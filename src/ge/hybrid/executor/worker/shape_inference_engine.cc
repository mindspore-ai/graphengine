/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hybrid/executor/worker/shape_inference_engine.h"
#include "graph/shape_refiner.h"
#include "graph/utils/node_utils.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
ShapeInferenceEngine::ShapeInferenceEngine(GraphExecutionContext *execution_context, SubgraphContext *subgraph_context)
    : execution_context_(execution_context), subgraph_context_(subgraph_context) {}

Status ShapeInferenceEngine::InferShape(NodeState &node_state) {
  // Wait for all input shape become valid
  GE_CHK_STATUS_RET_NOLOG(node_state.GetShapeInferenceState().AwaitShapesReady(*execution_context_));

  auto &node_item = *node_state.GetNodeItem();
  // Skip shape inference for node of type DEPEND_COMPUTE
  if (node_item.shape_inference_type == DEPEND_COMPUTE) {
    GELOGD("[%s] Skipping node with unknown shape type DEPEND_COMPUTE", node_item.NodeName().c_str());
    return SUCCESS;
  }

  // Clear shape range in case shape inference func forgot to do it
  if (node_item.shape_inference_type == DEPEND_SHAPE_RANGE) {
    // in case InferFunc forgot to reset output shape
    for (auto &output_desc : node_item.op_desc->GetAllOutputsDescPtr()) {
      output_desc->SetShape(GeShape({UNKNOWN_DIM_NUM}));
    }
  }

  // Wait for "const input nodes" if node's shape inference function requires any.
  GE_CHK_STATUS_RET_NOLOG(AwaitDependentNodes(node_state));

  // Do shape inference
  GELOGD("[%s] Start to invoke InferShapeAndType", node_item.NodeName().c_str());
  RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[InferShapeAndType] Start");
  GE_CHK_STATUS_RET(ShapeRefiner::InferShapeAndType(node_item.node), "Invoke InferShapeAndType failed.");
  RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[InferShapeAndType] End");

  // Check again to make sure shape is valid after shape inference
  if (node_item.shape_inference_type != DEPEND_SHAPE_RANGE) {
    bool is_unknown_shape = false;
    GE_CHK_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node_item.node, is_unknown_shape),
                      "Failed to get shape status. node = %s", node_item.NodeName().c_str());

    GE_CHK_BOOL_RET_STATUS(!is_unknown_shape, INTERNAL_ERROR, "[%s] Shape is still unknown after shape inference.",
                           node_item.NodeName().c_str());
  }

  GELOGD("[%s] [HybridTrace] After shape inference. Node = %s", node_item.NodeName().c_str(),
         node_item.DebugString().c_str());

  GELOGD("[%s] InferShapeAndType finished successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}

Status ShapeInferenceEngine::AwaitDependentNodes(NodeState &node_state) {
  auto &node_item = *node_state.GetNodeItem();
  for (auto &src_node : node_item.dependents_for_shape_inference) {
    GELOGI("[%s] Start to wait for data dependent node: %s", node_item.NodeName().c_str(), src_node->GetName().c_str());
    RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[AwaitNodeDone] [%s] Start",
                                 src_node->GetName().c_str());
    if (!subgraph_context_->Await(src_node)) {
      GELOGE(INTERNAL_ERROR, "[%s] Await node failed.", src_node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[AwaitNodeDone] [%s] End",
                                 src_node->GetName().c_str());
    GELOGI("[%s] Done waiting node.", src_node->GetName().c_str());
  }

  return SUCCESS;
}

Status ShapeInferenceEngine::PropagateOutputShapes(const NodeItem &node_item) {
  // output shape will not be valid until compute is done.
  bool shape_is_future =
    node_item.shape_inference_type == DEPEND_SHAPE_RANGE || node_item.shape_inference_type == DEPEND_COMPUTE;
  GELOGD("[%s] Start to propagate output shapes. shape_type = %d", node_item.NodeName().c_str(),
         node_item.shape_inference_type);

  // propagate each output
  for (int i = 0; i < node_item.num_outputs; ++i) {
    auto output_desc = node_item.op_desc->MutableOutputDesc(i);
    const auto &shape = output_desc->MutableShape();
    const auto &ori_shape = output_desc->GetOriginShape();
    auto &output_nodes = node_item.outputs[i];

    // propagate output to all sub-inputs
    for (auto &dst_input_index_and_node : output_nodes) {
      auto &dst_node_item = dst_input_index_and_node.second;
      auto dst_node_state = subgraph_context_->GetOrCreateNodeState(dst_node_item);
      GE_CHECK_NOTNULL(dst_node_state);

      GELOGI("[%s] Update dst node [%s], input index = %d", node_item.NodeName().c_str(),
             dst_node_item->NodeName().c_str(), dst_input_index_and_node.first);

      // in case type 3 and 4, shape will be valid after computing is done
      if (shape_is_future) {
        ShapeFuture future(node_item.node, i, subgraph_context_);
        dst_node_state->GetShapeInferenceState().UpdateInputShapeFuture(dst_input_index_and_node.first,
                                                                        std::move(future));
      } else {
        dst_node_state->GetShapeInferenceState().UpdateInputShape(dst_input_index_and_node.first, ori_shape, shape);
      }
    }
  }

  GELOGD("[%s] Propagating output shapes finished successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
