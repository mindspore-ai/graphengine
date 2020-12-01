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
    : execution_context_(execution_context),
      subgraph_context_(subgraph_context) {
}

Status ShapeInferenceEngine::InferShape(NodeState &node_state) {
  // Wait for all input shape become valid
  GE_CHK_STATUS_RET_NOLOG(node_state.GetShapeInferenceState().AwaitShapesReady(*execution_context_));

  auto &node_item = *node_state.GetNodeItem();

  // Wait for "const input nodes" if node's shape inference function requires any.
  // Even if output shape is static, there are cases that the const-input will be used in OpTiling and Execution
  GE_CHK_STATUS_RET_NOLOG(AwaitDependentNodes(node_state));
  if (node_item.is_output_shape_static) {
    return SUCCESS;
  }

  if (node_item.fused_subgraph != nullptr) {
    return InferShapeForSubgraph(node_item, *node_item.fused_subgraph);
  }

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

  // Do shape inference
  GELOGD("[%s] Start to invoke InferShapeAndType", node_item.NodeName().c_str());
  {
    std::lock_guard<std::mutex> lk(mu_);
    RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[InferShapeAndType] Start");
    GE_CHK_STATUS_RET(ShapeRefiner::InferShapeAndTypeForRunning(node_item.node, true), "Invoke InferShapeAndType failed.");
    RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[InferShapeAndType] End");
  }
  // Check again to make sure shape is valid after shape inference
  if (node_item.shape_inference_type != DEPEND_SHAPE_RANGE) {
    bool is_unknown_shape = false;
    GE_CHK_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node_item.node, is_unknown_shape),
                      "Failed to get shape status. node = %s",
                      node_item.NodeName().c_str());

    GE_CHK_BOOL_RET_STATUS(!is_unknown_shape,
                           INTERNAL_ERROR,
                           "[%s] Shape is still unknown after shape inference.",
                           node_item.NodeName().c_str());
  }

  GELOGD("[%s] [HybridTrace] After shape inference. Node = %s",
         node_item.NodeName().c_str(),
         node_item.DebugString().c_str());

  GELOGD("[%s] InferShapeAndType finished successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}

Status ShapeInferenceEngine::AwaitDependentNodes(NodeState &node_state) {
  auto &node_item = *node_state.GetNodeItem();
  for (auto &src_node : node_item.dependents_for_shape_inference) {
    GELOGI("[%s] Start to wait for data dependent node: %s",
           node_item.NodeName().c_str(),
           src_node->GetName().c_str());
    RECORD_SHAPE_INFERENCE_EVENT(execution_context_,
                                 node_item.NodeName().c_str(),
                                 "[AwaitNodeDone] [%s] Start",
                                 src_node->GetName().c_str());
    if (!subgraph_context_->Await(src_node)) {
      GELOGE(INTERNAL_ERROR, "[%s] Await node failed.", src_node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    RECORD_SHAPE_INFERENCE_EVENT(execution_context_,
                                 node_item.NodeName().c_str(),
                                 "[AwaitNodeDone] [%s] End",
                                 src_node->GetName().c_str());
    GELOGI("[%s] Done waiting node.", src_node->GetName().c_str());
  }

  return SUCCESS;
}

Status ShapeInferenceEngine::PropagateOutputShapes(const NodeItem &node_item) {
  if (node_item.is_output_shape_static) {
    return SUCCESS;
  }

  // output shape will not be valid until compute is done.
  bool shape_is_future =
      node_item.shape_inference_type == DEPEND_SHAPE_RANGE || node_item.shape_inference_type == DEPEND_COMPUTE;
  GELOGD("[%s] Start to propagate output shapes. shape_type = %d",
         node_item.NodeName().c_str(),
         node_item.shape_inference_type);
  RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[PropagateOutputShapes] Start");
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

      GELOGI("[%s] Update dst node [%s], input index = %d",
             node_item.NodeName().c_str(),
             dst_node_item->NodeName().c_str(),
             dst_input_index_and_node.first);

      // in case type 3 and 4, shape will be valid after computing is done
      auto &infer_state = dst_node_state->GetShapeInferenceState();
      if (shape_is_future) {
        ShapeFuture future(node_item.node, i, subgraph_context_);
        infer_state.UpdateInputShapeFuture(dst_input_index_and_node.first,
                                           std::move(future));
      } else {
        GE_CHK_STATUS_RET_NOLOG(infer_state.UpdateInputShape(dst_input_index_and_node.first,
                                                             ori_shape,
                                                             shape));
      }
    }
  }
  RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[PropagateOutputShapes] End");
  GELOGD("[%s] Propagating output shapes finished successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}

Status ShapeInferenceEngine::InferShapeForSubgraph(const NodeItem &node_item, const FusedSubgraph &fused_subgraph) {
  GELOGD("[%s] Start to infer shape by fused subgraph", node_item.NodeName().c_str());
  for (auto &it : fused_subgraph.input_mapping) {
    auto parent_tensor_desc = node_item.MutableInputDesc(it.first);
    GE_CHECK_NOTNULL(parent_tensor_desc);
    GELOGD("Start to update shape by input[%u]", it.first);
    GELOGD("Update shape to [%s]", parent_tensor_desc->GetShape().ToString().c_str());
    GELOGD("Update original shape to [%s]", parent_tensor_desc->GetOriginShape().ToString().c_str());
    for (auto &tensor_desc : it.second) {
      tensor_desc->SetShape(parent_tensor_desc->GetShape());
      tensor_desc->SetOriginShape(parent_tensor_desc->GetOriginShape());
    }
  }

  for (auto &node : fused_subgraph.nodes) {
    GELOGD("[%s] Start to invoke InferShapeAndType", node->GetName().c_str());
    GE_CHK_STATUS_RET(ShapeRefiner::InferShapeAndType(node));
    GELOGD("[%s] Done invoking InferShapeAndType", node->GetName().c_str());
    GE_CHK_STATUS_RET(UpdatePeerNodeShape(*node),
                      "[%s] Failed to update shapes of peer node.",
                      node->GetName().c_str());
  }

  for (auto &it : fused_subgraph.output_mapping) {
    uint32_t parent_output_idx = it.first;
    const auto &op_desc = it.second;
    GELOGD("Update parent output[%d] by [%s]", parent_output_idx, op_desc->GetName().c_str());
    auto input_desc = op_desc->MutableInputDesc(0);
    GE_CHECK_NOTNULL(input_desc);
    auto parent_output_tensor_desc = node_item.op_desc->MutableOutputDesc(parent_output_idx);
    GE_CHECK_NOTNULL(parent_output_tensor_desc);
    GELOGD("Update shape to [%s]", input_desc->GetShape().ToString().c_str());
    GELOGD("Update original shape to [%s]", input_desc->GetOriginShape().ToString().c_str());
    parent_output_tensor_desc->SetOriginShape(input_desc->GetOriginShape());
    parent_output_tensor_desc->SetShape(input_desc->GetShape());
  }

  GELOGD("[%s] Done shape inference by subgraph successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}

Status ShapeInferenceEngine::UpdatePeerNodeShape(const Node &node) {
  auto op_desc = node.GetOpDesc();
  for (const auto &out_anchor : node.GetAllOutDataAnchors()) {
    auto output_tensor = op_desc->MutableOutputDesc(out_anchor->GetIdx());
    for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
      auto peer_node = peer_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);
      auto peer_op_desc = peer_node->GetOpDesc();
      GE_CHECK_NOTNULL(peer_op_desc);
      auto peer_input_desc = peer_op_desc->MutableInputDesc(peer_anchor->GetIdx());
      if (peer_input_desc == nullptr) {
        GELOGE(GRAPH_FAILED, "peer_input_desc is nullptr");
        continue;
      }

      GELOGI("Peer input op desc name is %s, need to flush: shape size is %zu, datatype is %d, original datatype is %d",
             peer_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
             output_tensor->GetShape().GetDimNum(), output_tensor->GetDataType(),
             output_tensor->GetOriginDataType());
      peer_input_desc->SetOriginShape(output_tensor->GetOriginShape());
      peer_input_desc->SetShape(output_tensor->GetShape());
      GELOGI("Peer input op desc name is %s, shape size is %zu, datatype is %d, original datatype is %d",
             peer_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
             peer_input_desc->GetShape().GetDimNum(), peer_input_desc->GetDataType(),
             peer_input_desc->GetOriginDataType());
    }
  }
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
