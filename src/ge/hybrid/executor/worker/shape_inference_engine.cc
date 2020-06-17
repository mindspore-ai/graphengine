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
#include "graph/runtime_inference_context.h"
#include "graph/utils/node_utils.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {

ShapeInferenceEngine::ShapeInferenceEngine(GraphExecutionContext *context) : context_(context) {}

Status ShapeInferenceEngine::Start(ThreadPool &pool) {
  GELOGI("RuntimeShapeInferenceEngine start.");
  pool.commit([&]() {
    auto ret = this->InferShapeProcess();
    InferenceDone(ret);
  });

  return SUCCESS;
}

Status ShapeInferenceEngine::InferShapeProcess() {
  GELOGI("RuntimeShapeInferenceEngine worker start.");
  const auto &root_nodes = context_->model->RootNodes();
  auto &complete_queue = context_->compile_queue;
  std::queue<InferenceState *> ready_nodes;
  for (auto &node_item : root_nodes) {
    auto infer_state = GetOrCreateEntry(*node_item);
    GE_CHECK_NOTNULL(infer_state);
    ready_nodes.emplace(infer_state);
  }

  while (!ready_nodes.empty()) {
    InferenceState *infer_state = ready_nodes.front();
    ready_nodes.pop();
    auto node_item = infer_state->node_item;
    // even for non-dynamic shape node, it is still necessary to wait for pending shapes if got any.
    // which indicates that the parent node is of type 4, in which case the inputs will be valid only
    // when computing is done.
    GE_CHK_STATUS_RET(infer_state->AwaitShapeFutures(context_), "Await shape failed.");
    GELOGI("[%s] Node is ready for shape inference.", node_item.NodeName().c_str());
    if (node_item.is_dynamic) {
      // may block
      RECORD_SHAPE_INFERENCE_EVENT(context_, node_item.NodeName().c_str(), "Start");
      GELOGI("[%s] Start to invoke InferShape", node_item.NodeName().c_str());
      auto ret = InferShape(*infer_state);
      if (ret != SUCCESS) {
        return ret;
      }

      RECORD_SHAPE_INFERENCE_EVENT(context_, node_item.NodeName().c_str(), "[CalcOpRunningParam] Start");
      GE_CHK_STATUS_RET(NodeExecutorManager::GetInstance().CalcOpRunningParam(*node_item.node),
                        "[%s] Failed to invoke CalcOpRunningParam.", node_item.NodeName().c_str());
      RECORD_SHAPE_INFERENCE_EVENT(context_, node_item.NodeName().c_str(), "[CalcOpRunningParam] End");
    } else {
      GELOGD("[%s] Skip static shape node", node_item.NodeName().c_str());
    }

    if (node_item.node_type != NETOUTPUT) {
      GELOGI("[%s] Push to compile queue", node_item.NodeName().c_str());
      // may block if full
      auto node_state = context_->GetOrCreateNodeState(node_item.node);
      complete_queue.Push(node_state);
    }

    // Propagate
    RECORD_SHAPE_INFERENCE_EVENT(context_, node_item.NodeName().c_str(), "[PropagateOutputShapes] Start");
    PropagateOutputShapes(*infer_state, ready_nodes);
    RECORD_SHAPE_INFERENCE_EVENT(context_, node_item.NodeName().c_str(), "[PropagateOutputShapes] End");
  }

  return SUCCESS;
}

void ShapeInferenceEngine::InferenceDone(Status status) {
  if (status != SUCCESS) {
    GELOGE(status, "Error occurred while shape inference");
    context_->OnError(status);
  } else {
    context_->compile_queue.Push(nullptr);
  }
  inference_states_.clear();
  GELOGI("RuntimeShapeInferenceEngine worker END");
}

Status ShapeInferenceEngine::InferShape(InferenceState &entry) {
  // input shapes are ready, wait for dependent data if has any
  const auto &node_item = entry.node_item;
  if (!node_item.dependent_node_list.empty()) {
    for (auto &src_node : node_item.dependent_node_list) {
      auto *src_node_item = context_->model->GetNodeItem(src_node);
      GELOGI("[%s] Start to wait for data dependent node: %s", node_item.NodeName().c_str(),
             src_node_item->NodeName().c_str());
      RECORD_SHAPE_INFERENCE_EVENT(context_, node_item.NodeName().c_str(), "[AwaitNodeDone] [%s] Start",
                                   src_node->GetName().c_str());
      if (!context_->cv_manager.Await(src_node)) {
        GELOGE(INTERNAL_ERROR, "[%s] Await node failed.", src_node_item->NodeName().c_str());
        return INTERNAL_ERROR;
      }

      RECORD_SHAPE_INFERENCE_EVENT(context_, node_item.NodeName().c_str(), "[AwaitNodeDone] [%s] End",
                                   src_node->GetName().c_str());
      GELOGI("[%s] Done waiting node.", src_node_item->NodeName().c_str());
    }
  }

  if (node_item.shape_inference_type == DEPEND_COMPUTE) {
    GELOGD("[%s] Skip node with unknown shape type DEPEND_COMPUTE", node_item.NodeName().c_str());
    return SUCCESS;
  }

  if (node_item.shape_inference_type == DEPEND_SHAPE_RANGE) {
    // in case InferFunc forgot to reset output shape
    for (auto &output_desc : node_item.op_desc->GetAllOutputsDescPtr()) {
      output_desc->SetShape(GeShape({UNKNOWN_DIM_NUM}));
    }
  }

  // do shape inference
  RECORD_SHAPE_INFERENCE_EVENT(context_, node_item.NodeName().c_str(), "[InferShape] Start");
  GELOGD("[%s] Start to invoke InferShapeAndType", node_item.NodeName().c_str());
  GE_CHK_STATUS_RET(ShapeRefiner::InferShapeAndType(node_item.node), "Invoke InferShapeAndType failed.");
  RECORD_SHAPE_INFERENCE_EVENT(context_, node_item.NodeName().c_str(), "[InferShape] End");

  // Check shape
  if (node_item.shape_inference_type != DEPEND_SHAPE_RANGE) {
    bool is_unknown_shape = false;
    GE_CHK_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node_item.node, is_unknown_shape),
                      "Failed to get shape status. node = %s", node_item.NodeName().c_str());

    GE_CHK_BOOL_RET_STATUS(!is_unknown_shape, INTERNAL_ERROR, "[%s] Shape is still unknown after shape inference.",
                           node_item.NodeName().c_str());
  }

  GELOGD("[%s] InferShapeAndType finished successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}

void ShapeInferenceEngine::PropagateOutputShapes(InferenceState &entry, std::queue<InferenceState *> &queue) {
  auto &node_item = entry.node_item;
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
      auto inference_state = GetOrCreateEntry(*dst_node_item);
      GELOGI("[%s] Update dst node [%s], input index = %d", node_item.NodeName().c_str(),
             dst_node_item->NodeName().c_str(), dst_input_index_and_node.first);

      // in case type 3/4, shape will be valid after computing is done
      if (shape_is_future) {
        ShapeFuture future(node_item.node, i, &context_->cv_manager);
        inference_state->UpdateInputShapeFuture(dst_input_index_and_node.first, std::move(future));
      } else {
        inference_state->UpdateInputShape(dst_input_index_and_node.first, ori_shape, shape);
      }

      if (inference_state->IsInputShapesReady()) {
        GELOGI("[%s] Node input shape is ready, add to queue.", inference_state->node_item.NodeName().c_str());
        queue.emplace(inference_state);
      }
    }
  }

  GELOGD("[%s] Propagating output shapes finished successfully.", node_item.NodeName().c_str());
}

ShapeInferenceEngine::InferenceState *ShapeInferenceEngine::GetOrCreateEntry(const NodeItem &node_item) {
  auto &node_state = inference_states_[node_item.node_id];
  if (node_state == nullptr) {
    node_state.reset(new (std::nothrow) InferenceState(node_item));
  }

  return node_state.get();
}

ShapeInferenceEngine::InferenceState::InferenceState(const NodeItem &node_item) : node_item(node_item) {
  this->num_pending_shapes = node_item.num_inputs;
}

void ShapeInferenceEngine::InferenceState::UpdateInputShape(uint32_t idx, const GeShape &ori_shape,
                                                            const GeShape &shape) {
  if (node_item.const_input_shapes.count(idx) != 0) {
    GELOGD("[%s] Trying to update constant shape, idx = %u. old shape = [%s], new shape = [%s]",
           node_item.NodeName().c_str(), idx, node_item.op_desc->MutableInputDesc(idx)->GetShape().ToString().c_str(),
           shape.ToString().c_str());
  }

  GELOGD("[%s] Update input shape [%u] with Shape: [%s] and OriginalShape: [%s]", node_item.NodeName().c_str(), idx,
         shape.ToString().c_str(), ori_shape.ToString().c_str());
  num_pending_shapes -= 1;
  node_item.op_desc->MutableInputDesc(idx)->SetShape(shape);
  node_item.op_desc->MutableInputDesc(idx)->SetOriginShape(ori_shape);
}

void ShapeInferenceEngine::InferenceState::UpdateInputShapeFuture(uint32_t idx, ShapeFuture &&future) {
  if (node_item.const_input_shapes.count(idx) != 0) {
    GELOGE(INTERNAL_ERROR, "[%s] Trying to update constant shape, idx = %u", node_item.NodeName().c_str(), idx);
    return;
  }

  GELOGD("[%s] Update input shape [%u] with ShapeFuture.", node_item.NodeName().c_str(), idx);
  num_pending_shapes -= 1;
  shape_futures.emplace_back(idx, std::move(future));
}

Status ShapeInferenceEngine::InferenceState::AwaitShapeFutures(GraphExecutionContext *context) {
  for (auto &p : shape_futures) {
    auto idx = p.first;
    auto &future = p.second;
    GeShape shape;
    GeShape ori_shape;
    RECORD_SHAPE_INFERENCE_EVENT(context, node_item.NodeName().c_str(), "[AwaitShape] [idx = %u] Start", idx);
    GE_CHK_STATUS_RET(future.Get(ori_shape, shape), "[%s] Get shape failed. index = %u", node_item.NodeName().c_str(),
                      idx);
    RECORD_SHAPE_INFERENCE_EVENT(context, node_item.NodeName().c_str(), "[AwaitShape] [idx = %u] End", idx);

    GELOGD("[%s] Update input shape [%u] with shape: [%s] and ori_shape: [%s]", node_item.NodeName().c_str(), idx,
           shape.ToString().c_str(), ori_shape.ToString().c_str());
    node_item.op_desc->MutableInputDesc(idx)->SetShape(std::move(shape));
    node_item.op_desc->MutableInputDesc(idx)->SetOriginShape(ori_shape);
  }

  return SUCCESS;
}

ShapeInferenceEngine::ShapeFuture::ShapeFuture(NodePtr src_node, uint32_t src_index, NodeDoneManager *node_done_manager)
    : src_node_(std::move(src_node)), src_index_(src_index), node_done_manager_(node_done_manager) {}
}  // namespace hybrid
}  // namespace ge