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

#include "hybrid/executor/node_state.h"
#include <chrono>
#include "framework/common/debug/log.h"
#include "graph/compute_graph.h"
#include "graph/utils/tensor_utils.h"
#include "hybrid_execution_context.h"
#include "subgraph_context.h"

namespace ge {
namespace hybrid {
namespace {
// 5s * 120, wait for 10m
constexpr auto kWaitInternal = 5;
constexpr auto kMaxWaitTimes = 120;
}
ShapeInferenceState::ShapeInferenceState(const NodeItem &node_item) : node_item(node_item) {
  this->num_pending_shapes_ = node_item.num_inputs - node_item.num_static_input_shapes;
  GELOGD("[%s] ShapeInferenceState created, pending shape count = %d",
         node_item.NodeName().c_str(),
         this->num_pending_shapes_);

  input_tensor_desc.resize(node_item.num_inputs);
  for (int i = 0; i < node_item.num_inputs; ++i) {
    node_item.GetInputDesc(i, input_tensor_desc[i]);
  }

  output_tensor_desc.resize(node_item.num_outputs);
  for (int i = 0; i < node_item.num_outputs; ++i) {
    node_item.GetOutputDesc(i, output_tensor_desc[i]);
  }
}

Status ShapeInferenceState::UpdateInputShape(int idx, const GeTensorDesc &target) {
  if (node_item.IsInputShapeStatic(idx)) {
    GELOGD("[%s] Trying to update static shape, idx = %d. old shape = [%s], new shape = [%s]",
           node_item.NodeName().c_str(),
           idx,
           node_item.MutableInputDesc(idx)->GetShape().ToString().c_str(),
           target.GetShape().ToString().c_str());
    return SUCCESS;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto &input_desc = input_tensor_desc[idx];
  GeShape shape = target.GetShape();
  input_desc.SetShape(shape);
  input_desc.SetOriginShape(target.GetOriginShape());
  int64_t tensor_size = -1;
  (void) TensorUtils::GetSize(target, tensor_size);
  if (tensor_size <= 0) {
    Format format = input_desc.GetFormat();
    DataType data_type = input_desc.GetDataType();
    if (TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "[Invoke][CalcTensorMemSize] failed for [%s] when ShapeInferenceState %s.", 
          node_item.NodeName().c_str(), __FUNCTION__);
      REPORT_CALL_ERROR("E19999", "CalcTensorMemSize failed for [%s] when ShapeInferenceState %s.", 
          node_item.NodeName().c_str(), __FUNCTION__);    
      return FAILED;
    }
  }
  GELOGD("[%s] Update input shape [%d] with Shape: [%s] and OriginalShape: [%s], size = %ld",
         node_item.NodeName().c_str(),
         idx,
         shape.ToString().c_str(),
         target.GetOriginShape().ToString().c_str(),
         tensor_size);
  (void) TensorUtils::SetSize(input_desc, tensor_size);
  if (--num_pending_shapes_ <= 0) {
    ready_cv_.notify_all();
  }

  return SUCCESS;
}

void ShapeInferenceState::UpdateInputShapeFuture(int idx, ShapeFuture &&future) {
  if (node_item.IsInputShapeStatic(idx)) {
    GELOGD("[%s] Trying to update constant shape, idx = %d", node_item.NodeName().c_str(), idx);
    return;
  }

  GELOGD("[%s] Update input shape [%d] with ShapeFuture.", node_item.NodeName().c_str(), idx);
  std::lock_guard<std::mutex> lk(mu_);
  shape_futures.emplace_back(idx, std::move(future));
  if (--num_pending_shapes_ == 0) {
    ready_cv_.notify_all();
  }
}

Status ShapeInferenceState::AwaitShapesReady(const GraphExecutionContext &context) {
  if (!node_item.is_dynamic) {
    return SUCCESS;
  }
  std::unique_lock<std::mutex> lk(mu_);
  if (num_pending_shapes_ > 0) {
    GELOGD("[%s] Await pending shape or shape future start.", node_item.NodeName().c_str());
    int try_count = 0;
    bool wait_success = false;
    while (try_count++ < kMaxWaitTimes) {
      if (ready_cv_.wait_for(lk, std::chrono::seconds(kWaitInternal), [&]() { return num_pending_shapes_ == 0; })) {
        GELOGD("[%s] Await pending shape or shape future end.", node_item.NodeName().c_str());
        wait_success = true;
        break;
      }

      if (context.is_eos_) {
        GELOGD("[%s] Await pending shape cancelled due to end of sequence", node_item.NodeName().c_str());
        return END_OF_SEQUENCE;
      }

      if (context.GetStatus() != SUCCESS) {
        GELOGE(FAILED, "[Check][Status][%s] Await pending shape cancelled when %s.", 
            node_item.NodeName().c_str(), __FUNCTION__);
        REPORT_CALL_ERROR("E19999", "[%s] Await pending shape cancelled when %s.", 
            node_item.NodeName().c_str(), __FUNCTION__);
        break;
      }
    }

    if (!wait_success) {
      GELOGE(FAILED, "[Check][Status][%s] Wait for shape timeout when %s.", 
          node_item.NodeName().c_str(), __FUNCTION__);
      REPORT_CALL_ERROR("E19999", "[%s] Wait for shape timeout when %s.", 
          node_item.NodeName().c_str(), __FUNCTION__);
      return FAILED;
    }
  }

  for (size_t i = 0; i < input_tensor_desc.size(); ++i) {
    auto dst_tensor_desc = node_item.op_desc->MutableInputDesc(i);
    if (dst_tensor_desc == nullptr) {
      continue;
    }

    auto &tensor_desc = input_tensor_desc[i];
    int64_t tensor_size = -1;
    (void) TensorUtils::GetSize(tensor_desc, tensor_size);

    dst_tensor_desc->SetShape(tensor_desc.MutableShape());
    dst_tensor_desc->SetOriginShape(tensor_desc.GetOriginShape());
    (void) TensorUtils::SetSize(*dst_tensor_desc, tensor_size);
  }

  for (auto &p : shape_futures) {
    auto idx = p.first;
    auto &future = p.second;
    RECORD_SHAPE_INFERENCE_EVENT(&context, node_item.NodeName().c_str(), "[AwaitShape] [idx = %u] Start", idx);
    const GeTensorDesc* src_tensor_desc = nullptr;
    GE_CHK_STATUS_RET_NOLOG(future.GetTensorDesc(&src_tensor_desc));
    GE_CHECK_NOTNULL(src_tensor_desc);
    RECORD_SHAPE_INFERENCE_EVENT(&context, node_item.NodeName().c_str(), "[AwaitShape] [idx = %u] End", idx);

    auto input_desc = node_item.MutableInputDesc(idx);
    GE_CHECK_NOTNULL(input_desc);
    int64_t tensor_size = -1;
    (void) TensorUtils::GetSize(*src_tensor_desc, tensor_size);
    GELOGD("[%s] Update input shape [%u] with shape: [%s] and ori_shape: [%s], index = %zu",
           node_item.NodeName().c_str(),
           idx,
           src_tensor_desc->GetShape().ToString().c_str(),
           src_tensor_desc->GetOriginShape().ToString().c_str(),
           tensor_size);
    input_desc->SetShape(src_tensor_desc->GetShape());
    input_desc->SetOriginShape(src_tensor_desc->GetOriginShape());
    (void) TensorUtils::SetSize(*input_desc, tensor_size);
  }

  return SUCCESS;
}

const vector<GeTensorDesc> &ShapeInferenceState::GetOutputTensorDesc() const {
    return output_tensor_desc;
}

Status ShapeInferenceState::UpdateOutputDesc() {
  for (size_t i = 0; i < output_tensor_desc.size(); ++i) {
    auto src_tensor_desc = node_item.MutableOutputDesc(i);
    GE_CHECK_NOTNULL(src_tensor_desc);
    auto &dst_tensor_desc = output_tensor_desc[i];
    dst_tensor_desc.SetShape(src_tensor_desc->MutableShape());
    dst_tensor_desc.SetOriginShape(src_tensor_desc->GetOriginShape());
    int64_t tensor_size = -1;
    (void) TensorUtils::GetSize(*src_tensor_desc, tensor_size);
    (void) TensorUtils::SetSize(dst_tensor_desc, tensor_size);
  }
  return SUCCESS;
}

ShapeFuture::ShapeFuture(NodeState *src_node,
                         uint32_t src_index,
                         SubgraphContext *subgraph_context)
    : src_node_(src_node), src_index_(src_index), subgraph_context_(subgraph_context) {
}

NodeState::NodeState(const NodeItem &node_item, SubgraphContext *subgraph_context)
    : node_item_(&node_item), shape_inference_state_(node_item), subgraph_context_(subgraph_context) {
  this->op_desc_ = node_item.node->GetOpDesc();
}

Status NodeState::AwaitInputTensors(GraphExecutionContext &context) const {
  for (auto &src_node : node_item_->dependents_for_execution) {
    GELOGD("[%s] Start to wait for data dependent node: [%s]",
           node_item_->NodeName().c_str(),
           src_node->GetName().c_str());
    RECORD_EXECUTION_EVENT(&context,
                           node_item_->NodeName().c_str(),
                           "[AwaitNodeDone] [%s] Start",
                           src_node->GetName().c_str());

    HYBRID_CHK_STATUS_RET(subgraph_context_->Await(src_node),
                          "[%s] Await node [%s] failed.",
                          GetName().c_str(),
                          src_node->GetName().c_str());

    RECORD_EXECUTION_EVENT(&context,
                           node_item_->NodeName().c_str(),
                           "[AwaitNodeDone] [%s] End",
                           src_node->GetName().c_str());
    GELOGD("[%s] Done waiting node.", src_node->GetName().c_str());
  }

  return SUCCESS;
}

Status NodeState::WaitForPrepareDone() {
  if (prepare_future_.valid()) {
    GELOGD("[%s] Start to wait for prepare future.", GetName().c_str());
    GE_CHK_STATUS_RET(prepare_future_.get(),
        "[Check][Status][%s] PreRun failed when NodeState %s.", GetName().c_str(), __FUNCTION__);
  }

  return SUCCESS;
}
Status NodeState::UpdateOutputShapes(int index, const GeShape &shape, const GeShape &ori_shape) {
  auto self_tensor_desc = op_desc_->MutableOutputDesc(index);
  GE_CHECK_NOTNULL(self_tensor_desc);
  self_tensor_desc->SetShape(shape);
  self_tensor_desc->SetOriginShape(ori_shape);
  return SUCCESS;
}

void NodeState::SetTaskContext(std::shared_ptr<TaskContext> &task_context) {
  task_context_ = task_context;
}

std::shared_ptr<TaskContext> NodeState::GetTaskContext() {
  return task_context_;
}

Status ShapeFuture::Get(GeShape &ori_shape, GeShape &shape) {
  GELOGD("Start to wait node: %s for getting shape", src_node_->GetName().c_str());
  HYBRID_CHK_STATUS_RET(subgraph_context_->Await(src_node_->GetNodeItem()->node), "cancelled");
  auto &output_desc = src_node_->GetShapeInferenceState().GetOutputTensorDesc().at(src_index_);
  shape = output_desc.GetShape();
  ori_shape = output_desc.GetOriginShape();
  GELOGD("Get shape from %s:%u. shape = [%s]", src_node_->GetName().c_str(), src_index_, shape.ToString().c_str());
  return SUCCESS;
}

Status ShapeFuture::GetTensorDesc(const GeTensorDesc **tensor_desc) {
  GE_CHECK_NOTNULL(tensor_desc);
  GELOGD("Start to wait node: %s for getting shape", src_node_->GetName().c_str());
  HYBRID_CHK_STATUS_RET(subgraph_context_->Await(src_node_->GetNodeItem()->node), "cancelled");
  *tensor_desc = &src_node_->GetShapeInferenceState().GetOutputTensorDesc().at(src_index_);
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
