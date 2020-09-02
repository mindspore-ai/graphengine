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
#include "hybrid_execution_context.h"
#include "subgraph_context.h"

namespace ge {
namespace hybrid {
namespace {
// 5s * 120, wait for 10m
constexpr auto kWaitInternal = 5;
constexpr auto kMaxWaitTimes = 120;
}  // namespace
ShapeInferenceState::ShapeInferenceState(const NodeItem &node_item) : node_item(node_item) {
  this->num_pending_shapes_ = node_item.num_inputs - node_item.num_static_input_shapes;
  GELOGD("[%s] ShapeInferenceState created, pending shape count = %d", node_item.NodeName().c_str(),
         this->num_pending_shapes_);
}

void ShapeInferenceState::UpdateInputShape(uint32_t idx, const GeShape &ori_shape, const GeShape &shape) {
  if (!node_item.is_dynamic || node_item.is_input_shape_static[idx]) {
    GELOGD("[%s] Trying to update static shape, idx = %u. old shape = [%s], new shape = [%s]",
           node_item.NodeName().c_str(), idx, node_item.op_desc->MutableInputDesc(idx)->GetShape().ToString().c_str(),
           shape.ToString().c_str());
    return;
  }

  GELOGD("[%s] Update input shape [%u] with Shape: [%s] and OriginalShape: [%s]", node_item.NodeName().c_str(), idx,
         shape.ToString().c_str(), ori_shape.ToString().c_str());

  std::lock_guard<std::mutex> lk(mu_);
  node_item.op_desc->MutableInputDesc(idx)->SetShape(shape);
  node_item.op_desc->MutableInputDesc(idx)->SetOriginShape(ori_shape);
  if (--num_pending_shapes_ == 0) {
    ready_cv_.notify_all();
  }
}

void ShapeInferenceState::UpdateInputShapeFuture(uint32_t idx, ShapeFuture &&future) {
  if (!node_item.is_dynamic || node_item.is_input_shape_static[idx]) {
    GELOGD("[%s] Trying to update constant shape, idx = %u", node_item.NodeName().c_str(), idx);
    return;
  }

  GELOGD("[%s] Update input shape [%u] with ShapeFuture.", node_item.NodeName().c_str(), idx);
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

      if (context.GetStatus() != SUCCESS) {
        GELOGE(FAILED, "[%s] Await pending shape cancelled", node_item.NodeName().c_str());
        break;
      }
    }

    if (!wait_success) {
      GELOGE(FAILED, "[%s] Wait for shape timeout.", node_item.NodeName().c_str());
      return FAILED;
    }
  }

  for (auto &p : shape_futures) {
    auto idx = p.first;
    auto &future = p.second;
    GeShape shape;
    GeShape ori_shape;
    RECORD_SHAPE_INFERENCE_EVENT(&context, node_item.NodeName().c_str(), "[AwaitShape] [idx = %u] Start", idx);
    GE_CHK_STATUS_RET(future.Get(ori_shape, shape), "[%s] Get shape failed. index = %u", node_item.NodeName().c_str(),
                      idx);
    RECORD_SHAPE_INFERENCE_EVENT(&context, node_item.NodeName().c_str(), "[AwaitShape] [idx = %u] End", idx);

    GELOGD("[%s] Update input shape [%u] with shape: [%s] and ori_shape: [%s]", node_item.NodeName().c_str(), idx,
           shape.ToString().c_str(), ori_shape.ToString().c_str());
    node_item.op_desc->MutableInputDesc(idx)->SetShape(std::move(shape));
    node_item.op_desc->MutableInputDesc(idx)->SetOriginShape(ori_shape);
  }

  return SUCCESS;
}

ShapeFuture::ShapeFuture(NodePtr src_node, uint32_t src_index, SubgraphContext *subgraph_context)
    : src_node_(std::move(src_node)), src_index_(src_index), subgraph_context_(subgraph_context) {}

NodeState::NodeState(const NodeItem &node_item, SubgraphContext *subgraph_context)
    : node_item_(&node_item), shape_inference_state_(node_item), subgraph_context_(subgraph_context) {
  this->op_desc_ = node_item.node->GetOpDesc();
}

Status NodeState::AwaitInputTensors(GraphExecutionContext &context) const {
  for (auto &src_node : node_item_->dependents_for_execution) {
    GELOGI("[%s] Start to wait for data dependent node: [%s]", node_item_->NodeName().c_str(),
           src_node->GetName().c_str());
    RECORD_EXECUTION_EVENT(&context, node_item_->NodeName().c_str(), "[AwaitNodeDone] [%s] Start",
                           src_node->GetName().c_str());
    if (!subgraph_context_->Await(src_node)) {
      GELOGE(INTERNAL_ERROR, "[%s] Await node [%s] failed.", GetName().c_str(), src_node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    RECORD_EXECUTION_EVENT(&context, node_item_->NodeName().c_str(), "[AwaitNodeDone] [%s] End",
                           src_node->GetName().c_str());
    GELOGI("[%s] Done waiting node.", src_node->GetName().c_str());
  }

  return SUCCESS;
}

Status NodeState::WaitForPrepareDone() {
  if (prepare_future_.valid()) {
    GELOGD("[%s] Start to wait for prepare future.", GetName().c_str());
    GE_CHK_STATUS_RET(prepare_future_.get(), "[%s] PreRun failed.", GetName().c_str());
  }

  return SUCCESS;
}

Status ShapeFuture::Get(GeShape &ori_shape, GeShape &shape) {
  GELOGI("Start to wait node: %s for getting shape", src_node_->GetName().c_str());
  if (!subgraph_context_->Await(src_node_)) {
    GELOGE(INTERNAL_ERROR, "cancelled");
    return INTERNAL_ERROR;
  }

  shape = src_node_->GetOpDesc()->MutableOutputDesc(src_index_)->MutableShape();
  ori_shape = src_node_->GetOpDesc()->MutableOutputDesc(src_index_)->GetOriginShape();
  GELOGI("Get shape from %s:%u. shape = [%s]", src_node_->GetName().c_str(), src_index_, shape.ToString().c_str());
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
