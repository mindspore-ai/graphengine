/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include "graph/manager/graph_manager_utils.h"

#include <set>

#include "graph/debug/ge_attr_define.h"

namespace ge {
GraphNode::GraphNode(const GraphId graph_id) : graph_id_(graph_id), sem_(1U) {}

GraphNode::~GraphNode() = default;

void GraphNode::Lock() {
  (void)sem_.Push(0U);
}

void GraphNode::Unlock() {
  uint8_t unused;
  (void)sem_.Pop(unused);
  (void)unused;
}

void GraphNode::IncreaseLoadCount() {
  const std::unique_lock<std::mutex> lock(load_count_mu_);
  if (load_record_ == kMaxLoadNum) {
    GELOGW("Reach the maximum of load_count:%u", kMaxLoadNum);
    return;
  }
  ++load_count_;
}

void GraphNode::SetLoaded() {
  --load_count_;
  ++load_record_;
  load_flag_ = true;
}

SubGraphInfo::SubGraphInfo() : subgraph_ptr_(nullptr), ge_model_ptr_(nullptr) {}

SubGraphInfo::~SubGraphInfo() {
}

GraphModelListener::GraphModelListener()
    : ModelListener(), result_code_(0U), is_finished_(false) {}

Status GraphModelListener::OnComputeDone(const uint32_t model_id, const uint32_t data_index, const uint32_t result_code,
                                         std::vector<ge::Tensor> &outputs) {
  (void)outputs;
  GELOGI(
      "[GraphManager] graph compute call back, model_id:%u, task_id:%u, "
      "resultCode:%u.",
      model_id, data_index, result_code);

  const std::lock_guard<std::mutex> lock(mutex_);
  result_code_ = result_code;
  is_finished_ = true;
  condition_.notify_all();

  return SUCCESS;
}

uint32_t GraphModelListener::GetResultCode() {
  // Pending until async execute graph complete
  std::unique_lock<std::mutex> lock(mutex_);
  if (!is_finished_) {
    GELOGI("[GetResultCode] wait model execute finished.");
    condition_.wait(lock);
  }

  if (!is_finished_) {
    REPORT_CALL_ERROR("E19999", "Model not run finish");
    GELOGE(INTERNAL_ERROR, "[Check][Param] model not run finish.");
    return INTERNAL_ERROR;
  }
  return result_code_;
}

Status GraphModelListener::ResetResult() {
  const std::lock_guard<std::mutex> lock(mutex_);
  result_code_ = 0U;
  is_finished_ = false;

  return SUCCESS;
}

void RunAsyncListener::SetCallback(const RunAsyncCallback &callback) {
  (void)sem_.Push(0U);
  callback_ = callback;
}

Status RunAsyncListener::OnComputeDone(const uint32_t model_id, const uint32_t data_index, const uint32_t result_code,
                                       std::vector<ge::Tensor> &outputs) {
  GELOGI("[GraphManager] run graph async call back, modelId:%u, taskId:%u, resultCode:%u.",
         model_id, data_index, result_code);
  GE_CHECK_NOTNULL(callback_);
  callback_(result_code, outputs);
  uint8_t unused;
  (void)sem_.Pop(unused);
  (void)unused;
  return SUCCESS;
}
}  // namespace ge
