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

#include "graph/manager/model_manager/event_manager.h"

#define RETURN_IF_COND_NOT_MET(condition, ...) \
  do {                                         \
    if (!(condition)) {                        \
      GELOGE(FAILED, __VA_ARGS__);             \
      return;                                  \
    }                                          \
  } while (0);

namespace ge {
Status EventManager::Init(size_t event_num) {
  if (this->inited_) {
    return SUCCESS;
  }

  rtEvent_t event = nullptr;
  current_idx_ = 0;
  for (size_t i = 0; i < event_num; ++i) {
    GE_CHK_RT_RET(rtEventCreate(&event));
    this->event_list_.push_back(event);
  }

  this->inited_ = true;

  return SUCCESS;
}

void EventManager::Release() noexcept {
  for (size_t i = 0; i < this->event_list_.size(); ++i) {
    rtError_t rt_ret = rtEventDestroy(this->event_list_[i]);
    RETURN_IF_COND_NOT_MET(rt_ret == RT_ERROR_NONE, "Destroy event failed, idx is %zu, ret is 0x%x.", i, rt_ret);
  }
  this->event_list_.clear();

  this->inited_ = false;
}

Status EventManager::EventRecord(size_t event_idx, rtStream_t stream) {
  GE_CHK_BOOL_RET_STATUS_NOLOG(this->inited_, INTERNAL_ERROR);

  GE_CHK_BOOL_RET_STATUS_NOLOG(event_idx < this->event_list_.size(), PARAM_INVALID);

  GE_CHK_RT_RET(rtEventRecord(this->event_list_[event_idx], stream));

  current_idx_ = static_cast<uint32_t>(event_idx);
  return SUCCESS;
}

Status EventManager::EventElapsedTime(size_t start_event_idx, size_t stop_event_idx, float &time) {
  GE_CHK_BOOL_RET_STATUS_NOLOG(this->inited_, INTERNAL_ERROR);

  GE_CHK_BOOL_RET_STATUS_NOLOG(start_event_idx < this->event_list_.size() &&
                               stop_event_idx < this->event_list_.size() && start_event_idx <= stop_event_idx,
                               PARAM_INVALID);

  GE_CHK_RT_RET(rtEventElapsedTime(&time, this->event_list_[start_event_idx], this->event_list_[stop_event_idx]));

  return SUCCESS;
}

Status EventManager::GetEvent(uint32_t index, rtEvent_t &event) {
  GE_CHK_BOOL_RET_STATUS_NOLOG(index < this->event_list_.size(), PARAM_INVALID);
  event = this->event_list_[index];
  return SUCCESS;
}
}  // namespace ge
