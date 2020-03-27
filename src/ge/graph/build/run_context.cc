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

#include "graph/build/run_context.h"

#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
RunContextUtil::~RunContextUtil() { DestroyRtModelStreamAndEvents(); }

Status RunContextUtil::InitMemInfo(uint8_t *data_mem_base, uint64_t data_mem_size, uint8_t *weight_mem_base,
                                   uint64_t weight_mem_size) {
  if ((data_mem_size > 0) && (data_mem_base == nullptr)) {
    GELOGE(PARAM_INVALID, "InitMemInfo param data_mem_base is null but data_mem_size = %lu.", data_mem_size);
    return PARAM_INVALID;
  }
  if ((weight_mem_size > 0) && (weight_mem_base == nullptr)) {
    GELOGE(PARAM_INVALID, "InitMemInfo param weight_mem_base is null but weight_mem_size = %lu.", weight_mem_size);
    return PARAM_INVALID;
  }
  data_mem_base_ = data_mem_base;
  data_mem_size_ = data_mem_size;
  weight_mem_base_ = weight_mem_base;
  weight_mem_size_ = weight_mem_size;
  return SUCCESS;
}

Status RunContextUtil::CreateRtModelStreamsAndEvents(uint32_t stream_num, uint32_t event_num) {
  // Create rt model
  rtError_t rt_ret = rtModelCreate(&rt_model_, 0);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtModelCreate failed. rt_ret = %d", static_cast<int>(rt_ret));
    return RT_FAILED;
  }

  // Create rt Stream and bind with model
  for (uint32_t i = 0; i < stream_num; ++i) {
    rtStream_t stream = nullptr;
    rt_ret = rtStreamCreate(&stream, 0);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtStreamCreate failed. rt_ret = %d, index = %u", static_cast<int>(rt_ret), i);
      return RT_FAILED;
    }
    stream_list_.emplace_back(stream);

    rt_ret = rtModelBindStream(rt_model_, stream, 0);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Bind stream and model failed. rt_ret = %d, index = %u", static_cast<int>(rt_ret), i);
      return RT_FAILED;
    }
  }

  // Create rt event
  for (uint32_t i = 0; i < event_num; ++i) {
    rtEvent_t event = nullptr;
    rt_ret = rtEventCreate(&event);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtEventCreate failed. rt_ret = %d, index = %u", static_cast<int>(rt_ret), i);
      return RT_FAILED;
    }
    event_list_.emplace_back(event);
  }
  return SUCCESS;
}

void RunContextUtil::DestroyRtModelStreamAndEvents() noexcept {
  rtError_t rt_ret;
  for (size_t i = 0; i < stream_list_.size(); i++) {
    // Unbind stream to model first
    (void)rtModelUnbindStream(rt_model_, stream_list_[i]);
    rt_ret = rtStreamDestroy(stream_list_[i]);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Destroy stream failed. rt_ret = %d, index = %zu.", static_cast<int>(rt_ret), i);
    }
  }
  stream_list_.clear();

  for (size_t i = 0; i < event_list_.size(); i++) {
    rt_ret = rtEventDestroy(event_list_[i]);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Destroy event failed. rt_ret = %d, index = %zu.", static_cast<int>(rt_ret), i);
    }
  }
  event_list_.clear();

  if (rt_model_ != nullptr) {
    rt_ret = rtModelDestroy(rt_model_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Destroy rt model failed. rt_ret = %d.", static_cast<int>(rt_ret));
    }
    rt_model_ = nullptr;
  }
}

Status RunContextUtil::CreateRunContext(Model &model, const ComputeGraphPtr &graph, Buffer &buffer,
                                        const uint64_t session_id) {
  GELOGI("Begin to Create RunContext, session_id = %lu", session_id);
  // check params
  if (graph == nullptr) {
    GELOGE(PARAM_INVALID, "CreateRunContext param graph is null. session_id=%lu", session_id);
    return PARAM_INVALID;
  }

  uint32_t stream_num = 0;
  if (!AttrUtils::GetInt(&model, ATTR_MODEL_STREAM_NUM, stream_num)) {
    GELOGE(INTERNAL_ERROR, "Get stream_num attr from model_def failed. session_id=%lu", session_id);
    return INTERNAL_ERROR;
  }
  GELOGI("Stream_num = %u", stream_num);

  uint32_t event_num = 0;
  if (!AttrUtils::GetInt(&model, ATTR_MODEL_EVENT_NUM, event_num)) {
    GELOGE(INTERNAL_ERROR, "Get event_num attr from model failed. session_id=%lu", session_id);
    return INTERNAL_ERROR;
  }
  GELOGI("Event_num = %u", event_num);

  Status ret = CreateRtModelStreamsAndEvents(stream_num, event_num);
  if (ret != SUCCESS) {
    GELOGE(ret, "CreateRtModelStreamsAndEvents failed. session_id=%lu", session_id);
    DestroyRtModelStreamAndEvents();
    return ret;
  }

  run_context_ = {rt_model_,        nullptr,          session_id, data_mem_size_, data_mem_base_,
                  weight_mem_size_, weight_mem_base_, buffer,     stream_list_,   event_list_};

  return SUCCESS;
}

RunContext &RunContextUtil::GetRunContext() { return run_context_; }
}  // namespace ge
