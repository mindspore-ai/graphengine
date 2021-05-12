/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common/omg_util.h"

namespace ge {
RunContextUtil::~RunContextUtil() { DestroyRtModelResources(); }

Status RunContextUtil::InitMemInfo(uint8_t *data_mem_base, uint64_t data_mem_size,
                                   std::map<int64_t, uint8_t *> mem_type_to_data_mem_base,
                                   std::map<int64_t, uint64_t> mem_type_to_data_mem_size, uint8_t *weight_mem_base,
                                   uint64_t weight_mem_size) {
  if ((data_mem_size > 0) && (data_mem_base == nullptr)) {
    REPORT_INNER_ERROR("E19999", "InitMemInfo param data_mem_base is null but data_mem_size = %lu", data_mem_size);
    GELOGE(PARAM_INVALID, "[Check][Param] InitMemInfo param data_mem_base is null but data_mem_size = %lu.",
           data_mem_size);
    return PARAM_INVALID;
  }
  if ((weight_mem_size > 0) && (weight_mem_base == nullptr)) {
    REPORT_INNER_ERROR("E19999", "InitMemInfo param weight_mem_base is null but weight_mem_size = %lu",
                       weight_mem_size);
    GELOGE(PARAM_INVALID, "[Check][Param] InitMemInfo param weight_mem_base is null but weight_mem_size = %lu.",
           weight_mem_size);
    return PARAM_INVALID;
  }
  if (mem_type_to_data_mem_base.empty() || mem_type_to_data_mem_size.empty() ||
      mem_type_to_data_mem_base.size() != mem_type_to_data_mem_size.size()) {
    REPORT_INNER_ERROR("E19999", "InitMemInfo param mem_type_to_data_mem_base size[%zu] "
                       "is not equal to the size of mem_type_to_data_mem_size[%zu].",
                       mem_type_to_data_mem_base.size(), mem_type_to_data_mem_size.size());
    GELOGE(PARAM_INVALID,
           "[Check][Param] InitMemInfo param mem_type_to_data_mem_base size[%zu] is not equal to the size of "
           "mem_type_to_data_mem_size[%zu].", mem_type_to_data_mem_base.size(), mem_type_to_data_mem_size.size());
    return PARAM_INVALID;
  }
  data_mem_base_ = data_mem_base;
  data_mem_size_ = data_mem_size;
  weight_mem_base_ = weight_mem_base;
  weight_mem_size_ = weight_mem_size;
  mem_type_to_data_mem_base_ = mem_type_to_data_mem_base;
  mem_type_to_data_mem_size_ = mem_type_to_data_mem_size;
  return SUCCESS;
}

Status RunContextUtil::CreateRtModelResources(uint32_t stream_num, uint32_t event_num, uint32_t label_num) {
  // Create rt model
  rtError_t rt_ret = rtModelCreate(&rt_model_, 0);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "call rtModelCreate failed, ret:%d,", static_cast<int>(rt_ret));
    GELOGE(RT_FAILED, "[Call][RtModelCreate] failed. rt_ret = %d", static_cast<int>(rt_ret));
    return RT_FAILED;
  }

  // Create rt Stream and bind with model
  for (uint32_t i = 0; i < stream_num; ++i) {
    rtStream_t stream = nullptr;
    rt_ret = rtStreamCreate(&stream, 0);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "call rtStreamCreate failed, ret:%d, index:%u,",
                        static_cast<int>(rt_ret), i);
      GELOGE(RT_FAILED, "[Call][RtStreamCreate] failed. rt_ret = %d, index = %u", static_cast<int>(rt_ret), i);
      return RT_FAILED;
    }
    stream_list_.emplace_back(stream);

    rt_ret = rtModelBindStream(rt_model_, stream, 0);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "call rtModelBindStream failed, ret:%d, index:%u,",
                        static_cast<int>(rt_ret), i);
      GELOGE(RT_FAILED, "[Bind][StreamAndModel] failed. rt_ret = %d, index = %u", static_cast<int>(rt_ret), i);
      return RT_FAILED;
    }
  }

  // Create rt event
  uint32_t create_flag = static_cast<uint32_t>((event_num > kEventReuseThreshold) ? RT_EVENT_WITH_FLAG :
                                                                                    RT_EVENT_DEFAULT);
  for (uint32_t i = 0; i < event_num; ++i) {
    rtEvent_t event = nullptr;
    rt_ret = rtEventCreateWithFlag(&event, create_flag);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "call rtEventCreate failed, ret:%d, index:%u,",
                        static_cast<int>(rt_ret), i);
      GELOGE(RT_FAILED, "[Call][RtEventCreate] failed. rt_ret = %d, index = %u", static_cast<int>(rt_ret), i);
      return RT_FAILED;
    }
    event_list_.emplace_back(event);
  }

  // Create rt label
  for (uint32_t i = 0; i < label_num; ++i) {
    rtLabel_t label = nullptr;
    rt_ret = rtLabelCreateV2(&label, rt_model_);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "call rtLabelCreateV2 failed, ret:%d, index:%u,",
                        static_cast<int>(rt_ret), i);
      GELOGE(RT_FAILED, "[Call][RtLabelCreate] failed. rt_ret = %d, index = %u", static_cast<int>(rt_ret), i);
      return RT_FAILED;
    }
    label_list_.emplace_back(label);
  }

  return SUCCESS;
}

void RunContextUtil::DestroyRtModelResources() noexcept {
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

  for (size_t i = 0; i < label_list_.size(); ++i) {
    rt_ret = rtLabelDestroy(label_list_[i]);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Destroy label failed. rt_ret = %d, index = %zu.", static_cast<int>(rt_ret), i);
    }
  }
  label_list_.clear();

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
  GELOGD("Begin to Create RunContext, session_id = %lu", session_id);
  // check params
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param graph nullptr, session_id:%lu,", session_id);
    GELOGE(PARAM_INVALID, "[Check][Param] CreateRunContext param graph is null. session_id=%lu", session_id);
    return PARAM_INVALID;
  }

  uint32_t stream_num = 0;
  if (!AttrUtils::GetInt(&model, ATTR_MODEL_STREAM_NUM, stream_num)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s failed from model, session_id:%lu,",
                       ATTR_MODEL_STREAM_NUM.c_str(), session_id);
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s failed from model. session_id=%lu",
           ATTR_MODEL_STREAM_NUM.c_str(), session_id);
    return INTERNAL_ERROR;
  }
  GELOGD("Stream_num = %u", stream_num);

  uint32_t event_num = 0;
  if (!AttrUtils::GetInt(&model, ATTR_MODEL_EVENT_NUM, event_num)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s failed from model, session_id:%lu,",
                       ATTR_MODEL_EVENT_NUM.c_str(), session_id);
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s failed from model, session_id:%lu,",
           ATTR_MODEL_EVENT_NUM.c_str(), session_id);
    return INTERNAL_ERROR;
  }
  GELOGD("Event_num = %u", event_num);

  uint32_t label_num = 0;
  if (!AttrUtils::GetInt(&model, ATTR_MODEL_LABEL_NUM, label_num)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s failed from model, session_id:%lu,",
                       ATTR_MODEL_LABEL_NUM.c_str(), session_id);
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s failed from model, session_id:%lu,",
           ATTR_MODEL_LABEL_NUM.c_str(), session_id);
    return INTERNAL_ERROR;
  }
  GELOGD("Label_num = %u", label_num);

  Status ret = CreateRtModelResources(stream_num, event_num, label_num);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Create][RtModelResources] failed. session_id=%lu", session_id);
    DestroyRtModelResources();
    return ret;
  }

  GELOGI("CreateRunContext: data_mem_base_ = %p, weight_mem_base_ = %p, memory_size = %lu, weight_size = %lu",
         data_mem_base_, weight_mem_base_, data_mem_size_, weight_mem_size_);

  PrintMemInfo();

  run_context_ = {rt_model_,
                  nullptr,
                  session_id,
                  data_mem_size_,
                  data_mem_base_,
                  mem_type_to_data_mem_size_,
                  mem_type_to_data_mem_base_,
                  weight_mem_size_,
                  weight_mem_base_,
                  buffer,
                  stream_list_,
                  event_list_,
                  label_list_};
  return SUCCESS;
}

void RunContextUtil::PrintMemInfo() {
  for (auto iter : mem_type_to_data_mem_base_) {
    GELOGD("CreateRunContext: memory type = %ld, data memory base = %p", iter.first, iter.second);
  }

  for (auto iter : mem_type_to_data_mem_size_) {
    GELOGD("CreateRunContext: memory type = %ld, data memory size = %lu", iter.first, iter.second);
  }
}

RunContext &RunContextUtil::GetRunContext() { return run_context_; }
}  // namespace ge
