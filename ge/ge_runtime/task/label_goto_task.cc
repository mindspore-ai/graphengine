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

#include "ge_runtime/task/label_goto_task.h"
#include "ge_runtime/task/task_factory.h"
#include "framework/common/util.h"

namespace ge {
namespace model_runner {
LabelGotoTask::LabelGotoTask(const ModelContext &model_context, const std::shared_ptr<LabelGotoTaskInfo> &task_info)
    : TaskRepeater<LabelGotoTaskInfo>(model_context, task_info), task_info_(task_info) {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
    return;
  }
  auto stream_list = model_context.stream_list();
  auto label_list = model_context.label_list();
  uint32_t stream_id = task_info->stream_id();
  uint32_t label_id = task_info->label_id();
  GELOGI("Stream list size:%zu, stream id:%u.", stream_list.size(), stream_id);
  GELOGI("Label list size:%zu, label id:%u.", label_list.size(), label_id);
  if (stream_id >= stream_list.size() || label_id >= label_list.size()) {
    GELOGW("Stream/Label id invalid.");
    return;
  }
  stream_ = stream_list[stream_id];
  label_ = label_list[label_id];
}

LabelGotoTask::~LabelGotoTask() {
  GE_FREE_RT_LOG(label_info_);
  GE_FREE_RT_LOG(index_value_);
}

bool LabelGotoTask::Distribute() {
  GELOGI("LabelGotoTask Distribute start.");
  if (!CheckParamValid()) {
    return false;
  }

  const std::vector<void *> label_list = { label_ };
  rtError_t rt_ret = rtMalloc(&index_value_, sizeof(uint64_t), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  uint64_t branch_index = 0;
  rt_ret = rtMemcpy(index_value_, sizeof(uint64_t), &branch_index, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  uint32_t label_info_size = sizeof(rtLabelDevInfo) * label_list.size();
  rt_ret = rtMalloc(&label_info_, label_info_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  rt_ret = rtLabelListCpy(label_list.data(), label_list.size(), label_info_, label_info_size);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  rt_ret = rtLabelSwitchByIndex(index_value_, label_list.size(), label_info_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: %#x", rt_ret);
    return false;
  }

  GELOGI("DistributeTask end.");
  return true;
}

bool LabelGotoTask::CheckParamValid() {
  if (stream_ == nullptr) {
    GELOGE(PARAM_INVALID, "stream is null!");
    return false;
  }

  if (label_ == nullptr) {
    GELOGE(PARAM_INVALID, "label is null!");
    return false;
  }

  if (label_info_ != nullptr) {
    GELOGE(PARAM_INVALID, "label_info_ has dirty data.");
    return false;
  }

  if (index_value_ != nullptr) {
    GELOGE(PARAM_INVALID, "index_value_ has dirty data.");
    return false;
  }

  return true;
}

REGISTER_TASK(TaskInfoType::LABEL_GOTO, LabelGotoTask, LabelGotoTaskInfo);
}  // namespace model_runner
}  // namespace ge
