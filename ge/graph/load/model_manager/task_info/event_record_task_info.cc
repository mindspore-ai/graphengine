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

#include "graph/load/model_manager/task_info/event_record_task_info.h"

#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/davinci_model.h"

namespace ge {
Status EventRecordTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("EventRecordTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr when EventRecordTaskInfo %s", __FUNCTION__);
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  const auto &eventList = davinci_model->GetEventList();
  if (task_def.event_id() >= eventList.size()) {
    REPORT_INNER_ERROR("E19999", "Task event_id:%u > model event size:%zu, check invalid when EventRecordTaskInfo %s",
                       task_def.event_id(), eventList.size(), __FUNCTION__);
    GELOGE(INTERNAL_ERROR, "event list size:%zu, cur:%u!", eventList.size(), task_def.event_id());
    return INTERNAL_ERROR;
  }

  event_ = eventList[task_def.event_id()];

  return SUCCESS;
}

Status EventRecordTaskInfo::Distribute() {
  GELOGI("EventRecordTaskInfo Distribute Start.");
  rtError_t rt_ret = rtEventRecord(event_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtEventRecord fail ret:0x%X, when EventRecordTaskInfo %s", rt_ret, __FUNCTION__);
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_EVENT_RECORD, EventRecordTaskInfo);
}  // namespace ge
