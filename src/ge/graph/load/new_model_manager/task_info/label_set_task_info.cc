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

#include "graph/load/new_model_manager/task_info/label_set_task_info.h"

#include "framework/common/debug/ge_log.h"
#include "graph/load/new_model_manager/davinci_model.h"

namespace ge {
Status LabelSetTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("LabelSetTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  uint32_t label_id = task_def.label_id();
  if (label_id > davinci_model->BatchNum()) {
    GELOGE(PARAM_INVALID, "labelId is invalid! labelId=%u, labelListSize=%u", label_id, davinci_model->BatchNum());
    return PARAM_INVALID;
  }

  if (!davinci_model->GetLabelList().empty()) {
    label_ = davinci_model->GetLabelList()[label_id];
  }

  return SUCCESS;
}

Status LabelSetTaskInfo::Distribute() {
  GELOGI("LabelSetTaskInfo Distribute Start.");
  rtError_t rt_ret = rtLabelSet(label_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_LABEL_SET, LabelSetTaskInfo);
}  // namespace ge
