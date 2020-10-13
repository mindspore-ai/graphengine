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

#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
Status LabelSetTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("LabelSetTaskInfo Init Start.");
  GE_CHECK_NOTNULL(davinci_model);

  if (SetStream(task_def.stream_id(), davinci_model->GetStreamList()) != SUCCESS) {
    return FAILED;
  }

  // Get LabelSet task def
  const domi::LabelSetDef &label_set = task_def.label_set();
  OpDescPtr op_desc = davinci_model->GetOpByIndex(label_set.op_index());
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Task op index:%u out of range!", label_set.op_index());
    return INTERNAL_ERROR;
  }

  uint32_t label_index = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, label_index)) {
    GELOGE(INTERNAL_ERROR, "LabelSetTaskInfo: %s attr [%s] not exist.",
           op_desc->GetName().c_str(), ATTR_NAME_LABEL_SWITCH_INDEX.c_str());
    return INTERNAL_ERROR;
  }

  const vector<rtLabel_t> &label_list = davinci_model->GetLabelList();
  if (label_index >= label_list.size()) {
    GELOGE(INTERNAL_ERROR, "LabelSetTaskInfo: Invalid label id:%u, label size:%zu", label_index, label_list.size());
    return INTERNAL_ERROR;
  }
  label_ = label_list[label_index];

  GELOGI("LabelSetTaskInfo Init success, label id:%u, label:%p.", label_index, label_);
  return SUCCESS;
}

Status LabelSetTaskInfo::Distribute() {
  GELOGI("LabelSetTaskInfo Distribute Start.");
  rtError_t rt_ret = rtLabelSet(label_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("LabelSetTaskInfo Distribute Success.");
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_LABEL_SET, LabelSetTaskInfo);
}  // namespace ge
