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

#include "graph/load/model_manager/task_info/label_goto_ex_task_info.h"

#include "graph/load/model_manager/davinci_model.h"

namespace ge {
constexpr uint8_t kGotoBranchMax = 1;

LabelGotoExTaskInfo::~LabelGotoExTaskInfo() {
  GE_FREE_RT_LOG(args_);
  GE_FREE_RT_LOG(index_value_);
}

Status LabelGotoExTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("LabelGotoExTaskInfo Init Start.");
  GE_CHECK_NOTNULL(davinci_model);

  if (SetStream(task_def.stream_id(), davinci_model->GetStreamList()) != SUCCESS) {
    return FAILED;
  }

  // Get LabelGotoEx task def
  const domi::LabelGotoExDef &label_goto = task_def.label_goto_ex();
  OpDescPtr op_desc = davinci_model->GetOpByIndex(label_goto.op_index());
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Task op index:%u out of range!", label_goto.op_index());
    return INTERNAL_ERROR;
  }

  uint32_t label_index = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, label_index)) {
    GELOGE(INTERNAL_ERROR, "LabelGotoExTaskInfo: %s attr [%s] not exist.",
           op_desc->GetName().c_str(), ATTR_NAME_LABEL_SWITCH_INDEX.c_str());
    return INTERNAL_ERROR;
  }

  const vector<rtLabel_t> &label_list = davinci_model->GetLabelList();
  if (label_index >= label_list.size()) {
    GELOGE(PARAM_INVALID, "LabelGotoExTaskInfo: Invalid label id:%u, label size:%zu", label_index, label_list.size());
    return INTERNAL_ERROR;
  }
  GE_CHECK_NOTNULL(label_list[label_index]);
  vector<rtLabel_t> label_used = { label_list[label_index] };

  rtMemType_t memory_type = op_desc->HasAttr(ATTR_NAME_MEMORY_TYPE_RANGE) ? RT_MEMORY_TS_4G : RT_MEMORY_HBM;
  GELOGI("memory_type: %u", memory_type);
  args_size_ = kGotoBranchMax * sizeof(rtLabelDevInfo);
  rtError_t rt_ret = rtMalloc(&args_, args_size_, memory_type);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMalloc failed, error: %#x", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtLabelListCpy(label_used.data(), label_used.size(), args_, args_size_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtLabelListCpy failed, error: %#x", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtMalloc(&index_value_, sizeof(uint64_t), memory_type);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMalloc failed, error: %#x", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  uint64_t branch_index = 0;
  rt_ret = rtMemcpy(index_value_, sizeof(uint64_t), &branch_index, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMemcpy failed, error: %#x", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("LabelGotoExTaskInfo Init Success, label id:%u, label:%p.", label_index, label_list[label_index]);
  return SUCCESS;
}

Status LabelGotoExTaskInfo::Distribute() {
  GELOGI("LabelGotoExTaskInfo Distribute Start.");
  GE_CHECK_NOTNULL(args_);
  GE_CHECK_NOTNULL(index_value_);
  if (args_size_ == 0) {
    GELOGE(PARAM_INVALID, "branch max: %u, args size: %u invalid.", kGotoBranchMax, args_size_);
    return PARAM_INVALID;
  }

  rtError_t rt_ret = rtLabelSwitchByIndex(index_value_, kGotoBranchMax, args_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("LabelGotoExTaskInfo Distribute Success.");
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_STREAM_LABEL_GOTO, LabelGotoExTaskInfo);
}  // namespace ge
