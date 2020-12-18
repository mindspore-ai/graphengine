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

#include "graph/load/new_model_manager/task_info/stream_active_task_info.h"

#include <vector>

#include "framework/common/debug/ge_log.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
Status StreamActiveTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("StreamActiveTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  auto stream_active_def = task_def.stream_active();
  uint32_t op_index = stream_active_def.op_index();

  uint32_t internal_index = davinci_model->GetFlowctrlIndex(op_index);

  // get StreamActive op
  OpDescPtr op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  std::vector<uint32_t> active_stream_index_list;
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_index_list)) {
    GELOGE(INTERNAL_ERROR, "StreamActiveOp get attr ACTIVE_STREAM fail, node name:%s.", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (internal_index >= active_stream_index_list.size()) {
    GELOGE(INTERNAL_ERROR, "InitStreamSwitchTaskInfo stream id index invalid. index:%u, list size:%zu.", internal_index,
           active_stream_index_list.size());
    return INTERNAL_ERROR;
  }

  if (active_stream_index_list[internal_index] >= davinci_model->GetStreamList().size()) {
    GELOGE(INTERNAL_ERROR, "InitStreamSwitchTaskInfo stream index invalid. index:%u, stream list size:%zu.",
           active_stream_index_list[internal_index], davinci_model->GetStreamList().size());
    return INTERNAL_ERROR;
  }

  active_stream_ = davinci_model->GetStreamList()[active_stream_index_list[internal_index]];
  active_stream_id_ = stream_active_def.active_stream_id();
  GELOGI("InitStreamActiveTaskInfo Init Success, index:%u, activeStream:%p, activeStreamID:%u.",
         internal_index, active_stream_, active_stream_id_);

  return SUCCESS;
}

Status StreamActiveTaskInfo::Distribute() {
  GELOGI("StreamActiveTaskInfo Distribute Start.");
  rtError_t rt_ret = rtStreamActive(active_stream_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("StreamActiveTaskInfo Distribute Success. activeStreamID:%p.", active_stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_STREAM_ACTIVE, StreamActiveTaskInfo);
}  // namespace ge
