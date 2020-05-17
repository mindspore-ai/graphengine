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

#include "graph/load/new_model_manager/task_info/stream_switch_task_info.h"

#include <vector>

#include "framework/common/debug/ge_log.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
const uint32_t kTrueBranchStreamNum = 1;
}  // namespace

Status StreamSwitchTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("StreamSwitchTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return FAILED;
  }

  auto stream_switch_def = task_def.stream_switch();
  uint32_t op_index = stream_switch_def.op_index();

  // get StreamSwitch op
  auto op_desc = davinci_model->GetOpList()[op_index];
  GE_CHECK_NOTNULL(op_desc);
  auto input_data_addr = ModelUtils::GetInputDataAddrs(davinci_model->GetRuntimeParam(), op_desc);
  if (!input_data_addr.empty() && input_data_addr.size() >= domi::STREAM_SWITCH_INPUT_NUM) {
    input_ptr_ = input_data_addr[0];
    value_ptr_ = input_data_addr[1];
  }

  uint32_t cond = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_STREAM_SWITCH_COND, cond)) {
    GELOGE(INTERNAL_ERROR, "StreamSwitchOp get attr STREAM_SWITCH_COND fail.");
    return INTERNAL_ERROR;
  }
  cond_ = static_cast<rtCondition_t>(cond);

  size_t input_size = op_desc->GetInputsSize();
  if (input_data_addr.size() != domi::STREAM_SWITCH_INPUT_NUM || input_size != domi::STREAM_SWITCH_INPUT_NUM) {
    GELOGE(INTERNAL_ERROR, "Input num should be %u inputAddr size:%zu, inputDesc size:%zu.",
           domi::STREAM_SWITCH_INPUT_NUM, input_data_addr.size(), input_size);
    return INTERNAL_ERROR;
  }

  vector<uint32_t> active_stream_list;
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list)) {
    GELOGE(INTERNAL_ERROR, "StreamSwitchOp get attr ACTIVE_STREAM_LIST fail.");
    return INTERNAL_ERROR;
  }

  if (active_stream_list.size() != kTrueBranchStreamNum) {
    GELOGE(FAILED, "Stream num of switch true branch must be %u.", kTrueBranchStreamNum);
    return FAILED;
  }

  size_t true_stream_index = active_stream_list.front();
  if (true_stream_index >= davinci_model->GetStreamList().size()) {
    GELOGE(INTERNAL_ERROR, "InitStreamSwitchTaskInfo stream index invalid. index:%zu, stream list size:%zu.",
           true_stream_index, davinci_model->GetStreamList().size());
    return INTERNAL_ERROR;
  }

  true_stream_ = davinci_model->GetStreamList()[true_stream_index];
  true_stream_id_ = stream_switch_def.true_stream_id();

  if (op_desc->HasAttr(ATTR_NAME_SWITCH_DATA_TYPE)) {
    int64_t data_type = 0;
    if (!AttrUtils::GetInt(op_desc, ATTR_NAME_SWITCH_DATA_TYPE, data_type)) {
      GELOGE(FAILED, "StreamSwitchOp[node:%s] get attr SWITCH_DATA_TYPE fail.", op_desc->GetName().c_str());
      return FAILED;
    }
    data_type_ = static_cast<rtSwitchDataType_t>(data_type);
  }

  GELOGI("InitStreamSwitchTaskInfo Init Success, cond:%d, trueStream:%p, trueStreamID:%u, datatype:%d.", cond_,
         true_stream_, true_stream_id_, data_type_);

  return SUCCESS;
}

Status StreamSwitchTaskInfo::Distribute() {
  GELOGI("StreamSwitchTaskInfo Distribute Start.");
  rtError_t rt_ret = rtStreamSwitchEx(input_ptr_, cond_, value_ptr_, true_stream_, stream_, data_type_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  GELOGI("StreamSwitchTaskInfo Distribute Success. cond:%d, stream:%p, datatype:%d.", cond_, true_stream_, data_type_);
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_STREAM_SWITCH, StreamSwitchTaskInfo);
}  // namespace ge
