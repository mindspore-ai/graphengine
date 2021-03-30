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

#include "graph/load/model_manager/task_info/stream_switch_task_info.h"

#include <vector>

#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
const uint32_t kTrueBranchStreamNum = 1;
}  // namespace

Status StreamSwitchTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("StreamSwitchTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr when StreamSwitchTaskInfo %s", __FUNCTION__);
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
  OpDescPtr op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  auto input_data_addr = ModelUtils::GetInputDataAddrs(davinci_model->GetRuntimeParam(), op_desc);
  SetInputAndValuePtr(davinci_model, input_data_addr);
  uint32_t cond = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_STREAM_SWITCH_COND, cond)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail when StreamSwitchTaskInfo %s",
                       ATTR_NAME_STREAM_SWITCH_COND.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), __FUNCTION__);
    GELOGE(INTERNAL_ERROR, "StreamSwitchOp get attr STREAM_SWITCH_COND fail.");
    return INTERNAL_ERROR;
  }
  cond_ = static_cast<rtCondition_t>(cond);

  size_t input_size = op_desc->GetInputsSize();
  if (input_data_addr.size() != STREAM_SWITCH_INPUT_NUM || input_size != STREAM_SWITCH_INPUT_NUM) {
    REPORT_INNER_ERROR("E19999", "input_data_addr.size():%zu or input size:%zu != STREAM_SWITCH_INPUT_NUM:%u "
                       "in op:%s(%s), check invalid when StreamSwitchTaskInfo %s", input_data_addr.size(), input_size,
                       STREAM_SWITCH_INPUT_NUM, op_desc->GetName().c_str(), op_desc->GetType().c_str(), __FUNCTION__);
    GELOGE(INTERNAL_ERROR, "Input num should be %u. inputAddr size:%zu, inputDesc size:%zu.",
           STREAM_SWITCH_INPUT_NUM, input_data_addr.size(), input_size);
    return INTERNAL_ERROR;
  }

  vector<uint32_t> active_stream_list;
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail when StreamSwitchTaskInfo %s",
                       ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), __FUNCTION__);
    GELOGE(INTERNAL_ERROR, "StreamSwitchOp get attr ACTIVE_STREAM_LIST fail.");
    return INTERNAL_ERROR;
  }

  if (active_stream_list.size() != kTrueBranchStreamNum) {
    REPORT_INNER_ERROR("E19999", "active_stream_list.size():%zu in op:%s(%s) != kTrueBranchStreamNum:%u, "
                       "check invalid when StreamSwitchTaskInfo %s", active_stream_list.size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), kTrueBranchStreamNum, __FUNCTION__);
    GELOGE(FAILED, "Stream num of switch true branch must be %u.", kTrueBranchStreamNum);
    return FAILED;
  }

  size_t true_stream_index = active_stream_list.front();
  if (true_stream_index >= davinci_model->GetStreamList().size()) {
    REPORT_INNER_ERROR("E19999", "active_stream_index:%zu in op:%s(%s) >= stream list size:%zu in model,"
                       "check invalid when StreamSwitchTaskInfo %s", true_stream_index,
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), davinci_model->GetStreamList().size(),
                       __FUNCTION__);
    GELOGE(INTERNAL_ERROR, "InitStreamSwitchTaskInfo stream index invalid. index:%zu, stream list size:%zu.",
           true_stream_index, davinci_model->GetStreamList().size());
    return INTERNAL_ERROR;
  }

  true_stream_ = davinci_model->GetStreamList()[true_stream_index];
  true_stream_id_ = stream_switch_def.true_stream_id();
  davinci_model->DisableZeroCopy(input_ptr_);
  davinci_model->DisableZeroCopy(value_ptr_);

  if (op_desc->HasAttr(ATTR_NAME_SWITCH_DATA_TYPE)) {
    int64_t data_type = 0;
    if (!AttrUtils::GetInt(op_desc, ATTR_NAME_SWITCH_DATA_TYPE, data_type)) {
      REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail when StreamSwitchTaskInfo %s",
                         ATTR_NAME_SWITCH_DATA_TYPE.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), __FUNCTION__);
      GELOGE(FAILED, "StreamSwitchOp[node:%s] get attr SWITCH_DATA_TYPE fail.", op_desc->GetName().c_str());
      return FAILED;
    }
    data_type_ = static_cast<rtSwitchDataType_t>(data_type);
  }

  GELOGI("InitStreamSwitchTaskInfo Init Success, cond:%d, trueStream:%p, trueStreamID:%u, datatype:%d.",
         cond_, true_stream_, true_stream_id_, data_type_);

  return SUCCESS;
}

Status StreamSwitchTaskInfo::Distribute() {
  GELOGI("StreamSwitchTaskInfo Distribute Start.");
  rtError_t rt_ret = rtStreamSwitchEx(input_ptr_, cond_, value_ptr_, true_stream_, stream_, data_type_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamSwitchEx fail, ret:0x%X, when StreamSwitchTaskInfo %s",
                      rt_ret, __FUNCTION__);
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("StreamSwitchTaskInfo Distribute Success. cond:%d, stream:%p, datatype:%d.", cond_, true_stream_, data_type_);
  return SUCCESS;
}
Status StreamSwitchTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GE_CHECK_NOTNULL(davinci_model);
  auto stream_switch_def = task_def.stream_switch();
  uint32_t op_index = stream_switch_def.op_index();
  GELOGI("Begin to calculate args, op_index is: %u", op_index);
  auto op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Calc opType[%s] args size. Node name is [%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  if (op_desc->GetInputsSize() != STREAM_SWITCH_INPUT_NUM) {
    REPORT_INNER_ERROR("E19999", "input size:%zu in op:%s(%s) != STREAM_SWITCH_INPUT_NUM:%u,"
                       "check invalid when StreamSwitchTaskInfo %s", op_desc->GetInputsSize(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), STREAM_SWITCH_INPUT_NUM, __FUNCTION__);
    GELOGE(FAILED, "Stream switch op only have one data input. Now input size is %zu", op_desc->GetInputsSize());
    return FAILED;
  }
  for (uint32_t i = 0; i < STREAM_SWITCH_INPUT_NUM; ++i) {
    string input_tensor_name = op_desc->GetInputNameByIndex(i);
    int64_t fixed_addr_offset = davinci_model->GetFixedAddrsSize(input_tensor_name);
    fixed_addr_offset_.emplace_back(fixed_addr_offset);
    auto tensor_desc = op_desc->GetInputDesc(i);
    int64_t tensor_size = 0;
    GE_CHK_STATUS(TensorUtils::GetSize(tensor_desc, tensor_size));
    davinci_model->SetTotalFixedAddrsSize(input_tensor_name, tensor_size);
    GELOGI("Calculate stream switch task args , tensor size is %ld, fixed addr[%u] offset %ld", tensor_size, i,
           fixed_addr_offset);
  }
  return SUCCESS;
}

void StreamSwitchTaskInfo::SetInputAndValuePtr(DavinciModel *davinci_model, const vector<void *> &input_data_addrs) {
  if (davinci_model->IsKnownNode() && fixed_addr_offset_.size() == STREAM_SWITCH_INPUT_NUM) {
    input_ptr_ = davinci_model->GetCurrentFixedAddr(fixed_addr_offset_[0]);
    value_ptr_ = davinci_model->GetCurrentFixedAddr(fixed_addr_offset_[1]);
  } else {
    if (!input_data_addrs.empty() && input_data_addrs.size() >= STREAM_SWITCH_INPUT_NUM) {
      input_ptr_ = input_data_addrs[0];
      value_ptr_ = input_data_addrs[1];
    }
  }
}
REGISTER_TASK_INFO(RT_MODEL_TASK_STREAM_SWITCH, StreamSwitchTaskInfo);
}  // namespace ge
