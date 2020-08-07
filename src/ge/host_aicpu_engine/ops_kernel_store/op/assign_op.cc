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

#include "host_aicpu_engine/ops_kernel_store/op/assign_op.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "host_aicpu_engine/ops_kernel_store/op/op_factory.h"

namespace {
const size_t kAssignInputNum = 2;
const size_t kAssignRefInputIndex = 0;
const size_t kAssignValueInputIndex = 1;
const size_t kAssignRefOutputIndex = 0;
}  // namespace

namespace ge {
namespace host_aicpu {
Status AssignOp::Compute(const ge::OpDescPtr &op_desc_ptr, const std::vector<ge::GeTensorPtr> &inputs,
                         std::vector<ge::GeTensorPtr> &outputs) {
  GELOGI("AssignOp [%s, %s] compute begin.", node_.GetName().c_str(), node_.GetType().c_str());
  if (inputs.size() != kAssignInputNum) {
    GELOGE(PARAM_INVALID, "Number of input for AssignOp must be %zu.", kAssignInputNum);
    return PARAM_INVALID;
  }
  auto &ref_input = inputs[kAssignRefInputIndex];
  const auto &value_input = inputs[kAssignValueInputIndex];
  ref_input->SetData(value_input->GetData().GetData(), value_input->GetData().GetSize());
  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(kAssignRefOutputIndex),
                                                value_input->GetData().GetData(), value_input->GetData().GetSize());
  GE_CHECK_NOTNULL(output_ptr);
  outputs.push_back(output_ptr);
  GELOGI("AssignOp [%s, %s] compute success.", node_.GetName().c_str(), node_.GetType().c_str());
  return SUCCESS;
}

REGISTER_OP_CREATOR(Assign, AssignOp);
}  // namespace host_aicpu
}  // namespace ge
