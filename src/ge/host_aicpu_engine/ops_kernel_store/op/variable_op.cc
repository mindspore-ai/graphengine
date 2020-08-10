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

#include "host_aicpu_engine/ops_kernel_store/op/variable_op.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "host_aicpu_engine/ops_kernel_store/op/op_factory.h"

namespace {
const size_t kInputSize = 1;
}

namespace ge {
namespace host_aicpu {
Status VariableOp::Compute(const ge::OpDescPtr &op_desc_ptr, const std::vector<ge::GeTensorPtr> &inputs,
                           std::vector<ge::GeTensorPtr> &outputs) {
  GELOGI("VariableOp [%s, %s] compute begin.", node_.GetName().c_str(), node_.GetType().c_str());
  if (inputs.size() != kInputSize) {
    GELOGE(PARAM_INVALID, "Number of input for VariableOp must be %zu.", kInputSize);
    return PARAM_INVALID;
  }
  GeTensorPtr output_ptr =
    MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0), inputs[0]->GetData().GetData(), inputs[0]->GetData().GetSize());
  GE_CHECK_NOTNULL(output_ptr);
  outputs.push_back(output_ptr);
  GELOGI("VariableOp [%s, %s] compute success.", node_.GetName().c_str(), node_.GetType().c_str());
  return SUCCESS;
}

REGISTER_OP_CREATOR(Variable, VariableOp);
REGISTER_OP_CREATOR(Constant, VariableOp);
}  // namespace host_aicpu
}  // namespace ge
