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

#include "hybrid/node_executor/host_cpu/kernel/variable_kernel.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "hybrid/node_executor/host_cpu/kernel_factory.h"

namespace ge {
namespace hybrid {
namespace host_cpu {
Status VariableKernel::Compute(TaskContext& context) {
  GELOGI("[%s] compute begin.", node_->GetName().c_str());

  auto tensor = context.GetVariable(node_->GetName());
  if (tensor == nullptr) {
    GELOGE(PARAM_INVALID, "tensor is NULL.");
    return PARAM_INVALID;
  }
  // Constant & Variable Op has and only has one output
  GE_CHK_STATUS_RET(context.SetOutput(0, *tensor), "[%s] Failed to set output.", context.GetNodeName());
  GELOGI("[%s] compute success.", node_->GetName().c_str());
  return SUCCESS;
}

REGISTER_KERNEL_CREATOR(Variable, VariableKernel);
REGISTER_KERNEL_CREATOR(Constant, VariableKernel);
}  // namespace host_cpu
}  // namespace hybrid
}  // namespace ge
