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

#include "hybrid/node_executor/host_cpu/kernel/assign_kernel.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "hybrid/node_executor/host_cpu/kernel_factory.h"

namespace {
const size_t kAssignInputNum = 2;
const size_t kAssignRefInputIndex = 0;
const size_t kAssignValueInputIndex = 1;
const size_t kAssignRefOutputIndex = 0;
}

namespace ge {
namespace hybrid {
namespace host_cpu {
Status AssignKernel::Compute(TaskContext& context) {
  GELOGI("[%s] compute begin.", node_->GetName().c_str());

  auto ref_tensor = context.MutableInput(kAssignRefInputIndex);
  GE_CHECK_NOTNULL(ref_tensor);
  const auto value_tensor = context.GetInput(kAssignValueInputIndex);
  GE_CHECK_NOTNULL(value_tensor);
  if (value_tensor->GetSize() > ref_tensor->GetSize()) {
    GELOGE(INTERNAL_ERROR, "[%s] value_input_size=%zu, but ref_input_size=%zu.",
           node_->GetName().c_str(), value_tensor->GetSize(), ref_tensor->GetSize());
    return INTERNAL_ERROR;
  }

  GELOGI("[%s] value_input_data=%p, ref_input_size=%zu, value_input_size=%zu.",
         node_->GetName().c_str(), ref_tensor->GetData(), ref_tensor->GetSize(), value_tensor->GetSize());
  if (value_tensor->GetSize() > 0) {
    GE_CHK_RT_RET(rtMemcpy(ref_tensor->MutableData(), ref_tensor->GetSize(), value_tensor->GetData(),
                           value_tensor->GetSize(), RT_MEMCPY_HOST_TO_HOST));
  }
  GE_CHK_STATUS_RET(context.SetOutput(kAssignRefOutputIndex, *ref_tensor),
                    "[%s] Failed to set output.", context.GetNodeName());

  GELOGI("[%s] compute success.", node_->GetName().c_str());
  return SUCCESS;
}

REGISTER_KERNEL_CREATOR(Assign, AssignKernel);
}  // namespace host_cpu
}  // namespace hybrid
}  // namespace ge
