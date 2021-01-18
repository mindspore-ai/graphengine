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

#include "hybrid/node_executor/host_cpu/kernel/data_kernel.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "hybrid/node_executor/host_cpu/kernel_factory.h"

namespace {
constexpr size_t kDataInputIndex = 0;
constexpr size_t kDataOutputIndex = 0;
}

namespace ge {
namespace hybrid {
namespace host_cpu {
Status DataKernel::Compute(TaskContext& context) {
  auto input = context.MutableInput(kDataInputIndex);
  GE_CHECK_NOTNULL(input);
  GE_CHK_STATUS_RET(context.SetOutput(kDataOutputIndex, *input), "[%s] Failed to set output.", context.GetNodeName())
  GELOGD("[%s] compute success.", node_->GetName().c_str());
  return SUCCESS;
}

REGISTER_KERNEL_CREATOR(Data, DataKernel);
}  // namespace host_cpu
}  // namespace hybrid
}  // namespace ge
