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

#ifndef GE_HYBRID_HOST_CPU_KERNEL_VARIABLE_KERNEL_H_
#define GE_HYBRID_HOST_CPU_KERNEL_VARIABLE_KERNEL_H_

#include "hybrid/node_executor/host_cpu/kernel/kernel.h"

namespace ge {
namespace hybrid {
namespace host_cpu {
class VariableKernel : public Kernel {
 public:
  VariableKernel(const NodePtr &node) : Kernel(node) {}
  ~VariableKernel() override = default;
  VariableKernel &operator=(const VariableKernel &op) = delete;
  VariableKernel(const VariableKernel &op) = delete;

  /**
   *  @brief compute for node_task.
   *  @return result
   */
  Status Compute(TaskContext& context) override;
};
}  // namespace host_cpu
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_HOST_CPU_KERNEL_VARIABLE_KERNEL_H_
