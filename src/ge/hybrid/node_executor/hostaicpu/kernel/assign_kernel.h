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

#ifndef GE_HYBRID_HOST_AICPU_KERNEL_ASSIGN_KERNEL_H_
#define GE_HYBRID_HOST_AICPU_KERNEL_ASSIGN_KERNEL_H_

#include "hybrid/node_executor/hostaicpu/kernel/kernel.h"

namespace ge {
namespace hybrid {
namespace host_aicpu {
class AssignKernel : public Kernel {
 public:
  AssignKernel(const NodePtr &node) : Kernel(node) {}
  ~AssignKernel() override = default;
  AssignKernel &operator=(const AssignKernel &op) = delete;
  AssignKernel(const AssignKernel &op) = delete;

  /**
   *  @brief compute for node_task.
   *  @return result
   */
  Status Compute(TaskContext &context) override;
};
}  // namespace host_aicpu
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_HOST_AICPU_KERNEL_ASSIGN_KERNEL_H_
