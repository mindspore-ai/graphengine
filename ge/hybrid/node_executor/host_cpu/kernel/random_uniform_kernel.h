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

#ifndef GE_HYBRID_HOST_CPU_KERNEL_RANDOM_UNIFORM_KERNEL_H_
#define GE_HYBRID_HOST_CPU_KERNEL_RANDOM_UNIFORM_KERNEL_H_

#include "hybrid/node_executor/host_cpu/kernel/kernel.h"

namespace ge {
namespace hybrid {
namespace host_cpu {
class RandomUniformKernel : public Kernel {
 public:
  RandomUniformKernel(const NodePtr &node) : Kernel(node) {}
  ~RandomUniformKernel() override = default;
  RandomUniformKernel &operator=(const RandomUniformKernel &op) = delete;
  RandomUniformKernel(const RandomUniformKernel &op) = delete;

  /**
   *  @brief compute for node_task.
   *  @return result
   */
  Status Compute(TaskContext& context) override;

 private:
  template <typename T>
  Status Generate(const ge::OpDescPtr &op_desc_ptr, int64_t seed, int64_t seed2, TaskContext& context);

  static Status GenerateFP16(const ge::OpDescPtr &op_desc_ptr, int64_t seed, int64_t seed2, TaskContext& context);
};
}  // namespace host_cpu
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_HOST_CPU_KERNEL_RANDOM_UNIFORM_KERNEL_H_
