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

#ifndef GE_HYBRID_HOST_CPU_KERNEL_KERNEL_H_
#define GE_HYBRID_HOST_CPU_KERNEL_KERNEL_H_

#include "common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "hybrid/node_executor/task_context.h"

namespace ge {
namespace hybrid {
namespace host_cpu {
/**
 * The base class for all host_kernel.
 */
class Kernel {
 public:
  Kernel(const NodePtr &node) : node_(node) {}
  virtual ~Kernel() = default;
  virtual Status Compute(TaskContext& context) = 0;

 protected:
  const NodePtr &node_;
};
}  // namespace host_cpu
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_HOST_CPU_KERNEL_KERNEL_H_
