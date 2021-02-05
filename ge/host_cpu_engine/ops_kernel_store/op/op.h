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

#ifndef GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_OP_OP_H_
#define GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_OP_OP_H_

#include <climits>
#include <string>
#include <vector>
#include "common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/node.h"

namespace ge {
namespace host_cpu {
/**
 * The base class for all op.
 */
class GE_FUNC_VISIBILITY Op {
 public:
  Op(const Node &node, RunContext &run_context) : run_context_(run_context), node_(node) {}
  virtual ~Op() = default;
  virtual Status Run() = 0;

 protected:
  const RunContext &run_context_;
  const Node &node_;
};
}  // namespace host_cpu
}  // namespace ge

#endif  // GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_OP_OP_H_
