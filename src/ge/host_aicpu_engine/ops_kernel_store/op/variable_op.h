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

#ifndef GE_HOST_AICPU_ENGINE_OPS_KERNEL_STORE_OP_VARIABLE_OP_H_
#define GE_HOST_AICPU_ENGINE_OPS_KERNEL_STORE_OP_VARIABLE_OP_H_

#include "host_aicpu_engine/ops_kernel_store/op/op.h"

namespace ge {
namespace host_aicpu {
class VariableOp : public Op {
 public:
  VariableOp(const Node &node, RunContext &run_context) : Op(node, run_context) {}
  ~VariableOp() override = default;
  VariableOp &operator=(const VariableOp &op) = delete;
  VariableOp(const VariableOp &op) = delete;

  /**
   *  @brief compute for node_task.
   *  @return result
   */
  Status Compute(const ge::OpDescPtr &op_desc_ptr, const std::vector<ge::GeTensorPtr> &inputs,
                 std::vector<ge::GeTensorPtr> &outputs) override;
};
}  // namespace host_aicpu
}  // namespace ge

#endif  // GE_HOST_AICPU_ENGINE_OPS_KERNEL_STORE_OP_VARIABLE_OP_H_
