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

#include "host_aicpu_engine/ops_kernel_store/op/host_op.h"
#include "framework/common/util.h"
#include "host_aicpu_engine/ops_kernel_store/op/op_factory.h"

namespace ge {
namespace host_aicpu {
Status HostOp::Run() {
  // no need to generate device task
  return SUCCESS;
}

REGISTER_OP_CREATOR(NoOp, HostOp);
REGISTER_OP_CREATOR(Variable, HostOp);
REGISTER_OP_CREATOR(Constant, HostOp);
REGISTER_OP_CREATOR(Assign, HostOp);
REGISTER_OP_CREATOR(RandomUniform, HostOp);
}  // namespace host_aicpu
}  // namespace ge
