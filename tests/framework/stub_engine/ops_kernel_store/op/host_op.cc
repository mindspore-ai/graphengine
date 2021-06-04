/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "inc/st_types.h"
#include "stub_engine/ops_kernel_store/op/host_op.h"
#include "framework/common/util.h"
#include "stub_engine/ops_kernel_store/op/stub_op_factory.h"

namespace ge {
namespace st {
Status HostOp::Run() {
  // no need to generate device task
  return SUCCESS;
}
REGISTER_OP_CREATOR(Enter, RTSLib, HostOp);
REGISTER_OP_CREATOR(Merge, RTSLib, HostOp);
REGISTER_OP_CREATOR(Switch, RTSLib, HostOp);
REGISTER_OP_CREATOR(Less, AiCoreLib, HostOp);
REGISTER_OP_CREATOR(NextIteration, AiCoreLib, HostOp);
REGISTER_OP_CREATOR(LoopCond, RTSLib, HostOp);
REGISTER_OP_CREATOR(Exit, RTSLib, HostOp);
REGISTER_OP_CREATOR(StreamMerge, RTSLib, HostOp);
REGISTER_OP_CREATOR(StreamSwitch, RTSLib, HostOp);
REGISTER_OP_CREATOR(StreamActive, RTSLib, HostOp);
REGISTER_OP_CREATOR(Cast, AiCoreLib, HostOp);
REGISTER_OP_CREATOR(Transdata, AiCoreLib, HostOp);
}  // namespace st
}  // namespace ge
