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

#include "ge_default_running_env.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"

FAKE_NS_BEGIN
namespace {
std::vector<FakeEngine> default_engines = {FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib"),
                                           FakeEngine("VectorEngine").KernelInfoStore("VectorLib"),
                                           FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib"),
                                           FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib"),
                                           FakeEngine("DNN_HCCL").KernelInfoStore("HcclLib"),
                                           FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib")};

std::vector<FakeOp> fake_ops = {
  FakeOp(ENTER).InfoStoreAndBuilder("RTSLib"),        FakeOp(MERGE).InfoStoreAndBuilder("RTSLib"),
  FakeOp(SWITCH).InfoStoreAndBuilder("RTSLib"),       FakeOp(LOOPCOND).InfoStoreAndBuilder("RTSLib"),
  FakeOp(STREAMMERGE).InfoStoreAndBuilder("RTSLib"),  FakeOp(STREAMSWITCH).InfoStoreAndBuilder("RTSLib"),
  FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"), FakeOp(EXIT).InfoStoreAndBuilder("RTSLib"),

  FakeOp(LESS).InfoStoreAndBuilder("AiCoreLib"),      FakeOp(NEXTITERATION).InfoStoreAndBuilder("AiCoreLib"),
  FakeOp(CAST).InfoStoreAndBuilder("AiCoreLib"),      FakeOp(TRANSDATA).InfoStoreAndBuilder("AiCoreLib"),
  FakeOp(NOOP).InfoStoreAndBuilder("AiCoreLib"),      FakeOp(VARIABLE).InfoStoreAndBuilder("AiCoreLib"),
  FakeOp(CONSTANT).InfoStoreAndBuilder("AiCoreLib"),  FakeOp(ASSIGN).InfoStoreAndBuilder("AiCoreLib"),
  FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib"),       FakeOp(MUL).InfoStoreAndBuilder("AiCoreLib"),
  FakeOp(DATA).InfoStoreAndBuilder("AiCoreLib"),      FakeOp(NETOUTPUT).InfoStoreAndBuilder("AiCoreLib"),

};
}  // namespace

void GeDefaultRunningEnv::InstallTo(GeRunningEnvFaker& ge_env) {
  for (auto& fake_engine : default_engines) {
    ge_env.Install(fake_engine);
  }

  for (auto& fake_op : fake_ops) {
    ge_env.Install(fake_op);
  }
}

FAKE_NS_END