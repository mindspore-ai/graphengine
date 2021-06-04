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

#include "stub_engine.h"
#include <map>
#include <memory>
#include <string>
#include <securec.h>
#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"
#include "inc/st_types.h"

namespace ge {
namespace st {
StubEngine &StubEngine::Instance() {
  static StubEngine instance;
  return instance;
}

Status StubEngine::Initialize(const std::map<string, string> &options) {
  for (const auto engine_2_lib : kStubEngine2KernelLib) {
    auto ops_kernel_store = MakeShared<StubOpsKernelInfoStore>(engine_2_lib.second);
    if (ops_kernel_store == nullptr) {
      return FAILED;
    }
    ops_kernel_store_map_.insert(make_pair(engine_2_lib.second, ops_kernel_store));
  }
  return SUCCESS;
}

void StubEngine::GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  for (const auto name_2_ops_kernel_store : ops_kernel_store_map_) {
    ops_kernel_map[name_2_ops_kernel_store.first] = name_2_ops_kernel_store.second;
  }
}

void StubEngine::GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &) {
  // no optimizer for host cpu engine
}

Status StubEngine::Finalize() {
  return SUCCESS;
}
}  // namespace st
}  // namespace ge

ge::Status Initialize(const std::map<string, string> &options) {
  return ge::st::StubEngine::Instance().Initialize(options);
}

void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  ge::st::StubEngine::Instance().GetOpsKernelInfoStores(ops_kernel_map);
}

void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers) {
  ge::st::StubEngine::Instance().GetGraphOptimizerObjs(graph_optimizers);
}

ge::Status Finalize() {
  return ge::st::StubEngine::Instance().Finalize();
}
