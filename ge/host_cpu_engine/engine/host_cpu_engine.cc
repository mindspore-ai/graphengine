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

#include "host_cpu_engine/engine/host_cpu_engine.h"
#include <map>
#include <memory>
#include <string>
#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"
#include "host_cpu_engine/common/constant/constant.h"
#include "host_cpu_engine/ops_kernel_store/host_cpu_ops_kernel_info.h"

namespace ge {
namespace host_cpu {
HostCpuEngine &HostCpuEngine::Instance() {
  static HostCpuEngine instance;
  return instance;
}

Status HostCpuEngine::Initialize(const std::map<string, string> &options) {
  if (ops_kernel_store_ == nullptr) {
    ops_kernel_store_ = MakeShared<HostCpuOpsKernelInfoStore>();
    if (ops_kernel_store_ == nullptr) {
      GELOGE(FAILED, "Make HostCpuOpsKernelInfoStore failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

void HostCpuEngine::GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  if (ops_kernel_store_ != nullptr) {
    // add buildin opsKernel to opsKernelInfoMap
    ops_kernel_map[kHostCpuOpKernelLibName] = ops_kernel_store_;
  }
}

void HostCpuEngine::GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &) {
  // no optimizer for host cpu engine
}

Status HostCpuEngine::Finalize() {
  ops_kernel_store_ = nullptr;
  return SUCCESS;
}
}  // namespace host_cpu
}  // namespace ge

ge::Status Initialize(const std::map<string, string> &options) {
  return ge::host_cpu::HostCpuEngine::Instance().Initialize(options);
}

void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  ge::host_cpu::HostCpuEngine::Instance().GetOpsKernelInfoStores(ops_kernel_map);
}

void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers) {
  ge::host_cpu::HostCpuEngine::Instance().GetGraphOptimizerObjs(graph_optimizers);
}

ge::Status Finalize() { return ge::host_cpu::HostCpuEngine::Instance().Finalize(); }
