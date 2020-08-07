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

#include "host_aicpu_engine/engine/host_aicpu_engine.h"
#include <map>
#include <memory>
#include <string>
#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"
#include "host_aicpu_engine/common/constant/constant.h"
#include "host_aicpu_engine/ops_kernel_store/host_aicpu_ops_kernel_info.h"

namespace ge {
namespace host_aicpu {
HostAiCpuEngine &HostAiCpuEngine::Instance() {
  static HostAiCpuEngine instance;
  return instance;
}

Status HostAiCpuEngine::Initialize(const std::map<string, string> &options) {
  if (ops_kernel_store_ == nullptr) {
    ops_kernel_store_ = MakeShared<HostAiCpuOpsKernelInfoStore>();
    if (ops_kernel_store_ == nullptr) {
      GELOGE(FAILED, "Make HostAiCpuOpsKernelInfoStore failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

void HostAiCpuEngine::GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  if (ops_kernel_store_ != nullptr) {
    // add buildin opsKernel to opsKernelInfoMap
    ops_kernel_map[kHostAiCpuOpKernelLibName] = ops_kernel_store_;
  }
}

void HostAiCpuEngine::GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &) {
  // no optimizer for host aicpu engine
}

Status HostAiCpuEngine::Finalize() {
  ops_kernel_store_ = nullptr;
  return SUCCESS;
}
}  // namespace host_aicpu
}  // namespace ge

ge::Status Initialize(const std::map<string, string> &options) {
  return ge::host_aicpu::HostAiCpuEngine::Instance().Initialize(options);
}

void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map) {
  ge::host_aicpu::HostAiCpuEngine::Instance().GetOpsKernelInfoStores(ops_kernel_map);
}

void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers) {
  ge::host_aicpu::HostAiCpuEngine::Instance().GetGraphOptimizerObjs(graph_optimizers);
}

ge::Status Finalize() { return ge::host_aicpu::HostAiCpuEngine::Instance().Finalize(); }
