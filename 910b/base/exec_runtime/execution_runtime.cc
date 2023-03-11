/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "exec_runtime/execution_runtime.h"
#include "mmpa/mmpa_api.h"
#include "runtime/rt.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"

namespace ge {
namespace {
constexpr const char_t *kHeterogeneousRuntimeLibName = "libmodel_deployer.so";
constexpr const char_t *kResourceConfigPath = "RESOURCE_CONFIG_PATH";
constexpr const char_t *kEsClusterConfigPath = "ESCLUSTER_CONFIG_PATH";
}
std::mutex ExecutionRuntime::mu_;
bool ExecutionRuntime::heterogeneous_enabled_ = true;
bool ExecutionRuntime::deploy_with_flow_ = true;
void *ExecutionRuntime::handle_;
std::shared_ptr<ExecutionRuntime> ExecutionRuntime::instance_;

void ExecutionRuntime::SetExecutionRuntime(const std::shared_ptr<ExecutionRuntime> &instance) {
  const std::lock_guard<std::mutex> lk(mu_);
  instance_ = instance;
}

ExecutionRuntime *ExecutionRuntime::GetInstance() {
  const std::lock_guard<std::mutex> lk(mu_);
  return instance_.get();
}

Status ExecutionRuntime::InitHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  if (LoadHeterogeneousLib() != ge::SUCCESS) {
    FinalizeExecutionRuntime();
    return FAILED;
  }
  if (SetupHeterogeneousRuntime(options) != ge::SUCCESS) {
    FinalizeExecutionRuntime();
    return FAILED;
  }
  return SUCCESS;
}

Status ExecutionRuntime::LoadHeterogeneousLib() {
  const auto open_flag =
      static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL));
  handle_ = mmDlopen(kHeterogeneousRuntimeLibName, open_flag);
  if (handle_ == nullptr) {
    const auto *error_msg = mmDlerror();
    GE_IF_BOOL_EXEC(error_msg == nullptr, error_msg = "unknown error");
    GELOGE(FAILED, "[Dlopen][So] failed, so name = %s, error_msg = %s", kHeterogeneousRuntimeLibName, error_msg);
    return FAILED;
  }
  GELOGD("Open %s succeeded", kHeterogeneousRuntimeLibName);
  return SUCCESS;
}

Status ExecutionRuntime::SetupHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  using InitFunc = Status(*)(const std::map<std::string, std::string> &);
  const auto init_func = reinterpret_cast<InitFunc>(mmDlsym(handle_, "InitializeHelperRuntime"));
  if (init_func == nullptr) {
    GELOGE(FAILED, "[Dlsym] failed to find function: InitializeHelperRuntime");
    return FAILED;
  }
  GE_CHK_STATUS_RET(init_func(options), "Failed to invoke InitializeHelperRuntime");
  return SUCCESS;
}

void ExecutionRuntime::FinalizeExecutionRuntime() {
  const auto instance = GetInstance();
  if (instance != nullptr) {
    (void) instance->Finalize();
    instance_ = nullptr;
  }

  if (handle_ != nullptr) {
    GELOGD("close so: %s", kHeterogeneousRuntimeLibName);
    (void) mmDlclose(handle_);
    handle_ = nullptr;
  }
}

void ExecutionRuntime::UpdateGraphOptions(const std::string &key, const std::string &value) {
  auto graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[key] = value;
  GetThreadLocalContext().SetGraphOption(graph_options);
}

bool ExecutionRuntime::IsHeterogeneous() {
  if (handle_ != nullptr) {
    return true;
  }

  int32_t heterogeneous_flag = 0;
  if (rtGetIsHeterogenous(&heterogeneous_flag) == RT_ERROR_NONE &&
    heterogeneous_flag == ge::ExecutionRuntime::kRuntimeTypeHeterogeneous) {
    GELOGI("Get is heterogenous flag success, heterogeneous_flag = %d", heterogeneous_flag);
    return true;
  }

  constexpr size_t kMaxPathLen = 1024UL;
  char_t file_path[kMaxPathLen]{};
  int32_t ret = mmGetEnv(kResourceConfigPath, file_path, kMaxPathLen);
  if (ret == EN_OK) {
    GELOGI("Get resource config path success, path = %s", file_path);
    UpdateGraphOptions(RESOURCE_CONFIG_PATH, file_path);
    const std::lock_guard<std::mutex> lk(mu_);
    deploy_with_flow_ = false;
    return true;
  }

  ret = mmGetEnv(kEsClusterConfigPath, file_path, kMaxPathLen);
  if (ret == EN_OK) {
    GELOGI("Get embedding service cluster config path success, path = %s", file_path);
    const std::lock_guard<std::mutex> lk(mu_);
    deploy_with_flow_ = false;
    return true;
  }
  return false;
}

bool ExecutionRuntime::IsMbufAllocatorEnabled() {
  const int32_t base = 10;
  const char *enable_mbuf_allocator = std::getenv("ENABLE_MBUF_ALLOCATOR");
  if ((enable_mbuf_allocator != nullptr) &&
      (static_cast<int32_t>(std::strtol(enable_mbuf_allocator, nullptr, base) == 1))) {
    GELOGD("ENABLE_MBUF_ALLOCATOR=[%s].", enable_mbuf_allocator);
    return true;
  }
  return false;
}

bool ExecutionRuntime::IsHeterogeneousEnabled() {
  const std::lock_guard<std::mutex> lk(mu_);
  return heterogeneous_enabled_;
}

bool ExecutionRuntime::DeployWithFlow() {
  const std::lock_guard<std::mutex> lk(mu_);
  return deploy_with_flow_;
}

void ExecutionRuntime::DisableHeterogeneous() {
  const std::lock_guard<std::mutex> lk(mu_);
  heterogeneous_enabled_ = false;
}

bool ExecutionRuntime::IsInHeterogeneousExecutor() {
  // heterogeneous is disable iff. in executor
  return !IsHeterogeneousEnabled();
}
}  // namespace ge