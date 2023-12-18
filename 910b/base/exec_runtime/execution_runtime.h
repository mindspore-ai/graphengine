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
#ifndef BASE_RUNTIME_EXECUTION_RUNTIME_H_
#define BASE_RUNTIME_EXECUTION_RUNTIME_H_

#include <memory>
#include <map>
#include <string>
#include "external/ge/ge_api_error_codes.h"
#include "common/model/model_deploy_resource.h"
#include "exec_runtime/deploy/model_deployer.h"
#include "exec_runtime/deploy/exchange_service.h"

namespace ge {
class ExecutionRuntime {
 public:
  ExecutionRuntime() = default;
  GE_DELETE_ASSIGN_AND_COPY(ExecutionRuntime);
  virtual ~ExecutionRuntime() = default;

  static ExecutionRuntime *GetInstance();

  static Status InitHeterogeneousRuntime(const std::map<std::string, std::string> &options);

  static void FinalizeExecutionRuntime();

  static bool IsHeterogeneous();

  static bool IsHeterogeneousHost();

  static void SetExecutionRuntime(const std::shared_ptr<ExecutionRuntime> &instance);

  static bool IsMbufAllocatorEnabled();

  static bool IsHeterogeneousEnabled();

  static void DisableHeterogeneous();

  static void EnableInHeterogeneousExecutor();

  static void EnableGlobalInHeterogeneousExecutor();

  static bool IsInHeterogeneousExecutor();

  /// Initialize ExecutionRuntime
  /// @param execution_runtime    instance of execution runtime
  /// @param options              options for initialization
  /// @return                     SUCCESS if initialization is successful, otherwise returns appropriate error code
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;

  /// Finalize ExecutionRuntime
  virtual Status Finalize() = 0;

  /// Get ModelDeployer
  /// @return                       pointer to the instance of ModelDeployer
  virtual ModelDeployer &GetModelDeployer() = 0;

  /// Get ExchangeService
  /// @return                       pointer to the instance of ExchangeService
  virtual ExchangeService &GetExchangeService() = 0;

  virtual const std::string &GetCompileHostResourceType() const;

  virtual const std::map<std::string, std::string> &GetCompileDeviceInfo() const;
  static const int32_t kRuntimeTypeHeterogeneous = 1;
 private:
  static Status LoadHeterogeneousLib();
  static Status SetupHeterogeneousRuntime(const std::map<std::string, std::string> &options);
  static void UpdateGraphOptions(const std::string &key, const std::string &value);
  static std::mutex mu_;
  static bool heterogeneous_enabled_;
  thread_local static bool in_heterogeneous_executor_;
  static bool global_in_heterogeneous_executor_;
  static void *handle_;
  static std::shared_ptr<ExecutionRuntime> instance_;
};
}  // namespace ge

#endif  // BASE_RUNTIME_EXECUTION_RUNTIME_H_
