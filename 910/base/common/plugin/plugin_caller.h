/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef GE_COMMON_GE_PLUGIN_CALLER_H_
#define GE_COMMON_GE_PLUGIN_CALLER_H_

#include <mutex>
#include "external/ge/ge_api_types.h"
#include "framework/common/debug/log.h"
#include "common/ge_common/scope_guard.h"
#include "mmpa/mmpa_api.h"
#include "inc/common/checker.h"

namespace ge {
class PluginCaller {
 public:
  explicit PluginCaller(const std::string &lib_name) : handle_(nullptr), lib_name_(lib_name) {}
  ~PluginCaller() = default;
  template <typename RetType, typename FuncType, typename... Args>
  auto CallFunction(const std::string &function_name, Args... args) -> RetType {
    const Status ret = LoadLib();
    GE_CHK_BOOL_RET_SPECIAL_STATUS(ret == NOT_CHANGED, SUCCESS, "[Load][Lib] failed, function_name = %s.",
                                   function_name.c_str());
    GE_ASSERT_SUCCESS(ret);
    GE_MAKE_GUARD(unload_libs, [this]() { UnloadLib(); });
    const auto func = reinterpret_cast<FuncType>(mmDlsym(handle_, function_name.c_str()));
    GE_ASSERT_NOTNULL(func, "Failed to find symbol %s.", function_name.c_str());
    return func(args...);
  }

 private:
  ge::Status LoadLib();
  void UnloadLib();
  void ReportInnerError() const;
  void *handle_;
  std::string lib_name_;
  std::mutex mutex_;
};
}  // namespace ge

#endif  // GE_COMMON_GE_PLUGIN_CALLER_H_
