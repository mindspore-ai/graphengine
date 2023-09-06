/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef AIR_CXX_BASE_COMMON_GLOBAL_VARIABLES_DIAGNOSE_SWITCH_H_
#define AIR_CXX_BASE_COMMON_GLOBAL_VARIABLES_DIAGNOSE_SWITCH_H_
#include <cstdint>
#include <map>
#include <mutex>
#include "framework/common/debug/ge_log.h"
#include "runtime/subscriber/built_in_subscriber_definitions.h"

namespace ge {
struct DiagnoseSwitchHandler {
  void *arg;
  void (*on_event)(void *arg, uint64_t event);
};
class SingleDiagnoseSwitch {
 public:
  uint64_t GetEnableFlag() const {
    return enable_flag_;
  }
  void SetEnableFlag(uint64_t enable_flags) {
    if (enable_flag_ != enable_flags) {
      enable_flags == 0UL ? enable_flag_ = 0UL : enable_flag_ |= enable_flags;
      PublishEvent(enable_flag_);
    }
  }
  void RegisterHandler(const void *key, DiagnoseSwitchHandler handler) {
    if (handler.on_event == nullptr) {
      GELOGW("Register diagnose handler use a nullptr, ignore it");
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_lock_);
    if (!keys_to_handler_.emplace(key, handler).second) {
      return;
    }
    handler.on_event(handler.arg, GetEnableFlag());
  }
  void UnregisterHandler(const void * const key) {
    const std::lock_guard<std::mutex> lock(mutex_lock_);
    (void)keys_to_handler_.erase(key);
  }
  size_t GetHandleSize() const {
    std::lock_guard<std::mutex> lock(mutex_lock_);
    return keys_to_handler_.size();
  }

 private:
  void PublishEvent(uint64_t enable_flags) {
    std::lock_guard<std::mutex> lock(mutex_lock_);
    for (const auto &key_to_handler : keys_to_handler_) {
      auto &handler = key_to_handler.second;
      handler.on_event(handler.arg, enable_flags);
    }
  }

 private:
  uint64_t enable_flag_{0UL};
  std::map<const void *, DiagnoseSwitchHandler> keys_to_handler_;
  mutable std::mutex mutex_lock_;
};
namespace diagnoseSwitch {
SingleDiagnoseSwitch &MutableProfiling();

const SingleDiagnoseSwitch &GetProfiling();

SingleDiagnoseSwitch &MutableDumper();

const SingleDiagnoseSwitch &GetDumper();

void EnableDataDump();

void EnableOverflowDump();

void EnableExceptionDump();

void EnableHostDump();

void EnableTrainingTrace();

void EnableGeHostProfiling();

void EnableDeviceProfiling();

void EnableTaskTimeProfiling();

void EnableCannHostProfiling();

void EnableMemoryProfiling();

void EnableProfiling(const std::vector<gert::ProfilingType> &prof_type);

void DisableProfiling();

void DisableDumper();
}  // namespace diagnoseSwitch
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_GLOBAL_VARIABLES_DIAGNOSE_SWITCH_H_
