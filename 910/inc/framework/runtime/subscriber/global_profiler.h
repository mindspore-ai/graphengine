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
#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_GLOBAL_PROFILER_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_GLOBAL_PROFILER_H_

#include <algorithm>
#include <memory>
#include <unordered_map>
#include "mmpa/mmpa_api.h"
#include "built_in_subscriber_definitions.h"
#include "common/debug/ge_log.h"
#include "framework/common/ge_visibility.h"
#include "runtime/subscriber/executor_subscriber_c.h"

namespace gert {
struct ProfilingData {
  uint64_t name_idx;
  uint64_t type_idx;
  ExecutorEvent event;
  std::chrono::time_point<std::chrono::system_clock> timestamp;
  int64_t thread_id;
};
class GlobalProfiler {
 public:
  GlobalProfiler() = default;
  void Record(uint64_t name_idx, uint64_t type_idx, ExecutorEvent event,
                  std::chrono::time_point<std::chrono::system_clock> timestamp) {
    auto index = count_.load();
    ++count_;
    if (index >= kProfilingDataCap) {
      return;
    }
    thread_local static auto tid = static_cast<int64_t>(mmGetTid());
    records_[index] = {name_idx, type_idx, event, timestamp, tid};
  }
  void Dump(std::ostream &out_stream, std::vector<std::string> &idx_to_str) const;
  size_t GetCount() const {
    return count_.load();
  }

 private:
  std::atomic<size_t> count_{0UL};
  ProfilingData records_[kProfilingDataCap];
};

class VISIBILITY_EXPORT GlobalProfilingWrapper {
 public:
  GlobalProfilingWrapper(const GlobalProfilingWrapper &) = delete;
  GlobalProfilingWrapper(GlobalProfilingWrapper &&) = delete;
  GlobalProfilingWrapper &operator=(const GlobalProfilingWrapper &) = delete;
  GlobalProfilingWrapper &operator=(GlobalProfilingWrapper &&) = delete;

  static GlobalProfilingWrapper *GetInstance() {
    static GlobalProfilingWrapper global_prof_wrapper;
    return &global_prof_wrapper;
  }

  static void OnGlobalProfilingSwitch(void *ins, uint64_t enable_flags);

  void Init(uint64_t enable_flags);

  void Free() {
    global_profiler_.reset(nullptr);
    SetEnableFlags(0UL);
  }

  GlobalProfiler *GetGlobalProfiler() const {
    return global_profiler_.get();
  }

  void SetEnableFlags(const uint64_t enable_flags) {
    enable_flags_ = enable_flags;
  }

  uint64_t GetRecordCount() {
    if (global_profiler_ == nullptr) {
      return 0UL;
    }
    return global_profiler_->GetCount();
  }

  uint64_t GetEnableFlags() const {
    return enable_flags_;
  }

  bool IsEnabled(ProfilingType profiling_type) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return enable_flags_ & BuiltInSubscriberUtil::EnableBit<ProfilingType>(profiling_type);
  }

  void DumpAndFree(std::ostream &out_stream) {
    Dump(out_stream);
    Free();
  }
  void Dump(std::ostream &out_stream) {
    if (global_profiler_ != nullptr) {
      global_profiler_->Dump(out_stream, idx_to_str_);
    }
  }
  void Record(uint64_t name_idx, uint64_t type_idx, ExecutorEvent event,
              std::chrono::time_point<std::chrono::system_clock> timestamp) {
    if (global_profiler_ != nullptr) {
      global_profiler_->Record(name_idx, type_idx, event, timestamp);
    }
  }

  uint64_t RegisterString(const std::string &name) {
    const std::lock_guard<std::mutex> lk(register_mutex_);
    RegisterBuiltInString();
    const auto iter = std::find(idx_to_str_.begin(), idx_to_str_.end(), name);
    if (iter == idx_to_str_.end()) {
      idx_to_str_[str_idx_] = name;
      ++str_idx_;
      if (str_idx_ >= idx_to_str_.size()) {
        idx_to_str_.resize(idx_to_str_.size() * kDouble);
      }
      return str_idx_ - 1UL;
    } else {
      return iter - idx_to_str_.begin();
    }
  }

  static uint64_t RegisterStringHash(const std::string &str);

  const std::vector<std::string> &GetIdxToStr() const {
    return idx_to_str_;
  }
  void RegisterBuiltInString();
 private:
  GlobalProfilingWrapper();

 private:
  std::unique_ptr<GlobalProfiler> global_profiler_{nullptr};
  uint64_t enable_flags_{0UL};
  uint64_t str_idx_{0UL};
  bool is_builtin_string_registered_{false};
  std::vector<std::string> idx_to_str_;
  std::mutex register_mutex_;
  std::mutex mutex_;
};

class ScopeProfiler {
 public:
  ScopeProfiler(const size_t element, const size_t event) : element_(element), event_(event) {
    if (GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kGeHost)) {
      start_trace_ = std::chrono::system_clock::now();
    }
  }

  void SetElement(const size_t element) {
    element_ = element;
  }

  ~ScopeProfiler() {
    if (GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kGeHost)) {
      GlobalProfilingWrapper::GetInstance()->Record(element_, event_, kExecuteStart, start_trace_);
      GlobalProfilingWrapper::GetInstance()->Record(element_, event_, kExecuteEnd, std::chrono::system_clock::now());
    }
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_trace_;
  size_t element_;
  size_t event_;
};
}  // namespace gert

#define GE_PROFILING_START(event)                                                             \
  std::chrono::time_point<std::chrono::system_clock> event##start_time;                       \
  if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kGeHost)) { \
    event##start_time = std::chrono::system_clock::now();                                     \
  }

#define GE_PROFILING_END(name_idx, type_idx, event)                                                         \
  do {                                                                                                      \
    if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kGeHost)) {             \
      gert::GlobalProfilingWrapper::GetInstance()->Record(name_idx, type_idx, ExecutorEvent::kExecuteStart, \
                                                          event##start_time);                               \
      gert::GlobalProfilingWrapper::GetInstance()->Record(name_idx, type_idx, ExecutorEvent::kExecuteEnd,   \
                                                          std::chrono::system_clock::now());                \
    }                                                                                                       \
  } while (false)

#define RT2_PROFILING_SCOPE(element, event) gert::ScopeProfiler profiler((element), event)
#define RT2_PROFILING_SCOPE_CONST(element, event) const gert::ScopeProfiler profiler((element), (event))
#define RT2_PROFILING_SCOPE_ELEMENT(element) profiler.SetElement(element)
#endif
