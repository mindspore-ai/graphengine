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
#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_GLOBAL_PROFILING_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_GLOBAL_PROFILING_H_

#include <algorithm>
#include <memory>
#include <unordered_map>
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
};
class GlobalProfiler {
 public:
  GlobalProfiler() = default;
  void Record(uint64_t name_idx, uint64_t type_idx, ExecutorEvent event,
              std::chrono::time_point<std::chrono::system_clock> timestamp) {
    auto index = count_++;
    if (index >= kProfilingDataCap) {
      return;
    }
    records_[index] = {name_idx, type_idx, event, timestamp};
  }
  void Dump(std::ostream &out_stream, std::vector<std::string> &idx_to_str) const;
  size_t GetCount() const {
    return count_;
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

  void SetEnableFlags(uint64_t enable_flags) {
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

  bool IsEnable(ProfilingType profiling_type) const {
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

  uint64_t RegisterString(const char *name) {
    const std::lock_guard<std::mutex> lk(register_mutex_);
    std::string str_name = name;
    const auto iter = std::find(idx_to_str_.begin(), idx_to_str_.end(), str_name);
    if (iter == idx_to_str_.end()) {
      idx_to_str_[str_idx_] = str_name;
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

 private:
  GlobalProfilingWrapper();

 private:
  std::unique_ptr<GlobalProfiler> global_profiler_{nullptr};
  uint64_t enable_flags_{0UL};
  uint64_t str_idx_{0UL};
  std::vector<std::string> idx_to_str_;
  std::mutex register_mutex_;
};
}  // namespace gert

#endif
