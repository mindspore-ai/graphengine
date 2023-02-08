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

#ifndef GE_COMMON_PROFILING_PROPERTIES_H_
#define GE_COMMON_PROFILING_PROPERTIES_H_

#include <mutex>
#include <string>
#include <vector>
#include <atomic>

#include "framework/common/ge_types.h"
#include "toolchain/prof_callback.h"
#include "common/profiling_definitions.h"

namespace ge {
class ProfilingProperties {
 public:
  static ProfilingProperties &Instance();
  void SetLoadProfiling(const bool is_load_profiling);
  bool IsLoadProfiling();
  void SetExecuteProfiling(const bool is_exec_profiling);
  void SetExecuteProfiling(const std::map<std::string, std::string> &options);
  bool IsExecuteProfiling();
  void SetTrainingTrace(const bool is_train_trace);
  bool ProfilingTrainingTraceOn() const {
    return is_training_trace_;
  }
  void SetIfInited(const bool is_inited) { is_inited_.store(is_inited); }
  bool ProfilingInited() const { return is_inited_.load(); }
  void SetFpBpPoint(const std::string &fp_point, const std::string &bp_point);
  bool ProfilingOn() const {
    return is_load_profiling_ && is_execute_profiling_;
  }
  void GetFpBpPoint(std::string &fp_point, std::string &bp_point);
  void ClearProperties();
  void SetOpDetailProfiling(const bool is_op_detail_profiling);
  bool IsOpDetailProfiling();
  bool IsDynamicShapeProfiling() const;
  void UpdateDeviceIdCommandParams(const std::string &config_data);
  const std::string &GetDeviceConfigData() const;
  bool IsTrainingModeProfiling() const;
  void SetProfilingLoadOfflineFlag(const bool is_load_offline);
  bool IsTaskEventProfiling() const {
    return is_task_event_profiling_.load();
  }
  void SetTaskEventProfiling(const bool is_task_event_profiling) {
    is_task_event_profiling_.store(is_task_event_profiling);
    // task event profiling need reinit profiling context
    profiling::ProfilingContext::GetInstance().Init();
  }

 private:
  ProfilingProperties() noexcept : is_op_detail_profiling_(false) {}
  ~ProfilingProperties() = default;
  std::mutex mutex_;
  std::mutex point_mutex_;
  bool is_load_profiling_ = false;
  bool is_execute_profiling_ = false;
  bool is_training_trace_ = false;
  bool is_load_offline_flag_ = false;
  std::atomic<bool> is_inited_{false};
  std::string device_command_params_;  // key: device_id, value: profiling config data
  std::atomic<bool> is_op_detail_profiling_{false};
  std::atomic<bool> is_task_event_profiling_{false};
  std::string fp_point_;
  std::string bp_point_;
};
}  // namespace ge

#endif  // GE_COMMON_PROFILING_PROPERTIES_H_
