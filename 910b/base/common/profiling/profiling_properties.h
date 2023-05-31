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
struct DeviceSubsInfo {
  uint64_t module;
  uint32_t subscribe_count;
};

struct ProfSubscribeInfo {
  bool is_subscribe;
  uint64_t prof_switch;
  uint32_t graph_id;
};
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
  void UpdateSubscribeDeviceModuleMap(const std::string &prof_type, const uint32_t device_id, const uint64_t module);
  void UpdateDeviceIdModuleMap(const std::string &prof_type, const uint64_t module,
                               const std::vector<int32_t> &device_list);
  bool ProfilingSubscribeOn() const;
  void SetSubscribeInfo(const uint64_t prof_switch, const uint32_t model_id, const bool is_subscribe);
  void CleanSubscribeInfo();
  void ClearDeviceIdMap();
  void InsertSubscribeGraphId(const uint32_t graph_id);
  void RemoveSubscribeGraphId(const uint32_t graph_id);
  const std::map<uint32_t, DeviceSubsInfo> &GetDeviceSubInfo() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return subs_dev_module_;
  }

  const ProfSubscribeInfo &GetSubscribeInfo() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return subscribe_info_;
  }

  std::atomic<uint32_t> &MutableSubscribeCount() {
    return subscribe_count_;
  }

  const std::set<uint32_t> &GetSubscribeGraphId() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return subscribe_graph_id_;
  }
 private:
  ProfilingProperties() noexcept {}
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
  std::atomic<uint32_t> subscribe_count_{0U};
  std::map<int32_t, uint64_t> device_id_module_map_; // key: device_id, value: profiling on module
  std::map<uint32_t, DeviceSubsInfo> subs_dev_module_; // key: device_id, value: profiling on module
  ProfSubscribeInfo subscribe_info_{false, 0U, 0U};
  std::set<uint32_t> subscribe_graph_id_;
};
}  // namespace ge

#endif  // GE_COMMON_PROFILING_PROPERTIES_H_
