/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef GE_COMMON_PROFILING_PROFILING_MANAGER_H_
#define GE_COMMON_PROFILING_PROFILING_MANAGER_H_

#include <nlohmann/json.hpp>
#include <mutex>
#include <map>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "external/register/register_types.h"
#include "toolchain/prof_callback.h"

using std::map;
using std::string;
using std::vector;
using Json = nlohmann::json;

namespace {
  const std::string GE_PROFILING_MODULE = "Framework";
}  // namespace
namespace ge {
struct DeviceSubsInfo {
  uint64_t module;
  uint32_t subscribe_count;
};

struct MsprofCallback {
  MsprofCtrlCallback msprofCtrlCallback;
  MsprofReporterCallback msprofReporterCallback;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ProfilingManager {
 public:
  ProfilingManager();
  virtual ~ProfilingManager();
  static ProfilingManager &Instance();
  Status Init(const Options &options);
  Status ProfInit(uint64_t module);
  Status ProfFinalize();
  Status ProfStartProfiling(uint64_t module, const std::map<std::string, std::string> &config_para);
  Status ProfStopProfiling(uint64_t module, const std::map<std::string, std::string> &config_para);
  Status ProfModelSubscribe(uint64_t module, void *model);
  Status ProfModelUnsubscribe(void *model);
  void StopProfiling();
  bool ProfilingTrainingTraceOn() const { return is_training_trace_; }
  bool ProfilingModelLoadOn() const { return is_load_profiling_; }
  bool ProfilingModelExecuteOn() const;
  bool ProfilingOn() const { return is_load_profiling_ && is_execute_profiling_; } // is_execute_profiling_ only used by ge option and env
  void ReportProfilingData(uint32_t model_id, const std::vector<TaskDescInfo> &task_desc_info,
                           const std::vector<ComputeGraphDescInfo> &compute_graph_desc_info);
  void ProfilingTaskDescInfo(uint32_t model_id, const std::vector<TaskDescInfo> &task_desc_info,
                             const int32_t &device_id);
  void ProfilingGraphDescInfo(uint32_t model_id, const std::vector<ComputeGraphDescInfo> &compute_graph_desc_info,
                              const int32_t &device_id);
  Status PluginInit() const;
  void PluginUnInit() const;
  Status CallMsprofReport(ReporterData &reporter_data) const;
  struct MsprofCallback &GetMsprofCallback() { return prof_cb_; }
  void SetMsprofCtrlCallback(MsprofCtrlCallback func) { prof_cb_.msprofCtrlCallback = func; }
  void SetMsprofReporterCallback(MsprofReporterCallback func) { prof_cb_.msprofReporterCallback = func; }
  void GetFpBpPoint(std::string &fp_point, std::string &bp_point);
 private:
  Status InitFromOptions(const Options &options, MsprofGeOptions &prof_conf);
  Status ParseOptions(const std::string &options);
  Status ProfParseParam(const std::map<std::string, std::string> &config_para, int32_t &device_num,
                        vector<int32_t> &device_list);
  Status ProfParseDeviceId(const std::map<std::string, std::string> &config_para,
                               vector<int32_t> &device_list);
  uint64_t GetProfilingModule();
  void GraphDescReport(const int32_t &device_id, const string &data);
  void UpdateDeviceIdModuleMap(string prof_type, uint64_t module, const vector<int32_t> &device_list);
  void UpdateSubscribeDeviceModuleMap(std::string prof_type, uint32_t device_id, uint64_t module);

  bool is_load_profiling_;
  bool is_execute_profiling_;
  bool is_training_trace_;
  vector<int32_t> device_id_;
  map<int32_t, uint64_t> device_id_module_map_; // key: device_id, value: profiling on module
  map<uint32_t, DeviceSubsInfo> subs_dev_module_; // key: device_id, value: profiling on module
  uint32_t subscribe_count_;
  std::mutex mutex_;
  MsprofCallback prof_cb_;
  std::string fp_point_;
  std::string bp_point_;
};
}  // namespace ge
#endif  // GE_COMMON_PROFILING_PROFILING_MANAGER_H_
