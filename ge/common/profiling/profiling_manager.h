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
#include "toolchain/prof_engine.h"
#include "toolchain/prof_mgr_core.h"
#include "toolchain/prof_acl_api.h"

using std::map;
using std::string;
using std::vector;
using Json = nlohmann::json;

namespace {
const std::string GE_PROFILING_MODULE = "Framework";
}  // namespace
namespace ge {
// register Plugin
class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY PluginImpl : public Msprof::Engine::PluginIntf {
 public:
  explicit PluginImpl(const std::string &module);
  ~PluginImpl() {}

  int Init(const Msprof::Engine::Reporter *reporter);
  int UnInit();
  static Msprof::Engine::Reporter *GetPluginReporter() { return reporter_; }

 private:
  static Msprof::Engine::Reporter *reporter_;
  std::string module_;
};

// register Engine
class ProfilingEngineImpl : public Msprof::Engine::EngineIntf {
 public:
  ProfilingEngineImpl() {}
  ~ProfilingEngineImpl() {}

  Msprof::Engine::PluginIntf *CreatePlugin();
  int ReleasePlugin(Msprof::Engine::PluginIntf *plugin);
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ProfilingManager {
 public:
  ProfilingManager();
  virtual ~ProfilingManager();
  static ProfilingManager &Instance();
  ge::Status Init(const Options &options, bool convert_2_phy_device_id = false);
  ge::Status InitFromOptions(const Options &options);
  ge::Status InitFromAclCfg(const std::string &config);
  ge::Status StartProfiling(int32_t iter, int32_t device_id);
  ge::Status ProfInit(uint64_t module);
  ge::Status ProfFinalize();
  ge::Status ProfStartProfiling(uint64_t module, const std::map<std::string, std::string> &config_para);
  ge::Status ProfStopProfiling(uint64_t module, const std::map<std::string, std::string> &config_para);
  void StopProfiling();
  bool ProfilingOpTraceOn() const { return is_op_trace_; }
  bool ProfilingLoadFlag() const { return is_load_; }
  bool ProfilingTrainingTraceOn() const { return is_training_trace_; }
  bool ProfilingModelLoadOn() const { return is_load_profiling_; }
  bool ProfilingModelExecuteOn() const;
  bool ProfilingOn() const { return is_load_profiling_ && is_execute_profiling_; }  // only used  by command pattern
  int32_t GetOpTraceIterNum() const { return op_trace_iter_num_; }
  void ReportProfilingData(const std::vector<TaskDescInfo> &task_desc_info,
                           const std::vector<ComputeGraphDescInfo> &compute_graph_desc_info);
  void Report(const int32_t &device_id, const string &data, Msprof::Engine::Reporter &reporter,
              Msprof::Engine::ReporterData &reporter_data);
  void ProfilingTaskDescInfo(const std::vector<TaskDescInfo> &task_desc_info, const int32_t &device_id);
  void ProfilingGraphDescInfo(const std::vector<ComputeGraphDescInfo> &compute_graph_desc_info,
                              const int32_t &device_id);
  void SetProfilingConfig(const string &profiling_cfg);
  vector<int32_t> GetProfilingDeviceId() const { return device_id_; }
  void PluginUnInit(const std::string &module) const;

 private:
  ge::Status ParseFeaturesFromAclCfg(const Json &feature);
  ge::Status ProfParseParam(const std::map<std::string, std::string> &config_para, int32_t &device_num,
                            vector<int32_t> &device_list);
  ge::Status ProfParseDeviceId(const std::map<std::string, std::string> &config_para, vector<int32_t> &device_list);
  uint64_t GetProfilingModule();
  void UpdateDeviceIdModuleMap(string prof_type, uint64_t module, const vector<int32_t> &device_list);
  bool is_load_profiling_ = false;
  bool is_execute_profiling_ = false;
  bool is_op_trace_ = false;
  bool is_load_ = false;
  bool is_training_trace_ = false;
  bool is_acl_api_mode_ = false;
  int32_t op_trace_iter_num_ = 0;
  string job_id_;
  string prof_dir_;
  vector<int32_t> device_id_;
  vector<string> op_trace_conf_;
  vector<string> profiling_opts_;
  vector<void *> prof_handle_vec_;
  string recv_profiling_config_;
  string send_profiling_config_;
  string system_trace_conf_;
  string task_trace_conf_;
  const ProfilingEngineImpl engine_;
  map<int32_t, uint64_t> device_id_module_map_;  // key: device_id, value: profiling on module
  std::mutex mutex_;
};
}  // namespace ge
#endif  // GE_COMMON_PROFILING_PROFILING_MANAGER_H_
