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

#include <map>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "external/register/register_types.h"
#include "toolchain/prof_engine.h"
#include "toolchain/prof_mgr_core.h"

using std::map;
using std::string;
using std::vector;

namespace ge {
class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ProfilingManager {
 public:
  ProfilingManager();
  virtual ~ProfilingManager();
  static ProfilingManager &Instance();
  ge::Status Init(const Options &options);
  ge::Status InitFromEnv(const Options &options);
  ge::Status InitFromAclCfg(const std::string &config);
  ge::Status StartProfiling(int32_t iter);
  void StopProfiling();
  bool ProfilingOpTraceOn() const { return is_op_trace_; }
  bool ProfilingLoadFlag() const { return is_load_; }
  bool ProfilingOn() const { return is_profiling_; }
  int32_t GetOpTraceIterNum() const { return op_trace_iter_num_; }
  void ReportProfilingData(const std::map<uint32_t, std::string> &op_task_id_map);
  void SetProfilingConfig(const string &profiling_cfg);

 private:
  bool is_profiling_ = false;
  bool is_op_trace_ = false;
  bool is_load_ = false;
  int32_t op_trace_iter_num_ = 0;
  string job_id_;
  int32_t device_id_ = 0;
  vector<string> op_trace_conf_;
  vector<string> profiling_opts_;
  void *prof_handle = nullptr;
  string recv_profiling_config_;
  string send_profiling_config_;
};

///
/// @brief register Plugin
///
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

///
/// @brief register Engine
///
class ProfilingEngineImpl : public Msprof::Engine::EngineIntf {
 public:
  ProfilingEngineImpl() {}
  ~ProfilingEngineImpl() {}

  Msprof::Engine::PluginIntf *CreatePlugin();
  int ReleasePlugin(Msprof::Engine::PluginIntf *plugin);
};
}  // namespace ge
#endif  // GE_COMMON_PROFILING_PROFILING_MANAGER_H_
