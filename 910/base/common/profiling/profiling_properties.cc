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

#include "profiling_properties.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/ge_context.h"
#include "mmpa/mmpa_api.h"
#include "common/ge_inner_attrs.h"
#include "nlohmann/json.hpp"
#include "common/global_variables/diagnose_switch.h"

namespace {
const uint64_t kMsProfOptionsMaxlen = 2048U;
const std::string kFpPoint = "fp_point";
const std::string kBpPoint = "bp_point";
const std::string kProfilingExecuteOn = "1";
const std::string kProfilingExecuteOff = "0";
}  // namespace ge

namespace ge {

ProfilingProperties& ProfilingProperties::Instance() {
  static ProfilingProperties profiling_properties;
  return profiling_properties;
}

void ProfilingProperties::SetLoadProfiling(const bool is_load_profiling) {
  const std::lock_guard<std::mutex> lock(mutex_);
  is_load_profiling_ = is_load_profiling;
}
bool ProfilingProperties::IsLoadProfiling() {
  const std::lock_guard<std::mutex> lock(mutex_);
  return is_load_profiling_;
}

void ProfilingProperties::SetExecuteProfiling(const bool is_exec_profiling) {
  const std::lock_guard<std::mutex> lock(mutex_);
  is_execute_profiling_ = is_exec_profiling;
}

void ProfilingProperties::SetExecuteProfiling(const std::map<std::string, std::string> &options) {
  const std::map<std::string, std::string>::const_iterator &iter = options.find(kProfilingIsExecuteOn);
  if (iter != options.end()) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (iter->second == kProfilingExecuteOn) {
      is_execute_profiling_ = true;
    } else if (iter->second == kProfilingExecuteOff) {
      is_execute_profiling_ = false;
    } else {
      GELOGW("Set execute profiling failed, set value[%s]", iter->second.c_str());
    }
  }
}

bool ProfilingProperties::IsExecuteProfiling() {
  const std::lock_guard<std::mutex> lock(mutex_);
  return is_execute_profiling_;
}

void ProfilingProperties::SetTrainingTrace(const bool is_train_trace) {
  const std::lock_guard<std::mutex> lock(mutex_);
  is_training_trace_ = is_train_trace;
}

void ProfilingProperties::SetOpDetailProfiling(const bool is_op_detail_profiling) {
  is_op_detail_profiling_.store(is_op_detail_profiling);
}

bool ProfilingProperties::IsOpDetailProfiling() {
  return is_op_detail_profiling_.load();
}
bool ProfilingProperties::IsDynamicShapeProfiling() const {
  // current use the same switch with op detail.
  return is_op_detail_profiling_.load();
}
void ProfilingProperties::GetFpBpPoint(std::string &fp_point, std::string &bp_point) {
  // Env or options mode, fp_point_/bp_point_ have initiliazed on profiling init
  const std::lock_guard<std::mutex> lock(mutex_);
  if ((!fp_point_.empty()) && (!bp_point_.empty())) {
    fp_point = fp_point_;
    bp_point = bp_point_;
    GELOGI("Bp Fp have been initialized in env or options. bp_point: %s, fp_point: %s", bp_point.c_str(),
           fp_point.c_str());
    return;
  }
  // ProfApi mode and training trace is set
  // Parse options first
  bool is_profiling_valid = false;
  std::string profiling_options;
  if ((ge::GetContext().GetOption(OPTION_EXEC_PROFILING_OPTIONS, profiling_options) == SUCCESS) &&
      (!profiling_options.empty())) {
    is_profiling_valid = true;
  } else {
    char_t env_profiling_options[kMsProfOptionsMaxlen] = {};
    const INT32 ret = mmGetEnv("PROFILING_OPTIONS", &env_profiling_options[0], kMsProfOptionsMaxlen);
    if (ret != EN_OK) {
      GELOGI("PROFILING_OPTIONS env is not exist.");
      return;
    }
    GELOGI("Parse env PROFILING_OPTIONS:%s.", &env_profiling_options[0]);
    profiling_options = &env_profiling_options[0];
    is_profiling_valid = true;
  }
  if (is_profiling_valid) {
    try {
      const nlohmann::json prof_options = nlohmann::json::parse(profiling_options);
      if (prof_options.contains(kFpPoint)) {
        fp_point_ = prof_options[kFpPoint];
      }
      if (prof_options.contains(kBpPoint)) {
        bp_point_ = prof_options[kBpPoint];
      }
      fp_point = fp_point_;
      bp_point = bp_point_;
      if ((!fp_point_.empty()) && (!bp_point_.empty())) {
        GELOGI("Training trace bp fp is set, bp_point:%s, fp_point:%s.", bp_point_.c_str(), fp_point_.c_str());
      }
    } catch (nlohmann::json::exception &e) {
      GELOGE(ge::FAILED, "Nlohmann json prof options is invalid, catch exception:%s", e.what());
      return;
    }
  }

  return;
}

void ProfilingProperties::SetFpBpPoint(const std::string &fp_point, const std::string &bp_point) {
  const std::lock_guard<std::mutex> lock(mutex_);
  fp_point_ = fp_point;
  bp_point_ = bp_point;
}

void ProfilingProperties::UpdateDeviceIdCommandParams(const std::string &config_data) {
  device_command_params_ = config_data;
}

const std::string &ProfilingProperties::GetDeviceConfigData() const {
  return device_command_params_;
}

void ProfilingProperties::ClearProperties() {
  const std::lock_guard<std::mutex> lock(mutex_);
  diagnoseSwitch::DisableProfiling();
  is_load_profiling_ = false;
  is_op_detail_profiling_.store(false);
  is_execute_profiling_ = false;
  is_training_trace_ = false;
  fp_point_.clear();
  bp_point_.clear();
}

bool ProfilingProperties::IsTrainingModeProfiling() const {
  if (is_load_offline_flag_) {
    return false;
  }
  return domi::GetContext().train_flag;
}

void ProfilingProperties::SetProfilingLoadOfflineFlag(const bool is_load_offline) {
  is_load_offline_flag_ = is_load_offline;
}

}  // namespace ge
