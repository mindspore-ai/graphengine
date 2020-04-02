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

#include "common/profiling/profiling_manager.h"

#include <nlohmann/json.hpp>
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/string_util.h"
#include "graph/ge_context.h"
#include "runtime/base.h"

using Json = nlohmann::json;

namespace {
const char *const kJobID = "jobID";
const char *const kDeviceID = "deviceID";
const char *const kStartCfg = "startCfg";
const char *const kFeatures = "features";
const char *const kConf = "conf";
const char *const kEvents = "events";
const char *const kAiCoreEvents = "ai_core_events";
const char *const kName = "name";
const char *const kTraceID = "traceId";
}  // namespace

namespace ge {
ProfilingManager::ProfilingManager() {}

ProfilingManager::~ProfilingManager() {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ProfilingManager &ProfilingManager::Instance() {
  static ProfilingManager profiling_manager;
  return profiling_manager;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ge::Status ProfilingManager::Init(const Options &options) {
#ifdef DAVINCI_SUPPORT_PROFILING
  device_id_ = options.device_id;
  job_id_ = options.job_id;

  Status ret;
  if (!recv_profiling_config_.empty()) {
    GELOGI("Profiling json config from acl:%s", recv_profiling_config_.c_str());
    ret = InitFromAclCfg(recv_profiling_config_);
  } else {
    ret = InitFromEnv(options);
  }
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to init profiling.");
    return ret;
  }

  if (is_profiling_) {
    // register Framework to profiling
    const ProfilingEngineImpl engine_0;
    int result = Msprof::Engine::RegisterEngine("Framework", &engine_0);
    if (result != 0) {
      GELOGE(FAILED, "Register profiling engine failed.");
      return FAILED;
    }
    // profiling startup first time
    ret = StartProfiling(0);
    if (ret != SUCCESS) {
      GELOGE(ret, "Profiling start failed.");
      return FAILED;
    }
    GELOGI("Profiling init succ.");
  }
#endif
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ge::Status ProfilingManager::InitFromAclCfg(
  const std::string &config) {
#ifdef DAVINCI_SUPPORT_PROFILING
  try {
    is_profiling_ = false;
    profiling_opts_.clear();
    op_trace_conf_.clear();
    Json start_prof_conf = Json::parse(config);
    Json &prof_conf = start_prof_conf[kStartCfg][0];
    job_id_ = prof_conf[kJobID];

    GELOGI("Profiling json config from acl:%s", config.c_str());
    Json &features = prof_conf[kFeatures];
    for (size_t i = 0; i < features.size(); ++i) {
      Json &feature = features[i];
      if ((feature.find(kName) == feature.end()) || feature[kName].is_null()) {
        continue;
      }

      const std::string &name = feature[kName];
      if (name == "op_trace") {
        GELOGI("Op trace config from acl");
        Json &conf = feature[kConf];
        Json &events = conf[kEvents];
        const std::string &ai_core_events = events[kAiCoreEvents];
        GELOGI("Op trace config from acl ai_core_events:%s", ai_core_events.c_str());
        is_op_trace_ = true;
        // op trace get conf
        ProfMgrConf prof_mgr_conf;
        int result = ProfMgrGetConf(ai_core_events, &prof_mgr_conf);
        if (result != 0) {
          GELOGE(FAILED, "ProfMgrGetConf failed.");
          return FAILED;
        }
        op_trace_conf_ = prof_mgr_conf.conf;
        op_trace_iter_num_ = static_cast<int32_t>(op_trace_conf_.size());
        GELOGI("Op trace profiling iter num %d,", op_trace_iter_num_);
      } else if (name == "task_trace") {
        is_op_trace_ = false;
        GELOGI("Task trace config from acl");
      }
      profiling_opts_.push_back(name);
    }

    is_profiling_ = true;
  } catch (Json::parse_error &e) {
    GELOGE(FAILED, "Json conf is not invalid !");
    return ge::PARAM_INVALID;
  }
#endif
  return ge::SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ge::Status ProfilingManager::InitFromEnv(const Options &options) {
#ifdef DAVINCI_SUPPORT_PROFILING
  const char *is_profiling = std::getenv("PROFILING_MODE");
  const char *prof_options = std::getenv("PROFILING_OPTIONS");
  GELOGI("The profiling in options is %s, %s", is_profiling, prof_options);
  if ((is_profiling == nullptr) || (strcmp("true", is_profiling) != 0) || (prof_options == nullptr)) {
    // default training trace on
    is_profiling_ = false;
    return SUCCESS;
  } else {
    std::string prof_options_str = std::string(prof_options);
    profiling_opts_ = StringUtils::Split(prof_options_str, ':');
    is_profiling_ = true;
  }

  // features:'training_trace', 'task_trace' or 'op_trace'  etc
  if (!profiling_opts_.empty()) {
    if (profiling_opts_[0] == "op_trace") {
      is_op_trace_ = true;
      // op trace get conf
      ProfMgrConf prof_mgr_conf;
      int result = ProfMgrGetConf("", &prof_mgr_conf);
      if (result != 0) {
        GELOGE(FAILED, "ProfMgrGetConf failed.");
        return FAILED;
      }
      op_trace_conf_ = prof_mgr_conf.conf;
      op_trace_iter_num_ = static_cast<int32_t>(op_trace_conf_.size());
      GELOGI("op trace profiling iter num %d,", op_trace_iter_num_);
    } else {
      is_op_trace_ = false;
      op_trace_iter_num_ = 1;
    }
  }
#endif
  return ge::SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ge::Status ProfilingManager::StartProfiling(int32_t iter_num) {
#ifdef DAVINCI_SUPPORT_PROFILING
  if (!profiling_opts_.empty()) {
    GELOGI("Start profiling index is %d", iter_num);
    // current one docker only use one device
    Json p_device;

    try {
      // profiling need physical_device_id
      p_device[kDeviceID] = std::to_string(device_id_);
      p_device[kJobID] = job_id_;
      p_device[kTraceID] = std::to_string(GetContext().TraceId());

      Json features;
      if (is_op_trace_) {
        Json f;
        f[kName] = "op_trace";
        Json conf;
        if (op_trace_conf_.size() <= static_cast<size_t>(iter_num)) {
          GELOGE(FAILED, "Op trace iter num is invalid!");
          return FAILED;
        }
        conf = nlohmann::json::parse(op_trace_conf_[iter_num]);
        f[kConf] = conf;
        features[0] = f;
        if (iter_num == 0) {
          is_load_ = true;
        }
      } else {
        for (std::vector<std::string>::size_type i = 0; i < profiling_opts_.size(); i++) {
          Json f;
          f[kName] = profiling_opts_[i];
          features[i] = f;
        }
        is_load_ = true;
      }
      p_device[kFeatures] = features;
      // only one device, but sProfMgrStartUp API require for device list
      Json devices;
      devices[0] = p_device;

      Json start_cfg;
      start_cfg[kStartCfg] = devices;

      // convert json to string
      std::stringstream ss;
      ss << start_cfg;
      send_profiling_config_ = ss.str();
      GELOGI("Profiling config %s\n", send_profiling_config_.c_str());
    } catch (Json::parse_error &e) {
      GELOGE(FAILED, "Op trace json conf is not invalid !");
      return FAILED;
    }

    // runtime startup for profiling
    GE_CHK_RT_RET(rtProfilerStart());

    // call profiling startup API
    ProfMgrCfg prof_cfg = {send_profiling_config_};
    prof_handle = ProfMgrStartUp(&prof_cfg);
    if (prof_handle == nullptr) {
      GELOGW("ProfMgrStartUp failed.");
      return FAILED;
    }
  }
#endif
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void ProfilingManager::StopProfiling() {
#ifdef DAVINCI_SUPPORT_PROFILING
  Msprof::Engine::Reporter *reporter = PluginImpl::GetPluginReporter();
  if (reporter != nullptr) {
    int ret = reporter->Flush();
    GELOGI("Report data end, ret is %d", ret);
  }

  rtError_t rt_ret = rtProfilerStop();
  if (rt_ret != RT_ERROR_NONE) {
    GELOGI("Call rtProfilerStop ret:%d", rt_ret);
  }

  if (prof_handle != nullptr) {
    int result = ProfMgrStop(prof_handle);
    if (result != 0) {
      GELOGW("ProfMgr stop return fail:%d.", result);
      return;
    }
  }
  is_load_ = false;
  recv_profiling_config_ = "";
  GELOGI("Stop Profiling success.");
#endif
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void ProfilingManager::ReportProfilingData(
  const std::map<uint32_t, std::string> &op_task_id_map) {
#ifdef DAVINCI_SUPPORT_PROFILING
  Msprof::Engine::Reporter *reporter = PluginImpl::GetPluginReporter();
  if (reporter == nullptr) {
    GELOGI("Profiling report is nullptr!");
    return;
  }
  std::string data;
  for (const auto &iter : op_task_id_map) {
    data = iter.second + ' ' + std::to_string(iter.first) + ';';
    Msprof::Engine::ReporterData reporter_data{};
    reporter_data.deviceId = device_id_;
    reporter_data.data = (unsigned char *)data.c_str();
    reporter_data.dataLen = data.size();
    int ret = memcpy_s(reporter_data.tag, MSPROF_ENGINE_MAX_TAG_LEN + 1, "framework", sizeof("framework"));
    if (ret != EOK) {
      GELOGE(ret, "Report data tag memcpy error!");
      return;
    }
    ret = reporter->Report(&reporter_data);
    if (ret != SUCCESS) {
      GELOGE(ret, "Reporter data fail!");
      return;
    }
  }
  GELOGI("Report profiling data for GE end.");
#endif
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void ProfilingManager::SetProfilingConfig(
  const std::string &profiling_cfg) {
  recv_profiling_config_ = profiling_cfg;
}

/**
 * @brief Profiling PluginImpl
 */
// PluginImpl static variable init
Msprof::Engine::Reporter *PluginImpl::reporter_ = nullptr;

PluginImpl::PluginImpl(const std::string &module) : module_(module) { GELOGI("Create PluginImpl\n"); }

int PluginImpl::Init(const Msprof::Engine::Reporter *reporter) {
  GELOGI("PluginImpl init");
  reporter_ = const_cast<Msprof::Engine::Reporter *>(reporter);
  return 0;
}

int PluginImpl::UnInit() {
  GELOGI("PluginImpl Uninit");
  reporter_ = nullptr;
  return 0;
}

Msprof::Engine::PluginIntf *ProfilingEngineImpl::CreatePlugin() {
  GELOGI(" Create Plugin");
  return new (std::nothrow) PluginImpl("Framework");
}

int ProfilingEngineImpl::ReleasePlugin(Msprof::Engine::PluginIntf *plugin) {
  if (plugin != nullptr) {
    delete plugin;
    plugin = nullptr;
  }
  return 0;
}
}  // namespace ge
