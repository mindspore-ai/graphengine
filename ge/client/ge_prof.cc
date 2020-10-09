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

#include "ge/ge_prof.h"
#include "ge/ge_api.h"
#include "init/gelib.h"
#include "common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/profiling/profiling_manager.h"
#include "graph/load/graph_loader.h"
#include "toolchain/prof_acl_api.h"

using std::map;
using std::string;
using std::vector;

namespace {
const uint32_t kMaxDeviceNum = 64;
const uint32_t kDeviceListIndex = 3;
const std::string kProfilingInit = "prof_init";
const std::string kProfilingFinalize = "prof_finalize";
const std::string kProfilingStart = "prof_start";
const std::string kProfilingStop = "prof_stop";
const std::string kDeviceNums = "devNums";
const std::string kDeviceIdList = "devIdList";
const std::string kAicoreMetrics = "aicoreMetrics";

const std::map<ge::ProfilingAicoreMetrics, std::string> kProfAicoreMetricsToString = {
    {ge::kAicoreArithmaticThroughput, "AICORE_ARITHMATIC_THROUGHPUT"},
    {ge::kAicorePipeline, "AICORE_PIPELINE"},
    {ge::kAicoreSynchronization, "AICORE_SYNCHRONIZATION"},
    {ge::kAicoreMemory, "AICORE_MEMORY"},
    {ge::kAicoreInternalMemory, "AICORE_INTERNAL_MEMORY"},
    {ge::kAicoreStall, "AICORE_STALL"}};
}  // namespace

static bool g_graph_prof_init_ = false;
static std::mutex g_prof_mutex_;

namespace ge {
struct aclgrphProfConfig {
  ProfConfig config;
};

Status aclgrphProfInit(const char *profiler_path, uint32_t length) {
  GELOGT(TRACE_INIT, "Graph prof init start");

  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Ge client is not initialized.");
    return FAILED;
  }

  std::lock_guard<std::mutex> lock(g_prof_mutex_);
  if (g_graph_prof_init_) {
    GELOGW("Multi graph profiling initializations.");
    return GE_PROF_MULTI_INIT;
  }

  Status ret = CheckPath(profiler_path, length);
  if (ret != SUCCESS) {
    GELOGE(ret, "Profiling config path is invalid.");
    return ret;
  }
  // if command mode is set, just return
  if (ProfilingManager::Instance().ProfilingOn()) {
    GELOGW("Graph prof init failed, cause profiling command pattern is running.");
    return GE_PROF_MODE_CONFLICT;
  }

  ret = ProfInit(profiler_path);
  if (ret != SUCCESS) {
    GELOGE(ret, "ProfInit init fail");
    return ret;
  }

  GraphLoader graph_loader;
  Command command;
  command.cmd_params.clear();
  command.cmd_type = kProfilingInit;
  command.module_index = PROF_MODEL_LOAD;
  ret = graph_loader.CommandHandle(command);
  if (ret != SUCCESS) {
    GELOGE(ret, "Handle profiling command %s failed, config = %s", kProfilingInit.c_str(), profiler_path);
    return ret;
  }
  if (!g_graph_prof_init_) {
    g_graph_prof_init_ = true;
    GELOGI("Profiling init successfully.");
  }

  GELOGI("Successfully execute GraphProfInit.");
  return SUCCESS;
}

Status aclgrphProfFinalize() {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Ge client is not initialized.");
    return FAILED;
  }
  std::lock_guard<std::mutex> lock(g_prof_mutex_);
  // if command mode is set, just return
  if (ProfilingManager::Instance().ProfilingOn()) {
    GELOGW("Graph prof finalize failed, cause profiling command pattern is running.");
    return GE_PROF_MODE_CONFLICT;
  }

  if (!g_graph_prof_init_) {
    GELOGE(GE_PROF_NOT_INIT, "Graph not profiling initialize.");
    return GE_PROF_NOT_INIT;
  }
  GraphLoader graph_loader;
  Command command;
  command.cmd_params.clear();
  command.cmd_type = kProfilingFinalize;
  Status ret = graph_loader.CommandHandle(command);
  if (ret != SUCCESS) {
    GELOGE(ret, "Handle profiling command %s failed.", kProfilingFinalize.c_str());
    return ret;
  }

  ret = ProfFinalize();
  if (ret != SUCCESS) {
    GELOGE(ret, "Finalize profiling failed, result = %d", ret);
  }

  if (ret == SUCCESS) {
    g_graph_prof_init_ = false;
    GELOGI("Successfully execute GraphProfFinalize.");
  }
  return ret;
}

bool TransProfConfigToParam(const aclgrphProfConfig *profiler_config, vector<string> &prof_config_params) {
  prof_config_params.clear();
  prof_config_params.emplace_back(kDeviceNums);
  prof_config_params.emplace_back(std::to_string(profiler_config->config.devNums));
  prof_config_params.emplace_back(kDeviceIdList);
  std::string devID = "";
  if (profiler_config->config.devNums == 0) {
    GELOGW("The device num is invalid.");
    return false;
  }
  for (uint32_t i = 0; i < profiler_config->config.devNums; i++) {
    devID.append(std::to_string(profiler_config->config.devIdList[i]));
    if (i != profiler_config->config.devNums - 1) {
      devID.append(",");
    }
  }

  prof_config_params.push_back(devID);
  prof_config_params.push_back(kAicoreMetrics);
  auto iter =
      kProfAicoreMetricsToString.find(static_cast<ProfilingAicoreMetrics>(profiler_config->config.aicoreMetrics));
  if (iter == kProfAicoreMetricsToString.end()) {
    GELOGW("The prof aicore metrics is invalid.");
    return false;
  }
  prof_config_params.push_back(iter->second);
  return true;
}

bool isProfConfigValid(const uint32_t *deviceid_list, uint32_t device_nums) {
  if (deviceid_list == nullptr) {
    GELOGE(PARAM_INVALID, "deviceIdList is nullptr");
    return false;
  }
  if (device_nums == 0 || device_nums > kMaxDeviceNum) {
    GELOGE(PARAM_INVALID, "The device nums is invalid.");
    return false;
  }

  // real device num
  int32_t dev_count = 0;
  rtError_t rt_err = rtGetDeviceCount(&dev_count);
  if (rt_err != RT_ERROR_NONE) {
    GELOGE(INTERNAL_ERROR, "Get the Device count fail.");
    return false;
  }

  if (device_nums > static_cast<uint32_t>(dev_count)) {
    GELOGE(PARAM_INVALID, "Device num(%u) is not in range 1 ~ %d.", device_nums, dev_count);
    return false;
  }

  std::unordered_set<uint32_t> record;
  for (size_t i = 0; i < device_nums; ++i) {
    uint32_t dev_id = deviceid_list[i];
    if (dev_id >= static_cast<uint32_t>(dev_count)) {
      GELOGE(PARAM_INVALID, "Device id %u is not in range 0 ~ %d(exclude %d)", dev_id, dev_count, dev_count);
      return false;
    }
    if (record.count(dev_id) > 0) {
      GELOGE(PARAM_INVALID, "Device id %u is duplicatedly set", dev_id);
      return false;
    }
    record.insert(dev_id);
  }
  return true;
}

aclgrphProfConfig *aclgrphProfCreateConfig(uint32_t *deviceid_list, uint32_t device_nums,
                                           ProfilingAicoreMetrics aicore_metrics, ProfAicoreEvents *aicore_events,
                                           uint64_t data_type_config) {
  if (!isProfConfigValid(deviceid_list, device_nums)) {
    return nullptr;
  }
  aclgrphProfConfig *config = new (std::nothrow) aclgrphProfConfig();
  if (config == nullptr) {
    GELOGE(INTERNAL_ERROR, "new aclgrphProfConfig fail");
    return nullptr;
  }
  config->config.devNums = device_nums;
  if (memcpy_s(config->config.devIdList, sizeof(config->config.devIdList), deviceid_list,
               device_nums * sizeof(uint32_t)) != EOK) {
    GELOGE(INTERNAL_ERROR, "copy devID failed. size = %u", device_nums);
    delete config;
    return nullptr;
  }

  config->config.aicoreMetrics = static_cast<ProfAicoreMetrics>(aicore_metrics);
  config->config.dataTypeConfig = data_type_config;
  GELOGI("Successfully create prof config.");
  return config;
}

Status aclgrphProfDestroyConfig(aclgrphProfConfig *profiler_config) {
  if (profiler_config == nullptr) {
    GELOGE(PARAM_INVALID, "destroy profilerConfig failed, profilerConfig must not be nullptr");
    return PARAM_INVALID;
  }

  delete profiler_config;
  GELOGI("Successfully destroy prof config.");
  return SUCCESS;
}

Status aclgrphProfStart(aclgrphProfConfig *profiler_config) {
  if (profiler_config == nullptr) {
    GELOGE(PARAM_INVALID, "aclgrphProfConfig is invalid.");
    return FAILED;
  }
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Ge client is not initialized.");
    return FAILED;
  }

  std::lock_guard<std::mutex> lock(g_prof_mutex_);
  // if command mode is set, just return
  if (ProfilingManager::Instance().ProfilingOn()) {
    GELOGW("Graph prof finalize failed, cause profiling command pattern is running.");
    return GE_PROF_MODE_CONFLICT;
  }
  if (!g_graph_prof_init_) {
    GELOGE(GE_PROF_NOT_INIT, "Graph not profiling initialize.");
    return GE_PROF_NOT_INIT;
  }

  Status ret = ProfStartProfiling(&profiler_config->config);
  if (ret != SUCCESS) {
    GELOGE(ret, "Start profiling failed, prof result = %d", ret);
    return FAILED;
  }

  std::vector<string> prof_params;
  if (!TransProfConfigToParam(profiler_config, prof_params)) {
    GELOGE(PARAM_INVALID, "Transfer profilerConfig to string vector failed");
    return PARAM_INVALID;
  }

  GraphLoader graph_loader;
  Command command;
  command.cmd_params.clear();
  command.cmd_type = kProfilingStart;
  command.cmd_params = prof_params;
  command.module_index = profiler_config->config.dataTypeConfig;
  GELOGI("Profiling will start, device nums:%s , deviceID:[%s], data type config: 0x%llx", prof_params[0].c_str(),
         prof_params[kDeviceListIndex].c_str(), command.module_index);
  ret = graph_loader.CommandHandle(command);
  if (ret != SUCCESS) {
    GELOGE(ret, "Handle profiling command failed");
    return FAILED;
  }

  GELOGI("Successfully execute GraphProfStartProfiling.");

  return SUCCESS;
}

Status aclgrphProfStop(aclgrphProfConfig *profiler_config) {
  if (profiler_config == nullptr) {
    GELOGE(PARAM_INVALID, "aclgrphProfConfig is invalid.");
    return FAILED;
  }
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Ge client is not initialized.");
    return FAILED;
  }

  std::lock_guard<std::mutex> lock(g_prof_mutex_);
  // if command mode is set, just return
  if (ProfilingManager::Instance().ProfilingOn()) {
    GELOGW("Graph prof finalize failed, cause profiling command pattern is running.");
    return GE_PROF_MODE_CONFLICT;
  }
  if (!g_graph_prof_init_) {
    GELOGE(GE_PROF_NOT_INIT, "Graph not profiling initialize.");
    return GE_PROF_NOT_INIT;
  }

  Status ret = ProfStopProfiling(&profiler_config->config);
  if (ret != SUCCESS) {
    GELOGE(ret, "Stop profiling failed, prof result = %d", ret);
    return ret;
  }

  std::vector<string> prof_params;
  if (!TransProfConfigToParam(profiler_config, prof_params)) {
    GELOGE(PARAM_INVALID, "Transfer profilerConfig to string vector failed");
    return PARAM_INVALID;
  }

  GraphLoader graph_loader;
  Command command;
  command.cmd_params.clear();
  command.cmd_type = kProfilingStop;
  command.cmd_params = prof_params;
  command.module_index = profiler_config->config.dataTypeConfig;
  GELOGI("Profiling will stop, device nums:%s , deviceID:[%s], data type config: 0x%llx", prof_params[0].c_str(),
         prof_params[kDeviceListIndex].c_str(), command.module_index);
  ret = graph_loader.CommandHandle(command);
  if (ret != SUCCESS) {
    GELOGE(ret, "Handle profiling command failed");
    return FAILED;
  }

  GELOGI("Successfully execute GraphProfStopProfiling.");
  return SUCCESS;
}
}  // namespace ge
