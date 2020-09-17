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

#include "init/gelib.h"

#include <dlfcn.h>
#include <cstdlib>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "common/ge/ge_util.h"
#include "common/ge/plugin_manager.h"
#include "common/profiling/profiling_manager.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "ge/ge_api_types.h"
#include "ge_local_engine/engine/host_cpu_engine.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/graph_var_manager.h"
#include "omm/csa_interact.h"
#include "runtime/kernel.h"

using Json = nlohmann::json;

namespace ge {
namespace {
const int kDecimal = 10;
const int kSocVersionLen = 50;
const int kDefaultDeviceIdForTrain = 0;
const int kDefaultDeviceIdForInfer = -1;
}  // namespace
static std::shared_ptr<GELib> instancePtr_ = nullptr;

// Initial each module of GE, if one failed, release all
Status GELib::Initialize(const map<string, string> &options) {
  GELOGI("initial start");
  GEEVENT("[GEPERFTRACE] GE Init Start");
  // Multiple initializations are not allowed
  instancePtr_ = MakeShared<GELib>();
  if (instancePtr_ == nullptr) {
    GELOGE(GE_CLI_INIT_FAILED, "GeLib initialize failed, malloc shared_ptr failed.");
    return GE_CLI_INIT_FAILED;
  }

  map<string, string> new_options;
  Status ret = instancePtr_->SetRTSocVersion(options, new_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "GeLib initial failed.");
    return ret;
  }
  GetMutableGlobalOptions().insert(new_options.begin(), new_options.end());
  GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  GE_TIMESTAMP_START(Init);
  ret = instancePtr_->InnerInitialize(new_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "GeLib initial failed.");
    instancePtr_ = nullptr;
    return ret;
  }
  GE_TIMESTAMP_END(Init, "GELib::Initialize");
  return SUCCESS;
}

Status GELib::InnerInitialize(const map<string, string> &options) {
  // Multiple initializations are not allowed
  if (init_flag_) {
    GELOGW("multi initializations");
    return SUCCESS;
  }

  GELOGI("GE System initial.");
  GE_TIMESTAMP_START(SystemInitialize);
  Status initSystemStatus = SystemInitialize(options);
  GE_TIMESTAMP_END(SystemInitialize, "InnerInitialize::SystemInitialize");
  if (initSystemStatus != SUCCESS) {
    GELOGE(initSystemStatus);
    RollbackInit();
    return initSystemStatus;
  }

  GELOGI("engineManager initial.");
  GE_TIMESTAMP_START(EngineInitialize);
  Status initEmStatus = engineManager_.Initialize(options);
  GE_TIMESTAMP_END(EngineInitialize, "InnerInitialize::EngineInitialize");
  if (initEmStatus != SUCCESS) {
    GELOGE(initEmStatus);
    RollbackInit();
    return initEmStatus;
  }

  GELOGI("opsManager initial.");
  GE_TIMESTAMP_START(OpsManagerInitialize);
  Status initOpsStatus = opsManager_.Initialize(options);
  GE_TIMESTAMP_END(OpsManagerInitialize, "InnerInitialize::OpsManagerInitialize");
  if (initOpsStatus != SUCCESS) {
    GELOGE(initOpsStatus);
    RollbackInit();
    return initOpsStatus;
  }

  GELOGI("sessionManager initial.");
  GE_TIMESTAMP_START(SessionManagerInitialize);
  Status initSmStatus = sessionManager_.Initialize(options);
  GE_TIMESTAMP_END(SessionManagerInitialize, "InnerInitialize::SessionManagerInitialize");
  if (initSmStatus != SUCCESS) {
    GELOGE(initSmStatus);
    RollbackInit();
    return initSmStatus;
  }

  GELOGI("memoryMallocSize initial.");
  GE_TIMESTAMP_START(SetMemoryMallocSize);
  Status initMemStatus = VarManager::Instance(0)->SetMemoryMallocSize(options);
  GE_TIMESTAMP_END(SetMemoryMallocSize, "InnerInitialize::SetMemoryMallocSize");
  if (initMemStatus != SUCCESS) {
    GELOGE(initMemStatus, "failed to set malloc size");
    RollbackInit();
    return initMemStatus;
  }

  GELOGI("Start to initialize HostCpuEngine");
  GE_TIMESTAMP_START(HostCpuEngineInitialize);
  Status initHostCpuEngineStatus = HostCpuEngine::GetInstance().Initialize();
  GE_TIMESTAMP_END(HostCpuEngineInitialize, "InnerInitialize::HostCpuEngineInitialize");
  if (initHostCpuEngineStatus != SUCCESS) {
    GELOGE(initHostCpuEngineStatus, "Failed to initialize HostCpuEngine");
    RollbackInit();
    return initHostCpuEngineStatus;
  }

  init_flag_ = true;
  GELOGI("GeLib initial success.");
  return SUCCESS;
}

Status GELib::SystemInitialize(const map<string, string> &options) {
  Status status = FAILED;
  auto iter = options.find(OPTION_GRAPH_RUN_MODE);
  if (iter != options.end()) {
    if (GraphRunMode(std::strtol(iter->second.c_str(), nullptr, kDecimal)) >= TRAIN) {
      is_train_mode_ = true;
    }
  }

  iter = options.find(HEAD_STREAM);
  head_stream_ = (iter != options.end()) ? std::strtol(iter->second.c_str(), nullptr, kDecimal) : false;

  iter = options.find(OPTION_EXEC_ENABLE_DUMP);
  if (iter != options.end()) {
    int32_t enable_dump_flag = 1;
    auto path_iter = options.find(OPTION_EXEC_DUMP_PATH);
    if (iter->second == std::to_string(enable_dump_flag) && path_iter != options.end()) {
      std::string dump_path = path_iter->second;
      if (!dump_path.empty() && dump_path[dump_path.size() - 1] != '/') {
        dump_path = dump_path + "/" + CurrentTimeInStr() + "/";
      }

      PropertiesManager::Instance().AddDumpPropertyValue(DUMP_ALL_MODEL, {});
      GELOGD("Get dump path %s successfully", dump_path.c_str());
      PropertiesManager::Instance().SetDumpOutputPath(dump_path);
    }
    auto step_iter = options.find(OPTION_EXEC_DUMP_STEP);
    if (step_iter != options.end()) {
      std::string dump_step = step_iter->second;
      GELOGD("Get dump step %s successfully", dump_step.c_str());
      PropertiesManager::Instance().SetDumpStep(dump_step);
    }
    auto mode_iter = options.find(OPTION_EXEC_DUMP_MODE);
    if (mode_iter != options.end()) {
      std::string dump_mode = mode_iter->second;
      GELOGD("Get dump mode %s successfully", dump_mode.c_str());
      PropertiesManager::Instance().SetDumpMode(dump_mode);
    }
  }

  // In train and infer, profiling is always needed.
  InitOptions(options);
  InitProfiling(this->options_);
  // 1.`is_train_mode_` means case: train
  // 2.`(!is_train_mode_) && (options_.device_id != kDefaultDeviceIdForInfer)` means case: online infer
  // these two case need call `InitSystemWithOptions->rtGetDeviceIndexByPhyId`
  // to convert phy device id to logical device id
  // note:rtGetDeviceIndexByPhyId return `0` logical id when input phy device id is `0`
  if (is_train_mode_ || (options_.device_id != kDefaultDeviceIdForInfer)) {
    status = InitSystemWithOptions(this->options_);
  } else {
    status = InitSystemWithoutOptions();
  }
  return status;
}

void GELib::InitProfiling(Options &options) {
  GELOGI("Init Profiling. session Id: %ld, device id:%d ", options.session_id, options.device_id);
  std::lock_guard<std::mutex> lock(status_mutex_);
  GetContext().Init();
  // Profiling init
  if (ProfilingManager::Instance().Init(options) != SUCCESS) {
    GELOGW("Profiling init failed.");
  }
}

Status GELib::SetRTSocVersion(const map<string, string> &options, map<string, string> &new_options) {
  GELOGI("Start to set SOC_VERSION");
  new_options.insert(options.begin(), options.end());
  auto it = new_options.find(ge::SOC_VERSION);
  if (it != new_options.end()) {
    GE_CHK_RT_RET(rtSetSocVersion(it->second.c_str()));
    GELOGI("Succeeded in setting SOC_VERSION[%s] to runtime.", it->second.c_str());
  } else {
    GELOGI("SOC_VERSION is not exist in options");
    char version[kSocVersionLen] = {0};
    rtError_t rt_ret = rtGetSocVersion(version, kSocVersionLen);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtGetSocVersion failed"); return FAILED;)
    GELOGI("Succeeded in getting SOC_VERSION[%s] from runtime.", version);
    new_options.insert(std::make_pair(ge::SOC_VERSION, version));
  }
  return SUCCESS;
}

void GELib::InitOptions(const map<string, string> &options) {
  this->options_.session_id = 0;
  auto iter = options.find(OPTION_EXEC_SESSION_ID);
  if (iter != options.end()) {
    this->options_.session_id = std::strtoll(iter->second.c_str(), nullptr, kDecimal);
  }
  this->options_.device_id = is_train_mode_ ? kDefaultDeviceIdForTrain : kDefaultDeviceIdForInfer;
  iter = options.find(OPTION_EXEC_DEVICE_ID);
  if (iter != options.end()) {
    this->options_.device_id = static_cast<int32_t>(std::strtol(iter->second.c_str(), nullptr, kDecimal));
  }
  iter = options.find(OPTION_EXEC_JOB_ID);
  if (iter != options.end()) {
    this->options_.job_id = iter->second.c_str();
  }
  this->options_.isUseHcom = false;
  iter = options.find(OPTION_EXEC_IS_USEHCOM);
  if (iter != options.end()) {
    std::istringstream(iter->second) >> this->options_.isUseHcom;
  }
  this->options_.isUseHvd = false;
  iter = options.find(OPTION_EXEC_IS_USEHVD);
  if (iter != options.end()) {
    std::istringstream(iter->second) >> this->options_.isUseHvd;
  }
  this->options_.deployMode = false;
  iter = options.find(OPTION_EXEC_DEPLOY_MODE);
  if (iter != options.end()) {
    std::istringstream(iter->second) >> this->options_.deployMode;
  }
  iter = options.find(OPTION_EXEC_POD_NAME);
  if (iter != options.end()) {
    this->options_.podName = iter->second.c_str();
  }
  iter = options.find(OPTION_EXEC_PROFILING_MODE);
  if (iter != options.end()) {
    this->options_.profiling_mode = iter->second.c_str();
  }
  iter = options.find(OPTION_EXEC_PROFILING_OPTIONS);
  if (iter != options.end()) {
    this->options_.profiling_options = iter->second.c_str();
  }
  iter = options.find(OPTION_EXEC_RANK_ID);
  if (iter != options.end()) {
    this->options_.rankId = std::strtoll(iter->second.c_str(), nullptr, kDecimal);
  }
  iter = options.find(OPTION_EXEC_RANK_TABLE_FILE);
  if (iter != options.end()) {
    this->options_.rankTableFile = iter->second.c_str();
  }
  this->options_.enable_atomic = true;
  iter = options.find(OPTION_EXEC_ATOMIC_FLAG);
  GE_IF_BOOL_EXEC(iter != options.end(),
                  this->options_.enable_atomic = std::strtol(iter->second.c_str(), nullptr, kDecimal));
  GELOGI("ge InnerInitialize, the enable_atomic_flag in options_ is %d", this->options_.enable_atomic);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status GELib::InitSystemWithOptions(Options &options) {
  std::string mode = is_train_mode_ ? "Training" : "Online infer";
  GELOGI("%s init GELib. session Id:%ld, device id :%d ", mode.c_str(), options.session_id, options.device_id);
  GEEVENT("System init with options begin, job id %s", options.job_id.c_str());
  std::lock_guard<std::mutex> lock(status_mutex_);
  GE_IF_BOOL_EXEC(is_system_inited && !is_shutdown,
                  GELOGW("System init with options is already inited and not shutdown.");
                  return SUCCESS);

  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  Status initMmStatus = MemManager::Instance().Initialize(mem_type);
  if (initMmStatus != SUCCESS) {
    GELOGE(initMmStatus, "[Initialize] MemoryAllocatorManager initialize failed.");
    return initMmStatus;
  }

  // Update CSA file
  CsaInteract::GetInstance().Init(options.device_id, GetContext().TraceId());
  Status ret = CsaInteract::GetInstance().WriteJobState(JOBSTATE_RUNNING, JOBSUBSTATE_ENV_INIT);
  GE_LOGE_IF(ret != SUCCESS, "write job state failed, ret:%u", ret);
  options.physical_device_id = options.device_id;

  // The physical ID is transferred to the logical ID. FMK receives physical ID and needs to be converted
  uint32_t dev_logic_index = 0;
  rtError_t rt_ret = rtGetDeviceIndexByPhyId(static_cast<uint32_t>(options.device_id), &dev_logic_index);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  GELOGE(rt_ret, "rtGetDeviceIndexByPhyId transform index by phyId %d failed", options.device_id);
                  CsaInteract::GetInstance().WriteErrorCode(rt_ret, ERROR_MODULE_RUNTIME, JOBSUBSTATE_ENV_INIT);
                  return FAILED);
  options.device_id = static_cast<int32_t>(dev_logic_index);
  GELOGI("rtGetDeviceIndexByPhyId physical device id:%d,logical device id:%u", options.device_id, dev_logic_index);

  GetContext().SetCtxDeviceId(dev_logic_index);

  GE_CHK_RT_RET(rtSetDevice(options.device_id));

  // In the scenario that the automatic add fusion is set, but there is no cleanaddr operator,
  // maybe need to check it
  is_system_inited = true;
  is_shutdown = false;

  GELOGI("%s init GELib success.", mode.c_str());

  return SUCCESS;
}

Status GELib::SystemShutdownWithOptions(const Options &options) {
  std::string mode = is_train_mode_ ? "Training" : "Online infer";
  GELOGI("%s finalize GELib begin.", mode.c_str());

  std::lock_guard<std::mutex> lock(status_mutex_);
  GE_IF_BOOL_EXEC(is_shutdown || !is_system_inited,
                  GELOGW("System Shutdown with options is already is_shutdown or system does not inited. "
                         "is_shutdown:%d is_omm_inited:%d",
                         is_shutdown, is_system_inited);
                  return SUCCESS);

  GE_CHK_RT(rtDeviceReset(options.device_id));

  // Update CSA file
  Status ret = CsaInteract::GetInstance().WriteJobState(JOBSTATE_SUCCEED);
  GE_LOGE_IF(ret != SUCCESS, "write job state failed, ret:%u", ret);

  is_system_inited = false;
  is_shutdown = true;

  GELOGI("%s finalize GELib success.", mode.c_str());
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status GELib::InitSystemWithoutOptions() {
  GELOGI("Inference Init GELib begin.");

  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  Status initMmStatus = MemManager::Instance().Initialize(mem_type);
  if (initMmStatus != SUCCESS) {
    GELOGE(initMmStatus, "[Initialize] MemoryAllocatorManager initialize failed.");
    return initMmStatus;
  }

  static bool is_inited = false;
  if (is_inited) {
    GELOGW("System init without options is already inited,  don't need to init again.");
    return SUCCESS;
  }
  is_inited = true;
  GELOGI("Inference init GELib success.");

  return SUCCESS;
}

string GELib::GetPath() { return PluginManager::GetPath(); }

// Finalize all modules
Status GELib::Finalize() {
  GELOGI("finalization start");
  // Finalization is not allowed before initialization
  if (!init_flag_) {
    GELOGW("not initialize");
    return SUCCESS;
  }
  Status final_state = SUCCESS;
  Status mid_state;
  GELOGI("engineManager finalization.");
  mid_state = engineManager_.Finalize();
  if (mid_state != SUCCESS) {
    GELOGW("engineManager finalize failed");
    final_state = mid_state;
  }
  GELOGI("sessionManager finalization.");
  mid_state = sessionManager_.Finalize();
  if (mid_state != SUCCESS) {
    GELOGW("sessionManager finalize failed");
    final_state = mid_state;
  }

  GELOGI("opsManager finalization.");
  mid_state = opsManager_.Finalize();
  if (mid_state != SUCCESS) {
    GELOGW("opsManager finalize failed");
    final_state = mid_state;
  }

  GELOGI("VarManagerPool finalization.");
  VarManagerPool::Instance().Destory();

  GELOGI("MemManager finalization.");
  MemManager::Instance().Finalize();

  GELOGI("HostCpuEngine finalization.");
  HostCpuEngine::GetInstance().Finalize();

  // Shut down profiling
  ShutDownProfiling();

  if (is_train_mode_ || (options_.device_id != kDefaultDeviceIdForInfer)) {
    GELOGI("System ShutDown.");
    mid_state = SystemShutdownWithOptions(this->options_);
    if (mid_state != SUCCESS) {
      GELOGW("System shutdown with options failed");
      final_state = mid_state;
    }
  }

  is_train_mode_ = false;

  GetMutableGlobalOptions().erase(ENABLE_SINGLE_STREAM);

  instancePtr_ = nullptr;
  init_flag_ = false;
  if (final_state != SUCCESS) {
    GELOGE(FAILED, "MemManager finalization.");
    return final_state;
  }
  GELOGI("finalization success.");
  return SUCCESS;
}

void GELib::ShutDownProfiling() {
  std::lock_guard<std::mutex> lock(status_mutex_);

  if (!ProfilingManager::Instance().ProfilingOpTraceOn() && ProfilingManager::Instance().ProfilingOn()) {
    ProfilingManager::Instance().StopProfiling();
  }
  if (ProfilingManager::Instance().ProfilingOn()) {
    ProfilingManager::Instance().PluginUnInit(GE_PROFILING_MODULE);
  }
}

// Get Singleton Instance
std::shared_ptr<GELib> GELib::GetInstance() { return instancePtr_; }

void GELib::RollbackInit() {
  if (engineManager_.init_flag_) {
    (void)engineManager_.Finalize();
  }
  if (opsManager_.init_flag_) {
    (void)opsManager_.Finalize();
  }
  if (sessionManager_.init_flag_) {
    (void)sessionManager_.Finalize();
  }
  MemManager::Instance().Finalize();
  VarManagerPool::Instance().Destory();
}
}  // namespace ge
