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
#include "framework/common/debug/ge_log.h"
#include "common/ge/plugin_manager.h"
#include "common/ge/ge_util.h"
#include "common/profiling/profiling_manager.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/graph_var_manager.h"
#include "runtime/kernel.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "ge/ge_api_types.h"
#include <cstdlib>
#include "graph/load/new_model_manager/model_manager.h"
#include "omm/csa_interact.h"
#include "common/properties_manager.h"

using Json = nlohmann::json;

namespace ge {
namespace {
const int kDecimal = 10;
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
  GetMutableGlobalOptions().insert(options.begin(), options.end());
  GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  GE_TIMESTAMP_START(Init);
  Status ret = instancePtr_->InnerInitialize(options);
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
  Status init_system_status = SystemInitialize(options);
  if (init_system_status != SUCCESS) {
    GELOGE(init_system_status);
    RollbackInit();
    return init_system_status;
  }

  GELOGI("engineManager initial.");
  Status init_em_status = engine_manager_.Initialize(options);
  if (init_em_status != SUCCESS) {
    GELOGE(init_em_status);
    RollbackInit();
    return init_em_status;
  }

  GELOGI("opsManager initial.");
  Status init_ops_status = ops_manager_.Initialize(options);
  if (init_ops_status != SUCCESS) {
    GELOGE(init_ops_status);
    RollbackInit();
    return init_ops_status;
  }

  GELOGI("sessionManager initial.");
  Status init_sm_status = session_manager_.Initialize(options);
  if (init_sm_status != SUCCESS) {
    GELOGE(init_sm_status);
    RollbackInit();
    return init_sm_status;
  }

  GELOGI("memoryMallocSize initial.");
  Status init_mem_status = VarManager::Instance(0)->SetMemoryMallocSize(options);
  if (init_mem_status != SUCCESS) {
    GELOGE(init_mem_status, "failed to set malloc size");
    RollbackInit();
    return init_mem_status;
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
        dump_path += "/";
      }

      PropertiesManager::Instance().AddDumpPropertyValue(DUMP_ALL_MODEL, {});
      PropertiesManager::Instance().SetDumpOutputPath(dump_path);
    }
  }

  if (is_train_mode_) {
    InitOptions(options);
    status = InitSystemWithOptions(this->options_);
  } else {
    status = InitSystemWithoutOptions();
  }
  return status;
}

void GELib::InitOptions(const map<string, string> &options) {
  this->options_.session_id = 0;
  auto iter = options.find(OPTION_EXEC_SESSION_ID);
  if (iter != options.end()) {
    this->options_.session_id = std::strtoll(iter->second.c_str(), nullptr, kDecimal);
  }
  this->options_.device_id = 0;
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
  this->options_.deployMode = false;
  iter = options.find(OPTION_EXEC_DEPLOY_MODE);
  if (iter != options.end()) {
    std::istringstream(iter->second) >> this->options_.deployMode;
  }

  iter = options.find(OPTION_EXEC_POD_NAME);
  if (iter != options.end()) {
    this->options_.podName = iter->second.c_str();
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
  GELOGI("Training init GELib. session Id:%ld, device id :%d ", options.session_id, options.device_id);
  GEEVENT("System init with options begin, job id %s", options.job_id.c_str());
  std::lock_guard<std::mutex> lock(status_mutex_);
  GE_IF_BOOL_EXEC(is_system_inited && !is_shutdown,
                  GELOGW("System init with options is already inited and not shutdown.");
                  return SUCCESS);
  GetContext().Init();

  // profiling init
  if (ProfilingManager::Instance().Init(options) != SUCCESS) {
    GELOGW("Profiling init failed.");
  }

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

  GELOGI("Training init GELib success.");

  return SUCCESS;
}

Status GELib::SystemShutdownWithOptions(const Options &options) {
  GELOGI("Training finalize GELib begin.");

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

  if (!ProfilingManager::Instance().ProfilingOpTraceOn() && ProfilingManager::Instance().ProfilingOn()) {
    ProfilingManager::Instance().StopProfiling();
  }

  is_system_inited = false;
  is_shutdown = true;

  GELOGI("Training finalize GELib success.");

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

  GELOGI("engineManager finalization.");
  Status final_em_status = engine_manager_.Finalize();
  GELOGI("sessionManager finalization.");
  Status final_sm_status = session_manager_.Finalize();

  if (final_em_status != SUCCESS) {
    GELOGE(final_em_status);
    return final_em_status;
  }

  if (final_sm_status != SUCCESS) {
    GELOGE(final_sm_status);
    return final_sm_status;
  }

  GELOGI("opsManager finalization.");
  Status final_ops_status = ops_manager_.Finalize();
  if (final_ops_status != SUCCESS) {
    GELOGE(final_ops_status);
    return final_ops_status;
  }

  GELOGI("VarManagerPool finalization.");
  VarManagerPool::Instance().Destroy();

  GELOGI("MemManager finalization.");
  MemManager::Instance().Finalize();

#ifdef DAVINCI_CLOUD
  if (is_train_mode_) {
    GELOGI("System ShutDown.");
    Status shutdown_status = SystemShutdownWithOptions(this->options_);
    if (shutdown_status != SUCCESS) {
      GELOGE(shutdown_status);
      return shutdown_status;
    }
  }
  is_train_mode_ = false;
#endif

  instancePtr_ = nullptr;
  init_flag_ = false;
  GELOGI("finalization success.");
  return SUCCESS;
}

// Get Singleton Instance
std::shared_ptr<GELib> GELib::GetInstance() { return instancePtr_; }

void GELib::RollbackInit() {
  if (engine_manager_.init_flag_) {
    (void)engine_manager_.Finalize();
  }
  if (ops_manager_.init_flag_) {
    (void)ops_manager_.Finalize();
  }
  if (session_manager_.init_flag_) {
    (void)session_manager_.Finalize();
  }
  MemManager::Instance().Finalize();
  VarManagerPool::Instance().Destroy();
}
}  // namespace ge
