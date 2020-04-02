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

#include "ge/ge_api.h"
#include <iostream>
#include "common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge/datatype_util.h"
#include "proto/ge_api.pb.h"
#include "graph/model_serialize.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/utils/tensor_adapter.h"
#include "init/gelib.h"
#include "session/session_manager.h"
#include "graph/opsproto_manager.h"
#include "graph/utils/type_utils.h"
#include "register/op_registry.h"

using domi::GetContext;
using domi::OpRegistry;
using std::map;
using std::string;
using std::vector;

namespace ge {
static const int32_t kMaxStrLen = 128;
static bool kGeInitialized = false;
static std::mutex kGeReleaseMutex;  // GEFinalize and ~Session use

void GetOpsProtoPath(std::string &opsproto_path) {
  GELOGI("Enter get ops proto path schedule");
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env != nullptr) {
    std::string path = path_env;
    opsproto_path = (path + "/op_proto/built-in/" + ":") + (path + "/op_proto/custom/");
    GELOGI("Get opsproto so path from env: %s", path.c_str());
    return;
  }
  std::string path_base = PluginManager::GetPath();
  GELOGI("path_base is %s", path_base.c_str());
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);
  opsproto_path = (path_base + "ops/op_proto/built-in/" + ":") + (path_base + "ops/op_proto/custom/");
}

Status CheckDumpAndReuseMemory(const std::map<string, string> &options) {
  const int kDecimal = 10;
  auto dump_op_env = std::getenv("DUMP_OP");
  int dump_op_flag = (dump_op_env != nullptr) ? std::strtol(dump_op_env, nullptr, kDecimal) : 0;
  auto disable_reuse_memory_iter = options.find("ge.exec.disableReuseMemory");
  if (disable_reuse_memory_iter != options.end()) {
    if (disable_reuse_memory_iter->second == "0") {
      GELOGD("ge.exec.disableReuseMemory=0, reuse memory is open");
      if (dump_op_flag) {
        GELOGW("Will dump incorrect op data with GE Option ge.exec.disableReuseMemory=0");
      }
    } else if (disable_reuse_memory_iter->second == "1") {
      GELOGD("ge.exec.disableReuseMemory=1, reuse memory is close");
    } else {
      GELOGE(PARAM_INVALID, "CheckDumpAndReuseMemory ge.exec.disableReuseMemory is valid");
      return FAILED;
    }
  } else {
    if (dump_op_flag) {
      GELOGW("Will dump incorrect op data with default reuse memory");
    }
  }
  return SUCCESS;
}

Status CheckOptionsValid(const std::map<string, string> &options) {
  // check job_id is valid
  auto job_id_iter = options.find(OPTION_EXEC_JOB_ID);
  if (job_id_iter != options.end()) {
    if (job_id_iter->second.length() > kMaxStrLen) {
      GELOGE(PARAM_INVALID, "CheckOptionsValid job_id failed, string len > %d", kMaxStrLen);
      return FAILED;
    }
  }

  // Check ge.exec.disableReuseMemory and env DUMP_OP
  if (CheckDumpAndReuseMemory(options) != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

void SaveDdkVersion(const std::map<string, string> &options) {
  auto ddk_option = options.find(DDK_VERSION_FLAG);
  if (ddk_option != options.end()) {
    auto ddk_version = ddk_option->second;
    if (!ddk_version.empty()) {
      GELOGI("Input ddk version : %s.", ddk_version.c_str());
      domi::GetContext().ddk_version = ddk_version;
    }
  } else {
    GELOGW("No ddkVersion!");
    return;
  }
}

// Initialize GE, prepare for execution, call GELib::Initialize
Status GEInitialize(const std::map<string, string> &options) {
  GELOGT(TRACE_INIT, "GEInitialize start");
  // 0.check init status
  if (kGeInitialized) {
    GELOGW("GEInitialize is called more than once");
    return SUCCESS;
  }
  // Load OpsProto lib plugin
  std::string opsproto_path;
  GetOpsProtoPath(opsproto_path);
  OpsProtoManager *manager = OpsProtoManager::Instance();
  std::map<string, string> option_tmp;
  option_tmp.emplace(std::pair<string, string>(string("ge.opsProtoLibPath"), opsproto_path));
  bool is_proto_init = manager->Initialize(option_tmp);
  if (!is_proto_init) {
    GELOGE(GE_CLI_INIT_FAILED, "geInitialize failed, ops proto path is invalid.");
    return FAILED;
  }

  // check options is valid
  if (CheckOptionsValid(options) != SUCCESS) {
    return FAILED;
  }

  SaveDdkVersion(options);

  // call Initialize
  GELOGT(TRACE_RUNNING, "Initializing environment");
  Status ret = ge::GELib::Initialize(options);
  if (ret != SUCCESS) {
    GELOGE(GE_CLI_INIT_FAILED, "geInitialize failed, error code = %u", ret);
    return FAILED;
  }

  // 7.check return status, return
  if (!kGeInitialized) {
    // Initialize success, first time calling initialize
    kGeInitialized = true;
  }

  GELOGT(TRACE_STOP, "GEInitialize finished");
  return ret;
}

// GE finalize, releasing all resources
Status GEFinalize() {
  GELOGT(TRACE_INIT, "GEFinalize start");
  // check init status
  if (!kGeInitialized) {
    GELOGW("GEFinalize is called before GEInitialize");
    return SUCCESS;
  }

  std::lock_guard<std::mutex> lock(kGeReleaseMutex);
  // call Finalize
  GELOGT(TRACE_RUNNING, "Finalizing environment");
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "GEFinalize Failed: GE not initialized");
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  Status ret = instance_ptr->Finalize();
  GELOGI("GEFinalize finalize gelib ret=%u", ret);
  if (ret != SUCCESS) {
    GELOGE(ret, "GEFinalize Failed");
    return FAILED;
  }

  if (kGeInitialized && ret == SUCCESS) {
    kGeInitialized = false;
  }

  GELOGT(TRACE_STOP, "GEFinalize finished");
  return ret;
}

// Initialize session，which calls innerSession
Session::Session(const std::map<string, string> &options) {
  GELOGT(TRACE_INIT, "Session Constructor start");
  // check init status
  sessionId_ = 0;
  if (!kGeInitialized) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED);
    return;
  }
  // call Initialize
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Session Constructor failed");
    return;
  }

  GELOGT(TRACE_RUNNING, "Creating session");
  uint64_t session_id = 0;
  Status ret = instance_ptr->SessionManagerObj().CreateSession(options, session_id);
  GELOGT(TRACE_RUNNING, "Session id is %lu", session_id);

  // check return status, return, update session id if success
  if (ret == SUCCESS) {
    sessionId_ = session_id;
  } else {
    GELOGE(ret, "Session constructor failed, session Id not initialized");
    return;
  }
  GELOGT(TRACE_STOP, "Session Constructor finished");
}

// session destructor
Session::~Session() {
  GELOGT(TRACE_INIT, "Session Destructor start");
  // 0.check init status
  if (!kGeInitialized) {
    GELOGW("GE is not yet initialized or is finalized.");
    return;
  }

  Status ret = FAILED;
  std::lock_guard<std::mutex> lock(kGeReleaseMutex);
  try {
    uint64_t session_id = sessionId_;
    // call DestroySession
    std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
      GELOGW("GE is not yet initialized or is finalized.");
      return;
    }
    GELOGT(TRACE_RUNNING, "Session id is %lu", session_id);

    GELOGT(TRACE_RUNNING, "Destroying session");

    ret = instance_ptr->SessionManagerObj().DestroySession(session_id);
  } catch (google::protobuf::FatalException &e) {
    GELOGE(GE_CLI_SESS_DESTROY_FAILED, "SessionDestructor throws FatalException");
  }

  // check return status, return, update session id if success
  if (ret != SUCCESS) {
    GELOGE(ret, "Session Destructor failed");
  }

  GELOGT(TRACE_STOP, "Session Destructor finished");
}

Status Session::AddGraph(uint32_t graph_id, const Graph &graph) {
  std::map<std::string, std::string> options;
  return AddGraph(graph_id, graph, options);
}

Status Session::AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options) {
  GELOGT(TRACE_INIT, "Start to add graph in Session. graph_id: %u, sessinon_id: %lu.", graph_id, sessionId_);
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "AddGraph failed in Sesson.");
    return FAILED;
  }
  GELOGD("Adding graph to session");
  Status ret = instance_ptr->SessionManagerObj().AddGraph(sessionId_, graph_id, graph, options);
  if (ret != SUCCESS) {
    GELOGE(ret, "AddGraph failed in Session.");
    return FAILED;
  }
  GELOGD("AddGraph finished in Session.");
  return ret;
}

Status Session::RemoveGraph(uint32_t graph_id) {
  GELOGT(TRACE_INIT, "Session RemoveGraph start");

  // call RemoveGraph
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (!instance_ptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Session RemoveGraph failed");
    return FAILED;
  }

  GELOGT(TRACE_RUNNING, "Removing Graph from session");
  Status ret = instance_ptr->SessionManagerObj().RemoveGraph(sessionId_, graph_id);
  // check return status, return
  if (ret != SUCCESS) {
    GELOGE(ret, "session RemoveGraph failed");
    return FAILED;
  }
  GELOGT(TRACE_STOP, "Session RemoveGraph finished");
  return ret;
}

void PrintOutputResult(std::vector<Tensor> &outputs) {
  if (outputs.empty() || outputs[0].GetData() == nullptr) {
    GELOGW("outputs is empty or data is nullptr.");
    return;
  }

  size_t out_buf_size = outputs[0].GetSize();
  TensorDesc desc(outputs[0].GetTensorDesc());
  DataType data_type = desc.GetDataType();
  auto iter = CONST_OPDATA_TYPE_SIZE_MAP.find(data_type);
  if (iter == CONST_OPDATA_TYPE_SIZE_MAP.end()) {
    GELOGI("DataType %s has not defined size", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return;
  }
  size_t length = CONST_OPDATA_TYPE_SIZE_MAP[data_type];
  for (size_t i = 0; i < 10 && i < (out_buf_size / length); ++i) {  // take first 10 at most
    switch (data_type) {
      case DT_BOOL:
      case DT_INT8:
      case DT_UINT8:
        GELOGI("output data[%zu]=%d", i, *(reinterpret_cast<int8_t *>(outputs[0].GetData()) + i));
        break;
      case DT_INT16:
      case DT_UINT16:
        GELOGI("output data[%zu]=%d", i, *(reinterpret_cast<int16_t *>(outputs[0].GetData()) + i));
        break;
      case DT_INT32:
      case DT_UINT32:
        GELOGI("output data[%zu]=%d", i, *(reinterpret_cast<int32_t *>(outputs[0].GetData()) + i));
        break;
      case DT_INT64:
      case DT_UINT64:
        GELOGI("output data[%zu]=%ld", i, *(reinterpret_cast<int64_t *>(outputs[0].GetData()) + i));
        break;
      case DT_FLOAT:
        GELOGI("output data[%zu]=%f", i, *(reinterpret_cast<float *>(outputs[0].GetData()) + i));
        break;
      case DT_DOUBLE:
        GELOGI("output data[%zu]=%lf", i, *(reinterpret_cast<double *>(outputs[0].GetData()) + i));
        break;
      default:
        GELOGI("Output datatype %s is not support print.", TypeUtils::DataTypeToSerialString(data_type).c_str());
        return;
    }
  }
}

Status Session::RunGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  GELOGT(TRACE_INIT, "Session RunGraph start");

  std::vector<Tensor> graph_inputs = inputs;
  // call RunGraph
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Session RunGraph failed");
    return FAILED;
  }
  GELOGT(TRACE_RUNNING, "Running Graph");
  Status ret = instance_ptr->SessionManagerObj().RunGraph(sessionId_, graph_id, graph_inputs, outputs);
  // check return status
  if (ret != SUCCESS) {
    GELOGE(ret, "Session RunGraph failed");
    return FAILED;
  }

  // print output
  if (outputs.size() > 0) {
    PrintOutputResult(outputs);
  }

  // return
  GELOGT(TRACE_STOP, "Session RunGraph finished");
  return ret;
}

Status Session::RegisterCallBackFunc(const std::string &key, const pCallBackFunc &callback) {
  GELOGW(
    "The callback function will not be checked. Please ensure that the implementation of the function is trusted.");
  return ge::GELib::GetInstance()->SessionManagerObj().RegisterCallBackFunc(sessionId_, key, callback);
}

Status Session::RunGraphAsync(uint32_t graph_id, const std::vector<TensorInfo> &inputs,
                              std::vector<TensorInfo> &outputs, std::function<void(Status)> callback) {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "SessionConstructor failed");
    return FAILED;
  }
  GELOGT(TRACE_RUNNING, "Run Graph Asynchronously");
  GELOGW(
    "The callback function will not be checked. Please ensure that the implementation of the function is trusted.");

  Status ret =
    ge::GELib::GetInstance()->SessionManagerObj().RunGraphAsync(sessionId_, graph_id, inputs, outputs, callback);
  if (ret != SUCCESS) {
    GELOGE(ret, "SessionManager RunGraphAsync failed");
    return FAILED;
  }
  return SUCCESS;
}
bool Session::IsGraphNeedRebuild(uint32_t graph_id) {
  return ge::GELib::GetInstance()->SessionManagerObj().IsGraphNeedRebuild(sessionId_, graph_id);
}
}  // namespace ge
