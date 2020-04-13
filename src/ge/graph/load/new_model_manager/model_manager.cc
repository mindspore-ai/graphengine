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

#include "graph/load/new_model_manager/model_manager.h"

#include <string>

#include "common/l2_cache_optimize.h"
#include "common/profiling/profiling_manager.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"

namespace ge {
thread_local uint32_t device_count = 0;
namespace {
const int kCmdParSize = 2;
const int kDumpCmdPairSize = 2;
}  // namespace

std::shared_ptr<ModelManager> ModelManager::GetInstance() {
  static const std::shared_ptr<ModelManager> instance_ptr =
    shared_ptr<ModelManager>(new (std::nothrow) ModelManager(), ModelManager::FinalizeForPtr);
  return instance_ptr;
}

ModelManager::ModelManager() { max_model_id_ = 0; }

Status ModelManager::KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType op_type, uint64_t session_id, uint32_t model_id) {
  STR_FWK_OP_KERNEL param_base = {};
  void *devicebase = nullptr;
  void *aicpu_kernel_addr = nullptr;
  const uint32_t kKernelType = 0;
  param_base.fwkKernelType = kKernelType;
  param_base.fwkKernelBase.fwk_kernel.opType = op_type;
  param_base.fwkKernelBase.fwk_kernel.sessionID = session_id;
  if (op_type == aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_DESTROY) {
    std::vector<uint64_t> v_aicpu_kernel;
    std::string model_key = std::to_string(session_id) + "_" + std::to_string(model_id);
    auto iter = model_aicpu_kernel_.find(model_key);
    if (iter != model_aicpu_kernel_.end()) {
      GELOGD("kernel destroy session_id %lu, model_id %u.", session_id, model_id);
      v_aicpu_kernel = model_aicpu_kernel_.at(model_key);
      // Insert size of aicpu kernel vector in the first element
      v_aicpu_kernel.insert(v_aicpu_kernel.begin(), v_aicpu_kernel.size());

      auto kernel_size = sizeof(uint64_t) * (v_aicpu_kernel.size());
      rtError_t rt_ret = rtMalloc(&aicpu_kernel_addr, kernel_size, RT_MEMORY_HBM);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "rtMalloc error, ret: 0x%X", rt_ret);
                      return RT_FAILED;)

      rt_ret = rtMemcpy(aicpu_kernel_addr, kernel_size, v_aicpu_kernel.data(), kernel_size, RT_MEMCPY_HOST_TO_DEVICE);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtMemcpy to input_output_addr_ error: 0x%X", rt_ret);
                      GE_CHK_RT(rtFree(aicpu_kernel_addr)); return FAILED;)
      uint64_t kernel_id_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(aicpu_kernel_addr));
      param_base.fwkKernelBase.fwk_kernel.kernelID = kernel_id_addr;
      // Remove model key from map
      model_aicpu_kernel_.erase(iter);
    }
  }

  rtError_t rt_ret = rtMalloc(&(devicebase), sizeof(STR_FWK_OP_KERNEL), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "malloc device memory failed.");
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    return FAILED;
  }

  rt_ret =
    rtMemcpy(devicebase, sizeof(STR_FWK_OP_KERNEL), &param_base, sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "memory copy to device failed.");
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    GE_CHK_RT(rtFree(devicebase));
    return FAILED;
  }

  rtStream_t stream = nullptr;
  rt_ret = rtStreamCreate(&stream, 0);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "create stream failed.");
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    GE_CHK_RT(rtFree(devicebase));
    return FAILED;
  }

  rt_ret = rtKernelLaunchEx(devicebase, sizeof(STR_FWK_OP_KERNEL), 0, stream);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "rtKernelLaunchEx failed.");
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    GE_CHK_RT(rtFree(devicebase));
    GE_CHK_RT(rtStreamDestroy(stream));
    return FAILED;
  }
  rt_ret = rtStreamSynchronize(stream);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "rtStreamSynchronize failed.");
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    GE_CHK_RT(rtFree(devicebase));
    GE_CHK_RT(rtStreamDestroy(stream));
    return FAILED;
  }
  if (aicpu_kernel_addr != nullptr) {
    rt_ret = rtFree(aicpu_kernel_addr);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(rt_ret, "free memory failed.");
      GE_CHK_RT(rtFree(devicebase));
      GE_CHK_RT(rtStreamDestroy(stream));
      return FAILED;
    }
  }
  rt_ret = rtFree(devicebase);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "free memory failed.");
    GE_CHK_RT(rtStreamDestroy(stream));
    return FAILED;
  }
  rt_ret = rtStreamDestroy(stream);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "rtStreamDestroy failed.");
    return FAILED;
  }
  return SUCCESS;
}

void ModelManager::DestroyAicpuSession(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(sess_ids_mutex_);
  auto it = sess_ids_.find(session_id);
  if (it == sess_ids_.end()) {
    GELOGI("The session: %lu not created.", session_id);
    return;
  } else {
    Status ret = KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_SESSION_DESTROY, session_id, 0);
    if (ret != SUCCESS) {
      GELOGW("The session: %lu destroy failed.", session_id);
    } else {
      (void)sess_ids_.erase(session_id);
      GELOGI("The session: %lu destroyed.", session_id);
    }
  }
}

ge::Status ModelManager::DestroyAicpuKernel(uint64_t session_id, uint32_t model_id) {
  GELOGD("destroy aicpu kernel in session_id %lu, model_id %u.", session_id, model_id);
  std::lock_guard<std::mutex> lock(sess_ids_mutex_);
  std::string model_key = std::to_string(session_id) + "_" + std::to_string(model_id);
  if (model_aicpu_kernel_.find(model_key) != model_aicpu_kernel_.end()) {
    Status ret = KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_DESTROY, session_id, model_id);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Destroy aicpu kernel failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

ge::Status ModelManager::CreateAicpuKernel(uint64_t session_id, uint32_t model_id, uint64_t kernel_id) {
  std::vector<uint64_t> v_aicpu_kernel;
  std::lock_guard<std::mutex> lock(sess_ids_mutex_);
  std::string model_key = std::to_string(session_id) + "_" + std::to_string(model_id);
  if (model_aicpu_kernel_.find(model_key) != model_aicpu_kernel_.end()) {
    v_aicpu_kernel = model_aicpu_kernel_.at(model_key);
  }
  v_aicpu_kernel.push_back(kernel_id);
  model_aicpu_kernel_[model_key] = v_aicpu_kernel;
  return SUCCESS;
}

ModelManager::~ModelManager() {
  std::lock_guard<std::mutex> lock(map_mutex_);
  model_map_.clear();
  model_aicpu_kernel_.clear();

  GE_IF_BOOL_EXEC(device_count > 0, GE_CHK_RT(rtDeviceReset(0)));
}

///
/// @ingroup domi_ome
/// @brief set Device. If no device available, return failure
/// @return Status run result
/// @author
///
Status ModelManager::SetDevice(int32_t deviceId) const {
  GE_CHK_RT_RET(rtSetDevice(deviceId));

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief load model online
/// @return Status run result
///
Status ModelManager::LoadModelOnline(uint32_t &model_id, shared_ptr<ge::Model> &model,
                                     std::shared_ptr<ModelListener> listener) {
  GE_CHK_BOOL_RET_STATUS(listener.get() != nullptr, PARAM_INVALID, "Param incorrect, listener is null");
  GenModelId(&model_id);

  GE_CHK_STATUS_RET(SetDevice(static_cast<int32_t>(GetContext().DeviceId())), "Set device failed, model id:%u.",
                    model_id);

  std::shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(0, listener);
  if (davinci_model == nullptr) {
    GELOGE(FAILED, "davinci_model is nullptr");
    return FAILED;
  }

  davinci_model->SetId(model_id);
  davinci_model->SetDeviceId(GetContext().DeviceId());

  Status ret = SUCCESS;
  do {
    GeModelPtr ge_model;
    GE_IF_BOOL_EXEC(
      ModelHelper::TransModelToGeModel(model, ge_model) != SUCCESS, GELOGW("trans model to ge_model failed."); break;);
    GE_TIMESTAMP_START(Assign);
    GE_IF_BOOL_EXEC(SUCCESS != (ret = davinci_model->Assign(ge_model)), GELOGW("assign model to modeldef failed.");
                    break;);
    GE_TIMESTAMP_END(Assign, "GraphLoader::ModelAssign");

    GE_TIMESTAMP_START(Init);
    GE_IF_BOOL_EXEC(SUCCESS != (ret = davinci_model->Init()), GELOGW("DavinciInit failed."); break;);
    GE_TIMESTAMP_END(Init, "GraphLoader::ModelInit");

    InsertModel(model_id, davinci_model);

    GELOGI("Parse model %u success.", model_id);
  } while (0);

  GE_CHK_RT(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));

  return ret;
}

void ModelManager::InsertModel(uint32_t id, std::shared_ptr<DavinciModel> &davinci_model) {
  GE_CHK_BOOL_EXEC(davinci_model != nullptr, return, "davinci_model ptr is null, id: %u", id);
  std::lock_guard<std::mutex> lock(map_mutex_);
  model_map_[id] = davinci_model;
}

Status ModelManager::DeleteModel(uint32_t id) {
  std::lock_guard<std::mutex> lock(map_mutex_);

  auto it = model_map_.find(id);
  if (it == model_map_.end()) {
    GELOGE(PARAM_INVALID, "model id %u does not exists.", id);
    return PARAM_INVALID;
  }

  (void)model_map_.erase(it);
  free_model_id_.push_back(id);
  return SUCCESS;
}

std::shared_ptr<DavinciModel> ModelManager::GetModel(uint32_t id) {
  std::lock_guard<std::mutex> lock(map_mutex_);

  auto it = model_map_.find(id);
  return (it == model_map_.end()) ? nullptr : it->second;
}

Status ModelManager::Unload(uint32_t model_id) {
  GE_CHK_STATUS_RET(DeleteModel(model_id), "failed to unload model id: %u", model_id);
  if (device_count > 0) {
    device_count--;
    GELOGI("Unload model %u success.", model_id);
  } else {
    GELOGI("Unload model %u success.no need reset device,device_count: %u", model_id, device_count);
  }

  return SUCCESS;
}

Status ModelManager::UnloadModeldef(uint32_t model_id) {
  GE_CHK_STATUS_RET(DeleteModel(model_id), "failed to unload modeldef id: %u", model_id);
  return SUCCESS;
}

Status ModelManager::DataInput(const InputData &input_data, OutputData &output_data) {
  GELOGI("calling the DataInput");

  SysMode mode = DavinciModel::GetSysMode();
  if ((mode == RESET) || (mode == STOP)) {
    GELOGE(domi::MODEL_NOT_READY, "System mode is reset or stop");
    return domi::MODEL_NOT_READY;
  }

  shared_ptr<InputDataWrapper> data_wrap(new (std::nothrow) InputDataWrapper());
  GE_CHECK_NOTNULL(data_wrap);

  Status status = data_wrap->Init(input_data, output_data);
  if (status != SUCCESS) {
    GELOGE(domi::PUSH_DATA_FAILED, "Init InputDataWrapper failed, input data index: %u.", input_data.index);
    return domi::PUSH_DATA_FAILED;
  }

  uint32_t model_id = input_data.model_id;
  output_data.model_id = model_id;

  std::shared_ptr<DavinciModel> model = GetModel(model_id);

  GE_CHK_BOOL_RET_STATUS(model != nullptr, PARAM_INVALID, "Invalid Model ID %u in InputData! ", model_id);

  GE_IF_BOOL_EXEC(model->GetDataInputTid() == 0, model->SetDataInputTid(mmGetTid()));

  DataInputer *inputer = model->GetDataInputer();
  GE_CHECK_NOTNULL(inputer);
  if (inputer->Push(data_wrap) != SUCCESS) {
    GELOGE(domi::DATA_QUEUE_ISFULL, "Data queue is full, please call again later, model_id %u ", model_id);
    return domi::DATA_QUEUE_ISFULL;
  }
  GELOGD("Data input success, model id:%u", model_id);

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief load Input and output TensorInfor for Model
/// @return Status run result
///
Status ModelManager::DataInputTensor(uint32_t model_id, const std::vector<TensorInfo> &inputs,
                                     std::vector<TensorInfo> &outputs) {
  SysMode mode = DavinciModel::GetSysMode();
  if ((mode == RESET) || (mode == STOP)) {
    GELOGE(domi::MODEL_NOT_READY, "System mode is reset or stop");
    return domi::MODEL_NOT_READY;
  }
  std::shared_ptr<DavinciModel> model = GetModel(model_id);
  GE_CHECK_NOTNULL(model);

  InputData input_data;
  input_data.model_id = model_id;
  input_data.timeout = 0;
  input_data.timestamp = 0;
  input_data.index = 0;

  std::size_t index = 0;
  for (const auto &item : model->GetOpList()) {
    auto op = item.second;
    GE_CHECK_NOTNULL(op);
    if (op->GetType() == DATA) {
      GE_CHECK_GE(inputs.size(), 1);
      GE_CHECK_GE(inputs.size() - 1, index);

      DataBuffer data;
      data.data = inputs[index].data.data;
      data.length = inputs[index].data.length;
      input_data.blobs.push_back(data);
      index++;
    }
  }

  CHECK_FALSE_EXEC(input_data.blobs.size() >= inputs.size(),
                   GELOGW("cur_inputs size = %zu, inputs size = %zu.", input_data.blobs.size(), inputs.size()););

  OutputData output_data;
  output_data.model_id = model_id;
  output_data.index = 0;
  for (size_t i = 0; i < outputs.size(); i++) {
    DataBuffer data;
    data.data = outputs[i].data.data;
    data.length = outputs[i].data.length;
    output_data.blobs.push_back(data);
  }

  shared_ptr<InputDataWrapper> data_wrap(new (std::nothrow) InputDataWrapper());
  GE_CHECK_NOTNULL(data_wrap);

  GE_CHK_STATUS_EXEC(data_wrap->Init(input_data, output_data), return domi::PUSH_DATA_FAILED,
                     "Init InputDataWrapper failed,input data model_id is : %u.", model_id);

  GE_CHK_BOOL_RET_STATUS(model != nullptr, PARAM_INVALID, "Invalid Model ID %u in InputData! ", model_id);

  DataInputer *inputer = model->GetDataInputer();
  GE_CHECK_NOTNULL(inputer);

  GE_CHK_STATUS_EXEC(inputer->Push(data_wrap), return domi::DATA_QUEUE_ISFULL,
                     "Data queue is full, please call again later, model_id %u ", model_id);

  GELOGD("Data input success, model id:%u", model_id);

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief create model thread, start to execute model
/// @param [in] model_id Model ID to be started
/// @return Status model run result
/// @author
///
Status ModelManager::Start(uint32_t model_id) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);

  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID, "Invalid Model ID %u to start! ", model_id);

  Status status = davinci_model->ModelRunStart();
  if (status == SUCCESS) {
    GELOGI("Start model %u success.", model_id);
  }

  return status;
}

///
/// @ingroup domi_ome
/// @brief Model ID stop
/// @only when unloaded
/// @param [in] model_id Model ID to be stopped
/// @return Status model stop result
/// @author
///
Status ModelManager::Stop(uint32_t model_id) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID, "Invalid Model ID %u to stop!", model_id);

  Status status = davinci_model->ModelRunStop();
  if (status == SUCCESS) {
    GELOGI("Stop model %u success.", model_id);
  }

  return status;
}

///
/// @ingroup domi_ome
/// @brief Command handle
/// @iterator 1 only Ieference, Debug 2 modes
/// @param [in] command command to handle
/// @return Status command handle result
/// @author
///
Status ModelManager::HandleCommand(const Command &command) {
  static const std::map<std::string, std::function<uint32_t(const Command &)>> cmds = {
    {"profile", HandleProfileCommand}, {"dump", HandleDumpCommand}, {"profiling", HandleAclProfilingCommand}};

  auto iter = cmds.find(command.cmd_type);
  if (iter == cmds.end()) {
    GELOGE(PARAM_INVALID, "Unsupported command: %s", command.cmd_type.c_str());
    return PARAM_INVALID;
  } else {
    return iter->second(command);
  }
}

Status ModelManager::HandleAclProfilingCommand(const Command &command) {
  if (command.cmd_params.size() < kCmdParSize) {
    GELOGE(PARAM_INVALID, "When the cmd_type is 'profiling', the size of cmd_params must larger than 2.");
    return PARAM_INVALID;
  }

  std::string map_key = command.cmd_params[0];
  std::string value = command.cmd_params[1];
  if (map_key == PROFILE_CONFIG) {
    ProfilingManager::Instance().SetProfilingConfig(value);
  }

  return SUCCESS;
}

Status ModelManager::HandleProfileCommand(const Command &command) {
  if (command.cmd_params.size() < kCmdParSize) {
    GELOGE(PARAM_INVALID, "When the cmd_type is 'profile', the size of cmd_params must larger than 2.");
    return PARAM_INVALID;
  }

  std::string map_key = command.cmd_params[0];
  std::string value = command.cmd_params[1];

  GELOGI("Profiling mode, Command key:%s , value:%s ", map_key.c_str(), value.c_str());

  auto iter = PROFILE_COMPONENT_MAP.find(map_key);
  if (iter != PROFILE_COMPONENT_MAP.end()) {
    std::string property_value = (value == "on") ? "1" : "0";
    PropertiesManager::Instance().SetPropertyValue(iter->second, property_value);
  }

  if ((map_key == PROFILER_JOBCTX || map_key == PROFILER_TARGET_PATH || map_key == RTS_PROFILE_PATH)) {
    PropertiesManager::Instance().SetPropertyValue(map_key, value);
  }

  if ((map_key == PROFILE_STOP_KEY) && (value == PROFILE_STOP_VALUE)) {
    rtError_t rt_ret = rtProfilerStop();
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(PARAM_INVALID, "Call rtProfilerStop ret:%d", rt_ret);
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status ModelManager::HandleDumpCommand(const Command &command) {
  if (command.cmd_params.size() % kDumpCmdPairSize != 0) {
    GELOGE(PARAM_INVALID, "When the cmd_type is 'dump', the size of cmd_params must be a even number.");
    return PARAM_INVALID;
  }

  std::string dump_status("off");
  std::string dump_model(DUMP_ALL_MODEL);
  std::string dump_path("/");
  std::set<std::string> dump_layers;
  std::string dump_layer_count;

  auto iter_dump_status = std::find(command.cmd_params.begin(), command.cmd_params.end(), DUMP_STATUS);
  if (iter_dump_status != command.cmd_params.end()) {
    ++iter_dump_status;
    if (iter_dump_status == command.cmd_params.end()) {
      GELOGE(PARAM_INVALID, "Invalid access.");
      return PARAM_INVALID;
    }

    dump_status = *iter_dump_status;
    GELOGI("dump status = %s.", dump_status.c_str());
  }

  auto iter_dump_model = std::find(command.cmd_params.begin(), command.cmd_params.end(), DUMP_MODEL);
  if (iter_dump_model != command.cmd_params.end()) {
    ++iter_dump_model;
    if (iter_dump_model == command.cmd_params.end()) {
      GELOGE(PARAM_INVALID, "Invalid access.");
      return PARAM_INVALID;
    }

    dump_model = *iter_dump_model;
    GELOGI("dump model = %s.", dump_model.c_str());
  }

  if (dump_status == "off" || dump_status == "OFF") {
    PropertiesManager::Instance().DeleteDumpPropertyValue(dump_model);
    return SUCCESS;
  }

  for (size_t i = 0; i < command.cmd_params.size() / kDumpCmdPairSize; ++i) {
    if (command.cmd_params.at(i * kDumpCmdPairSize).find(DUMP_LAYER) != std::string::npos) {
      GELOGI("dump layer: %s.", command.cmd_params.at(i * kDumpCmdPairSize + 1).c_str());
      dump_layers.insert(command.cmd_params.at(i * kDumpCmdPairSize + 1));
    }
  }

  auto iter_dump_path = std::find(command.cmd_params.begin(), command.cmd_params.end(), DUMP_FILE_PATH);
  if (iter_dump_path != command.cmd_params.end()) {
    ++iter_dump_path;
    if (iter_dump_path == command.cmd_params.end()) {
      GELOGE(PARAM_INVALID, "Invalid access.");
      return PARAM_INVALID;
    }

    dump_path = *iter_dump_path;

    if (!dump_path.empty() && dump_path[dump_path.size() - 1] != '/') {
      dump_path += "/";
    }
    GELOGI("dump path = %s.", dump_path.c_str());
  }

  PropertiesManager::Instance().AddDumpPropertyValue(dump_model, dump_layers);
  PropertiesManager::Instance().SetDumpOutputPath(dump_path);
  return SUCCESS;
}

Status ModelManager::GetMaxUsedMemory(const uint32_t model_id, uint64_t &max_size) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID, "GetMaxUsedMemory Failed, Invalid Model ID %u !",
                         model_id);

  max_size = davinci_model->TotalMemSize();
  return SUCCESS;
}

Status ModelManager::GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "GetInputOutputDescInfo Failed, Invalid Model ID %u !", model_id);

  return davinci_model->GetInputOutputDescInfo(input_desc, output_desc);
}

Status ModelManager::GetInputOutputDescInfoForZeroCopy(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                                       vector<InputOutputDescInfo> &output_desc) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "GetInputOutputDescInfo Failed, Invalid Model ID %u !", model_id);

  return davinci_model->GetInputOutputDescInfoForZeroCopy(input_desc, output_desc);
}

Status ModelManager::GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc,
                                            std::vector<uint32_t> &inputFormats, std::vector<uint32_t> &outputFormats) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "GetInputOutputDescInfo Failed, Invalid Model ID %u !", model_id);

  return davinci_model->GetInputOutputDescInfo(input_desc, output_desc, inputFormats, outputFormats);
}

Status ModelManager::GetInputOutputDescInfoForZeroCopy(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                                       vector<InputOutputDescInfo> &output_desc,
                                                       std::vector<uint32_t> &inputFormats,
                                                       std::vector<uint32_t> &outputFormats) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "GetInputOutputDescInfo Failed, Invalid Model ID %u !", model_id);

  return davinci_model->GetInputOutputDescInfoForZeroCopy(input_desc, output_desc, inputFormats, outputFormats);
}

Status ModelManager::LoadModelOffline(uint32_t &model_id, const ModelData &model, shared_ptr<ModelListener> listener,
                                      void *dev_ptr, size_t mem_size, void *weight_ptr, size_t weight_size) {
  GE_CHK_BOOL_RET_STATUS(model.key.empty() || access(model.key.c_str(), F_OK) == 0, PARAM_INVALID,
                         "input key file path is not valid!");
  GenModelId(&model_id);

  shared_ptr<DavinciModel> davinci_model = nullptr;

  ModelHelper model_helper;
  Status ret = model_helper.LoadModel(model);
  if (ret != SUCCESS) {
    GELOGE(ret, "load model failed.");
    return ret;
  }

  do {
    GeModelPtr ge_model = model_helper.GetGeModel();
    try {
      davinci_model = std::make_shared<DavinciModel>(model.priority, listener);
    } catch (std::bad_alloc &) {
      GELOGE(FAILED, "Make shared failed");
      return FAILED;
    } catch (...) {
      GELOGE(FAILED, "Make shared failed since other exception raise");
      return FAILED;
    }
    ret = davinci_model->Assign(ge_model);
    if (ret != SUCCESS) {
      GELOGW("assign model failed.");
      break;
    }
    davinci_model->SetId(model_id);
    ret = davinci_model->Init(dev_ptr, mem_size, weight_ptr, weight_size);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, break, "DavinciInit failed.");

    InsertModel(model_id, davinci_model);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(davinci_model == nullptr, ret = PARAM_INVALID; break, "Insert model failed");

    GELOGI("Parse model %u success.", model_id);

    GE_IF_BOOL_EXEC(ret == SUCCESS, device_count++);
    return SUCCESS;
  } while (0);

  return ret;
}

///
/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [out] model_id: model id for manager.
/// @param [in] model_data: Model data load from offline model file.
/// @param [in] input_que_ids: input queue ids from user, num equals Data Op.
/// @param [in] output_que_ids: input queue ids from user, num equals NetOutput Op.
/// @return: 0 for success / others for fail
///
Status ModelManager::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                                    const std::vector<uint32_t> &input_queue_ids,
                                    const std::vector<uint32_t> &output_queue_ids) {
  GE_CHK_BOOL_RET_STATUS(model_data.key.empty() || access(model_data.key.c_str(), F_OK) == 0, PARAM_INVALID,
                         "input key file path is not valid!");

  ModelHelper model_helper;
  Status ret = model_helper.LoadModel(model_data);
  if (ret != SUCCESS) {
    GELOGE(ret, "load model failed.");
    return ret;
  }

  shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(model_data.priority, nullptr);
  if (davinci_model == nullptr) {
    GELOGE(FAILED, "create model failed.");
    return FAILED;
  }

  ret = davinci_model->Assign(model_helper.GetGeModel());
  if (ret != SUCCESS) {
    GELOGE(ret, "assign model failed.");
    return ret;
  }

  GenModelId(&model_id);
  davinci_model->SetId(model_id);
  davinci_model->SetSessionId(model_id);
  ret = davinci_model->SetQueIds(input_queue_ids, output_queue_ids);
  if (ret != SUCCESS) {
    GELOGE(ret, "set model queue ids failed.");
    return ret;
  }

  ret = davinci_model->Init();
  if (ret != SUCCESS) {
    GELOGE(ret, "init model failed.");
    return ret;
  }

  InsertModel(model_id, davinci_model);
  GELOGI("Parse model %u success.", model_id);

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief  ACL case, not start new thread, return result
/// @param [in] model_id  mode id
/// @param [in] stream   model stream
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  input data
/// @param [out] output_data  output data
///
Status ModelManager::ExecuteModel(uint32_t model_id, rtStream_t stream, bool async_mode, const InputData &input_data,
                                  OutputData &output_data) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID, "Invalid Model ID %u to start! ", model_id);

  Status status = davinci_model->NnExecute(stream, async_mode, input_data, output_data);
  if (status == SUCCESS) {
    GELOGI("Execute model %u success.", model_id);
  }

  return status;
}

Status ModelManager::CreateAicpuSession(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(sess_ids_mutex_);
  auto it = sess_ids_.find(session_id);
  // never been created by any model
  if (it == sess_ids_.end()) {
    Status ret = KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_SESSION_CREATE, session_id, 0);
    if (ret == SUCCESS) {
      (void)sess_ids_.insert(session_id);
      GELOGI("The session: %lu create success.", session_id);
    }
    return ret;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief get model memory size and weight
/// @param [in] const ModelData model: model type
/// @param [out] size_t memSize: model memory usage
///           size_t weightSize: model weight and memory size
/// @return SUCCESS success / others failure
///
Status ModelManager::GetModelMemAndWeightSize(const ModelData &model, size_t &mem_size, size_t &weight_size) {
  uint8_t *model_data = nullptr;
  uint32_t model_len = 0;
  Status ret = DavinciModelParser::ParseModelContent(model, model_data, model_len);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ret, "parse model content failed!");

  OmFileLoadHelper om_file_helper;
  ret = om_file_helper.Init(model_data, model_len);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ret, "om file helperInit failed!");

  ModelPartition task_partition;
  if (om_file_helper.GetModelPartition(ModelPartitionType::TASK_INFO, task_partition) != SUCCESS) {
    GELOGE(FAILED, "get task model partition failed.");
    return FAILED;
  }

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  if (model_task_def == nullptr) {
    return FAILED;
  }
  if (task_partition.size != 0) {
    if (!ReadProtoFromArray(task_partition.data, static_cast<int>(task_partition.size), model_task_def.get())) {
      GELOGE(FAILED, "ReadProtoFromArray failed.");
      return FAILED;
    }
  }

  ModelPartition partition_weight;
  ret = om_file_helper.GetModelPartition(ModelPartitionType::WEIGHTS_DATA, partition_weight);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ret, "Get weight partition failed. ret = %u", ret);

  mem_size = model_task_def->memory_size();
  weight_size = partition_weight.size;
  return SUCCESS;
}

void ModelManager::GenModelId(uint32_t *id) {
  if (id == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(map_mutex_);
  if (free_model_id_.empty()) {
    *id = ++max_model_id_;
  } else {
    *id = free_model_id_.back();
    free_model_id_.pop_back();
  }
}
}  // namespace ge
