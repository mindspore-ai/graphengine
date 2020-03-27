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

#include "executor/ge_executor.h"

#include <ctime>

#include <iostream>

#include "cce/cce.h"
#include "cce/compiler_stub.h"
#include "cce/aicpu_engine.h"
#include "cce/fwk_adpt_struct.h"
#include "common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"
#include "common/helper/model_helper.h"
#include "common/util.h"
#include "graph/execute/graph_execute.h"
#include "graph/load/graph_loader.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/model.h"
#include "graph/utils/graph_utils.h"
#include "mmpa/mmpa_api.h"
#include "single_op/single_op_manager.h"
#include "framework/common/util.h"
#include "common/profiling/profiling_manager.h"

namespace {
const uint64_t kDynamicImageSizeParamNum = 2;
}  // namespace

namespace ge {
bool GeExecutor::is_init_ = false;

class ModelListenerAdapter : public ModelListener {
 public:
  domi::Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result_code) {
    if (listener == nullptr) {
      GELOGE(ge::FAILED, "listener is null.");
      return FAILED;
    }
    return listener->OnComputeDone(model_id, data_index, result_code);
  }

  std::shared_ptr<ge::ModelListener> listener;
};

ge::Status TransferDomiErrorCode(const uint32_t error_code) {
  switch (error_code) {
    case ge::PARAM_INVALID:
    case domi::PARAM_INVALID:
      return ge::PARAM_INVALID;
    case ge::INTERNAL_ERROR:
    case domi::INTERNAL_ERROR:
      return ge::INTERNAL_ERROR;
    default:
      return ge::FAILED;
  }
}

void GetGeTensorDescFromDomiInfo(std::vector<ge::TensorDesc> &ge_descs,
                                 const std::vector<InputOutputDescInfo> &domi_descs,
                                 const std::vector<uint32_t> &formats) {
  uint32_t idx = 0;
  for (auto desc_item : domi_descs) {
    ge::TensorDesc ge_desc;
    ge_desc.SetName(desc_item.name);
    ge_desc.SetDataType(static_cast<DataType>(desc_item.data_type));
    ge_desc.SetFormat(static_cast<ge::Format>(formats[idx]));
    std::vector<int64_t> shape_dims;
    for (auto dim : desc_item.shape_info.dims) {
      shape_dims.push_back(dim);
    }
    Shape ge_shape(shape_dims);
    ge_desc.SetShape(ge_shape);
    ge_desc.SetSize(desc_item.size);
    ge_descs.emplace_back(ge_desc);
    ++idx;
  }
}

void GetDomiInputData(const ge::RunModelData &input_data, InputData &inputs) {
  inputs.index = input_data.index;
  inputs.model_id = input_data.model_id;
  inputs.timestamp = input_data.timestamp;
  inputs.timeout = input_data.timeout;
  inputs.request_id = input_data.request_id;
  for (const auto &data_item : input_data.blobs) {
    DataBuffer data_buf{data_item.data, data_item.length, data_item.isDataSupportMemShare};
    inputs.blobs.emplace_back(data_buf);
  }
}

void GetDomiOutputData(const ge::RunModelData &output_data, OutputData &outputs) {
  outputs.index = output_data.index;
  outputs.model_id = output_data.model_id;
  for (const auto &data_item : output_data.blobs) {
    DataBuffer data_buf(data_item.data, data_item.length, data_item.isDataSupportMemShare);
    outputs.blobs.emplace_back(data_buf);
  }
}

GeExecutor::GeExecutor() {}

Status GeExecutor::Initialize() {
  GELOGI("Init ge_executor begin.");
  if (is_init_) {
    GELOGW("Already inited, don't need to init again.");
    return ge::SUCCESS;
  }

  std::vector<rtMemType_t> mem_type(1, RT_MEMORY_HBM);
  auto ret = MemManager::Instance().Initialize(mem_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "Memory Manager init fail.");
    return ret;
  }

  // Start profiling
  int32_t device_id = 0;
  rtError_t rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "runtime get device_id failed, current device_id:%d", device_id);
    return FAILED;
  }
  GELOGI("current device_id:%d", device_id);
  Options profiling_options;
  profiling_options.device_id = device_id;
  profiling_options.job_id = 0;
  ProfilingManager::Instance().Init(profiling_options);
  if (ProfilingManager::Instance().Init(profiling_options) != SUCCESS) {
    GELOGE(FAILED, "Failed to init profiling.");
    return FAILED;
  }

  is_init_ = true;
  GELOGI("Init ge_executor over.");
  return ge::SUCCESS;
}

// Load model
Status GeExecutor::LoadModelOffline(uint32_t &model_id, const std::string &path, const std::string &key,
                                    int32_t priority, std::shared_ptr<ge::ModelListener> listener) {
  GELOGI("load model offline begin.");
  if (!is_init_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  string file_path = RealPath(path.c_str());
  if (file_path.empty()) {
    GELOGE(ge::FAILED, "fileath is invalid. please check your text file '%s'.", path.c_str());
    return ge::FAILED;
  }

  std::shared_ptr<ModelListenerAdapter> listener_adapter = MakeShared<ModelListenerAdapter>();
  if (listener_adapter == nullptr) {
    GELOGE(MEMALLOC_FAILED, "ModelListenerAdapter make shared failed!");
    return ge::FAILED;
  }
  listener_adapter->listener = listener;

  Status ret = GraphLoader::LoadModelFromFile(path, key, priority, listener_adapter, model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[GeExecutor] LoadModelFromFile failed");
    return TransferDomiErrorCode(ret);
  }
  return SUCCESS;
}

Status GeExecutor::LoadModel(uint32_t &model_id, const ModelData &model_data,
                             std::shared_ptr<ge::ModelListener> listener) {
  GELOGI("Load model begin, model_id:%u.", model_id);
  if (!is_init_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  Status ret;
  std::shared_ptr<ModelListenerAdapter> listener_adapter = MakeShared<ModelListenerAdapter>();
  if (listener_adapter == nullptr) {
    GELOGE(MEMALLOC_FAILED, "ModelListenerAdapter make shared failed!");
    return ge::FAILED;
  }
  listener_adapter->listener = listener;

  ret = GraphLoader::LoadModel(model_data, listener_adapter, model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[GeExecutor] LoadModel failed.");
    return TransferDomiErrorCode(ret);
  }
  return ret;
}

Status GeExecutor::UnloadModel(uint32_t model_id) {
  GELOGI("unload model %u begin.", model_id);
  if (!is_init_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  // Stop profiling
  if (!ProfilingManager::Instance().ProfilingOpTraceOn() && ProfilingManager::Instance().ProfilingOn()) {
    ProfilingManager::Instance().StopProfiling();
  }
  return GraphLoader::UnloadModel(model_id);
}

Status GeExecutor::RunModel(const ge::RunModelData &input_data, ge::RunModelData &output_data) {
  GELOGI("run model begin.");
  if (!is_init_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  InputData inputs;
  GetDomiInputData(input_data, inputs);
  OutputData outputs;
  GetDomiOutputData(output_data, outputs);

  return GraphExecutor::DataInput(inputs, outputs);
}

// Get input and output descriptor
Status GeExecutor::GetModelDescInfo(uint32_t model_id, std::vector<ge::TensorDesc> &input_desc,
                                    std::vector<ge::TensorDesc> &output_desc) {
  GELOGI("get model desc info begin.");
  if (!is_init_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  std::vector<InputOutputDescInfo> input_desc_infos;
  std::vector<InputOutputDescInfo> output_desc_infos;
  std::vector<uint32_t> input_formats;
  std::vector<uint32_t> output_formats;
  GELOGI("GetInputOutputDescInfo via new ome.");

  Status ret = GraphExecutor::GetInputOutputDescInfo(model_id, input_desc_infos, output_desc_infos,
                                                     input_formats, output_formats);
  if (ret != domi::SUCCESS) {
    GELOGE(ret, "GetInputOutputDescInfo  failed. ret = %u", ret);
    return TransferDomiErrorCode(ret);
  }

  if (input_formats.size() != input_desc_infos.size()) {
    GELOGE(ge::FAILED, "input_formats.size() != input_desc_infos.size().");
    return ge::FAILED;
  }

  if (output_formats.size() != output_desc_infos.size()) {
    GELOGE(ge::FAILED, "output_formats.size() != output_desc_infos.size().");
    return ge::FAILED;
  }

  // Transfer data to TensorDesc
  GetGeTensorDescFromDomiInfo(input_desc, input_desc_infos, input_formats);
  GetGeTensorDescFromDomiInfo(output_desc, output_desc_infos, output_formats);

  GELOGI("get model desc info end.");
  return ge::SUCCESS;
}

Status GeExecutor::GetModelDescInfoForZeroCopy(uint32_t model_id, std::vector<ge::TensorDesc> &input_desc,
                                               std::vector<TensorDesc> &output_desc) {
  GELOGI("get model desc info for zero copy begin.");
  if (!is_init_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  std::vector<InputOutputDescInfo> input_desc_infos;
  std::vector<InputOutputDescInfo> output_desc_infos;
  std::vector<uint32_t> input_formats;
  std::vector<uint32_t> output_formats;
  GELOGI("GetInputOutputDescInfoForZeroCopy via new ome.");

  Status ret = GraphExecutor::GetInputOutputDescInfoForZeroCopy(model_id, input_desc_infos, output_desc_infos,
                                                                input_formats, output_formats);
  if (ret != domi::SUCCESS) {
    GELOGE(ret, "Get DescInfo For ZeroCopy failed. ret = %u", ret);
    return TransferDomiErrorCode(ret);
  }

  if (input_formats.size() != input_desc_infos.size()) {
    GELOGE(ge::FAILED, "input_formats.size() != input_desc_infos.size().");
    return ge::FAILED;
  }

  if (output_formats.size() != output_desc_infos.size()) {
    GELOGE(ge::FAILED, "output_formats.size() != output_desc_infos.size().");
    return ge::FAILED;
  }

  GetGeTensorDescFromDomiInfo(input_desc, input_desc_infos, input_formats);
  GetGeTensorDescFromDomiInfo(output_desc, output_desc_infos, output_formats);

  GELOGI("get model desc info for zero copy end.");
  return ge::SUCCESS;
}

Status GeExecutor::CommandHandle(const Command &command) {
  GELOGI("command handle begin.");
  Status ret = GraphLoader::CommandHandle(command);
  if (ret != SUCCESS) {
    GELOGE(ret, "CommandHandle: Command Handle failed.");
    return TransferDomiErrorCode(ret);
  }
  return SUCCESS;
}

Status GeExecutor::GetMaxUsedMemory(uint32_t model_id, uint32_t &max_size) {
  uint64_t max_mem_size = 0;
  Status ret = GraphLoader::GetMaxUsedMemory(model_id, max_mem_size);
  max_size = static_cast<uint32_t>(max_mem_size);
  return ret;
}

///
/// @ingroup ge
/// @brief Load data from model file to memory
/// @param [in] const std::string &path: Offline model file path
/// @param [out] domi::ModelData &model_data: Offline model memory data
/// @return SUCCESS handle successfully / others handle failed
///
Status GeExecutor::LoadDataFromFile(const std::string &path, ModelData &model_data) {
  string file_path = RealPath(path.c_str());
  if (file_path.empty()) {
    GELOGE(ge::FAILED, "file_path is invalid. please check your text file '%s'.", path.c_str());
    return ge::FAILED;
  }
  GELOGI("load model_data from file: %s.", path.c_str());
  std::string key_path;
  int32_t priority = 0;
  Status ret = GraphLoader::LoadDataFromFile(path, key_path, priority, model_data);
  if (ret != SUCCESS) {
    if (model_data.model_data != nullptr) {
      delete[] static_cast<char *>(model_data.model_data);
      model_data.model_data = nullptr;
    }
  }

  return ret;
}

///
/// @ingroup ge
/// @brief Load model from offline model memory data
/// @param [in] domi::ModelData &model_data: Offline model data
///            void *dev_ptr: Input/Output memory start address
///            size_t memsize: Input/Output memory length
///            void *weight_ptr: Weight memory start address
///            size_t weightsize: Weight memory length
/// @param [out] uint32_t &model_id: identification after model loading
/// @return SUCCESS handle successfully / others handle failed
///
Status GeExecutor::LoadModelFromData(uint32_t &model_id, const ModelData &model_data, void *dev_ptr,
                                     size_t mem_size, void *weight_ptr, size_t weight_size) {
  return GraphLoader::LoadModelFromData(model_id, model_data, dev_ptr, mem_size, weight_ptr, weight_size);
}

///
/// @ingroup ge
/// @brief Load task list from ModelData with queue.
/// @param [out] model_id: model id allocate from manager.
/// @param [in] ge_model_data: Model data load from offline model.
/// @param [in] input_queue_ids: input queue ids create from user.
/// @param [in] output_queue_ids: input queue ids create from user.
/// @return: 0 for success / others for fail
///
Status GeExecutor::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                                  const std::vector<uint32_t> &input_queue_ids,
                                  const std::vector<uint32_t> &output_queue_ids) {
  return GraphLoader::LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids);
}

///
/// @ingroup ge
/// @brief Synchronous execution of offline model(Do not create thread)
/// @param [in] uint32_t modelId: Model ID to execute
///             void* stream: stream to execute
///             const domi::InputData *input_data: Model input data
///             bool async_mode: is asynchronize mode.
/// @param [out] domi::OutputData *output_data: Model output data
/// @return SUCCESS handle successfully / others handle failed
///
Status GeExecutor::ExecModel(uint32_t model_id, void *stream, const ge::RunModelData &input_data,
                             ge::RunModelData &output_data, bool async_mode) {
  if (!is_init_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  InputData input_data_tmp;
  OutputData output_data_tmp;
  GetDomiInputData(input_data, input_data_tmp);
  GetDomiOutputData(output_data, output_data_tmp);

  return GraphLoader::ExecuteModel(model_id, stream, async_mode, input_data_tmp, output_data_tmp);
}

///
/// @ingroup ge
/// @brief Get weight memory size from model file
/// @param [in] const std::string &path: Offline model file path
/// @param [out] size_t &mem_size Execution memory size
///              size_t &weight_size Weight memory space size
/// @return SUCCESS handle successfully / others handle failed
///
Status GeExecutor::GetMemAndWeightSize(const std::string &path, size_t &mem_size, size_t &weight_size) {
  ModelData model;
  std::string key;
  Status ret = ge::GraphLoader::LoadDataFromFile(path, key, 0, model);
  if ((ret != SUCCESS) || (model.model_data == nullptr)) {
    GELOGE(ret, "Load data from file failed. ret = %d", ret);
    return ret;
  }

  ret = ge::ModelManager::GetModelMemAndWeightSize(model, mem_size, weight_size);

  delete[] static_cast<char *>(model.model_data);
  model.model_data = nullptr;

  return ret;
}

///
/// @ingroup ge
/// @brief Get weight memory size from model file
/// @param [in] const void *model_data Offline model buffer
///             size_t model_size Offline model buffer length
/// @param [out] size_t &mem_size Execution memory size
///              size_t &weight_size Weight memory space size
/// @return SUCCESS handle successfully / others handle failed
///
Status GeExecutor::GetMemAndWeightSize(const void *model_data, size_t model_size, size_t &mem_size,
                                       size_t &weight_size) {
  if (model_data == nullptr) {
    GELOGE(PARAM_INVALID, "invalid model data!");
    return PARAM_INVALID;
  }

  ModelData model;
  model.model_data = const_cast<void *>(model_data);
  model.model_len = static_cast<uint32_t>(model_size);

  return ge::ModelManager::GetModelMemAndWeightSize(model, mem_size, weight_size);
}

Status GeExecutor::LoadSingleOp(const std::string &model_name,
                                const ge::ModelData &model_data,
                                void *stream,
                                SingleOp **single_op) {
  return SingleOpManager::GetInstance().GetOpFromModel(model_name, model_data, stream, single_op);
}

Status GeExecutor::ExecuteAsync(SingleOp *executor, const std::vector<DataBuffer> &inputs,
                                std::vector<DataBuffer> &outputs) {
  if (executor == nullptr) {
    GELOGE(PARAM_INVALID, "param is NULL");
    return PARAM_INVALID;
  }

  return executor->ExecuteAsync(inputs, outputs);
}

Status GeExecutor::ReleaseSingleOpResource(void *stream) {
  return SingleOpManager::GetInstance().ReleaseResource(stream);
}
}  // namespace ge
