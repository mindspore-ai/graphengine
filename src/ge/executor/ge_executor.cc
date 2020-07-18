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
#include <cce/cce.h>
#include <cce/compiler_stub.h>
#include <ctime>
#include <iostream>
#include "common/debug/log.h"
#include "common/ge/ge_util.h"
#include "common/helper/model_helper.h"
#include "common/profiling/profiling_manager.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/execute/graph_execute.h"
#include "graph/load/graph_loader.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/model.h"
#include "graph/utils/graph_utils.h"
#include "mmpa/mmpa_api.h"
#include "single_op/single_op_manager.h"

namespace {
const size_t kDynamicBatchSizeVecSize = 1;
const size_t kStaticBatchInfoSize = 1;
const size_t kDynamicImageSizeVecSize = 2;
const size_t kDynamicImageSizeInputSize = 2;
const char *const kBatchLabel = "Batch_";

ge::Status TransferDomiErrorCode(const uint32_t errorCode) {
  switch (errorCode) {
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
                                 const std::vector<ge::InputOutputDescInfo> &domi_descs,
                                 const std::vector<uint32_t> &formats) {
  uint32_t idx = 0;
  for (auto desc_item : domi_descs) {
    ge::TensorDesc ge_desc;
    ge_desc.SetName(desc_item.name);
    ge_desc.SetDataType(static_cast<ge::DataType>(desc_item.data_type));
    ge_desc.SetFormat(static_cast<ge::Format>(formats[idx]));
    std::vector<int64_t> shape_dims;
    for (auto dim : desc_item.shape_info.dims) {
      shape_dims.push_back(dim);
    }
    ge::Shape ge_shape(shape_dims);
    ge_desc.SetShape(ge_shape);
    ge_desc.SetSize(desc_item.size);
    ge_descs.emplace_back(ge_desc);
    ++idx;
  }
}

void GetDomiInputData(const ge::RunModelData &input_data, ge::InputData &inputs) {
  inputs.index = input_data.index;
  inputs.model_id = input_data.modelId;
  inputs.timestamp = input_data.timestamp;
  inputs.timeout = input_data.timeout;
  inputs.request_id = input_data.request_id;
  for (const auto &data_item : input_data.blobs) {
    ge::DataBuffer dataBuf{data_item.data, data_item.length, data_item.isDataSupportMemShare};
    inputs.blobs.emplace_back(dataBuf);
  }
}

void GetDomiOutputData(const ge::RunModelData &output_data, ge::OutputData &outputs) {
  outputs.index = output_data.index;
  outputs.model_id = output_data.modelId;
  for (const auto &data_item : output_data.blobs) {
    ge::DataBuffer dataBuf(data_item.data, data_item.length, data_item.isDataSupportMemShare);
    outputs.blobs.emplace_back(dataBuf);
  }
}

void SetDynamicInputDataFlag(const ge::RunModelData &input_data, const std::vector<std::vector<int64_t>> batch_info,
                             ge::InputData &inputs) {
  inputs.is_dynamic_batch = true;
  std::string batch_label;
  for (size_t i = 0; i < batch_info.size(); ++i) {
    if (batch_info[i].size() == kDynamicBatchSizeVecSize &&
        batch_info[i][0] == static_cast<int64_t>(input_data.dynamic_batch_size)) {
      batch_label = kBatchLabel + std::to_string(i);
      inputs.batch_label = batch_label;
      break;
    } else if (batch_info[i].size() == kDynamicImageSizeVecSize &&
               batch_info[i][0] == static_cast<int64_t>(input_data.dynamic_image_height) &&
               batch_info[i][1] == static_cast<int64_t>(input_data.dynamic_image_width)) {
      batch_label = kBatchLabel + std::to_string(i);
      inputs.batch_label = batch_label;
      break;
    }
  }
  GELOGI("current batch label:%s", batch_label.c_str());
}

bool IsDynamicBatchSizeMatchModel(uint64_t batch_size, const vector<std::vector<int64_t>> &batch_info) {
  if (batch_info.empty()) {
    GELOGE(ge::FAILED, "Dynamic batch info is empty.");
    return false;
  }

  for (auto batch : batch_info) {
    if (batch.size() != kDynamicBatchSizeVecSize) {
      GELOGE(ge::FAILED, "Dynamic batch param num is %zu, current batch size is %zu.", kDynamicBatchSizeVecSize,
             batch.size());
      return false;
    }
    if (batch[0] == static_cast<int64_t>(batch_size)) {
      return true;
    }
  }
  GELOGE(ge::FAILED, "Dynamic batch %lu can not match the gear of model.", batch_size);
  return false;
}

bool IsDynamicImageSizeMatchModel(uint64_t image_height, uint64_t image_width,
                                  const vector<std::vector<int64_t>> &batch_info) {
  if (batch_info.empty()) {
    GELOGE(ge::FAILED, "Dynamic batch info is empty.");
    return false;
  }

  for (auto resolution : batch_info) {
    if (resolution.size() != kDynamicImageSizeVecSize) {
      GELOGE(ge::FAILED, "Dynamic resolution param num is %zu, current resolution size is %zu.",
             kDynamicImageSizeVecSize, resolution.size());
      return false;
    }
    if (resolution[0] == static_cast<int64_t>(image_height) && resolution[1] == static_cast<int64_t>(image_width)) {
      return true;
    }
  }

  GELOGE(ge::FAILED, "Dynamic resolution (%lu,%lu) can not match the gear of model.", image_height, image_width);
  return false;
}
}  // namespace

namespace ge {
bool GeExecutor::isInit_ = false;
class ModelListenerAdapter : public ModelListener {
 public:
  domi::Status OnComputeDone(uint32_t model_id, uint32_t dataIndex, uint32_t resultCode,
                             std::vector<ge::OutputTensorInfo> &outputs) {
    if (listener == nullptr) {
      GELOGE(ge::FAILED, "listener is null.");
      return FAILED;
    }
    return listener->OnComputeDone(model_id, dataIndex, resultCode, outputs);
  }

  std::shared_ptr<ge::ModelListener> listener;
};

GeExecutor::GeExecutor() {}

Status GeExecutor::Initialize() {
  GELOGI("Init GeExecutor begin.");
  if (isInit_) {
    GELOGW("Already initialized, no need to be initialized again.");
    return ge::SUCCESS;
  }

  std::vector<rtMemType_t> mem_type(1, RT_MEMORY_HBM);
  auto ret = MemManager::Instance().Initialize(mem_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "Memory Manager init failed.");
    return ret;
  }

  // Start profiling
  Options profiling_options;
  profiling_options.device_id = 0;
  profiling_options.job_id = "";
  ProfilingManager::Instance().Init(profiling_options);

  isInit_ = true;
  GELOGI("Init GeExecutor over.");
  return ge::SUCCESS;
}

Status GeExecutor::Finalize() {
  GELOGI("Uninit GeExecutor begin.");
  if (isInit_ == false) {
    GELOGW("GeExecutor has not been initialized.");
    return ge::SUCCESS;
  }

  // Stop profiling
  if (ProfilingManager::Instance().ProfilingOn()) {
    ProfilingManager::Instance().StopProfiling();
    ProfilingManager::Instance().PluginUnInit(GE_PROFILING_MODULE);
  }

  GELOGI("Uninit GeExecutor over.");
  return ge::SUCCESS;
}

Status GeExecutor::SetDynamicBatchSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                       uint64_t batch_size) {
  if (dynamic_input_addr == nullptr) {
    GELOGE(FAILED, "Dynamic input addr is nullptr!");
    return FAILED;
  }

  uint64_t size = sizeof(uint64_t);
  if (length < size) {
    GELOGE(FAILED, "Dynamic input size [%lu] is less than [%lu]!", length, size);
    return FAILED;
  }

  // Verify whether the input dynamic batch matches the model gear
  std::vector<std::vector<int64_t>> batch_info;
  std::vector<uint64_t> batch_num{batch_size};
  Status ret = GraphExecutor::GetDynamicBatchInfo(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Get dynamic input info failed.");
    return FAILED;
  }

  if (!IsDynamicBatchSizeMatchModel(batch_size, batch_info)) {
    GELOGE(FAILED, "The current dynamic input does not match the gear of the model.");
    return FAILED;
  }

  ret = GraphExecutor::SetDynamicSize(model_id, batch_num);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Set dynamic size failed");
    return FAILED;
  }
  // memcpy dynamic_batch_size from host to device
  if (rtMemcpy(dynamic_input_addr, length, &batch_size, size, RT_MEMCPY_HOST_TO_DEVICE) != RT_ERROR_NONE) {
    GELOGE(FAILED, "memcpy dynamic batch input data failed!");
    return FAILED;
  }
  return SUCCESS;
}

Status GeExecutor::SetDynamicImageSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                       uint64_t image_height, uint64_t image_width) {
  if (dynamic_input_addr == nullptr) {
    GELOGE(FAILED, "Dynamic input addr is nullptr!");
    return FAILED;
  }

  uint64_t dynamic_input_size = kDynamicImageSizeInputSize * sizeof(uint64_t);
  if (length < dynamic_input_size) {
    GELOGE(FAILED, "Dynamic input size [%lu] is less than [%lu]!", length, dynamic_input_size);
    return FAILED;
  }

  // Verify whether the input dynamic resolution matches the model gear
  std::vector<std::vector<int64_t>> batch_info;
  std::vector<uint64_t> batch_num{image_height, image_width};
  Status ret = GraphExecutor::GetDynamicBatchInfo(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Get dynamic input info failed.");
    return FAILED;
  }

  if (!IsDynamicImageSizeMatchModel(image_height, image_width, batch_info)) {
    GELOGE(FAILED, "The current dynamic input does not match the gear of the model.");
    return FAILED;
  }

  ret = GraphExecutor::SetDynamicSize(model_id, batch_num);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Set dynamic size failed");
    return FAILED;
  }
  // Memcpy dynamic resolution height from host to device
  if (rtMemcpy(dynamic_input_addr, sizeof(uint64_t), &image_height, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE) !=
      RT_ERROR_NONE) {
    GELOGE(FAILED, "memcpy dynamic resolution input data failed!");
    return FAILED;
  }

  uint64_t remain_size = length - sizeof(uint64_t);
  // Memcpy dynamic resolution width from host to device
  if (rtMemcpy(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(dynamic_input_addr) + sizeof(uint64_t)),
               remain_size, &image_width, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE) != RT_ERROR_NONE) {
    GELOGE(FAILED, "memcpy dynamic resolution input data failed!");
    return FAILED;
  }
  return SUCCESS;
}

Status GeExecutor::GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info) {
  GELOGI("Begin to get current shape");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }
  Status ret = GraphExecutor::GetCurShape(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Get current shape failed");
    return FAILED;
  }
  return SUCCESS;
}

Status GeExecutor::SetDynamicAippData(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                      const std::vector<kAippDynamicBatchPara> &aippBatchPara,
                                      const kAippDynamicPara &aippParms) {
  GELOGI("Enter to SetDynamicAippData.");
  if (dynamic_input_addr == nullptr) {
    GELOGE(FAILED, "Dynamic aipp input addr is nullptr!");
    return FAILED;
  }
  if (aippBatchPara.empty()) {
    GELOGE(FAILED, "aippBatchPara is empty.");
    return FAILED;
  }
  uint64_t batch_num = aippBatchPara.size();
  uint64_t real_aippParms_size = sizeof(kAippDynamicPara) - sizeof(kAippDynamicBatchPara);
  uint64_t struct_len = batch_num * sizeof(kAippDynamicBatchPara) + real_aippParms_size;
  GELOGI(
    "Get acl input dynamic aipp data, model_id is %u, length is %lu,"
    "batch num is %lu, struct_len is %lu",
    model_id, length, batch_num, struct_len);
  if (struct_len > length) {
    GELOGE(FAILED, "input dynamic aipp param len [%lu] is larger than aipp_data size [%lu]", struct_len, length);
    return FAILED;
  }
  // Memcpy real kAippDynamicBatchPara from host to device
  if (rtMemcpy(dynamic_input_addr, length, &aippParms, real_aippParms_size, RT_MEMCPY_HOST_TO_DEVICE) !=
      RT_ERROR_NONE) {
    GELOGE(FAILED, "memcpy real_aippParms_size failed!");
    return FAILED;
  }
  uint64_t remain_len = length - real_aippParms_size;
  uint8_t *aipp_batch_para_dev = reinterpret_cast<uint8_t *>(dynamic_input_addr) + real_aippParms_size;

  for (uint64_t i = 0; i < batch_num; ++i) {
    if (rtMemcpy(reinterpret_cast<void *>(aipp_batch_para_dev + i * sizeof(kAippDynamicBatchPara)),
                 (remain_len - i * sizeof(kAippDynamicBatchPara)), &(aippBatchPara[i]), sizeof(kAippDynamicBatchPara),
                 RT_MEMCPY_HOST_TO_DEVICE) != RT_ERROR_NONE) {
      GELOGE(FAILED, "memcpy kAippDynamicBatchPara input data failed!");
      return FAILED;
    }
  }
  return SUCCESS;
}

// Load model
Status GeExecutor::LoadModelOffline(uint32_t &model_id, const std::string &path, const std::string &key,
                                    int32_t priority, std::shared_ptr<ge::ModelListener> listener) {
  GELOGI("load model offline begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

  string filePath = RealPath(path.c_str());
  if (filePath.empty()) {
    GELOGE(ge::FAILED, "File path is invalid. please check your text file '%s'.", path.c_str());
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
  GELOGI("Load model begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

  std::shared_ptr<ModelListenerAdapter> listener_adapter = MakeShared<ModelListenerAdapter>();
  if (listener_adapter == nullptr) {
    GELOGE(MEMALLOC_FAILED, "ModelListenerAdapter make shared failed!");
    return ge::FAILED;
  }
  listener_adapter->listener = listener;

  Status ret = GraphLoader::LoadModel(model_data, listener_adapter, model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[GeExecutor] LoadModel failed.");
    return TransferDomiErrorCode(ret);
  }
  return ret;
}

Status GeExecutor::UnloadModel(uint32_t model_id) {
  GELOGI("unload model %u begin.", model_id);
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }
  Status ret = GraphLoader::DestroyAicpuSessionForInfer(model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[GraphLoader] DestroyAicpuSessionForInfer failed.");
    return FAILED;
  }
  return GraphLoader::UnloadModel(model_id);
}

Status GeExecutor::RunModel(const ge::RunModelData &input_data, ge::RunModelData &output_data) {
  GELOGI("run model begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
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
                                    std::vector<ge::TensorDesc> &output_desc, bool new_model_desc) {
  GELOGI("get model desc info begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

  std::vector<InputOutputDescInfo> input_desc_infos;
  std::vector<InputOutputDescInfo> output_desc_infos;
  std::vector<uint32_t> input_formats;
  std::vector<uint32_t> output_formats;

  Status ret = GraphExecutor::GetInputOutputDescInfo(model_id, input_desc_infos, output_desc_infos, input_formats,
                                                     output_formats, new_model_desc);
  if (ret != domi::SUCCESS) {
    GELOGE(ret, "GetInputOutputDescInfo failed. ret = %u", ret);
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

///
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [in] model_id
/// @param [out] batch_info
/// @return execute result
///
Status GeExecutor::GetDynamicBatchInfo(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info) {
  GELOGI("Begin to get dynamic batch info.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

  Status ret = GraphExecutor::GetDynamicBatchInfo(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetDynamicBatchInfo failed.");
    return ret;
  }

  GELOGI("Get dynamic batch info succ.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get AIPP input format
/// @param [in] model_id
/// @param [in] index
/// @param [out] input_format
/// @return execute result
///
Status GeExecutor::GetAIPPInfo(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_info) {
  GELOGI("Begin to GetAIPPInfo.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }
  Status ret = GraphExecutor::GetAIPPInfo(model_id, index, aipp_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetAIPPInfo failed.");
    return ret;
  }
  GELOGI("GetAIPPInfo succ.");
  return SUCCESS;
}
Status GeExecutor::GetModelAttr(uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info) {
  GELOGI("Begin to get dynamic batch output shape info");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }
  Status ret = GraphExecutor::GetModelAttr(model_id, dynamic_output_shape_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "Get dynamic batch output shape info failed.");
    return ret;
  }

  GELOGI("Get dynamic batch output shape info succ.");
  return SUCCESS;
}

Status GeExecutor::GetModelDescInfoForZeroCopy(uint32_t model_id, std::vector<ge::TensorDesc> &input_desc,
                                               std::vector<TensorDesc> &output_desc) {
  GELOGI("get model desc info for zero copy begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

  std::vector<InputOutputDescInfo> input_desc_infos;
  std::vector<InputOutputDescInfo> output_desc_infos;
  std::vector<uint32_t> input_formats;
  std::vector<uint32_t> output_formats;

  Status ret = GraphExecutor::GetInputOutputDescInfoForZeroCopy(model_id, input_desc_infos, output_desc_infos,
                                                                input_formats, output_formats);
  if (ret != domi::SUCCESS) {
    GELOGE(ret, "Get DescInfo from zero copy failed. ret = %u", ret);
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

  GELOGI("get model desc info from zero copy end.");
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
  GELOGI("Get max used memory begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

  uint64_t max_mem_size = 0;
  Status ret = GraphLoader::GetMaxUsedMemory(model_id, max_mem_size);
  max_size = static_cast<uint32_t>(max_mem_size);
  return ret;
}

/**
 * @ingroup ge
 * @brief Load data from model file to memory
 * @param [in] const std::string &path: Offline model file path
 * @param [out] domi::ModelData &model_data: Offline model memory data
 * @return SUCCESS handle successfully / others handle failed
 */
Status GeExecutor::LoadDataFromFile(const std::string &path, ModelData &model_data) {
  GELOGI("Load data from file begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

  string filePath = RealPath(path.c_str());
  if (filePath.empty()) {
    GELOGE(ge::FAILED, "File path is invalid. please check your text file '%s'.", path.c_str());
    return ge::FAILED;
  }
  GELOGI("load modelData from file: %s.", path.c_str());
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

/**
* @ingroup ge
* @brief Load model from offline model memory data
* @param [in] domi::ModelData &model_data: Offline model data
              void *dev_ptr: Input/Output memory start address
              size_t memsize: Input/Output memory length
              void *weight_ptr: Weight memory start address
              size_t weightsize: Weight memory length
* @param [out] uint32_t &model_id: identification after model loading
* @return SUCCESS handle successfully / others handle failed
*/
Status GeExecutor::LoadModelFromData(uint32_t &model_id, const ModelData &model_data, void *dev_ptr, size_t mem_size,
                                     void *weight_ptr, size_t weight_size) {
  GELOGI("Load model from data begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  return GraphLoader::LoadModelFromData(model_id, model_data, dev_ptr, mem_size, weight_ptr, weight_size);
}

/**
 * @ingroup ge
 * @brief Load task list from ModelData with queue.
 * @param [out] model_id: model id allocate from manager.
 * @param [in] ge_model_data: Model data load from offline model.
 * @param [in] input_queue_ids: input queue ids create from user.
 * @param [in] output_queue_ids: input queue ids create from user.
 * @return: 0 for success / others for fail
 */
Status GeExecutor::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                                  const std::vector<uint32_t> &input_queue_ids,
                                  const std::vector<uint32_t> &output_queue_ids) {
  GELOGI("Load model with queue begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }
  return GraphLoader::LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids);
}

/**
* @ingroup ge
* @brief Synchronous execution of offline model(Do not create thread)
* @param [in] uint32_t model_id: Model ID to execute
              void* stream: stream to execute
              const domi::InputData *input_data: Model input data
              bool async_mode: is asynchronize mode.
* @param [out] domi::OutputData *output_data: Model output data
* @return SUCCESS handle successfully / others handle failed
*/
Status GeExecutor::ExecModel(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                             ge::RunModelData &run_output_data, bool async_mode) {
  GELOGI("Execute model begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

  InputData input_data;
  OutputData output_data;
  GetDomiInputData(run_input_data, input_data);
  GetDomiOutputData(run_output_data, output_data);

  if ((run_input_data.dynamic_batch_size != 0) || (run_input_data.dynamic_image_width != 0) ||
      (run_input_data.dynamic_image_height != 0)) {
    std::vector<std::vector<int64_t>> batch_info;
    Status ret = GraphExecutor::GetDynamicBatchInfo(model_id, batch_info);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Get dynamic input info failed.");
      return FAILED;
    }
    if (!batch_info.empty()) {
      SetDynamicInputDataFlag(run_input_data, batch_info, input_data);
    }
  }

  return GraphLoader::ExecuteModel(model_id, stream, async_mode, input_data, output_data);
}

/**
* @ingroup ge
* @brief Get weight memory size from model file
* @param [in] const std::string &path: Offline model file path
* @param [out] size_t &mem_size Execution memory size
               size_t &weight_size Weight memory space size
* @return SUCCESS handle successfully / others handle failed
*/
Status GeExecutor::GetMemAndWeightSize(const std::string &path, size_t &mem_size, size_t &weight_size) {
  GELOGI("Get memory and weight size from file begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

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

/**
* @ingroup ge
* @brief Get weight memory size from model file
* @param [in] const void *model_data Offline model buffer
              size_t model_size Offline model buffer length
* @param [out] size_t &mem_size Execution memory size
               size_t &weight_size Weight memory space size
* @return SUCCESS handle successfully / others handle failed
*/
Status GeExecutor::GetMemAndWeightSize(const void *model_data, size_t model_size, size_t &mem_size,
                                       size_t &weight_size) {
  GELOGI("Get memory and weight size from data begin.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "GeExecutor has not been initialized!");
    return GE_EXEC_NOT_INIT;
  }

  if (model_data == nullptr) {
    GELOGE(PARAM_INVALID, "invalid model data!");
    return PARAM_INVALID;
  }

  ModelData model;
  model.model_data = const_cast<void *>(model_data);
  model.model_len = static_cast<uint32_t>(model_size);

  return ge::ModelManager::GetModelMemAndWeightSize(model, mem_size, weight_size);
}

Status GeExecutor::LoadSingleOp(const std::string &model_name, const ge::ModelData &modelData, void *stream,
                                SingleOp **single_op) {
  return SingleOpManager::GetInstance().GetOpFromModel(model_name, modelData, stream, single_op);
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

Status GeExecutor::GetBatchInfoSize(uint32_t model_id, size_t &shape_count) {
  std::vector<std::vector<int64_t>> batch_info;
  Status ret = GetDynamicBatchInfo(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "Calc batch info size failed. ret = %d", ret);
    return ret;
  }
  if (batch_info.empty()) {
    shape_count = kStaticBatchInfoSize;
  } else {
    shape_count = batch_info.size();
  }
  return SUCCESS;
}

Status GeExecutor::GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &orig_input_info) {
  GELOGI("Begin to GetOrigInputInfo.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  Status ret = GraphExecutor::GetOrigInputInfo(model_id, index, orig_input_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetOrigInputInfo failed.");
    return ret;
  }

  GELOGI("GetOrigInputInfo succ.");
  return SUCCESS;
}

Status GeExecutor::GetAllAippInputOutputDims(uint32_t model_id, uint32_t index,
                                             std::vector<InputOutputDims> &input_dims,
                                             std::vector<InputOutputDims> &output_dims) {
  GELOGI("Begin to GetAllAippInputOutputDims.");
  if (!isInit_) {
    GELOGE(GE_EXEC_NOT_INIT, "not inited yet!");
    return GE_EXEC_NOT_INIT;
  }

  Status ret = GraphExecutor::GetAllAippInputOutputDims(model_id, index, input_dims, output_dims);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetAllAippInputOutputDims failed.");
    return ret;
  }

  GELOGI("GetAllAippInputOutputDims succ.");
  return SUCCESS;
}
}  // namespace ge
