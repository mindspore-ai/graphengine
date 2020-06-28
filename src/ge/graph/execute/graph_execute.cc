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

#include "graph/execute/graph_execute.h"

#include <memory>
#include <string>

#include "common/ge_inner_error_codes.h"
#include "common/model_parser/base.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "omm/csa_interact.h"
#include "runtime/dev.h"
#include "runtime/mem.h"

namespace ge {
GraphExecutor::GraphExecutor()
    : init_flag_(false),
      train_graph_flag_(false),
      sync_run_mutex_(nullptr),
      condition_(nullptr),
      graph_run_listener_(nullptr),
      graph_context_(nullptr),
      last_graph_id_(UINT32_MAX),
      malloc_flag_(false) {}

GraphExecutor::~GraphExecutor() {
  outputs_desc_.clear();
  if (malloc_flag_) {
    for (auto &buffer_addr : buffer_addr_) {
      rtError_t rt_ret;
      rt_ret = rtFreeHost(buffer_addr);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "[GraphManager] subgraph free buffer failed, ret: 0x%X", rt_ret);
      }
    }
  }
  malloc_flag_ = false;
  buffer_addr_.clear();
}

Status GraphExecutor::SetCondition(std::mutex *mutex, std::condition_variable *cond,
                                   std::shared_ptr<GraphModelListener> listener) {
  if (mutex == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[SetCondition] input param mutex is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  if (cond == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[SetCondition] input param cond is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  if (listener == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[SetCondition] input param listener is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  sync_run_mutex_ = mutex;
  condition_ = cond;

  graph_run_listener_ = listener;

  init_flag_ = true;

  return SUCCESS;
}

Status GraphExecutor::SetGraphContext(GraphContextPtr graph_context_ptr) {
  if (graph_context_ptr == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[SetGraphContext] input param graph_context_ptr is nullptr");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  graph_context_ = graph_context_ptr;
  return SUCCESS;
}

Status GraphExecutor::SetDynamicSize(uint32_t model_id, const std::vector<uint64_t> &batch_num) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->SetDynamicSize(model_id, batch_num);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "SetDynamicSize failed");
    return ret;
  }
  return SUCCESS;
}

void GraphExecutor::SetTrainFlag(bool is_train_graph) { train_graph_flag_ = is_train_graph; }

Status GraphExecutor::FreeInOutBuffer() {
  if (malloc_flag_) {
    for (auto iter = buffer_addr_.begin(); iter != buffer_addr_.end(); ++iter) {
      rtError_t rt_ret;
      rt_ret = rtFreeHost(*iter);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "[GraphManager] subgraph free buffer failed, ret: 0x%X", rt_ret);
        (void)buffer_addr_.erase(buffer_addr_.begin(), iter);
        return GE_GRAPH_FREE_FAILED;
      }
    }
    buffer_addr_.clear();

    malloc_flag_ = false;
    return SUCCESS;
  } else {
    GELOGI("[GraphManager] not malloc buffer.");
    return SUCCESS;
  }
}

Status GraphExecutor::MallocInOutBuffer(const std::vector<uint32_t> &buffer_size, std::vector<void *> &data_addr) {
  if (malloc_flag_) {
    auto all_size_same = true;
    if (buffer_size.size() == buffer_size_.size()) {
      for (size_t i = 0; i < buffer_size.size(); i++) {
        if (buffer_size[i] != buffer_size_[i]) {
          all_size_same = false;
          break;
        }
      }
    } else {
      all_size_same = false;
    }
    if (all_size_same) {
      data_addr = buffer_addr_;
      return SUCCESS;
    }
    buffer_size_.clear();
    auto rt_ret = FreeInOutBuffer();
    if (rt_ret != SUCCESS) {
      GELOGE(RT_FAILED, "[SubGraphInfo] MallocInOutBuffer free buffer failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  rtError_t rt_ret;
  for (size_t i = 0; i < buffer_size.size(); ++i) {
    void *tmp_buf = nullptr;
    rt_ret = rtMallocHost(&tmp_buf, buffer_size[i]);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[GraphManager] subgraph malloc buffer failed, ret: 0x%X", rt_ret);
      return GE_GRAPH_MALLOC_FAILED;
    }
    malloc_flag_ = true;
    data_addr.push_back(tmp_buf);
    buffer_addr_.push_back(tmp_buf);
  }
  buffer_size_ = buffer_size;
  return SUCCESS;
}

Status GraphExecutor::PrepareInputData(const std::vector<GeTensor> &input_tensor, InputData &graph_input_data,
                                       OutputData &graph_output_data, std::vector<InputOutputDescInfo> &output_desc) {
  // Preprocessing input data
  graph_input_data.index = 0;
  graph_input_data.timeout = 0;
  graph_input_data.timestamp = 0;
  std::size_t inputSize = input_tensor.size();
  std::size_t output_size = output_desc.size();
  std::vector<uint32_t> bufferSizeVec;
  std::vector<void *> addrVec;

  for (std::size_t i = 0; i < inputSize; ++i) {
    const GeTensor *InTensor = &input_tensor[i];
    GE_CHECK_NOTNULL(InTensor);
    bufferSizeVec.push_back(InTensor->GetData().size());
  }

  for (const auto &desc : output_desc) {
    bufferSizeVec.push_back(desc.size);
  }

  Status ret = MallocInOutBuffer(bufferSizeVec, addrVec);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_MALLOC_FAILED, "[GraphExecutor] Malloc mem failed");
    return GE_GRAPH_MALLOC_FAILED;
  }

  for (std::size_t i = 0; i < input_tensor.size() && i < addrVec.size(); ++i) {
    const GeTensor *in_tensor = &input_tensor[i];
    GE_CHECK_NOTNULL(in_tensor);
    if ((addrVec[i] != nullptr) && (in_tensor->GetData().data() != nullptr)) {
      rtError_t rt_ret = rtMemcpy(addrVec[i], bufferSizeVec[i], in_tensor->GetData().data(),
                                  in_tensor->GetData().size(), RT_MEMCPY_HOST_TO_HOST);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
        return RT_FAILED;
      }
    }

    DataBuffer in_data_buf;
    in_data_buf.data = reinterpret_cast<uint8_t *>(addrVec[i]);
    in_data_buf.length = in_tensor->GetData().size();
    in_data_buf.isDataSupportMemShare = false;
    graph_input_data.blobs.push_back(in_data_buf);
  }

  graph_output_data.index = 0;

  for (std::size_t j = 0; j < output_size; j++) {
    auto desc = output_desc[j];
    uint32_t buffer_size = desc.size;

    DataBuffer out_data_buf;
    out_data_buf.data = reinterpret_cast<uint8_t *>(addrVec[inputSize + j]);
    out_data_buf.length = buffer_size;
    out_data_buf.isDataSupportMemShare = false;
    graph_output_data.blobs.push_back(out_data_buf);
  }

  return SUCCESS;
}

Status GraphExecutor::SyncExecuteModel(uint32_t model_id, const std::vector<GeTensor> &input_tensor,
                                       std::vector<GeTensor> &output_tensor) {
  // Prepare input and output
  std::vector<InputOutputDescInfo> inputs_desc;
  std::vector<InputOutputDescInfo> output_desc;

  GELOGI("[ExecuteGraph] GetInputOutputDescInfo via new ome begin.");
  Status ret = GetInputOutputDescInfo(model_id, inputs_desc, output_desc);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_GET_IN_OUT_FAILED, "[GraphExecutor] GetInputOutputDescInfo failed, modelId=%u.", model_id);
    return GE_GRAPH_GET_IN_OUT_FAILED;
  }
  outputs_desc_.assign(output_desc.begin(), output_desc.end());

  InputData input_data;
  OutputData output_data;
  input_data.model_id = model_id;
  ret = PrepareInputData(input_tensor, input_data, output_data, output_desc);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_PREPARE_FAILED, "[GraphExecutor] PrepareInputData failed, modelId=%u.", model_id);
    return GE_GRAPH_PREPARE_FAILED;
  }

  if (graph_run_listener_->ResetResult() != SUCCESS) {
    GELOGE(GE_GRAPH_EXECUTE_FAILED, "Reset result failed");
    return GE_GRAPH_EXECUTE_FAILED;
  }

  // Run mode async
  GELOGI("[ExecuteGraph] DataInput via new ome begin.");
  ret = DataInput(input_data, output_data);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_DATA_INPUT_FAILED, "[GraphExecutor] push data failed, modelId=%u.", model_id);
    return GE_GRAPH_DATA_INPUT_FAILED;
  }
  GELOGI("[GraphExecutor] input data push to wrapper finish, waiting for result...");

  // Pending until async execute graph complete
  {
    std::unique_lock<std::mutex> ulock(*sync_run_mutex_);
    if (!graph_run_listener_->IsFinished()) {
      (*condition_).wait(ulock);
    }

    // Run graph return
    uint32_t result_code = graph_run_listener_->GetResultCode();
    if (result_code != SUCCESS && result_code != END_OF_SEQUENCE) {
      GELOGE(GE_GRAPH_EXECUTE_FAILED, "[GraphExecutor] execute model failed, ret=%u, modelId=%u.", result_code,
             model_id);
      return GE_GRAPH_EXECUTE_FAILED;
    }
  }
  for (size_t i = 0; i < output_data.blobs.size(); ++i) {
    DataBuffer outputDataTmp = output_data.blobs[i];
    CHECK_FALSE_EXEC(outputDataTmp.length != 0,
                     GELOGE(GE_GRAPH_EXECUTE_FAILED, "Failed to allocate memory, length is 0.");
                     return GE_GRAPH_EXECUTE_FAILED);
    std::unique_ptr<uint8_t> outBufTmp(new (std::nothrow) uint8_t[outputDataTmp.length]);
    if (outBufTmp == nullptr) {
      GELOGE(FAILED, "Failed to allocate memory.");
      return FAILED;
    }
    GE_PRINT_DYNAMIC_MEMORY(new, "the output memory of data on training.", sizeof(uint8_t) * outputDataTmp.length)
    rtError_t ret_value =
      rtMemcpy(outBufTmp.get(), outputDataTmp.length, outputDataTmp.data, outputDataTmp.length, RT_MEMCPY_HOST_TO_HOST);
    CHECK_FALSE_EXEC(ret_value == RT_ERROR_NONE,
                     GELOGE(GE_GRAPH_EXECUTE_FAILED, "Call rt api rtMemcpy failed, ret: 0x%X", ret);
                     return GE_GRAPH_EXECUTE_FAILED);
    GeTensor outTensor;
    std::vector<int64_t> shapeDims;
    for (const auto &dim : output_desc[i].shape_info.dims) {
      shapeDims.push_back(dim);
    }

    GeShape outShape(shapeDims);
    outTensor.MutableTensorDesc().SetShape(outShape);
    outTensor.MutableTensorDesc().SetDataType((DataType)output_desc[i].data_type);
    (void)outTensor.SetData(outBufTmp.get(), outputDataTmp.length);
    output_tensor.push_back(outTensor);
  }

  GELOGI("[GraphExecutor] execute model success, modelId=%u.", model_id);

  return SUCCESS;
}

void GraphExecutor::InitModelIdInfo(std::vector<uint32_t> &out_model_id_info,
                                    std::vector<SubGraphInfoPtr> &sub_graph_vec, uint32_t output_size) {
  for (uint32_t i = 0; i < output_size; i++) {
    for (size_t j = 0; j < sub_graph_vec.size(); j++) {
      if (sub_graph_vec[j]->GetOutputFlag().size() == output_size && sub_graph_vec[j]->GetOutputFlag().at(i)) {
        out_model_id_info.push_back(sub_graph_vec[j]->GetModelIdInfo().model_id);
      }
    }
  }
}

Status GraphExecutor::FreeExecuteMemory() {
  auto ret = FreeInOutBuffer();
  if (ret != SUCCESS) {
    GELOGE(ret, "[FreeExecuteMemory] FreeInOutBuffer Error!");
    return ret;
  }

  return SUCCESS;
}

Status GraphExecutor::ExecuteGraph(GraphId graph_id, const GeRootModelPtr &ge_root_model,
                                   const std::vector<GeTensor> &input_tensor, std::vector<GeTensor> &output_tensor) {
  if (graph_id != last_graph_id_) {
    auto ret = FreeExecuteMemory();
    if (ret != SUCCESS) {
      return ret;
    }
  }
  last_graph_id_ = graph_id;

  if (!init_flag_) {
    GELOGE(GE_GRAPH_EXECUTE_NOT_INIT, "[GraphExecutor] AI Core Engine without calling SetCondition!");
    return GE_GRAPH_EXECUTE_NOT_INIT;
  }
  GE_CHECK_NOTNULL_EXEC(ge_root_model, return FAILED);
  Status ret = SyncExecuteModel(ge_root_model->GetModelId(), input_tensor, output_tensor);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_SYNC_MODEL_FAILED, "[GraphExecutor] SyncExecuteModel Error!");
    return GE_GRAPH_SYNC_MODEL_FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::ExecuteGraphAsync(GraphId graph_id, const GeRootModelPtr &ge_root_model,
                                        const std::vector<InputTensorInfo> &input_tensor) {
  GELOGI("[GraphExecutor] Start to async execute graph, graph_id=%u", graph_id);
  if (graph_id != last_graph_id_) {
    auto ret = FreeExecuteMemory();
    if (ret != SUCCESS) {
      return ret;
    }
  }
  last_graph_id_ = graph_id;
  GE_CHECK_NOTNULL_EXEC(ge_root_model, return FAILED);
  Status ret = AsyncExecuteModel(ge_root_model->GetModelId(), input_tensor);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_SYNC_MODEL_FAILED, "[GraphExecutor] AsyncExecuteModel Error!");
    return GE_GRAPH_SYNC_MODEL_FAILED;
  }

  GELOGI("[GraphExecutor] Async execute graph success, graph_id=%u", graph_id);
  return SUCCESS;
}

Status GraphExecutor::AsyncExecuteModel(uint32_t model_id, const std::vector<InputTensorInfo> &inputs) {
  try {
    auto model_manager = ge::ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    GELOGI("RunAsync begin.model_id %u", model_id);

    Status ret = model_manager->DataInputTensor(model_id, inputs);
    if (ret != SUCCESS) {
      GELOGE(ret, "RunAsync: DataInput fail");
      return ret;
    }

    GELOGI("RunAsync success.");
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "RunAsync failed, bad memory allocation occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
    return MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "RunAsync failed, some exceptions occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
    return FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::DataInput(const InputData &input_data, OutputData &output_data) {
  try {
    auto model_manager = ge::ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->DataInput(input_data, output_data);
    if (ret != SUCCESS) {
      GELOGE(ret, "DataInput: DataInput failed.");
      CsaInteract::GetInstance().WriteErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
      return ret;
    }
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "DataInput failed, bad memory allocation occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
    return MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "DataInput failed, some exceptions occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
    return FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                             vector<InputOutputDescInfo> &output_desc) {
  try {
    auto model_manager = ge::ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->GetInputOutputDescInfo(model_id, input_desc, output_desc);
    if (ret != SUCCESS) {
      GELOGE(ret, "GetInputOutputDescInfo  failed.");
      CsaInteract::GetInstance().WriteErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
      return ret;
    }
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "GetInputOutputDescInfo failed, bad memory allocation occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
    return MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "GetInputOutputDescInfo failed, some exceptions occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
    return FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                             vector<InputOutputDescInfo> &output_desc,
                                             std::vector<uint32_t> &input_formats, std::vector<uint32_t> &out_formats) {
  try {
    auto model_manager = ge::ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->GetInputOutputDescInfo(model_id, input_desc, output_desc, input_formats, out_formats);
    if (ret != SUCCESS) {
      GELOGE(ret, "GetInputOutputDescInfo  failed.");
      CsaInteract::GetInstance().WriteErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
      return ret;
    }
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "GetInputOutputDescInfo failed, bad memory allocation occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
    return MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "GetInputOutputDescInfo failed, some exceptions occur !");
    CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
    return FAILED;
  }

  return SUCCESS;
}
///
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [in] model_id
/// @param [out] batch_info
/// @return execute result
///
Status GraphExecutor::GetDynamicBatchInfo(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetDynamicBatchInfo(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetDynamicBatchInfo failed.");
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetCurShape(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "GetCurShape failed");
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetModelAttr(uint32_t model_id, std::vector<string> &dynamic_output_shape_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetModelAttr(model_id, dynamic_output_shape_info);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "GetModelAttr failed");
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetInputOutputDescInfoForZeroCopy(uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                                        vector<InputOutputDescInfo> &output_desc,
                                                        std::vector<uint32_t> &input_formats,
                                                        std::vector<uint32_t> &out_formats) {
  try {
    auto model_manager = ge::ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret =
      model_manager->GetInputOutputDescInfoForZeroCopy(model_id, input_desc, output_desc, input_formats, out_formats);
    if (ret != SUCCESS) {
      GELOGE(ret, "GetInputOutputDescInfoForZeroCopy failed.");
      return ret;
    }
  } catch (std::bad_alloc &) {
    GELOGE(MEMALLOC_FAILED, "GetInputOutputDescInfoForZeroCopy failed, bad memory allocation occur !");
    return MEMALLOC_FAILED;
  } catch (...) {
    GELOGE(FAILED, "GetInputOutputDescInfoForZeroCopy failed, some exceptions occur !");
    return FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::GetAIPPInfo(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetAIPPInfo(model_id, index, aipp_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetAIPPInfo failed.");
    return ret;
  }

  return SUCCESS;
}

Status GraphExecutor::GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &orig_input_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetOrigInputInfo(model_id, index, orig_input_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetOrigInputInfo failed.");
    return ret;
  }

  return SUCCESS;
}

Status GraphExecutor::GetAllAippInputOutputDims(uint32_t model_id, uint32_t index,
                                                std::vector<InputOutputDims> &input_dims,
                                                std::vector<InputOutputDims> &output_dims) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetAllAippInputOutputDims(model_id, index, input_dims, output_dims);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetAllAippInputOutputDims failed.");
    return ret;
  }

  return SUCCESS;
}

}  // namespace ge
