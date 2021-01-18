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

#include "hybrid/executor/hybrid_model_async_executor.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_context.h"
#include "omm/csa_interact.h"

namespace ge {
namespace hybrid {
namespace {
const int kDataOutputIndex = 0;
}
HybridModelAsyncExecutor::HybridModelAsyncExecutor(HybridModel *model)
    : model_(model), run_flag_(false) {
}

HybridModelAsyncExecutor::~HybridModelAsyncExecutor() {
  if (stream_ != nullptr) {
    GE_CHK_RT(rtStreamDestroy(stream_));
  }
}

void HybridModelAsyncExecutor::SetDeviceId(uint32_t device_id) {
  device_id_ = device_id;
}

void HybridModelAsyncExecutor::SetModelId(uint32_t model_id) {
  model_id_ = model_id;
}

Status HybridModelAsyncExecutor::EnqueueData(const shared_ptr<InputDataWrapper> &data) {
  GE_CHK_STATUS_EXEC(data_inputer_->Push(data), return domi::DATA_QUEUE_ISFULL,
                     "Data queue is full, please call again later, model_id %u ", model_id_);
  GELOGD("EnqueueData successfully. model_id = %u, data_index = %u", data->GetInput().model_id, data->GetInput().index);
  return SUCCESS;
}

Status HybridModelAsyncExecutor::Start(const std::shared_ptr<ModelListener> &listener) {
  GELOGD("HybridModelExecutor::Start IN, has listener = %d", listener != nullptr);
  std::lock_guard<std::mutex> lk(mu_);
  GE_CHK_BOOL_RET_STATUS(!run_flag_, INTERNAL_ERROR, "Model already started.");

  run_flag_ = true;
  listener_ = listener;
  future_ = std::async(std::launch::async, [&]() -> Status {
    GetThreadLocalContext() = *executor_->GetContext()->ge_context;
    GetContext().SetSessionId(executor_->GetContext()->session_id);
    return RunInternal();
  });

  GE_CHK_BOOL_RET_STATUS(future_.valid(), INTERNAL_ERROR, "Failed to start.");
  GELOGD("HybridModelExecutor::Start successfully");
  return SUCCESS;
}

Status HybridModelAsyncExecutor::Stop() {
  std::lock_guard<std::mutex> lk(mu_);
  run_flag_ = false;
  data_inputer_->Stop();

  Status ret = SUCCESS;
  if (future_.valid()) {
    ret = future_.get();
  }

  if (stream_ != nullptr) {
    GE_CHK_RT(rtStreamDestroy(stream_));
    stream_ = nullptr;
  }

  return ret;
}

Status HybridModelAsyncExecutor::Init() {
  data_inputer_ = std::unique_ptr<DataInputer>(new(std::nothrow) DataInputer());
  GE_CHECK_NOTNULL(data_inputer_);
  GE_CHK_RT_RET(rtStreamCreate(&stream_, RT_STREAM_PRIORITY_DEFAULT));

  executor_ = std::unique_ptr<HybridModelExecutor>(new(std::nothrow) HybridModelExecutor(model_, device_id_, stream_));
  GE_CHECK_NOTNULL(executor_);
  GE_CHK_STATUS_RET(executor_->Init(), "Failed to init hybrid engine");
  GE_CHK_STATUS_RET(InitInputDesc(), "Failed to init input tensors");
  return SUCCESS;
}

Status HybridModelAsyncExecutor::PreRun(InputData &current_data, HybridModelExecutor::ExecuteArgs &args) {
  GE_CHK_STATUS_RET(SyncVarData(), "Failed to sync var data");
  RECORD_MODEL_EXECUTION_EVENT(executor_->GetContext(), "[SyncVarData] End");
  GE_CHK_STATUS_RET(PrepareInputs(current_data, args), "Failed to copy input data to model");
  RECORD_MODEL_EXECUTION_EVENT(executor_->GetContext(), "[CopyInputData] End");
  return SUCCESS;
}

Status HybridModelAsyncExecutor::RunInternal() {
  auto device_id = static_cast<int32_t>(device_id_);
  GELOGD("Hybrid model start. model_id = %u, device_id = %u", model_id_, device_id_);
  GE_CHK_RT_RET(rtSetDevice(device_id));
  // DeviceReset before thread run finished!
  GE_MAKE_GUARD(not_used_var, [&] { GE_CHK_RT(rtDeviceReset(device_id)); });

  while (run_flag_) {
    std::shared_ptr<InputDataWrapper> data_wrapper;
    Status ret = data_inputer_->Pop(data_wrapper);
    if (data_wrapper == nullptr || ret != SUCCESS) {
      GELOGI("data_wrapper is null!, ret = %u", ret);
      continue;
    }

    GELOGI("Getting the input data, model_id:%u", model_id_);
    GE_IF_BOOL_EXEC(!run_flag_, break);
    InputData current_data = data_wrapper->GetInput();
    GELOGI("Model thread Run begin, model id:%u, data index:%u.", model_id_, current_data.index);

    RECORD_MODEL_EXECUTION_EVENT(executor_->GetContext(), "[RunInternal] [iteration = %d] Start", iterator_count_);
    HybridModelExecutor::ExecuteArgs args;
    ret = PreRun(current_data, args);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ret != SUCCESS, (void) HandleResult(ret, current_data.index, args, data_wrapper->GetOutput());
        CsaInteract::GetInstance().StoreInternalErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
        continue, "PreRun failed.");  // [No need to check value]

    ret = executor_->Execute(args);
    ret = HandleResult(ret, current_data.index, args, data_wrapper->GetOutput());
    if (ret != SUCCESS) {
      CsaInteract::GetInstance().StoreInternalErrorCode(ret, ERROR_MODULE_RUNTIME, JOBSUBSTATE_GRAPH_EXEC);
      continue;
    }

    RECORD_MODEL_EXECUTION_EVENT(executor_->GetContext(), "[RunInternal] [iteration = %d] End", iterator_count_);
    iterator_count_++;
    GELOGI("run iterator count is %lu", iterator_count_);
  }

  CsaInteract::GetInstance().WriteInternalErrorCode();
  GELOGI("Model run end, model id:%u", model_id_);
  return SUCCESS;
}

Status HybridModelAsyncExecutor::HandleResult(Status exec_ret,
                                              uint32_t data_id,
                                              HybridModelExecutor::ExecuteArgs &args,
                                              OutputData *output_data) {
  GELOGD("Start to handle result. model id = %u, data index = %u, execution ret = %u", model_id_, data_id, exec_ret);
  std::vector<ge::OutputTensorInfo> output_tensor_info_list;
  if (args.is_eos) {
    GELOGI("End of sequence, model id = %u", model_id_);
    GE_CHK_STATUS_RET_NOLOG(OnComputeDone(data_id, END_OF_SEQUENCE, output_tensor_info_list));
    return SUCCESS;
  }

  if (exec_ret != SUCCESS) {
    GELOGE(exec_ret, "Failed to execute graph. model_id = %u", model_id_);
    return OnComputeDone(data_id, INTERNAL_ERROR, output_tensor_info_list);
  }

  GE_CHECK_NOTNULL(output_data);
  auto ret = CopyOutputs(args, output_data, output_tensor_info_list);
  if (ret != SUCCESS) {
    OnComputeDone(data_id, INTERNAL_ERROR, output_tensor_info_list);
    return INTERNAL_ERROR;
  }

  GELOGD("Executed graph successfully, model id = %u, data_index = %u", model_id_, data_id);
  return OnComputeDone(data_id, SUCCESS, output_tensor_info_list);
}

Status HybridModelAsyncExecutor::SyncVarData() {
  GELOGI("Sync var data, model id:%u", model_id_);

  TensorValue *global_step_var = model_->GetVariable(NODE_NAME_GLOBAL_STEP);
  if (global_step_var != nullptr) {
    std::vector<uint64_t> v_step;
    v_step.push_back(iterator_count_);
    GE_CHK_RT_RET(rtMemcpy(global_step_var->MutableData(),
                           global_step_var->GetSize(),
                           v_step.data(),
                           v_step.size() * sizeof(uint64_t),
                           RT_MEMCPY_HOST_TO_DEVICE));
  } else {
    GELOGD("No GLOBAL_STEP variable was found.");
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::PrepareInputs(const InputData &current_data, HybridModelExecutor::ExecuteArgs &args) {
  if (current_data.blobs.size() < input_tensor_desc_.size()) {
    GELOGE(PARAM_INVALID, "Blob size mismatches, expect at least %zu, but got %zu",
           input_tensor_desc_.size(), current_data.blobs.size());
    return PARAM_INVALID;
  }

  auto allocator = NpuMemoryAllocator::GetAllocator(device_id_);
  GE_CHECK_NOTNULL(allocator);
  args.input_desc.resize(input_tensor_desc_.size());
  const std::vector<DataBuffer> &blobs = current_data.blobs;
  for (size_t input_index = 0; input_index < input_tensor_desc_.size(); ++input_index) {
    auto tensor_size = input_sizes_[input_index];
    if (is_input_dynamic_[input_index]) {
      if (input_index >= current_data.shapes.size()) {
        GELOGE(PARAM_INVALID, "Shape index out of range, index = %zu, shape size = %zu",
               input_index, current_data.shapes.size());
        return PARAM_INVALID;
      }
      auto &tensor_desc = input_tensor_desc_[input_index];
      tensor_desc->SetShape(GeShape(current_data.shapes[input_index]));
      args.input_desc[input_index] = tensor_desc;
      GELOGD("Update shape of input[%u] to [%s]", input_index, tensor_desc->MutableShape().ToString().c_str());
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorMemorySizeInBytes(*tensor_desc, tensor_size),
                              "Failed to calc tensor size, index = %zu, shape = [%s]",
                              input_index,
                              tensor_desc->GetShape().ToString().c_str());
      GELOGD("Input tensor[%zu] size = %zu", input_index, tensor_size);
    }

    GE_CHECK_GE(tensor_size, 0);
    AllocationAttr attr;
    if (GetContext().GetHostExecFlag()) {
      attr.SetMemType(HOST_DDR);
    }
    auto tensor_buffer = TensorBuffer::Create(allocator, tensor_size, &attr);
    GE_CHECK_NOTNULL(tensor_buffer);
    args.inputs.emplace_back(std::shared_ptr<TensorBuffer>(tensor_buffer.release()));

    GELOGD("To copy input data for input[%u]", input_index);
    const DataBuffer &data_buf = blobs[input_index];
    auto mem_size = static_cast<uint64_t>(tensor_size);
    GE_CHK_BOOL_RET_STATUS(mem_size >= data_buf.length,
                           PARAM_INVALID,
                           "input data size(%lu) does not match model required size(%lu), ret failed.",
                           data_buf.length,
                           mem_size);

    GELOGI("[IMAS]CopyPlainData memcpy graph_%u type[F] output[%u] memaddr[%p] mem_size[%u] datasize[%lu]",
           model_->root_runtime_param_.graph_id,
           input_index,
           args.inputs[input_index].GetData(),
           mem_size,
           data_buf.length);
    GE_CHK_RT_RET(rtMemcpy(args.inputs[input_index].MutableData(),
                           mem_size,
                           data_buf.data,
                           data_buf.length,
                           RT_MEMCPY_HOST_TO_DEVICE));
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::InitInputDesc() {
  int input_index = 0;
  for (const auto &input_node : model_->GetRootGraphItem()->GetInputNodes()) {
    GELOGD("Init input[%u], node = %s, is_dynamic = %d",
           input_index,
           input_node->NodeName().c_str(),
           input_node->is_dynamic);
    auto output_desc = input_node->MutableOutputDesc(kDataOutputIndex);
    GE_CHECK_NOTNULL(output_desc);
    int64_t tensor_size = -1;
    if (!input_node->is_dynamic) {
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetSize(*output_desc, tensor_size),
                              "Failed to get size from %s",
                              input_node->NodeName().c_str());

      if (tensor_size == 0) {
        GELOGW("[%s] Tensor size == 0", input_node->NodeName().c_str());
        GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorMemorySizeInBytes(*output_desc, tensor_size),
                                "Failed to calc tensor size");
        GELOGD("[%s] Tensor size updated to %ld", input_node->NodeName().c_str(), tensor_size);
      }
    }

    input_sizes_.emplace(input_index, tensor_size);
    input_tensor_desc_.emplace(input_index, output_desc);
    is_input_dynamic_.push_back(input_node->is_dynamic);
    input_index += 1;
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::OnComputeDone(uint32_t data_index, uint32_t result_code,
                                               std::vector<ge::OutputTensorInfo> &outputs) {
  GELOGD("OnComputeDone. model id = %u, data index = %u, execution ret = %u", model_id_, data_index, result_code);
  if (listener_ != nullptr) {
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_index, result_code, outputs),
                  "OnComputeDone failed");
  }

  return result_code;
}

Status HybridModelAsyncExecutor::CopyOutputs(HybridModelExecutor::ExecuteArgs &args,
                                             OutputData *output_data,
                                             std::vector<ge::OutputTensorInfo> &outputs) {
  // copy output data from op to designated position
  std::vector<ConstGeTensorDescPtr> &output_tensor_desc_list = args.output_desc;
  std::vector<TensorValue> &output_tensors = args.outputs;
  if (output_tensor_desc_list.size() != output_tensors.size()) {
    GELOGE(INTERNAL_ERROR,
           "Output sizes mismatch. From op_desc = %zu, and from output tensors = %zu",
           output_tensor_desc_list.size(),
           output_tensors.size());
    return INTERNAL_ERROR;
  }

  GELOGD("Number of outputs = %zu", output_tensor_desc_list.size());
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    GELOGD("Start to process output[%zu]", i);
    auto &output_tensor = output_tensors[i];
    auto &tensor_desc = output_tensor_desc_list.at(i);
    GE_CHECK_NOTNULL(tensor_desc);
    int64_t output_size = -1;
    GE_CHK_GRAPH_STATUS_RET(TensorUtils::CalcTensorMemSize(tensor_desc->GetShape(),
                                                           tensor_desc->GetFormat(),
                                                           tensor_desc->GetDataType(),
                                                           output_size),
                            "Failed to calc tensor size for output[%zu]. shape = [%s], type = %s, format = %s",
                            i,
                            tensor_desc->GetShape().ToString().c_str(),
                            TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(),
                            TypeUtils::FormatToSerialString(tensor_desc->GetFormat()).c_str());

    GELOGD("Got tensor size for output[%zu] successfully. shape = [%s], type = %s, format = %s, size = %ld",
           i,
           tensor_desc->GetShape().ToString().c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(tensor_desc->GetFormat()).c_str(),
           output_size);

    GE_CHECK_GE(output_size, 0);
    GE_CHECK_LE(output_size, UINT32_MAX);
    if (output_tensor.GetSize() < static_cast<size_t>(output_size)) {
      GELOGE(INTERNAL_ERROR,
             "output[%zu] tensor size(%zu) is not enough for output shape [%s]",
             i, output_tensor.GetSize(), tensor_desc->GetShape().ToString().c_str());
      return INTERNAL_ERROR;
    }

    ge::OutputTensorInfo output;
    output.data_type = static_cast<uint32_t>(tensor_desc->GetDataType());
    output.dims = tensor_desc->GetShape().GetDims();
    output.length = output_size;
    if (output_size > 0) {
      std::unique_ptr<uint8_t[]> data_buf(new(std::nothrow) uint8_t[output_size]);
      GE_CHECK_NOTNULL(data_buf);
      GE_CHK_RT_RET(rtMemcpy(data_buf.get(),
                             output_size,
                             output_tensor.GetData(),
                             output_size,
                             RT_MEMCPY_DEVICE_TO_HOST));
      output.data = std::move(data_buf);
      output_data->blobs.emplace_back(data_buf.get(), static_cast<uint32_t>(output_size), false);
    } else {
      GELOGW("Output[%zu] is empty. shape = [%s]", i, tensor_desc->GetShape().ToString().c_str());
      output.data = nullptr;
      output_data->blobs.emplace_back(nullptr, 0U, false);
    }

    outputs.emplace_back(std::move(output));
    GELOGD("Output[%zu] added, type = %s, shape = [%s], size = %ld",
           i,
           TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(),
           tensor_desc->GetShape().ToString().c_str(),
           output_size);
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::Execute(const std::vector<DataBuffer> &inputs,
                                         const std::vector<GeTensorDesc> &input_desc,
                                         std::vector<DataBuffer> &outputs,
                                         std::vector<GeTensorDesc> &output_desc) {
  GELOGI("Start to execute model.");

  HybridModelExecutor::ExecuteArgs args;
  args.inputs.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    TensorValue tensor_value(inputs[i].data, inputs[i].length);
    args.inputs[i] = tensor_value;
  }
  GE_CHK_STATUS_RET(executor_->Execute(args), "Failed to execute model.");
  for (const auto &output_tensor_desc : args.output_desc) {
    output_desc.emplace_back(*output_tensor_desc);
  }

  for (size_t i = 0; i < args.outputs.size(); ++i) {
    int64_t output_real_size = 0;
    ge::graphStatus graph_status = TensorUtils::GetTensorSizeInBytes(output_desc[i], output_real_size);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Get tensor size in bytes failed.");
      return FAILED;
    }
    if (output_real_size > 0) {
      if (outputs[i].length < static_cast<uint64_t>(output_real_size)) {
        GELOGE(FAILED, "output idx[%zu], the memory size of output[%lu] given by "
                       "user should be greater than or equal to the real size of output[%ld]",
               i, outputs[i].length, output_real_size);
        return FAILED;
      }
      GE_CHK_RT_RET(rtMemcpy(outputs[i].data, outputs[i].length, args.outputs[i].GetData(), output_real_size, 
                    RT_MEMCPY_DEVICE_TO_DEVICE));
    }
    outputs[i].length = output_real_size;
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::Execute(const vector<GeTensor> &inputs, vector<GeTensor> &outputs) {
  GELOGD("Start to execute model.");
  // prepare inputs
  InputData input_data;
  for (auto &tensor : inputs) {
    DataBuffer buffer;
    buffer.data = const_cast<uint8_t *>(tensor.GetData().GetData());
    buffer.length = tensor.GetData().size();
    input_data.blobs.emplace_back(buffer);
    input_data.shapes.emplace_back(tensor.GetTensorDesc().GetShape().GetDims());
  }

  HybridModelExecutor::ExecuteArgs args;
  GE_CHK_STATUS_RET(PrepareInputs(input_data, args), "Failed to copy input data to model");
  GELOGD("Done copying input data successfully.");
  GE_CHK_STATUS_RET(executor_->Execute(args), "Failed to execute model.");

  std::vector<ge::OutputTensorInfo> output_tensor_info_list;
  OutputData output_data;
  GE_CHK_STATUS_RET(CopyOutputs(args, &output_data, output_tensor_info_list), "Failed to copy outputs.");
  GELOGD("Done copying output data successfully. output count = %zu", output_tensor_info_list.size());

  int out_index = 0;
  outputs.resize(output_tensor_info_list.size());
  for (auto &out_tensor_info : output_tensor_info_list) {
    auto &ge_tensor = outputs[out_index];
    if (out_tensor_info.length > 0) {
      GE_CHK_GRAPH_STATUS_RET(ge_tensor.SetData(out_tensor_info.data.get(), out_tensor_info.length),
                              "Failed to set output[%d].", out_index);
    }

    ge_tensor.MutableTensorDesc() = *args.output_desc[out_index];
    GELOGD("Set output[%d], tensor size = %ld, shape = [%s]",
           out_index,
           out_tensor_info.length,
           ge_tensor.MutableTensorDesc().MutableShape().ToString().c_str());
    ++out_index;
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
