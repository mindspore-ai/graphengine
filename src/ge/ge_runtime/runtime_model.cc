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

#include "ge_runtime/runtime_model.h"

#include <set>

#include "./model_context.h"
#include "./task/task.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "framework/common/op/op_parser_util.h"
#include "graph/types.h"
#include "ge_runtime/op_info_utils.h"
#include "task/task_factory.h"

namespace ge {
namespace model_runner {
RuntimeModel::~RuntimeModel() {
  GELOGI("RuntimeModel destructor start");

  // Release task first, hccl task hold stream
  task_list_.clear();

  // Unbind rtModel from all task related streams
  RtModelUnbindStream();

  // Release all task related streams
  RtStreamDestory();

  // Release rtlabel resource
  RtLabelDestory();

  // Release rtEvent resourece
  RtEventDestory();

  GELOGI("Do RtModelDestory");
  // Release all rt_model
  RtModelDestory();
}

bool RuntimeModel::InitStream(std::shared_ptr<DavinciModel> &davinci_model) {
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "Davinci model is null.");
    return false;
  }

  std::set<int64_t> wait_active_streams;
  std::set<int64_t> force_copy_streams;

  for (const auto &stream_id : davinci_model->GetWaitActiveStreams()) {
    GELOGI("stream id %u is wait active stream.", stream_id);
    (void)wait_active_streams.insert(stream_id);
  }

  for (const auto &stream_id : davinci_model->GetForceCopyStreams()) {
    GELOGI("stream id %u is force copy stream.", stream_id);
    (void)force_copy_streams.insert(stream_id);
  }

  GELOGI("stream number:%u", davinci_model->GetStreamNum());
  for (uint32_t i = 0; i < davinci_model->GetStreamNum(); ++i) {
    rtStream_t stream = nullptr;
    uint32_t flag = (force_copy_streams.find(i) != force_copy_streams.end())
                    ? (RT_STREAM_PERSISTENT | RT_STREAM_FORCE_COPY)
                    : (RT_STREAM_PERSISTENT);

    rtError_t rt_ret = rtStreamCreateWithFlags(&stream, davinci_model->GetPriority(), flag);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api rtStreamCreate failed, ret: 0x%X", rt_ret);
      return false;
    }

    GELOGI("rtStreamCreateWithFlags end.");

    stream_list_.emplace_back(stream);

    // Bind rt_model_handle_ to all task related streams
    flag = (wait_active_streams.find(i) != wait_active_streams.end()) ? (static_cast<uint32_t>(RT_INVALID_FLAG))
                                                                      : (static_cast<uint32_t>(RT_HEAD_STREAM));
    rt_ret = rtModelBindStream(rt_model_handle_, stream, flag);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api rtModelBindStream failed, ret: 0x%X", rt_ret);
      return false;
    }
  }

  return true;
}

bool RuntimeModel::InitEvent(uint32_t event_num) {
  GELOGI("event number:%u.", event_num);
  for (uint32_t i = 0; i < event_num; ++i) {
    rtEvent_t rt_event;
    rtError_t rt_ret = rtEventCreate(&rt_event);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api rtEventCreate failed, i; %u; ret: 0x%X", i, rt_ret);
      return false;
    }
    event_list_.push_back(rt_event);
  }
  return true;
}

bool RuntimeModel::InitLabel(uint32_t batch_num) {
  GELOGI("batch number:%u.", batch_num);
  for (uint32_t i = 0; (batch_num != 0 && i <= batch_num); ++i) {
    rtLabel_t rt_lLabel = nullptr;
    rtError_t rt_ret = rtLabelCreate(&rt_lLabel);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api rtLabelCreate failed, i; %u; ret: 0x%X", i, rt_ret);
      return false;
    }

    if (rt_lLabel == nullptr) {
      GELOGE(RT_FAILED, "rtLabel is nullptr!");
      return false;
    }

    label_list_.emplace_back(rt_lLabel);
  }
  return true;
}

bool RuntimeModel::InitResource(std::shared_ptr<DavinciModel> &davinci_model) {
  GELOGI("InitResource start");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci model is null");
    return false;
  }
  rtError_t rt_ret = rtModelCreate(&rt_model_handle_, 0);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api rtModelCreate failed, ret: 0x%X", rt_ret);
    return false;
  }

  // Create rtStream for rt_model_handle_
  rt_ret = rtStreamCreate(&rt_model_stream_, davinci_model->GetPriority());
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api rtStreamCreate failed, ret: 0x%X", rt_ret);
    return false;
  }
  GELOGI("rtStreamCreate end");

  if (!InitStream(davinci_model)) {
    return false;
  }

  if (!InitEvent(davinci_model->GetEventNum())) {
    return false;
  }

  if (!InitLabel(davinci_model->GetBatchNum())) {
    return false;
  }

  GELOGI("InitResource succ");
  return true;
}

void RuntimeModel::GenerateTask(uint32_t device_id, uint64_t session_id, std::shared_ptr<DavinciModel> &davinci_model) {
  GELOGI("GenerateTask start.");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci model is null");
    return;
  }
  auto task_infos = davinci_model->GetTaskInfoList();
  ModelContext model_context(device_id, session_id, davinci_model->GetPriority(), rt_model_handle_, rt_model_stream_,
                             stream_list_, label_list_, event_list_);
  for (auto &task_info : task_infos) {
    auto task = TaskFactory::GetInstance().Create(model_context, task_info);
    task_list_.push_back(task);
  }
  GELOGI("GenerateTask succ.");
}

bool RuntimeModel::LoadTask() {
  GELOGI("LoadTask start.");
  for (auto &task : task_list_) {
    if (task == nullptr) {
      GELOGE(PARAM_INVALID, "task is null.");
      continue;
    }
    bool ret = task->Distribute();
    if (!ret) {
      GELOGE(FAILED, "task distribute fail.");
      return false;
    }

    uint32_t task_id = 0;
    rtError_t rt_ret = rtModelGetTaskId(rt_model_handle_, &task_id);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X.", rt_ret);
      return false;
    }
    task_id_list_.push_back(task_id);
  }
  GELOGI("Distribute task succ.");

  auto rt_ret = rtModelLoadComplete(rt_model_handle_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api rtModelLoadComplete failed, ret: 0x%X.", rt_ret);
    return false;
  }

  GELOGI("LoadTask succ.");
  return true;
}

bool RuntimeModel::Load(uint32_t device_id, uint64_t session_id, std::shared_ptr<DavinciModel> &davinci_model) {
  bool status = InitResource(davinci_model);
  if (!status) {
    GELOGE(FAILED, "InitResource failed.");
    return status;
  }

  status = InitDataInfo(davinci_model);
  if (!status) {
    GELOGE(FAILED, "InitDataInfo failed.");
    return status;
  }

  status = InitOutputInfo(davinci_model);
  if (!status) {
    GELOGE(FAILED, "InitOutputInfo failed.");
    return status;
  }

  status = InitConstantInfo(davinci_model);
  if (!status) {
    GELOGE(FAILED, "InitConstantInfo failed.");
    return status;
  }

  GenerateTask(device_id, session_id, davinci_model);

  status = LoadTask();
  if (!status) {
    GELOGE(FAILED, "DistributeTask failed");
    return status;
  }

  return status;
}

bool RuntimeModel::Run() {
  GELOGI("Davinci task run start");
  rtError_t ret = rtModelExecute(rt_model_handle_, rt_model_stream_, 0);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Model execute failed, ret = 0x%X", ret);
    return false;
  }

  GELOGI("Run rtModelExecute success");

  ret = rtStreamSynchronize(rt_model_stream_);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Model stream sync failed, ret = 0x%X", ret);
    return false;
  }

  GELOGI("Davinci task run succ.");
  return true;
}

void RuntimeModel::RtModelUnbindStream() noexcept {
  for (size_t i = 0; i < stream_list_.size(); i++) {
    if (rtModelUnbindStream(rt_model_handle_, stream_list_[i]) != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Unbind stream from model failed! Index: %zu", i);
      return;
    }
  }
}

void RuntimeModel::RtStreamDestory() noexcept {
  if (rtStreamDestroy(rt_model_stream_) != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Destroy stream for rt_model failed!");
    return;
  }

  for (size_t i = 0; i < stream_list_.size(); i++) {
    if (rtStreamDestroy(stream_list_[i]) != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Destroy stream failed! Index: %zu", i);
      return;
    }
  }
}

void RuntimeModel::RtLabelDestory() noexcept {
  for (size_t i = 0; i < label_list_.size(); i++) {
    if (rtLabelDestroy(label_list_[i]) != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Destroy label failed! Index: %zu.", i);
      return;
    }
  }
}

void RuntimeModel::RtModelDestory() noexcept {
  rtError_t ret = rtModelDestroy(rt_model_handle_);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", ret);
    return;
  }
}

void RuntimeModel::RtEventDestory() noexcept {
  for (size_t i = 0; i < event_list_.size(); i++) {
    if (rtEventDestroy(event_list_[i]) != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Destroy event failed! Index: %zu", i);
      return;
    }
  }
}

bool RuntimeModel::InitDataInfo(std::shared_ptr<DavinciModel> &davinci_model) {
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci model is null");
    return false;
  }
  data_info_list_ = davinci_model->GetDataInfoList();
  for (auto &data_info : data_info_list_) {
    cce::ccTensorDescriptor_t input_desc = nullptr;
    cce::ccTensorDescriptor_t output_desc = nullptr;
    if (data_info == nullptr) {
      GELOGE(PARAM_INVALID, "data info ptr is null.");
      return false;
    }

    if (data_info->input_tensors.empty() || data_info->output_tensors.empty()) {
      GELOGE(PARAM_INVALID, "data info input tensors size %zu, output tensor size %zu.",
             data_info->input_tensors.size(), data_info->output_tensors.size());
      return false;
    }

    if (static_cast<Format>(data_info->input_tensors[0].format) != FORMAT_FILTER_HWCK) {
      bool ret = OpInfoUtils::InitTensorDescriptor(data_info->input_tensors[0].format,
                                                   data_info->input_tensors[0].datatype,
                                                   data_info->input_tensors[0].dims, input_desc,
                                                   data_info->input_tensors[0].real_dim_cnt);
      if (!ret) {
        GELOGE(FAILED, "InitTensorDescriptor Fail.");
        OpInfoUtils::DestroyTensorDescriptor(input_desc);
        return false;
      }

      input_tensor_desc_list_[data_info->name] = input_desc;
    }

    if (static_cast<Format>(data_info->output_tensors[0].format) != FORMAT_FRACTAL_Z) {
      bool ret = OpInfoUtils::InitTensorDescriptor(data_info->output_tensors[0].format,
                                                   data_info->output_tensors[0].datatype,
                                                   data_info->output_tensors[0].dims, output_desc,
                                                   data_info->output_tensors[0].real_dim_cnt);
      if (!ret) {
        GELOGE(FAILED, "InitTensorDescriptor Fail.");
        OpInfoUtils::DestroyTensorDescriptor(output_desc);
        return false;
      }

      output_tensor_desc_list_[data_info->name] = output_desc;
    }
  }

  return true;
}

bool RuntimeModel::InitOutputInfo(std::shared_ptr<DavinciModel> &davinci_model) {
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci model is null");
    return false;
  }
  output_info_list_ = davinci_model->GetOutputInfoList();
  return true;
}

bool RuntimeModel::CopyInputData(const InputData &input_data) {
  if (input_data.blobs.size() != data_info_list_.size()) {
    GELOGE(PARAM_INVALID, "The input data list size (%zu) does not match the model input list size (%zu)",
           input_data.blobs.size(), data_info_list_.size());
    return false;
  }

  for (const auto &data_info : data_info_list_) {
    if (data_info == nullptr) {
      GELOGE(PARAM_INVALID, "data info is null.");
      return false;
    }

    bool ret = CopyInputDataToModel(input_data.blobs, data_info);
    if (!ret) {
      GELOGE(FAILED, "Copy input data to model ret fail, data_info: %s, model id: %u", data_info->name.c_str(),
             input_data.model_id);
      return false;
    }
  }

  return true;
}

bool RuntimeModel::CopyInputDataToModel(const std::vector<DataBuffer> &data, const std::shared_ptr<OpInfo> &data_info) {
  if (data_info == nullptr) {
    GELOGE(PARAM_INVALID, "data info is empty.");
    return false;
  }
  GELOGI("Start copy input data to model, data info: %s.", data_info->name.c_str());
  if (data.empty()) {
    GELOGE(PARAM_INVALID, "data buffer is empty.");
    return false;
  }

  // Check size
  if (data_info->input_tensors.size() != 1 || data_info->output_tensors.size() != 1) {
    GELOGE(PARAM_INVALID, "Data Op has invalid input_desc_size(%zu) or output_desc_size(%zu)",
           data_info->input_tensors.size(), data_info->output_tensors.size());
    return false;
  }

  // Process filter weight input while online
  if (OpInfoUtils::NeedTransFilter(data_info)) {
    bool ret = OpInfoUtils::TransFilterData(data_info, data[data_info->index].data, data[data_info->index].length);
    if (!ret) {
      GELOGE(FAILED, "TransFilterData fail.");
      return false;
    }
    return true;
  }

  if (data_info->input_tensors[0].size >= data[data_info->index].length) {
    GELOGE(PARAM_INVALID, "The input data size(%u) does not match model required size(%u), ret fail.",
           data[data_info->index].length, data_info->input_tensors[0].size);
    return false;
  }

  // float to float16
  bool need_trans_flag = OpInfoUtils::IsInputTensorNeedTrans(data_info);
  if (need_trans_flag) {
    return CopyTransData(data, data_info);
  } else {
    return CopyHostData(data, data_info);
  }
}

bool RuntimeModel::CopyHostData(const std::vector<DataBuffer> &data, const std::shared_ptr<OpInfo> &data_info) const {
  GELOGI("Start CopyHostData.");
  if (data.empty()) {
    GELOGE(PARAM_INVALID, "data buffer is empty.");
    return false;
  }

  if (data_info == nullptr) {
    GELOGE(PARAM_INVALID, "data info is null.");
    return false;
  }

  void *host_data_addr = data[data_info->index].data;
  uint32_t copy_size = data[data_info->index].length;
  GELOGD("data output tensor is aipp tensor,copy data only.");

  const std::vector<uintptr_t> &outputs = data_info->output_addrs;
  if (outputs.empty()) {
    GELOGE(PARAM_INVALID, "Output addrs is empty.");
    return false;
  }

  // Copy input data to data nodes
  void *data_out_addr = reinterpret_cast<void *>(outputs[0]);

  rtError_t rt_ret = rtMemcpy(data_out_addr, copy_size, host_data_addr, copy_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  return true;
}

bool RuntimeModel::CopyTransData(const std::vector<DataBuffer> &data, const std::shared_ptr<OpInfo> &data_info) {
  GELOGI("Start CopyTransData.");
  if (data.empty()) {
    GELOGE(PARAM_INVALID, "data buffer is empty.");
    return false;
  }

  if (data_info == nullptr) {
    GELOGE(PARAM_INVALID, "data info is null.");
    return false;
  }

  if (data_info->output_tensors.empty()) {
    GELOGE(PARAM_INVALID, "data info output tensors is empty.");
    return false;
  }

  const std::vector<uintptr_t> &outputs = data_info->output_addrs;
  if (outputs.empty()) {
    GELOGE(PARAM_INVALID, "output addrs is empty.");
    return false;
  }

  void *fp16_data_addr = nullptr;
  uint32_t copy_size = data_info->output_tensors[0].size;
  GE_MAKE_GUARD_RTMEM(fp16_data_addr);

  rtError_t rt_ret = rtMallocHost(&fp16_data_addr, copy_size);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  cce::ccStatus_t cc_ret = cce::ccTransTensor(input_tensor_desc_list_[data_info->name], data[data_info->index].data,
                                              output_tensor_desc_list_[data_info->name], fp16_data_addr, copy_size);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    return false;
  }
  void *host_data_addr = fp16_data_addr;

  GELOGI("data output tensor is not aipp tensor,call cce trans tensor.");
  GELOGI("output[0]=%ld, copy_size=%u", outputs[0], copy_size);

  rt_ret = rtMemcpy(reinterpret_cast<void *>(outputs[0]), copy_size, host_data_addr, copy_size,
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  return true;
}

bool RuntimeModel::InitConstantInfo(std::shared_ptr<DavinciModel> &davinci_model) {
  // Const no input, only 1 output, and this output has no data
  // weight data copy to output mem
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "Davinci model is null.");
    return false;
  }
  constant_info_list_ = davinci_model->GetConstantInfoList();

  for (const auto &constant : constant_info_list_) {
    if (constant == nullptr) {
      GELOGE(PARAM_INVALID, "constant is null");
      continue;
    }
    if (constant->output_tensors.empty()) {
      GELOGE(PARAM_INVALID, "Output tensors is empty");
      return false;
    }

    if (constant->weight_tensors.empty()) {
      GELOGE(PARAM_INVALID, "Weight tensors is empty");
      return false;
    }

    if (constant->output_tensors[0].size < constant->weight_data.size()) {
      GELOGE(PARAM_INVALID, "Output size:%u less than weight data size:%zu",
             constant->output_tensors[0].size, constant->weight_data.size());
      return false;
    }

    if (constant->weight_data.empty()) {
      GELOGW("Const op:%s has no weight data.", constant->name.c_str());
      continue;
    }

    if (constant->weight_tensors[0].datatype == DT_STRING) {
      /// If tensor is a scaler, it's shape size if zero, according ge_tensor.cc.
      /// The logic of GetShapeSize is wrong, the scaler tensor's GetShapeSize is zero
      /// and that of unknown shape is zero too.
      /// Unknown shape will not appear here, so we can use zero judge a tensor is scaler or not.
      int64_t elem_num = (constant->weight_tensors[0].GetShapeSize() == 0) ?
                         1 : constant->weight_tensors[0].GetShapeSize();
      if (constant->weight_data.size() < sizeof(uint64_t)) {
        GELOGE(FAILED, "weight_data size is smaller than sizeof(uint64_t)");
        return false;
      }
      uint64_t *buff = reinterpret_cast<uint64_t *>(const_cast<char *>(constant->weight_data.data()));
      int64_t offset = elem_num * 8;
      uintptr_t hbm_raw_data_base_addr = reinterpret_cast<uintptr_t>(constant->output_addrs[0]) + offset;
      for (int64_t i = elem_num - 1; i >= 0; --i) {
        buff[i] = hbm_raw_data_base_addr + (buff[i] - buff[0]);
      }
    }

    rtError_t rt_ret = rtMemcpy(reinterpret_cast<void *>(constant->output_addrs[0]), constant->output_tensors[0].size,
                                constant->weight_data.data(), constant->weight_data.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtGetFunctionByName failed, ret: 0x%X", rt_ret);
      return false;
    }
  }

  return true;
}

bool RuntimeModel::GetInputOutputDescInfo(bool zero_copy,
                                          std::vector<InputOutputDescInfo> *input_desc,
                                          std::vector<InputOutputDescInfo> *output_desc,
                                          std::vector<uint32_t> *input_format,
                                          std::vector<uint32_t> *output_format) {
  if ((data_info_list_.empty()) || (data_info_list_[0]->input_tensors.size() != 1)) {
    // Maybe there is no datainput info while online
    if (!zero_copy && input_format == nullptr) {
      GELOGW("Data op List is null or input_desc size is not 1!");
    } else {
      GELOGE(FAILED, "Data op List is null or input_desc size is not 1!");
      return false;
    }
  }

  bool ret = GetInputDescInfo(input_desc, input_format);
  if (!ret) {
    GELOGE(FAILED, "Get input desc info failed.");
    return false;
  }

  ret = GetOutputDescInfo(output_desc, output_format);
  if (!ret) {
    GELOGE(FAILED, "Get output desc info failed.");
    return false;
  }

  std::vector<uint32_t> output_size_list;
  std::vector<uint32_t> output_memory_size_list;
  for (const auto &output_op : output_info_list_) {
    if (!OpInfoUtils::GetOutputSize(output_op, output_size_list, output_memory_size_list)) {
      GELOGE(FAILED, "GetOutputSize fail.");
      return false;
    }
  }

  if (output_desc->size() != output_size_list.size()) {
    GELOGE(INTERNAL_ERROR, "output_desc size[%zu] not equal output_size_list_[%zu] size!", output_desc->size(),
           output_size_list.size());
    return false;
  }

  const std::vector<uint32_t> &size_list = (zero_copy) ? (output_memory_size_list) : (output_size_list);
  for (size_t i = 0; i < output_size_list.size(); ++i) {
    output_desc->at(i).size = size_list[i];
  }

  return true;
}

bool RuntimeModel::GetInputDescInfo(std::vector<InputOutputDescInfo> *input_desc,
                                    std::vector<uint32_t> *formats) {
  if (input_desc == nullptr) {
    GELOGE(PARAM_INVALID, "Input desc is null.");
    return false;
  }

  // Analyze input dimension information
  for (size_t index = 0; index < data_info_list_.size(); ++index) {
    if (data_info_list_[index]->input_tensors.empty()) {
      GELOGE(INTERNAL_ERROR, "data info list index %zu input tensors is empty.", index);
      return false;
    }
    InputOutputDescInfo input;
    uint32_t n, c, h, w;
    Format format = static_cast<Format>(data_info_list_[index]->input_tensors[0].format);
    if (format == FORMAT_NHWC) {
      n = kNhwcDimN;
      c = kNhwcDimC;
      h = kNhwcDimH;
      w = kNhwcDimW;
    } else {
      n = kNchwDimN;
      c = kNchwDimC;
      h = kNchwDimH;
      w = kNchwDimW;
    }

    if (data_info_list_[index]->input_tensors[0].dims.size() == static_cast<size_t>(domi::NORMAL_TENSOR_SIZE)) {
      input.shape_info.num = data_info_list_[index]->input_tensors[0].GetDim(n);
      input.shape_info.height = data_info_list_[index]->input_tensors[0].GetDim(h);
      input.shape_info.width = data_info_list_[index]->input_tensors[0].GetDim(w);
      input.shape_info.channel = data_info_list_[index]->input_tensors[0].GetDim(c);
    }
    // Original network dimension
    for (size_t k = 0; k < data_info_list_[index]->input_tensors[0].dims.size(); ++k) {
      input.shape_info.dims.push_back(data_info_list_[index]->input_tensors[0].GetDim(k));
    }

    input.data_type = data_info_list_[index]->input_tensors[0].datatype;
    input.name = data_info_list_[index]->name;
    input.size = data_info_list_[index]->input_tensors[0].size;

    input_desc->push_back(input);
    if (formats != nullptr) {
      formats->push_back(format);
    }
  }

  return true;
}

bool RuntimeModel::GetOutputDescInfo(std::vector<InputOutputDescInfo> *output_desc,
                                     std::vector<uint32_t> *formats) {
  if (output_desc == nullptr) {
    GELOGE(PARAM_INVALID, "Output desc is null.");
    return false;
  }

  // Analyze output dimension information
  for (size_t i = 0; i < output_info_list_.size(); ++i) {
    const auto &op_info = output_info_list_[i];
    if (op_info == nullptr) {
      GELOGE(PARAM_INVALID, "Op info at %zu is null.", i);
      return false;
    }
    auto out_size = static_cast<uint32_t>(op_info->output_tensors.size());
    for (uint32_t index = 0; index < out_size; ++index) {
      bool is_output = op_info->output_tensors[index].is_output;
      if (!is_output) {
        continue;
      }

      std::string output_name;
      InputOutputDescInfo output;
      uint32_t format_result;
      CreateOutput(index, *op_info, &output, &format_result);

      std::vector<std::string> src_name = op_info->src_name;
      std::vector<int64_t> src_index = op_info->src_index;
      if (op_info->type == kNetOutPut) {
        GELOGI("Op info %s index %zu is NETOUTPUT.", op_info->name.c_str(), i);
        if (index >= src_name.size() || index >= src_index.size()) {
          GELOGE(INTERNAL_ERROR, "Construct output_name failed.");
          return false;
        }
        output_name = std::string("output_") + std::to_string(index) + "_" + src_name[index] + "_" +
            std::to_string(src_index[index]);
      } else {
        GELOGI("Op info %s index %zu is not NETOUTPUT, type: %s.", op_info->name.c_str(), i, op_info->type.c_str());
        output_name = std::string("output_") + std::to_string(i) + "_" + op_info->name + "_" + std::to_string(index);
      }
      output.name = output_name;

      output_desc->push_back(output);
      if (formats != nullptr) {
        formats->push_back(format_result);
      }
    }
  }
  return true;
}

void RuntimeModel::CreateOutput(uint32_t index, const OpInfo &op_info, InputOutputDescInfo *output,
                                uint32_t *format_result) {
  if (output == nullptr) {
    GELOGE(PARAM_INVALID, "Output desc is null.");
    return;
  }

  int64_t dims[] = {1, 1, 1, 1};
  if (index >= op_info.output_tensors.size()) {
    GELOGE(PARAM_INVALID, "op_info %s output_tensors size %zu, but index %u.", op_info.name.c_str(),
           op_info.output_tensors.size(), index);
    return;
  }

  TensorInfo output_tensor = op_info.output_tensors[index];
  Format format = static_cast<Format>(output_tensor.format);
  if (format_result != nullptr) {
    *format_result = format;
  }

  if (format == FORMAT_ND) {  // For ND tensor
    for (size_t i = 0; i < output_tensor.dims.size() && i < (sizeof(dims) / sizeof(dims[0])); ++i) {
      dims[i] = static_cast<uint32_t>(output_tensor.GetDim(i));
    }
  } else if (format == FORMAT_NHWC) {  // For FORMAT_NHWC
    dims[0] = output_tensor.GetDim(kNhwcDimN);
    dims[1] = output_tensor.GetDim(kNhwcDimC);
    dims[2] = output_tensor.GetDim(kNhwcDimH);
    dims[3] = output_tensor.GetDim(kNhwcDimW);
  } else {  // For FORMAT_NCHW
    dims[0] = output_tensor.GetDim(kNchwDimN);
    dims[1] = output_tensor.GetDim(kNchwDimC);
    dims[2] = output_tensor.GetDim(kNchwDimH);
    dims[3] = output_tensor.GetDim(kNchwDimW);
  }

  output->shape_info.num = dims[0];      // 0: First dim
  output->shape_info.channel = dims[1];  // 1: Second dim
  output->shape_info.height = dims[2];   // 2: Third dim
  output->shape_info.width = dims[3];    // 3: Forth dim

  if (index >= op_info.input_tensors.size()) {
    GELOGE(PARAM_INVALID, "input tensors size %zu less than index %u.", op_info.input_tensors.size(), index);
    return;
  }

  if (op_info.input_tensors[index].format == FORMAT_FRACTAL_Z) {  // FraczToHWCK
    int64_t k = output_tensor.GetDim(0);                          // 0: First dim
    int64_t c = output_tensor.GetDim(1);                          // 1: Second dim
    int64_t h = output_tensor.GetDim(2);                          // 2: Third dim
    int64_t w = output_tensor.GetDim(3);                          // 3: Forth dim
    output->shape_info.dims.push_back(h);
    output->shape_info.dims.push_back(w);
    output->shape_info.dims.push_back(c);
    output->shape_info.dims.push_back(k);

    if (format_result != nullptr) {
      *format_result = FORMAT_HWCN;
    }
  } else {
    for (size_t j = 0; j < output_tensor.dims.size(); ++j) {
      output->shape_info.dims.push_back(output_tensor.GetDim(j));
    }
  }

  output->data_type = output_tensor.datatype;
}

const std::vector<uint32_t> &RuntimeModel::GetTaskIdList() const { return task_id_list_; }

}  // namespace model_runner
}  // namespace ge
