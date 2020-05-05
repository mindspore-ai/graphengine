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
  if (task_list_.empty()) {
    GELOGE(FAILED, "Task list is empty");
    return false;
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

bool RuntimeModel::InitDataInfo(std::shared_ptr<DavinciModel> &davinci_model) { return true; }

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
  return true;
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
      GELOGE(PARAM_INVALID, "Output size:%u less than weight data size:%zu", constant->output_tensors[0].size,
             constant->weight_data.size());
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
      int64_t elem_num =
        (constant->weight_tensors[0].GetShapeSize() == 0) ? 1 : constant->weight_tensors[0].GetShapeSize();
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

bool RuntimeModel::GetInputOutputDescInfo(bool zero_copy, std::vector<InputOutputDescInfo> *input_desc,
                                          std::vector<InputOutputDescInfo> *output_desc,
                                          std::vector<uint32_t> *input_format, std::vector<uint32_t> *output_format) {
  return true;
}

bool RuntimeModel::GetInputDescInfo(std::vector<InputOutputDescInfo> *input_desc, std::vector<uint32_t> *formats) {
  return true;
}

bool RuntimeModel::GetOutputDescInfo(std::vector<InputOutputDescInfo> *output_desc, std::vector<uint32_t> *formats) {
  return true;
}

void RuntimeModel::CreateOutput(uint32_t index, const OpInfo &op_info, InputOutputDescInfo *output,
                                uint32_t *format_result) {}

const std::vector<uint32_t> &RuntimeModel::GetTaskIdList() const { return task_id_list_; }

}  // namespace model_runner
}  // namespace ge
