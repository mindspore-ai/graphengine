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

#include "ge_runtime/task/label_goto_task.h"
#include "ge_runtime/task/task_factory.h"

namespace ge {
namespace model_runner {
std::weak_ptr<LabelGotoTask::LabelManager> LabelGotoTask::LabelManager::instance_;
std::mutex LabelGotoTask::LabelManager::instance_mutex_;

LabelGotoTask::LabelGotoTask(const ModelContext &model_context, const std::shared_ptr<LabelGotoTaskInfo> &task_info)
    : TaskRepeater<LabelGotoTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      label_(nullptr),
      index_value_(nullptr) {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
    return;
  }
  auto stream_list = model_context.stream_list();
  auto label_list = model_context.label_list();
  rt_model_handle_ = model_context.rt_model_handle();
  uint32_t stream_id = task_info->stream_id();
  label_id_ = task_info->label_id();
  GELOGI("Stream list size:%zu, stream id:%u.", stream_list.size(), stream_id);
  GELOGI("Label list size:%zu, label id:%u.", label_list.size(), label_id_);
  if (stream_id >= stream_list.size() || label_id_ >= label_list.size()) {
    GELOGW("Stream/Label id invalid.");
    return;
  }
  stream_ = stream_list[stream_id];
  label_ = label_list[label_id_];
  label_manager_ = LabelManager::GetInstance();
  if (label_manager_ == nullptr) {
    GELOGW("Get label manager instance failed.");
    return;
  }
  label_info_ = label_manager_->GetLabelInfo(rt_model_handle_, label_id_, label_);
}

LabelGotoTask::~LabelGotoTask() {
  if (index_value_ != nullptr) {
    rtError_t rt_ret = rtFree(index_value_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtFree index_value_ failed! ret: 0x%X.", rt_ret);
    }
    index_value_ = nullptr;
  }
}

bool LabelGotoTask::Distribute() {
  GELOGI("LabelGotoTask Distribute start.");
  if (stream_ == nullptr) {
    GELOGE(PARAM_INVALID, "stream is null!");
    return false;
  }
  if (label_ == nullptr) {
    GELOGE(PARAM_INVALID, "label is null!");
    return false;
  }

  if (label_info_ == nullptr) {
    GELOGE(PARAM_INVALID, "label info is null!");
    return false;
  }

  if (index_value_ == nullptr) {
    rtError_t rt_ret = rtMalloc(&index_value_, sizeof(uint64_t), RT_MEMORY_HBM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }

    uint64_t index = 0;
    rt_ret = rtMemcpy(index_value_, sizeof(uint64_t), &index, sizeof(index), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }
  }

  void *label_info = label_info_->GetLabelInfo();
  rtError_t rt_ret = rtLabelSwitchByIndex(index_value_, 1, label_info, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  GELOGI("DistributeTask end.");
  return true;
}

LabelGotoTask::LabelGuard::~LabelGuard() {
  void *label_info = GetLabelInfo();
  if (label_info != nullptr) {
    rtError_t rt_ret = rtFree(label_info);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtFree label_info failed! ret: 0x%X.", rt_ret);
    }
  }
}

std::shared_ptr<LabelGotoTask::LabelManager> LabelGotoTask::LabelManager::GetInstance() {
  std::lock_guard<std::mutex> lock(instance_mutex_);
  auto instance = instance_.lock();
  if (instance != nullptr) {
    return instance;
  }

  instance = std::make_shared<LabelManager>();
  instance_ = instance;
  return instance;
}

std::shared_ptr<LabelGotoTask::LabelGuard> LabelGotoTask::LabelManager::GetLabelInfo(rtModel_t model, uint32_t label_id,
                                                                                     void *label) {
  std::lock_guard<std::mutex> lock(model_info_mapping_mutex_);
  rtError_t rt_ret;
  auto model_iter = model_info_mapping_.find(model);
  if (model_iter == model_info_mapping_.end()) {
    model_info_mapping_.emplace(model, std::map<uint32_t, std::weak_ptr<LabelGuard>>());
    model_iter = model_info_mapping_.find(model);
  }

  std::map<uint32_t, std::weak_ptr<LabelGuard>> &label_map = model_iter->second;
  auto label_iter = label_map.find(label_id);
  if (label_iter != label_map.end()) {
    auto label_guard = label_iter->second.lock();
    if (label_guard != nullptr) {
      GELOGI("model %p find same label id.", model, label_id);
      return label_guard;
    }
  }

  GELOGI("Alloc label id %u for model %p.", label_id, model);
  void *label_info;
  std::vector<void *> label_list = {label};
  uint32_t label_info_size = sizeof(rtLabelDevInfo) * label_list.size();
  rt_ret = rtMalloc(&label_info, label_info_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return nullptr;
  }

  rt_ret = rtLabelListCpy(label_list.data(), label_list.size(), label_info, label_info_size);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return nullptr;
  }

  auto label_guard = std::make_shared<LabelGuard>(label_info);
  label_map.emplace(label_id, label_guard);
  return label_guard;
}
REGISTER_TASK(TaskInfoType::LABEL_GOTO, LabelGotoTask, LabelGotoTaskInfo);

}  // namespace model_runner
}  // namespace ge
