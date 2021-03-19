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

#ifndef GE_GE_RUNTIME_TASK_LABEL_GOTO_TASK_H_
#define GE_GE_RUNTIME_TASK_LABEL_GOTO_TASK_H_

#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include "ge_runtime/task/task.h"

namespace ge {
namespace model_runner {
class LabelGotoTask : public TaskRepeater<LabelGotoTaskInfo> {
 public:
  LabelGotoTask(const ModelContext &model_context, const std::shared_ptr<LabelGotoTaskInfo> &task_info);

  ~LabelGotoTask() override;

  bool Distribute() override;

 private:
  class LabelGuard;
  class LabelManager;

  std::shared_ptr<LabelGotoTaskInfo> task_info_;
  void *stream_;
  void *label_;
  std::shared_ptr<LabelGuard> label_info_;
  void *index_value_;
  uint32_t label_id_;
  rtModel_t rt_model_handle_;
  std::shared_ptr<LabelManager> label_manager_;
};

class LabelGotoTask::LabelGuard {
 public:
  explicit LabelGuard(void *label_info) : label_info_(reinterpret_cast<uintptr_t>(label_info)) {}
  ~LabelGuard();
  void *GetLabelInfo() { return reinterpret_cast<void *>(label_info_); }

 private:
  uintptr_t label_info_;
};

class LabelGotoTask::LabelManager {
 public:
  static std::shared_ptr<LabelManager> GetInstance();
  std::shared_ptr<LabelGuard> GetLabelInfo(rtModel_t model, uint32_t label_id, void *label);

 private:
  std::mutex model_info_mapping_mutex_;
  std::map<rtModel_t, std::map<uint32_t, std::weak_ptr<LabelGuard>>> model_info_mapping_;

  static std::weak_ptr<LabelManager> instance_;
  static std::mutex instance_mutex_;
};
}  // namespace model_runner
}  // namespace ge

#endif  // GE_GE_RUNTIME_TASK_LABEL_GOTO_TASK_H_
