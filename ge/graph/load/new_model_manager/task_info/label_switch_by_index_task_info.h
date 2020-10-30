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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_LABEL_SWITCH_BY_INDEX_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_LABEL_SWITCH_BY_INDEX_TASK_INFO_H_

#include "graph/load/new_model_manager/task_info/task_info.h"

namespace ge {
class LabelSwitchByIndexTaskInfo : public TaskInfo {
 public:
  LabelSwitchByIndexTaskInfo()
      : index_value_(nullptr), branch_max_(0), args_(nullptr), args_size_(0), fixed_addr_offset_(0) {}

  ~LabelSwitchByIndexTaskInfo() override;

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

 private:
  void *index_value_;    // switch index input.
  uint32_t branch_max_;  // max branch count.
  void *args_;           // label info memory.
  uint32_t args_size_;   // label info length.
  std::vector<rtLabel_t> label_list_;
  int64_t fixed_addr_offset_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_LABEL_SWITCH_BY_INDEX_TASK_INFO_H_