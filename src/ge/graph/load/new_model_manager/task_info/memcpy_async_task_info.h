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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ASYNC_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ASYNC_TASK_INFO_H_

#include "graph/load/new_model_manager/task_info/task_info.h"
#include "graph/op_desc.h"

namespace ge {
class MemcpyAsyncTaskInfo : public TaskInfo {
 public:
  MemcpyAsyncTaskInfo() : dst_(nullptr), dst_max_(0), src_(nullptr), count_(0), kind_(0), memory_4g_(nullptr) {}

  ~MemcpyAsyncTaskInfo() override {
    src_ = nullptr;
    dst_ = nullptr;

    if (memory_4g_ != nullptr) {
      rtError_t ret = rtFree(memory_4g_);
      if (ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", ret);
      }
      memory_4g_ = nullptr;
    }
  }

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  Status UpdateArgs() override;

  Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

 private:
  Status AllocTsMemoryForMemcpy(const OpDescPtr &op_desc, DavinciModel *davinci_model);
  uint8_t *dst_;
  uint64_t dst_max_;
  uint8_t *src_;
  uint64_t count_;
  uint32_t kind_;
  DavinciModel *davinci_model_ = nullptr;
  uint32_t args_offset_ = 0;
  domi::MemcpyAsyncDef memcpy_async;
  void *memory_4g_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ASYNC_TASK_INFO_H_
