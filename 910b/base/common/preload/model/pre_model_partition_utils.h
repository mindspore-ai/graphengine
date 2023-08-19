/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef GE_COMMON_PRELOAD_PRE_MODEL_PARTITION_UTILS_H_
#define GE_COMMON_PRELOAD_PRE_MODEL_PARTITION_UTILS_H_
#include "common/preload/model/pre_model_utils.h"
#include "common/preload/model/pre_model_types.h"

namespace ge {
struct PartitionKernelArgsParam {
  uint8_t type;
  uint64_t offset;
  uint64_t para;
};

struct TaskBuildBuf {
  std::shared_ptr<uint8_t> buf;
  uint32_t orgi_size;
  uint32_t used_size;
};

class PreModelPartitionUtils {
 public:
  PreModelPartitionUtils() = default;
  ~PreModelPartitionUtils() = default;
  static PreModelPartitionUtils &GetInstance() {
    static PreModelPartitionUtils instance;
    return instance;
  }
  void Reset();
  void AddPreTaskDescInfo(const std::vector<PreTaskDescInfo> &pre_task_desc_infos);
  Status InitTaskBuildMem(const uint32_t huge_stream_size, const uint32_t stream_size);
  Status InitTaskBuildMem(const uint32_t task_num);
  Status PreparePartitionData(const EngineType type);
  Status CheckNanoPartitionType(const uint8_t type) const;
  std::shared_ptr<TaskBuildBuf> &GetTaskBuildBuf() { return task_build_buf_; }
  std::shared_ptr<TaskBuildBuf> &GetNanoTaskBuildBuf(const uint8_t type) {
    return nano_partition_type_to_buf_[type];
  }
  void AddNanoHostFuncParamData(const std::shared_ptr<uint8_t> &nano_hostfunc_param_data);

 private:
  Status PrepareDefaultPartitionData(const PreTaskDescInfo &task_desc);
  Status PrepareNanoPartitionData(const PreTaskDescInfo &task_desc);
  Status UpdateSqeInfo(const PreTaskDescInfo &task_desc, std::shared_ptr<TaskBuildBuf> task_build_buf) const;
  Status InitTaskBuildMem(const tagRtTaskBuffType type, const uint32_t task_num);
  Status GenTaskBuildBuf(std::shared_ptr<TaskBuildBuf> &build_buf, const uint32_t orgi_size) const;
  std::vector<PreTaskDescInfo> pre_task_desc_info_;
  std::vector<KernelArgsInfo> kernel_args_info_;
  std::vector<PartitionKernelArgsParam> partition_kernel_args_param_;
  uint64_t kernel_args_info_size_ = 0UL;
  std::shared_ptr<TaskBuildBuf> task_build_buf_ = nullptr;
  std::unordered_map<uint8_t, std::shared_ptr<TaskBuildBuf>> nano_partition_type_to_buf_;
  uint32_t total_tlv_len_ = 0U;
  std::vector<std::shared_ptr<uint8_t>> nano_hostfunc_param_data_;
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_PRE_MODEL_PARTITION_UTILS_H_