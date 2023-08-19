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
#ifndef GE_COMMON_PRELOAD_PRE_DAVINCI_MODEL_H_
#define GE_COMMON_PRELOAD_PRE_DAVINCI_MODEL_H_
#include "common/preload/model/pre_model_types.h"
#include "common/preload/model/pre_model_utils.h"
#include "framework/common/util.h"
#include "external/ge/ge_api_types.h"
#include "proto/task.pb.h"
#include "common/model/ge_model.h"

namespace ge {
class PreDavinciModel {
 public:
  PreDavinciModel() = default;
  virtual ~PreDavinciModel() = default;
  void Assign(const GeModelPtr &ge_model);
  virtual Status Init();
  virtual Status DoPartitionProcess();
  Status DoTaskSink(const EngineType engine_type);
  Status InitNodes(const ComputeGraphPtr &compute_graph);
  void InitKernelOffset();
  void InitRuntimeParams();
  void DoReset() const;

 private:
  // get Op
  OpDescPtr GetOpByIndex(const uint32_t op_index) const;
  Status GetEngineName(const EngineType engine_type, const uint32_t task_type,
                       const uint32_t kernel_type, std::string &engine_name) const;
  std::string GetEngineNameByType(const uint32_t type,
                                  const std::map<uint32_t, std::string> type_to_engine_name) const;

 protected:
  GeModelPtr ge_model_;
  std::map<int64_t, OpDescPtr> op_list_;
  uint32_t model_id_{0U};
  uint32_t huge_stream_size_{0U};
  uint32_t task_num_{0U};
  PreRuntimeParam runtime_param_;
  std::unordered_map<std::string, uint32_t> names_to_bin_offset_;
  std::unordered_map<int64_t, uint32_t> zero_copy_offset_to_ids_;
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_PRE_DAVINCI_MODEL_H_