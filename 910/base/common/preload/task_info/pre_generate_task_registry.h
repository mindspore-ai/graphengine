/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef GE_COMMON_PRELOAD_TASK_INFO_PRE_GENERATE_TASK_REGISTRY_H_
#define GE_COMMON_PRELOAD_TASK_INFO_PRE_GENERATE_TASK_REGISTRY_H_
#include <string>
#include <functional>
#include <vector>
#include "common/preload/task_info/pre_task_status.h"
#include "common/preload/model/pre_model_utils.h"

namespace ge {
struct PreTaskInput {
  PreRuntimeParam rts_param;
  std::unordered_map<std::string, uint32_t> names_to_bin_offset;
  std::unordered_map<int64_t, uint32_t> zero_copy_offset_to_id;
};
struct PreTaskResult {
  PreTaskStatus status;
  std::vector<PreTaskDescInfo> pre_task_desc_infos;
};
class PreGenerateTaskRegistry {
 public:
  using PreGenerateTask = PreTaskResult (*)(const domi::TaskDef &task_def, const OpDescPtr &op_desc,
                                            const PreTaskInput &pre_task_input);
  static PreGenerateTaskRegistry &GetInstance();
  PreGenerateTask FindPreGenerateTask(const std::string &func_name);
  void Register(const std::string &func_name, PreGenerateTask func);

 private:
  std::unordered_map<std::string, PreGenerateTask> names_to_register_task_;
};

class PreGenerateTaskRegister {
 public:
  PreGenerateTaskRegister(const std::string &func_name, PreGenerateTaskRegistry::PreGenerateTask func) noexcept;
};
}  // namespace ge

#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif

#define GE_REFISTER_PRE_GENERATE_TASK(type, func, counter)                                                             \
  static const ge::PreGenerateTaskRegister g_register_pre_generate_task_##counter ATTRIBUTE_USED =                     \
      ge::PreGenerateTaskRegister((type), (func))
#define REFISTER_PRE_GENERATE_TASK(type, func)                                                                         \
  GE_REFISTER_PRE_GENERATE_TASK((type), (func), __COUNTER__)

#endif  // GE_COMMON_PRELOAD_PRE_GENERATE_TASK_REGISTRY_H_