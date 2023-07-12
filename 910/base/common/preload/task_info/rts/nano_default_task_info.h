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
#ifndef GE_COMMON_PRELOAD_TASKINFO_RTS_NANO_DEFAULT_TASK_INFO_H_
#define GE_COMMON_PRELOAD_TASKINFO_RTS_NANO_DEFAULT_TASK_INFO_H_
#include "common/preload/task_info/pre_generate_task_registry.h"

namespace ge {
PreTaskResult GenerateDefaultTask(const domi::TaskDef &task_def, const OpDescPtr &op_desc,
                                  const PreTaskInput &pre_task_input);
}  // namespace ge

#endif  // GE_COMMON_PRELOAD_TASKINFO_RTS_NANO_DEFAULT_TASK_INFO_H_
