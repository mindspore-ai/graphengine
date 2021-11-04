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

#ifndef CCE_RUNTIME_RT_DFX_H
#define CCE_RUNTIME_RT_DFX_H

#include "base.h"

#if defined(__cplusplus)
extern "C" {
#endif

// max task tag buffer is 1024(include '\0')
#define TASK_TAG_MAX_LEN    1024U

/**
 * @brief set task tag.
 * once set is only use by one task and thread local.
 * attention:
 *  1. it's used for dump current task in active stream now.
 *  2. it must be called be for task submit and will be invalid after task submit.
 * @param [in] taskTag  task tag, usually it's can be node name or task name.
 *                      must end with '\0' and max len is TASK_TAG_MAX_LEN.
 * @return RT_ERROR_NONE for ok
 * @return other failed
 */
RTS_API rtError_t rtSetTaskTag(const char_t *taskTag);

#if defined(__cplusplus)
}
#endif
#endif // CCE_RUNTIME_RT_DFX_H
