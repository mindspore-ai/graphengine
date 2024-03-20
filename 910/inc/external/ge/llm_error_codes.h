/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
  
#ifndef AIR_RUNTIME_LLM_ENGINE_INC_EXTERNAL_LLM_ERROR_CODES_H_
#define AIR_RUNTIME_LLM_ENGINE_INC_EXTERNAL_LLM_ERROR_CODES_H_

#include "external/ge_common/ge_api_error_codes.h"

#ifndef LLM_ERROR_CODES
#define LLM_ERROR_CODES

namespace ge {
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_WAIT_PROC_TIMEOUT, 1);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_KV_CACHE_NOT_EXIST, 2);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_REPEAT_REQUEST, 3);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_REQUEST_ALREADY_COMPLETED, 4);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_PARAM_INVALID, 5);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_ENGINE_FINALIZED, 6);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_NOT_YET_LINK, 7);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_ALREADY_LINK, 8);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_LINK_FAILED, 9);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_UNLINK_FAILED, 10);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_NOTIFY_PROMPT_UNLINK_FAILED, 11);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_CLUSTER_NUM_EXCEED_LIMIT, 12);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_PROCESSING_LINK, 13);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_DEVICE_OUT_OF_MEMORY, 14);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_PREFIX_ALREADY_EXIST, 15);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_PREFIX_NOT_EXIST, 16);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_SEQ_LEN_OVER_LIMIT, 17);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_NO_FREE_BLOCK, 18);
GE_ERRORNO_DEFINE(0b01, 0b01, 0b000, 8, 11, LLM_BLOCKS_OUT_OF_MEMORY, 19);
}  // namespace ge
#endif
#endif  // AIR_RUNTIME_LLM_ENGINE_INC_EXTERNAL_LLM_ERROR_CODES_H_
