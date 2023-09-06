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

#ifndef ACLDVPP_BASE_H_
#define ACLDVPP_BASE_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t acldvppStatus;

#define ACLDVPP_SUCCESS 0
#define ACLDVPP_ERR_PARAM_NULLPTR 106001
#define ACLDVPP_ERR_PARAM_INVALID 106002
#define ACLDVPP_ERR_RUNTIME_ERROR 306001

#define ACLDVPP_ERR_INNER 506000
#define ACLDVPP_ERR_INNER_CREATE_EXECUTOR 506101
#define ACLDVPP_ERR_INNER_NOT_TRANS_EXECUTOR 506102
#define ACLDVPP_ERR_INNER_NULLPTR 506103

#ifdef __cplusplus
}
#endif
#endif // ACLDVPP_BASE_H_