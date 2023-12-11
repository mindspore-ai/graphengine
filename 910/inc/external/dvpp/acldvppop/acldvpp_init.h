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

#ifndef ACLDVPP_INIT_H_
#define ACLDVPP_INIT_H_

#include "acldvpp_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief DVPP初始化函数，同步接口。
* @param [in] configPath: 预留参数，配置文件所在路径的指针，包含文件名，当前需要配置为null
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppInit(const char *configPath);

/**
* @brief DVPP去初始化函数，同步接口。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppFinalize();

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_INIT_H_