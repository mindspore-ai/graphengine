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

#ifndef INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_
#define INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOGI(fmt, ...)                                                                            \
  do {                                                                                            \
      printf("[INFO] GE %s:%s:%d: "#fmt "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__);       \
  } while (false)
#define LOGD(fmt, ...)                                                                            \
  do {                                                                                            \
      printf("[DEBUG] GE %s:%s:%d: "#fmt "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__);      \
  } while (false)
#define LDGW(fmt, ...)                                                                            \
  do {                                                                                            \
      printf("[WARN] GE %s:%s:%d: "#fmt "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__);       \
  } while (false)
#define LOGE(fmt, ...)                                                                            \
  do {                                                                                            \
      printf("[ERROR] GE %s:%s:%d: "#fmt "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__);      \
  } while (false)
#define EVENT(fmt, ...)                                                                           \
  do {                                                                                            \
      printf("[EVENT] GE %s:%s:%d: "#fmt "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__);      \
  } while (false)

#ifdef __cplusplus
}
#endif
#endif  // INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_
