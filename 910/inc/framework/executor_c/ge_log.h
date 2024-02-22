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
#ifndef INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_
#define INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#if !defined (LITEOS_PRINT)
#include "toolchain/slog.h"
#endif
#include "mmpa_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GE_GET_ERRORNO_STR "GE_ERRORNO_STR"
#define GE_GET_ERROR_LOG_HEADER "[GE][MODULE]"
#define GE_MODULE_NAME ((int32_t)GE)

#ifndef FAILED
#define FAILED (-1)
#endif
// trace status of log
enum TraceStatus { TRACE_INT = 0, TRACE_RUNNING, TRACE_WAITTING, TRACE_STOP };

static inline uint64_t GetTid(void) {
  const uint64_t tid = (uint64_t)(mmGetTaskId());
  return tid;
}

#ifdef LITEOS_PRINT
#define GELOGD(fmt, ...)                                                                                 \
  do {                                                                                                   \
    drv_uart_send("[DEBUG][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)

#define GELOGI(fmt, ...)                                                                                \
  do {                                                                                                  \
    drv_uart_send("[INFO][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)

#define GELOGW(fmt, ...)                                                                                   \
  do {                                                                                                     \
    drv_uart_send("[WARNING][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)

#define GELOGE(ERROR_CODE, fmt, ...)                                                                    \
  do {                                                                                                  \
    drv_uart_send("[ERROR][%s:%d]%ld ErrorNo:%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(),    \
      (long int)ERROR_CODE, ##__VA_ARGS__);                                                             \
  } while (false)

#define GEEVENT(fmt, ...)                                                                                \
  do {                                                                                                   \
    drv_uart_send("[EVENT][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)
#elif defined(RUN_TEST)
#define GELOGD(fmt, ...)                                                                          \
do {                                                                                              \
    printf("[DEBUG][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)

#define GELOGI(fmt, ...)                                                                         \
  do {                                                                                           \
    printf("[INFO][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)

#define GELOGW(fmt, ...)                                                                            \
  do {                                                                                              \
    printf("[WARNING][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)

#define GELOGE(ERROR_CODE, fmt, ...)                                                                                \
  do {                                                                                                              \
    printf("[ERROR][%s:%d]%ld ErrorNo:%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), (long int)ERROR_CODE, \
           ##__VA_ARGS__);                                                                                          \
  } while (false)

#define GEEVENT(fmt, ...)                                                                         \
  do {                                                                                            \
    printf("[EVENT][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)
#else
#define GELOGD(fmt, ...)                                                                           \
  do {                                                                                             \
    DlogDebugInner(GE_MODULE_NAME, "[DEBUG][%s:%d]%ld " fmt "\n", __FILE__, __LINE__,              \
                   (long int)GetTid(), ##__VA_ARGS__);                                             \
  } while (false)

#define GELOGI(fmt, ...)                                                                           \
  do {                                                                                             \
    DlogInfoInner(GE_MODULE_NAME, "[INFO][%s:%d]%ld " fmt "\n", __FILE__, __LINE__,                \
                  (long int)GetTid(), ##__VA_ARGS__);                                              \
  } while (false)

#define GELOGW(fmt, ...)                                                                           \
  do {                                                                                             \
    DlogWarnInner(GE_MODULE_NAME, "[WARNING][%s:%d]%ld " fmt "\n", __FILE__, __LINE__,             \
                  (long int)GetTid(), ##__VA_ARGS__);                                              \
  } while (false)

#define GEEVENT(fmt, ...)                                                                          \
  do {                                                                                             \
    DlogEventInner(GE_MODULE_NAME, "[Event][%s:%d]%ld " fmt "\n", __FILE__, __LINE__,              \
                   (long int)GetTid(), ##__VA_ARGS__);                                             \
  } while (false)

#define GELOGE(ERROR_CODE, fmt, ...)                                                               \
  do {                                                                                             \
    DlogErrorInner(GE_MODULE_NAME, "[ERROR][%s:%d]%ld:ErrorNo:%u(%s)%s " fmt "\n",                 \
                   __FILE__, __LINE__, (long int)GetTid(), (uint32_t)ERROR_CODE,                   \
                   GE_GET_ERRORNO_STR, GE_GET_ERROR_LOG_HEADER, ##__VA_ARGS__);                    \
  } while (false)
#endif

#ifdef __cplusplus
}
#endif
#endif  // INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_
