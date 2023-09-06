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
#include "toolchain/slog.h"
#include <unistd.h>
#include <sys/syscall.h>

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

static inline uint64_t GetTid() {
  const uint64_t tid = (uint64_t)(syscall(__NR_gettid));
  return tid;
}

static inline bool IsLogEnable(const int32_t module_name, const int32_t log_level) {
  const int32_t enable = CheckLogLevel(module_name, log_level);
  return (enable == 1);
}

#define LOGE(fmt, ...)                                                      \
  do {                                                                      \
    printf("[ERROR]GE %s:%d:" fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
  } while (false)

#ifdef RUN_TEST
#define GELOGD(fmt, ...)                                                                          \
  do {                                                                                            \
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
#define GELOGD(fmt, ...)                                                       \
  do {                                                                         \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {                             \
      dlog_debug(GE_MODULE_NAME, "%" PRIu64 ":" fmt, GetTid(), ##__VA_ARGS__); \
    }                                                                          \
  } while (false)

#define GELOGI(fmt, ...)                                                      \
  do {                                                                        \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {                             \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 ":" fmt, GetTid(), ##__VA_ARGS__); \
    }                                                                         \
  } while (false)

#define GELOGW(fmt, ...)                                                      \
  do {                                                                        \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_WARN)) {                             \
      dlog_warn(GE_MODULE_NAME, "%" PRIu64 ":" fmt, GetTid(), ##__VA_ARGS__); \
    }                                                                         \
  } while (false)

#define GELOGE(ERROR_CODE, fmt, ...)                                                                       \
  do {                                                                                                     \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {                                                         \
      dlog_error(GE_MODULE_NAME, "%" PRIu64 ":ErrorNo:%" PRIuLEAST8 "(%s)%s " fmt, GetTid(), (ERROR_CODE), \
                 GE_GET_ERRORNO_STR, GE_GET_ERROR_LOG_HEADER, ##__VA_ARGS__);                              \
    }                                                                                                      \
  } while (false)

#define GEEVENT(fmt, ...)                                                                                           \
  do {                                                                                                              \
    dlog_event((uint32_t)(RUN_LOG_MASK) | (uint32_t)(GE_MODULE_NAME), "%" PRIu64 ":" fmt, GetTid(), ##__VA_ARGS__); \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {                                                                  \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 ":" fmt, GetTid(), ##__VA_ARGS__);                                       \
    }                                                                                                               \
  } while (false)
#endif

#ifdef __cplusplus
}
#endif
#endif  // INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_
