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

#ifndef INC_FRAMEWORK_COMMON_DEBUG_GE_LOG_H_
#define INC_FRAMEWORK_COMMON_DEBUG_GE_LOG_H_

#include <cstdint>
#include <unistd.h>
#include <sys/syscall.h>

#include "framework/common/ge_inner_error_codes.h"
#include "toolchain/slog.h"

#define GE_MODULE_NAME static_cast<int>(GE)

// trace status of log
enum TraceStatus { TRACE_INIT = 0, TRACE_RUNNING, TRACE_WAITING, TRACE_STOP };

#define GELOGE(ERROR_CODE, ...) GE_LOG_ERROR(GE_MODULE_NAME, ERROR_CODE, __VA_ARGS__)
#define GELOGW(...) GE_LOG_WARN(GE_MODULE_NAME, __VA_ARGS__)
#define GELOGI(...) GE_LOG_INFO(GE_MODULE_NAME, __VA_ARGS__)
#define GELOGD(...) GE_LOG_DEBUG(GE_MODULE_NAME, __VA_ARGS__)
#define GEEVENT(...) GE_LOG_EVENT(GE_MODULE_NAME, __VA_ARGS__)
#define GELOGO(...) GE_LOG_OPLOG(GE_MODULE_NAME, __VA_ARGS__)
#define GELOGT(VALUE, ...) GE_LOG_TRACE(GE_MODULE_NAME, VALUE, __VA_ARGS__)

inline bool IsLogEnable(int module_name, int log_level) {
  int32_t enable = CheckLogLevel(module_name, log_level);
  // 1:enable, 0:disable
  if (enable == 1) {
    return true;
  }
  return false;
}

inline pid_t GetTid() {
  thread_local static pid_t tid = syscall(__NR_gettid);
  return tid;
}

#define GE_LOG_ERROR(MOD_NAME, ERROR_CODE, fmt, ...)                                       \
  dlog_error(MOD_NAME, "%lu %s: ErrorNo: %d(%s) " fmt, GetTid(), __FUNCTION__, ERROR_CODE, \
             ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()), ##__VA_ARGS__)
#define GE_LOG_WARN(MOD_NAME, fmt, ...) \
  if (IsLogEnable(MOD_NAME, DLOG_WARN)) dlog_warn(MOD_NAME, "%lu %s:" fmt, GetTid(), __FUNCTION__, ##__VA_ARGS__)
#define GE_LOG_INFO(MOD_NAME, fmt, ...) \
  if (IsLogEnable(MOD_NAME, DLOG_INFO)) dlog_info(MOD_NAME, "%lu %s:" fmt, GetTid(), __FUNCTION__, ##__VA_ARGS__)
#define GE_LOG_DEBUG(MOD_NAME, fmt, ...) \
  if (IsLogEnable(MOD_NAME, DLOG_DEBUG)) dlog_debug(MOD_NAME, "%lu %s:" fmt, GetTid(), __FUNCTION__, ##__VA_ARGS__)
#define GE_LOG_EVENT(MOD_NAME, fmt, ...) dlog_event(MOD_NAME, "%lu %s:" fmt, GetTid(), __FUNCTION__, ##__VA_ARGS__)
#define GE_LOG_OPLOG(MOD_NAME, fmt, ...) \
  Dlog(MOD_NAME, DLOG_OPLOG, "%lu %s:" fmt, GetTid(), __FUNCTION__, ##__VA_ARGS__)

#define GE_LOG_TRACE(MOD_NAME, value, fmt, ...)                                                                       \
  do {                                                                                                                \
    TraceStatus stat = value;                                                                                         \
    const char *const TraceStatStr[] = {"INIT", "RUNNING", "WAITING", "STOP"};                                        \
    int idx = static_cast<int>(stat);                                                                                 \
    char *k = const_cast<char *>("status");                                                                           \
    char *v = const_cast<char *>(TraceStatStr[idx]);                                                                  \
    KeyValue kv = {k, v};                                                                                             \
    DlogWithKV(static_cast<int>(MOD_NAME), DLOG_TRACE, &kv, 1, "%lu %s:" fmt, GetTid(), __FUNCTION__, ##__VA_ARGS__); \
  } while (0)

// print memory when it is greater than 1KB.
#define GE_PRINT_DYNAMIC_MEMORY(FUNC, PURPOSE, SIZE)                                                        \
  do {                                                                                                      \
    if ((SIZE) > 1024) {                                                                                    \
      GELOGI("MallocMemory, func=%s, size=%zu, purpose=%s", (#FUNC), static_cast<size_t>(SIZE), (PURPOSE)); \
    }                                                                                                       \
  } while (0);
#endif  // INC_FRAMEWORK_COMMON_DEBUG_GE_LOG_H_
