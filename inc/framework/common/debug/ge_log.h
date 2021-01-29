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

#include "framework/common/ge_inner_error_codes.h"
#include "toolchain/slog.h"
#ifdef __GNUC__
#include <unistd.h>
#include <sys/syscall.h>
#else
#include "mmpa/mmpa_api.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GE_MODULE_NAME static_cast<int>(GE)

// trace status of log
enum TraceStatus { TRACE_INIT = 0, TRACE_RUNNING, TRACE_WAITING, TRACE_STOP };

class GE_FUNC_VISIBILITY GeLog {
 public:
  static uint64_t GetTid() {
#ifdef __GNUC__
    thread_local static uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    thread_local static uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
    return tid;
  }
};

inline bool IsLogEnable(int module_name, int log_level) {
  int32_t enable = CheckLogLevel(module_name, log_level);
  // 1:enable, 0:disable
  return (enable == 1);
}

#define GELOGE(ERROR_CODE, fmt, ...)                                                                    \
  dlog_error(GE_MODULE_NAME, "%lu %s: ErrorNo: %d(%s) " fmt, GeLog::GetTid(), __FUNCTION__, ERROR_CODE, \
             ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()), ##__VA_ARGS__)
#define GELOGW(fmt, ...)                      \
  if (IsLogEnable(GE_MODULE_NAME, DLOG_WARN)) \
  dlog_warn(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), __FUNCTION__, ##__VA_ARGS__)
#define GELOGI(fmt, ...)                      \
  if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) \
  dlog_info(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), __FUNCTION__, ##__VA_ARGS__)
#define GELOGD(fmt, ...)                       \
  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) \
  dlog_debug(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), __FUNCTION__, ##__VA_ARGS__)

#define GEEVENT(fmt, ...) dlog_event(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), __FUNCTION__, ##__VA_ARGS__)

#define GELOGT(VALUE, fmt, ...)                                                                                    \
  do {                                                                                                             \
    TraceStatus stat = VALUE;                                                                                      \
    const char *const TraceStatStr[] = {"INIT", "RUNNING", "WAITING", "STOP"};                                     \
    int idx = static_cast<int>(stat);                                                                              \
    char *k = const_cast<char *>("status");                                                                        \
    char *v = const_cast<char *>(TraceStatStr[idx]);                                                               \
    KeyValue kv = {k, v};                                                                                          \
    DlogWithKV(static_cast<int>(GE_MODULE_NAME), DLOG_TRACE, &kv, 1, "%lu %s:" fmt, GeLog::GetTid(), __FUNCTION__, \
               ##__VA_ARGS__);                                                                                     \
  } while (0)

#define GE_LOG_ERROR(MOD_NAME, ERROR_CODE, fmt, ...)                                              \
  dlog_error(MOD_NAME, "%lu %s: ErrorNo: %d(%s) " fmt, GeLog::GetTid(), __FUNCTION__, ERROR_CODE, \
             ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()), ##__VA_ARGS__)

// print memory when it is greater than 1KB.
#define GE_PRINT_DYNAMIC_MEMORY(FUNC, PURPOSE, SIZE)                                                        \
  do {                                                                                                      \
    if ((SIZE) > 1024) {                                                                                    \
      GELOGI("MallocMemory, func=%s, size=%zu, purpose=%s", (#FUNC), static_cast<size_t>(SIZE), (PURPOSE)); \
    }                                                                                                       \
  } while (0);
#ifdef __cplusplus
}
#endif
#endif  // INC_FRAMEWORK_COMMON_DEBUG_GE_LOG_H_
