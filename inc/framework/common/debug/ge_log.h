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

#define GE_MODULE_NAME GE

// trace status of log
enum TraceStatus { TRACE_INIT = 0, TRACE_RUNNING, TRACE_WAITING, TRACE_STOP };

#define GELOGE(ERROR_CODE, ...) GE_LOG_ERROR(GE_MODULE_NAME, ERROR_CODE, __VA_ARGS__)
#define GELOGW(...) GE_LOG_WARN(GE_MODULE_NAME, __VA_ARGS__)
#define GELOGI(...) GE_LOG_INFO(GE_MODULE_NAME, __VA_ARGS__)
#define GELOGD(...) GE_LOG_DEBUG(GE_MODULE_NAME, __VA_ARGS__)
#define GEEVENT(...) GE_LOG_EVENT(GE_MODULE_NAME, __VA_ARGS__)
#define GELOGO(...) GE_LOG_OPLOG(GE_MODULE_NAME, __VA_ARGS__)
#define GELOGT(VALUE, ...) GE_LOG_TRACE(GE_MODULE_NAME, VALUE, __VA_ARGS__)

inline bool IsLogEnable(int module_name, int log_level) noexcept {
  int32_t enable_event = 0;
  int32_t dlog_level = dlog_getlevel(module_name, &enable_event);
  if (dlog_level <= log_level) {
    return true;
  }
  return false;
}

/*lint --emacro((773),GE_TIMESTAMP_START)*/
/*lint -esym(773,GE_TIMESTAMP_START)*/
#define GE_TIMESTAMP_START(stage) uint64_t startUsec_##stage = ge::GetCurrentTimestap()

#define GE_TIMESTAMP_END(stage, stage_name)                                           \
  do {                                                                                \
    uint64_t endUsec_##stage = ge::GetCurrentTimestap();                              \
    GEEVENT("[GEPERFTRACE] The time cost of %s is [%lu] micro second.", (stage_name), \
            (endUsec_##stage - startUsec_##stage));                                   \
  } while (0);

#define GE_TIMESTAMP_CALLNUM_START(stage)                \
  uint64_t startUsec_##stage = ge::GetCurrentTimestap(); \
  uint64_t call_num_of##stage = 0;                       \
  uint64_t time_of##stage = 0

#define GE_TIMESTAMP_RESTART(stage) (startUsec_##stage = ge::GetCurrentTimestap())

#define GE_TIMESTAMP_ADD(stage)                                   \
  time_of##stage += ge::GetCurrentTimestap() - startUsec_##stage; \
  call_num_of##stage++

#define GE_TIMESTAMP_CALLNUM_END(stage, stage_name)                                                                 \
  GEEVENT("[GEPERFTRACE] The time cost of %s is [%lu] micro second, call num is %lu", (stage_name), time_of##stage, \
          call_num_of##stage)

#define GE_LOG_ERROR(MOD_NAME, ERROR_CODE, fmt, ...)                                           \
  dlog_error(static_cast<int>(MOD_NAME), "%s: ErrorNo: %d(%s) " fmt, __FUNCTION__, ERROR_CODE, \
             ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()), ##__VA_ARGS__)
#define GE_LOG_WARN(MOD_NAME, fmt, ...)                   \
  if (IsLogEnable(static_cast<int>(MOD_NAME), DLOG_WARN)) \
  dlog_warn(static_cast<int>(MOD_NAME), "%s:" fmt, __FUNCTION__, ##__VA_ARGS__)
#define GE_LOG_INFO(MOD_NAME, fmt, ...)                   \
  if (IsLogEnable(static_cast<int>(MOD_NAME), DLOG_INFO)) \
  dlog_info(static_cast<int>(MOD_NAME), "%s:" fmt, __FUNCTION__, ##__VA_ARGS__)
#define GE_LOG_DEBUG(MOD_NAME, fmt, ...)                   \
  if (IsLogEnable(static_cast<int>(MOD_NAME), DLOG_DEBUG)) \
  dlog_debug(static_cast<int>(MOD_NAME), "%s:" fmt, __FUNCTION__, ##__VA_ARGS__)
#define GE_LOG_EVENT(MOD_NAME, fmt, ...) dlog_event(static_cast<int>(MOD_NAME), "%s:" fmt, __FUNCTION__, ##__VA_ARGS__)
#define GE_LOG_OPLOG(MOD_NAME, fmt, ...) \
  Dlog(static_cast<int>(MOD_NAME), DLOG_OPLOG, "%s:" fmt, __FUNCTION__, ##__VA_ARGS__)
#define GE_LOG_TRACE(MOD_NAME, value, fmt, ...)                                                         \
  do {                                                                                                  \
    TraceStatus stat = value;                                                                           \
    const char *const TraceStatStr[] = {"INIT", "RUNNING", "WAITING", "STOP"};                          \
    int idx = static_cast<int>(stat);                                                                   \
    char *k = const_cast<char *>("status");                                                             \
    char *v = const_cast<char *>(TraceStatStr[idx]);                                                    \
    KeyValue kv = {k, v};                                                                               \
    DlogWithKV(static_cast<int>(MOD_NAME), DLOG_TRACE, &kv, 1, "%s:" fmt, __FUNCTION__, ##__VA_ARGS__); \
  } while (0)
#endif  // INC_FRAMEWORK_COMMON_DEBUG_GE_LOG_H_
