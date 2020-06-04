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

#ifndef INC_FRAMEWORK_COMMON_DEBUG_LOG_H_
#define INC_FRAMEWORK_COMMON_DEBUG_LOG_H_

#include <string>

#include "cce/cce_def.hpp"
#include "common/string_util.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "ge/ge_api_error_codes.h"

using cce::CC_STATUS_SUCCESS;
using cce::ccStatus_t;

#if !defined(__ANDROID__) && !defined(ANDROID)
#define DOMI_LOGE(...) GE_LOG_ERROR(GE_MODULE_NAME, ge::FAILED, __VA_ARGS__)
#else
#include <android/log.h>
#if defined(BUILD_VERSION_PERF)
#define DOMI_LOGE(fmt, ...)
#else
// The Android system has strict log control. Do not modify the log.
#define DOMI_LOGE(fmt, ...) \
  __android_log_print(ANDROID_LOG_ERROR, "NPU_FMK", "%s %s(%d)::" #fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif
#endif

// ge marco
#define GE_LOGI_IF(condition, ...) \
  if ((condition)) {               \
    GELOGI(__VA_ARGS__);           \
  }

#define GE_LOGW_IF(condition, ...) \
  if ((condition)) {               \
    GELOGW(__VA_ARGS__);           \
  }

#define GE_LOGE_IF(condition, ...) \
  if ((condition)) {               \
    DOMI_LOGE(__VA_ARGS__);        \
  }

// If expr is not SUCCESS, print the log and return the same value
#define GE_CHK_STATUS_RET(expr, ...)   \
  do {                                 \
    const ge::Status _status = (expr); \
    if (_status != ge::SUCCESS) {      \
      DOMI_LOGE(__VA_ARGS__);          \
      return _status;                  \
    }                                  \
  } while (0);

// If expr is not SUCCESS, print the log and do not execute return
#define GE_CHK_STATUS(expr, ...)       \
  do {                                 \
    const ge::Status _status = (expr); \
    if (_status != ge::SUCCESS) {      \
      DOMI_LOGE(__VA_ARGS__);          \
    }                                  \
  } while (0);

// If expr is not SUCCESS, return the same value
#define GE_CHK_STATUS_RET_NOLOG(expr)  \
  do {                                 \
    const ge::Status _status = (expr); \
    if (_status != ge::SUCCESS) {      \
      return _status;                  \
    }                                  \
  } while (0);

// If expr is not GRAPH_SUCCESS, print the log and return FAILED
#define GE_CHK_GRAPH_STATUS_RET(expr, ...) \
  do {                                     \
    if ((expr) != ge::GRAPH_SUCCESS) {     \
      DOMI_LOGE(__VA_ARGS__);              \
      return FAILED;                       \
    }                                      \
  } while (0);

// If expr is not SUCCESS, print the log and execute a custom statement
#define GE_CHK_STATUS_EXEC(expr, exec_expr, ...)                  \
  do {                                                            \
    const ge::Status _status = (expr);                            \
    GE_CHK_BOOL_EXEC(_status == SUCCESS, exec_expr, __VA_ARGS__); \
  } while (0);

// If expr is not true, print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS(expr, _status, ...)                                                         \
  do {                                                                                                     \
    bool b = (expr);                                                                                       \
    if (!b) {                                                                                              \
      std::string msg;                                                                                     \
      (void)msg.append(ge::StringUtils::FormatString(__VA_ARGS__));                                        \
      (void)msg.append(                                                                                    \
        ge::StringUtils::FormatString(" Error Code:0x%X(%s)", _status, GET_ERRORNO_STR(_status).c_str())); \
      DOMI_LOGE("%s", msg.c_str());                                                                        \
      return _status;                                                                                      \
    }                                                                                                      \
  } while (0);

// If expr is not true, print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS_NOLOG(expr, _status, ...) \
  do {                                                   \
    bool b = (expr);                                     \
    if (!b) {                                            \
      return _status;                                    \
    }                                                    \
  } while (0);

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_EXEC(expr, exec_expr, ...) \
  {                                            \
    bool b = (expr);                           \
    if (!b) {                                  \
      DOMI_LOGE(__VA_ARGS__);                  \
      exec_expr;                               \
    }                                          \
  };

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_EXEC_WARN(expr, exec_expr, ...) \
  {                                                 \
    bool b = (expr);                                \
    if (!b) {                                       \
      GELOGW(__VA_ARGS__);                          \
      exec_expr;                                    \
    }                                               \
  };
// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_EXEC_INFO(expr, exec_expr, ...) \
  {                                                 \
    bool b = (expr);                                \
    if (!b) {                                       \
      GELOGI(__VA_ARGS__);                          \
      exec_expr;                                    \
    }                                               \
  };

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_TRUE_EXEC_INFO(expr, exec_expr, ...) \
  {                                                      \
    bool b = (expr);                                     \
    if (b) {                                             \
      GELOGI(__VA_ARGS__);                               \
      exec_expr;                                         \
    }                                                    \
  };

// If expr is true, print logs and execute custom statements
#define GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(expr, exec_expr, ...) \
  {                                                          \
    bool b = (expr);                                         \
    if (b) {                                                 \
      DOMI_LOGE(__VA_ARGS__);                                \
      exec_expr;                                             \
    }                                                        \
  };
// If expr is true, print the Information log and execute a custom statement
#define GE_CHK_TRUE_EXEC_INFO(expr, exec_expr, ...) \
  {                                                 \
    bool b = (expr);                                \
    if (b) {                                        \
      GELOGI(__VA_ARGS__);                          \
      exec_expr;                                    \
    }                                               \
  };

// If expr is not SUCCESS, print the log and execute the expression + return
#define GE_CHK_BOOL_TRUE_RET_VOID(expr, exec_expr, ...) \
  {                                                     \
    bool b = (expr);                                    \
    if (b) {                                            \
      DOMI_LOGE(__VA_ARGS__);                           \
      exec_expr;                                        \
      return;                                           \
    }                                                   \
  };

// If expr is not SUCCESS, print the log and execute the expression + return _status
#define GE_CHK_BOOL_TRUE_EXEC_RET_STATUS(expr, _status, exec_expr, ...) \
  {                                                                     \
    bool b = (expr);                                                    \
    if (b) {                                                            \
      DOMI_LOGE(__VA_ARGS__);                                           \
      exec_expr;                                                        \
      return _status;                                                   \
    }                                                                   \
  };

// If expr is not true, execute a custom statement
#define GE_CHK_BOOL_EXEC_NOLOG(expr, exec_expr) \
  {                                             \
    bool b = (expr);                            \
    if (!b) {                                   \
      exec_expr;                                \
    }                                           \
  };

// -----------------runtime related macro definitions-------------------------------
// If expr is not RT_ERROR_NONE, print the log
#define GE_CHK_RT(expr)                                    \
  do {                                                     \
    rtError_t _rt_ret = (expr);                            \
    if (_rt_ret != RT_ERROR_NONE) {                        \
      DOMI_LOGE("Call rt api failed, ret: 0x%X", _rt_ret); \
    }                                                      \
  } while (0);

// If expr is not RT_ERROR_NONE, print the log and execute the exec_expr expression
#define GE_CHK_RT_EXEC(expr, exec_expr)                    \
  {                                                        \
    rtError_t _rt_ret = (expr);                            \
    if (_rt_ret != RT_ERROR_NONE) {                        \
      DOMI_LOGE("Call rt api failed, ret: 0x%X", _rt_ret); \
      exec_expr;                                           \
    }                                                      \
  }

// If expr is not RT_ERROR_NONE, print the log and return
#define GE_CHK_RT_RET(expr)                                \
  do {                                                     \
    rtError_t _rt_ret = (expr);                            \
    if (_rt_ret != RT_ERROR_NONE) {                        \
      DOMI_LOGE("Call rt api failed, ret: 0x%X", _rt_ret); \
      return ge::RT_FAILED;                                \
    }                                                      \
  } while (0);

// ------------------------cce related macro definitions----------------------------
// If expr is not CC_STATUS_SUCCESS, print the log
#define GE_CHK_CCE(expr)                                    \
  do {                                                      \
    ccStatus_t _cc_ret = (expr);                            \
    if (_cc_ret != CC_STATUS_SUCCESS) {                     \
      DOMI_LOGE("Call cce api failed, ret: 0x%X", _cc_ret); \
    }                                                       \
  } while (0);

// If expr is true, execute exec_expr without printing logs
#define GE_IF_BOOL_EXEC(expr, exec_expr) \
  {                                      \
    if (expr) {                          \
      exec_expr;                         \
    }                                    \
  }

// If make_shared is abnormal, print the log and execute the statement
#define GE_MAKE_SHARED(exec_expr0, exec_expr1) \
  try {                                        \
    exec_expr0;                                \
  } catch (const std::bad_alloc &) {           \
    DOMI_LOGE("Make shared failed");           \
    exec_expr1;                                \
  }

#endif  // INC_FRAMEWORK_COMMON_DEBUG_LOG_H_
