/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef COMMON_GRAPH_DEBUG_GE_LOG_H_
#define COMMON_GRAPH_DEBUG_GE_LOG_H_

#include "graph/ge_error_codes.h"
#include "framework/common/debug/ge_log.h"

#define GE_LOGE(...) GE_LOG_ERROR(GE_MODULE_NAME, ge::FAILED, __VA_ARGS__)

#define GE_LOGI_IF(condition, ...) \
  if ((condition)) {               \
    GELOGI(__VA_ARGS__);           \
  }

#define GE_LOGW_IF(condition, ...) \
  if ((condition)) {               \
    GELOGW(__VA_ARGS__);           \
  }

#define GE_LOGE_IF(condition, ...)          \
  if ((condition)) {                        \
    GELOGE(ge::FAILED, __VA_ARGS__);        \
  }

#define GE_CHK_STATUS_RET_NOLOG(expr)       \
  do {                                      \
    const ge::graphStatus _status = (expr); \
    if (ge::SUCCESS != _status) {           \
      return _status;                       \
    }                                       \
  } while (0)

#define GE_CHK_BOOL_RET_STATUS(expr, _status, ...) \
  do {                                             \
    bool b = (expr);                               \
    if (!b) {                                      \
      GELOGE(ge::FAILED, __VA_ARGS__);             \
      return _status;                              \
    }                                              \
  } while (0)

#define GE_CHK_BOOL_EXEC_NOLOG(expr, exec_expr) \
  {                                             \
    bool b = (expr);                            \
    if (!b) {                                   \
      exec_expr;                                \
    }                                           \
  }

#define GE_IF_BOOL_EXEC(expr, exec_expr) \
  {                                      \
    if (expr) {                          \
      exec_expr;                         \
    }                                    \
  }

#define GE_RETURN_WITH_LOG_IF_ERROR(expr, ...) \
  do {                                         \
    const ge::graphStatus _status = (expr);    \
    if (_status) {                             \
      GELOGE(ge::FAILED, __VA_ARGS__);         \
      return _status;                          \
    }                                          \
  } while (0)

// If expr is true, the log is printed and a custom statement is executed
#define GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(expr, exec_expr, ...) \
  {                                                          \
    bool b = (expr);                                         \
    if (b) {                                                 \
      GELOGE(ge::FAILED, __VA_ARGS__);                       \
      exec_expr;                                             \
    }                                                        \
  }

// Only check error log
#define GE_CHK_BOOL_ONLY_LOG(expr, ...) \
  do {                                  \
    bool b = (expr);                    \
    if (!b) {                           \
      GELOGI(__VA_ARGS__);              \
    }                                   \
  } while (0)

// If expr is not true, do not print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS_NOLOG(expr, _status, ...) \
  do {                                                   \
    bool b = (expr);                                     \
    if (!b) {                                            \
      return _status;                                    \
    }                                                    \
  } while (0)

// If expr is not true, the log is printed and a custom statement is executed
#define GE_CHK_BOOL_EXEC(expr, exec_expr, ...)           \
  {                                                      \
    bool b = (expr);                                     \
    if (!b) {                                            \
      GELOGE(ge::FAILED, __VA_ARGS__);                   \
      exec_expr;                                         \
    }                                                    \
  }

// If expr is not true, the log is printed and a custom statement is executed
#define GE_CHK_BOOL_EXEC_INFO(expr, exec_expr, ...)     \
  {                                                     \
    bool b = (expr);                                    \
    if (!b) {                                           \
      GELOGI(__VA_ARGS__);                              \
      exec_expr;                                        \
    }                                                   \
  }

// If expr is not GRAPH_SUCCESS, print the log and return the same value
#define GE_CHK_STATUS_RET(expr, ...)                          \
  do {                                                        \
    const ge::graphStatus _status = (expr);                   \
    if (ge::SUCCESS != _status) {                             \
      GELOGE(ge::FAILED, __VA_ARGS__);                        \
      return _status;                                         \
    }                                                         \
  } while (0)

#define GE_MAKE_SHARED(exec_expr0, exec_expr1)                \
  try {                                                       \
    exec_expr0;                                               \
  } catch (...) {                                             \
    GELOGE(ge::FAILED, "Make shared failed");                 \
    exec_expr1;                                               \
  }

#endif  // COMMON_GRAPH_DEBUG_GE_LOG_H_

