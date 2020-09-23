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

#ifndef COMMON_GRAPH_DEBUG_GE_UTIL_H_
#define COMMON_GRAPH_DEBUG_GE_UTIL_H_

#include <limits.h>
#include <math.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_log.h"
#include "graph/ge_error_codes.h"

#if !defined(__ANDROID__) && !defined(ANDROID)
#define GE_DYNAMIC_CAST dynamic_cast
#define GE_DYNAMIC_POINTER_CAST std::dynamic_pointer_cast
#else
#define GE_DYNAMIC_CAST static_cast
#define GE_DYNAMIC_POINTER_CAST std::static_pointer_cast
#endif

#define GE_RETURN_IF_ERROR(expr)            \
  do {                                      \
    const ::ge::optStatus _status = (expr); \
    if (_status) return _status;            \
  } while (0)

#define GE_RETURN_WITH_LOG_IF_INFO(expr, ...) \
  do {                                        \
    const ::ge::optStatus _status = (expr);   \
    if (_status) {                            \
      GELOGI(__VA_ARGS__);                    \
      return _status;                         \
    }                                         \
  } while (0)

// Verify whether the parameter is true. If yes, return graph failed and record the error log
#define GE_RETURN_WITH_LOG_IF_TRUE(condition, ...) \
  do {                                             \
    if (condition) {                               \
      GELOGE(ge::GRAPH_FAILED, __VA_ARGS__);       \
      return ge::GRAPH_FAILED;                     \
    }                                              \
  } while (0)

// Verify whether the parameter is false. If yes, return graph failed and record the error log
#define GE_RETURN_WITH_LOG_IF_FALSE(condition, ...) \
  do {                                              \
    bool _condition = (condition);                  \
    if (!_condition) {                              \
      GELOGE(ge::GRAPH_FAILED, __VA_ARGS__);        \
      return ge::GRAPH_FAILED;                      \
    }                                               \
  } while (0)

// Verify whether the parameter is true. If yes, return GRAPH_PARAM_INVALID and record the error log
#define GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(condition, ...) \
  do {                                                       \
    if (condition) {                                         \
      GELOGE(ge::GRAPH_PARAM_INVALID, __VA_ARGS__);          \
      return ge::GRAPH_PARAM_INVALID;                        \
    }                                                        \
  } while (0)

// Verify whether the parameter is false. If yes, return GRAPH_PARAM_INVALID and record the error log
#define GE_RT_PARAM_INVALID_WITH_LOG_IF_FALSE(condition, ...) \
  do {                                                        \
    bool _condition = (condition);                            \
    if (!_condition) {                                        \
      GELOGE(ge::GRAPH_PARAM_INVALID, __VA_ARGS__);           \
      return ge::GRAPH_PARAM_INVALID;                         \
    }                                                         \
  } while (0)

// Verify whether the parameter is null. If yes, return GRAPH_PARAM_INVALID and record the error log
#define GE_CHECK_NOTNULL(val)                                               \
  do {                                                                      \
    if (val == nullptr) {                                                   \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] must not be null.", #val); \
      return ge::GRAPH_PARAM_INVALID;                                       \
    }                                                                       \
  } while (0)

// Verify whether the parameter is null. If yes, return GRAPH_PARAM_INVALID and record the error log
#define GE_CHECK_NOTNULL_EXEC(val, expr)                                    \
  do {                                                                      \
    if (val == nullptr) {                                                   \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] must not be null.", #val); \
      expr;                                                                 \
    }                                                                       \
  } while (0)

// Verify whether the parameter is null. If yes, return false and record the error log
#define GE_RT_FALSE_CHECK_NOTNULL(val)                               \
  do {                                                               \
    if (val == nullptr) {                                            \
      GELOGE(ge::GRAPH_FAILED, "param[%s] must not be null.", #val); \
      return false;                                                  \
    }                                                                \
  } while (0)

// Check whether the parameter is out of range
#define GE_CHECK_SIZE(size)                                                \
  do {                                                                     \
    if (size == 0) {                                                       \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] is out of range", #size); \
      return ge::GRAPH_PARAM_INVALID;                                      \
    }                                                                      \
  } while (0)

///
/// @ingroup GE_common
/// eg:GE_DEFINE_BYTE_SIZE(filter_byte, filter.data().size(), sizeof(float));
///
#define GE_DEFINE_BYTE_SIZE(_var_name, _expr, _sizeof)                               \
  uint32_t _var_name;                                                                \
  do {                                                                               \
    uint32_t _expr_size = (_expr);                                                   \
    uint32_t _sizeof_size = (_sizeof);                                               \
    if (_expr_size > (0xffffffff) / _sizeof_size) {                                  \
      GELOGE(ge::GRAPH_PARAM_INVALID, "byte size : %s is out of range", #_var_name); \
      return ge::GRAPH_PARAM_INVALID;                                                \
    }                                                                                \
    _var_name = _sizeof_size * _expr_size;                                           \
  } while (0);

// Check whether the container is empty
#define GE_CHECK_VECTOR_NOT_EMPTY(vector)                           \
  do {                                                              \
    if (vector.empty()) {                                           \
      GELOGE(ge::GRAPH_FAILED, "param[#vector] is empty", #vector); \
      return ge::GRAPH_FAILED;                                      \
    }                                                               \
  } while (0)

// Check whether the container is empty and return the specified status code
#define GE_CHECK_VECTOR_NOT_EMPTY_RET_STATUS(vector, _status) \
  do {                                                        \
    if (vector.empty()) {                                     \
      GELOGE(_status, "param[%s] is empty", #vector);         \
      return _status;                                         \
    }                                                         \
  } while (0)

///
/// @ingroup GE_common
/// @brief This macro provides the ability to disable copying constructors and assignment operators.
///        It is usually placed under private
///
#define GE_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName &) = delete;        \
  void operator=(const TypeName &) = delete

/// Check whether the size is 0 or out of range
/// @param：size：Size to be verified
#define GE_CHECK_SIZE_RANGE(size)                                          \
  do {                                                                     \
    if (size == 0 || size >= UINT_MAX / 4) {                               \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] is out of range", #size); \
      return ge::GRAPH_PARAM_INVALID;                                      \
    }                                                                      \
  } while (0)

#define GE_CHECK_SHORT_SIZE_RANGE(size)                                    \
  do {                                                                     \
    if (size == 0 || size >= UINT_MAX / 2) {                               \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] is out of range", #size); \
      return ge::GRAPH_PARAM_INVALID;                                      \
    }                                                                      \
  } while (0)

#define GE_CHECK_POSITIVE_SIZE_RANGE(size)                                          \
  do {                                                                              \
    if (size <= 0) {                                                                \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] is not a positive number", #size); \
      return ge::GRAPH_PARAM_INVALID;                                               \
    }                                                                               \
  } while (0)

#define GE_CHECK_POSITIVE_SHORT_SIZE_RANGE(size)                           \
  do {                                                                     \
    if (size <= 0 || size == 0 || size >= UINT_MAX / 4) {                  \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] is out of range", #size); \
      return ge::GRAPH_PARAM_INVALID;                                      \
    }                                                                      \
  } while (0)

// Verify that the value on the left is greater than or equal to the value on the right
#define GE_CHECK_GE(lhs, rhs)                                                    \
  do {                                                                           \
    if (lhs < rhs) {                                                             \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] is less than[%s]", #lhs, #rhs); \
      return ge::GRAPH_PARAM_INVALID;                                            \
    }                                                                            \
  } while (0)

// Check whether the parameters are equal
#define GE_CHECK_EQ(val1, val2)                                                        \
  do {                                                                                 \
    if (val1 != val2) {                                                                \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] is not equals to[%s]", #val1, #val2); \
      return ge::GRAPH_PARAM_INVALID;                                                  \
    }                                                                                  \
  } while (0)

// Verify that the value on the left is less than or equal to the value on the right
#define GE_CHECK_LE(lhs, rhs)                                                       \
  do {                                                                              \
    if (lhs > rhs) {                                                                \
      GELOGE(ge::GRAPH_PARAM_INVALID, "param[%s] is greater than[%s]", #lhs, #rhs); \
      return ge::GRAPH_PARAM_INVALID;                                               \
    }                                                                               \
  } while (0)

// Check whether the parameters are equal
#define GE_CHECK_EQ_WITH_LOG(val1, val2, ...)       \
  do {                                              \
    if (val1 != val2) {                             \
      GELOGE(ge::GRAPH_PARAM_INVALID, __VA_ARGS__); \
      return ge::GRAPH_PARAM_INVALID;               \
    }                                               \
  } while (0)

// If expr is false, the custom statement is executed
#define CHECK_FALSE_EXEC(expr, exec_expr, ...) \
  do {                                         \
    bool b = (expr);                           \
    if (!b) {                                  \
      exec_expr;                               \
    }                                          \
  } while (0)

#define GE_DELETE_NEW_SINGLE(var) \
  do {                            \
    if (var != nullptr) {         \
      delete var;                 \
      var = nullptr;              \
    }                             \
  } while (0)

#define GE_DELETE_NEW_ARRAY(var) \
  do {                           \
    if (var != nullptr) {        \
      delete[] var;              \
      var = nullptr;             \
    }                            \
  } while (0)

template <typename T, typename... Args>
static inline std::shared_ptr<T> ComGraphMakeShared(Args &&... args) {
  using T_nc = typename std::remove_const<T>::type;
  std::shared_ptr<T> ret(new (std::nothrow) T_nc(std::forward<Args>(args)...));
  return ret;
}

#endif  // COMMON_GRAPH_DEBUG_GE_UTIL_H_
