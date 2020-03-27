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

#ifndef GE_COMMON_MATH_MATH_UTIL_H_
#define GE_COMMON_MATH_MATH_UTIL_H_

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "framework/common/debug/log.h"
#include "framework/common/fmk_error_codes.h"

namespace ge {
///
/// @ingroup math_util
/// @brief check whether int32 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
///
inline Status CheckIntAddOverflow(int a, int b) {
  if (((b > 0) && (a > (INT_MAX - b))) || ((b < 0) && (a < (INT_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int64 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
///
inline Status CheckInt64AddOverflow(int64_t a, int64_t b) {
  if (((b > 0) && (a > (INT64_MAX - b))) || ((b < 0) && (a < (INT64_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int32 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
///
inline Status CheckInt32AddOverflow(int32_t a, int32_t b) {
  if (((b > 0) && (a > (INT32_MAX - b))) || ((b < 0) && (a < (INT32_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether uint32 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
///
inline Status CheckUint32AddOverflow(uint32_t a, uint32_t b) {
  if (a > (UINT32_MAX - b)) {
    return FAILED;
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
///
inline Status CheckIntSubOverflow(int a, int b) {
  if (((b > 0) && (a < (INT_MIN + b))) || ((b < 0) && (a > (INT_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int32 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
///
inline Status CheckInt32SubOverflow(int32_t a, int32_t b) {
  if (((b > 0) && (a < (INT32_MIN + b))) || ((b < 0) && (a > (INT32_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
///
inline Status CheckIntMulOverflow(int a, int b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
///
inline Status CheckInt32MulOverflow(int32_t a, int32_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT32_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT32_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT32_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT32_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int64 int32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
///
inline Status CheckInt64Int32MulOverflow(int64_t a, int32_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int64 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
///
inline Status CheckInt64Uint32MulOverflow(int64_t a, uint32_t b) {
  if (a == 0 || b == 0) {
    return SUCCESS;
  }
  if (a > 0) {
    if (a > (INT64_MAX / b)) {
      return FAILED;
    }
  } else {
    if (a < (INT64_MIN / b)) {
      return FAILED;
    }
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether uint32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
///
inline Status CheckUint32MulOverflow(uint32_t a, uint32_t b) {
  if (a == 0 || b == 0) {
    return SUCCESS;
  }

  if (a > (UINT32_MAX / b)) {
    return FAILED;
  }

  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int division can result in overflow
/// @param [in] a  dividend
/// @param [in] b  divisor
/// @return Status
///
inline Status CheckIntDivOverflow(int a, int b) {
  if ((b == 0) || ((a == INT_MIN) && (b == -1))) {
    return FAILED;
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int32 division can result in overflow
/// @param [in] a  dividend
/// @param [in] b  divisor
/// @return Status
///
inline Status CheckInt32DivOverflow(int32_t a, int32_t b) {
  if ((b == 0) || ((a == INT32_MIN) && (b == -1))) {
    return FAILED;
  }
  return SUCCESS;
}

#define FMK_INT_ADDCHECK(a, b)                                                          \
  if (CheckIntAddOverflow((a), (b)) != SUCCESS) {                                       \
    GELOGE(INTERNAL_ERROR, "Int %d and %d addition can result in overflow!", (a), (b)); \
    return INTERNAL_ERROR;                                                              \
  }

#define FMK_INT32_ADDCHECK(a, b)                                                          \
  if (CheckInt32AddOverflow((a), (b)) != SUCCESS) {                                       \
    GELOGE(INTERNAL_ERROR, "Int32 %d and %d addition can result in overflow!", (a), (b)); \
    return INTERNAL_ERROR;                                                                \
  }

#define FMK_UINT32_ADDCHECK(a, b)                                                                         \
  if (CheckUint32AddOverflow((a), (b)) != SUCCESS) {                                                      \
    GELOGE(INTERNAL_ERROR, "UINT32 %u and %u addition can result in overflow!", static_cast<uint32_t>(a), \
           static_cast<uint32_t>(b));                                                                     \
    return INTERNAL_ERROR;                                                                                \
  }

#define FMK_INT_SUBCHECK(a, b)                                                             \
  if (CheckIntSubOverflow((a), (b)) != SUCCESS) {                                          \
    GELOGE(INTERNAL_ERROR, "INT %d and %d subtraction can result in overflow!", (a), (b)); \
    return INTERNAL_ERROR;                                                                 \
  }

#define FMK_INT32_SUBCHECK(a, b)                                                             \
  if (CheckInt32SubOverflow((a), (b)) != SUCCESS) {                                          \
    GELOGE(INTERNAL_ERROR, "INT32 %d and %d subtraction can result in overflow!", (a), (b)); \
    return INTERNAL_ERROR;                                                                   \
  }

#define FMK_INT_MULCHECK(a, b)                                                                \
  if (CheckIntMulOverflow((a), (b)) != SUCCESS) {                                             \
    GELOGE(INTERNAL_ERROR, "INT %d and %d multiplication can result in overflow!", (a), (b)); \
    return INTERNAL_ERROR;                                                                    \
  }

#define FMK_INT32_MULCHECK(a, b)                                                                \
  if (CheckInt32MulOverflow((a), (b)) != SUCCESS) {                                             \
    GELOGE(INTERNAL_ERROR, "INT32 %d and %d multiplication can result in overflow!", (a), (b)); \
    return INTERNAL_ERROR;                                                                      \
  }

#define FMK_UINT32_MULCHECK(a, b)                                                                               \
  if (CheckUint32MulOverflow((a), (b)) != SUCCESS) {                                                            \
    GELOGE(INTERNAL_ERROR, "UINT32 %u and %u multiplication can result in overflow!", static_cast<uint32_t>(a), \
           static_cast<uint32_t>(b));                                                                           \
    return INTERNAL_ERROR;                                                                                      \
  }

#define FMK_INT_DIVCHECK(a, b)                                                          \
  if (CheckIntDivOverflow((a), (b)) != SUCCESS) {                                       \
    GELOGE(INTERNAL_ERROR, "INT %d and %d division can result in overflow!", (a), (b)); \
    return INTERNAL_ERROR;                                                              \
  }

#define FMK_INT32_DIVCHECK(a, b)                                                          \
  if (CheckInt32DivOverflow((a), (b)) != SUCCESS) {                                       \
    GELOGE(INTERNAL_ERROR, "INT32 %d and %d division can result in overflow!", (a), (b)); \
    return INTERNAL_ERROR;                                                                \
  }

#define FMK_INT64_UINT32_MULCHECK(a, b)                                                                 \
  if (CheckInt64Uint32MulOverflow((a), (b)) != SUCCESS) {                                               \
    GELOGE(INTERNAL_ERROR, "INT64 %ld and UINT32 %u multiplication can result in overflow!", (a), (b)); \
    return INTERNAL_ERROR;                                                                              \
  }
}  // namespace ge

#endif  // GE_COMMON_MATH_MATH_UTIL_H_
