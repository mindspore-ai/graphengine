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

#ifndef GE_COMMON_MATH_MATH_UTIL_H_
#define GE_COMMON_MATH_MATH_UTIL_H_

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "common/fp16_t.h"
#include "framework/common/debug/log.h"
#include "framework/common/fmk_error_codes.h"

namespace ge {
inline size_t MemSizeAlign(const size_t bytes, const uint32_t aligns = 32U) {
  const size_t align_size = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
  return (((bytes + align_size) - 1U) / align_size) * align_size;
}

/// @ingroup math_util
/// @brief check whether int32 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckIntAddOverflow(const int32_t a, const int32_t b) {
  if (((b > 0) && (a > (INT32_MAX - b))) || ((b < 0) && (a < (INT32_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int8 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckInt8AddOverflow(const int8_t a, const int8_t b) {
  if (((b > 0) && (a > (INT8_MAX - b))) || ((b < 0) && (a < (INT8_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int16 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckInt16AddOverflow(const int16_t a, const int16_t b) {
  if (((b > 0) && (a > (INT16_MAX - b))) || ((b < 0) && (a < (INT16_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckInt32AddOverflow(const int32_t a, const int32_t b) {
  if (((b > 0) && (a > (INT32_MAX - b))) || ((b < 0) && (a < (INT32_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int64 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckInt64AddOverflow(const int64_t a, const int64_t b) {
  if (((b > 0) && (a > (std::numeric_limits<int64_t>::max() - b))) ||
      ((b < 0) && (a < (std::numeric_limits<int64_t>::min() - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint8 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckUint8AddOverflow(const uint8_t a, const uint8_t b) {
  if (static_cast<int32_t>(a) > (static_cast<int32_t>(UINT8_MAX) - static_cast<int32_t>(b))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint16 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckUint16AddOverflow(const uint16_t a, const uint16_t b) {
  if (static_cast<int32_t>(a) > (static_cast<int32_t>(UINT16_MAX) - static_cast<int32_t>(b))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint32 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckUint32AddOverflow(const uint32_t a, const uint32_t b) {
  if (a > (UINT32_MAX - b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint64 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckUint64AddOverflow(const uint64_t a, const uint64_t b) {
  if (a > (UINT64_MAX - b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether size_t addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckSizeTAddOverflow(const size_t a, const size_t b) {
  if (a > (SIZE_MAX - b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether fp16_t addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFp16AddOverflow(const fp16_t &a, const fp16_t &b) {
  const fp16_t result = static_cast<fp16_t>(a) + static_cast<fp16_t>(b);
  if (Fp16IsInvalid(result.val)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether float addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFloatAddOverflow(const float32_t a, const float32_t b) {
  if (!std::isfinite(static_cast<float32_t>(a) + static_cast<float32_t>(b))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether double addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckDoubleAddOverflow(const float64_t a, const float64_t b) {
  if (!std::isfinite(static_cast<float64_t>(a) + static_cast<float64_t>(b))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32_t subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckIntSubOverflow(const int32_t a, const int32_t b) {
  if (((b > 0) && (a < (INT_MIN + b))) || ((b < 0) && (a > (INT_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int8 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckInt8SubOverflow(const int8_t a, const int8_t b) {
  if (((b > 0) && (a < (INT8_MIN + b))) || ((b < 0) && (a > (INT8_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int16 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckInt16SubOverflow(const int16_t a, const int16_t b) {
  if (((b > 0) && (a < (INT16_MIN + b))) || ((b < 0) && (a > (INT16_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckInt32SubOverflow(const int32_t a, const int32_t b) {
  if (((b > 0) && (a < (INT32_MIN + b))) || ((b < 0) && (a > (INT32_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int64 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckInt64SubOverflow(const int64_t a, const int64_t b) {
  if (((b > 0L) && (a < (std::numeric_limits<int64_t>::min() + b))) ||
      ((b < 0L) && (a > (std::numeric_limits<int64_t>::max() + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint8 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckUint8SubOverflow(const uint8_t a, const uint8_t b) {
  if (static_cast<int32_t>(a) < static_cast<int32_t>(b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint16 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckUint16SubOverflow(const uint16_t a, const uint16_t b) {
  if (static_cast<int32_t>(a) < static_cast<int32_t>(b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint32 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckUint32SubOverflow(const uint32_t a, const uint32_t b) {
  if (a < b) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint64 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckUint64SubOverflow(const uint64_t a, const uint64_t b) {
  if (a < b) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether size_t subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckSizeTSubOverflow(const size_t a, const size_t b) {
  if (a < b) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether fp16_t subtraction can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFp16SubOverflow(const fp16_t &a, const fp16_t &b) {
  const fp16_t result = static_cast<fp16_t>(a) - static_cast<fp16_t>(b);
  if (Fp16IsInvalid(result.val)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether float subtraction can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFloatSubOverflow(const float32_t a, const float32_t b) {
  if (!std::isfinite(a - b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether double subtraction can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckDoubleSubOverflow(const float64_t a, const float64_t b) {
  if (!std::isfinite(a - b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32_t multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckIntMulOverflow(const int32_t a, const int32_t b) {
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
/// @ingroup math_util
/// @brief check whether int8 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt8MulOverflow(const int8_t a, const int8_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT8_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT8_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT8_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT8_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int16 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt16MulOverflow(const int16_t a, const int16_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT16_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT16_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT16_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT16_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt32MulOverflow(const int32_t a, const int32_t b) {
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

/// @ingroup math_util
/// @brief check whether int64 int32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt64Int32MulOverflow(const int64_t a, const int32_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (std::numeric_limits<int64_t>::max() / b)) {
        return FAILED;
      }
    } else {
      if (b < (std::numeric_limits<int64_t>::min() / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (std::numeric_limits<int64_t>::min() / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (std::numeric_limits<int64_t>::max() / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

/// @ingroup math_util
/// @brief check whether int64 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt64MulOverflow(const int64_t a, const int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (std::numeric_limits<int64_t>::max() / b)) {
        return FAILED;
      }
    } else {
      if (b < (std::numeric_limits<int64_t>::min() / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (std::numeric_limits<int64_t>::min() / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (std::numeric_limits<int64_t>::max() / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int64 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt64Uint32MulOverflow(const int64_t a, const uint32_t b) {
  if ((a == 0L) || (b == 0U)) {
    return SUCCESS;
  }
  if (a > 0) {
    if (a > (std::numeric_limits<int64_t>::max() / static_cast<int64_t>(b))) {
      return FAILED;
    }
  } else {
    if (a < (std::numeric_limits<int64_t>::min() / static_cast<int64_t>(b))) {
      return FAILED;
    }
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint8 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckUint8MulOverflow(const uint8_t a, const uint8_t b) {
  if ((a == 0U) || (b == 0U)) {
    return SUCCESS;
  }

  if (static_cast<int32_t>(a) > (UINT8_MAX / static_cast<int32_t>(b))) {
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint16 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckUint16MulOverflow(const uint16_t a, const uint16_t b) {
  if ((a == 0U) || (b == 0U)) {
    return SUCCESS;
  }

  if (static_cast<int32_t>(a) > (UINT16_MAX / static_cast<int32_t>(b))) {
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckUint32MulOverflow(const uint32_t a, const uint32_t b) {
  if ((a == 0U) || (b == 0U)) {
    return SUCCESS;
  }

  if (a > (UINT32_MAX / b)) {
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint64 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckUint64MulOverflow(const uint64_t a, const uint64_t b) {
  if ((a == 0UL) || (b == 0UL)) {
    return SUCCESS;
  }

  if (a > (UINT64_MAX / b)) {
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether size_t multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckSizeTMulOverflow(const size_t a, const size_t b) {
  if ((a == 0UL) || (b == 0UL)) {
    return SUCCESS;
  }

  if (a > (SIZE_MAX / b)) {
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether fp16_t multiplication can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFp16MulOverflow(const fp16_t &a, const fp16_t &b) {
  const fp16_t result = static_cast<fp16_t>(a) * static_cast<fp16_t>(b);
  if (Fp16IsInvalid(result.val)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether float multiplication can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFloatMulOverflow(const float32_t a, const float32_t b) {
  if (!std::isfinite(static_cast<float32_t>(a) * static_cast<float32_t>(b))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether double multiplication can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckDoubleMulOverflow(const float64_t a, const float64_t b) {
  if (!std::isfinite(static_cast<float64_t>(a) * static_cast<float64_t>(b))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32_t division can result in overflow
/// @param [in] a  dividend
/// @param [in] b  divisor
/// @return Status
inline Status CheckIntDivOverflow(const int32_t a, const int32_t b) {
  if ((b == 0) || ((a == INT_MIN) && (b == -1))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32 division can result in overflow
/// @param [in] a  dividend
/// @param [in] b  divisor
/// @return Status
inline Status CheckInt32DivOverflow(const int32_t a, const int32_t b) {
  if ((b == 0) || ((a == INT32_MIN) && (b == -1))) {
    return FAILED;
  }
  return SUCCESS;
}

#define FMK_INT_ADDCHECK(a, b)                                                        \
  if (ge::CheckIntAddOverflow(static_cast<int32_t>(a), static_cast<int32_t>(b)) != SUCCESS) { \
    GELOGW("Int %d and %d addition can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                  \
    return INTERNAL_ERROR;                                                            \
  }

#define FMK_INT8_ADDCHECK(a, b)                                                       \
  if (ge::CheckInt8AddOverflow(static_cast<int8_t>(a), static_cast<int8_t>(b)) != SUCCESS) { \
    GELOGW("Int8 %d and %d addition can result in overflow!", static_cast<int8_t>(a), \
           static_cast<int8_t>(b));                                                   \
    return INTERNAL_ERROR;                                                            \
  }

#define FMK_INT16_ADDCHECK(a, b)                                                        \
  if (ge::CheckInt16AddOverflow(static_cast<int16_t>(a), static_cast<int16_t>(b)) != SUCCESS) { \
    GELOGW("Int16 %d and %d addition can result in overflow!", static_cast<int16_t>(a), \
           static_cast<int16_t>(b));                                                    \
    return INTERNAL_ERROR;                                                              \
  }

#define FMK_INT32_ADDCHECK(a, b)                                                        \
  if (ge::CheckInt32AddOverflow(static_cast<int32_t>(a), static_cast<int32_t>(b)) != SUCCESS) { \
    GELOGW("Int32 %d and %d addition can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                    \
    return INTERNAL_ERROR;                                                              \
  }

#define FMK_INT64_ADDCHECK(a, b)                                                          \
  if (ge::CheckInt64AddOverflow(static_cast<int64_t>(a), static_cast<int64_t>(b)) != SUCCESS) { \
    GELOGW("Int64 %ld and %ld addition can result in overflow!", static_cast<int64_t>(a), \
           static_cast<int64_t>(b));                                                      \
    return INTERNAL_ERROR;                                                                \
  }

#define FMK_UINT8_ADDCHECK(a, b)                                                        \
  if (ge::CheckUint8AddOverflow(static_cast<uint8_t>(a), static_cast<uint8_t>(b)) != SUCCESS) { \
    GELOGW("Uint8 %u and %u addition can result in overflow!", static_cast<uint8_t>(a), \
           static_cast<uint8_t>(b));                                                    \
    return INTERNAL_ERROR;                                                              \
  }

#define FMK_UINT16_ADDCHECK(a, b)                                                         \
  if (ge::CheckUint16AddOverflow(static_cast<uint16_t>(a), static_cast<uint16_t>(b)) != SUCCESS) { \
    GELOGW("UINT16 %u and %u addition can result in overflow!", static_cast<uint16_t>(a), \
           static_cast<uint16_t>(b));                                                     \
    return INTERNAL_ERROR;                                                                \
  }

#define FMK_UINT32_ADDCHECK(a, b)                                                         \
  if (ge::CheckUint32AddOverflow(static_cast<uint32_t>(a), static_cast<uint32_t>(b)) != SUCCESS) { \
    GELOGW("Uint32 %u and %u addition can result in overflow!", static_cast<uint32_t>(a), \
           static_cast<uint32_t>(b));                                                     \
    return INTERNAL_ERROR;                                                                \
  }

#define FMK_UINT64_ADDCHECK(a, b)                                                           \
  if (ge::CheckUint64AddOverflow(static_cast<uint64_t>(a), static_cast<uint64_t>(b)) != SUCCESS) { \
    GELOGW("Uint64 %lu and %lu addition can result in overflow!", static_cast<uint64_t>(a), \
           static_cast<uint64_t>(b));                                                       \
    return INTERNAL_ERROR;                                                                  \
  }

#define FMK_SIZET_ADDCHECK(a, b)                                                           \
  if (ge::CheckSizeTAddOverflow(static_cast<size_t>(a), static_cast<size_t>(b)) != SUCCESS) { \
    GELOGW("size_t %zu and %zu addition can result in overflow!", static_cast<size_t>(a), \
           static_cast<size_t>(b));                                                       \
    return INTERNAL_ERROR;                                                                  \
  }

#define FMK_FP16_ADDCHECK(a, b)                                                      \
  if (ge::CheckFp16AddOverflow(static_cast<fp16_t>(a), static_cast<fp16_t>(b)) != SUCCESS) { \
    GELOGW("Fp16 %f and %f addition can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                   \
    return INTERNAL_ERROR;                                                           \
  }

#define FMK_FLOAT_ADDCHECK(a, b)                                                      \
  if (ge::CheckFloatAddOverflow(static_cast<float>(a), static_cast<float>(b)) != SUCCESS) { \
    GELOGW("Float %f and %f addition can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                    \
    return INTERNAL_ERROR;                                                            \
  }

#define FMK_DOUBLE_ADDCHECK(a, b)                                                         \
  if (ge::CheckDoubleAddOverflow(static_cast<double>(a), static_cast<double>(b)) != SUCCESS) { \
    GELOGW("Double %lf and %lf addition can result in overflow!", static_cast<double>(a), \
           static_cast<double>(b));                                                       \
    return INTERNAL_ERROR;                                                                \
  }

#define FMK_INT_SUBCHECK(a, b)                                                           \
  if (ge::CheckIntSubOverflow(static_cast<int32_t>(a), static_cast<int32_t>(b)) != SUCCESS) { \
    GELOGW("Int %d and %d subtraction can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                     \
    return INTERNAL_ERROR;                                                               \
  }

#define FMK_INT8_SUBCHECK(a, b)                                                          \
  if (ge::CheckInt8SubOverflow(static_cast<int8_t>(a), static_cast<int8_t>(b)) != SUCCESS) { \
    GELOGW("Int8 %d and %d subtraction can result in overflow!", static_cast<int8_t>(a), \
           static_cast<int8_t>(b));                                                      \
    return INTERNAL_ERROR;                                                               \
  }

#define FMK_INT16_SUBCHECK(a, b)                                                           \
  if (ge::CheckInt16SubOverflow(static_cast<int16_t>(a), static_cast<int16_t>(b)) != SUCCESS) { \
    GELOGW("Int16 %d and %d subtraction can result in overflow!", static_cast<int16_t>(a), \
           static_cast<int16_t>(b));                                                       \
    return INTERNAL_ERROR;                                                                 \
  }

#define FMK_INT32_SUBCHECK(a, b)                                                           \
  if (ge::CheckInt32SubOverflow(static_cast<int32_t>(a), static_cast<int32_t>(b)) != SUCCESS) { \
    GELOGW("Int32 %d and %d subtraction can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                       \
    return INTERNAL_ERROR;                                                                 \
  }

#define FMK_INT64_SUBCHECK(a, b)                                                             \
  if (ge::CheckInt64SubOverflow(static_cast<int64_t>(a), static_cast<int64_t>(b)) != SUCCESS) { \
    GELOGW("Int64 %ld and %ld subtraction can result in overflow!", static_cast<int64_t>(a), \
           static_cast<int64_t>(b));                                                         \
    return INTERNAL_ERROR;                                                                   \
  }

#define FMK_UINT8_SUBCHECK(a, b)                                                           \
  if (ge::CheckUint8SubOverflow(static_cast<uint8_t>(a), static_cast<uint8_t>(b)) != SUCCESS) { \
    GELOGW("Uint8 %u and %u subtraction can result in overflow!", static_cast<uint8_t>(a), \
           static_cast<uint8_t>(b));                                                       \
    return INTERNAL_ERROR;                                                                 \
  }

#define FMK_UINT16_SUBCHECK(a, b)                                                            \
  if (ge::CheckUint16SubOverflow(static_cast<uint16_t>(a), static_cast<uint16_t>(b)) != SUCCESS) { \
    GELOGW("Uint16 %u and %u subtraction can result in overflow!", static_cast<uint16_t>(a), \
           static_cast<uint16_t>(b));                                                        \
    return INTERNAL_ERROR;                                                                   \
  }

#define FMK_UINT32_SUBCHECK(a, b)                                                            \
  if (ge::CheckUint32SubOverflow(static_cast<uint32_t>(a), static_cast<uint32_t>(b)) != SUCCESS) { \
    GELOGW("Uint32 %u and %u subtraction can result in overflow!", static_cast<uint32_t>(a), \
           static_cast<uint32_t>(b));                                                        \
    return INTERNAL_ERROR;                                                                   \
  }

#define FMK_UINT64_SUBCHECK(a, b)                                                              \
  if (ge::CheckUint64SubOverflow(static_cast<uint64_t>(a), static_cast<uint64_t>(b)) != SUCCESS) { \
    GELOGW("Uint64 %lu and %lu subtraction can result in overflow!", static_cast<uint64_t>(a), \
           static_cast<uint64_t>(b));                                                          \
    return INTERNAL_ERROR;                                                                     \
  }

#define FMK_SIZET_SUBCHECK(a, b)                                                              \
  if (ge::CheckSizeTSubOverflow(static_cast<size_t>(a), static_cast<size_t>(b)) != SUCCESS) { \
    GELOGW("size_t %zu and %zu subtraction can result in overflow!", static_cast<size_t>(a), \
           static_cast<size_t>(b));                                                          \
    return INTERNAL_ERROR;                                                                     \
  }

#define FMK_FP16_SUBCHECK(a, b)                                                         \
  if (ge::CheckFp16SubOverflow(static_cast<fp16_t>(a), static_cast<fp16_t>(b)) != SUCCESS) { \
    GELOGW("Fp16 %f and %f subtraction can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                      \
    return INTERNAL_ERROR;                                                              \
  }

#define FMK_FLOAT_SUBCHECK(a, b)                                                         \
  if (ge::CheckFloatSubOverflow(static_cast<float>(a), static_cast<float>(b)) != SUCCESS) { \
    GELOGW("Float %f and %f subtraction can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                       \
    return INTERNAL_ERROR;                                                               \
  }

#define FMK_DOUBLE_SUBCHECK(a, b)                                                            \
  if (ge::CheckDoubleSubOverflow(static_cast<double>(a), static_cast<double>(b)) != SUCCESS) { \
    GELOGW("Double %lf and %lf subtraction can result in overflow!", static_cast<double>(a), \
           static_cast<double>(b));                                                          \
    return INTERNAL_ERROR;                                                                   \
  }

#define FMK_INT_MULCHECK(a, b)                                                              \
  if (ge::CheckIntMulOverflow(static_cast<int32_t>(a), static_cast<int32_t>(b)) != SUCCESS) { \
    GELOGW("Int %d and %d multiplication can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                        \
    return INTERNAL_ERROR;                                                                  \
  }

#define FMK_INT8_MULCHECK(a, b)                                                             \
  if (ge::CheckInt8MulOverflow(static_cast<int8_t>(a), static_cast<int8_t>(b)) != SUCCESS) { \
    GELOGW("Int8 %d and %d multiplication can result in overflow!", static_cast<int8_t>(a), \
           static_cast<int8_t>(b));                                                         \
    return INTERNAL_ERROR;                                                                  \
  }

#define FMK_INT16_MULCHECK(a, b)                                                              \
  if (ge::CheckInt16MulOverflow(static_cast<int16_t>(a), static_cast<int16_t>(b)) != SUCCESS) { \
    GELOGW("Int16 %d and %d multiplication can result in overflow!", static_cast<int16_t>(a), \
           static_cast<int16_t>(b));                                                          \
    return INTERNAL_ERROR;                                                                    \
  }

#define FMK_INT32_MULCHECK(a, b)                                                              \
  if (ge::CheckInt32MulOverflow(static_cast<int32_t>(a), static_cast<int32_t>(b)) != SUCCESS) { \
    GELOGW("Int32 %d and %d multiplication can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                          \
    return INTERNAL_ERROR;                                                                    \
  }

#define FMK_INT64_MULCHECK(a, b)                                                                \
  if (ge::CheckInt64MulOverflow(static_cast<int64_t>(a), static_cast<int64_t>(b)) != SUCCESS) { \
    GELOGW("Int64 %ld and %ld multiplication can result in overflow!", static_cast<int64_t>(a), \
           static_cast<int64_t>(b));                                                            \
    return INTERNAL_ERROR;                                                                      \
  }

#define FMK_UINT8_MULCHECK(a, b)                                                              \
  if (ge::CheckUint8MulOverflow(static_cast<uint8_t>(a), static_cast<uint8_t>(b)) != SUCCESS) { \
    GELOGW("Uint8 %u and %u multiplication can result in overflow!", static_cast<uint8_t>(a), \
           static_cast<uint8_t>(b));                                                          \
    return INTERNAL_ERROR;                                                                    \
  }

#define FMK_UINT16_MULCHECK(a, b)                                                               \
  if (ge::CheckUint16MulOverflow(static_cast<uint16_t>(a), static_cast<uint16_t>(b)) != SUCCESS) { \
    GELOGW("Uint16 %u and %u multiplication can result in overflow!", static_cast<uint16_t>(a), \
           static_cast<uint16_t>(b));                                                           \
    return INTERNAL_ERROR;                                                                      \
  }

#define FMK_UINT32_MULCHECK(a, b)                                                               \
  if (ge::CheckUint32MulOverflow(static_cast<uint32_t>(a), static_cast<uint32_t>(b)) != SUCCESS) { \
    GELOGW("Uint32 %u and %u multiplication can result in overflow!", static_cast<uint32_t>(a), \
           static_cast<uint32_t>(b));                                                           \
    return INTERNAL_ERROR;                                                                      \
  }

#define FMK_UINT64_MULCHECK(a, b)                                                                 \
  if (ge::CheckUint64MulOverflow(static_cast<uint64_t>(a), static_cast<uint64_t>(b)) != SUCCESS) { \
    GELOGW("Uint64 %lu and %lu multiplication can result in overflow!", static_cast<uint64_t>(a), \
           static_cast<uint64_t>(b));                                                             \
    return INTERNAL_ERROR;                                                                        \
  }

#define FMK_SIZET_MULCHECK(a, b)                                                                 \
  if (ge::CheckSizeTMulOverflow(static_cast<size_t>(a), static_cast<size_t>(b)) != SUCCESS) { \
    GELOGW("size_t %zu and %zu multiplication can result in overflow!", static_cast<size_t>(a), \
           static_cast<size_t>(b));                                                             \
    return INTERNAL_ERROR;                                                                        \
  }

#define FMK_FP16_MULCHECK(a, b)                                                            \
  if (ge::CheckFp16MulOverflow(static_cast<fp16_t>(a), static_cast<fp16_t>(b)) != SUCCESS) { \
    GELOGW("Fp16 %f and %f multiplication can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                         \
    return INTERNAL_ERROR;                                                                 \
  }

#define FMK_FLOAT_MULCHECK(a, b)                                                            \
  if (ge::CheckFloatMulOverflow(static_cast<float>(a), static_cast<float>(b)) != SUCCESS) { \
    GELOGW("Float %f and %f multiplication can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                          \
    return INTERNAL_ERROR;                                                                  \
  }

#define FMK_DOUBLE_MULCHECK(a, b)                                                               \
  if (ge::CheckDoubleMulOverflow(static_cast<double>(a), static_cast<double>(b)) != SUCCESS) {  \
    GELOGW("Double %lf and %lf multiplication can result in overflow!", static_cast<double>(a), \
           static_cast<double>(b));                                                             \
    return INTERNAL_ERROR;                                                                      \
  }

#define FMK_INT_DIVCHECK(a, b)                                                        \
  if (CheckIntDivOverflow(static_cast<int32_t>(a), static_cast<int32_t>(b)) != SUCCESS) { \
    GELOGW("Int %d and %d division can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                  \
    return INTERNAL_ERROR;                                                            \
  }

#define FMK_INT32_DIVCHECK(a, b)                                                        \
  if (CheckInt32DivOverflow(static_cast<int32_t>(a), static_cast<int32_t>(b)) != SUCCESS) { \
    GELOGW("Int32 %d and %d division can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                    \
    return INTERNAL_ERROR;                                                              \
  }

#define FMK_INT64_UINT32_MULCHECK(a, b)                                                               \
  if (ge::CheckInt64Uint32MulOverflow(static_cast<uint32_t>(a), static_cast<uint32_t>(b)) != SUCCESS) { \
    GELOGW("Int64 %ld and Uint32 %u multiplication can result in overflow!", static_cast<int64_t>(a), \
           static_cast<uint32_t>(b));                                                                 \
    return INTERNAL_ERROR;                                                                            \
  }

#define FMK_FP16_ZEROCHECK(a)                                                                            \
  if ((fabs((a)) < DBL_EPSILON) || ((a) < 0)) {                                                        \
    GELOGW("Fp16 %f can not less than or equal to zero! ", (a));                                         \
    return INTERNAL_ERROR;                                                                               \
  }

#define FMK_FLOAT_ZEROCHECK(a)                                                                           \
  if ((fabs((a)) < FLT_EPSILON) || ((a) < 0)) {                                                       \
    GELOGW("Float %f can not less than or equal to zero! ", (a));                                        \
    return INTERNAL_ERROR;                                                                               \
  }

#define FMK_DOUBLE_ZEROCHECK(a)                                                                          \
  if ((fabs((a)) < DBL_EPSILON) || ((a) < 0)) {                                                        \
    GELOGW("Double %lf can not less than or equal to zero! ", (a));                                      \
    return INTERNAL_ERROR;                                                                               \
  }
}  // namespace ge
#endif  // GE_COMMON_MATH_MATH_UTIL_H_
