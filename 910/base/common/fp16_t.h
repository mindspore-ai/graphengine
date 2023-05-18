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

#ifndef GE_COMMON_FP16_T_H_
#define GE_COMMON_FP16_T_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "external/graph/types.h"
#include "graph/def_types.h"

namespace ge {
enum class DimIndex : uint32_t {
  kDim0 = 0,
  kDim1,
  kDim2,
  kDim3,
  kDim4,
  kDim5,
  kDim6,
  kDim7,
  kDim8,
  kDim9,
  kDim10,
  kDim11,
  kDim12,
  kDim13,
  kDim14,
  kDim15,
  kDim16,
};

enum class BitShift : uint32_t{
  kBitShift2 = 2,
  kBitShift3 = 3,
  kBitShift4 = 4,
  kBitShift5 = 5,
  kBitShift6 = 6,
  kBitShift7 = 7,
  kBitShift8 = 8,
  kBitShift9 = 9,
  kBitShift10 = 10,
  kBitShift11 = 11,
  kBitShift12 = 12,
  kBitShift13 = 13,
  kBitShift14 = 14,
  kBitShift15 = 15,
  kBitShift16 = 16,
  kBitShift20 = 20,
  kBitShift24 = 24,
  kBitShift27 = 27,
  kBitShift28 = 28,
  kBitShift31 = 31,
  kBitShift32 = 32,
  kBitShift36 = 36,
  kBitShift40 = 40,
  kBitShift44 = 44,
  kBitShift48 = 48,
  kBitShift52 = 52,
  kBitShift56 = 56,
  kBitShift59 = 59,
  kBitShift60 = 60,
  kBitShift63 = 63,
  kBitShift64 = 64,
  kBitShift128 = 128,
  kBitShift255 = 255,
  kBitShift256 = 256,
  kBitShift512 = 512,
  kBitShift768 = 768,
  kBitShift784 = 784,
  kBitShift1020 = 1020,
  kBitShift1024 = 1024,
  kBitShift3136 = 3136,
  kBitShift4096 = 4096,
  kBitShift6144 = 6144,
  kBitShift10240 = 10240,
  kBitShift65536 = 65536
};
/// @ingroup fp16 basic parameter
/// @brief   fp16 exponent bias
constexpr uint16_t kFp16ExpBias = 15U;
/// @ingroup fp16 basic parameter
/// @brief   the mantissa bit length of fp16 is 10
constexpr uint16_t kFp16ManLen = 10U;
/// @ingroup fp16 basic parameter
/// @brief   bit index of sign in fp16
constexpr uint16_t kFp16SignIndex = 15U;
/// @ingroup fp16 basic parameter
/// @brief   sign mask of fp16         (1 00000 00000 00000)
constexpr uint16_t kFp16SignMask = 0x8000U;
/// @ingroup fp16 basic parameter
/// @brief   exponent mask of fp16     (  11111 00000 00000)
constexpr uint16_t kFp16ExpMask = 0x7C00U;
/// @ingroup fp16 basic parameter
/// @brief   mantissa mask of fp16     (        11111 11111)
constexpr uint16_t kFp16ManMask = 0x03FFU;
/// @ingroup fp16 basic parameter
/// @brief   hide bit of mantissa of fp16(   1 00000 00000)
constexpr uint16_t kFp16ManHideBit = 0x0400U;
/// @ingroup fp16 basic parameter
/// @brief   maximum value            (0111 1011 1111 1111)
constexpr uint16_t kFp16Max = 0x7BFFU;
/// @ingroup fp16 basic parameter
/// @brief   absolute maximum value   (0111 1111 1111 1111)
constexpr uint16_t kFp16AbsMax = 0x7FFFU;
/// @ingroup fp16 basic parameter
/// @brief   maximum exponent value of fp16 is 15(11111)
constexpr uint16_t kFp16MaxExp = 0x001FU;
/// @ingroup fp16 basic parameter
/// @brief   maximum mantissa value of fp16(11111 11111)
constexpr uint16_t kFp16MaxMan = 0x03FFU;
/// @ingroup fp16 basic operator
/// @brief   get sign of fp16
inline uint16_t Fp16ExtracSign(const uint16_t x) {
  return ((x >> 15U) & 1U);
}
/// @ingroup fp16 basic operator
/// @brief   get exponent of fp16
inline uint16_t Fp16ExtracExp(const uint16_t x) {
  return ((x >> 10U) & kFp16MaxExp);
}
/// @ingroup fp16 basic operator
/// @brief   get mantissa of fp16
inline uint16_t Fp16ExtracMan(const uint16_t x) {
  const uint8_t result = ((static_cast<uint32_t>(x >> 10U) & 0x1FU) > 0U) ? 1U : 0U;
  return static_cast<uint16_t>((static_cast<uint32_t>(x >> 0U) & 0x3FFU) | (result * 0x400U));
}
/// @ingroup fp16 basic operator
/// @brief   constructor of fp16 from sign exponent and mantissa
inline uint16_t Fp16Constructor(const uint16_t s, const uint16_t e, const uint16_t m) {
  return (static_cast<uint16_t>(s << kFp16SignIndex) | static_cast<uint16_t>(e << kFp16ManLen) | (m & kFp16MaxMan));
}
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is zero
inline bool Fp16IsZero(const uint16_t x) {
  return ((x & kFp16AbsMax) == 0U);
}
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is a denormalized value
inline bool Fp16IsDenorm(const uint16_t x) {
  return ((x & kFp16ExpMask) == 0U);
}
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is invalid
inline bool Fp16IsInvalid(const uint16_t x) {
  return ((x & kFp16ExpMask) == kFp16ExpMask);
}
/// @ingroup fp32 basic parameter
/// @brief   fp32 exponent bias
constexpr uint32_t kFp32ExpBias = 127U;
/// @ingroup fp32 basic parameter
/// @brief   the mantissa bit length of float/fp32 is 23
constexpr uint16_t kFp32ManLen = 23U;
/// @ingroup fp32 basic parameter
/// @brief   bit index of sign in float/fp32
constexpr uint16_t kFp32SignIndex = 31U;
/// @ingroup fp32 basic parameter
/// @brief   sign mask of fp32         (1 0000 0000  0000 0000 0000 0000 000)
constexpr uint32_t kFp32SignMask = 0x80000000U;
/// @ingroup fp32 basic parameter
/// @brief   exponent mask of fp32     (  1111 1111  0000 0000 0000 0000 000)
constexpr uint32_t kFp32ExpMask = 0x7F800000U;
/// @ingroup fp32 basic parameter
/// @brief   mantissa mask of fp32     (             1111 1111 1111 1111 111)
constexpr uint32_t kFp32ManMask = 0x007FFFFFU;
/// @ingroup fp32 basic parameter
/// @brief   hide bit of mantissa of fp32      (  1  0000 0000 0000 0000 000)
constexpr uint32_t kFp32ManHideBit = 0x00800000U;
/// @ingroup fp32 basic parameter
/// @brief   absolute maximum value    (0 1111 1111  1111 1111 1111 1111 111)
constexpr uint32_t kFp32AbsMax = 0x7FFFFFFFU;
/// @ingroup fp32 basic operator
/// @brief   constructor of fp32 from sign exponent and mantissa
inline uint32_t Fp32Constructor(const uint32_t s, const uint32_t e, const uint32_t m) {
  return ((s << kFp32SignIndex) | (e << kFp32ManLen) | (m & 0x7FFFFFU));
}
/// @ingroup fp64 basic parameter
/// @brief   fp64 exponent bias
constexpr uint64_t kFp64ExpBias = 1023U;
/// @ingroup fp64 basic parameter
/// @brief   the mantissa bit length of double/fp64 is 52
constexpr uint16_t kFp64ManLen = 52U;
/// @ingroup fp64 basic parameter
/// @brief   bit index of sign in double/fp64 is 63
constexpr uint16_t kFp64SignIndex = 63U;
/// @ingroup fp64 basic parameter
/// @brief   sign mask of fp64                (1 000                   (total 63bits 0))
constexpr uint64_t kFp64SignMask = 0x8000000000000000ULL;
/// @ingroup fp64 basic parameter
/// @brief   exponent mask of fp64            (0 1 11111 11111  0000?-?-(total 52bits 0))
constexpr uint64_t kFp64ExpMask = 0x7FF0000000000000ULL;
/// @ingroup fp64 basic parameter
/// @brief   mantissa mask of fp64            (                 1111?-?-(total 52bits 1))
constexpr uint64_t kFp64ManMask = 0x000FFFFFFFFFFFFFULL;
/// @ingroup fp64 basic parameter
/// @brief   hide bit of mantissa of fp64     (               1 0000?-?-(total 52bits 0))
constexpr uint64_t kFp64ManHideBit = 0x0010000000000000ULL;
/// @ingroup integer special value judgment
/// @brief   maximum positive value of int8_t            (0111 1111)
constexpr int8_t kInt8Max = 0x7F;
/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 8 bits length  (1111 111)
constexpr uint8_t kBitLen8Max = 0xFFU;
/// @ingroup integer special value judgment
/// @brief   maximum positive value of int16_t           (0111 1111 1111 1111)
constexpr int16_t kInt16Max = 0x7FFF;
/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 16 bits length (1111 1111 1111 1111)
constexpr uint16_t kBitLen16Max = 0xFFFFU;
/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 32 bits length (1111 1111 1111 1111 1111 1111 1111 1111)
constexpr uint32_t kBitLen32Max = 0xFFFFFFFFU;

/// @ingroup fp16_t enum
/// @brief   round mode of last valid digital
enum class TagFp16RoundMode : uint32_t {
  kRoundToNearest = 0,  // < round to nearest even
  kRoundByTruncated,    // < round by truncated
  kRoundModeReserved,
};

/// @ingroup fp16_t
/// @brief   Half precision float
///          bit15:       1 bit SIGN      +---+-----+------------+
///          bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
///          bit0-9:      10bit MAN       +---+-----+------------+
using fp16_t = class TagFp16 final {
 public:
  uint16_t val;

 public:
  /// @ingroup fp16_t constructor
  /// @brief   Constructor without any param(default constructor)
  TagFp16(void) : TagFp16(0x0U) {}
  /// @ingroup fp16_t constructor
  /// @brief   Constructor with an uint16_t value
  TagFp16(const uint16_t ui_val) : val(ui_val) {}
  /// @ingroup fp16_t constructor
  /// @brief   Constructor with a fp16_t object(copy constructor)
  TagFp16(const TagFp16 &fp) = default;

  /// @ingroup fp16_t math operator
  /// @param [in] fp fp16_t object to be added
  /// @brief   Override addition operator to performing fp16_t addition
  /// @return  Return fp16_t result of adding this and fp
  TagFp16 operator+(const TagFp16 fp) const;
  /// @ingroup fp16_t math operator
  /// @param [in] fp fp16_t object to be subtracted
  /// @brief   Override addition operator to performing fp16_t subtraction
  /// @return  Return fp16_t result of subtraction fp from this
  TagFp16 operator-(const TagFp16 fp) const;
  /// @ingroup fp16_t math operator
  /// @param [in] fp fp16_t object to be multiplied
  /// @brief   Override multiplication operator to performing fp16_t multiplication
  /// @return  Return fp16_t result of multiplying this and fp
  TagFp16 operator*(const TagFp16 fp) const;

  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t if-equal comparison
  /// @return  Return boolean result of if-equal comparison of this and fp.
  friend bool operator==(const TagFp16 lhs, const TagFp16 rhs) noexcept;
  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t greater-than comparison
  /// @return  Return boolean result of greater-than comparison of this and fp.
  friend bool operator>(const TagFp16 lhs, const TagFp16 rhs) noexcept;
  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t greater-equal comparison
  /// @return  Return boolean result of greater-equal comparison of this and fp.
  friend bool operator>=(const TagFp16 lhs, const TagFp16 rhs) noexcept;
  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t less-equal comparison
  /// @return  Return boolean result of less-equal comparison of this and fp.
  friend bool operator<=(const TagFp16 lhs, const TagFp16 rhs) noexcept;

  /// @ingroup fp16_t math evaluation operator
  /// @param [in] fp fp16_t object to be copy to fp16_t
  /// @brief   Override basic evaluation operator to copy fp16_t to a new fp16_t
  /// @return  Return fp16_t result from fp
  TagFp16 &operator=(const TagFp16 &fp) & = default;
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] f_val float object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert float to fp16_t
  /// @return  Return fp16_t result from f_val
  TagFp16 &operator=(const float32_t f_val) &;
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] d_val double object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert double to fp16_t
  /// @return  Return fp16_t result from d_val
  TagFp16 &operator=(const float64_t d_val) &;
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] i_val int32_t object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert int32_t to fp16_t
  /// @return  Return fp16_t result from i_val
  TagFp16 &operator=(const int32_t i_val) &;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to float/fp32
  /// @return  Return float/fp32 value of fp16_t
  explicit operator float32_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to double/fp64
  /// @return  Return double/fp64 value of fp16_t
  explicit operator float64_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to int8_t
  /// @return  Return int8_t value of fp16_t
  explicit operator int8_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to uint8_t
  /// @return  Return uint8_t value of fp16_t
  explicit operator uint8_t() const;
  /// @ingroup fp16_t conversion
  /// @brief   Override convert operator to convert fp16_t to int16_t
  /// @return  Return int16_t value of fp16_t
  explicit operator int16_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to uint16_t
  /// @return  Return uint16_t value of fp16_t
  explicit operator uint16_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to int32_t
  /// @return  Return int32_t value of fp16_t
  explicit operator int32_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to uint32_t
  /// @return  Return uint32_t value of fp16_t
  explicit operator uint32_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to int64_t
  /// @return  Return int64_t value of fp16_t
  explicit operator int64_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to uint64_t
  /// @return  Return uint64_t value of fp16_t
  explicit operator uint64_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to float/fp32
  /// @return  Return float/fp32 value of fp16_t
  float32_t ToFloat() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to double/fp64
  /// @return  Return double/fp64 value of fp16_t
  float64_t ToDouble() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to int8_t
  /// @return  Return int8_t value of fp16_t
  int8_t ToInt8() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to uint8_t
  /// @return  Return uint8_t value of fp16_t
  uint8_t ToUInt8() const;
  /// @ingroup fp16_t conversion
  /// @brief   Convert fp16_t to int16_t
  /// @return  Return int16_t value of fp16_t
  int16_t ToInt16() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to uint16_t
  /// @return  Return uint16_t value of fp16_t
  uint16_t ToUInt16() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to int32_t
  /// @return  Return int32_t value of fp16_t
  int32_t ToInt32() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to uint32_t
  /// @return  Return uint32_t value of fp16_t
  uint32_t ToUInt32() const;
};

/// @ingroup fp16_t public method
/// @param [in]     val signature is negative
/// @param [in|out] s   sign of fp16_t object
/// @param [in|out] e   exponent of fp16_t object
/// @param [in|out] m   mantissa of fp16_t object
/// @brief   Extract the sign, exponent and mantissa of a fp16_t object
void ExtractFp16(const uint16_t val, uint16_t &s, int16_t &e, uint16_t &m);
/// @ingroup fp16_t public method
/// @param [in]     negative sign is negative
/// @param [in|out] man      mantissa to be reverse
/// @brief   Calculate a mantissa's complement (add ont to it's radix-minus-one complement)
/// @return  Return complement of man
template <typename T>
void ReverseMan(const bool negative, T &man) {
  if (negative) {
    man = (~(man)) + 1U;
  }
}
/// @ingroup fp16_t public method
/// @param [in] e_a exponent of one fp16_t/float number
/// @param [in] m_a mantissa of one fp16_t/float number
/// @param [in] e_b exponent of another fp16_t/float number
/// @param [in] m_b mantissa of another fp16_t/float number
/// @brief   choose mantissa to be shift right whoes exponent is less than another one
/// @return  Return mantissawhoes exponent is less than another one
template <typename T>
auto MinMan(const int16_t e_a, T m_a, const int16_t e_b, T m_b) -> T {
  return (e_a > e_b) ? m_b : m_a;
}
/// @ingroup fp16_t public method
/// @param [in] man   mantissa to be operate
/// @param [in] shift right shift bits
/// @brief   right shift a mantissa
/// @return  Return right-shift mantissa
template <typename T>
auto RightShift(T man, const int16_t shift) -> T {
  constexpr uint64_t bits = static_cast<uint64_t>(sizeof(T) * 8U);  // one byte have 8 bits
  constexpr T mask = static_cast<T>(1U) << (bits - 1U);
  int32_t loop_cnt = static_cast<int32_t>(shift);
  while (loop_cnt-- != 0) {
    man = ((man & mask) | (man >> 1U));
  }
  return man;
}
/// @ingroup fp16_t public method
/// @param [in] e_a exponent of one temp fp16_t number
/// @param [in] m_a mantissa of one temp fp16_t number
/// @param [in] e_b exponent of another temp fp16_t number
/// @param [in] m_b mantissa of another temp fp16_t number
/// @brief   Get mantissa sum of two temp fp16_t numbers, T support types: uint16_t/uint32_t/uint64_t
/// @return  Return mantissa sum
template <typename T>
auto GetManSum(const int16_t e_a, const T &m_a, const int16_t e_b, const T &m_b) -> T {
  T sum = 0U;
  if (e_a != e_b) {
    T m_tmp = 0U;
    const int16_t e_tmp = static_cast<int16_t>(std::abs(static_cast<int32_t>(e_a - e_b)));
    if (e_a > e_b) {
      m_tmp = m_b;
      m_tmp = RightShift(m_tmp, e_tmp);
      sum = m_a + m_tmp;
    } else {
      m_tmp = m_a;
      m_tmp = RightShift(m_tmp, e_tmp);
      sum = m_tmp + m_b;
    }
  } else {
    sum = m_a + m_b;
  }
  return sum;
}
/// @ingroup fp16_t public method
/// @param [in] bit0    whether the last preserved bit is 1 before round
/// @param [in] bit1    whether the abbreviation's highest bit is 1
/// @param [in] bitLeft whether the abbreviation's bits which not contain highest bit grater than 0
/// @param [in] man     mantissa of a fp16_t or float number, support types: uint16_t/uint32_t/uint64_t
/// @param [in] shift   abbreviation bits
/// @brief    Round fp16_t or float mantissa to nearest value
/// @return   Returns true if round 1,otherwise false;
template <typename T>
auto ManRoundToNearest(const bool bit0, const bool bit1, const bool bitLeft, T man, const uint16_t shift = 0U) -> T {
  const uint32_t mark = (bit1 && (bitLeft || bit0)) ? 1U : 0U;
  man = static_cast<uint32_t>(man >> shift) + mark;
  return man;
}
/// @ingroup fp16_t public method
/// @param [in] man    mantissa of a float number, support types: uint16_t/uint32_t/uint64_t
/// @brief   Get bit length of a uint32_t number
/// @return  Return bit length of man
template <typename T>
int16_t GetManBitLength(T man) {
  int16_t len = 0;
  while (man != 0U) {
    man >>= 1U;
    len++;
  }
  return len;
}
}  // namespace ge
#endif  // GE_COMMON_FP16_T_H_
