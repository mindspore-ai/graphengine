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

#ifndef GE_COMMON_FP16_T_H_
#define GE_COMMON_FP16_T_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ge {
/**
 *@ingroup fp16 basic parameter
 *@brief   fp16 exponent bias
 */
#define FP16_EXP_BIAS (15)
/**
 *@ingroup fp16 basic parameter
 *@brief   the mantissa bit length of fp16 is 10
 */
#define FP16_MAN_LEN (10)
/**
 *@ingroup fp16 basic parameter
 *@brief   bit index of sign in fp16
 */
#define FP16_SIGN_INDEX (15)
/**
 *@ingroup fp16 basic parameter
 *@brief   exponent mask of fp16     (  11111 00000 00000)
 */
#define FP16_EXP_MASK (0x7C00)
/**
 *@ingroup fp16 basic parameter
 *@brief   mantissa mask of fp16     (        11111 11111)
 */
#define FP16_MAN_MASK (0x03FF)
/**
 *@ingroup fp16 basic parameter
 *@brief   clear bit of mantissa of fp16(   1 00000 00000)
 */
#define FP16_MAN_HIDE_BIT (0x0400)
/**
 *@ingroup fp16 basic parameter
 *@brief   maximum value            (0111 1011 1111 1111)
 */
#define FP16_MAX (0x7BFF)
/**
 *@ingroup fp16 basic parameter
 *@brief   maximum value            (0111 1011 1111 1111)
 */
/**
 *@ingroup fp16 basic parameter
 *@brief   maximum exponent value of fp16 is 15(11111)
 */
#define FP16_MAX_EXP (0x001F)
/**
 *@ingroup fp16 basic parameter
 *@brief   maximum mantissa value of fp16(11111 11111)
 */
#define FP16_MAX_MAN (0x03FF)
/**
 *@ingroup fp16 basic operator
 *@brief   get sign of fp16
 */
#define FP16_EXTRAC_SIGN(x) (((x) >> 15) & 1)
/**
 *@ingroup fp16 basic operator
 *@brief   get exponent of fp16
 */
#define FP16_EXTRAC_EXP(x) (((x) >> 10) & FP16_MAX_EXP)
/**
 *@ingroup fp16 basic operator
 *@brief   get mantissa of fp16
 */
#define FP16_EXTRAC_MAN(x) ((x & 0x3FF) | (((((x) >> 10) & 0x1F) > 0 ? 1 : 0) * 0x400))
/**
 *@ingroup fp16 basic operator
 *@brief   constructor of fp16 from sign exponent and mantissa
 */
#define FP16_CONSTRUCTOR(s, e, m) (((s) << FP16_SIGN_INDEX) | ((e) << FP16_MAN_LEN) | ((m)&FP16_MAX_MAN))
/**
 *@ingroup fp16 special value judgment
 *@brief   whether a fp16 is invalid
 */
#define FP16_IS_INVALID(x) ((x & FP16_EXP_MASK) == FP16_EXP_MASK)

/**
 *@ingroup fp32 basic parameter
 *@brief   fp32 exponent bias
 */
#define FP32_EXP_BIAS (127)
/**
 *@ingroup fp32 basic parameter
 *@brief   the mantissa bit length of float/fp32 is 23
 */
#define FP32_MAN_LEN (23)
/**
 *@ingroup fp32 basic parameter
 *@brief   bit index of sign in float/fp32
 */
#define FP32_SIGN_INDEX (31)
/**
 *@ingroup fp32 basic parameter
 *@brief   sign mask of fp32         (1 0000 0000  0000 0000 0000 0000 000)
 */
#define FP32_SIGN_MASK (0x80000000u)
/**
 *@ingroup fp32 basic parameter
 *@brief   exponent mask of fp32     (  1111 1111  0000 0000 0000 0000 000)
 */
#define FP32_EXP_MASK (0x7F800000u)
/**
 *@ingroup fp32 basic parameter
 *@brief   mantissa mask of fp32     (             1111 1111 1111 1111 111)
 */
#define FP32_MAN_MASK (0x007FFFFFu)
/**
 *@ingroup fp32 basic parameter
 *@brief   hide bit of mantissa of fp32      (  1  0000 0000 0000 0000 000)
 */
#define FP32_MAN_HIDE_BIT (0x00800000u)
/**
 *@ingroup fp32 basic parameter
 *@brief   absolute maximum value    (0 1111 1111  1111 1111 1111 1111 111)
 */
#define FP32_ABS_MAX (0x7FFFFFFFu)
/**
 *@ingroup fp32 basic parameter
 *@brief   maximum mantissa value of fp32    (1111 1111 1111 1111 1111 111)
 */
#define FP32_MAX_MAN (0x7FFFFF)
/**
 *@ingroup fp32 basic operator
 *@brief   constructor of fp32 from sign exponent and mantissa
 */
#define FP32_CONSTRUCTOR(s, e, m) (((s) << FP32_SIGN_INDEX) | ((e) << FP32_MAN_LEN) | ((m)&FP32_MAX_MAN))
/**
 *@ingroup fp64 basic parameter
 *@brief   the mantissa bit length of double/fp64 is 52
 */
#define FP64_MAN_LEN (52)
/**
 *@ingroup fp64 basic parameter
 *@brief   bit index of sign in double/fp64 is 63
 */
#define FP64_SIGN_INDEX (63)
/**
 *@ingroup fp64 basic parameter
 *@brief   sign mask of fp64                 (1 000                   (total 63bits 0))
 */
#define FP64_SIGN_MASK (0x8000000000000000LLu)
/**
 *@ingroup fp64 basic parameter
 *@brief   exponent mask of fp64            (0 1 11111 11111  0000?-?-(total 52bits 0))
 */
#define FP64_EXP_MASK (0x7FF0000000000000LLu)
/**
 *@ingroup fp64 basic parameter
 *@brief   mantissa mask of fp64            (                 1111?-?-(total 52bits 1))
 */
#define FP64_MAN_MASK (0x000FFFFFFFFFFFFFLLu)
/**
 *@ingroup fp64 basic parameter
 *@brief   hide bit of mantissa of fp64     (               1 0000?-?-(total 52bits 0))
 */
#define FP64_MAN_HIDE_BIT (0x0010000000000000LLu)
/**
 *@ingroup integer special value judgment
 *@brief   maximum positive value of int8_t            (0111 1111)
 */
#define INT8_T_MAX (0x7F)
/**
 *@ingroup integer special value judgment
 *@brief   maximum positive value of int32_t           (0111 1111 1111 1111 1111 1111 1111 1111)
 */
#define INT32_T_MAX (0x7FFFFFFFu)
/**
 *@ingroup integer special value judgment
 *@brief   maximum value of a data with 32 bits length (1111 1111 1111 1111 1111 1111 1111 1111)
 */
#define BIT_LEN32_MAX (0xFFFFFFFFu)
/**
 *@ingroup fp16_t enum
 *@brief   round mode of last valid digital
 */
typedef enum TagFp16RoundMode {
  ROUND_TO_NEAREST = 0, /**< round to nearest even */
  ROUND_BY_TRUNCATED,   /**< round by truncated    */
  ROUND_MODE_RESERVED,
} fp16RoundMode_t;

/**
 *@ingroup fp16_t
 *@brief   Half precision float
 *         bit15:       1 bit SIGN      +---+-----+------------+
 *         bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
 *         bit0-9:      10bit MAN       +---+-----+------------+
 *
 */
using fp16_t = struct TagFp16 {
  uint16_t val;

 public:
  /**
   *@ingroup fp16_t constructor
   *@brief   Constructor without any param(default constructor)
   */
  TagFp16(void) { val = 0x0u; }
  /**
   *@ingroup fp16_t constructor
   *@brief   Constructor with an uint16_t value
   */
  TagFp16(const uint16_t &ui_val) : val(ui_val) {}
  /**
   *@ingroup fp16_t constructor
   *@brief   Constructor with a fp16_t object(copy constructor)
   */
  TagFp16(const TagFp16 &fp) : val(fp.val) {}

  /**
   *@ingroup fp16_t copy assign
   *@brief   copy assign
   */
  TagFp16 &operator=(const TagFp16 &fp);
  /**
   *@ingroup fp16_t math evaluation operator
   *@param [in] fVal float object to be converted to fp16_t
   *@brief   Override basic evaluation operator to convert float to fp16_t
   *@return  Return fp16_t result from fVal
   */
  TagFp16 &operator=(const float &fVal);
  /**
   *@ingroup fp16_t math evaluation operator
   *@param [in] iVal int32_t object to be converted to fp16_t
   *@brief   Override basic evaluation operator to convert int32_t to fp16_t
   *@return  Return fp16_t result from iVal
   */
  TagFp16 &operator=(const int32_t &iVal);
  /**
   *@ingroup fp16_t math conversion
   *@brief   Override convert operator to convert fp16_t to float/fp32
   *@return  Return float/fp32 value of fp16_t
   */
  operator float() const;

  /**
   *@ingroup fp16_t math conversion
   *@brief   Override convert operator to convert fp16_t to int32_t
   *@return  Return int32_t value of fp16_t
   */
  operator int32_t() const;

  /**
   *@ingroup fp16_t math conversion
   *@brief   Convert fp16_t to float/fp32
   *@return  Return float/fp32 value of fp16_t
   */
  float toFloat() const;

  /**
   *@ingroup fp16_t math conversion
   *@brief   Convert fp16_t to int32_t
   *@return  Return int32_t value of fp16_t
   */
  int32_t toInt32() const;
};
inline bool operator>(const TagFp16 &lhs, const TagFp16 &rhs) { return lhs.toFloat() > rhs.toFloat(); }
inline bool operator<(const TagFp16 &lhs, const TagFp16 &rhs) { return lhs.toFloat() < rhs.toFloat(); }
inline bool operator==(const TagFp16 &lhs, const TagFp16 &rhs) { return lhs.toFloat() == rhs.toFloat(); }
inline bool operator!=(const TagFp16 &lhs, const TagFp16 &rhs) { return lhs.toFloat() != rhs.toFloat(); }

/**
 *@ingroup fp16_t public method
 *@param [in]     val signature is negative
 *@param [in|out] s   sign of fp16_t object
 *@param [in|out] e   exponent of fp16_t object
 *@param [in|out] m   mantissa of fp16_t object
 *@brief   Extract the sign, exponent and mantissa of a fp16_t object
 */
void ExtractFP16(const uint16_t &val, uint16_t *s, int16_t *e, uint16_t *m);

/**
 *@ingroup fp16_t public method
 *@param [in] bit0    whether the last preserved bit is 1 before round
 *@param [in] bit1    whether the abbreviation's highest bit is 1
 *@param [in] bit_left whether the abbreviation's bits which not contain highest bit grater than 0
 *@param [in] man     mantissa of a fp16_t or float number, support types: uint16_t/uint32_t/uint64_t
 *@param [in] shift   abbreviation bits
 *@brief    Round fp16_t or float mantissa to nearest value
 *@return   Returns true if round 1,otherwise false;
 */
template <typename T>
T ManRoundToNearest(bool bit0, bool bit1, bool bit_left, T man, uint16_t shift = 0) {
  man = (man >> shift) + ((bit1 && (bit_left || bit0)) ? 1 : 0);
  return man;
}

/**
 *@ingroup fp16_t public method
 *@param [in] man    mantissa of a float number, support types: uint16_t/uint32_t/uint64_t
 *@brief   Get bit length of a uint32_t number
 *@return  Return bit length of man
 */
template <typename T>
int16_t GetManBitLength(T man) {
  int16_t len = 0;
  while (man) {
    man >>= 1;
    len++;
  }
  return len;
}
};  // namespace ge

#endif  // GE_COMMON_FP16_T_H_
