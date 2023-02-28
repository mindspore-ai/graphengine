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

#include "common/fp16_t.h"

namespace ge {
/// @ingroup fp16_t global filed
/// @brief   round mode of last valid digital
const TagFp16RoundMode g_round_mode = TagFp16RoundMode::kRoundToNearest;

void ExtractFp16(const uint16_t &val, uint16_t &s, int16_t &e, uint16_t &m) {
  // 1.Extract
  s = Fp16ExtracSign(val);
  e = static_cast<int16_t>(Fp16ExtracExp(val));
  m = Fp16ExtracMan(val);
  // Denormal
  if (e == 0) {
    e = 1;
  }
}
/// @ingroup fp16_t static method
/// @param [in] man       truncated mantissa
/// @param [in] shift_out left shift bits based on ten bits
/// @brief   judge whether to add one to the result while converting fp16_t to other datatype
/// @return  Return true if add one, otherwise false
static bool Fp16IsRoundOne(const uint64_t man, const uint16_t trunc_len) {
  uint64_t mask0 = 0x4UL;
  uint64_t mask1 = 0x2UL;
  uint64_t mask2;
  const uint16_t shift_out = trunc_len - static_cast<uint16_t>(kDim2);
  mask0 = mask0 << shift_out;
  mask1 = mask1 << shift_out;
  mask2 = mask1 - 1UL;

  const bool last_bit = ((man & mask0) > 0UL);
  bool trunc_high = false;
  bool trunc_left = false;
  if (g_round_mode == TagFp16RoundMode::kRoundToNearest) {
    trunc_high = ((man & mask1) > 0UL);
    trunc_left = ((man & mask2) > 0UL);
  }
  return (trunc_high && (trunc_left || last_bit));
}
/// @ingroup fp16_t public method
/// @param [in] exp       exponent of fp16_t value
/// @param [in] man       exponent of fp16_t value
/// @brief   normalize fp16_t value
/// @return
static void Fp16NormalizeVal(int16_t &exp_val, uint16_t &man) {
  // set to invalid data
  if (exp_val >= static_cast<int16_t>(kFp16MaxExp)) {
    exp_val = static_cast<int16_t>(kFp16MaxExp);
    man = static_cast<uint16_t>(kFp16MaxMan);
  } else if ((exp_val == 0) && (man == kFp16ManHideBit)) {
    exp_val++;
    man = 0U;
  } else {
    // do nothing
  }
}

/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to float/fp32
/// @return  Return float/fp32 value of fp_val which is the value of fp16_t object
static float32_t Fp16ToFloatVal(const uint16_t &fp_val) {
  uint16_t hf_sign;
  uint16_t hf_man;
  int16_t hf_exp;
  ExtractFp16(fp_val, hf_sign, hf_exp, hf_man);

  while ((hf_man != 0U) && ((hf_man & kFp16ManHideBit) == 0U)) {
    hf_man <<= 1U;
    hf_exp--;
  }

  uint32_t e_ret;
  uint32_t m_ret;
  const uint32_t s_ret = hf_sign;
  if (hf_man == 0U) {
    e_ret = 0U;
    m_ret = 0U;
  } else {
    e_ret = (static_cast<uint32_t>(hf_exp) - static_cast<uint32_t>(kFp16ExpBias)) + kFp32ExpBias;
    m_ret = static_cast<uint32_t>(hf_man) & static_cast<uint32_t>(kFp16ManMask);
    m_ret = m_ret << static_cast<uint32_t>(kFp32ManLen - kFp16ManLen);
  }
  const uint32_t f_val = Fp32Constructor(s_ret, e_ret, m_ret);
  const auto p_ret_v = PtrToPtr<const uint32_t, const float32_t>(&f_val);

  return *p_ret_v;
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to double/fp64
/// @return  Return double/fp64 value of fp_val which is the value of fp16_t object
static float64_t Fp16ToDouble(const uint16_t &fp_val) {
  uint16_t hf_sign;
  uint16_t hf_man;
  int16_t hf_exp;
  ExtractFp16(fp_val, hf_sign, hf_exp, hf_man);

  while ((hf_man != 0U) && ((hf_man & kFp16ManHideBit) == 0U)) {
    hf_man <<= 1U;
    hf_exp--;
  }

  uint64_t e_ret;
  uint64_t m_ret;
  const uint64_t s_ret = hf_sign;
  if (hf_man == 0U) {
    e_ret = 0UL;
    m_ret = 0UL;
  } else {
    e_ret = (static_cast<uint64_t>(hf_exp) - static_cast<uint64_t>(kFp16ExpBias)) + kFp64ExpBias;
    m_ret = static_cast<uint64_t>(hf_man) & static_cast<uint64_t>(kFp16ManMask);
    m_ret = m_ret << static_cast<uint64_t>(kFp64ManLen - kFp16ManLen);
  }
  const uint64_t f_val = (s_ret << kFp64SignIndex) | (e_ret << kFp64ManLen) | (m_ret);
  const auto p_ret_v = PtrToPtr<const uint64_t, const float64_t>(&f_val);

  return *p_ret_v;
}
/// @ingroup fp16_t static method
/// @param [in] s_ret       sign of fp16_t value
/// @param [in] long_int_m   man uint64_t value of fp16_t object
/// @param [in] shift_out   shift offset
/// @brief   calculate uint8 value by sign,man and shift offset
/// @return Return uint8 value of fp16_t object
static uint8_t GetUint8ValByMan(uint8_t s_ret, const uint64_t &long_int_m, const uint16_t &shift_out) {
  bool need_round = Fp16IsRoundOne(long_int_m, shift_out + kFp16ManLen);
  auto m_ret = static_cast<uint8_t>((long_int_m >> static_cast<uint32_t>(kFp16ManLen + shift_out)) & kBitLen8Max);
  need_round = need_round && (((s_ret == 0U) && (m_ret < kInt8Max)) ||
		              ((s_ret == 1U) && (static_cast<int32_t>(m_ret) <= kInt8Max)));
  if (need_round) {
    m_ret++;
  }
  if (s_ret != 0U) {
    m_ret = static_cast<uint8_t>(~m_ret) + 1U;
  }
  if (m_ret == 0U) {
    s_ret = 0U;
  }
  return ((static_cast<uint8_t>(s_ret << static_cast<uint8_t>(kBitShift7))) | (m_ret));
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to int8_t
/// @return  Return int8_t value of fp_val which is the value of fp16_t object
static int8_t Fp16ToInt8(const uint16_t &fp_val) {
  int8_t ret;
  uint8_t ret_v;
  // 1.get s_ret and shift it to bit0.
  const uint8_t s_ret = static_cast<uint8_t>(Fp16ExtracSign(fp_val));
  // 2.get hf_e and hf_m
  uint16_t hf_e = Fp16ExtracExp(fp_val);
  const uint16_t hf_m = Fp16ExtracMan(fp_val);

  if (Fp16IsDenorm(fp_val)) {  // Denormalized number
    ret_v = 0U;
    ret = *(PtrToPtr<uint8_t, int8_t>(&ret_v));
    return ret;
  }

  uint64_t long_int_m = hf_m;
  uint8_t overflow_flag = 0U;
  uint16_t shift_out = 0U;
  if (Fp16IsInvalid(fp_val)) {  // Inf or NaN
    overflow_flag = 1U;
  } else {
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1U;
        if ((s_ret == 1U) && (long_int_m >= 0x20000U)) {  // sign=1,negative number(<0)
          long_int_m = 0x20000U;                     // 10 0000 0000 0000 0000  10(fp16_t-man)+7(int8)=17bit
          overflow_flag = 1U;
          break;
        } else if ((s_ret != 1U) && (long_int_m >= 0x1FFFFU)) {  // sign=0,positive number(>0)
          long_int_m = 0x1FFFFU;                            // 01 1111 1111 1111 1111  10(fp16_t-man)+7(int8)
          overflow_flag = 1U;
          break;
        } else {
          // do nothing
        }
      } else {
        hf_e++;
        shift_out++;
      }
    }
  }
  if (overflow_flag != 0U) {
    ret_v = static_cast<uint8_t>(kInt8Max) + s_ret;
  } else {
    // Generate final result
    ret_v = GetUint8ValByMan(s_ret, long_int_m, shift_out);
  }

  ret = *(PtrToPtr<uint8_t, int8_t>(&ret_v));
  return ret;
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to uint8_t
/// @return  Return uint8_t value of fp_val which is the value of fp16_t object
static uint8_t Fp16ToUInt8(const uint16_t &fp_val) {
  uint8_t m_ret = 0U;
  // 1.get s_ret and shift it to bit0.
  const uint8_t s_ret = static_cast<uint8_t>(Fp16ExtracSign(fp_val));
  // 2.get hf_e and hf_m
  uint16_t hf_e = Fp16ExtracExp(fp_val);
  const uint16_t hf_m = Fp16ExtracMan(fp_val);

  if (Fp16IsDenorm(fp_val)) {  // Denormalized number
    return 0U;
  }

  if (Fp16IsInvalid(fp_val)) {  // Inf or NaN
    m_ret = kBitLen8Max;
  } else {
    uint64_t long_int_m = hf_m;
    uint8_t overflow_flag = 0U;
    uint16_t shift_out = 0U;
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1U;
        if (long_int_m >= 0x40000UL) {  // overflow 0100 0000 0000 0000 0000
          long_int_m = 0x3FFFFUL;       // 11 1111 1111 1111 1111   10(fp16_t-man)+8(uint8)=18bit
          overflow_flag = 1U;
          m_ret = kBitLen8Max;
          break;
        }
      } else {
        hf_e++;
        shift_out++;
      }
    }
    if (overflow_flag == 0U) {
      const bool need_round = Fp16IsRoundOne(long_int_m, shift_out + kFp16ManLen);
      m_ret = static_cast<uint8_t>((long_int_m >> static_cast<uint32_t>(kFp16ManLen + shift_out)) & kBitLen8Max);
      if (need_round && (m_ret != kBitLen8Max)) {
        m_ret++;
      }
    }
  }

  if (s_ret == 1U) {  // Negative number
    m_ret = 0U;
  }
  // m_ret equal to final result
  return m_ret;
}
/// @ingroup fp16_t static method
/// @param [in] s_ret       sign of fp16_t value
/// @param [in] long_int_m   man uint64_t value of fp16_t object
/// @param [in] shift_out   shift offset
/// @brief   calculate uint16 value by sign,man and shift offset
/// @return Return uint16 value of fp16_t object
static uint16_t GetUint16ValByMan(uint16_t s_ret, const uint64_t &long_int_m, const uint16_t &shift_out) {
  const bool need_round = Fp16IsRoundOne(long_int_m, shift_out + kFp16ManLen);
  auto m_ret = static_cast<uint16_t>((long_int_m >> static_cast<uint32_t>(kFp16ManLen + shift_out)) & kBitLen16Max);
  if (need_round && (static_cast<int32_t>(m_ret) < kInt16Max)) {
    m_ret++;
  }
  if (s_ret != 0U) {
    m_ret = static_cast<uint16_t>(~m_ret) + 1U;
  }
  if (m_ret == 0U) {
    s_ret = 0U;
  }
  return ((static_cast<uint16_t>(s_ret << static_cast<uint16_t>(kBitShift15))) | (m_ret));
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to int16_t
/// @return  Return int16_t value of fp_val which is the value of fp16_t object
static int16_t Fp16ToInt16(const uint16_t &fp_val) {
  int16_t ret;
  uint16_t ret_v;
  // 1.get s_ret and shift it to bit0.
  const uint16_t s_ret = Fp16ExtracSign(fp_val);
  // 2.get hf_e and hf_m
  uint16_t hf_e = Fp16ExtracExp(fp_val);
  const uint16_t hf_m = Fp16ExtracMan(fp_val);

  if (Fp16IsDenorm(fp_val)) {  // Denormalized number
    ret_v = 0U;
    ret = static_cast<int16_t>(*(PtrToPtr<uint16_t, uint8_t>(&ret_v)));
    return ret;
  }

  uint64_t long_int_m = hf_m;
  uint8_t overflow_flag = 0U;
  uint16_t shift_out = 0U;
  if (Fp16IsInvalid(fp_val)) {  // Inf or NaN
    overflow_flag = 1U;
  } else {
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1U;
        if ((s_ret == 1U) && (long_int_m > 0x2000000UL)) {  // sign=1,negative number(<0)
          long_int_m = 0x2000000UL;                    // 10(fp16_t-man)+15(int16)=25bit
          overflow_flag = 1U;
          break;
        } else if ((s_ret != 1U) && (long_int_m >= 0x1FFFFFFUL)) {  // sign=0,positive number(>0) Overflow
          long_int_m = 0x1FFFFFFUL;                            // 10(fp16_t-man)+15(int16)=25bit
          overflow_flag = 1U;
          break;
        } else {
          // do nothing
        }
      } else {
        hf_e++;
        shift_out++;
      }
    }
  }
  if (overflow_flag != 0U) {
    ret_v = static_cast<uint16_t>(kInt16Max) + s_ret;
  } else {
    // Generate final result
    ret_v = GetUint16ValByMan(s_ret, long_int_m, shift_out);
  }
  ret = *(PtrToPtr<uint16_t, int16_t>(&ret_v));
  return ret;
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to uint16_t
/// @return  Return uint16_t value of fp_val which is the value of fp16_t object
static uint16_t Fp16ToUInt16(const uint16_t &fp_val) {
  uint16_t m_ret = 0U;
  // 1.get s_ret and shift it to bit0.
  const uint16_t s_ret = Fp16ExtracSign(fp_val);
  // 2.get hf_e and hf_m
  uint16_t hf_e = Fp16ExtracExp(fp_val);
  const uint16_t hf_m = Fp16ExtracMan(fp_val);

  if (Fp16IsDenorm(fp_val)) {  // Denormalized number
    return 0U;
  }

  if (Fp16IsInvalid(fp_val)) {  // Inf or NaN
    m_ret = kBitLen8Max;
  } else {
    uint64_t long_int_m = hf_m;
    uint16_t shift_out = 0U;
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1U;
      } else {
        hf_e++;
        shift_out++;
      }
    }
    const bool need_round = Fp16IsRoundOne(long_int_m, shift_out + kFp16ManLen);
    m_ret = static_cast<uint16_t>((long_int_m >> static_cast<uint32_t>(kFp16ManLen + shift_out)) & kBitLen16Max);
    if (need_round && (m_ret != kBitLen16Max)) {
      m_ret++;
    }
  }

  if (s_ret == 1U) {  // Negative number
    m_ret = 0U;
  }
  // m_ret equal to final result
  return m_ret;
}
/// @ingroup fp16_t math convertion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to int32_t
/// @return  Return int32_t value of fp_val which is the value of fp16_t object
static int32_t Fp16ToInt32(const uint16_t &fp_val) {
  uint32_t ret_v;
  // 1.get s_ret and shift it to bit0.
  uint32_t s_ret = static_cast<uint32_t>(Fp16ExtracSign(fp_val));
  // 2.get hf_e and hf_m
  uint16_t hf_e = Fp16ExtracExp(fp_val);
  const uint16_t hf_m = Fp16ExtracMan(fp_val);

  if (Fp16IsInvalid(fp_val)) {  // Inf or NaN
    ret_v = s_ret + 0x7FFFFFFFU;
  } else {
    uint64_t long_int_m = hf_m;
    uint16_t shift_out = 0U;

    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1U;
      } else {
        hf_e++;
        shift_out++;
      }
    }
    const bool need_round = Fp16IsRoundOne(long_int_m, shift_out + kFp16ManLen);
    auto m_ret = static_cast<uint32_t>((long_int_m >> static_cast<uint32_t>(kFp16ManLen + shift_out)) & kBitLen32Max);
    if (need_round && (m_ret < 0x7FFFFFFFU)) {
      m_ret++;
    }

    if (s_ret == 1U) {
      m_ret = (~m_ret) + 1U;
    }
    if (m_ret == 0U) {
      s_ret = 0U;
    }
    // Generate final result
    ret_v = (s_ret << static_cast<uint32_t>(kBitShift31)) | (m_ret);
  }

  return *(PtrToPtr<uint32_t, int32_t>(&ret_v));
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to uint32_t
/// @return  Return uint32_t value of fp_val which is the value of fp16_t object
static uint32_t Fp16ToUInt32(const uint16_t &fp_val) {
  uint32_t m_ret;
  // 1.get s_ret and shift it to bit0.
  const uint32_t s_ret = static_cast<uint32_t>(Fp16ExtracSign(fp_val));
  // 2.get hf_e and hf_m
  uint16_t hf_e = Fp16ExtracExp(fp_val);
  const uint16_t hf_m = Fp16ExtracMan(fp_val);

  if (Fp16IsDenorm(fp_val)) {  // Denormalized number
    return 0U;
  }

  if (Fp16IsInvalid(fp_val)) {  // Inf or NaN
    m_ret = kBitLen8Max;
  } else {
    uint64_t long_int_m = hf_m;
    uint16_t shift_out = 0U;
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1U;
      } else {
        hf_e++;
        shift_out++;
      }
    }
    const bool need_round = Fp16IsRoundOne(long_int_m, shift_out + kFp16ManLen);
    m_ret = static_cast<uint32_t>(long_int_m >> static_cast<uint32_t>(kFp16ManLen + shift_out)) & kBitLen32Max;
    if (need_round && (m_ret != kBitLen32Max)) {
      m_ret++;
    }
  }

  if (s_ret == 1U) {  // Negative number
    m_ret = 0U;
  }
  // m_ret equal to final result
  return m_ret;
}
static uint16_t Fp16AddCalVal(const uint16_t &s_ret, int16_t e_ret, uint16_t m_ret, uint32_t m_trunc,
                              const uint16_t shift_out) {
  const uint16_t m_min = kFp16ManHideBit << shift_out;
  const uint16_t m_max = m_min << 1U;
  // Denormal
  while ((m_ret < m_min) && (e_ret > 0)) {  // the value of m_ret should not be smaller than 2^23
    m_ret = m_ret << 1U;
    m_ret += static_cast<uint16_t>((kFp32SignMask & m_trunc) >> kFp32SignIndex);
    m_trunc = m_trunc << 1U;
    e_ret = e_ret - 1;
  }
  while (m_ret >= m_max) {  // the value of m_ret should be smaller than 2^24
    m_trunc = m_trunc >> 1U;
    m_trunc = m_trunc | (kFp32SignMask * (m_ret & 1U));
    m_ret = m_ret >> 1U;
    e_ret = e_ret + 1;
  }

  const bool b_last_bit = ((m_ret & 1U) > 0U);
  const bool b_trunc_high = (g_round_mode == TagFp16RoundMode::kRoundToNearest) && ((m_trunc & kFp32SignMask) > 0U);
  const bool b_trunc_left = (g_round_mode == TagFp16RoundMode::kRoundToNearest) && ((m_trunc & kFp32AbsMax) > 0U);
  m_ret = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, m_ret, shift_out);
  while (m_ret >= m_max) {
    m_ret = m_ret >> 1U;
    e_ret = e_ret + 1;
  }

  if ((e_ret == 0) && (m_ret <= m_max)) {
    m_ret = m_ret >> 1U;
  }
  Fp16NormalizeVal(e_ret, m_ret);
  const uint16_t ret = Fp16Constructor(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  return ret;
}
/// @ingroup fp16_t math operator
/// @param [in] v_1 left operator value of fp16_t object
/// @param [in] v_2 right operator value of fp16_t object
/// @brief   Performing fp16_t addition
/// @return  Return fp16_t result of adding this and fp
static uint16_t Fp16Add(const uint16_t v_1, const uint16_t v_2) {
  uint16_t s_a;
  uint16_t s_b;
  int16_t e_a;
  int16_t e_b;
  uint32_t m_a;
  uint32_t m_b;
  uint16_t m_a_tmp;
  uint16_t m_b_tmp;
  uint16_t shift_out = 0U;
  // 1.Extract
  ExtractFp16(v_1, s_a, e_a, m_a_tmp);
  ExtractFp16(v_2, s_b, e_b, m_b_tmp);
  m_a = m_a_tmp;
  m_b = m_b_tmp;

  uint16_t sum;
  uint16_t s_ret;
  if (s_a != s_b) {
    ReverseMan(s_a > 0U, m_a);
    ReverseMan(s_b > 0U, m_b);
    sum = static_cast<uint16_t>(GetManSum(e_a, m_a, e_b, m_b));
    s_ret = (sum & kFp16SignMask) >> kFp16SignIndex;
    ReverseMan(static_cast<bool>(s_ret > 0U), m_a);
    ReverseMan(static_cast<bool>(s_ret > 0U), m_b);
  } else {
    sum = static_cast<uint16_t>(GetManSum(e_a, m_a, e_b, m_b));
    s_ret = s_a;
  }

  if (sum == 0U) {
    shift_out = 3U;  // shift to left 3 bits
    m_a = m_a << shift_out;
    m_b = m_b << shift_out;
  }

  uint32_t m_trunc = 0U;
  const int16_t e_ret = std::max(e_a, e_b);
  const int16_t e_tmp = static_cast<int16_t>(std::abs(e_a - e_b));
  if (e_a > e_b) {
    m_trunc = (m_b << (static_cast<uint32_t>(kBitShift32) - static_cast<uint32_t>(e_tmp)));
    m_b = RightShift(m_b, e_tmp);
  } else if (e_a < e_b) {
    m_trunc = (m_a << (static_cast<uint32_t>(kBitShift32) - static_cast<uint32_t>(e_tmp)));
    m_a = RightShift(m_a, e_tmp);
  } else {
      // do nothing
  }
  // calculate mantissav
  const auto m_ret = static_cast<uint16_t>(m_a + m_b);
  return Fp16AddCalVal(s_ret, e_ret, m_ret, m_trunc, shift_out);
}
/// @ingroup fp16_t math operator
/// @param [in] v_1 left operator value of fp16_t object
/// @param [in] v_2 right operator value of fp16_t object
/// @brief   Performing fp16_t subtraction
/// @return  Return fp16_t result of subtraction fp from this
static uint16_t Fp16Sub(const uint16_t v_1, const uint16_t v_2) {
  // Reverse
  const uint16_t tmp = (static_cast<uint16_t>(~(v_2)) & kFp16SignMask) | (v_2 & kFp16AbsMax);
  return Fp16Add(v_1, tmp);
}
/// @ingroup fp16_t math operator
/// @param [in] v_1 left operator value of fp16_t object
/// @param [in] v_2 right operator value of fp16_t object
/// @brief   Performing fp16_t multiplication
/// @return  Return fp16_t result of multiplying this and fp
static uint16_t Fp16Mul(const uint16_t v_1, const uint16_t v_2) {
  uint16_t s_a;
  uint16_t s_b;
  int16_t e_a;
  int16_t e_b;
  uint32_t m_a;
  uint32_t m_b;
  uint16_t s_ret;
  uint16_t m_ret;
  int16_t e_ret;
  uint32_t mul_m;
  uint16_t m_a_tmp;
  uint16_t m_b_tmp;
  // 1.Extract
  ExtractFp16(v_1, s_a, e_a, m_a_tmp);
  ExtractFp16(v_2, s_b, e_b, m_b_tmp);
  m_a = m_a_tmp;
  m_b = m_b_tmp;

  e_ret = ((e_a + e_b) - static_cast<int16_t>(kFp16ExpBias)) - static_cast<int16_t>(kDim10);
  mul_m = m_a * m_b;
  s_ret = s_a ^ s_b;

  const uint32_t m_min = kFp16ManHideBit;
  const uint32_t m_max = static_cast<uint16_t>(m_min << 1U);
  uint32_t m_trunc = 0U;
  // the value of m_ret should not be smaller than 2^23
  while ((mul_m < m_min) && (e_ret > 1)) {
    mul_m = mul_m << 1U;
    e_ret = e_ret - 1;
  }
  while ((mul_m >= m_max) || (e_ret < 1)) {
    m_trunc = m_trunc >> 1U;
    m_trunc = m_trunc | (kFp32SignMask * (mul_m & 1U));
    mul_m = mul_m >> 1U;
    e_ret = e_ret + 1;
  }
  const bool b_last_bit = ((mul_m & 1U) > 0U);
  const bool b_trunc_high = (g_round_mode == TagFp16RoundMode::kRoundToNearest) && ((m_trunc & kFp32SignMask) > 0U);
  const bool b_trunc_left = (g_round_mode == TagFp16RoundMode::kRoundToNearest) && ((m_trunc & kFp32AbsMax) > 0U);
  mul_m = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, mul_m);

  while ((mul_m >= m_max) || (e_ret < 0)) {
    mul_m = mul_m >> 1U;
    e_ret = e_ret + 1;
  }

  if ((e_ret == 1) && (mul_m < kFp16ManHideBit)) {
    e_ret = 0;
  }
  m_ret = static_cast<uint16_t>(mul_m);

  Fp16NormalizeVal(e_ret, m_ret);

  const uint16_t ret = Fp16Constructor(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  return ret;
}

// operate
TagFp16 TagFp16::operator+(const TagFp16 fp) const {
  const uint16_t ret_val = Fp16Add(val, fp.val);
  const fp16_t ret(ret_val);
  return ret;
}
TagFp16 TagFp16::operator-(const TagFp16 fp) const {
  const uint16_t ret_val = Fp16Sub(val, fp.val);
  const TagFp16 ret(ret_val);
  return ret;
}
TagFp16 TagFp16::operator*(const TagFp16 fp) const {
  const uint16_t ret_val = Fp16Mul(val, fp.val);
  const fp16_t ret(ret_val);
  return ret;
}

// compare
bool TagFp16::operator==(const TagFp16 &fp) const {
  bool result = true;
  if (Fp16IsZero(val) && Fp16IsZero(fp.val)) {
    result = true;
  } else {
    result = ((val & kBitLen16Max) == (fp.val & kBitLen16Max));  // bit compare
  }
  return result;
}
bool TagFp16::operator>(const TagFp16 &fp) const {
  uint16_t s_a;
  uint16_t s_b;
  uint16_t e_a;
  uint16_t e_b;
  uint16_t m_a;
  uint16_t m_b;
  bool result = true;

  // 1.Extract
  s_a = Fp16ExtracSign(val);
  s_b = Fp16ExtracSign(fp.val);
  e_a = Fp16ExtracExp(val);
  e_b = Fp16ExtracExp(fp.val);
  m_a = Fp16ExtracMan(val);
  m_b = Fp16ExtracMan(fp.val);

  // Compare
  if ((s_a == 0U) && (s_b > 0U)) {  // +  -
    // -0=0
    result = !(Fp16IsZero(val) && Fp16IsZero(fp.val));
  } else if ((s_a == 0U) && (s_b == 0U)) {  // + +
    if (e_a > e_b) {                      // e_a - e_b >= 1; Va always larger than Vb
      result = true;
    } else if (e_a == e_b) {
      result = m_a > m_b;
    } else {
      result = false;
    }
  } else if ((s_a > 0U) && (s_b > 0U)) {  // - -    opposite to  + +
    if (e_a < e_b) {
      result = true;
    } else if (e_a == e_b) {
      result = m_a < m_b;
    } else {
      result = false;
    }
  } else {  // -  +
    result = false;
  }

  return result;
}
bool TagFp16::operator>=(const TagFp16 &fp) const {
  bool result = true;
  if ((*this) > fp) {
    result = true;
  } else if ((*this) == fp) {
    result = true;
  } else {
    result = false;
  }

  return result;
}
bool TagFp16::operator<=(const TagFp16 &fp) const {
  bool result = true;
  if ((*this) > fp) {
    result = false;
  } else {
    result = true;
  }

  return result;
}

// evaluation
TagFp16 &TagFp16::operator=(const TagFp16 &fp) {
  if (&fp == this) {
    return *this;
  }
  val = fp.val;
  return *this;
}
TagFp16 &TagFp16::operator=(const float32_t &f_val) {
  uint16_t s_ret;
  uint16_t m_ret;
  int16_t e_ret;
  uint32_t e_f;
  uint32_t m_f;
  const uint32_t ui32_v = *(PtrToPtr<const float32_t, const uint32_t>(&f_val));  // 1:8:23bit sign:exp:man
  uint32_t m_len_delta;

  s_ret = static_cast<uint16_t>((ui32_v & kFp32SignMask) >> kFp32SignIndex);  // 4Byte->2Byte
  e_f = (ui32_v & kFp32ExpMask) >> kFp32ManLen;                               // 8 bit exponent
  m_f = (ui32_v & kFp32ManMask);  // 23 bit mantissa dont't need to care about denormal
  m_len_delta = kFp32ManLen - kFp16ManLen;

  bool need_round = false;
  // Exponent overflow/NaN converts to signed inf/NaN
  if (e_f > 0x8FU) {  // 0x8Fu:142=127+15
    e_ret = static_cast<int16_t>(kFp16MaxExp) - 1;
    m_ret = kFp16MaxMan;
  } else if (e_f <= 0x70U) {  // 0x70u:112=127-15 Exponent underflow converts to denormalized half or signed zero
    e_ret = 0;
    if (e_f >= 0x67U) {  // 0x67:103=127-24 Denormal
      m_f = (m_f | kFp32ManHideBit);
      const uint16_t shift_out = kFp32ManLen;
      const uint64_t m_tmp = (static_cast<uint64_t>(m_f)) << (e_f - 0x67U);

      need_round = Fp16IsRoundOne(m_tmp, shift_out);
      m_ret = static_cast<uint16_t>(m_tmp >> shift_out);
      if (need_round) {
        m_ret++;
      }
    } else if ((e_f == 0x66U) && (m_f > 0U)) {  // 0x66:102 Denormal 0<f_v<min(Denormal)
      m_ret = 1U;
    } else {
      m_ret = 0U;
    }
  } else {  // Regular case with no overflow or underflow
    e_ret = static_cast<int16_t>(e_f - 0x70U);

    need_round = Fp16IsRoundOne(static_cast<uint64_t>(m_f), static_cast<uint16_t>(m_len_delta));
    m_ret = static_cast<uint16_t>(m_f >> m_len_delta);
    if (need_round) {
      m_ret++;
    }
    if ((m_ret & kFp16ManHideBit) != 0U) {
      e_ret++;
    }
  }

  Fp16NormalizeVal(e_ret, m_ret);
  val = Fp16Constructor(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  return *this;
}
static void SetValByUint32Val(const uint32_t &input_val, const uint16_t &sign, uint16_t &ret_val) {
  int16_t e_ret;
  uint32_t m_tmp = (input_val & kFp32AbsMax);
  const uint16_t len = static_cast<uint16_t>(GetManBitLength(m_tmp));
  if (len > static_cast<uint16_t>(kDim11)) {
    e_ret = kFp16ExpBias + kFp16ManLen;
    uint32_t m_trunc = 0U;
    uint32_t trunc_mask = 1U;
    const uint16_t e_tmp = len - static_cast<uint16_t>(kDim11);
    for (uint16_t i = 1U; i < e_tmp; i++) {
      trunc_mask = (trunc_mask << 1U) + 1U;
    }
    m_trunc = (m_tmp & trunc_mask) << static_cast<uint32_t>((static_cast<uint16_t>(kBitShift32) - e_tmp));
    for (uint16_t i = 0U; i < e_tmp; i++) {
      m_tmp = (m_tmp >> 1U);
      e_ret = e_ret + 1;
    }
    const bool b_last_bit = ((m_tmp & 1U) > 0U);
    const bool b_trunc_high = (g_round_mode == TagFp16RoundMode::kRoundToNearest) && ((m_trunc & kFp32SignMask) > 0U);
    const bool b_trunc_left = (g_round_mode == TagFp16RoundMode::kRoundToNearest) && ((m_trunc & kFp32AbsMax) > 0U);
    m_tmp = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, m_tmp);
    const uint32_t m_max = static_cast<uint32_t>(kFp16ManHideBit << 1U);
    while ((m_tmp >= m_max) || (e_ret < 0)) {
      m_tmp = m_tmp >> 1U;
      e_ret = e_ret + 1;
    }
    if (e_ret >= kFp16MaxExp) {
      e_ret = static_cast<int16_t>(kFp16MaxExp) - 1;
      m_tmp = kFp16MaxMan;
    }
  } else {
    e_ret = static_cast<int16_t>(kFp16ExpBias);
    m_tmp = m_tmp << static_cast<uint32_t>(static_cast<uint16_t>(kDim11) - len);
    e_ret = e_ret + static_cast<int16_t>(len - 1U);
  }
  const auto m_ret = static_cast<uint16_t>(m_tmp);
  ret_val = Fp16Constructor(sign, static_cast<uint16_t>(e_ret), m_ret);
}
TagFp16 &TagFp16::operator=(const int32_t &i_val) {
  if (i_val == 0) {
    val = 0U;
  } else {
    uint32_t ui_val = *(PtrToPtr<const int32_t, const uint32_t>(&i_val));
    const auto s_ret = static_cast<uint16_t>(ui_val >> static_cast<uint32_t>(kBitShift31));
    if (s_ret != 0U) {
      int32_t iValM = -i_val;
      ui_val = *(PtrToPtr<int32_t, uint32_t>(&iValM));
    }
    SetValByUint32Val(ui_val, s_ret, val);
  }
  return *this;
}
TagFp16 &TagFp16::operator=(const float64_t &d_val) {
  uint16_t s_ret;
  uint16_t m_ret;
  int16_t e_ret;
  uint64_t e_d;
  uint64_t m_d;
  const uint64_t ui64_v = *(PtrToPtr<const float64_t, const uint64_t>(&d_val));  // 1:11:52bit sign:exp:man
  uint32_t m_len_delta;

  s_ret = static_cast<uint16_t>((ui64_v & kFp64SignMask) >> static_cast<uint16_t>(kFp64SignIndex));  // 4Byte
  e_d = (ui64_v & kFp64ExpMask) >> static_cast<uint32_t>(kFp64ManLen);                               // 10 bit exponent
  m_d = (ui64_v & kFp64ManMask);                                              // 52 bit mantissa
  m_len_delta = kFp64ManLen - kFp16ManLen;

  bool need_round = false;
  // Exponent overflow/NaN converts to signed inf/NaN
  if (e_d >= 0x410U) {  // 0x410:1040=1023+16
    e_ret = static_cast<int16_t>(kFp16MaxExp) - 1;
    m_ret = kFp16MaxMan;
    val = Fp16Constructor(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  } else if (e_d <= 0x3F0U) {  // Exponent underflow converts to denormalized half or signed zero
    // 0x3F0:1008=1023-15
    // Signed zeros, denormalized floats, and floats with small
    // exponents all convert to signed zero half precision.
    e_ret = 0;
    if (e_d >= 0x3E7U) {  // 0x3E7u:999=1023-24 Denormal
      // Underflows to a denormalized value
      m_d = (kFp64ManHideBit | m_d);
      const uint16_t shift_out = kFp64ManLen;
      const uint64_t m_tmp = (static_cast<uint64_t>(m_d)) << (e_d - 0x3E7U);

      need_round = Fp16IsRoundOne(m_tmp, shift_out);
      m_ret = static_cast<uint16_t>(m_tmp >> shift_out);
      if (need_round) {
        m_ret++;
      }
    } else if ((e_d == 0x3E6U) && (m_d > 0U)) {
      m_ret = 1U;
    } else {
      m_ret = 0U;
    }
  } else {  // Regular case with no overflow or underflow
    e_ret = static_cast<int16_t>(e_d) - 0x3F0;

    need_round = Fp16IsRoundOne(m_d, static_cast<uint16_t>(m_len_delta));
    m_ret = static_cast<uint16_t>(m_d >> m_len_delta);
    if (need_round) {
      m_ret++;
    }
    if ((m_ret & kFp16ManHideBit) != 0U) {
      e_ret++;
    }
  }

  Fp16NormalizeVal(e_ret, m_ret);
  val = Fp16Constructor(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  return *this;
}

// convert
TagFp16::operator float32_t() const {
  return Fp16ToFloatVal(val);
}
TagFp16::operator float64_t() const {
  return Fp16ToDouble(val);
}
TagFp16::operator int8_t() const {
  return Fp16ToInt8(val);
}
TagFp16::operator uint8_t() const {
  return Fp16ToUInt8(val);
}
TagFp16::operator int16_t() const {
  return Fp16ToInt16(val);
}
TagFp16::operator uint16_t() const {
  return Fp16ToUInt16(val);
}
TagFp16::operator int32_t() const {
  return Fp16ToInt32(val);
}
TagFp16::operator uint32_t() const {
  return Fp16ToUInt32(val);
}
// Cannot be used, just in order to solve the compile error
TagFp16::operator int64_t() const {
  return 0;
}
// Cannot be used, just in order to solve the compile error
TagFp16::operator uint64_t() const {
  return 0U;
}

float32_t fp16_t::ToFloat() const {
  return Fp16ToFloatVal(val);
}
float64_t fp16_t::ToDouble() const {
  return Fp16ToDouble(val);
}
int8_t fp16_t::ToInt8() const {
  return Fp16ToInt8(val);
}
uint8_t fp16_t::ToUInt8() const {
  return Fp16ToUInt8(val);
}
int16_t fp16_t::ToInt16() const {
  return Fp16ToInt16(val);
}
uint16_t fp16_t::ToUInt16() const {
  return Fp16ToUInt16(val);
}
int32_t fp16_t::ToInt32() const {
  return Fp16ToInt32(val);
}
uint32_t fp16_t::ToUInt32() const {
  return Fp16ToUInt32(val);
}
}  // namespace ge
