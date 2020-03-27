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

#include "common/fp16_t.h"

#include "external/register/register_types.h"

namespace {
const int32_t kInt32SymbolShift = 31;
const int32_t kBitShift_32 = 32;
const int32_t kDim_2 = 2;
const int32_t kDim_11 = 11;
}  // namespace

namespace ge {
union Fp16ToFloatData {
  uint32_t uint_data;
  float float_data;
};

///
/// @ingroup fp16_t global filed
/// @brief   round mode of last valid digital
///
const fp16RoundMode_t g_round_mode = ROUND_TO_NEAREST;

void ExtractFP16(const uint16_t &val, uint16_t *s, int16_t *e, uint16_t *m) {
  // 1.Extract
  *s = static_cast<uint16_t>(FP16_EXTRAC_SIGN(val));
  *e = static_cast<uint16_t>(FP16_EXTRAC_EXP(val));
  *m = static_cast<uint16_t>(FP16_EXTRAC_MAN(val));

  // Denormal
  if ((*e) == 0) {
    *e = 1;
  }
}

///
/// @ingroup fp16_t static method
/// @param [in] man       truncated mantissa
/// @param [in] shift_out left shift bits based on ten bits
/// @brief   judge whether to add one to the result while converting fp16_t to other datatype
/// @return  Return true if add one, otherwise false
///
static bool IsRoundOne(uint64_t man, uint16_t trunc_len) {
  uint64_t mask0 = 0x4;
  uint64_t mask1 = 0x2;
  uint64_t mask2;
  uint16_t shift_out = static_cast<uint16_t>(trunc_len - kDim_2);
  mask0 = mask0 << shift_out;
  mask1 = mask1 << shift_out;
  mask2 = mask1 - 1;

  bool last_bit = ((man & mask0) > 0);
  bool trunc_high = false;
  bool trunc_left = false;
  if (g_round_mode == ROUND_TO_NEAREST) {
    trunc_high = ((man & mask1) > 0);
    trunc_left = ((man & mask2) > 0);
  }
  return (trunc_high && (trunc_left || last_bit));
}

///
/// @ingroup fp16_t public method
/// @param [in] exp       exponent of fp16_t value
/// @param [in] man       exponent of fp16_t value
/// @brief   normalize fp16_t value
/// @return
///
static void Fp16Normalize(int16_t &exp, uint16_t &man) {
  if (exp >= FP16_MAX_EXP) {
    exp = FP16_MAX_EXP - 1;
    man = FP16_MAX_MAN;
  } else if (exp == 0 && man == FP16_MAN_HIDE_BIT) {
    exp++;
    man = 0;
  }
}

// Evaluation
fp16_t &fp16_t::operator=(const fp16_t &fp) {
  if (&fp == this) {
    return *this;
  }
  val = fp.val;
  return *this;
}

fp16_t &fp16_t::operator=(const float &f_val) {
  uint16_t s_ret, m_ret;
  int16_t e_ret;
  uint32_t e_f, m_f;
  uint32_t ui32_v = *(reinterpret_cast<const uint32_t *>(&f_val));  // 1:8:23bit sign:exp:man
  uint32_t m_len_delta;

  s_ret = static_cast<uint16_t>((ui32_v & FP32_SIGN_MASK) >> FP32_SIGN_INDEX);  // 4Byte->2Byte
  e_f = (ui32_v & FP32_EXP_MASK) >> FP32_MAN_LEN;                               // 8 bit exponent
  m_f = (ui32_v & FP32_MAN_MASK);  // 23 bit mantissa dont't need to care about denormal
  m_len_delta = FP32_MAN_LEN - FP16_MAN_LEN;

  // Exponent overflow/NaN converts to signed inf/NaN
  if (e_f > 0x8Fu) {  // 0x8Fu:142=127+15
    e_ret = FP16_MAX_EXP - 1;
    m_ret = FP16_MAX_MAN;
  } else if (e_f <= 0x70u) {  // 0x70u:112=127-15 Exponent underflow converts to denormalized half or signed zero
    e_ret = 0;
    if (e_f >= 0x67) {  // 0x67:103=127-24 Denormal
      m_f = (m_f | FP32_MAN_HIDE_BIT);
      uint16_t shift_out = FP32_MAN_LEN;
      uint64_t m_tmp = (static_cast<uint64_t>(m_f)) << (e_f - 0x67);

      bool need_round = IsRoundOne(m_tmp, shift_out);
      m_ret = static_cast<uint16_t>(m_tmp >> shift_out);
      if (need_round) {
        m_ret++;
      }
    } else if (e_f == 0x66 && m_f > 0) {  // 0x66:102 Denormal 0<f_v<min(Denormal)
      m_ret = 1;
    } else {
      m_ret = 0;
    }
  } else {  // Regular case with no overflow or underflow
    e_ret = static_cast<int16_t>(e_f - 0x70u);

    bool need_round = IsRoundOne(m_f, static_cast<uint16_t>(m_len_delta));
    m_ret = static_cast<uint16_t>(m_f >> m_len_delta);
    if (need_round) {
      m_ret++;
    }
    if (m_ret & FP16_MAN_HIDE_BIT) {
      e_ret++;
    }
  }

  Fp16Normalize(e_ret, m_ret);
  val = FP16_CONSTRUCTOR(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  return *this;
}

fp16_t &fp16_t::operator=(const int32_t &i_val) {
  if (i_val == 0) {
    val = 0;
  } else {
    uint32_t ui_val = *(reinterpret_cast<const uint32_t *>(&i_val));
    uint16_t s_ret = static_cast<uint16_t>(ui_val >> kInt32SymbolShift);
    if (s_ret) {
      int32_t i_val_m = -i_val;
      ui_val = *(reinterpret_cast<uint32_t *>(&i_val_m));
    }
    int16_t e_ret;
    uint32_t m_tmp = (ui_val & FP32_ABS_MAX);
    uint32_t m_min = FP16_MAN_HIDE_BIT;
    uint32_t m_max = m_min << 1;
    int32_t len = static_cast<int32_t>(GetManBitLength(m_tmp));
    if (len > kDim_11) {
      e_ret = FP16_EXP_BIAS + FP16_MAN_LEN;
      uint32_t m_trunc = 0;
      uint32_t trunc_mask = 1;
      int32_t e_tmp = len - kDim_11;
      for (int i = 1; i < e_tmp; i++) {
        trunc_mask = (trunc_mask << 1) + 1;
      }
      m_trunc = (m_tmp & trunc_mask) << static_cast<uint32_t>(kBitShift_32 - e_tmp);
      for (int i = 0; i < e_tmp; i++) {
        m_tmp = (m_tmp >> 1);
        e_ret = e_ret + 1;
      }
      bool b_last_bit = ((m_tmp & 1) > 0);
      bool b_trunc_high = false;
      bool b_trunc_left = false;
      if (g_round_mode == ROUND_TO_NEAREST) {  // trunc
        b_trunc_high = ((m_trunc & FP32_SIGN_MASK) > 0);
        b_trunc_left = ((m_trunc & FP32_ABS_MAX) > 0);
      }
      m_tmp = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, m_tmp);
      while (m_tmp >= m_max || e_ret < 0) {
        m_tmp = m_tmp >> 1;
        e_ret = e_ret + 1;
      }
      if (e_ret >= FP16_MAX_EXP) {
        e_ret = FP16_MAX_EXP - 1;
        m_tmp = FP16_MAX_MAN;
      }
    } else {
      e_ret = FP16_EXP_BIAS;
      m_tmp = m_tmp << static_cast<uint32_t >(kDim_11 - len);
      e_ret = e_ret + (len - 1);
    }
    uint16_t m_ret = static_cast<uint16_t>(m_tmp);
    val = FP16_CONSTRUCTOR(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  }
  return *this;
}

///
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to float/fp32
/// @return  Return float/fp32 value of fp_val which is the value of fp16_t object
///
float Fp16ToFloat(const uint16_t &fp_val) {
  float ret;

  uint16_t hf_sign, hf_man;
  int16_t hf_exp;
  ExtractFP16(fp_val, &hf_sign, &hf_exp, &hf_man);

  while (hf_man && !(hf_man & FP16_MAN_HIDE_BIT)) {
    hf_man <<= 1;
    hf_exp--;
  }

  uint32_t s_ret, e_ret, m_ret, f_val;

  s_ret = hf_sign;
  if (!hf_man) {
    e_ret = 0;
    m_ret = 0;
  } else {
    e_ret = static_cast<uint32_t>(hf_exp - FP16_EXP_BIAS + FP32_EXP_BIAS);
    m_ret = hf_man & FP16_MAN_MASK;
    m_ret = m_ret << (FP32_MAN_LEN - FP16_MAN_LEN);
  }
  f_val = FP32_CONSTRUCTOR(s_ret, e_ret, m_ret);
  Fp16ToFloatData data;
  data.uint_data = f_val;
  ret = data.float_data;

  return ret;
}

///
/// @ingroup fp16_t math convertion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to int32_t
/// @return  Return int32_t value of fp_val which is the value of fp16_t object
///
int32_t Fp16ToInt32(const uint16_t &fp_val) {
  int32_t ret;
  uint32_t ret_v;
  uint32_t s_ret;
  uint16_t hf_e, hf_m;

  // 1.Get s_ret and shift it to bit0.
  s_ret = FP16_EXTRAC_SIGN(fp_val);
  // 2.Get hf_e and hf_m
  hf_e = FP16_EXTRAC_EXP(fp_val);
  hf_m = FP16_EXTRAC_MAN(fp_val);

  if (FP16_IS_INVALID(fp_val)) {  // Inf or NaN
    ret_v = INT32_T_MAX + s_ret;
  } else {
    uint64_t long_int_m = hf_m;
    uint16_t shift_out = 0;

    while (hf_e != FP16_EXP_BIAS) {
      if (hf_e > FP16_EXP_BIAS) {
        hf_e--;
        long_int_m = long_int_m << 1;
      } else {
        hf_e++;
        shift_out++;
      }
    }
    uint32_t m_ret;
    bool need_round = IsRoundOne(long_int_m, shift_out + FP16_MAN_LEN);
    m_ret = static_cast<uint32_t>((long_int_m >> (FP16_MAN_LEN + shift_out)) & BIT_LEN32_MAX);
    if (need_round && m_ret < INT32_T_MAX) {
      m_ret++;
    }

    if (s_ret == 1) {
      m_ret = (~m_ret) + 1;
    }
    if (m_ret == 0) {
      s_ret = 0;
    }
    // Generate final result
    ret_v = (s_ret << kInt32SymbolShift) | (m_ret);
  }

  ret = *(reinterpret_cast<int32_t *>(&ret_v));
  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY float fp16_t::toFloat() const { return Fp16ToFloat(val); }

int32_t fp16_t::toInt32() const { return Fp16ToInt32(val); }

// Convert
fp16_t::operator float() const { return Fp16ToFloat(val); }

fp16_t::operator int32_t() const { return Fp16ToInt32(val); }
}  // namespace ge
