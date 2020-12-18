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

#ifndef INC_GRAPH_USR_TYPES_H_
#define INC_GRAPH_USR_TYPES_H_

#include <atomic>
#include <memory>
#include <vector>
namespace ge {
#define USR_TYPE_DEC(type, name)                              \
  inline void set_##name(const type &value) { name = value; } \
  type *mutable_##name() { return &name; }

#define USR_TYPE_HAS_DEC(type, name)                                                      \
  inline void set_##name(const type &value) { name = value; }                             \
                                                                                          \
 private:                                                                                 \
  bool has_mutable_##name{false};                                                         \
                                                                                          \
 public:                                                                                  \
  bool has_##name() const { return (has_mutable_##name) || QuantizeFactorHasData(name); } \
  type *mutable_##name() {                                                                \
    has_mutable_##name = true;                                                            \
    return &name;                                                                         \
  }

#define USR_TYPE_BYTES_DEC(name)                                                \
  inline void clear_##name() { name.clear(); }                                  \
  inline void set_##name(const void *value, size_t size) {                      \
    name.assign(reinterpret_cast<uint8_t *>(const_cast<void *>(value)),         \
                reinterpret_cast<uint8_t *>(const_cast<void *>(value)) + size); \
  }

enum UsrQuantizeScaleType { USR_VECTOR_SCALE = 0, USR_SCALAR_SCALE = 1 };
enum UsrQuantizeScaleMode { USR_NORMAL_MODE = 0, USR_SQRT_MODE = 1 };
enum UsrQuantizeAlgorithm {
  USR_NON_OFFSET_ALGO = 0,
  USR_HALF_OFFSET_ALGO = 1,
  USR_ALL_OFFSET_ALGO = 2,
};

struct UsrQuantizeFactor {
 public:
  // QuantizeScaleMode scale_mode;
  UsrQuantizeScaleMode scale_mode{USR_NORMAL_MODE};
  std::vector<uint8_t> scale_value;
  int64_t scale_offset{0};
  std::vector<uint8_t> offset_data_value;
  int64_t offset_data_offset{0};
  std::vector<uint8_t> offset_weight_value;
  int64_t offset_weight_offset{0};
  std::vector<uint8_t> offset_pad_value;
  int64_t offset_pad_offset{0};

  USR_TYPE_DEC(UsrQuantizeScaleMode, scale_mode);
  USR_TYPE_BYTES_DEC(scale_value);

  USR_TYPE_DEC(int64_t, scale_offset);
  USR_TYPE_BYTES_DEC(offset_data_value);
  USR_TYPE_DEC(int64_t, offset_data_offset);

  USR_TYPE_BYTES_DEC(offset_weight_value);
  USR_TYPE_DEC(int64_t, offset_weight_offset);
  USR_TYPE_BYTES_DEC(offset_pad_value);
  USR_TYPE_DEC(int64_t, offset_pad_offset);
};

static inline bool QuantizeFactorHasData(const UsrQuantizeFactor &factor) {
  return factor.scale_value.size() > 0 || factor.offset_data_value.size() > 0 ||
         factor.offset_weight_value.size() > 0 || factor.offset_pad_value.size() > 0;
}

struct UsrQuantizeCalcFactor {
 public:
  std::vector<uint8_t> offsetw;
  int64_t offsetw_offset{0};
  std::vector<uint8_t> offsetd;
  int64_t offsetd_offset{0};
  std::vector<uint8_t> scalereq;
  int64_t scaledreq_offset{0};
  std::vector<uint8_t> offsetdnext;
  int64_t offsetdnext_offset{0};

  USR_TYPE_BYTES_DEC(offsetw);
  USR_TYPE_DEC(int64_t, offsetw_offset);
  USR_TYPE_BYTES_DEC(offsetd);
  USR_TYPE_DEC(int64_t, offsetd_offset);
  USR_TYPE_BYTES_DEC(scalereq);
  USR_TYPE_DEC(int64_t, scaledreq_offset);
  USR_TYPE_BYTES_DEC(offsetdnext);
  USR_TYPE_DEC(int64_t, offsetdnext_offset);
};

static inline bool QuantizeFactorHasData(const UsrQuantizeCalcFactor &factor) {
  return factor.offsetw.size() > 0 || factor.offsetd.size() > 0 || factor.scalereq.size() > 0 ||
         factor.offsetdnext.size() > 0;
}

struct UsrQuantizeFactorParams {
  UsrQuantizeAlgorithm quantize_algo{USR_NON_OFFSET_ALGO};
  UsrQuantizeScaleType scale_type{USR_VECTOR_SCALE};
  UsrQuantizeFactor quantize_param;
  UsrQuantizeFactor dequantize_param;
  UsrQuantizeFactor requantize_param;
  UsrQuantizeCalcFactor quantizecalc_param;
  USR_TYPE_DEC(UsrQuantizeAlgorithm, quantize_algo);
  USR_TYPE_DEC(UsrQuantizeScaleType, scale_type);
  USR_TYPE_HAS_DEC(UsrQuantizeFactor, quantize_param);
  USR_TYPE_HAS_DEC(UsrQuantizeFactor, dequantize_param);
  USR_TYPE_HAS_DEC(UsrQuantizeFactor, requantize_param);
  USR_TYPE_HAS_DEC(UsrQuantizeCalcFactor, quantizecalc_param);
};

#undef USR_TYPE_DEC
#undef USR_TYPE_HAS_DEC
#undef USR_TYPE_BYTES_DEC
}  // namespace ge

#endif  // INC_GRAPH_USR_TYPES_H_

