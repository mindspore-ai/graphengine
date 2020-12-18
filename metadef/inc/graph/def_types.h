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

#ifndef INC_GRAPH_DEF_TYPES_H_
#define INC_GRAPH_DEF_TYPES_H_

#include <atomic>
#include <memory>
#include <vector>
#include "graph/attr_value_serializable.h"
#include "graph/buffer.h"
namespace ge {
#define DEF_TYPE_DEC(type, name)                              \
  inline void set_##name(const type &value) { name = value; } \
  type *mutable_##name() { return &name; }

#define DEF_TYPE_HAS_DEC(type, name)                                                      \
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

#define DEF_TYPE_VEC_DEC(type, name)                                     \
  inline int name##_size() const { return name.size(); }                 \
  inline void clear_##name() { name.clear(); }                           \
  inline void set_##name(int index, type value) { name[index] = value; } \
  inline void add_##name(type value) { name.push_back(value); }          \
  inline std::vector<type> *mutable_##name() { return &name; }

#define DEF_TYPE_BYTES_DEC(name)                                                                                \
  inline void clear_##name() { name.ClearBuffer(); }                                                            \
  inline void set_##name(const void *value, size_t size) {                                                      \
    name = Buffer::CopyFrom((const uint8_t *)(value), size); }                                        \
  inline Buffer *mutable_##name() { return &name; }

struct CompressInfo {
 public:
  CompressInfo() {}
  CompressInfo(int32_t blockRow, int32_t blockCol, int32_t fractalK, int32_t fractalN, int32_t lastFractalK,
               int32_t lastFractalN, int32_t cubeSize, int32_t loadDir) {
    blockrow = blockRow;
    blockcol = blockCol;
    fractalk = fractalK;
    fractaln = fractalN;
    lastfractalk = lastFractalK;
    lastfractaln = lastFractalN;
    cubesize = cubeSize;
    loaddir = loadDir;
  }

  int32_t blockrow{0};      // Block row
  int32_t blockcol{0};      // Block col
  int32_t fractalk{0};      // Fractal K
  int32_t fractaln{0};      // Fractal N
  int32_t lastfractalk{0};  // K of last fractal
  int32_t lastfractaln{0};  // N of last fractal
  int32_t cubesize{0};      // Cube's length
  int32_t loaddir{0};       // Data load directtiono 0:col load 1:row load
  DEF_TYPE_DEC(int32_t, blockrow);
  DEF_TYPE_DEC(int32_t, blockcol);
  DEF_TYPE_DEC(int32_t, fractalk);
  DEF_TYPE_DEC(int32_t, fractaln);
  DEF_TYPE_DEC(int32_t, lastfractalk);
  DEF_TYPE_DEC(int32_t, lastfractaln);
  DEF_TYPE_DEC(int32_t, cubesize);
  DEF_TYPE_DEC(int32_t, loaddir);

  GE_SERIALIZABLE(blockrow, blockcol, fractalk, fractaln, lastfractalk, lastfractaln, cubesize, loaddir);
};

enum QuantizeScaleType { VECTOR_SCALE = 0, SCALAR_SCALE = 1 };
enum QuantizeScaleMode { NORMAL_MODE = 0, SQRT_MODE = 1 };
enum QuantizeAlgorithm {
  NON_OFFSET_ALGO = 0,
  HALF_OFFSET_ALGO = 1,
  ALL_OFFSET_ALGO = 2,
};
struct QuantizeFactor {
 public:
  // QuantizeScaleMode scale_mode;
  uint32_t scale_mode{0};
  Buffer scale_value;
  int64_t scale_offset{0};
  Buffer offset_data_value;
  int64_t offset_data_offset{0};
  Buffer offset_weight_value;
  int64_t offset_weight_offset{0};
  Buffer offset_pad_value;
  int64_t offset_pad_offset{0};

  DEF_TYPE_DEC(uint32_t, scale_mode);
  DEF_TYPE_BYTES_DEC(scale_value);

  DEF_TYPE_DEC(int64_t, scale_offset);
  DEF_TYPE_BYTES_DEC(offset_data_value);
  DEF_TYPE_DEC(int64_t, offset_data_offset);

  DEF_TYPE_BYTES_DEC(offset_weight_value);
  DEF_TYPE_DEC(int64_t, offset_weight_offset);
  DEF_TYPE_BYTES_DEC(offset_pad_value);
  DEF_TYPE_DEC(int64_t, offset_pad_offset);

  GE_SERIALIZABLE(scale_mode, scale_value, scale_offset, offset_data_value, offset_data_offset, offset_weight_value,
                  offset_weight_offset, offset_pad_value, offset_pad_offset)
};

static inline bool QuantizeFactorHasData(const QuantizeFactor &factor) {
  return factor.scale_value.GetSize() > 0 || factor.offset_data_value.GetSize() > 0 ||
         factor.offset_weight_value.GetSize() > 0 || factor.offset_pad_value.GetSize() > 0;
}

struct AllOffsetQuantizeInfo {
 public:
  AllOffsetQuantizeInfo() {}
  AllOffsetQuantizeInfo(float s, int32_t o) : scale(s), offset(o) {}
  float scale{0};
  int32_t offset{0};

  DEF_TYPE_DEC(float, scale);
  DEF_TYPE_DEC(int32_t, offset);

  GE_SERIALIZABLE(scale, offset)
};

struct QuantizeCalcFactor {
 public:
  Buffer offsetw;
  int64_t offsetw_offset{0};
  Buffer offsetd;
  int64_t offsetd_offset{0};
  Buffer scalereq;
  int64_t scaledreq_offset{0};
  Buffer offsetdnext;
  int64_t offsetdnext_offset{0};

  DEF_TYPE_BYTES_DEC(offsetw);
  DEF_TYPE_DEC(int64_t, offsetw_offset);
  DEF_TYPE_BYTES_DEC(offsetd);
  DEF_TYPE_DEC(int64_t, offsetd_offset);
  DEF_TYPE_BYTES_DEC(scalereq);
  DEF_TYPE_DEC(int64_t, scaledreq_offset);
  DEF_TYPE_BYTES_DEC(offsetdnext);
  DEF_TYPE_DEC(int64_t, offsetdnext_offset);

  GE_SERIALIZABLE(offsetw, offsetw_offset, offsetd, offsetd_offset, scalereq, scaledreq_offset, offsetdnext,
                  offsetdnext_offset);
};

static inline bool QuantizeFactorHasData(const QuantizeCalcFactor &factor) {
  return factor.offsetw.GetSize() > 0 || factor.offsetd.GetSize() > 0 || factor.scalereq.GetSize() > 0 ||
         factor.offsetdnext.GetSize() > 0;
}

struct QuantizeFactorParams {
  uint32_t quantize_algo{0};
  uint32_t scale_type{0};
  QuantizeFactor quantize_param;
  QuantizeFactor dequantize_param;
  QuantizeFactor requantize_param;
  QuantizeCalcFactor quantizecalc_param;
  DEF_TYPE_DEC(uint32_t, quantize_algo);
  DEF_TYPE_DEC(uint32_t, scale_type);
  DEF_TYPE_HAS_DEC(QuantizeFactor, quantize_param);
  DEF_TYPE_HAS_DEC(QuantizeFactor, dequantize_param);
  DEF_TYPE_HAS_DEC(QuantizeFactor, requantize_param);
  DEF_TYPE_HAS_DEC(QuantizeCalcFactor, quantizecalc_param);

  GE_SERIALIZABLE(quantize_algo, scale_type, quantize_param, dequantize_param, requantize_param, quantizecalc_param,
                  has_mutable_quantize_param, has_mutable_dequantize_param, has_mutable_requantize_param,
                  has_mutable_quantizecalc_param);
};

#undef DEF_TYPE_DEC
}  // namespace ge

#endif  // INC_GRAPH_DEF_TYPES_H_
