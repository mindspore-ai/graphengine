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

#include "common/formats/format_transfers/format_transfer_fractal_z.h"

#include <securec.h>
#include <memory>

#include "common/debug/log.h"
#include "common/formats/utils/formats_definitions.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
namespace {
Status CheckDataTypeSupport(DataType data_type) { return GetSizeByDataType(data_type) > 0 ? SUCCESS : UNSUPPORTED; }

/**
 * FZ represents the weight of convolution,.
 * After the conversion to two-dimensional matrix, the memory arrangement is small n and large Z.
 * If 4D(eg.NCHW) is used to represent convolution kernel, N is width, HWC is height.
 *
 * frac_z axises: (C1*H*W, No, Ni, C0), which Ni = 16, C0 = 16/32, No = Ceil(N/Ni), C1 = Ceil(C/C0)
 * @return
 */
Status TransShapeToFz(int64_t n, int64_t c, int64_t h, int64_t w, DataType data_type, std::vector<int64_t> &dst_shape) {
  auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    return UNSUPPORTED;
  }

  auto c1 = Ceil(c, c0);
  auto no = Ceil(n, static_cast<int64_t>(kNiSize));

  dst_shape.clear();
  dst_shape.push_back(h * w * c1);
  dst_shape.push_back(no);
  dst_shape.push_back(kNiSize);
  dst_shape.push_back(c0);
  if (!IsShapeValid(dst_shape)) {
    GELOGE(PARAM_INVALID, "Failed to check dst shape %s", ShapeToString(dst_shape).c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status TransShapeNchwToFz(const std::vector<int64_t> &src_shape, DataType data_type, std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
    return PARAM_INVALID;
  }

  auto n = src_shape.at(kNchwN);
  auto c = src_shape.at(kNchwC);
  auto h = src_shape.at(kNchwH);
  auto w = src_shape.at(kNchwW);
  return TransShapeToFz(n, c, h, w, data_type, dst_shape);
}

Status TransShapeHwcnToFz(const std::vector<int64_t> &src_shape, DataType data_type, std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
    return PARAM_INVALID;
  }

  auto h = src_shape.at(kHwcnH);
  auto w = src_shape.at(kHwcnW);
  auto c = src_shape.at(kHwcnC);
  auto n = src_shape.at(kHwcnN);

  return TransShapeToFz(n, c, h, w, data_type, dst_shape);
}

Status TransShapeNhwcToFz(const std::vector<int64_t> &src_shape, DataType data_type, std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kNhwcDimsNum)) {
    return PARAM_INVALID;
  }

  auto n = src_shape.at(kNhwcN);
  auto h = src_shape.at(kNhwcH);
  auto w = src_shape.at(kNhwcW);
  auto c = src_shape.at(kNhwcC);

  return TransShapeToFz(n, c, h, w, data_type, dst_shape);
}

Status TransFormatFromNchwToFz(const TransArgs &args, TransResult &result) {
  int64_t n = args.src_shape.at(kNchwN);
  int64_t c = args.src_shape.at(kNchwC);
  int64_t h = args.src_shape.at(kNchwH);
  int64_t w = args.src_shape.at(kNchwW);

  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  int64_t c1 = Ceil(c, c0);

  int64_t hw = h * w;
  int64_t chw = c * hw;
  int64_t nchw = n * chw;
  int64_t hwc0 = hw * c0;

  // horizontal fractal matrix count (N)
  int64_t hf_cnt = Ceil(n, static_cast<int64_t>(kNiSize));
  // vertical fractal matrix count (C1HWC0)
  int64_t vf_cnt = c1 * hw;
  // elements count in one fractal
  int64_t fractal_ele_cnt = c0 * kNiSize;
  int64_t total_ele_cnt = hf_cnt * vf_cnt * fractal_ele_cnt;
  int size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = total_ele_cnt * size;
  GE_CHK_BOOL_EXEC_NOLOG(dst_size != 0, result.length = static_cast<size_t>(dst_size); return SUCCESS;);

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      dst == nullptr,
      GELOGE(OUT_OF_MEMORY, "Failed to trans format from %s to %s, can not alloc the memory for dst buf %ld",
             TypeUtils::FormatToSerialString(args.src_format).c_str(),
             TypeUtils::FormatToSerialString(args.dst_format).c_str(), dst_size);
      return OUT_OF_MEMORY;);

  for (int64_t vfi = 0; vfi < vf_cnt; vfi++) {
    // vertical fractal matrix base index
    auto vf_base_i = vfi * hf_cnt;
    for (int64_t hfi = 0; hfi < hf_cnt; hfi++) {
      // global fractal matrix index
      auto gfi = vf_base_i + hfi;
      auto src_n_offset = hfi * chw * kNiSize;
      auto src_f_offset = src_n_offset + vfi % hw + vfi / hw * hwc0;
      for (int64_t row = 0; row < c0; row++) {
        auto src_ci = vfi / hw * c0 + row;
        auto src_row_offset = src_f_offset + row * hw;
        for (int col = 0; col < kNiSize; col++) {
          auto src_ni = hfi * kNiSize + col;
          auto src_offset = src_row_offset + chw * col;
          // pad 0
          // 1. src_ni grater than n
          // 2. src_ci grater than c
          // 3. source address grater than original array size
          auto need_pad_zero = src_ni >= n || src_offset >= nchw || src_ci >= c;
          auto idx = gfi * fractal_ele_cnt + col * c0 + row;
          auto offset = idx * size;
          auto protected_size = dst_size - offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                    ? dst_size - offset
                                    : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
          errno_t ret = EOK;
          if (need_pad_zero) {
            ret = memset_s(dst.get() + offset, static_cast<size_t>(protected_size), 0, static_cast<size_t>(size));
          } else {
            if (protected_size < size) {
              std::string error = "Failed to operate the dst memory, protected_size is " +
                  FmtToStr(protected_size) + " and size is " + FmtToStr(size);
              GE_ERRORLOG_AND_ERRORMSG(INTERNAL_ERROR, error.c_str());
              return INTERNAL_ERROR;
            }
            char *dst_data = reinterpret_cast<char *>(dst.get() + offset);
            const char *src_data = reinterpret_cast<const char *>(args.data + src_offset * size);
            for (int64_t index = 0; index < size; index++) {
              *dst_data++ = *src_data++;
            }
          }
          if (ret != EOK) {
            GELOGE(INTERNAL_ERROR, "Failed to operate the dst memory at offset %ld, error-code %d pad mode %d", offset,
                   ret, need_pad_zero);
            return INTERNAL_ERROR;
          }
        }
      }
    }
  }

  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return SUCCESS;
}

Status TransFormatHwcnToFz(const TransArgs &args, TransResult &result) {
  int64_t h = args.src_shape[kHwcnH];
  int64_t w = args.src_shape[kHwcnW];
  int64_t c = args.src_shape[kHwcnC];
  int64_t n = args.src_shape[kHwcnN];
  int64_t n1n0 = Ceil(n, static_cast<int64_t>(kNiSize)) * kNiSize;
  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  int64_t c1 = Ceil(c, c0);

  auto cn = c * n;
  auto wcn = w * cn;
  auto n1n0c0 = n1n0 * c0;
  auto wn1n0c0 = w * n1n0c0;
  auto hwn1n0c0 = h * wn1n0c0;

  int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = 1;
  for (auto dim : args.dst_shape) {
    dst_size *= dim;
  }
  dst_size *= data_size;
  GE_CHK_BOOL_EXEC_NOLOG(dst_size != 0, result.length = static_cast<size_t>(dst_size); return SUCCESS;);

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      dst == nullptr,
      GELOGE(OUT_OF_MEMORY, "Failed to trans format from %s to %s, can not alloc the memory for dst buf %ld",
             TypeUtils::FormatToSerialString(args.src_format).c_str(),
             TypeUtils::FormatToSerialString(args.dst_format).c_str(), dst_size);
      return OUT_OF_MEMORY;);

  for (int64_t c1i = 0; c1i < c1; c1i++) {
    for (int64_t hi = 0; hi < h; hi++) {
      for (int64_t wi = 0; wi < w; wi++) {
        for (int64_t n1n0i = 0; n1n0i < n1n0; n1n0i++) {
          for (int64_t c0i = 0; c0i < c0; c0i++) {
            int64_t dst_idx = c1i * hwn1n0c0 + hi * wn1n0c0 + wi * n1n0c0 + n1n0i * c0 + c0i;
            int64_t dst_offset = dst_idx * data_size;
            auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                      ? dst_size - dst_offset
                                      : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
            auto pad_zero = ((c1i * c0 + c0i) >= c) || (n1n0i >= n);
            errno_t ret = EOK;
            if (pad_zero) {
              ret = memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0,
                             static_cast<size_t>(data_size));
            } else {
              if (protected_size < data_size) {
                GELOGE(INTERNAL_ERROR, "Failed to operate the dst memory, protected_size is %ld and size is %ld",
                       protected_size, data_size);
                return INTERNAL_ERROR;
              }
              int64_t src_idx = hi * wcn + wi * cn + (c1i * c0 + c0i) * n + n1n0i;
              char *dst_data = reinterpret_cast<char *>(dst.get() + dst_offset);
              const char *src_data = reinterpret_cast<const char *>(args.data + src_idx * data_size);
              for (int64_t index = 0; index < data_size; index++) {
                *dst_data++ = *src_data++;
              }
            }
            if (ret != EOK) {
              GELOGE(INTERNAL_ERROR, "Failed to operate the dst memory at offset %ld, error-code %d, pad mode %d",
                     dst_offset, ret, pad_zero);
              return INTERNAL_ERROR;
            }
          }
        }
      }
    }
  }

  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return SUCCESS;
}

Status TransFormatNhwcToFz(const TransArgs &args, TransResult &result) {
  int64_t n = args.src_shape[kNhwcN];
  int64_t h = args.src_shape[kNhwcH];
  int64_t w = args.src_shape[kNhwcW];
  int64_t c = args.src_shape[kNhwcC];
  auto wc = w * c;
  auto hwc = h * w * c;

  int64_t n1n0 = Ceil(n, static_cast<int64_t>(kNiSize)) * kNiSize;
  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  int64_t c1 = Ceil(c, c0);
  auto n1n0c0 = n1n0 * c0;
  auto wn1n0c0 = w * n1n0c0;
  auto hwn1n0c0 = h * wn1n0c0;

  int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = 1;
  for (auto dim : args.dst_shape) {
    dst_size *= dim;
  }
  dst_size *= data_size;
  GE_CHK_BOOL_EXEC_NOLOG(dst_size != 0, result.length = static_cast<size_t>(dst_size); return SUCCESS;);

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      dst == nullptr,
      GELOGE(OUT_OF_MEMORY, "Failed to trans format from %s to %s, can not alloc the memory for dst buf %ld",
             TypeUtils::FormatToSerialString(args.src_format).c_str(),
             TypeUtils::FormatToSerialString(args.dst_format).c_str(), dst_size);
      return OUT_OF_MEMORY;);

  for (int64_t c1i = 0; c1i < c1; c1i++) {
    for (int64_t hi = 0; hi < h; hi++) {
      for (int64_t wi = 0; wi < w; wi++) {
        for (int64_t n1n0i = 0; n1n0i < n1n0; n1n0i++) {
          for (int64_t c0i = 0; c0i < c0; c0i++) {
            int64_t dst_idx = c1i * hwn1n0c0 + hi * wn1n0c0 + wi * n1n0c0 + n1n0i * c0 + c0i;
            int64_t dst_offset = dst_idx * data_size;
            auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                      ? dst_size - dst_offset
                                      : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
            auto pad_zero = ((c1i * c0 + c0i) >= c) || (n1n0i >= n);
            errno_t ret = EOK;
            if (pad_zero) {
              ret = memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0,
                             static_cast<size_t>(data_size));
            } else {
              if (protected_size < data_size) {
                GELOGE(INTERNAL_ERROR, "Failed to operate the dst memory, protected_size is %ld and size is %ld",
                       protected_size, data_size);
                return INTERNAL_ERROR;
              }
              int64_t src_idx = n1n0i * hwc + hi * wc + wi * c + (c1i * c0 + c0i);
              char *dst_data = reinterpret_cast<char *>(dst.get() + dst_offset);
              const char *src_data = reinterpret_cast<const char *>(args.data + src_idx * data_size);
              for (int64_t index = 0; index < data_size; index++) {
                *dst_data++ = *src_data++;
              }
            }
            if (ret != EOK) {
              GELOGE(INTERNAL_ERROR, "Failed to operate the dst memory at offset %ld, error-code %d, pad mode %d",
                     dst_offset, ret, pad_zero);
              return INTERNAL_ERROR;
            }
          }
        }
      }
    }
  }

  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return SUCCESS;
}
}  // namespace

Status FormatTransferFractalZ::TransFormat(const TransArgs &args, TransResult &result) {
  GELOGD("Begin to trans format from %s to %s, src shape %s, data type %s, dst shape %s",
         TypeUtils::FormatToSerialString(args.src_format).c_str(),
         TypeUtils::FormatToSerialString(args.dst_format).c_str(), ShapeToString(args.src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(), ShapeToString(args.dst_shape).c_str());
  std::vector<int64_t> expect_shape;
  auto ret = TransShape(args.src_format, args.src_shape, args.src_data_type, args.dst_format, expect_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if (!IsTransShapeDstCorrect(args, expect_shape)) {
    return PARAM_INVALID;
  }

  if (args.src_format == FORMAT_NHWC && args.dst_format == FORMAT_FRACTAL_Z) {
    return TransFormatNhwcToFz(args, result);
  }

  if (args.src_format == FORMAT_HWCN && args.dst_format == FORMAT_FRACTAL_Z) {
    return TransFormatHwcnToFz(args, result);
  }

  if (args.src_format == FORMAT_NCHW && args.dst_format == FORMAT_FRACTAL_Z) {
    return TransFormatFromNchwToFz(args, result);
  }

  return UNSUPPORTED;
}

Status FormatTransferFractalZ::TransShape(Format src_format, const std::vector<int64_t> &src_shape, DataType data_type,
                                          Format dst_format, std::vector<int64_t> &dst_shape) {
  if (CheckDataTypeSupport(data_type) != SUCCESS) {
    return UNSUPPORTED;
  }

  if (src_format == FORMAT_NHWC && dst_format == FORMAT_FRACTAL_Z) {
    return TransShapeNhwcToFz(src_shape, data_type, dst_shape);
  }
  if (src_format == FORMAT_HWCN && dst_format == FORMAT_FRACTAL_Z) {
    return TransShapeHwcnToFz(src_shape, data_type, dst_shape);
  }
  if (src_format == FORMAT_NCHW && dst_format == FORMAT_FRACTAL_Z) {
    return TransShapeNchwToFz(src_shape, data_type, dst_shape);
  }

  return UNSUPPORTED;
}

REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_NCHW, FORMAT_FRACTAL_Z)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_HWCN, FORMAT_FRACTAL_Z)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_NHWC, FORMAT_FRACTAL_Z)
}  // namespace formats
}  // namespace ge
