/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#include "formats/format_transfers/format_transfer_fractal_z_tbe.h"

#include "formats/utils/formats_definitions.h"
#include "formats/utils/formats_trans_utils.h"
#include "framework/common/util.h"

namespace ge {
namespace formats {
namespace {
enum class NdDimIndex {
  k2dC, k2dN, k2dDimsNum
};

/**
 * FZ represents the weight of convolution,.
 * After the conversion to two-dimensional matrix, the memory arrangement is small n and large Z.
 * If 4D(eg.NCHW) is used to represent convolution kernel, N is width, HWC is height.
 *
 * frac_z axises: (C1*H*W, No, Ni, C0), which Ni = 16, C0 = 16/32, No = Ceil(N/Ni), C1 = Ceil(C/C0)
 * @return
 */
Status TransShapeNdToFz(const std::vector<int64_t> &src_shape, const DataType data_type,
                        std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, static_cast<int64_t>(NdDimIndex::k2dDimsNum))) {
    GELOGE(FAILED, "src_shape is valid");
    return FAILED;  // Only support 2D to fracz
  }

  const int64_t c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    GELOGE(FAILED, "data_type is valid");
    return FAILED;
  }

  const int64_t h = 1;
  const int64_t w = 1;  // The shape conversion in 2d format is the same as 1,1,c,n
  const int64_t c = src_shape.at(static_cast<size_t>(NdDimIndex::k2dC));
  const int64_t n = src_shape.at(static_cast<size_t>(NdDimIndex::k2dN));
  const int64_t c1 = Ceil(c, c0);
  const int64_t no = Ceil(n, static_cast<int64_t>(kNiSize));

  dst_shape.clear();
  dst_shape.push_back(h * w * c1);
  dst_shape.push_back(no);
  dst_shape.push_back(kNiSize);
  dst_shape.push_back(c0);
  if (!IsShapeValid(dst_shape)) {
    GELOGE(FAILED, "dst_shape is valid");
    return FAILED;
  }
  return SUCCESS;
}

Status TransFormatNdToFz(const TransArgs &args, TransResult &result) {
  int64_t dst_size = 1;
  for (const auto dim : args.dst_shape) {
    dst_size *= dim;
  }

  const int64_t data_size = GetSizeByDataType(args.src_data_type);
  dst_size *= data_size;
  if (dst_size == 0) {
    result.length = static_cast<size_t>(dst_size);
    return SUCCESS;
  }

  const std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  GE_CHECK_NOTNULL(dst);

  const int64_t c = args.src_shape[static_cast<size_t>(NdDimIndex::k2dC)];
  const int64_t n = args.src_shape[static_cast<size_t>(NdDimIndex::k2dN)];
  const int64_t n1n0 = Ceil(n, static_cast<int64_t>(kNiSize)) * kNiSize;
  const int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  const int64_t c1 = Ceil(c, c0);
  const int64_t hwn1n0c0 = n1n0 * c0;
  for (int64_t c1i = 0; c1i < c1; ++c1i) {
    for (int64_t n1n0i = 0; n1n0i < n1n0; ++n1n0i) {
      for (int64_t c0i = 0; c0i < c0; ++c0i) {
        const int64_t dst_idx = (c1i * hwn1n0c0) + (n1n0i * c0) + c0i;
        const int64_t dst_offset = dst_idx * data_size;
        const auto protected_size = ((dst_size - dst_offset) < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)) ?
                                    (dst_size - dst_offset) : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
        const bool pad_zero = (((c1i * c0) + c0i) >= c) || (n1n0i >= n);
        errno_t ret;
        if (pad_zero) {
          ret = memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0,
                         static_cast<size_t>(data_size));
        } else {
          const int64_t src_idx = (((c1i * c0) + c0i) * n) + n1n0i;
          ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), args.data + (src_idx * data_size),
                         static_cast<size_t>(data_size));
        }
        if (ret != EOK) {
          GELOGE(FAILED, "ret != EOK");
          return FAILED;
        }
      }
    }
  }

  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return SUCCESS;
}
}  // namespace

Status FormatTransferFractalZTbe::TransFormat(const TransArgs &args, TransResult &result) {
  std::vector<int64_t> expect_shape;
  const auto ret = TransShape(args.src_format, args.src_shape, args.src_data_type, args.dst_format, expect_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if ((!args.dst_shape.empty()) && (args.dst_shape != expect_shape)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "dst_shape id empty or valid");
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  return TransFormatNdToFz(args, result);
}

Status FormatTransferFractalZTbe::TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                                             const DataType data_type, const Format dst_format,
                                             std::vector<int64_t> &dst_shape) {
  const Format src_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(src_format)));
  const Format dst_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(dst_format)));
  if (GetSizeByDataType(data_type) <= 0) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  if ((src_primary_format == FORMAT_ND) && (dst_primary_format == FORMAT_FRACTAL_Z)) {
    return TransShapeNdToFz(src_shape, data_type, dst_shape);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferFractalZTbe, FORMAT_ND, FORMAT_FRACTAL_Z)
}  // namespace formats
}  // namespace ge
