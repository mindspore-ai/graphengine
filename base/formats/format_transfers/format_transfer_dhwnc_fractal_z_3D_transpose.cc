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
#include "formats/format_transfers/format_transfer_dhwnc_fractal_z_3D_transpose.h"

#include <securec.h>
#include <memory>

#include "formats/utils/formats_definitions.h"
#include "formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
namespace {
Status CheckDataTypeSupportForDhwncToFz3DT(const DataType dtype) {
  return (GetSizeByDataType(dtype) > 0) ? SUCCESS : UNSUPPORTED;
}

Status TransShapeToFzForDhwncToFz3DT(const int64_t d, const int64_t n, const int64_t c, const int64_t h,
                                     const int64_t w, const int64_t c0, std::vector<int64_t> &dst_shape) {
  if (c0 <= 0) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  const auto c1 = Ceil(c, c0);
  const auto no = Ceil(n, static_cast<int64_t>(kNiSize));

  dst_shape.clear();
  dst_shape.push_back(d * c1 * h * w);
  dst_shape.push_back(no);
  dst_shape.push_back(kNiSize);
  dst_shape.push_back(c0);

  return SUCCESS;
}

Status TransShapeDhwncToFz3DTranspose(const std::vector<int64_t> &src_shape, const Format &dst_format,
                                      std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kDhwncDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  const auto d = src_shape.at(kDhwncD);
  const auto h = src_shape.at(kDhwncH);
  const auto w = src_shape.at(kDhwncW);
  const auto n = src_shape.at(kDhwncN);
  const auto c = src_shape.at(kDhwncC);
  const auto c0 = GetC0Value(static_cast<int32_t>(dst_format));

  // exchange n c, normalize process with dhwcn to fraz3D
  return TransShapeToFzForDhwncToFz3DT(d, c, n, h, w, c0, dst_shape);
}
Status TransFormatDhwncToFz3DTranspose(const TransArgs &args, TransResult &result) {
  if (!CheckShapeValid(args.src_shape, kDhwncDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  const int64_t d = args.src_shape[kDhwncD];
  const int64_t h = args.src_shape[kDhwncH];
  const int64_t w = args.src_shape[kDhwncW];
  // exchange nc ,for normalize process with dhwcn to Fz3D
  const int64_t c = args.src_shape[kDhwncN];
  const int64_t n = args.src_shape[kDhwncC];
  const int64_t n1n0 = Ceil(n, static_cast<int64_t>(kNiSize)) * kNiSize;
  const int64_t c0 = GetC0Value(static_cast<int32_t>(args.dst_format));
  const int64_t c1 = Ceil(c, c0);

  const auto cn = c * n;
  const auto wcn = w * cn;
  const auto n1n0c0 = n1n0 * c0;
  const auto wn1n0c0 = w * n1n0c0;
  const auto hwn1n0c0 = h * wn1n0c0;
  const auto c1hwn1n0c0 = c1 * hwn1n0c0;

  const int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = 1;
  for (const auto dim : args.dst_shape) {
    dst_size *= dim;
  }
  dst_size *= data_size;
  if (dst_size == 0) {
    result.length = static_cast<size_t>(dst_size);
    return SUCCESS;
  }

  const std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][DSTMemory]Failed to allcoate memory "
           "for dst buf %" PRId64 " when trans format from %s to %s",
           dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to allcoate memory for dst buf %" PRId64 " "
                      "when trans format from %s to %s",
                      dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  for (int64_t di = 0; di < d; di++) {
    for (int64_t c1i = 0; c1i < c1; c1i++) {
      for (int64_t hi = 0; hi < h; hi++) {
        for (int64_t wi = 0; wi < w; wi++) {
          for (int64_t n1n0i = 0; n1n0i < n1n0; n1n0i++) {
            for (int64_t c0i = 0; c0i < c0; c0i++) {
              const int64_t dst_idx =
                  (di * c1hwn1n0c0) + (c1i * hwn1n0c0) + (hi * wn1n0c0) + (wi * n1n0c0) + (n1n0i * c0) + c0i;
              const int64_t dst_offset = dst_idx * data_size;
              const auto protected_size = ((dst_size - dst_offset) < static_cast<int64_t>(SECUREC_MEM_MAX_LEN))
                                          ? (dst_size - dst_offset)
                                          : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
              const auto pad_zero = (((c1i * c0) + c0i) >= c) || (n1n0i >= n);
              errno_t ret;
              if (pad_zero) {
                ret = memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0,
                               static_cast<size_t>(data_size));
              } else {
                const auto hwcn = h * wcn;
                const int64_t src_idx = (di * hwcn) + (hi * wcn) + (wi * cn) + (((c1i * c0) + c0i) * n) + n1n0i;
                ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size),
                               args.data + (src_idx * data_size), static_cast<size_t>(data_size));
              }
              if (ret != EOK) {
                GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Operate][DSTMemory]Failed at "
                       "offset %" PRId64 ", error-code %d, pad mode %d",
		       dst_offset, ret, static_cast<int32_t>(pad_zero));
                REPORT_CALL_ERROR("E19999", "Failed to operate dst memory at offset %" PRId64 ", "
                                  "error-code %d, pad mode %d", dst_offset, ret, static_cast<int32_t>(pad_zero));
                return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
              }
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

Status FormatTransferDhwncFractalZ3DTranspose::TransFormat(const TransArgs &args, TransResult &result) {
  GELOGD("Begin to trans format from %s to %s, src shape %s, data type %s, dst shape %s",
         TypeUtils::FormatToSerialString(args.src_format).c_str(),
         TypeUtils::FormatToSerialString(args.dst_format).c_str(), ShapeToString(args.src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(), ShapeToString(args.dst_shape).c_str());
  std::vector<int64_t> expect_shape;
  const auto ret = TransShape(args.src_format, args.src_shape, args.src_data_type, args.dst_format, expect_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if (!IsTransShapeDstCorrect(args, expect_shape)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  if ((args.src_primary_format == ge::FORMAT_DHWNC) && (args.dst_primary_format == ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE)) {
    return TransFormatDhwncToFz3DTranspose(args, result);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

Status FormatTransferDhwncFractalZ3DTranspose::TransShape(const Format src_format,
                                                          const std::vector<int64_t> &src_shape,
                                                          const DataType data_type,
                                                          const Format dst_format,
                                                          std::vector<int64_t> &dst_shape) {
  if (CheckDataTypeSupportForDhwncToFz3DT(data_type) != SUCCESS) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  const Format src_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(src_format)));
  const Format dst_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(dst_format)));
  if ((src_primary_format == FORMAT_DHWNC) && (dst_primary_format == FORMAT_FRACTAL_Z_3D_TRANSPOSE)) {
    return TransShapeDhwncToFz3DTranspose(src_shape, dst_format, dst_shape);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferDhwncFractalZ3DTranspose, FORMAT_DHWNC, FORMAT_FRACTAL_Z_3D_TRANSPOSE)
}  // namespace formats
}  // namespace ge
