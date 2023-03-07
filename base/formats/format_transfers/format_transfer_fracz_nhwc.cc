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

#include "formats/format_transfers/format_transfer_fracz_nhwc.h"

#include <securec.h>
#include <memory>

#include "formats/utils/formats_definitions.h"
#include "formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
namespace {
bool CheckDataTypeSupportedForFracZToNhwc(const DataType data_type) { return GetSizeByDataType(data_type) > 0; }

Status CheckArgsForFracZToNhwc(const TransArgs &args) {
  const auto src_shape = args.src_shape;
  const auto dst_shape = args.dst_shape;
  if ((args.src_primary_format != FORMAT_FRACTAL_Z) || (args.dst_primary_format != FORMAT_NHWC)) {
    const std::string error = "Dose not support trans format from " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_primary_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_primary_format));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }
  if (!CheckDataTypeSupportedForFracZToNhwc(args.src_data_type)) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Check][DataType]Failed, "
           "shape from FORMAT_FRACTAL_Z to NCHW, invalid data type %s",
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_INNER_ERROR("E19999", "Failed to trans shape from FORMAT_FRACTAL_Z to NCHW, invalid data type %s",
                       TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (!CheckShapeValid(src_shape, kFracZDimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, src shape %s",
           ShapeToString(src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Src shape %s check invalid", ShapeToString(src_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  if (!CheckShapeValid(dst_shape, kNhwcDimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check invalid", ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  const int64_t c0 =  GetC0Value(static_cast<int32_t>(args.src_format));
  if (c0 < 0) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  const int64_t c1 = Ceil(dst_shape.at(kNhwcC), c0);
  const int64_t n0 = Ceil(dst_shape.at(kNhwcN), static_cast<int64_t>(kNiSize));
  if ((src_shape.at(kFracZHWC1) != (dst_shape.at(kNhwcH) * dst_shape.at(kNhwcW) * c1)) ||
      (src_shape.at(kFracZC0) != c0) || (src_shape.at(kFracZNi) != kNiSize) || (src_shape.at(kFracZN0) != n0)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID,
           "[Check][Shape]Failed to check relationship between src and dst shape, "
           "src shape %s, dst shape %s",
           ShapeToString(src_shape).c_str(), ShapeToString(dst_shape).c_str());
    REPORT_INNER_ERROR("E19999", "Failed to check relationship between src and dst shape, "
                       "src shape %s, dst shape %s",
                       ShapeToString(src_shape).c_str(), ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  return SUCCESS;
}

Status GetDstDataAfterTransForFracZToNhwc(const TransArgs &args, TransResult &result,
                                          const int32_t size, const int64_t total_size) {
  const std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[total_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Allocate][DSTMemory]Failed, memory for dst buf %" PRId64 ", "
           "shape %s when trans format from %s to %s",
           total_size, ShapeToString(args.dst_shape).c_str(),
           TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to alloc the memory for dst buf %" PRId64 ", "
                      "shape %s when trans format from %s to %s",
                      total_size, ShapeToString(args.dst_shape).c_str(),
                      TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  const auto n0 = args.src_shape.at(kFracZN0);
  const auto ni = args.src_shape.at(kFracZNi);
  const auto c0 = args.src_shape.at(kFracZC0);
  const auto h = args.dst_shape.at(kNhwcH);
  const auto w = args.dst_shape.at(kNhwcW);
  const auto c = args.dst_shape.at(kNhwcC);
  const auto n = args.dst_shape.at(kNhwcN);
  const int64_t nc = ni * n0;
  const int64_t ncc0 = nc * c0;
  const int64_t wncc0 = w * ncc0;
  const int64_t hwncc0 = h * wncc0;
  const int64_t wc = w * c;
  const int64_t hwc = h * wc;

  for (int64_t n_idx = 0; n_idx < n; n_idx++) {
    const int64_t n_head_addr = n_idx * hwc;
    for (int64_t h_idx = 0; h_idx < h; h_idx++) {
      const int64_t h_head_addr = n_head_addr + (h_idx * wc);
      for (int64_t w_idx = 0; w_idx < w; w_idx++) {
        const int64_t w_head_addr = h_head_addr + (w_idx * c);
        for (int64_t c_idx = 0; c_idx < c; c_idx++) {
          const int64_t dst_idx = w_head_addr + c_idx;
          const int64_t c1_idx = c_idx / c0;
          const int64_t c0_idx = c_idx % c0;
          const int64_t nc_idx = n_idx;
          const int64_t src_idx = (c1_idx * hwncc0) + (h_idx * wncc0) + (w_idx * ncc0) + (nc_idx * c0) + c0_idx;
          const auto src_offset = src_idx * size;
          const auto dst_offset = dst_idx * size;
          const auto protected_size = ((total_size - dst_offset) < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)) ?
                                      (total_size - dst_offset) : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
          GE_CHECK_GE(protected_size, 0);
          const auto ret =
              memcpy_s(PtrAdd(dst.get(), static_cast<size_t>(total_size), static_cast<size_t>(dst_offset)),
                       static_cast<size_t>(protected_size),
                       args.data + src_offset, static_cast<size_t>(size));
          if (ret != EOK) {
            GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED,
                   "[Operate][Memory]Failed to copy data from FracZ offset %" PRId64 " to "
                   "NCHW[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] offset %" PRId64 ", err-code %d",
                   src_offset, n_idx, c_idx, h_idx, w_idx, dst_offset, ret);
            REPORT_CALL_ERROR("E19999", "Failed to copy data from FracZ offset %" PRId64 " to "
                              "NCHW[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] offset %" PRId64 ", "
			      "err-code %d", src_offset, n_idx, c_idx, h_idx, w_idx, dst_offset, ret);
            return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
          }
        }
      }
    }
  }
  result.data = dst;
  result.length = static_cast<size_t>(total_size);
  return SUCCESS;
}
}  // namespace

Status FormatTransferFracZNhwc::TransFormat(const TransArgs &args, TransResult &result) {
  Status ret = CheckArgsForFracZToNhwc(args);
  if (ret != SUCCESS) {
    return ret;
  }
  const int32_t size = GetSizeByDataType(args.src_data_type);
  const auto total_size = GetItemNumByShape(args.dst_shape) * size;
  if (total_size <= 0) {
    const int64_t src_size = GetItemNumByShape(args.src_shape);
    if ((total_size == 0) && (src_size == 0)) {
      result.length = static_cast<size_t>(total_size);
      return SUCCESS;
    }

    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Get][ShapeSize]Failed, total size %" PRId64 " from dst shape %s, "
           "src shape %s", total_size,
           ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to get total size %" PRId64 " from dst shape %s, src shape %s",
                      total_size,
                      ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GELOGD("Begin to trans format from FracZ to NHWC, src shape %s, data type %s, dst shape %s, memory size %" PRId64 "",
         ShapeToString(args.src_shape).c_str(), TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
         ShapeToString(args.dst_shape).c_str(), total_size);
  ret = GetDstDataAfterTransForFracZToNhwc(args, result, size, total_size);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][Data]Failed, after trans, src shape %s, data type %s, "
           "dst shape %s, memory size %" PRId64 ", error_code %u",
           ShapeToString(args.src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
           ShapeToString(args.dst_shape).c_str(), total_size, ret);
    REPORT_CALL_ERROR("E19999", "Failed to get data after trans, src shape %s, data type %s, "
                      "dst shape %s, memory size %" PRId64 ", error_code %u",
                      ShapeToString(args.src_shape).c_str(),
                      TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
                      ShapeToString(args.dst_shape).c_str(), total_size, ret);
    return ret;
  }
  return SUCCESS;
}

Status FormatTransferFracZNhwc::TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                                           const DataType data_type,
                                           const Format dst_format, std::vector<int64_t> &dst_shape) {
  (void)src_format;
  (void)src_shape;
  (void)data_type;
  (void)dst_format;
  (void)dst_shape;
  GELOGD("The shape derivation from FracZ to NHWC is not unique. Trans shape in this direction is not supported");
  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferFracZNhwc, FORMAT_FRACTAL_Z, FORMAT_NHWC)
}  // namespace formats
}  // namespace ge
