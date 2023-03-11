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

#include "formats/format_transfers/format_transfer_nhwc_nc1hwc0.h"

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
bool CheckDataTypeSupportedNhwcToNc1hwc0(const DataType &data_type) { return GetSizeByDataType(data_type) > 0; }

Status TransShapeNhwcToNc1hwc0(const std::vector<int64_t> &src_shape, const DataType data_type, const int64_t cube_size,
                               std::vector<int64_t> &dst_shape) {
  int64_t c0 = (cube_size <= 0) ? GetCubeSizeByDataType(data_type) : cube_size;
  if (c0 <= 0) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Get][Cube]Failed, the data type %s is invalid",
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to get cube size, the data type %s is invalid",
                      TypeUtils::DataTypeToSerialString(data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  dst_shape.clear();
  dst_shape.push_back(src_shape.at(kNhwcN));
  dst_shape.push_back(Ceil(src_shape.at(kNhwcC), c0));
  dst_shape.push_back(src_shape.at(kNhwcH));
  dst_shape.push_back(src_shape.at(kNhwcW));
  dst_shape.push_back(c0);
  if (!CheckShapeValid(dst_shape, kNc1hwc0DimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check invalid",
                      ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return SUCCESS;
}

Status CheckArgsForNhwcToNc1hwc0(const TransArgs &args) {
  if ((args.src_primary_format != FORMAT_NHWC) || (args.dst_primary_format != FORMAT_NC1HWC0)) {
    const std::string error = "Dose not support trans format from " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_primary_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_primary_format));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }
  if (!CheckDataTypeSupportedNhwcToNc1hwc0(args.src_data_type)) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Check][DataType]Failed from NHWC to NC1HWC0, "
           "invalid data type %s",
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_INNER_ERROR("E19999", "Failed to trans shape from NHWC to NC1HWC0, invalid data type %s",
                       TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (!CheckShapeValid(args.src_shape, kNhwcDimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, src shape %s",
           ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Src shape %s check invalid",
                      ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  if (!CheckShapeValid(args.dst_shape, kNc1hwc0DimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(args.dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check valid",
                      ShapeToString(args.dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  std::vector<int64_t> expect_dst_shape;
  const int64_t c0 = GetC0Value(static_cast<int32_t>(args.dst_format));
  const auto ret = TransShapeNhwcToNc1hwc0(args.src_shape, args.src_data_type, c0, expect_dst_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if (args.dst_shape != expect_dst_shape) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID,
           "[Trans][Format]Failed , the src shape %s and dst shape %s are not compatible. "
           "expect dst shape %s",
           ShapeToString(args.src_shape).c_str(), ShapeToString(args.dst_shape).c_str(),
           ShapeToString(expect_dst_shape).c_str());
    REPORT_CALL_ERROR("E19999",  "Failed to trans format, the src shape %s and "
                      "dst shape %s are not compatible. expect dst shape %s",
                      ShapeToString(args.src_shape).c_str(), ShapeToString(args.dst_shape).c_str(),
                      ShapeToString(expect_dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  return SUCCESS;
}

Status GetDstDataAfterTransNhwcToNc1hwc0(const TransArgs &args, TransResult &result,
                                         const int32_t size, const int64_t total_size) {
  const std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[total_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allcoate][Memory]Failed, memory for dst buf %" PRId64 ", "
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

  const auto n = args.src_shape.at(kNhwcN);
  const auto h = args.src_shape.at(kNhwcH);
  const auto w = args.src_shape.at(kNhwcW);
  const auto c = args.src_shape.at(kNhwcC);
  const auto c1 = args.dst_shape.at(kNc1hwc0C1);
  const auto c0 = args.dst_shape.at(kNc1hwc0C0);
  const int64_t wc = w * c;
  const int64_t hwc = h * wc;
  const int64_t wc0 = w * c0;
  const int64_t hwc0 = h * wc0;
  const int64_t c1hwc0 = c1 * hwc0;

  for (int64_t n_idx = 0; n_idx < n; n_idx++) {
    const int64_t n_head_addr = n_idx * c1hwc0;
    for (int64_t c1_idx = 0; c1_idx < c1; c1_idx++) {
      const int64_t c1_head_addr = n_head_addr + (c1_idx * hwc0);
      for (int64_t h_idx = 0; h_idx < h; h_idx++) {
        const int64_t h_head_addr = c1_head_addr + (h_idx * wc0);
        for (int64_t w_idx = 0; w_idx < w; w_idx++) {
          const int64_t w_head_addr = h_head_addr + (w_idx * c0);
          for (int64_t c0_idx = 0; c0_idx < c0; c0_idx++) {
            const int64_t dst_idx = c0_idx + w_head_addr;
            const int64_t dst_offset = dst_idx * size;
            const auto protected_size = ((total_size - dst_offset) < static_cast<int64_t>(SECUREC_MEM_MAX_LEN))
                                            ? (total_size - dst_offset)
                                            : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
            GE_CHECK_GE(protected_size, 0);
            const int64_t c_idx = c0_idx + (c1_idx * c0);
            const int64_t src_idx = (n_idx * hwc) + (h_idx * wc) + (w_idx * c) + c_idx;

            if (c_idx < c) {
              const auto src_offset = src_idx * size;
              const auto ret =
                  memcpy_s(PtrAdd(dst.get(), static_cast<size_t>(total_size), static_cast<size_t>(dst_offset)),
                           static_cast<size_t>(protected_size),
                           args.data + src_offset, static_cast<size_t>(size));
              if (ret != EOK) {
                GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED,
                       "[Operate][Memory]Failed to copy data from NHWC[%" PRId64 ", %" PRId64 ", %" PRId64 ", "
                       "%" PRId64 "] " "offset %" PRId64 " to NC1HWC0[%" PRId64 ", %" PRId64 ", %" PRId64 ", "
                       "%" PRId64 ", %" PRId64 "] offset %" PRId64 " err-code %d", n_idx, h_idx, w_idx,
                       c_idx, src_offset, n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                REPORT_CALL_ERROR("E19999", "Failed to copy data from NHWC[%" PRId64 ", %" PRId64 ", %" PRId64 ", "
                                  "%" PRId64 "] " "offset %" PRId64 " to " "NC1HWC0[%" PRId64 ", %" PRId64 ", "
                                  "%" PRId64 ", %" PRId64 ", %" PRId64 "] offset %" PRId64 " err-code %d",
                                  n_idx, h_idx, w_idx, c_idx, src_offset,
                                  n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
              }
            } else {
              const auto ret =
                  memset_s(PtrAdd(dst.get(), static_cast<size_t>(total_size), static_cast<size_t>(dst_offset)),
                           static_cast<size_t>(protected_size),
                           0, static_cast<size_t>(size));
              if (ret != EOK) {
                GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED,
                       "[Operate][Memory]Failed to set 0 to "
                       "NC1HWC0[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 ", "
                       "%" PRId64 "] offset %" PRId64 " base err-code %d",
                       n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                REPORT_CALL_ERROR("E19999", "Failed to set 0 to "
                                  "NC1HWC0[%" PRId64 ", %" PRId64 ", %" PRId64 ", "
                                  "%" PRId64 ", %" PRId64 "] offset %" PRId64 " base err-code %d",
                                  n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
              }
            }
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

Status FormatTransferNhwcNc1hwc0::TransFormat(const TransArgs &args, TransResult &result) {
  Status ret = CheckArgsForNhwcToNc1hwc0(args);
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

    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Get][ShapeSize]Failed, "
           "total size %" PRId64 " from dst shape %s, src shape %s", total_size,
           ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "[Get][Shape]Failed, total size %" PRId64 " from "
                      "dst shape %s, src shape %s", total_size,
                      ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  GELOGD("Begin to trans format from NHWC to NC1HWC0, src shape %s, data type %s, dst shape %s, "
         "memory size %" PRId64 "", ShapeToString(args.src_shape).c_str(),
	 TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
         ShapeToString(args.dst_shape).c_str(), total_size);

  ret = GetDstDataAfterTransNhwcToNc1hwc0(args, result, size, total_size);
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

Status FormatTransferNhwcNc1hwc0::TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                                             const DataType data_type, const Format dst_format,
                                             std::vector<int64_t> &dst_shape) {
  const Format src_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(src_format)));
  (void)dst_format;
  const auto c0 = GetC0Value(static_cast<int32_t>(dst_format));
  if ((src_primary_format == FORMAT_NHWC) && CheckDataTypeSupportedNhwcToNc1hwc0(data_type)) {
    if (!CheckShapeValid(src_shape, kNhwcDimsNum)) {
      GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, src shape %s",
             ShapeToString(src_shape).c_str());
      REPORT_CALL_ERROR("E19999", "Src shape %s check invalid",
                        ShapeToString(src_shape).c_str());
      return ACL_ERROR_GE_SHAPE_INVALID;
    }
    return TransShapeNhwcToNc1hwc0(src_shape, data_type, c0, dst_shape);
  } else if (src_primary_format != FORMAT_NHWC) {
    return ACL_ERROR_GE_FORMAT_INVALID;
  } else {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
}

REGISTER_FORMAT_TRANSFER(FormatTransferNhwcNc1hwc0, FORMAT_NHWC, FORMAT_NC1HWC0)
}  // namespace formats
}  // namespace ge
