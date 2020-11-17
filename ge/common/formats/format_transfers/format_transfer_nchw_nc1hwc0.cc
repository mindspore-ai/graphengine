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

#include "common/formats/format_transfers/format_transfer_nchw_nc1hwc0.h"

#include <securec.h>
#include <memory>

#include "common/formats/utils/formats_definitions.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
namespace {
Status TransShapeNchwToNc1hwc0(const std::vector<int64_t> &src_shape, DataType data_type,
                               std::vector<int64_t> &dst_shape) {
  int64_t c0 = GetCubeSizeByDataType(data_type);
  if (c0 <= 0) {
    GELOGE(PARAM_INVALID, "Failed to get cube size, the data type is invalid");
    return PARAM_INVALID;
  }
  if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
    GELOGE(PARAM_INVALID, "Failed to check src shape %s", ShapeToString(src_shape).c_str());
    return PARAM_INVALID;
  }
  dst_shape.clear();
  dst_shape.push_back(src_shape.at(kNchwN));
  dst_shape.push_back(Ceil(src_shape.at(kNchwC), c0));
  dst_shape.push_back(src_shape.at(kNchwH));
  dst_shape.push_back(src_shape.at(kNchwW));
  dst_shape.push_back(c0);
  if (!CheckShapeValid(dst_shape, kNc1hwc0DimsNum)) {
    GELOGE(PARAM_INVALID, "Failed to check dst shape %s", ShapeToString(dst_shape).c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status CheckArgsForNchwToNc1hwc0(const TransArgs &args) {
  if (args.src_format != FORMAT_NCHW || args.dst_format != FORMAT_NC1HWC0) {
    std::string error = "Dose not support trans format from " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format)) + ;
    GE_ERRORLOG_AND_ERRORMSG(UNSUPPORTED, error.c_str());
    return UNSUPPORTED;
  }
  std::vector<int64_t> expect_5d_shape;
  auto ret = TransShapeNchwToNc1hwc0(args.src_shape, args.src_data_type, expect_5d_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if (expect_5d_shape != args.dst_shape) {
    GELOGE(PARAM_INVALID,
           "Failed to trans format, the src and dst shape are not compatible. data"
           " type %s, src shape %s, dst shape %s, expect dst shape %s",
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(), ShapeToString(args.src_shape).c_str(),
           ShapeToString(args.dst_shape).c_str(), ShapeToString(expect_5d_shape).c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status GetDstDataAfterTrans(const TransArgs &args, TransResult &result, const int size, const int64_t total_size) {
  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[total_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(OUT_OF_MEMORY,
           "Failed to trans format from %s to %s, can not alloc the memory for"
           " dst buf %ld, shape %s",
           TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str(), total_size, ShapeToString(args.dst_shape).c_str());
    return OUT_OF_MEMORY;
  }

  auto n = args.src_shape.at(kNchwN);
  auto c = args.src_shape.at(kNchwC);
  auto h = args.src_shape.at(kNchwH);
  auto w = args.src_shape.at(kNchwW);

  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  if (c0 <= 0) {
    GELOGE(INTERNAL_ERROR, "The c0 is invalid %ld", c0);
    return PARAM_INVALID;
  }
  int64_t c1 = (c - 1) / c0 + 1;
  int64_t hw = h * w;
  int64_t chw = c * hw;
  int64_t hwc0 = hw * c0;
  int64_t c1hwc0 = c1 * hwc0;
  int64_t wc0 = w * c0;

  for (int64_t n_idx = 0; n_idx < n; n_idx++) {
    int64_t n_head_addr = n_idx * c1hwc0;
    for (int64_t c1_idx = 0; c1_idx < c1; c1_idx++) {
      int64_t c1_head_addr = n_head_addr + c1_idx * hwc0;
      for (int64_t h_idx = 0; h_idx < h; h_idx++) {
        int64_t h_head_addr = c1_head_addr + h_idx * wc0;
        for (int64_t w_idx = 0; w_idx < w; w_idx++) {
          int64_t w_head_addr = h_head_addr + w_idx * c0;
          for (int64_t c0_idx = 0; c0_idx < c0; c0_idx++) {
            int64_t dst_index = c0_idx + w_head_addr;
            int64_t dst_offset = dst_index * size;
            auto protected_size = total_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                      ? total_size - dst_offset
                                      : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
            int64_t cIdx = c0_idx + c1_idx * c0;
            int64_t srcIdx = n_idx * chw + cIdx * hw + h_idx * w + w_idx;
            auto src_offset = srcIdx * size;

            if (cIdx < c) {
              auto ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), args.data + src_offset,
                                  static_cast<size_t>(size));
              if (ret != EOK) {
                GELOGE(INTERNAL_ERROR,
                       "Failed to copy data from NCHW[%ld] offset %ld to "
                       "NC1HWC0[%ld, %ld, %ld, %ld, %ld] offset %ld, err-code %d",
                       srcIdx, src_offset, n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                return INTERNAL_ERROR;
              }
            } else {
              auto ret =
                  memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0, static_cast<size_t>(size));
              if (ret != EOK) {
                GELOGE(INTERNAL_ERROR,
                       "Failed to set to 0 to "
                       "NC1HWC0[%ld, %ld, %ld, %ld, %ld] offset %ld, err-code %d",
                       n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                return INTERNAL_ERROR;
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

Status FormatTransferNchwNc1hwc0::TransFormat(const TransArgs &args, TransResult &result) {
  if (CheckArgsForNchwToNc1hwc0(args) != SUCCESS) {
    return PARAM_INVALID;
  }
  // Guarantee the validity of parameters in check function
  int size = GetSizeByDataType(args.src_data_type);
  auto total_size = GetItemNumByShape(args.dst_shape) * size;
  if (total_size <= 0) {
    int64_t src_size = GetItemNumByShape(args.src_shape);
    if (total_size == 0 && src_size == 0) {
      result.length = static_cast<size_t>(total_size);
      return SUCCESS;
    }

    GELOGE(INTERNAL_ERROR, "Get %ld total size from dst shape %s, src shape %s", total_size,
           ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    return PARAM_INVALID;
  }
  GELOGD(
      "Begin to trans format from NCHW to NC1HWC0, src shape %s, data type "
      "%s, dst shape %s memory size %ld",
      ShapeToString(args.src_shape).c_str(), TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
      ShapeToString(args.dst_shape).c_str(), total_size);
  if (GetDstDataAfterTrans(args, result, size, total_size) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to get data after trans, src shape %s, data type %s, dst shape %s, memory size %ld",
           ShapeToString(args.src_shape).c_str(), TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
           ShapeToString(args.dst_shape).c_str(), total_size);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status FormatTransferNchwNc1hwc0::TransShape(Format src_format, const std::vector<int64_t> &src_shape,
                                             DataType data_type, Format dst_format, std::vector<int64_t> &dst_shape) {
  if (src_format == FORMAT_NCHW) {
    return TransShapeNchwToNc1hwc0(src_shape, data_type, dst_shape);
  } else {
    return UNSUPPORTED;
  }
}

REGISTER_FORMAT_TRANSFER(FormatTransferNchwNc1hwc0, FORMAT_NCHW, FORMAT_NC1HWC0)
}  // namespace formats
}  // namespace ge
