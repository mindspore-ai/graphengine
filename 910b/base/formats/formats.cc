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

#include "formats/formats.h"

#include <securec.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
Status TransDataFormat(const TransArgs &args, TransResult &result) {
  const auto transfer = BuildFormatTransfer(args);
  if (transfer == nullptr) {
    const std::string error = "Failed to trans data from format " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }

  const auto src_shape_size = GetItemNumByShape(args.src_shape);
  if ((args.data == nullptr) && (src_shape_size != 0)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Shape]Failed, input data is null "
           "or shape size not euqal to 0, src_shape %s",
           ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to check shape, input data is null "
                      "or shape size not equal to 0, src_shape %s",
                      ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  return transfer->TransFormat(args, result);
}

Status TransTensorShape(const Format src_format, const std::vector<int64_t> &src_shape, const DataType data_type,
                        const Format dst_format, std::vector<int64_t> &dst_shape) {
  formats::TransArgs args;
  args.src_format = src_format;
  args.dst_format = dst_format;
  args.src_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(src_format)));
  args.dst_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(dst_format)));
  const auto transfer = BuildFormatTransfer(args);
  if (transfer == nullptr) {
    const std::string error = "Failed to trans data from format " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_primary_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_primary_format));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }

  return transfer->TransShape(src_format, src_shape, data_type, dst_format, dst_shape);
}

bool IsTransFormatSupport(const TransArgs &args) {
  return FormatTransferExists(args);
}
}  // namespace formats
}  // namespace ge
