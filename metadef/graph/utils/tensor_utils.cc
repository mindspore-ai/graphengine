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

#include "graph/utils/tensor_utils.h"
#include <cmath>

#include "debug/ge_log.h"
#include "framework/common/debug/ge_log.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/ge_tensor.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
// When nc1hwc0 dim size = 5, calc element count directly.
const uint32_t kNc1hwc0CalcByDimsSize = 5;

// Unknown shape element num
const int64_t kElementCntUnknownShape = -1;

// Unknown shape mem size
const int64_t kMemSizeUnknownShape = -1;

// Nchw and nhwc dim size must be 4
const uint32_t kDimSize4d = 4;

// C1HWNCoC0 dim size must be 6
const uint32_t kDimSizeC1hwncoc0 = 6;

// Cube size is 16
const uint32_t kTheCubeSize = 16;

// Default c0 size equals cube size.
const uint32_t kC0SizeDefault = kTheCubeSize;

// Size equals int8 cube size is 32
const uint32_t kC0SizeInt8 = 32;

// NCHW dim N index
const int32_t kNchwDimIdxN = 0;
// NCHW dim C index
const int32_t kNchwDimIdxC = 1;
// NCHW dim H index
const int32_t kNchwDimIdxH = 2;
// NCHW dim W index
const int32_t kNchwDimIdxW = 3;

const int kDataMemAlignSize = 32;
const int kNum2 = 2;
}  // namespace

///
/// Check if a * b overflow.
/// @param a multiplier
/// @param b Multiplicand
/// @return true: overflow
///         false: not overflow
///
static bool CheckMultiplyOverflowInt64(const int64_t &a, const int64_t &b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return true;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return true;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return true;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return true;
      }
    }
  }
  return false;
}

///
/// Calculate element num by dims directly.
/// @param dims dim info
/// @param element_cnt element count
/// @return GRAPH_SUCCESS:success
///         other:failed
///
static graphStatus CalcElementCntByDims(const std::vector<int64_t> &dims, int64_t &element_cnt) {
  element_cnt = 1;
  for (int64_t dim : dims) {
    if (CheckMultiplyOverflowInt64(element_cnt, dim)) {
      ErrorManager::GetInstance().ATCReportErrMessage(
        "E19013", {"function", "var1", "var2"},
        {"CheckMultiplyOverflowInt64", std::to_string(element_cnt), std::to_string(dim)});
      GELOGE(GRAPH_FAILED, "CalcElementCntByDims failed, when multiplying %ld and %ld.", element_cnt, dim);
      return GRAPH_FAILED;
    }
    element_cnt *= dim;
  }
  return GRAPH_SUCCESS;
}

///
/// Calculate fixed dims element num.
/// @param dims dim info
/// @param fixed_dim_size fixed dim size
/// @param element_cnt element count
/// @return GRAPH_SUCCESS:success
///         other:failed
///
static graphStatus CalcElementCntOfFixedDims(const std::vector<int64_t> &dims, Format format, uint32_t fixed_dim_size,
                                             int64_t &element_cnt) {
  if (dims.size() != fixed_dim_size) {
    GELOGW("Format %d(%s) need dim size=%u but %zu, calc as ND.", format,
           TypeUtils::FormatToSerialString(format).c_str(), fixed_dim_size, dims.size());
  }
  return CalcElementCntByDims(dims, element_cnt);
}

///
/// Get dim c0 size by type
/// @param data_type data type
/// @return c0 size
///
static uint32_t GetDimC0(DataType &data_type) {
  bool is_int8_size = (data_type == DT_INT8) || (data_type == DT_UINT8) || (data_type == DT_DUAL_SUB_UINT8) ||
                      (data_type == DT_DUAL_SUB_INT8) || (data_type == DT_BOOL) || (data_type == DT_QINT8);
  return is_int8_size ? kC0SizeInt8 : kC0SizeDefault;
}

///
/// Calculate nc1hwc0 element num.
/// @param dims dim info
/// @param data_type data type
/// @param element_cnt element count
/// @return GRAPH_SUCCESS:success
///         other:failed
///
static graphStatus CalcElementCntOfNc1hwc0(const std::vector<int64_t> &dims, DataType data_type, int64_t &element_cnt) {
  // When nc1hwc0 dims size = 5, no need split dim c
  if (dims.size() == kNc1hwc0CalcByDimsSize) {
    return CalcElementCntByDims(dims, element_cnt);
  } else if (dims.size() != kDimSize4d) {
    GELOGE(GRAPH_FAILED, "CalcElementCntOfNc1hwc0 failed as dims.size=%zu is not %u or %u.", dims.size(), kDimSize4d,
           kNc1hwc0CalcByDimsSize);
    return GRAPH_FAILED;
  }

  auto c0 = static_cast<int64_t>(GetDimC0(data_type));
  // Nc1hwc0 dims is according to nchw, dim c index is 1.
  auto c1 = static_cast<int64_t>(std::ceil(dims[kNchwDimIdxC] * 1.0 / c0));
  // Store dims is split c to c1 and c0.
  std::vector<int64_t> store_dims = {dims[kNchwDimIdxN], c1, dims[kNchwDimIdxH], dims[kNchwDimIdxW], c0};
  return CalcElementCntByDims(store_dims, element_cnt);
}

///
/// Calculate FractalZ element num.
/// @param dims dim info
/// @param data_type data type
/// @param element_cnt element count
/// @return GRAPH_SUCCESS:success
///         other:failed
///
static graphStatus CalcElementCntOfFractalZ(const std::vector<int64_t> &dims, DataType data_type,
                                            int64_t &element_cnt) {
  static char *parser_priority = std::getenv("PARSER_PRIORITY");
  if (parser_priority != nullptr && string(parser_priority) == "cce") {
    if (dims.size() != kDimSize4d) {
      GELOGE(GRAPH_FAILED, "CalcElementCntOfFractalZ failed as dims.size=%zu is not %u.", dims.size(), kDimSize4d);
      return GRAPH_FAILED;
    }
    auto c0 = static_cast<int64_t>(GetDimC0(data_type));
    // FractalZ dims is according to nchw, dim c index is 1.
    auto c1 = static_cast<int64_t>(std::ceil(dims[kNchwDimIdxC] * 1.0 / c0));

    // Spread NC1HWC0 as a two dimension array, n as column dimension,
    // C1HWC0 as row dimension
    std::vector<int64_t> r_count_vec = {c1, dims[kNchwDimIdxH], dims[kNchwDimIdxW], c0};

    int64_t r_count = 1;
    graphStatus graph_status = CalcElementCntByDims(r_count_vec, r_count);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(graph_status, "Calc [%ld, %ld, %ld, %ld] element count failed.", c1, dims[kNchwDimIdxH],
             dims[kNchwDimIdxW], c0);
      return graph_status;
    }

    // Cube count in n
    auto nc_cnt = static_cast<int64_t>(std::ceil(dims[kNchwDimIdxN] * 1.0 / kTheCubeSize));

    // Cube count in vertical direction(C1HWC0)
    int64_t vc_cnt = r_count / c0;
    // Element count in each cube
    int64_t cube_elem_cnt = c0 * kTheCubeSize;

    if (CheckMultiplyOverflowInt64(nc_cnt, vc_cnt)) {
      GELOGE(GRAPH_FAILED, "The multiplication of %ld and %ld is overflow.", nc_cnt, vc_cnt);
      return GRAPH_FAILED;
    }
    // Read data times needed by cube
    int64_t c_cnt = nc_cnt * vc_cnt;

    if (CheckMultiplyOverflowInt64(c_cnt, cube_elem_cnt)) {
      GELOGE(GRAPH_FAILED, "The multiplication of %ld and %ld is overflow.", c_cnt, cube_elem_cnt);
      return GRAPH_FAILED;
    }
    // Element count after fractal arrangement
    element_cnt = c_cnt * cube_elem_cnt;
    return GRAPH_SUCCESS;
  } else {
    return CalcElementCntByDims(dims, element_cnt);
  }
}

///
/// Calculate tensor element num.
/// @param dims dim info
/// @param format tensor format
/// @param data_type data type
/// @param element_cnt element count
/// @return GRAPH_SUCCESS:success
///         other:failed
///
static graphStatus CalcTensorElementCnt(const std::vector<int64_t> &dims, Format format, DataType data_type,
                                        int64_t &element_cnt) {
  const string format_str = TypeUtils::FormatToSerialString(format);
  // Check dims
  for (size_t i = 0; i < dims.size(); ++i) {
    int64_t dim = dims[i];
    if (dim < 0) {
      GELOGI("It's unknown shape, as dims[%zu]=%ld negative, format=%d(%s).", i, dim, format, format_str.c_str());
      element_cnt = kElementCntUnknownShape;
      return GRAPH_SUCCESS;
    } else if (dim == 0) {
      GELOGI("No need calc element count, as dims[%zu]=%ld, format=%d(%s).", i, dim, format, format_str.c_str());
      element_cnt = 0;
      return GRAPH_SUCCESS;
    }
  }

  graphStatus graph_status;
  switch (format) {
    case FORMAT_ND:
    case FORMAT_MD:
      graph_status = CalcElementCntByDims(dims, element_cnt);
      break;
    case FORMAT_NCHW:
    case FORMAT_HWCN:
    case FORMAT_NHWC:
    case FORMAT_CHWN:
      graph_status = CalcElementCntOfFixedDims(dims, format, kDimSize4d, element_cnt);
      break;
    case FORMAT_C1HWNCoC0:
      graph_status = CalcElementCntOfFixedDims(dims, format, kDimSizeC1hwncoc0, element_cnt);
      break;
    case FORMAT_NC1HWC0:
      graph_status = CalcElementCntOfNc1hwc0(dims, data_type, element_cnt);
      break;
    case FORMAT_FRACTAL_Z:
      graph_status = CalcElementCntOfFractalZ(dims, data_type, element_cnt);
      break;
    case FORMAT_FRACTAL_NZ:
    case FORMAT_FRACTAL_ZZ:
    case FORMAT_NDHWC:
    case FORMAT_NCDHW:
    case FORMAT_DHWCN:
    case FORMAT_DHWNC:
    case FORMAT_FRACTAL_Z_3D:
    case FORMAT_FRACTAL_Z_3D_TRANSPOSE:
    case FORMAT_NDC1HWC0:
    case FORMAT_FRACTAL_Z_C04:
    case FORMAT_FRACTAL_ZN_LSTM:
    case FORMAT_NC1HWC0_C04:
      graph_status = CalcElementCntByDims(dims, element_cnt);
      break;
    default:
      GELOGE(GRAPH_FAILED, "unsupported format, format=%d(%s).", format, format_str.c_str());
      graph_status = GRAPH_FAILED;
      break;
  }

  const string type_str = TypeUtils::DataTypeToSerialString(data_type);
  if (graph_status == GRAPH_SUCCESS) {
    GELOGD(
      "CalcTensorElementCnt end, format=%d(%s),"
      " data_type=%d(%s), element_cnt=%ld.",
      format, format_str.c_str(), data_type, type_str.c_str(), element_cnt);
  } else {
    GELOGE(GRAPH_FAILED, "CalcTensorElementCnt failed, format=%d(%s), data_type=%d(%s).", format, format_str.c_str(),
           data_type, type_str.c_str());
  }
  return graph_status;
}

///
/// Calculate tensor mem size.
/// @param shape tensor shape
/// @param format tensor format
/// @param data_type tensor data type
/// @param mem_size -1 means unknown shape,other means mem size
/// @return GRAPH_SUCCESS:success, other:failed
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::CalcTensorMemSize(const GeShape &shape,
                                                                                          Format format,
                                                                                          DataType data_type,
                                                                                          int64_t &mem_size) {
  const string format_str = TypeUtils::FormatToSerialString(format);
  const string type_str = TypeUtils::DataTypeToSerialString(data_type);
  uint32_t type_size = 0;
  bool result = TypeUtils::GetDataTypeLength(data_type, type_size);
  if (!result) {
    GELOGE(GRAPH_FAILED, "GetDataTypeLength failed, data_type=%d(%s).", data_type, type_str.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dims = shape.GetDims();
  int64_t element_cnt = 0;
  graphStatus status = CalcTensorElementCnt(dims, format, data_type, element_cnt);
  if (status != GRAPH_SUCCESS) {
    GELOGE(status, "CalcTensorElementCnt failed, status=%u format=%d(%s) data_type=%d(%s).", status, format,
           format_str.c_str(), data_type, type_str.c_str());
    return status;
  }
  // Support unknown shape
  if (element_cnt < 0) {
    mem_size = kMemSizeUnknownShape;
    GELOGD(
      "element_cnt is unknown. "
      "format=%d(%s), data_type=%d(%s), mem_size=%ld",
      format, format_str.c_str(), data_type, type_str.c_str(), mem_size);
    return GRAPH_SUCCESS;
  }
  auto type_size_int64 = static_cast<int64_t>(type_size);
  if (CheckMultiplyOverflowInt64(element_cnt, type_size_int64)) {
    GELOGE(GRAPH_FAILED, "CalcTensorMemSize overflow, when multiplying %ld and %ld, format=%d(%s), data_type=%d(%s).",
           element_cnt, type_size_int64, format, format_str.c_str(), data_type, type_str.c_str());
    return GRAPH_FAILED;
  }
  mem_size = element_cnt * type_size_int64;

  GELOGD(
    "CalcTensorMemSize end, "
    "format=%d(%s), data_type=%d(%s), mem_size=%ld",
    format, format_str.c_str(), data_type, type_str.c_str(), mem_size);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetTensorMemorySizeInBytes(const GeTensorDesc &desc_temp, int64_t &size_temp) {
  graphStatus graph_status = GetTensorSizeInBytes(desc_temp, size_temp);
  if (graph_status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  // 64-byte alignment, if size is 0, align to 32 bytes
  if (size_temp > (INT64_MAX - kNum2 * kDataMemAlignSize)) {
    GELOGW("The updated mem size %ld is bigger than INT64_MAX", size_temp);
  } else {
    size_temp = ((size_temp + kNum2 * kDataMemAlignSize - 1) / kDataMemAlignSize) * kDataMemAlignSize;
  }
  return GRAPH_SUCCESS;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetTensorSizeInBytes(const GeTensorDesc &desc_temp, int64_t &size_temp) {
  GeShape output_shape = desc_temp.GetShape();
  Format format = desc_temp.GetFormat();
  DataType data_type = desc_temp.GetDataType();
  int64_t output_mem_size = 0;
  graphStatus graph_status = CalcTensorMemSize(output_shape, format, data_type, output_mem_size);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "CalcTensorMemSize failed!");
    return GRAPH_FAILED;
  }

  if (output_mem_size < 0) {
    GELOGE(GRAPH_FAILED, "After calc concat tensor memory size, output_mem_size = %ld, out of data range [0, %ld]",
           output_mem_size, INT64_MAX);
    return GRAPH_FAILED;
  }

  size_temp = output_mem_size;
  return GRAPH_SUCCESS;
}
}  // namespace ge
