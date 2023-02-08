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

#include "formats/format_transfers/format_transfer_nchw_to_fz_c04.h"
#include "formats/format_transfers/format_transfer_transpose.h"

#include <securec.h>
#include <memory>
#include <cstdlib>

#include "formats/utils/formats_definitions.h"
#include "formats/utils/formats_trans_utils.h"
#include "common/math/math_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"

/** 【Explain about transfer from nchw to FZ_CO4】
 *  First Step: Padding in N and C axis. Here C must be less or equal than 4
 *      After Padding, it will be like (n = ceil(n,16)*16, 4, h, w)
 *  Second Step: transpose. It will be like (n = ceil(n,16)*16, h, w, 4)
 *  Third Step: View the 4D as 2D , first dim is N, second dim is h*w*c.
 *      Padding to (N, ceil(Z/16)*16)
 *  Last Step: View the (N, ceil(Z/16)*16) as 4D (N/16, 16, C/16, 16) and transpose to (C/16, N/16, 16, 16)
 */
namespace ge {
namespace formats {
namespace {
constexpr int64_t kMaxDimsNumC = 4;

Status CheckDataTypeSupport(const DataType data_type) {
  return (GetSizeByDataType(data_type) > 0) ? SUCCESS : UNSUPPORTED;
}

Status TransShapeOfNchw2Fz(const int64_t n, const int64_t c, const int64_t h, const int64_t w, const DataType data_type,
                           std::vector<int64_t> &dst_shape) {
  const auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  const auto chw = c * h * w;

  const auto first_dim = Ceil(chw, c0);
  const auto no = Ceil(n, static_cast<int64_t>(c0));

  dst_shape.clear();
  dst_shape.push_back(first_dim);
  dst_shape.push_back(no);
  dst_shape.push_back(c0);
  dst_shape.push_back(c0);

  if (!IsShapeValid(dst_shape)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check invalid", ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return SUCCESS;
}

Status TransShapeNchwToFzC04(const std::vector<int64_t> &src_shape, const DataType data_type,
                             std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  const auto n = src_shape.at(kNchwN);
  const auto c = src_shape.at(kNchwC);
  const auto h = src_shape.at(kNchwH);
  const auto w = src_shape.at(kNchwW);
  return TransShapeOfNchw2Fz(n, c, h, w, data_type, dst_shape);
}

Status TransFormatFromNchwToFzC04(const TransArgs &args, TransResult &result) {
  const int64_t n = args.src_shape.at(kNchwN);
  const int64_t c = args.src_shape.at(kNchwC);
  const int64_t h = args.src_shape.at(kNchwH);
  const int64_t w = args.src_shape.at(kNchwW);

  const int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  const int32_t size = GetSizeByDataType(args.src_data_type);

  const auto data = args.data;
  TransResult trans_result_1;
  const std::vector<int64_t> perm_arg_1 = {0, 2, 3, 1};
  const std::vector<int64_t> expect_shape = {n, h, w, c};
  auto ret = ge::formats::Transpose(data, args.src_shape, args.src_data_type, perm_arg_1, trans_result_1);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Trans][Formats]Failed from NCHW to HWCN, src_shape %s, src_data_type %s",
           ShapeToString(args.src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_CALL_ERROR("E19999", "Failede to trans formats from NCHW to HWCN, src_shape %s, "
                      "src_data_type %s",
                      ShapeToString(args.src_shape).c_str(),
                      TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ret;
  }

  TransArgs args_tmp = args;
  args_tmp.src_shape = expect_shape;
  args_tmp.data = trans_result_1.data.get();
  // check size it should be same with original
  const int64_t sum_size = n * c * h * w * size; // before has do check about mul
  const size_t expect_size = static_cast<size_t>(sum_size);
  if (trans_result_1.length != expect_size) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Shape]size %zu is not match expect size %zu "
           "after transpose",
           trans_result_1.length, expect_size);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  // prepare for padding in chw
  const int64_t tmp = h * w * c;
  const int64_t n_o = Ceil(n, static_cast<int64_t>(c0));
  const int64_t c_o = c0;
  const int64_t h_o = Ceil(tmp, c0);
  const int64_t w_o = c0;
  const std::vector<int64_t> shape_o = {n_o, c_o, h_o, w_o};

  // data overflow check totally
  GE_IF_BOOL_EXEC(CheckInt64MulOverflow(h_o, w_o) != SUCCESS,
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", h_o, w_o);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%" PRId64 "], "
                                    "B[%" PRId64 "]", h_o, w_o);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  GE_IF_BOOL_EXEC(CheckInt64MulOverflow(n_o, c_o) != SUCCESS,
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", n_o, c_o);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%" PRId64 "], "
                                    "B[%" PRId64 "]", n_o, c_o);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  const auto t1 = h_o * w_o;
  const auto t2 = n_o * c_o;
  GE_IF_BOOL_EXEC(CheckInt64MulOverflow(t1, t2) != SUCCESS,
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", t1, t2);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, "
                                    "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", t1, t2);
                  return ACL_ERROR_GE_INTERNAL_ERROR);

  const int64_t total_ele_cnt = n_o * c_o * h_o * w_o;
  GE_IF_BOOL_EXEC(CheckInt64MulOverflow(total_ele_cnt, static_cast<int64_t>(size)) != SUCCESS,
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                  "int64 mul overflow.A[%" PRId64 "], B[%d]", total_ele_cnt, size);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%" PRId64 "], "
                  "B[%d]", total_ele_cnt, size);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  const int64_t dst_size = total_ele_cnt * size;
  if (dst_size == 0) {
    result.length = 0U;
    return SUCCESS;
  }

  const std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Failed to alloc the memory for dst buf %" PRId64 " "
           "when trans format from %s to %s",
           dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    REPORT_CALL_ERROR("E19999",  "Failed to alloc the memory for dst buf %" PRId64 " "
                      "when trans format from %s to %s",
                      dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  const auto retMem = memset_s(dst.get(), static_cast<size_t>(dst_size), 0, static_cast<size_t>(dst_size));
  if (retMem != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Set][Memory]Failed, dst buf %" PRId64 ", error_code %d",
           dst_size, retMem);
    REPORT_CALL_ERROR("E19999", "Set memory failed, dst buf %" PRId64 ", error_code %d", dst_size, retMem);
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  // copy data
  const auto block = c * h * w * size;
  const auto stride = h_o * w_o * size;
  const auto p_s = trans_result_1.data.get();
  const auto p_d = dst.get();
  auto protected_size = dst_size;
  for (auto k = 0; k < n; k++) {
    const auto cpy_ret =
        memcpy_s(PtrAdd(p_d, static_cast<size_t>(dst_size), static_cast<size_t>(k) * static_cast<size_t>(stride)),
                 static_cast<size_t>(protected_size),
                 p_s + (k * block), static_cast<size_t>(block));
    if (cpy_ret != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Set][Memcpy]Failed, block %zu, stride %zu, "
             "protect_size %" PRId64 ", error_code %d", block, stride, protected_size, cpy_ret);
      REPORT_CALL_ERROR("E19999", "[Set][Memcpy]Failed, block %" PRIu64 ", stride %" PRIu64 ", "
                        "protect_size %" PRId64 ", error_code %d", block, stride, protected_size, cpy_ret);
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    protected_size = protected_size - block;
  }

  // transpose : 2,0,1,3
  const std::vector<int64_t> perm_arg_2 = {2, 0, 1, 3};
  ret = ge::formats::Transpose(dst.get(), shape_o, args.src_data_type, perm_arg_2, result);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Trans][Formats]Failed from NCHW to HWCN, error_code %u", ret);
    REPORT_CALL_ERROR("E19999", "Failed to trans formats from NCHW to HWCN, error_code %u", ret);
    return ret;
  }

  return SUCCESS;
}

Status PaddingNC(const TransArgs &args, TransArgs &args_tmp, std::shared_ptr<uint8_t> &dst) {
  args_tmp = args;
  auto src_shape = args_tmp.src_shape;
  if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  const int64_t c0 = GetCubeSizeByDataType(args.src_data_type);

  const auto n = src_shape.at(kNchwN);
  const auto c = src_shape.at(kNchwC);
  const auto h = src_shape.at(kNchwH);
  const auto w = src_shape.at(kNchwW);

  if (c > kMaxDimsNumC) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Invalid dim c num[%lu]. "
           "It should be in (0,4]", c);
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  const auto n_o = Ceil(n, c0) * c0;
  const auto c_o = kMaxDimsNumC;
  const auto h_o = h;
  const auto w_o = w;
  args_tmp.src_shape.at(kNchwN) = n_o;
  args_tmp.src_shape.at(kNchwC) = c_o;
  args_tmp.src_shape.at(kNchwH) = h_o;
  args_tmp.src_shape.at(kNchwW) = w_o;

  // data overflow check
  GE_IF_BOOL_EXEC(CheckInt64MulOverflow(h_o, w_o) != SUCCESS,
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", h_o, w_o);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%" PRId64 "], "
                                    "B[%" PRId64 "]", h_o, w_o);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  GE_IF_BOOL_EXEC(CheckInt64MulOverflow(n_o, c_o) != SUCCESS,
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", n_o, c_o);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%" PRId64 "], "
                                    "B[%" PRId64 "]", n_o, c_o);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  const auto t1 = h_o * w_o;
  const auto t2 = n_o * c_o;
  GE_IF_BOOL_EXEC(CheckInt64MulOverflow(t1, t2) != SUCCESS,
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", t1, t2);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%" PRId64 "], "
                                    "B[%" PRId64 "]", t1, t2);
                  return ACL_ERROR_GE_INTERNAL_ERROR);

  const int64_t total_ele_cnt = n_o * c_o * h_o * w_o;
  const int32_t size = GetSizeByDataType(args.src_data_type);
  GE_IF_BOOL_EXEC(CheckInt64MulOverflow(total_ele_cnt, static_cast<int64_t>(size)) != SUCCESS,
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%" PRId64 "], B[%d]", total_ele_cnt, size);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%" PRId64 "], "
                                    "B[%d]", total_ele_cnt, size);
                  return ACL_ERROR_GE_INTERNAL_ERROR);

  const int64_t dst_size = total_ele_cnt * size;
  if (dst_size == 0) {
    return SUCCESS;
  }

  dst.reset(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Failed to alloc the memory for dst buf %" PRId64 " when "
           "trans format from %s to %s",
           dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to alloc the memory for dst buf %" PRId64 " when "
                      "trans format from %s to %s",
                      dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  auto ret = memset_s(dst.get(), static_cast<size_t>(dst_size), 0, static_cast<size_t>(dst_size));
  if (ret != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Set][Memory]Failed, dst buf %" PRId64 ", error_code %d",
           dst_size, ret);
    REPORT_CALL_ERROR("E19999", "Set memory failed, dst buf %" PRId64 ", error_code %d", dst_size, ret);
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }

  const auto p_s = args.data;
  const auto p_d = dst.get();
  const auto block = h * w * size;
  auto protected_size = dst_size;

  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < c; j++) {
      const int64_t dst_offset = ((i * c_o * h_o * w_o) + (j * h_o * w_o)) * size;
      ret = memcpy_s(PtrAdd(p_d, static_cast<size_t>(dst_size), static_cast<size_t>(dst_offset)),
                     static_cast<size_t>(protected_size),
                     p_s + (((i * c * h * w) + (j * h * w)) * size), static_cast<size_t>(block));
      if (ret != EOK) {
        GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Set][Memcpy]Failed, block %zu, "
               "protect_size %" PRId64 ", error_code %d", block, protected_size, ret);
        REPORT_CALL_ERROR("E19999", "[Set][Memcpy]Failed, block %" PRIu64 ", protect_size %" PRId64 ", "
                          "error_code %d", block, protected_size, ret);
        return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      }
      protected_size = protected_size - block;
    }
  }
  args_tmp.data = dst.get();

  return SUCCESS;
}
}  // namespace

Status FormatTransferNchwToFZC04::TransFormat(const TransArgs &args, TransResult &result) {
  GELOGD("Begin to trans format from %s to %s, src shape %s, data type %s, dst shape %s",
         TypeUtils::FormatToSerialString(args.src_format).c_str(),
         TypeUtils::FormatToSerialString(args.dst_format).c_str(), ShapeToString(args.src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(), ShapeToString(args.dst_shape).c_str());
  TransArgs args_tmp = args;
  std::shared_ptr<uint8_t> dst = nullptr;
  auto ret = PaddingNC(args, args_tmp, dst);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Padding][NCAxis]Failed, error_code %u", ret);
    REPORT_CALL_ERROR("E19999", "Padding in NC axis failed, error_code %u", ret);
    return ret;
  }

  std::vector<int64_t> expect_shape;
  ret = TransShape(args_tmp.src_format, args_tmp.src_shape, args_tmp.src_data_type,
                   args_tmp.dst_format, expect_shape);
  if (ret != SUCCESS) {
    return ret;
  }

  if (!IsTransShapeDstCorrect(args_tmp, expect_shape)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  if ((args_tmp.src_format == FORMAT_NCHW) && (args_tmp.dst_format == FORMAT_FRACTAL_Z_C04)) {
    return TransFormatFromNchwToFzC04(args_tmp, result);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

Status FormatTransferNchwToFZC04::TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                                             const DataType data_type, const Format dst_format,
                                             std::vector<int64_t> &dst_shape) {
  const Format src_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(src_format)));
  const Format dst_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(dst_format)));
  if (CheckDataTypeSupport(data_type) != SUCCESS) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if ((src_primary_format == FORMAT_NCHW) && (dst_primary_format == FORMAT_FRACTAL_Z_C04)) {
    return TransShapeNchwToFzC04(src_shape, data_type, dst_shape);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferNchwToFZC04, FORMAT_NCHW, FORMAT_FRACTAL_Z_C04)
}  // namespace formats
}  // namespace ge
