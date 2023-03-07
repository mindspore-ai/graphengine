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

#include "formats/format_transfers/format_transfer_fz_c04.h"
#include "formats/format_transfers/format_transfer_transpose.h"

#include <securec.h>
#include <memory>
#include <cstdlib>

#include "formats/utils/formats_definitions.h"
#include "formats/utils/formats_trans_utils.h"
#include "common/math/math_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/math_util.h"
#include "formats/formats.h"
#include "common/checker.h"

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
constexpr int64_t kC04 = 4;
constexpr int64_t kC0Size = 16;
constexpr int64_t kCubeK = 16; // 当前硬件为固定值

Status CheckDataTypeSupport(const DataType data_type) {
  return (GetSizeByDataType(data_type) > 0) ? SUCCESS : UNSUPPORTED;
}

Status GetTotalDataSize(const std::vector<int64_t> &shape_4d, const int64_t dtype_size, int64_t &total_size) {
  // data overflow check totally
  // 输出的4d shape不一定是nchw格式，这里只是为了按顺序取每个轴的值
  const auto n_o = shape_4d.at(kNchwN);
  const auto c_o = shape_4d.at(kNchwC);
  const auto h_o = shape_4d.at(kNchwH);
  const auto w_o = shape_4d.at(kNchwW);
  int64_t t1, t2 = 0;
  GE_ASSERT_TRUE(!MulOverflow(h_o, w_o, t1),
                 "[Check][Shape]Failed, "
                 "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", h_o, w_o);
  GE_ASSERT_TRUE(!MulOverflow(n_o, c_o, t2),
                 "[Check][Shape]Failed, "
                 "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", n_o, c_o);
  int64_t total_ele_cnt = 0;
  GE_ASSERT_TRUE(!MulOverflow(t1, t2, total_ele_cnt),
                 "[Check][Shape]Failed, "
                 "int64 mul overflow.A[%" PRId64 "], B[%" PRId64 "]", t1, t2);
  GE_ASSERT_TRUE(!MulOverflow(total_ele_cnt, static_cast<int64_t>(dtype_size), total_size),
                 "[Check][Shape]Failed, "
                 "int64 mul overflow.A[%" PRId64 "], B[%d]", total_ele_cnt, dtype_size);
  return SUCCESS;
}

Status TransShapeNchwToFzC04(const std::vector<int64_t> &src_shape, const DataType data_type,
                             std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  const auto n = src_shape.at(kNchwN);
  const auto h = src_shape.at(kNchwH);
  const auto w = src_shape.at(kNchwW);

  const auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  // fzc04, c should be 4
  const auto chw = kC04 * h * w;

  const auto first_dim = Ceil(chw, c0);
  const auto no = Ceil(n, static_cast<int64_t>(c0));

  dst_shape.clear();
  dst_shape.push_back(first_dim);
  dst_shape.push_back(no);
  dst_shape.push_back(kCubeK);
  dst_shape.push_back(c0);

  if (!IsShapeValid(dst_shape)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check invalid", ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return SUCCESS;
}

Status TransFormatNchwToFzC04(const TransArgs &args, TransResult &result) {
  const int64_t n = args.src_shape.at(kNchwN);
  const int64_t c = args.src_shape.at(kNchwC);
  const int64_t h = args.src_shape.at(kNchwH);
  const int64_t w = args.src_shape.at(kNchwW);

  const int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  const int32_t dtype_size = GetSizeByDataType(args.src_data_type);

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
  const int64_t sum_size = n * c * h * w * dtype_size; // before has do check about mul
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
  int64_t dst_size = 0;
  GE_ASSERT_SUCCESS(GetTotalDataSize(shape_o, dtype_size, dst_size));
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
  const auto block = c * h * w * dtype_size;
  const auto stride = h_o * w_o * dtype_size;
  const auto p_start = trans_result_1.data.get();
  const auto p_dst = dst.get();
  auto protected_size = dst_size;
  for (auto k = 0; k < n; k++) {
    const auto cpy_ret =
        memcpy_s(PtrAdd(p_dst, static_cast<size_t>(dst_size), static_cast<size_t>(k) * static_cast<size_t>(stride)),
                 static_cast<size_t>(protected_size), p_start + (k * block), static_cast<size_t>(block));
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

  // malloc dst data and reset to 0
  int64_t dst_size = 0;
  const int32_t dtype_size = GetSizeByDataType(args.src_data_type);
  GE_ASSERT_SUCCESS(GetTotalDataSize(args_tmp.src_shape, static_cast<int64_t>(dtype_size), dst_size));
  if (dst_size == 0) {
    return SUCCESS;
  }
  dst.reset(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  GE_ASSERT_NOTNULL(dst, "Failed to alloc the memory for dst buf %" PRId64 " when "
           "trans format from %s to %s",
           dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
  auto ret = memset_s(dst.get(), static_cast<size_t>(dst_size), 0, static_cast<size_t>(dst_size));
  GE_ASSERT_EOK(ret, "[Set][Memory]Failed, dst buf %" PRId64 ", error_code %d", dst_size, ret);

  // copy data from nchw to padded n'4hw
  const auto p_start = args.data;
  const auto p_dst = dst.get();
  const auto block = h * w * dtype_size;
  auto protected_size = dst_size;

  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < c; j++) {
      const int64_t dst_offset = ((i * c_o * h_o * w_o) + (j * h_o * w_o)) * dtype_size;
      ret = memcpy_s(PtrAdd(p_dst, static_cast<size_t>(dst_size), static_cast<size_t>(dst_offset)),
                     static_cast<size_t>(protected_size),
                     p_start + (((i * c * h * w) + (j * h * w)) * dtype_size), static_cast<size_t>(block));
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

Status TransFormatFzC04ToHwcn(const TransArgs &args, TransResult &result) {
  const Format format_4d = args.dst_primary_format;
  if (format_4d != FORMAT_HWCN) {
    return GRAPH_FAILED;
  }
  const std::vector<int64_t> shape_4d = args.dst_shape;
  int64_t h_dim = shape_4d[kHwcnH];
  int64_t w_dim = shape_4d[kHwcnW];
  int64_t c_dim = shape_4d[kHwcnC];
  int64_t n_dim = shape_4d[kHwcnN];

  int64_t cin_ori = c_dim;
  int64_t cout_ori = n_dim;
  if (cin_ori == 0 || cout_ori == 0) {
    return GRAPH_FAILED;
  }

  const DataType data_type = args.src_data_type;
  const int64_t cube_k = GetCubeSizeByDataType(args.src_data_type);
  // groups as divisor can not equal to 0
  const int64_t groups = static_cast<int64_t>(args.src_sub_format);
  if (groups == 0) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "groups can not be 0.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  int64_t e_mult = std::min(
      Lcm(Lcm(cin_ori, cube_k) / (cin_ori), Lcm(cout_ori, static_cast<int64_t>(kCubeSize)) / (cout_ori)), groups);
  int64_t cin_opt = Ceil(e_mult * cin_ori, cube_k) * cube_k;
  int64_t c1 = cin_opt / cube_k;
  int64_t n1 = Ceil(e_mult * cout_ori, static_cast<int64_t>(kCubeSize));

  int64_t dtype_size = GetSizeByDataType(data_type);
  int64_t dst_size = 0;
  GE_ASSERT_SUCCESS(GetTotalDataSize(args.dst_shape, dtype_size, dst_size));
  int64_t src_size = 0;
  GE_ASSERT_SUCCESS(GetTotalDataSize(args.src_shape, dtype_size, src_size));
  // The input is empty tensor, we should return success directly.
  if (dst_size == 0) {
    result.length = static_cast<size_t>(dst_size);
    return GRAPH_FAILED;
  }
  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  GE_ASSERT_NOTNULL(dst, "Failed to allcoate memory for dst buf [%lld] when trans format from [%s] to [%s]", dst_size,
                    TypeUtils::FormatToSerialString(args.src_format).c_str(),
                    TypeUtils::FormatToSerialString(args.dst_format).c_str());
  GE_ASSERT_EOK(memset_s(dst.get(), static_cast<size_t>(dst_size), 0, static_cast<size_t>(dst_size)));

  int64_t protected_size = dst_size;
  for (int64_t c = 0; c < c_dim; c++) {
    for (int64_t h = 0; h < h_dim; h++) {
      for (int64_t w = 0; w < w_dim; w++) {
        for (int64_t n = 0; n < n_dim; n++) {
          int64_t common_factor = h * w_dim * c1 * kC04 + w * kC04 + c % kC04;
          int64_t idx_fz = (common_factor / cube_k * (cube_k * kNiSize * n1)) + (n / kNiSize * (cube_k * kNiSize)) +
                           (n % kNiSize * cube_k) + (common_factor % cube_k);
          int64_t idx_4d = (h * w_dim * c_dim * n_dim) + (w * c_dim * n_dim) + (c * n_dim) + n;
          auto ret = memcpy_s(PtrAdd(dst.get(), dst_size, idx_4d * dtype_size), protected_size,
                              PtrAdd(args.data, src_size, idx_fz * dtype_size), dtype_size);
          GE_ASSERT_EOK(ret, "[Set][Memcpy]Failed, stride %zu, protect_size %" PRId64 ", error_code %d", dtype_size,
                        protected_size);
          protected_size -= dtype_size;
        }
      }
    }
  }

  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return SUCCESS;
}
}  // namespace
Status FormatTransfer4DToFZC04::BuildTransArgsNchwToFzC04(const ge::formats::TransArgs &src_dst_args,
                                                          TransResult &nchw_result_holder,
                                                          ge::formats::TransArgs &nchw_dst_args) {
  if (src_dst_args.src_primary_format == FORMAT_NCHW) {
    return SUCCESS;
  }
  ge::formats::TransArgs src_to_nchw_args = src_dst_args;
  src_to_nchw_args.dst_format = static_cast<Format>(GetFormatFromSub(FORMAT_NCHW,
                                                                     GetSubFormat(src_dst_args.dst_sub_format)));
  src_to_nchw_args.dst_primary_format = FORMAT_NCHW;
  if (!ge::formats::IsTransFormatSupport(src_to_nchw_args)) {
    GELOGE(ACL_ERROR_GE_FORMAT_INVALID, "Not support trans format from %s to %s.",
           TypeUtils::FormatToSerialString(src_dst_args.src_format).c_str(),
           TypeUtils::FormatToSerialString(src_to_nchw_args.dst_format).c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }
  // get nchw shape from src
  std::vector<int64_t> nchw_shape;
  GE_ASSERT_SUCCESS(TransShapeFromSrcToNchw(src_to_nchw_args.src_format, src_to_nchw_args.src_shape,
                                            src_to_nchw_args.src_data_type, nchw_shape));
  src_to_nchw_args.dst_shape = nchw_shape;
  GE_ASSERT_SUCCESS(ge::formats::TransDataFormat(src_to_nchw_args, nchw_result_holder));

  std::vector<int64_t> dst_shape;
  nchw_dst_args = {nchw_result_holder.data.get(), FORMAT_NCHW, src_dst_args.dst_format,
                   FORMAT_NCHW,
                   src_dst_args.dst_primary_format,
                   FORMAT_RESERVED,
                   src_dst_args.dst_sub_format, kC0Size, kC0Size,
                   nchw_shape,
                   dst_shape,
                   src_dst_args.src_data_type};
  return SUCCESS;
}

Status FormatTransfer4DToFZC04::TransShapeFromSrcToNchw(const Format src_format, const std::vector<int64_t> &src_shape,
                                                        const DataType data_type, std::vector<int64_t> &nchw_shape) {
  const Format src_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(src_format)));
  if (src_primary_format == FORMAT_NCHW) {
    return SUCCESS;
  }
  ge::formats::TransArgs args{nullptr, src_format, FORMAT_NCHW, src_primary_format, FORMAT_NCHW,
                              FORMAT_RESERVED, FORMAT_RESERVED, kC0Size, kC0Size,
                              src_shape,  nchw_shape,  data_type};
  if (!ge::formats::IsTransFormatSupport(args)) {
    GELOGE(ACL_ERROR_GE_FORMAT_INVALID, "Not support trans shape from %s to %s",
           ge::TypeUtils::FormatToSerialString(src_format).c_str(),
           ge::TypeUtils::FormatToSerialString(FORMAT_NCHW).c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }
  GE_ASSERT_SUCCESS(ge::formats::TransTensorShape(src_format, src_shape, data_type, FORMAT_NCHW, nchw_shape));
  return SUCCESS;
}

Status FormatTransfer4DToFZC04::TransFormat(const TransArgs &args, TransResult &result) {
  GELOGD("Begin to trans format from %s to %s, src shape %s, data type %s, dst shape %s",
         TypeUtils::FormatToSerialString(args.src_format).c_str(),
         TypeUtils::FormatToSerialString(args.dst_format).c_str(), ShapeToString(args.src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(), ShapeToString(args.dst_shape).c_str());
  
  TransArgs nchw_to_dst_args = args;
  // try build nchw_to_dst_args
  TransResult src_nchw_result;
  GE_CHK_STATUS_RET(BuildTransArgsNchwToFzC04(args, src_nchw_result, nchw_to_dst_args));

  TransArgs args_tmp = nchw_to_dst_args;
  std::shared_ptr<uint8_t> dst = nullptr;
  auto ret = PaddingNC(nchw_to_dst_args, args_tmp, dst);
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

  if ((args_tmp.src_primary_format == FORMAT_NCHW) && (args_tmp.dst_primary_format == FORMAT_FRACTAL_Z_C04)) {
    return TransFormatNchwToFzC04(args_tmp, result);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

Status FormatTransfer4DToFZC04::TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                                           const DataType data_type, const Format dst_format,
                                           std::vector<int64_t> &dst_shape) {
  const Format dst_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(dst_format)));
  if (CheckDataTypeSupport(data_type) != SUCCESS) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  // try transhape from src_format to NCHW
  std::vector<int64_t> nchw_shape = src_shape;
  GE_ASSERT_SUCCESS(TransShapeFromSrcToNchw(src_format, src_shape, data_type, nchw_shape));

  if (dst_primary_format == FORMAT_FRACTAL_Z_C04) {
    return TransShapeNchwToFzC04(nchw_shape, data_type, dst_shape);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

/*
 * FzC04 axises: (Ceil(kC0*h*w*C1/16), N1, N0, 16), which N1 = 16, C0 = 4, N1 = Ceil(N/N0), C1 = Ceil(C/C0)
 */
Status FormatTransferFZC04To4D::TransFormat(const TransArgs &args, TransResult &result) {
  GELOGD("Begin to trans format from %s to %s, src shape %s, data type %s, dst shape %s",
         TypeUtils::FormatToSerialString(args.src_format).c_str(),
         TypeUtils::FormatToSerialString(args.dst_format).c_str(), ShapeToString(args.src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(), ShapeToString(args.dst_shape).c_str());
  if (args.src_primary_format != FORMAT_FRACTAL_Z_C04 || args.dst_primary_format != FORMAT_HWCN) {
    GELOGE(ACL_ERROR_GE_FORMAT_INVALID, "Src format is %s, dst format is %s, Not support.",
           TypeUtils::FormatToSerialString(args.src_primary_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_primary_format).c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }
  std::vector<int64_t> expect_fzc04_shape;
  FormatTransfer4DToFZC04 hwcn_fzc04_transfer;
  GE_ASSERT_SUCCESS(hwcn_fzc04_transfer.TransShape(args.dst_format, args.dst_shape, args.src_data_type, args.src_format,
                                                   expect_fzc04_shape));
  if (expect_fzc04_shape != args.src_shape) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "Src format %s, dts format %s. Shape not equivalent.",
           TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return TransFormatFzC04ToHwcn(args, result);
}

Status FormatTransferFZC04To4D::TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                                           const DataType data_type, const Format dst_format,
                                           std::vector<int64_t> &dst_shape) {
  (void)src_format;
  (void)src_shape;
  (void)data_type;
  (void)dst_format;
  (void)dst_shape;
  GELOGD("The shape derivation from FRACTAL_Z_C04 to 4D is not unique. Trans shape in this direction is not supported");
  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransfer4DToFZC04, FORMAT_NCHW, FORMAT_FRACTAL_Z_C04)
REGISTER_FORMAT_TRANSFER(FormatTransfer4DToFZC04, FORMAT_HWCN, FORMAT_FRACTAL_Z_C04)
REGISTER_FORMAT_TRANSFER(FormatTransferFZC04To4D, FORMAT_FRACTAL_Z_C04, FORMAT_HWCN)
}  // namespace formats
}  // namespace ge
