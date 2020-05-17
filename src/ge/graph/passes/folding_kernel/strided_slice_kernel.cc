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

#include "graph/passes/folding_kernel/strided_slice_kernel.h"

#include <memory>

#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/math/math_util.h"
#include "common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

using domi::STRIDE_SLICE_ATTR_BEGIN_MASK;
using domi::STRIDE_SLICE_ATTR_ELLIPSIS_MASK;
using domi::STRIDE_SLICE_ATTR_END_MASK;
using domi::STRIDE_SLICE_ATTR_NEW_AXIS_MASK;
using domi::STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK;
using domi::STRIDEDSLICE;

namespace ge {
namespace {
const int32_t kNumOne = 1;
const size_t kStridedSliceInputSize = 4;
const size_t kStridedSliceInputIndex0 = 0;
const size_t kStridedSliceInputIndex1 = 1;
const size_t kStridedSliceInputIndex2 = 2;
const size_t kStridedSliceInputIndex3 = 3;
const int32_t kDefaultSrideSize = 1;
}  // namespace
Status StridedSliceKernel::CheckAndGetAttr(const OpDescPtr &attr, const std::vector<ConstGeTensorPtr> &input,
                                           Attr &args) {
  int64_t begin_mask = 0;
  int64_t end_mask = 0;
  int64_t ellipsis_mask = 0;
  int64_t new_axis_mask = 0;
  int64_t shrink_axis_mask = 0;

  if (attr == nullptr) {
    GELOGE(PARAM_INVALID, "input opdescptr is nullptr.");
    return PARAM_INVALID;
  }
  if (input.size() != kStridedSliceInputSize) {
    GELOGE(PARAM_INVALID, "The number of input for strided slice must be %zu.", kStridedSliceInputSize);
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetInt(attr, STRIDE_SLICE_ATTR_BEGIN_MASK, begin_mask)) {
    GELOGE(PARAM_INVALID, "get begin_mask attr failed.");
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetInt(attr, STRIDE_SLICE_ATTR_END_MASK, end_mask)) {
    GELOGE(PARAM_INVALID, "get end_mask attr failed.");
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetInt(attr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, ellipsis_mask)) {
    GELOGE(PARAM_INVALID, "get ellipsis_mask attr failed.");
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetInt(attr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, new_axis_mask)) {
    GELOGE(PARAM_INVALID, "get new_axis_mask attr failed.");
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetInt(attr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, shrink_axis_mask)) {
    GELOGE(PARAM_INVALID, "get shrink_axis_mask attr failed.");
    return PARAM_INVALID;
  }
  if ((ellipsis_mask != 0) || (new_axis_mask != 0)) {
    GELOGW("ellipsis_mask or new_axis_mask must be 0 with optimizer.");
    return NOT_CHANGED;
  }
  const auto &input_desc = attr->MutableInputDesc(kStridedSliceInputIndex0);
  GE_CHECK_NOTNULL(input_desc);
  DataType data_type = input_desc->GetDataType();
  if ((data_type != DT_FLOAT) && (data_type != DT_INT32)) {
    GELOGW(
      "Data type of StridedSlice OP must be float or int32."
      "Constant folding will not be carried out in this condition"
      "which might affect the time performance but not the accuracy");
  }
  args.begin_mask = begin_mask;
  args.end_mask = end_mask;
  args.ellipsis_mask = ellipsis_mask;
  args.new_axis_mask = new_axis_mask;
  args.data_type = static_cast<int64_t>(data_type);
  args.shrink_axis_mask = shrink_axis_mask;

  ConstGeTensorPtr weight0 = input[kStridedSliceInputIndex0];
  ConstGeTensorPtr weight1 = input[kStridedSliceInputIndex1];
  ConstGeTensorPtr weight2 = input[kStridedSliceInputIndex2];
  ConstGeTensorPtr weight3 = input[kStridedSliceInputIndex3];
  if (CheckWeight(weight0, weight1, weight2, weight3) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Check And Get Attr failed.");
    return PARAM_INVALID;
  }

  return SUCCESS;
}
Status StridedSliceKernel::CheckWeight(const ConstGeTensorPtr &weight0, const ConstGeTensorPtr &weight1,
                                       const ConstGeTensorPtr &weight2, const ConstGeTensorPtr &weight3) const {
  if ((weight0 == nullptr) || (weight1 == nullptr) || (weight2 == nullptr) || (weight3 == nullptr)) {
    GELOGW("weight is nullptr.");
    return PARAM_INVALID;
  }
  if (!(weight1->GetTensorDesc().GetDataType() == DT_INT32 && weight2->GetTensorDesc().GetDataType() == DT_INT32 &&
        weight3->GetTensorDesc().GetDataType() == DT_INT32)) {
    GELOGE(INTERNAL_ERROR, "Data type of StridedSlice OP(begin,end,strides) must be int32.");
    return INTERNAL_ERROR;
  }

  // check data
  size_t weight0_size = weight0->GetData().size() / sizeof(int32_t);
  size_t weight1_size = weight1->GetData().size() / sizeof(int32_t);
  size_t weight2_size = weight2->GetData().size() / sizeof(int32_t);
  size_t weight3_size = weight3->GetData().size() / sizeof(int32_t);
  if ((weight0_size == 0) || (weight1_size == 0) || (weight2_size == 0) || (weight3_size == 0)) {
    GELOGW("Data size of inputs is 0.");
    return PARAM_INVALID;
  }

  // check dim size
  size_t weight0_dim_size = weight0->GetTensorDesc().GetShape().GetDimNum();
  if (!((weight0_dim_size >= weight1_size) && (weight1_size == weight2_size) && (weight1_size == weight3_size))) {
    GELOGW("The sizes of begin, end and stride is not supported.");
    return NOT_CHANGED;
  }

  return SUCCESS;
}

Status StridedSliceKernel::MaskCal(const bool &begin_mask_flag, const bool &end_mask_flag, const bool &shrink_mask_flag,
                                   int32_t &begin_i, int32_t &end_i, int32_t &dim_i) const {
  if (shrink_mask_flag) {
    begin_i = (begin_i < 0 ? (dim_i + begin_i) : begin_i);
    FMK_INT32_ADDCHECK(begin_i, kNumOne);
    end_i = begin_i + kNumOne;
  } else {
    if (begin_mask_flag) {
      begin_i = 0;
    } else {
      begin_i = (begin_i < 0 ? (dim_i + begin_i) : begin_i);
    }
    if (end_mask_flag) {
      end_i = dim_i;
    } else {
      end_i = (end_i < 0 ? (dim_i + end_i) : end_i);
    }
  }
  return SUCCESS;
}

void StridedSliceKernel::GetOutputDims(uint32_t dims_size, const std::vector<int64_t> &output_dims, const Attr &args,
                                       vector<int64_t> &v_dims) {
  for (uint32_t k = 0; k < dims_size; k++) {
    bool shrink_mask_i = (static_cast<uint32_t>(args.shrink_axis_mask) & (1 << k));
    if (shrink_mask_i) {
      continue;
    }
    v_dims.push_back(output_dims[k]);
  }
}

Status StridedSliceKernel::Compute(const ge::OpDescPtr attr, const std::vector<ge::ConstGeTensorPtr> &input,
                                   vector<ge::GeTensorPtr> &v_output) {
  GELOGI("StridedSliceKernel in.");
  Attr args;
  Status ret = CheckAndGetAttr(attr, input, args);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "Check And Get Attr failed.");
    return NOT_CHANGED;
  }

  ConstGeTensorPtr weight0 = input[kStridedSliceInputIndex0];
  ConstGeTensorPtr weight1 = input[kStridedSliceInputIndex1];
  ConstGeTensorPtr weight2 = input[kStridedSliceInputIndex2];
  ConstGeTensorPtr weight3 = input[kStridedSliceInputIndex3];

  const GeShape x_shape = weight0->GetTensorDesc().GetShape();
  size_t dim_size = x_shape.GetDimNum();
  size_t data_size = weight0->GetData().size() / sizeof(int32_t);

  const int32_t *begin = reinterpret_cast<const int32_t *>(weight1->GetData().data());
  const int32_t *end = reinterpret_cast<const int32_t *>(weight2->GetData().data());
  const int32_t *stride = reinterpret_cast<const int32_t *>(weight3->GetData().data());
  if ((begin == nullptr) || (end == nullptr) || (stride == nullptr)) {
    GELOGE(PARAM_INVALID, "input weight tensor is nullptr.");
    return NOT_CHANGED;
  }

  std::vector<int64_t> input_dims;
  std::vector<int64_t> begin_vec;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> stride_vec;
  int64_t dim_final;
  for (size_t i = 0; i < dim_size; i++) {
    int32_t begin_i = begin[i];
    int32_t end_i = end[i];
    int32_t stride_i = stride[i];
    int32_t dim_i = static_cast<int32_t>(x_shape.GetDim(i));
    GELOGI("%d\t %d\t %d\t %d", begin_i, end_i, stride_i, dim_i);
    uint32_t i_temp = static_cast<uint32_t>(i);
    bool begin_mask_i = (static_cast<uint32_t>(args.begin_mask) & (1 << i_temp));
    bool end_mask_i = (static_cast<uint32_t>(args.end_mask) & (1 << i_temp));
    bool shrink_mask_i = (static_cast<uint32_t>(args.shrink_axis_mask) & (1 << i_temp));
    ret = MaskCal(begin_mask_i, end_mask_i, shrink_mask_i, begin_i, end_i, dim_i);
    if (ret != SUCCESS) {
      GELOGW("MaskCal failed, because of data overflow.");
      return NOT_CHANGED;
    }
    if (stride_i == 0) {
      stride_i = kDefaultSrideSize;
    } else if (stride_i < 0) {
      stride_i = -stride_i;
      begin_i = x_shape.GetDim(i) - begin_i - 1;
      end_i = x_shape.GetDim(i) - end_i - 1;
    }
    if ((begin_i == 0) && (end_i == 0)) {
      dim_final = x_shape.GetDim(i);
    } else {
      dim_final = abs(end_i - begin_i) / stride_i;
    }
    output_dims.push_back(dim_final);
    input_dims.push_back(x_shape.GetDim(i));
    begin_vec.push_back(begin_i);
    stride_vec.push_back(stride_i);
  }

  // Index 0 can always gets a GeTensorDesc object from any OpDescPtr.
  auto output_tensor_desc = attr->GetOutputDesc(0);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "MakeShared GeTensor failed, node name %s.", attr->GetName().c_str());
    return NOT_CHANGED;
  }

  void *data = reinterpret_cast<void *>(const_cast<uint8_t *>(weight0->GetData().data()));
  GE_CHECK_NOTNULL(data);
  ret = OpUtils::SetOutputSliceData(data, static_cast<int64_t>(data_size), args.data_type, input_dims, begin_vec,
                                    output_dims, output_ptr.get(), stride_vec);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "SetOutputSliceData failed.");
    return NOT_CHANGED;
  }

  GeTensorDesc &t_d = output_ptr->MutableTensorDesc();
  t_d.SetDataType(static_cast<DataType>(args.data_type));

  uint32_t final_dim_size = static_cast<uint32_t>(output_dims.size());
  vector<int64_t> v_dims;
  GetOutputDims(final_dim_size, output_dims, args, v_dims);
  t_d.SetShape(GeShape(v_dims));
  v_output.push_back(output_ptr);
  GELOGI("StridedSliceKernel success.");
  return SUCCESS;
}
REGISTER_KERNEL(STRIDEDSLICE, StridedSliceKernel);
}  // namespace ge
