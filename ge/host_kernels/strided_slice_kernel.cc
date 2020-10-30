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

#include "host_kernels/strided_slice_kernel.h"

#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/math/math_util.h"
#include "common/op/ge_op_utils.h"
#include "external/graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "host_kernels/kernel_utils.h"
#include "inc/kernel_factory.h"
#include <memory>

namespace ge {
namespace {
const int32_t kNumOne = 1;
const size_t kStridedSliceInputSize = 4;
const size_t kStridedSliceInputIndex = 0;
const size_t kStridedSliceBeginIndex = 1;
const size_t kStridedSliceEndIndex = 2;
const size_t kStridedSliceStrideIndex = 3;
const int32_t kDefaultStrideSize = 1;
const std::set<DataType> kIndexNumberType = {DT_INT32, DT_INT64};

bool IsEllipsisMaskValid(const GeTensorDescPtr &input_desc, const int ellipsis_mask) {
  if (ellipsis_mask != 0) {
    auto ellipsis_num = 0;
    auto input_shape = input_desc->GetShape();
    bool ellipsis_mask_flag = false;
    for (size_t i = 0; i < input_shape.GetDimNum(); i++) {
      uint32_t i_temp = static_cast<uint32_t>(i);
      ellipsis_mask_flag = (static_cast<uint32_t>(ellipsis_mask) & (1 << i_temp));
      if (ellipsis_mask_flag) {
        ++ellipsis_num;
      }
      if (ellipsis_num > 1) {
        GELOGW("Only one non-zero bit is allowed in ellipsis_mask.");
        return false;
      }
    }
  }
  return true;
}
}  // namespace
Status StridedSliceKernel::Compute(const ge::OpDescPtr attr, const std::vector<ge::ConstGeTensorPtr> &input,
                                   vector<ge::GeTensorPtr> &v_output) {
  GELOGD("StridedSliceKernel in.");
  // 1.Check input and attrs
  if (CheckAndGetAttr(attr) != SUCCESS) {
    GELOGW("Check and get attrs failed.Ignore kernel.");
    return NOT_CHANGED;
  }
  if (CheckInputParam(input) != SUCCESS) {
    GELOGW("Check input params failed.Ignore kernel.");
    return NOT_CHANGED;
  }
  // 2.Init param with mask attrs.
  std::vector<int64_t> input_dims;
  std::vector<int64_t> begin_vec;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> stride_vec;
  if (InitParamWithAttrs(input, input_dims, begin_vec, output_dims, stride_vec) != SUCCESS) {
    GELOGW("Init param with mask attrs failed.Ignore kernel.");
    return NOT_CHANGED;
  }

  // 3.Set sliced data to output_ptr
  ConstGeTensorPtr weight0 = input[kStridedSliceInputIndex];
  auto data_type = weight0->GetTensorDesc().GetDataType();
  size_t data_size = weight0->GetData().size() / GetSizeByDataType(data_type);
  void *data = reinterpret_cast<void *>(const_cast<uint8_t *>(weight0->GetData().data()));
  GE_CHECK_NOTNULL(data);
  // Index 0 can always gets a GeTensorDesc object from any OpDescPtr.
  auto output_tensor_desc = attr->GetOutputDesc(0);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "MakeShared GeTensor failed, node name %s.", attr->GetName().c_str());
    return NOT_CHANGED;
  }
  auto ret = OpUtils::SetOutputSliceData(data, static_cast<int64_t>(data_size), data_type, input_dims, begin_vec,
                                         output_dims, output_ptr.get(), stride_vec);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "SetOutputSliceData failed.");
    return NOT_CHANGED;
  }

  // 4.Set output data_type and shape
  GeTensorDesc &t_d = output_ptr->MutableTensorDesc();
  t_d.SetDataType(static_cast<DataType>(data_type));

  auto final_dim_size = static_cast<uint32_t>(output_dims.size());
  vector<int64_t> v_dims;
  GetOutputDims(final_dim_size, output_dims, v_dims);
  t_d.SetShape(GeShape(v_dims));
  v_output.push_back(output_ptr);
  GELOGI("StridedSliceKernel success.");
  return SUCCESS;
}
Status StridedSliceKernel::CheckAndGetAttr(const OpDescPtr &attr) {
  if (attr == nullptr) {
    GELOGE(PARAM_INVALID, "input opdescptr is nullptr.");
    return PARAM_INVALID;
  }
  // Get all op attr value of strided_slice
  for (auto &attr_2_value : attr_value_map_) {
    if (!AttrUtils::GetInt(attr, attr_2_value.first, attr_2_value.second)) {
      GELOGE(PARAM_INVALID, "Get %s attr failed.", attr_2_value.first.c_str());
      return PARAM_INVALID;
    }
  }
  // Check ellipsis_mask is valid
  const auto &input_desc = attr->MutableInputDesc(kStridedSliceInputIndex);
  GE_CHECK_NOTNULL(input_desc);
  auto ellipsis_mask = attr_value_map_.at(STRIDE_SLICE_ATTR_ELLIPSIS_MASK);
  if (!IsEllipsisMaskValid(input_desc, ellipsis_mask)) {
    return PARAM_INVALID;
  }
  return SUCCESS;
}
Status StridedSliceKernel::CheckInputParam(const std::vector<ConstGeTensorPtr> &input) const {
  if (input.size() != kStridedSliceInputSize) {
    GELOGE(PARAM_INVALID, "The number of input for strided slice must be %zu.", kStridedSliceInputSize);
    return PARAM_INVALID;
  }

  ConstGeTensorPtr weight0 = input[kStridedSliceInputIndex];
  ConstGeTensorPtr begin_tensor = input[kStridedSliceBeginIndex];
  ConstGeTensorPtr end_tensor = input[kStridedSliceEndIndex];
  ConstGeTensorPtr stride_tensor = input[kStridedSliceStrideIndex];
  GE_CHECK_NOTNULL(weight0);
  GE_CHECK_NOTNULL(begin_tensor);
  GE_CHECK_NOTNULL(end_tensor);
  GE_CHECK_NOTNULL(stride_tensor);

  // check if begin,end,strides data type is supported
  auto begin_tensor_desc = begin_tensor->GetTensorDesc();
  auto end_tensor_desc = begin_tensor->GetTensorDesc();
  auto stride_tensor_desc = begin_tensor->GetTensorDesc();
  if (begin_tensor_desc.GetDataType() != end_tensor_desc.GetDataType() ||
      end_tensor_desc.GetDataType() != stride_tensor_desc.GetDataType()) {
    GELOGW("Data type of StridedSlice OP(begin,end,strides) must be same.");
    return PARAM_INVALID;
  }
  if (kIndexNumberType.find(begin_tensor_desc.GetDataType()) == kIndexNumberType.end()) {
    GELOGW("Data type of StridedSlice OP(begin,end,strides) must be int32 or int64.");
    return PARAM_INVALID;
  }

  // check data
  auto x_data_type = weight0->GetTensorDesc().GetDataType();
  auto x_data_size = GetSizeByDataType(x_data_type);
  if (x_data_size < 0) {
    GELOGW("Data type of x input %s is not supported.", TypeUtils::DataTypeToSerialString(x_data_type).c_str());
    return PARAM_INVALID;
  }
  size_t weight0_size = weight0->GetData().size() / x_data_size;
  size_t begin_data_size = begin_tensor->GetData().size() / sizeof(int32_t);
  size_t end_data_size = end_tensor->GetData().size() / sizeof(int32_t);
  size_t stride_data_size = stride_tensor->GetData().size() / sizeof(int32_t);
  if ((weight0_size == 0) || (begin_data_size == 0) || (end_data_size == 0) || (stride_data_size == 0)) {
    GELOGW("Data size of inputs is 0.");
    return PARAM_INVALID;
  }
  // check dim size
  if (!((begin_data_size == end_data_size) && (end_data_size == stride_data_size))) {
    GELOGW("The sizes of begin, end and stride is not supported.");
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status StridedSliceKernel::InitParamWithAttrs(const std::vector<ConstGeTensorPtr> &input,
                                              std::vector<int64_t> &input_dims, std::vector<int64_t> &begin_vec,
                                              std::vector<int64_t> &output_dims, std::vector<int64_t> &stride_vec) {
  ConstGeTensorPtr weight0 = input[kStridedSliceInputIndex];
  ConstGeTensorPtr begin_tensor = input[kStridedSliceBeginIndex];
  ConstGeTensorPtr end_tensor = input[kStridedSliceEndIndex];
  ConstGeTensorPtr stride_tensor = input[kStridedSliceStrideIndex];

  const GeShape x_shape = weight0->GetTensorDesc().GetShape();
  auto x_dims = x_shape.GetDims();
  auto x_dims_num = x_shape.GetDimNum();
  // handle new_axis_mask
  ExpandDimsWithNewAxis(begin_tensor, x_dims_num, x_dims);

  const int32_t *begin = reinterpret_cast<const int32_t *>(begin_tensor->GetData().data());
  const int32_t *end = reinterpret_cast<const int32_t *>(end_tensor->GetData().data());
  const int32_t *stride = reinterpret_cast<const int32_t *>(stride_tensor->GetData().data());
  auto begin_dim_num = begin_tensor->GetData().size() / sizeof(int32_t);
  auto min_dim = x_dims_num > begin_dim_num ? begin_dim_num : x_dims_num;
  for (size_t i = 0; i < x_dims.size(); ++i) {
    auto i_temp = static_cast<uint64_t>(i);
    bool new_axis_mask_flag =
      (static_cast<uint64_t>(attr_value_map_.at(STRIDE_SLICE_ATTR_NEW_AXIS_MASK)) & (1 << i_temp));
    if (new_axis_mask_flag) {
      output_dims.push_back(1);
      input_dims.push_back(1);
      begin_vec.push_back(0);
      stride_vec.push_back(1);
      continue;
    }

    int64_t begin_i = 0;
    int64_t end_i = 0;
    int64_t stride_i = 1;
    if (i < min_dim) {
      begin_i = begin[i];
      end_i = end[i];
      stride_i = stride[i];
    } else {
      begin_i = 0;
      end_i = x_dims.at(i);
      stride_i = 1;
    }
    GELOGD("Before mask calculate. Begin is : %d\t,end is : %d\t stride is : %d\t x_dim_i is : %d.", begin_i, end_i,
           stride_i, x_dims.at(i));
    auto ret = MaskCal(i, begin_i, end_i, x_dims.at(i));
    if (ret != SUCCESS) {
      GELOGW("MaskCal failed, because of data overflow.");
      return NOT_CHANGED;
    }
    int64_t dim_final;
    GELOGD("Before stride calculate. Begin is : %d\t,end is : %d\t stride is : %d\t x_dim_i is : %d.", begin_i, end_i,
           stride_i, x_dims.at(i));
    (void)StrideCal(x_dims.at(i), begin_i, end_i, stride_i, dim_final);
    output_dims.push_back(dim_final);
    input_dims.push_back(x_dims.at(i));
    begin_vec.push_back(begin_i);
    stride_vec.push_back(stride_i);
  }
  return SUCCESS;
}
void StridedSliceKernel::ExpandDimsWithNewAxis(const ConstGeTensorPtr &begin_tensor, const size_t x_dims_num,
                                               vector<int64_t> &x_dims) {
  auto begin_data_type_size = GetSizeByDataType(begin_tensor->GetTensorDesc().GetDataType());
  size_t begin_vec_size = begin_tensor->GetData().size() / begin_data_type_size;
  auto final_dim_num = x_dims_num < begin_vec_size ? begin_vec_size : x_dims_num;
  for (size_t i = 0; i < final_dim_num; i++) {
    auto i_temp = static_cast<uint64_t>(i);
    bool new_axis_mask_flag =
      (static_cast<uint64_t>(attr_value_map_.at(STRIDE_SLICE_ATTR_NEW_AXIS_MASK)) & (1 << i_temp));
    if (new_axis_mask_flag) {
      x_dims.insert(x_dims.begin() + i, 1);
    }
  }
}
Status StridedSliceKernel::MaskCal(const size_t i, int64_t &begin_i, int64_t &end_i, int64_t &dim_i) const {
  uint64_t i_temp = static_cast<uint64_t>(i);
  bool begin_mask_flag = (static_cast<uint64_t>(attr_value_map_.at(STRIDE_SLICE_ATTR_BEGIN_MASK)) & (1 << i_temp));
  bool end_mask_flag = (static_cast<uint64_t>(attr_value_map_.at(STRIDE_SLICE_ATTR_END_MASK)) & (1 << i_temp));
  bool ellipsis_mask_flag =
    (static_cast<uint64_t>(attr_value_map_.at(STRIDE_SLICE_ATTR_ELLIPSIS_MASK)) & (1 << i_temp));
  bool shrink_mask_flag =
    (static_cast<uint32_t>(attr_value_map_.at(STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK)) & (1 << i_temp));
  if (shrink_mask_flag) {
    begin_i = (begin_i < 0 ? (dim_i + begin_i) : begin_i);
    FMK_INT32_ADDCHECK(begin_i, kNumOne)
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
    if (ellipsis_mask_flag) {
      begin_i = 0;
      end_i = dim_i;
    }
  }
  return SUCCESS;
}
Status StridedSliceKernel::StrideCal(const int64_t x_dims_i, int64_t &begin_i, int64_t &end_i, int64_t &stride_i,
                                     int64_t &dim_final) const {
  if (stride_i == 0) {
    stride_i = kDefaultStrideSize;
  } else if (stride_i < 0) {
    stride_i = -stride_i;
    begin_i = x_dims_i - begin_i - 1;
    end_i = x_dims_i - end_i - 1;
  }

  if (end_i > x_dims_i) {
    end_i = x_dims_i;
  }

  if ((begin_i == 0) && (end_i == 0)) {
    dim_final = x_dims_i;
  } else {
    dim_final = abs(end_i - begin_i) / stride_i;
  }
  return SUCCESS;
}
void StridedSliceKernel::GetOutputDims(uint32_t dims_size, const std::vector<int64_t> &output_dims,
                                       vector<int64_t> &v_dims) {
  for (uint32_t k = 0; k < dims_size; k++) {
    bool shrink_mask_i = (static_cast<uint32_t>(attr_value_map_.at(STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK)) & (1 << k));
    if (shrink_mask_i) {
      continue;
    }
    v_dims.push_back(output_dims[k]);
  }
}
REGISTER_KERNEL(STRIDEDSLICE, StridedSliceKernel);
}  // namespace ge
