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

#include "graph/passes/folding_kernel/slice_kernel.h"

#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

using domi::SLICE;

namespace ge {
namespace {
const size_t kSliceInputSize = 3;
const size_t kSliceInputIndexX = 0;
const size_t kSliceInputIndexBegin = 1;
const size_t kSliceInputIndexSize = 2;
}  // namespace

Status SliceKernel::Compute(const OpDescPtr attr, const std::vector<ConstGeTensorPtr> &input,
                            vector<GeTensorPtr> &v_output) {
  GELOGI("SliceKernel in.");
  if (attr == nullptr) {
    GELOGW("Input opdescptr is nullptr.");
    return NOT_CHANGED;
  }
  // check input size
  if (input.size() != kSliceInputSize) {
    GELOGE(PARAM_INVALID, "The number of input for slice must be %zu.", kSliceInputSize);
    return NOT_CHANGED;
  }

  ConstGeTensorPtr x_ = input[kSliceInputIndexX];
  ConstGeTensorPtr begin = input[kSliceInputIndexBegin];
  ConstGeTensorPtr size = input[kSliceInputIndexSize];
  if (x_ == nullptr || begin == nullptr || size == nullptr) {
    GELOGE(PARAM_INVALID, "input tensor is nullptr.");
    return NOT_CHANGED;
  }

  // data type in input_x
  auto data_type = x_->GetTensorDesc().GetDataType();
  // check data type of begin and size
  if (begin->GetTensorDesc().GetDataType() != DT_INT32 || size->GetTensorDesc().GetDataType() != DT_INT32) {
    GELOGE(PARAM_INVALID, "Data type of begin and size for slice are not DT_INT32.");
    return NOT_CHANGED;
  }

  void *data = reinterpret_cast<void *>(const_cast<uint8_t *>(x_->GetData().data()));
  int32_t *begin_data = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(begin->GetData().GetData()));
  int32_t *size_data = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(size->GetData().GetData()));
  GE_CHECK_NOTNULL(data);
  GE_CHECK_NOTNULL(begin_data);
  GE_CHECK_NOTNULL(size_data);

  size_t data_size = x_->GetData().size() / sizeof(int32_t);
  size_t begin_size = begin->GetData().size() / sizeof(int32_t);
  size_t size_size = size->GetData().size() / sizeof(int32_t);
  const ge::GeShape &x_shape = x_->GetTensorDesc().GetShape();
  size_t dim_size = x_shape.GetDimNum();
  if (dim_size != begin_size || dim_size != size_size) {
    GELOGE(PARAM_INVALID, "Data type of begin and size for slice are not DT_INT32.");
    return NOT_CHANGED;
  }

  std::vector<int64_t> input_dims;
  std::vector<int64_t> begin_vec;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> stride_vec;
  for (size_t i = 0; i < dim_size; i++) {
    int32_t begin_i = begin_data[i];
    int32_t size_i = size_data[i];
    int64_t dim_i = x_shape.GetDim(i);
    if (size_i < 0) {
      GE_IF_BOOL_EXEC(((dim_i - begin_i) > INT32_MAX) || ((dim_i - begin_i) < INT32_MIN),
                      GELOGE(PARAM_INVALID, " %ld and %d sub can result in overflow!.", dim_i, begin_i);
                      return INTERNAL_ERROR);
      size_i = dim_i - begin_i;
    }
    input_dims.push_back(dim_i);
    begin_vec.push_back(begin_i);
    output_dims.push_back(size_i);
    stride_vec.push_back(1);
  }
  // construct tensorDesc
  ge::GeShape output_shape(output_dims);
  GeTensorDesc output_tensor_desc(output_shape, FORMAT_NCHW, data_type);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "make_shared ge::GeTensor failed, node name %s.", attr->GetName().c_str());
    return NOT_CHANGED;
  }
  Status ret = OpUtils::SetOutputSliceData(data, static_cast<int64_t>(data_size), data_type, input_dims, begin_vec,
                                           output_dims, output_ptr.get(), stride_vec);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "SetOutputSliceData failed.");
    return NOT_CHANGED;
  }
  v_output.push_back(output_ptr);
  GELOGI("SliceKernel success.");
  return SUCCESS;
}

REGISTER_KERNEL(SLICE, SliceKernel);
}  // namespace ge
