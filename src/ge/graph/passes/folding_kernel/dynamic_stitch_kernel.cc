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

#include "graph/passes/folding_kernel/dynamic_stitch_kernel.h"

#include <memory>

#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const int kDoubleAttrN = 2;
}  // namespace
Status DynamicStitchKernel::Compute(const OpDescPtr op_desc_ptr, const vector<ConstGeTensorPtr> &input,
                                    vector<GeTensorPtr> &v_output) {
  GELOGI("DynamicStitch Kernel in.");
  Status validate_ret = ValidateParams(op_desc_ptr, input);
  if (validate_ret != SUCCESS) {
    GELOGW("Dynamic stitch kernel params validate failed.");
    return validate_ret;
  }

  GE_CHECK_NOTNULL(input[n_]);
  auto data_type = input[n_]->GetTensorDesc().GetDataType();
  Status ret;
  switch (data_type) {
    case DT_INT8:
      ret = GenData<int8_t>(input, v_output);
      break;
    case DT_UINT8:
      ret = GenData<uint8_t>(input, v_output);
      break;
    case DT_INT16:
      ret = GenData<int16_t>(input, v_output);
      break;
    case DT_UINT16:
      ret = GenData<uint16_t>(input, v_output);
      break;
    case DT_INT32:
      ret = GenData<int32_t>(input, v_output);
      break;
    case DT_INT64:
      ret = GenData<int64_t>(input, v_output);
      break;
    case DT_BOOL:
      ret = GenData<bool>(input, v_output);
      break;
    case DT_FLOAT16:
      ret = GenData<fp16_t>(input, v_output);
      break;
    case DT_FLOAT:
      ret = GenData<float>(input, v_output);
      break;
    case DT_DOUBLE:
      ret = GenData<double>(input, v_output);
      break;
    default:
      ret = NOT_CHANGED;
      GELOGI("Dynamic stitch op not support data type of %s.", TypeUtils::DataTypeToSerialString(data_type).c_str());
      break;
  }
  if (ret != SUCCESS) {
    GELOGW("Dynamic stitch folding failed.");
    return ret;
  }
  GELOGI("Dynamic stitch end.");
  return SUCCESS;
}

Status DynamicStitchKernel::ValidateParams(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) {
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "input opdesc is nullptr.");
    return PARAM_INVALID;
  }
  // validate input
  // input[0]~input[N-1] is indices, input[N]~input[2N-1] is datas
  if (input.empty()) {
    GELOGI("Input is empty.Ignore dynamic stitch kernel.");
    return NOT_CHANGED;
  }
  // validate attrs
  if (!(AttrUtils::GetInt(op_desc_ptr, DYNAMIC_STITCH_ATTR_NAME_NUM, n_))) {
    GELOGW("Attr %s is not exist.", DYNAMIC_STITCH_ATTR_NAME_NUM.c_str());
    return NOT_CHANGED;
  }
  // validate attr N and input.size
  if ((kDoubleAttrN * n_) != static_cast<int>(input.size())) {
    GELOGW("Input size is not not match with attr N. Ignore dynamic stitch kernel.");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

template <typename T>
void DynamicStitchKernel::ComputeMergedShape(const vector<ConstGeTensorPtr> &input, GeShape &merged_shape,
                                             map<int32_t, T> &indice_data_mapping) {
  // data[i].shape = indices[i].shape + constant
  size_t indice_dim = input[0]->GetTensorDesc().GetShape().GetDimNum();
  // index n_ for input is less than size of input
  GeShape input_n_shape = input[n_]->GetTensorDesc().GetShape();
  int64_t dim_offset = (input_n_shape.GetDimNum() == indice_dim) ? 0 : input_n_shape.GetDim(indice_dim);

  int64_t merged_first_dim = 0;
  vector<int64_t> indice_dims;
  for (int i = 0; i < n_; i++) {
    // all index for input is less than size of input
    indice_dims = input[i]->GetTensorDesc().GetShape().GetDims();
    int32_t *input_indice = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(input[i]->GetData().data()));
    T *input_data = const_cast<T *>(reinterpret_cast<const T *>(input[i + n_]->GetData().data()));
    // scaler indice has one element
    if (indice_dims.empty()) {
      // if indice repeated, need new data replace old data
      indice_data_mapping[input_indice[0]] = input_data[0];
      merged_first_dim = (merged_first_dim > input_indice[0]) ? merged_first_dim : input_indice[0];
      continue;
    }
    // vector indice element mapping
    for (const auto &dim : indice_dims) {
      for (auto j = 0; j < dim; j++) {
        // if indice repeated, need new data replace old data
        indice_data_mapping[input_indice[j]] = input_data[j];
        merged_first_dim = (merged_first_dim > input_indice[j]) ? merged_first_dim : input_indice[j];
      }
    }
  }
  ++merged_first_dim;

  vector<int64_t> merged_dim_vec = {merged_first_dim};
  if (dim_offset != 0) {
    merged_dim_vec.emplace_back(dim_offset);
    GELOGI("merged_shape is [ %ld, %ld].", merged_first_dim, dim_offset);
  }
  merged_shape = GeShape(merged_dim_vec);
  GELOGI("merged_shape is [ %ld ].", merged_first_dim);
}

template <typename T>
Status DynamicStitchKernel::GenData(const vector<ConstGeTensorPtr> &input, vector<GeTensorPtr> &v_output) {
  GeShape merged_shape;
  map<int32_t, T> indice_data_mapping;
  ComputeMergedShape(input, merged_shape, indice_data_mapping);

  int64_t output_size = merged_shape.GetShapeSize();
  unique_ptr<T[]> buf(new (std::nothrow) T[output_size]());
  if (buf == nullptr) {
    GELOGE(MEMALLOC_FAILED, "new buf failed");
    return INTERNAL_ERROR;
  }
  for (const auto &indice_data : indice_data_mapping) {
    auto index = indice_data.first;
    buf[index] = indice_data.second;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>();
  if (output_ptr == nullptr) {
    GELOGW("Fail to malloc output.");
    return NOT_CHANGED;
  }
  auto dtype = input[n_]->GetTensorDesc().GetDataType();
  output_ptr->MutableTensorDesc().SetDataType(dtype);
  output_ptr->MutableTensorDesc().SetShape(merged_shape);

  uint32_t length = 1;
  if (!TypeUtils::GetDataTypeLength(dtype, length)) {
    GELOGW("Can't GetDataTypeLength of data_type: %s", TypeUtils::DataTypeToSerialString(dtype).c_str());
    return NOT_CHANGED;
  }
  GE_IF_BOOL_EXEC(output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()),
                                      static_cast<size_t>(output_size * length)) != GRAPH_SUCCESS,
                  GELOGE(INTERNAL_ERROR, "set data failed");
                  return NOT_CHANGED);
  v_output.push_back(output_ptr);
  return SUCCESS;
}

REGISTER_KERNEL(DYNAMICSTITCH, DynamicStitchKernel);
}  // namespace ge
