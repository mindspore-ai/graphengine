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

#include "graph/passes/folding_kernel/add_kernel.h"

#include <cfloat>

#include "graph/common/bcast.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kAddFirstInput = 0;
const size_t kAddSecondInput = 1;
const size_t kAddFirstOutput = 0;
const size_t kAddInputSize = 2;
const size_t kAddOutputSize = 1;

#define SET_BCAST_ADD_CASE(DTYPE, TYPE)                 \
  case (DTYPE):                                         \
    ret = BCastAdd<TYPE>(op_desc_ptr, input, v_output); \
    break;

#define SET_OVERFLOW_CHECK_SIGNED_CASE(DTYPE, MAX_VALUE, MIN_VALUE)                 \
  case (DTYPE):                                                                     \
    if (((y > 0) && (x > ((MAX_VALUE)-y))) || ((y < 0) && (x < ((MIN_VALUE)-y)))) { \
      overflow_flag = true;                                                         \
    }                                                                               \
    break;

#define SET_OVERFLOW_CHECK_UNSIGNED_CASE(DTYPE, TYPE, MAX_VALUE)          \
  case (DTYPE): {                                                         \
    TYPE threshold = static_cast<TYPE>(static_cast<TYPE>(MAX_VALUE) - y); \
    if (static_cast<TYPE>(x) > threshold) {                               \
      overflow_flag = true;                                               \
    }                                                                     \
    break;                                                                \
  }

}  // namespace

template <typename T>
bool AddKernel::OverflowCheck(const T &x, const T &y, DataType data_type) {
  bool overflow_flag = false;

  switch (data_type) {
    SET_OVERFLOW_CHECK_SIGNED_CASE(DT_INT8, INT8_MAX, INT8_MIN)
    SET_OVERFLOW_CHECK_SIGNED_CASE(DT_INT16, INT16_MAX, INT16_MIN)
    SET_OVERFLOW_CHECK_SIGNED_CASE(DT_INT32, INT32_MAX, INT32_MIN)
    SET_OVERFLOW_CHECK_SIGNED_CASE(DT_INT64, INT64_MAX, INT64_MIN)
    SET_OVERFLOW_CHECK_SIGNED_CASE(DT_FLOAT, FLT_MAX, FLT_MIN)
    SET_OVERFLOW_CHECK_SIGNED_CASE(DT_DOUBLE, DBL_MAX, DBL_MIN)
    SET_OVERFLOW_CHECK_UNSIGNED_CASE(DT_UINT8, uint8_t, UINT8_MAX)
    SET_OVERFLOW_CHECK_UNSIGNED_CASE(DT_UINT16, uint16_t, UINT16_MAX)
    SET_OVERFLOW_CHECK_UNSIGNED_CASE(DT_UINT32, uint32_t, UINT32_MAX)
    SET_OVERFLOW_CHECK_UNSIGNED_CASE(DT_UINT64, uint64_t, UINT64_MAX)
    default:
      break;
  }

  return overflow_flag;
}

template <typename InT>
Status AddKernel::BCastAdd(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                           std::vector<GeTensorPtr> &v_output) {
  // only broadcast shape
  BCast bcast;
  Status ret = bcast.GenerateBcastInfo(ge::BCast::TransShapeToDimVec(input[kAddFirstInput]->GetTensorDesc()),
                                       ge::BCast::TransShapeToDimVec(input[kAddSecondInput]->GetTensorDesc()));
  if (ret != SUCCESS) {
    GELOGE(ret, "Greater broadcasting failed.");
    return ret;
  }

  std::vector<int64_t> x_indexes;
  std::vector<int64_t> y_indexes;
  bcast.BCastIndexes(x_indexes, y_indexes);

  if (input[kAddFirstInput]->GetData().size() < sizeof(InT)) {
    GELOGE(FAILED, "The size of the first input is less than the size of the InT.");
    return FAILED;
  }
  auto x1_data = reinterpret_cast<const InT *>(input[kAddFirstInput]->GetData().data());

  if (input[kAddSecondInput]->GetData().size() < sizeof(InT)) {
    GELOGE(FAILED, "The size of the second input is less than the size of the InT.");
    return FAILED;
  }
  auto x2_data = reinterpret_cast<const InT *>(input[kAddSecondInput]->GetData().data());

  size_t data_num = x_indexes.size();
  InT *data = nullptr;
  data = new (std::nothrow) InT[data_num]();
  GE_CHECK_NOTNULL(data);

  DataType data_type = input[kAddFirstInput]->GetTensorDesc().GetDataType();
  for (size_t i = 0; i < data_num; i++) {
    auto x_index = *(x1_data + x_indexes[i]);
    auto y_index = *(x2_data + y_indexes[i]);
    if (OverflowCheck<InT>(x_index, y_index, data_type)) {
      GELOGE(PARAM_INVALID, "Result of add is overflow.");
      GE_DELETE_NEW_ARRAY(data);
      return PARAM_INVALID;
    }
    data[i] = x_index + y_index;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(kAddFirstOutput));
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed");
    GE_DELETE_NEW_ARRAY(data);
    return MEMALLOC_FAILED;
  }
  if (output_ptr->SetData(reinterpret_cast<uint8_t *>(data), data_num * sizeof(InT))) {
    GELOGW("GetRange: SetData failed");
  }
  GE_DELETE_NEW_ARRAY(data);

  output_ptr->MutableTensorDesc().SetDataType(data_type);
  vector<int64_t> bcast_dims = bcast.GetOutputShape();
  output_ptr->MutableTensorDesc().SetShape(GeShape(bcast_dims));
  v_output.push_back(output_ptr);

  return SUCCESS;
}

Status AddKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                          std::vector<GeTensorPtr> &v_output) {
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Op_desc_ptr must not be null.");
    return NOT_CHANGED;
  }
  // check how many inputs
  if ((input.size() != kAddInputSize) || (op_desc_ptr->GetOutputsSize() != kAddOutputSize)) {
    GELOGE(PARAM_INVALID, "The number of input for add must be %zu, output number must be %zu.", kAddInputSize,
           kAddOutputSize);
    return NOT_CHANGED;
  }

  // input vector elements must not be null
  if ((input[kAddFirstInput] == nullptr) || (input[kAddSecondInput] == nullptr)) {
    GELOGE(PARAM_INVALID, "Input vector elements must not be null.");
    return NOT_CHANGED;
  }

  // Inputs must have the same datatype.
  DataType data_type_0 = input[kAddFirstInput]->GetTensorDesc().GetDataType();
  DataType data_type_1 = input[kAddSecondInput]->GetTensorDesc().GetDataType();
  if (data_type_0 != data_type_1) {
    GELOGE(PARAM_INVALID, "Data type of inputs for add not matched, data_type_0:%s, data_type_1:%s",
           TypeUtils::DataTypeToSerialString(data_type_0).c_str(),
           TypeUtils::DataTypeToSerialString(data_type_1).c_str());
    return NOT_CHANGED;
  }

  // Checking whether the weightdef contains data
  if ((input[kAddFirstInput]->GetData().size() == 0) || (input[kAddSecondInput]->GetData().size() == 0)) {
    GELOGW("Data size of input0 is %zu, input1 is %zu.", input[kAddFirstInput]->GetData().size(),
           input[kAddSecondInput]->GetData().size());
    return NOT_CHANGED;
  }

  Status ret = NOT_CHANGED;
  switch (data_type_0) {
    SET_BCAST_ADD_CASE(DT_INT8, int8_t)
    SET_BCAST_ADD_CASE(DT_INT16, int16_t)
    SET_BCAST_ADD_CASE(DT_INT32, int32_t)
    SET_BCAST_ADD_CASE(DT_INT64, int64_t)
    SET_BCAST_ADD_CASE(DT_UINT8, uint8_t)
    SET_BCAST_ADD_CASE(DT_UINT16, uint16_t)
    SET_BCAST_ADD_CASE(DT_UINT32, uint32_t)
    SET_BCAST_ADD_CASE(DT_UINT64, uint64_t)
    SET_BCAST_ADD_CASE(DT_FLOAT, float)
    SET_BCAST_ADD_CASE(DT_DOUBLE, double)
    default:
      GELOGI("Add kernel data type %s not support.", TypeUtils::DataTypeToSerialString(data_type_0).c_str());
      return NOT_CHANGED;
  }

  if (ret != SUCCESS) {
    GELOGE(ret, "Greater broadcasting failed.");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

REGISTER_KERNEL(ADD, AddKernel);
}  // namespace ge
