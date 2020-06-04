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

#include "graph/passes/folding_kernel/concat_v2_kernel.h"

#include <memory>
#include <set>

#include "common/debug/log.h"
#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kConcatV2InputNum = 3;
const std::set<DataType> concatv2_supported_type = {DT_INT32, DT_FLOAT};

template <typename T>
void GetOutputData(std::vector<T> &y_data, int64_t loop, size_t &input_size,
                   const std::vector<ConstGeTensorPtr> &input) {
  for (int64_t i = 0; i < loop; i++) {
    for (size_t k = 0; k < input_size; k++) {
      GeShape datak_shape = input.at(k)->GetTensorDesc().GetShape();
      const T *datak = reinterpret_cast<const T *>(input.at(k)->GetData().data());
      int64_t gapk = datak_shape.GetShapeSize() / loop;  // [2,3] is 6/loop
      for (int64_t j = 0; j < gapk; j++) {
        y_data.push_back(datak[j + gapk * i]);
      }
    }
  }
}

#define SET_OUTPUT(DTYPE, TYPE)                                                                                  \
  case DTYPE:                                                                                                    \
    GetOutputData(y_data_##TYPE, loop, input_size, input);                                                       \
    (void)output_ptr->SetData(reinterpret_cast<uint8_t *>(y_data_##TYPE.data()), y_data_##TYPE.size() * length); \
    break;
}  // namespace

Status ConcatV2Kernel::Compute(const ge::OpDescPtr op_desc_ptr, const vector<ge::ConstGeTensorPtr> &input,
                               vector<ge::GeTensorPtr> &v_output) {
  GELOGI("ConcatV2Kernel in.");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "input opdesc is nullptr.");
    return PARAM_INVALID;
  }
  int tidx = -1;
  Status ret = ConcatV2PreCompute(input, tidx);
  if (ret != SUCCESS) {
    return ret;
  }

  size_t input_size = input.size();  // N + 1
  input_size--;                      // N

  ConstGeTensorPtr tensor0 = input.at(0);
  GE_CHECK_NOTNULL(tensor0);
  DataType data_type = tensor0->GetTensorDesc().GetDataType();
  uint32_t length = 0;
  if (!TypeUtils::GetDataTypeLength(data_type, length)) {
    GELOGW("Can't GetDataTypeLength of data_type: %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }

  std::vector<int32_t> y_data_int32_t;
  std::vector<float> y_data_float;

  // Index 0 can always gets a GeTensorDesc object from any OpDescPtr.
  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(0);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "MakeShared failed.");
    return MEMALLOC_FAILED;
  }

  GeShape data0_shape = tensor0->GetTensorDesc().GetShape();
  int64_t loop = 1;
  for (int i = 0; i < tidx; i++) {
    loop *= data0_shape.GetDim(i);
  }

  switch (data_type) {
    SET_OUTPUT(DT_INT32, int32_t)
    SET_OUTPUT(DT_FLOAT, float)
    default:
      break;
  }
  output_ptr->MutableTensorDesc().SetDataType(data_type);
  output_ptr->MutableTensorDesc().SetShape(GeShape({op_desc_ptr->GetOutputDesc(0).GetShape()}));
  v_output.push_back(output_ptr);
  GELOGI("ConcatV2Kernel success.");
  return SUCCESS;
}

Status ConcatV2Kernel::ConcatV2PreCompute(const std::vector<ConstGeTensorPtr> &input, int &tidx) {
  size_t input_size = input.size();
  // N >= 2 and N + 1 >= 3
  if (input_size < kConcatV2InputNum) {
    GELOGI("The number of input for ConcatV2 must not be less than %zu.", kConcatV2InputNum);
    return NOT_CHANGED;
  }

  for (size_t i = 0; i < input_size; i++) {
    if (input[i] == nullptr) {
      GELOGI("Input%zu must not be null.", i);
      return NOT_CHANGED;
    }
    if (input.at(i)->GetData().size() == 0) {
      GELOGI("Check data size fail. input%zu size is 0.", i);
      return NOT_CHANGED;
    }
  }

  input_size--;
  ConstGeTensorPtr tensor0 = input.at(0);
  GE_CHECK_NOTNULL(tensor0);
  DataType data_type = tensor0->GetTensorDesc().GetDataType();
  for (size_t i = 1; i < input_size; i++) {
    if (data_type != input.at(i)->GetTensorDesc().GetDataType()) {
      GELOGI("Data type of N inputs for ConcatV2 not the same, check input %zu failed.", i);
      return NOT_CHANGED;
    }
  }

  // check if input data type is supported
  if (concatv2_supported_type.find(data_type) == concatv2_supported_type.end()) {
    GELOGI("ConcatV2 does not support this Data type: %s.", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }

  ConstGeTensorPtr tensor_axis = input.at(input_size);
  GE_CHECK_NOTNULL(tensor_axis);
  const int *axis = reinterpret_cast<const int *>(tensor_axis->GetData().data());
  tidx = axis[0];                                                                // [-rank(values), rank(values))
  int dims = static_cast<int>(tensor0->GetTensorDesc().GetShape().GetDimNum());  // rank
  if (tidx < 0) {
    tidx += dims;
  }
  if (tidx < 0 || tidx > dims) {
    GELOGI("ConcatV2 tidx not legal.");
    return NOT_CHANGED;
  }

  return SUCCESS;
}

REGISTER_KERNEL(CONCATV2, ConcatV2Kernel);
}  // namespace ge
