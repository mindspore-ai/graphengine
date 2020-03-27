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

#include "graph/passes/folding_kernel/mul_kernel.h"

#include <memory>
#include <set>

#include "common/debug/log.h"
#include "common/math/math_util.h"
#include "common/types.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/common/bcast.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const std::set<DataType> mul_supported_type = {DT_INT32, DT_UINT32};

template <typename T>
Status IsOverflow(T const &a, T const &b, DataType &type) {
  switch (type) {
    case DT_INT32:
      return CheckInt32MulOverflow(a, b);
    case DT_UINT32:
      return CheckUint32MulOverflow(a, b);
    default:
      return FAILED;
  }
}

#define DEFINE_FUNC_WITH_STATUS_BY_TYPE(TYPE)                                             \
  std::function<TYPE(TYPE const &, TYPE const &, DataType &, Status &)> func_##TYPE = []( \
      TYPE const &a, TYPE const &b, DataType &type, Status &ret) -> TYPE {                \
    ret = IsOverflow(a, b, type);                                                         \
    if (ret != SUCCESS) {                                                                 \
      return static_cast<TYPE>(0);                                                        \
    }                                                                                     \
    return a * b;                                                                         \
  };

#define SET_BCAST_COMPUTE_CASE(DTYPE, TYPE)                           \
  case DTYPE:                                                         \
    ret = bcast.BCastComputeCheck(input, y_data_##TYPE, func_##TYPE); \
    break;

#define SET_OUTPUT(DTYPE, TYPE)                                                                                  \
  case DTYPE:                                                                                                    \
    (void)output_ptr->SetData(reinterpret_cast<uint8_t *>(y_data_##TYPE.data()), y_data_##TYPE.size() * length); \
    break;
DEFINE_FUNC_WITH_STATUS_BY_TYPE(int32_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(uint32_t)
}  // namespace

Status MulKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                          std::vector<GeTensorPtr> &v_output) {
  GELOGD("MulKernel in");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  Status ret = MulCheck(input);
  if (ret != SUCCESS) {
    return ret;
  }

  std::vector<int32_t> y_data_int32_t;
  std::vector<uint32_t> y_data_uint32_t;
  DataType data_type = input[0]->GetTensorDesc().GetDataType();
  BCast bcast;
  switch (data_type) {
    SET_BCAST_COMPUTE_CASE(DT_INT32, int32_t)
    SET_BCAST_COMPUTE_CASE(DT_UINT32, uint32_t)
    default:
      ret = NOT_CHANGED;
      break;
  }

  if (ret != SUCCESS) {
    GELOGW("BCastCompute fail, data_type: %s, ret: %s", TypeUtils::DataTypeToSerialString(data_type).c_str(),
           GET_ERRORNO_STR(ret).c_str());
    return NOT_CHANGED;
  }

  uint32_t length = 1;
  if (!TypeUtils::GetDataTypeLength(data_type, length)) {
    GELOGW("Can't GetDataTypeLength of data_type: %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed");
    return MEMALLOC_FAILED;
  }

  output_ptr->MutableTensorDesc().SetShape(GeShape(bcast.GetOutputShape()));
  // only return GRAPH_SUCCESS here
  switch (data_type) {
    SET_OUTPUT(DT_INT32, int32_t)
    SET_OUTPUT(DT_UINT32, uint32_t)
    default:
      break;
  }
  output_ptr->MutableTensorDesc().SetDataType(data_type);
  v_output.push_back(output_ptr);
  GELOGD("MulKernel success");

  return SUCCESS;
}

Status MulKernel::MulCheck(const std::vector<ConstGeTensorPtr> &input) {
  // check input number
  if (input.size() != static_cast<size_t>(MUL_INPUT_NUM)) {
    GELOGI("The number of input for Mul must be %u.", MUL_INPUT_NUM);
    return NOT_CHANGED;
  }

  ConstGeTensorPtr input_x1 = input.at(0);
  ConstGeTensorPtr input_x2 = input.at(1);
  GE_CHECK_NOTNULL(input_x1);
  GE_CHECK_NOTNULL(input_x2);
  // check whether there is data in Tensor
  if (input_x1->GetData().size() == 0 || input_x2->GetData().size() == 0) {
    GELOGI("Check data size fail. x1: %zu, x2: %zu", input_x1->GetData().size(), input_x2->GetData().size());
    return NOT_CHANGED;
  }

  // check whether the data types are the same
  DataType type = input_x1->GetTensorDesc().GetDataType();
  if (type != input_x2->GetTensorDesc().GetDataType()) {
    GELOGI("Data type of inputs for Mul not matched.");
    return NOT_CHANGED;
  }

  // check if input data type is supported
  if (mul_supported_type.find(type) == mul_supported_type.end()) {
    GELOGI("Mul does not support this Data type: %s", TypeUtils::DataTypeToSerialString(type).c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

REGISTER_KERNEL(MUL, MulKernel);
}  // namespace ge
