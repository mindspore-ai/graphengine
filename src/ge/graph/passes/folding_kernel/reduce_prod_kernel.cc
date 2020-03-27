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

#include "graph/passes/folding_kernel/reduce_prod_kernel.h"

#include <memory>
#include <set>

#include "common/math/math_util.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kReduceProdDataIndex = 0;
const size_t kReduceProdAxisIndex = 1;
const size_t kReduceProdMaxAxisRank = 1;
const size_t kReduceProdInputOnlyData = 1;
const size_t kReduceProdInputSize = 2;
const std::set<DataType> kReduceProdSupportedType = {DT_INT32};
}  // namespace

Status ReduceProdKernel::ReduceProdCheck(const ge::OpDescPtr &op_desc_ptr,
                                         const std::vector<ge::ConstGeTensorPtr> &input) const {
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "input opdesc is nullptr.");
    return PARAM_INVALID;
  }
  if (input.size() != kReduceProdInputSize) {
    if (input.size() == kReduceProdInputOnlyData) {
      // Input only data, which means calculate product for all elements in data_tensor.
      GELOGI("ReduceProd node input size is 1, which does not have param axis, node name %s",
             op_desc_ptr->GetName().c_str());
      return NOT_CHANGED;
    }
    GELOGE(PARAM_INVALID, "Unexpected ReduceProd node, node input size: %zu, node name: %s", input.size(),
           op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  ConstGeTensorPtr data_tensor = input.at(kReduceProdDataIndex);
  ConstGeTensorPtr axis_tensor = input.at(kReduceProdAxisIndex);
  GE_CHECK_NOTNULL(data_tensor);
  GE_CHECK_NOTNULL(axis_tensor);
  if (axis_tensor->GetTensorDesc().GetShape().GetDimNum() > kReduceProdMaxAxisRank) {
    GELOGE(PARAM_INVALID, "Axis must be at most rank 1, node node: %s", op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  if (data_tensor->GetData().size() == 0 || axis_tensor->GetData().size() == 0) {
    GELOGE(PARAM_INVALID, "ReduceProdKernel data size of inputs is 0, node node: %s", op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  DataType data_type = data_tensor->GetTensorDesc().GetDataType();
  if (kReduceProdSupportedType.find(data_type) == kReduceProdSupportedType.end()) {
    GELOGE(PARAM_INVALID, "ReduceProdKernel data type %s not support, node name: %s",
           TypeUtils::DataTypeToSerialString(data_type).c_str(), op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status ReduceProdKernel::AxisCal(const std::vector<ge::ConstGeTensorPtr> &input) {
  ConstGeTensorPtr data_tensor = input.at(kReduceProdDataIndex);
  ConstGeTensorPtr axis_tensor = input.at(kReduceProdAxisIndex);
  // support: compute for the first element of axis.
  vector<int64_t> data_dims = data_tensor->GetTensorDesc().GetShape().GetDims();
  size_t data_dim_size = data_dims.size();
  int32_t *axis = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(axis_tensor->GetData().GetData()));
  GE_CHECK_NOTNULL(axis);
  if (static_cast<size_t>(*axis) >= data_dim_size) {
    GELOGE(PARAM_INVALID, "axis is out of rank of data_dims, axis is %d.", *axis);
    return PARAM_INVALID;
  }
  axis_dim_ = data_dims[static_cast<size_t>(*axis)];
  head_dim_ = 1;
  end_dim_ = 1;
  bool axis_appear = false;
  for (size_t i = 0; i < data_dim_size; i++) {
    if (i == static_cast<size_t>(*axis)) {
      axis_appear = true;
      continue;
    }
    // data_dims is the vector of dims, element in data_dims isn't negative.
    if (axis_appear) {
      if (data_dims[i] != 0 && end_dim_ > (INT64_MAX / data_dims[i])) {
        GELOGE(INTERNAL_ERROR, "Product is overflow. multiplier 1: %ld. multiplier 2: %ld.", end_dim_, data_dims[i]);
        return INTERNAL_ERROR;
      }
      end_dim_ *= data_dims[i];
    } else {
      if (data_dims[i] != 0 && head_dim_ > (INT64_MAX / data_dims[i])) {
        GELOGE(INTERNAL_ERROR, "Product is overflow. multiplier 1: %ld. multiplier 2: %ld.", head_dim_, data_dims[i]);
        return INTERNAL_ERROR;
      }
      head_dim_ *= data_dims[i];
    }
  }
  return SUCCESS;
}

Status ReduceProdKernel::DataCal(const std::vector<ge::ConstGeTensorPtr> &input, ge::GeTensorPtr output_ptr) {
  ConstGeTensorPtr data_tensor = input.at(kReduceProdDataIndex);
  DataType data_dtype = data_tensor->GetTensorDesc().GetDataType();
  if (data_dtype == DT_INT32) {
    int32_t *input_data = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(data_tensor->GetData().GetData()));
    GE_CHECK_NOTNULL(input_data);
    size_t data_num = data_tensor->GetData().size() / sizeof(int32_t);
    unique_ptr<int32_t[]> buf(new (std::nothrow) int32_t[data_num]());
    if (buf == nullptr) {
      GELOGE(MEMALLOC_FAILED, "new buf failed");
      return INTERNAL_ERROR;
    }

    int32_t tmp_x = 1;
    for (int64_t i = 0; i < head_dim_; ++i) {
      for (int64_t j = 0; j < end_dim_; ++j) {
        // all index for input_data is less than size of input_data
        tmp_x = input_data[static_cast<size_t>(i * end_dim_ * axis_dim_ + j)];
        for (int64_t k = 1; k < axis_dim_; ++k) {
          int32_t tmp_y = input_data[static_cast<size_t>(i * end_dim_ * axis_dim_ + j + k * end_dim_)];
          if (CheckInt32MulOverflow(tmp_x, tmp_y) != SUCCESS) {
            GELOGE(INTERNAL_ERROR, "Product is overflow. multiplier 1: %d. multiplier 2: %d.", tmp_x, tmp_y);
            return INTERNAL_ERROR;
          }
          tmp_x *= tmp_y;
        }
        buf[static_cast<size_t>(i * end_dim_ + j)] = tmp_x;
      }
    }

    GE_IF_BOOL_EXEC(output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()),
                                        static_cast<size_t>(head_dim_ * end_dim_ * sizeof(int32_t))) != GRAPH_SUCCESS,
                    GELOGE(INTERNAL_ERROR, "set data failed");
                    return INTERNAL_ERROR);
    output_ptr->MutableTensorDesc().SetDataType(data_dtype);
  }
  return SUCCESS;
}

void ReduceProdKernel::ShapeCal(const ge::OpDescPtr &op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                                ge::GeTensorPtr output_ptr) {
  ConstGeTensorPtr data_tensor = input.at(kReduceProdDataIndex);
  ConstGeTensorPtr axis_tensor = input.at(kReduceProdAxisIndex);
  auto axis_data = axis_tensor->GetData().GetData();
  if (axis_data == nullptr) {
    GELOGE(FAILED, "Data of axis tensor is nullptr.");
    return;
  }
  vector<int64_t> data_dims = data_tensor->GetTensorDesc().GetShape().GetDims();
  int32_t data_dim_size = static_cast<int32_t>(data_dims.size());
  int32_t axis = *(const_cast<int32_t *>(reinterpret_cast<const int32_t *>(axis_data)));
  bool keep_dims = false;
  if (!AttrUtils::GetBool(op_desc_ptr, "keep_dims", keep_dims)) {
    GELOGI("Get the attr keep_dims was failed.");
  }

  if (keep_dims) {
    for (int32_t i = 0; i < data_dim_size; i++) {
      if (i == axis) {
        data_dims[i] = 1;
      }
    }
  } else {
    vector<int64_t> tmp_dims;
    for (int32_t i = 0; i < data_dim_size; i++) {
      if (i != axis) {
        tmp_dims.push_back(data_dims[i]);
      }
    }
    data_dims.clear();
    data_dims = tmp_dims;
  }
  output_ptr->MutableTensorDesc().SetShape(GeShape(data_dims));
}

Status ReduceProdKernel::ComputeNoAxis(const ge::OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                                       ge::GeTensorPtr output_ptr) {
  ConstGeTensorPtr data_tensor = input.at(kReduceProdDataIndex);
  GE_CHECK_NOTNULL(data_tensor);
  if (data_tensor->GetData().size() == 0) {
    GELOGE(PARAM_INVALID, "ReduceProdKernel data size of inputs is 0, node node: %s", op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  DataType data_type = data_tensor->GetTensorDesc().GetDataType();
  if (kReduceProdSupportedType.find(data_type) == kReduceProdSupportedType.end()) {
    GELOGE(PARAM_INVALID, "ReduceProdKernel data type %s not support, node name: %s",
           TypeUtils::DataTypeToSerialString(data_type).c_str(), op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }

  if (data_type == DT_INT32) {
    int32_t *input_data = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(data_tensor->GetData().GetData()));
    GE_CHECK_NOTNULL(input_data);
    size_t data_num = data_tensor->GetData().size() / sizeof(int32_t);
    unique_ptr<int32_t[]> buf(new (std::nothrow) int32_t[data_num]());
    if (buf == nullptr) {
      GELOGE(MEMALLOC_FAILED, "new buf failed");
      return INTERNAL_ERROR;
    }

    int32_t tmp_x = input_data[0];
    int32_t tmp_y = 1;
    for (size_t k = 1; k < data_num; ++k) {
      tmp_y = input_data[k];
      if (CheckInt32MulOverflow(tmp_x, tmp_y) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Product is overflow. multiplier 1: %d. multiplier 2: %d.", tmp_x, tmp_y);
        return INTERNAL_ERROR;
      }
      tmp_x *= tmp_y;
    }
    buf[0] = tmp_x;
    GE_IF_BOOL_EXEC(output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()), sizeof(int32_t)) != GRAPH_SUCCESS,
                    GELOGE(INTERNAL_ERROR, "set data failed");
                    return INTERNAL_ERROR);
    output_ptr->MutableTensorDesc().SetDataType(data_type);
    output_ptr->MutableTensorDesc().SetShape(GeShape());
  }
  return SUCCESS;
}

Status ReduceProdKernel::Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                                 std::vector<ge::GeTensorPtr> &v_output) {
  GELOGI("ReduceProdKernel in.");
  Status ret = ReduceProdCheck(op_desc_ptr, input);
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(PARAM_INVALID, "ReduceProdKernel input is invalid, failed to fold node.");
    return NOT_CHANGED;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>();
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "make_shared ge::GeTensor failed, node name %s.", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  if (ret == NOT_CHANGED) {
    // compute output tensor when no param axis
    ret = ComputeNoAxis(op_desc_ptr, input, output_ptr);
    if (ret != SUCCESS) {
      return NOT_CHANGED;
    }
  } else {
    // calculate axis to reduce
    ret = AxisCal(input);
    if (ret != SUCCESS) {
      return NOT_CHANGED;
    }
    // calculate data and data type
    ret = DataCal(input, output_ptr);
    if (ret != SUCCESS) {
      return NOT_CHANGED;
    }
    // calculate shape
    ShapeCal(op_desc_ptr, input, output_ptr);
  }

  // print output tensor information, and will be deleted
  GELOGD("ReduceProd op %s output tensor data size is %zu", op_desc_ptr->GetName().c_str(),
         output_ptr->GetData().size());
  vector<int64_t> data_dims = output_ptr->GetTensorDesc().GetShape().GetDims();
  GELOGD("ReduceProd op %s output tensor dim size is %zu", op_desc_ptr->GetName().c_str(), data_dims.size());

  v_output.emplace_back(output_ptr);
  GELOGI("ReduceProdKernel success.");
  return SUCCESS;
}
REGISTER_KERNEL(REDUCEPROD, ReduceProdKernel);
}  // namespace ge
