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

#include "framework/common/op/ge_op_utils.h"

#include "common/fp16_t.h"
#include "graph/anchor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "mmpa/mmpa_api.h"

// Get the value of key from attr
#define AIPP_GET_ATTR_VALUE(KEY, ATTR_TYPE)                          \
  if (aipp_attr.GetItem(#KEY).GetValue<ATTR_TYPE>(KEY) != SUCCESS) { \
    GELOGI("Attr %s will take default value.", #KEY);                \
    break;                                                           \
  }

// Converting aippparams and attrdefmap
#define AIPP_CONVERT_FORMAT_EX(KEY, ORG_TYPE, SAVE_TYPE, ATTR_TYPE) \
  do {                                                              \
    SAVE_TYPE KEY = static_cast<SAVE_TYPE>(0);                      \
    AIPP_GET_ATTR_VALUE(KEY, ATTR_TYPE)                             \
    aipp_params.set_##KEY(static_cast<ORG_TYPE>(KEY));              \
  } while (false)

// Converting aippparams and attrdefmap
#define AIPP_CONVERT_FORMAT(KEY, KEY_TYPE, ATTR_TYPE, PROTO_TYPE) \
  AIPP_CONVERT_FORMAT_EX(KEY, PROTO_TYPE, KEY_TYPE, ATTR_TYPE)

#define AIPP_CONVERT_INT(KEY, PROTO_TYPE) AIPP_CONVERT_FORMAT(KEY, int64_t, GeAttrValue::INT, PROTO_TYPE)

#define AIPP_CONVERT_BOOL(KEY) AIPP_CONVERT_FORMAT(KEY, bool, GeAttrValue::BOOL, bool)

#define AIPP_CONVERT_FLOAT(KEY) AIPP_CONVERT_FORMAT(KEY, float32_t, GeAttrValue::FLOAT, float32_t)

// Transform aippparams (with repeated decoration) and attrdefmap
#define AIPP_CONVERT_LIST_FORMAT(KEY, KEY_TYPE, REQUIRED, ATTR_TYPE, PROTO_TYPE) \
  do {                                                               \
    if (REQUIRED) {                                                  \
      KEY_TYPE KEY;                                                  \
      AIPP_GET_ATTR_VALUE(KEY, ATTR_TYPE)                            \
      aipp_params.add_##KEY(static_cast<PROTO_TYPE>(KEY));           \
    }                                                                \
  } while (false)

#define AIPP_CONVERT_LIST_INT(KEY, REQUIRED, PROTO_TYPE) \
  AIPP_CONVERT_LIST_FORMAT(KEY, int64_t, REQUIRED, GeAttrValue::INT, PROTO_TYPE)

#define AIPP_CONVERT_LIST_BOOL(KEY, REQUIRED) \
  AIPP_CONVERT_LIST_FORMAT(KEY, bool, REQUIRED, GeAttrValue::BOOL, bool)

#define AIPP_CONVERT_LIST_FLOAT(KEY, REQUIRED) \
  AIPP_CONVERT_LIST_FORMAT(KEY, float32_t, REQUIRED, GeAttrValue::FLOAT, float32_t)

namespace ge {
// General constant
const uint32_t kSliceDataNum = 2U;

// Add Sub Mul
const uint32_t ADD_INPUT_NUM = 2U;
const uint32_t MUL_INPUT_NUM = 2U;

// Permute
const int32_t PERMUTE_ORDER_NUM = 4;
// Ssd PriroBox
const float64_t SSD_PRIORBOX_ASPECT_RATIO_VALUE = 1.0;

// Switch
const uint32_t SWITCH_INPUT_NUM = 2U;
const uint32_t SWITCH_OUTPUT_NUM = 2U;
const uint32_t SWITCH_FALSE_OUTPUT = 0U;
const uint32_t SWITCH_TRUE_OUTPUT = 1U;
const uint32_t SWITCH_DATA_INPUT = 0U;
const uint32_t SWITCH_PRED_INPUT = 1U;

// Merge
const int32_t MERGE_DATA_OUTPUT = 0;
const int32_t MERGE_INDEX_OUTPUT = 1;

// FunctionOp
const uint32_t IF_COND_INPUT = 0U;
const uint32_t FOR_START_INPUT = 0U;
const uint32_t FOR_LIMIT_INPUT = 1U;
const uint32_t FOR_DELTA_INPUT = 2U;
const uint32_t FOR_DATA_INPUT = 3U;

const int32_t NORMAL_TENSOR_SIZE = 4;

Status OpUtils::ConvertAippParams(const GeAttrValue::NamedAttrs &aipp_attr,
                                  domi::AippOpParams &aipp_params) {
  AIPP_CONVERT_FORMAT_EX(aipp_mode, domi::AippOpParams::AippMode, int64_t, GeAttrValue::INT);
  AIPP_CONVERT_INT(related_input_rank, uint32_t);

  if (aipp_params.aipp_mode() == domi::AippOpParams::dynamic) {
    AIPP_CONVERT_INT(max_src_image_size, uint32_t);
    AIPP_CONVERT_BOOL(support_rotation);
  } else {
    AIPP_CONVERT_FORMAT_EX(input_format, domi::AippOpParams::InputFormat, int64_t, GeAttrValue::INT);
    AIPP_CONVERT_BOOL(csc_switch);
    AIPP_CONVERT_BOOL(crop);
    AIPP_CONVERT_INT(load_start_pos_w, int32_t);
    AIPP_CONVERT_INT(load_start_pos_h, int32_t);
    AIPP_CONVERT_INT(crop_size_w, int32_t);
    AIPP_CONVERT_INT(crop_size_h, int32_t);
    AIPP_CONVERT_BOOL(resize);
    AIPP_CONVERT_INT(resize_output_w, int32_t);
    AIPP_CONVERT_INT(resize_output_h, int32_t);
    AIPP_CONVERT_BOOL(padding);
    AIPP_CONVERT_INT(left_padding_size, int32_t);
    AIPP_CONVERT_INT(right_padding_size, int32_t);
    AIPP_CONVERT_INT(top_padding_size, int32_t);
    AIPP_CONVERT_INT(bottom_padding_size, int32_t);
    AIPP_CONVERT_INT(src_image_size_w, int32_t);
    AIPP_CONVERT_INT(src_image_size_h, int32_t);
    AIPP_CONVERT_FLOAT(cpadding_value);
    AIPP_CONVERT_BOOL(rbuv_swap_switch);
    AIPP_CONVERT_BOOL(ax_swap_switch);
    AIPP_CONVERT_BOOL(single_line_mode);
    AIPP_CONVERT_INT(mean_chn_0, int32_t);
    AIPP_CONVERT_INT(mean_chn_1, int32_t);
    AIPP_CONVERT_INT(mean_chn_2, int32_t);
    AIPP_CONVERT_FLOAT(min_chn_0);
    AIPP_CONVERT_FLOAT(min_chn_1);
    AIPP_CONVERT_FLOAT(min_chn_2);
    AIPP_CONVERT_LIST_FLOAT(var_reci_chn_0, true);
    AIPP_CONVERT_LIST_FLOAT(var_reci_chn_1, true);
    AIPP_CONVERT_LIST_FLOAT(var_reci_chn_2, true);
    AIPP_CONVERT_LIST_FLOAT(var_reci_chn_3, true);

    const bool csc_switch = aipp_params.csc_switch();
    AIPP_CONVERT_LIST_INT(matrix_r0c0, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(matrix_r0c1, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(matrix_r0c2, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(matrix_r1c0, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(matrix_r1c1, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(matrix_r1c2, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(matrix_r2c0, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(matrix_r2c1, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(matrix_r2c2, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(output_bias_0, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(output_bias_1, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(output_bias_2, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(input_bias_0, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(input_bias_1, csc_switch, int32_t);
    AIPP_CONVERT_LIST_INT(input_bias_2, csc_switch, int32_t);
  }

  return SUCCESS;
}

template <typename T>
void OpUtils::SliceData(const std::vector<char_t *> &input, const int64_t chunk_size, std::vector<char_t *> &output,
                        const int64_t begin, const int64_t out_dim, const int64_t stride) {
  char_t *sliced_data = nullptr;
  // chunk_size * (begin + (out_dim-1)*stride) always less than chunk_size * dim_i, no need to check.
  for (size_t j = 0UL; j < input.size(); j++) {
    sliced_data = input[j] + (static_cast<int64_t>(sizeof(T)) * begin * chunk_size);
    for (int64_t i = 0; i < out_dim; i++) {
      output.push_back(sliced_data + (static_cast<int64_t>(sizeof(T)) * i * chunk_size * stride));
    }
  }
}

template <typename T>
Status OpUtils::SetDataByDataType(const size_t out_size, const std::vector<char_t *> &chunk_input,
                                  const std::vector<char_t *> &chunk_output, GeTensor *const output) {
  const unique_ptr<T[]> output_data(new (std::nothrow) T[out_size]());
  if (output_data == nullptr) {
    GELOGE(MEMALLOC_FAILED, "[Malloc][Data]New buf failed");
    REPORT_CALL_ERROR("E19999", "New buf failed");
    return INTERNAL_ERROR;
  }

  if (!chunk_input.empty()) {
    for (size_t j = 0UL; j < out_size; j++) {
      const T *const value = reinterpret_cast<T *>(chunk_input[j]);
      output_data[j] = value[0];
    }
  } else {
    for (size_t j = 0UL; j < out_size; j++) {
      const T *const value = reinterpret_cast<T *>(chunk_output[j]);
      output_data[j] = value[0];
    }
  }

  // output_data != nullptr and out_size > 0, SetData always return success, no need to check value
  (void)output->SetData(reinterpret_cast<const uint8_t *const>(output_data.get()), out_size * sizeof(T));
  return SUCCESS;
}

template <typename T>
Status OpUtils::SetOutputSliceDataByDataType(void *const data, const int64_t data_size,
                                             const std::vector<int64_t> &input_dims,
                                             const std::vector<int64_t> &begin,
                                             const std::vector<int64_t> &output_dims,
                                             GeTensor *const output,
                                             const std::vector<int64_t> &stride) {
  std::vector<char_t*> chunk_input;
  std::vector<char_t *> chunk_output;
  chunk_input.push_back(reinterpret_cast<char_t *>(data));
  int64_t chunk_size = data_size;
  const size_t dim_size = input_dims.size();
  for (size_t i = 0UL; i < dim_size; i++) {
    const int64_t begin_i = begin[i];
    const int64_t size_i = output_dims[i];
    const int64_t dim_i = input_dims[i];
    const int64_t stride_i = stride[i];
    if (dim_i == 0L) {
      GELOGE(PARAM_INVALID, "[Check][Param]Invalid, Dim_i of size tensor is 0");
      REPORT_INNER_ERROR("E19999", "Dim_i of size tensor is 0, invalid");
      return PARAM_INVALID;
    }
    chunk_size = chunk_size / dim_i;

    if ((i % kSliceDataNum) == 0UL) {
      SliceData<T>(chunk_input, chunk_size, chunk_output, begin_i, size_i, stride_i);
      chunk_input.clear();
    } else {
      SliceData<T>(chunk_output, chunk_size, chunk_input, begin_i, size_i, stride_i);
      chunk_output.clear();
    }
  }

  const size_t out_size = chunk_input.size() + chunk_output.size();
  GE_CHK_BOOL_RET_STATUS(out_size > 0UL, FAILED, "Out_size <= 0");
  const Status ret = SetDataByDataType<T>(out_size, chunk_input, chunk_output, output);
  return ret;
}

Status OpUtils::SetOutputSliceData(void *const data, const int64_t data_size, const int32_t data_type,
                                   const std::vector<int64_t> &input_dims, const std::vector<int64_t> &begin,
                                   const std::vector<int64_t> &output_dims, GeTensor *const output,
                                   const std::vector<int64_t> &stride) {
  if ((data == nullptr) || (output == nullptr)) {
    GELOGE(PARAM_INVALID, "[Check][Param]Input param is nullptr");
    REPORT_INNER_ERROR("E19999", "Input param is nullptr");
    return PARAM_INVALID;
  }

  Status ret;
  switch (data_type) {
    case DT_INT32:
      ret = SetOutputSliceDataByDataType<int32_t>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_FLOAT:
      ret = SetOutputSliceDataByDataType<float>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_DOUBLE:
      ret = SetOutputSliceDataByDataType<double>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_FLOAT16:
      ret = SetOutputSliceDataByDataType<fp16_t>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_UINT8:
      ret = SetOutputSliceDataByDataType<uint8_t>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_INT8:
      ret = SetOutputSliceDataByDataType<int8_t>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_UINT16:
      ret = SetOutputSliceDataByDataType<uint16_t>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_INT16:
      ret = SetOutputSliceDataByDataType<int16_t>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_UINT32:
      ret = SetOutputSliceDataByDataType<uint32_t>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_UINT64:
      ret = SetOutputSliceDataByDataType<uint64_t>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    case DT_INT64:
      ret = SetOutputSliceDataByDataType<int64_t>(data, data_size, input_dims, begin, output_dims, output, stride);
      break;
    default:
      GELOGW("Unsupported data type: %s", TypeUtils::DataTypeToSerialString(static_cast<DataType>(data_type)).c_str());
      ret = PARAM_INVALID;
      break;
  }
  return ret;
}

// The caller guarantees that the input sensor is constant
Status OpUtils::GetShapeDataFromConstTensor(const ConstGeTensorPtr &tensor, const DataType type,
                                            std::vector<int64_t> &dims) {
  if (tensor == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param]Input tensor is nullptr");
    REPORT_INNER_ERROR("E19999", "Input tensor is nullptr");
    return PARAM_INVALID;
  }

  // If the tensor data is a vector, the shape dimension must be 1
  if (tensor->GetTensorDesc().GetShape().GetDims().size() > 1UL) {
    GELOGE(PARAM_INVALID, "[Check][Param]The dimension of the input tensor shape cannot be more than 1, it is %zu",
           tensor->GetTensorDesc().GetShape().GetDims().size());
    REPORT_CALL_ERROR("E19999", "The dimension of the input tensor shape %zu invalid, more than 1",
                      tensor->GetTensorDesc().GetShape().GetDims().size());
    return PARAM_INVALID;
  }

  if (type == DT_INT32) {
    const int32_t *const shape_data = reinterpret_cast<const int32_t *>(tensor->GetData().GetData());
    GE_CHECK_NOTNULL(shape_data);
    const size_t dims_num = tensor->GetData().size() / sizeof(int32_t);
    for (size_t i = 0UL; i < dims_num; i++) {
      dims.push_back(static_cast<int64_t>(shape_data[i]));
    }
  } else if (type == DT_INT64) {
    const int64_t *const shape_data = reinterpret_cast<const int64_t *>(tensor->GetData().GetData());
    GE_CHECK_NOTNULL(shape_data);
    const size_t dims_num = tensor->GetData().size() / sizeof(int64_t);
    for (size_t i = 0UL; i < dims_num; i++) {
      dims.push_back(shape_data[i]);
    }
  } else {
    GELOGE(PARAM_INVALID, "[Check][DataType]Invalid, type only can be DT_INT32 or DT_INT64, type is %s",
           TypeUtils::DataTypeToSerialString(type).c_str());
    REPORT_INNER_ERROR("E19999", "Data type %s check invalid, only can be DT_INT32 or DT_INT64",
                       TypeUtils::DataTypeToSerialString(type).c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}
}  // namespace ge
