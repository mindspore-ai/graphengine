/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "slice/data_slice_elementwise_impl.h"
#include "slice/data_slice_toolkit.h"
#include "slice/data_slice_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
static AxisInferRegister registerElementWise(ge::AxisType::ELEMENTWISE,
  [] (void) noexcept ->DataSliceInferBase* {return new (std::nothrow) DataSliceElementwiseImpl();});

static bool CheckOutCutInfoVaild(size_t index, OpDescPtr op_desc, const CutInfo &one_output_cutinfo,
    const std::vector<int64_t> &one_out_data_slice)
{
  if (static_cast<int64_t>(index) == one_output_cutinfo.second[0]) {
    if (one_out_data_slice.empty()) {
      GELOGE(FAILED, "The op[%s] output data slice cannot be empty.", op_desc->GetName().c_str());
      return false;
    }
    auto output_desc = op_desc->MutableOutputDesc(one_output_cutinfo.first);
    if (output_desc == nullptr) {
      GELOGE(FAILED, "The op[%s] output description is nullptr.", op_desc->GetName().c_str());
      return false;
    }
    int64_t output_dim = output_desc->MutableShape().GetDim(index);
    int64_t output_last_range = one_out_data_slice.back();
    if (output_last_range > output_dim) {
      GELOGE(FAILED, "The op[%s] output split range[%ld] larger than output dim[%ld].",
             op_desc->GetName().c_str(), output_last_range, output_dim);
      return false;
    }
  } else if (!one_out_data_slice.empty()) {
    GELOGE(FAILED, "The op[%s] output split range for non-split axis[%zu] is not empty.",
           op_desc->GetName().c_str(), index);
    return false;
  }
  return true;
}

static bool CheckOutDataSlice(OpDescPtr op_desc, const std::vector<CutInfo> &output_cutinfo,
    const DataSliceType &out_data_slice)
{
  if (output_cutinfo.size() != out_data_slice.size()) {
    GELOGE(FAILED, "The op[%s] output data slice info size[%u] is not equal to cut info size[%u].",
           op_desc->GetName().c_str(), output_cutinfo.size(), out_data_slice.size());
    return false;
  }
  for (size_t i = 0; i < out_data_slice.size(); ++i) {
    for (size_t j = 0; j < out_data_slice[i].size(); ++j) {
      if (!CheckOutCutInfoVaild(j, op_desc, output_cutinfo[i], out_data_slice[i][j])) {
        return false;
      }
    }
  }
  return true;
}

static std::vector<std::vector<int64_t>> GetInputSplitRanges(GeTensorDescPtr input_desc,
    const CutInfo &one_input_cutinfo, const std::vector<int64_t> &one_out_data_slice, bool &is_invalid_info)
{
  std::vector<std::vector<int64_t>> split_ranges;
  for (size_t j = 0; j < input_desc->MutableShape().GetDimNum(); ++j) {
    if (static_cast<int64_t>(j) == one_input_cutinfo.second[0]) {
      int64_t input_dim = input_desc->MutableShape().GetDim(j);
      if (one_out_data_slice.back() > input_dim) {
        GELOGE(FAILED, "The input split range[%ld] larger than input dim[%ld].", one_out_data_slice.back(), input_dim);
        is_invalid_info = true;
        break;
      }
    }
    split_ranges.push_back(one_out_data_slice);
  }
  return split_ranges;
}

// Elementwise
Status DataSliceElementwiseImpl::InferAxisSlice(Operator &op, const AxisTypeInfo &slice_info,
    const DataSliceType &out_data_slice, DataSliceType &in_data_slice)
{
  if (!in_data_slice.empty()) {
    GELOGE(FAILED, "The op[%s] input data slice is not empty.", DataSliceGetName(op).c_str());
    return FAILED;
  }
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "The op[%s] description is nullptr.", DataSliceGetName(op).c_str());
    return FAILED;
  }
  const std::vector<CutInfo> input_cutinfo = slice_info.GetRelateInputs();
  const std::vector<CutInfo> output_cutinfo = slice_info.GetRelateOutputs();
  if (input_cutinfo.empty() || output_cutinfo.empty()) {
    GELOGE(FAILED, "Get op[%s] input or output cut info empty.", DataSliceGetName(op).c_str());
    return FAILED;
  }
  // 遍历out_data_slice进行校验
  if (!CheckOutDataSlice(op_desc, output_cutinfo, out_data_slice)) {
    return FAILED;
  }

  // 遍历得到in_data_slice
  for (size_t i = 0; i < input_cutinfo.size(); ++i) {
    auto input_desc = op_desc->MutableInputDesc(input_cutinfo[i].first);
    if (input_desc == nullptr) {
      GELOGE(FAILED, "The op[%s] input description is nullptr.", DataSliceGetName(op).c_str());
      return FAILED;
    }
    bool is_invalid_info = false;
    std::vector<std::vector<int64_t>> split_ranges = GetInputSplitRanges(input_desc, input_cutinfo[i],
        out_data_slice[0][output_cutinfo[0].second[0]], is_invalid_info);
    if (is_invalid_info) {
      GELOGE(FAILED, "The op[%s] input split range larger than input dim.", op_desc->GetName().c_str());
      return FAILED;
    }
    in_data_slice.push_back(split_ranges);
  }
  GELOGI("Elementwise infer success, op:%s, type:%s, axis type:%d.",
         DataSliceGetName(op).c_str(), DataSliceGetOpType(op).c_str(), static_cast<int8_t>(slice_info.GetAxisType()));
  return SUCCESS;
}
}
