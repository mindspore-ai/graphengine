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

#include "graph/passes/folding_kernel/reformat_kernel.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/ge/ge_util.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kReFormatInputSize = 1;
const size_t kReformatFirstInput = 0;
const size_t kReformatFirstOutput = 0;
}  // namespace

Status ReFormatKernel::ValidateInput(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) {
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  if (op_desc_ptr->GetInputsSize() != kReFormatInputSize) {
    GELOGW("trans_op has more than 1 input_size.");
    return PARAM_INVALID;
  }
  if (input.empty()) {
    GELOGE(PARAM_INVALID, "Input tensor vector is empty");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status ReFormatKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                               std::vector<GeTensorPtr> &v_output) {
  GELOGD("ReFormatKernel begin.");
  Status status = ValidateInput(op_desc_ptr, input);
  if (status != SUCCESS) {
    return status;
  }

  ConstGeTensorPtr const_weight_ptr = input[kReformatFirstInput];
  if (const_weight_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, Input_0 is nullptr.");
    return NOT_CHANGED;
  }

  GeTensorDesc op_desc = op_desc_ptr->GetOutputDesc(kReformatFirstOutput);
  GeTensorDesc op_desc_in = op_desc_ptr->GetInputDesc(kReformatFirstInput);
  auto src_shape = op_desc_in.GetShape().GetDims();
  auto src_dtype = op_desc_in.GetDataType();
  auto dst_shape = op_desc.GetShape().GetDims();
  auto dst_dtype = op_desc.GetDataType();
  if (src_dtype != dst_dtype || src_shape != dst_shape) {
    GELOGW("Check params failed. src data type %s and shape %s should be equal to dst data type %s and shape %s",
           TypeUtils::DataTypeToSerialString(src_dtype).c_str(), formats::ShapeToString(src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(dst_dtype).c_str(), formats::ShapeToString(dst_shape).c_str());
    return NOT_CHANGED;
  }
  if (!KernelUtils::CheckSizeForTransOp(const_weight_ptr, op_desc_ptr)) {
    GELOGE(FAILED, "CheckSize failed, input size(shape %s) is not equal to weight size(shape %s)",
           formats::ShapeToString(src_shape).c_str(),
           formats::ShapeToString(const_weight_ptr->GetTensorDesc().GetShape()).c_str());
    return NOT_CHANGED;
  }
  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(kReformatFirstOutput));
  if (output_ptr == nullptr) {
    GELOGE(INTERNAL_ERROR, "Create shared ptr for GeTensor failed");
    return NOT_CHANGED;
  }
  GE_IF_BOOL_EXEC(output_ptr->SetData(input.at(0)->GetData()) != GRAPH_SUCCESS,
                  GELOGE(INTERNAL_ERROR, "set data failed");
                  return NOT_CHANGED);
  v_output.emplace_back(output_ptr);
  GELOGD("ReFormatKernel success.");
  return SUCCESS;
}

REGISTER_KERNEL(REFORMAT, ReFormatKernel);
}  // namespace ge
