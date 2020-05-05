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

#include "graph/passes/folding_kernel/rsqrt_kernel.h"

#include <cfloat>

#include <memory>

#include "common/debug/ge_log.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kRsqrtInputSize = 1;
const size_t kRsqrtInputIndex0 = 0;
}  // namespace
Status RsqrtKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                            std::vector<GeTensorPtr> &v_output) {
  GELOGI("RsqrtKernel in.");
  GE_CHECK_NOTNULL(op_desc_ptr);
  // check input size
  if (input.size() != kRsqrtInputSize) {
    GELOGW("The number of input for rsqrt must be %zu.", kRsqrtInputSize);
    return PARAM_INVALID;
  }

  ConstGeTensorPtr input_ = input.at(kRsqrtInputIndex0);
  GE_CHECK_NOTNULL(input_);
  const GeShape &x_shape = input_->GetTensorDesc().GetShape();

  size_t data_size = input_->GetData().size();
  size_t data_count = data_size / sizeof(float);

  // check whether input is zero
  for (size_t i = 0; i < data_count; i++) {
    if (fabs(*(reinterpret_cast<const float *>(input_->GetData().data()) + i)) < FLT_EPSILON) {
      GELOGW("input must be not equal 0.");
      return NOT_CHANGED;
    }
  }
  if (data_count > 0) {
    unique_ptr<float[]> buf(new (std::nothrow) float[data_count]());
    if (buf == nullptr) {
      GELOGE(MEMALLOC_FAILED, "new buf failed");
      return NOT_CHANGED;
    }

    for (size_t i = 0; i < data_count; i++) {
      float denominator = sqrt(*(reinterpret_cast<const float *>(input_->GetData().data()) + i));
      if (fabs(denominator) < FLT_EPSILON) {
        GELOGW("input must be not equal 0.");
        return NOT_CHANGED;
      }
      buf[i] = 1 / denominator;
    }

    // Index 0 can always gets a GeTensorDesc object from any OpDescPtr.
    auto output_tensor_desc = op_desc_ptr->GetOutputDesc(0);
    GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
    if (output_ptr == nullptr) {
      GELOGE(MEMALLOC_FAILED, "MakeShared GeTensor failed, node name %s.", op_desc_ptr->GetName().c_str());
      return NOT_CHANGED;
    }

    output_ptr->MutableTensorDesc().SetDataType(DT_FLOAT);
    GE_IF_BOOL_EXEC(output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()), data_size) != GRAPH_SUCCESS,
                    GELOGE(INTERNAL_ERROR, "set data failed");
                    return NOT_CHANGED);
    output_ptr->MutableTensorDesc().SetShape(x_shape);
    v_output.push_back(output_ptr);
  }
  GELOGI("RsqrtKernel success.");
  return SUCCESS;
}

REGISTER_KERNEL(RSQRT, RsqrtKernel);
}  // namespace ge
