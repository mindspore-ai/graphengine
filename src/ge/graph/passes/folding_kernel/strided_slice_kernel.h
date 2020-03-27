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

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_STRIDED_SLICE_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_STRIDED_SLICE_KERNEL_H_

#include <vector>

#include "inc/kernel.h"

namespace ge {
struct Attr {
  int64_t begin_mask;
  int64_t end_mask;
  int64_t ellipsis_mask;
  int64_t new_axis_mask;
  int64_t data_type;
  int64_t shrink_axis_mask;
};

class StridedSliceKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr attr, const std::vector<ConstGeTensorPtr> &input,
                 vector<GeTensorPtr> &v_output) override;

 private:
  Status CheckAndGetAttr(const OpDescPtr &attr, const std::vector<ConstGeTensorPtr> &input, Attr &args);
  Status CheckWeight(const ConstGeTensorPtr &weight0, const ConstGeTensorPtr &weight1, const ConstGeTensorPtr &weight2,
                     const ConstGeTensorPtr &weight3) const;
  void MaskCal(const bool &begin_mask_flag, const bool &end_mask_flag, const bool &shrink_mask_flag, int32_t &begin_i,
               int32_t &end_i, int32_t &dim_i) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_STRIDED_SLICE_KERNEL_H_
