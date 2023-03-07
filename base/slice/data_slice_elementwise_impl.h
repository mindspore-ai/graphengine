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
#ifndef SLICE_DATASLICE_DATA_SLICE_ELEMENTWISE_IMPL_H_
#define SLICE_DATASLICE_DATA_SLICE_ELEMENTWISE_IMPL_H_

#include "slice/data_slice_infer_base.h"
namespace ge {
// Elementwise
class DataSliceElementwiseImpl : public DataSliceInferBase {
 public:
  Status InferAxisSlice(Operator &op, const AxisTypeInfo &slice_info,
    const DataSliceType &out_data_slice, DataSliceType &in_data_slice) override;
};
}
#endif // SLICE_DATASLICE_DATA_SLICE_ELEMENTWISE_IMPL_H_
