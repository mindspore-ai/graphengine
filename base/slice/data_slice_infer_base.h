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
#ifndef SLICE_DATASLICE_DATA_SLICE_INFER_BASE_H_
#define SLICE_DATASLICE_DATA_SLICE_INFER_BASE_H_

#include <vector>
#include "external/graph/ge_error_codes.h"
#include "external/graph/operator.h"
#include "external/graph/types.h"
#include "external/ge/ge_api_types.h"
#include "graph/axis_type_info.h"
namespace ge {
using DataSliceType = std::vector<std::vector<std::vector<int64_t>>>;
class DataSliceInferBase {
 public:
  DataSliceInferBase() = default;
  virtual ~DataSliceInferBase() = default;
  virtual Status InferAxisSlice(ge::Operator &op, const AxisTypeInfo &slice_info,
    const DataSliceType &out_data_slice, DataSliceType &in_data_slice) = 0;
};
}
#endif // SLICE_DATASLICE_DATA_SLICE_INFER_BASE_H_
