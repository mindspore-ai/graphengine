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
#ifndef SLICE_DATASLICE_DATA_SLICE_HELPER_H_
#define SLICE_DATASLICE_DATA_SLICE_HELPER_H_

#include <vector>
#include "external/ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/axis_type_info.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
namespace ge {
class DataSliceHelper {
 public:
  static Status InferAxisSlice(OpDescPtr &op, const AxisTypeInfo &slice_info);
  static Status GetSliceInfo(OpDescPtr &op, std::vector<AxisTypeInfo> &axis_type_vec);
  static Status GetSliceInfo(const NodePtr &node, std::vector<AxisTypeInfo> &axis_type_vec);
  static Status InferDavinciAxisSlice(OpDescPtr &op, const AxisTypeInfo &slice_info);
  static Status GetDavinciSliceInfo(const NodePtr &node, std::vector<AxisTypeInfo> &axis_type_vec);
 private:
  // cut tensor: axis index: shape range
  using DataSliceType = std::vector<std::vector<std::vector<int64_t>>>;
  static Status SetInputSlice(OpDescPtr &op, const AxisTypeInfo &slice_info, DataSliceType &input_slice);
  static Status InferDavinciCommonOpSlice(OpDescPtr &op, const AxisTypeInfo &slice_info);
  static Status InferDavinciSpecialOpSlice(OpDescPtr &op, const AxisTypeInfo &slice_info,
    const InferAxisSliceFunc &node_slice_infer_ptr);
};
}
#endif // SLICE_DATASLICE_DATA_SLICE_HELPER_H_
