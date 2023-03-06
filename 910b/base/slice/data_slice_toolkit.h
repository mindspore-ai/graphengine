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
#ifndef SLICE_DATASLICE_DATA_SLICE_TOOLKIT_H_
#define SLICE_DATASLICE_DATA_SLICE_TOOLKIT_H_

#include <string>
#include "external/graph/ge_error_codes.h"
#include "external/ge/ge_api_types.h"
#include "external/graph/operator.h"
namespace ge {
template <typename T>
std::string DataSliceGetName(const T& op) {
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetName(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    return "None";
  }
  return op_ascend_name.GetString();
}

template <typename T>
std::string DataSliceGetOpType(const T& op) {
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetOpType(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    return "None";
  }
  return op_ascend_name.GetString();
}
}  // namespace ge
#endif // SLICE_DATASLICE_DATA_SLICE_TOOLKIT_H_
