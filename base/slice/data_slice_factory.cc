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
#include "slice/data_slice_factory.h"
#include "graph/operator_factory_impl.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
std::shared_ptr<DataSliceInferBase> DataSliceFactory::GetClassByAxisType(ge::AxisType axis_type) {
  DataSliceInferBase *instance = nullptr;
  auto it = class_map_.find(axis_type);
  if (it != class_map_.end()) {
    instance = it->second();
  }
  if (instance != nullptr) {
    return std::shared_ptr<DataSliceInferBase>(instance);
  }
  return nullptr;
}

void DataSliceFactory::RegistClass(ge::AxisType axis_type, std::function<DataSliceInferBase*(void)> infer_util_class) {
  auto ret = class_map_.insert(std::pair<ge::AxisType,
      std::function<DataSliceInferBase*(void)>>(axis_type, infer_util_class));
  if (!ret.second) {
    GELOGW("[DataSlice][Status] Axis type %d has already registed.", static_cast<int8_t>(axis_type));
  }
}
}
