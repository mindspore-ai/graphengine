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
#ifndef AIR_RUNTIME_VAR_MANAGER_H
#define AIR_RUNTIME_VAR_MANAGER_H
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor_data.h"
#include "external/ge/ge_api_types.h"
namespace gert {
class RtVarManager {
 public:
  RtVarManager() = default;
  virtual ~RtVarManager() = default;
  virtual ge::Status GetVarShapeAndMemory(const std::string &id, StorageShape &shape,
                                          TensorData &memory) const = 0;
};
} // namespace gert
#endif  // AIR_RUNTIME_VAR_MANAGER_H
