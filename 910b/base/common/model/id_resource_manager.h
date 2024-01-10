/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef AIR_CXX_ID_RESOURCE_MANAGER_H
#define AIR_CXX_ID_RESOURCE_MANAGER_H

#include <vector>
#include <algorithm>

#include "framework/common/ge_types.h"
#include "graph/ge_context.h"
#include "external/ge/ge_api_error_codes.h"

namespace ge {
constexpr int32_t kMaxIdResourceSize = 1024;
class IdResourceManager {
 public:
  static IdResourceManager &GetInstance() {
    static IdResourceManager instance;
    return instance;
  }

  Status Allocate(uint32_t &id);
  Status DeAllocate(const uint32_t &id);

 protected:
  std::vector<bool> resources_ = std::vector<bool>(kMaxIdResourceSize, false);
};

class ModelIdResourceManager : public IdResourceManager {
 public:
  static ModelIdResourceManager &GetInstance() {
    static ModelIdResourceManager instance;
    instance.resources_[0] = true;
    return instance;
  }
  Status GenerateModelId(uint32_t &id);
};
} // namespace ge
#endif  // AIR_CXX_ID_RESOURCE_MANAGER_H
