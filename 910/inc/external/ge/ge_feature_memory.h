/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef INC_EXTERNAL_GE_GE_FEATURE_MEMORY_H
#define INC_EXTERNAL_GE_GE_FEATURE_MEMORY_H

#include <memory>
#include <vector>
#include "ge_api_types.h"
#include "ge_error_codes.h"

namespace ge {
class GE_FUNC_VISIBILITY FeatureMemory {
 public:
  class Builder;

  class FeatureMemoryData;

  ~FeatureMemory();

  FeatureMemory &operator=(const FeatureMemory &) & = delete;
  FeatureMemory(const FeatureMemory &) = delete;

  ///
  /// @brief get memory type
  /// @return memory type
  ///
  MemoryType GetType() const;

  ///
  /// @brief get memory size
  /// @return memory size
  ///
  size_t GetSize() const;

  ///
  /// @brief is fixed feature memory
  /// @return true if feature memory is fixed, otherwise false
  ///
  bool IsFixed() const;

 private:
  FeatureMemory() = default;
  std::shared_ptr<FeatureMemoryData> data_{nullptr};
};
using FeatureMemoryPtr = std::shared_ptr<FeatureMemory>;
} // namespace ge
#endif  // INC_EXTERNAL_GE_GE_FEATURE_MEMORY_H
