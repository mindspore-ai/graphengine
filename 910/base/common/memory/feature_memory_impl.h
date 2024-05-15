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

#ifndef GE_COMMON_MEMORY__FEATURE_MEMORY_IMPL_H
#define GE_COMMON_MEMORY__FEATURE_MEMORY_IMPL_H

#include "external/ge/ge_feature_memory.h"

namespace ge {
class FeatureMemory::FeatureMemoryData {
 public:
  // 使用结构体,方便后续扩展is_refresh等
  struct MemoryAttr {
    bool is_fixed;
  };

  FeatureMemoryData() = default;
  ~FeatureMemoryData() = default;

  /// @brief get memory type
  /// @return memory type
  MemoryType GetType() const;

  /// @brief get memory size
  /// @return memory size
  size_t GetSize() const;

  /// @brief is fixed feature memory
  /// @return true if feature memory is fixed, otherwise false
  bool IsFixed() const;

 private:
  friend class FeatureMemory::Builder;
  void SetTypeSize(const MemoryType type, const size_t size, const FeatureMemoryData::MemoryAttr &mem_attr);

  MemoryType type_;
  size_t size_;
  bool is_fixed_;
};

class FeatureMemory::Builder {
 public:
  Builder() = default;
  ~Builder() = default;
  static FeatureMemoryPtr Build(const MemoryType type, const size_t size,
                                const FeatureMemoryData::MemoryAttr &mem_attr);
};
} // namespace ge
#endif  // GE_COMMON_MEMORY__FEATURE_MEMORY_IMPL_H
