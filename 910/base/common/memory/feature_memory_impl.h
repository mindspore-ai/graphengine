/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

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
