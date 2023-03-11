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
#ifndef AIR_MEM_ALLOCATOR_H
#define AIR_MEM_ALLOCATOR_H

#include "exe_graph/runtime/tensor_data.h"
#include "block.h"
#include "exe_graph/runtime/allocator.h"
#include "common/ge_visibility.h"
#include "ge/ge_api_error_codes.h"

namespace gert {
namespace memory {
struct MemAllocator {
  virtual Block *Malloc(size_t size, bool independence = false) = 0;
  virtual ~MemAllocator() = default;
};

struct MemSynchronizer {
  MemSynchronizer() = default;
  virtual ~MemSynchronizer() = default;
  // Wait until the memory is actually freed after task completed
  virtual void Synchronize() const = 0;
};
}

class VISIBILITY_EXPORT Allocators {
 public:
  memory::MemAllocator *GetAllocator(const TensorPlacement &placement, const size_t &usage);
  ge::Status SetAllocator(const TensorPlacement &placement, const size_t &usage,
                          std::shared_ptr<memory::MemAllocator> &allocator);
 private:
  std::shared_ptr<memory::MemAllocator> allocators[static_cast<size_t>(TensorPlacement::kTensorPlacementEnd)]
                                                  [static_cast<size_t>(AllocatorUsage::kEnd)];
};
}
#endif  // AIR_CXX_MEM_ALLOCATOR_H
