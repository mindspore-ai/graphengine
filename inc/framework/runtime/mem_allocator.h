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
#ifndef AIR_CXX_MEM_ALLOCATOR_H
#define AIR_CXX_MEM_ALLOCATOR_H
#include "block.h"
#include "exe_graph/runtime/allocator.h"

namespace gert {
namespace memory {
struct MemAllocator {
  virtual Block *Malloc(size_t size) = 0;
  virtual ~MemAllocator() = default;
};
}  // namespace memory

struct ExternalAllocators {
 public:
  memory::MemAllocator *GetAllocator(size_t placement, size_t usage);
  ge::Status SetAllocator(size_t placement, size_t usage, std::unique_ptr<memory::MemAllocator> allocator);

 private:
  std::unique_ptr<memory::MemAllocator> allocators[kTensorPlacementEnd][static_cast<size_t>(AllocatorUsage::kEnd)];
};
}  // namespace gert
#endif  // AIR_CXX_MEM_ALLOCATOR_H
