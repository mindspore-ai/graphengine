/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "npu_memory_allocator.h"
#include <mutex>
#include "framework/common/debug/log.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/graph_caching_allocator.h"

namespace ge {
namespace hybrid {
std::map<uint32_t, std::unique_ptr<NpuMemoryAllocator>> NpuMemoryAllocator::allocators_;
std::mutex NpuMemoryAllocator::mu_;

NpuMemoryAllocator *NpuMemoryAllocator::GetAllocator() {
  int32_t device_id = 0;
  if (rtGetDevice(&device_id) != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Failed to get device id");
    return nullptr;
  }

  GELOGD("Got device id = %d from context", device_id);
  return GetAllocator(static_cast<uint32_t>(device_id));
}

NpuMemoryAllocator::NpuMemoryAllocator(uint32_t device_id) : device_id_(device_id) {}

void *NpuMemoryAllocator::Allocate(std::size_t size, void *try_reuse_addr) {
  void *buffer =
    MemManager::CachingInstance(RT_MEMORY_HBM).Malloc(size, reinterpret_cast<uint8_t *>(try_reuse_addr), device_id_);
  if (buffer == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to malloc memory, device_id = %u, size = %zu", device_id_, size);
    return nullptr;
  }

  GELOGI("Allocating buffer of size %u successfully. device_id = %u, address = %p", size, device_id_, buffer);
  return buffer;
}

void NpuMemoryAllocator::Deallocate(void *data) {
  GELOGI("To deallocating buffer, addr = %p", data);
  if (data != nullptr) {
    GELOGI("Deallocating buffer successfully. addr = %p", data);
    MemManager::CachingInstance(RT_MEMORY_HBM).Free(reinterpret_cast<uint8_t *>(data), device_id_);
  }
}

NpuMemoryAllocator *NpuMemoryAllocator::GetAllocator(uint32_t device_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = allocators_.find(device_id);
  if (it == allocators_.end()) {
    auto allocator = std::unique_ptr<NpuMemoryAllocator>(new (std::nothrow) NpuMemoryAllocator(device_id));
    if (allocator == nullptr) {
      return nullptr;
    }

    allocators_.emplace(device_id, std::move(allocator));
  }

  return allocators_[device_id].get();
}

void NpuMemoryAllocator::DestroyAllocator() {
  std::lock_guard<std::mutex> lk(mu_);
  int device_id = 0;
  allocators_.erase(device_id);
}
}  // namespace hybrid
}  // namespace ge