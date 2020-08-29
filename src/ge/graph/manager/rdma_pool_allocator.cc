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

#include "graph/manager/rdma_pool_allocator.h"
#include "framework/common/debug/ge_log.h"

namespace {
const size_t kAlignedSize = 512;
const float kSplitThreshold = 0.5;

inline size_t GetAlignedBlockSize(size_t size) {
  if (size == 0) {
    return kAlignedSize;
  }
  return kAlignedSize * ((size + kAlignedSize - 1) / kAlignedSize);
}

inline bool ShouldSplit(const ge::Block *block, size_t size) {
  return static_cast<double>(size) <= (static_cast<double>(block->size) * kSplitThreshold);
}

inline bool CanMerge(ge::Block *block) { return block != nullptr && !block->allocated; }
}  // namespace

namespace ge {
RdmaPoolAllocator::RdmaPoolAllocator(rtMemType_t memory_type)
    : memory_type_(memory_type), block_bin_(BlockBin([](const Block *left, const Block *right) {
        if (left->size != right->size) {
          return left->size < right->size;
        }
        return reinterpret_cast<uintptr_t>(left->ptr) < reinterpret_cast<uintptr_t>(right->ptr);
      })) {}

Status RdmaPoolAllocator::Initialize() {
  memory_allocator_ = MemManager::Instance(memory_type_);
  if (memory_allocator_ == nullptr) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}
void RdmaPoolAllocator::Finalize() {
  for (auto it = allocated_blocks_.begin(); it != allocated_blocks_.end();) {
    auto block = it->second;
    allocated_blocks_.erase(it);
    delete block;
  }
  for (auto it = block_bin_.begin(); it != block_bin_.end();) {
    auto block = *it;
    block_bin_.erase(it);
    delete block;
  }

  if (rdma_base_addr_ != nullptr) {
    if (memory_allocator_->FreeMemory(rdma_base_addr_) != SUCCESS) {
      GELOGW("Free rdma pool memory failed");
    }
  }
}

Status RdmaPoolAllocator::InitMemory(size_t mem_size, uint32_t device_id) {
  if (rdma_base_addr_ != nullptr) {
    GELOGE(GE_MULTI_INIT, "Rdma pool has been malloced");
    return GE_MULTI_INIT;
  }
  const std::string purpose = "Memory for rdma pool.";
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  rdma_base_addr_ = memory_allocator_->MallocMemory(purpose, mem_size, device_id);
  if (rdma_base_addr_ == nullptr) {
    GELOGE(GE_GRAPH_MALLOC_FAILED, "Rdma pool memory malloc failed");
    return GE_GRAPH_MALLOC_FAILED;
  }
  rdma_mem_size_ = mem_size;
  // Init with a base block.
  auto *base_block = new (std::nothrow) Block(device_id, mem_size, rdma_base_addr_);
  if (base_block == nullptr) {
    GELOGE(GE_GRAPH_MALLOC_FAILED, "Block malloc failed");
    return GE_GRAPH_MALLOC_FAILED;
  }
  block_bin_.insert(base_block);
  return SUCCESS;
}

uint8_t *RdmaPoolAllocator::Malloc(size_t size, uint32_t device_id) {
  auto aligned_size = GetAlignedBlockSize(size);
  Block key(device_id, aligned_size, nullptr);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = block_bin_.lower_bound(&key);
  if (it != block_bin_.end()) {
    Block *block = *it;
    block_bin_.erase(it);
    block->allocated = true;
    if (block->ptr == nullptr) {
      GELOGE(INTERNAL_ERROR, "Rdmapool memory address is nullptr.");
      return nullptr;
    }
    allocated_blocks_.emplace(block->ptr, block);
    GELOGI("Find block size = %zu", block->size);

    if (ShouldSplit(block, aligned_size)) {
      auto *new_block =
        new (std::nothrow) Block(device_id, block->size - aligned_size, nullptr, block->ptr + aligned_size);
      if (new_block == nullptr) {
        GELOGW("Block split failed");
        return block->ptr;
      }
      new_block->next = block->next;
      if (block->next != nullptr) {
        block->next->prev = new_block;
      }
      new_block->prev = block;
      block->next = new_block;
      block->size = aligned_size;
      block_bin_.insert(new_block);
    }
    return block->ptr;
  }
  return nullptr;
}

Status RdmaPoolAllocator::Free(uint8_t *memory_addr, uint32_t device_id) {
  GELOGI("Free device id = %u", device_id);
  if (memory_addr == nullptr) {
    GELOGE(GE_GRAPH_FREE_FAILED, "Invalid memory pointer");
    return GE_GRAPH_FREE_FAILED;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = allocated_blocks_.find(memory_addr);
  if (it == allocated_blocks_.end()) {
    GELOGE(PARAM_INVALID, "Invalid memory pointer");
    return PARAM_INVALID;
  }
  Block *block = it->second;
  block->allocated = false;
  allocated_blocks_.erase(it);
  block_bin_.insert(block);
  // Each time merge with its pre and next.
  MergeBlockNearby(block, block->next);
  MergeBlockNearby(block->prev, block);
  return SUCCESS;
}

void RdmaPoolAllocator::MergeBlockNearby(Block *pre_block, Block *block) {
  if (!(CanMerge(pre_block) && CanMerge(block))) {
    return;
  }
  pre_block->size += block->size;
  pre_block->next = block->next;
  if (block->next != nullptr) {
    block->next->prev = pre_block;
  }
  block_bin_.erase(block);
  delete block;
}

Status RdmaPoolAllocator::GetBaseAddr(uint64_t &base_addr, uint64_t &mem_size) {
  if (rdma_base_addr_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "Rdma base addr is nullptr.");
    return INTERNAL_ERROR;
  }
  base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdma_base_addr_));
  mem_size = rdma_mem_size_;
  return SUCCESS;
}
}  // namespace ge
