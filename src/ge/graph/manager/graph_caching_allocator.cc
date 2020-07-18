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

#include "graph/manager/graph_caching_allocator.h"

#include <set>
#include <string>
#include <utility>

#include "framework/common/debug/ge_log.h"
#include "graph/manager/graph_mem_allocator.h"

namespace ge {
const size_t bin_ranges[kNumBins] = {kRoundBlockSize * kKByteSize,
                                     8 * kMByteSize,
                                     32 * kMByteSize,
                                     128 * kMByteSize,
                                     kGByteSize,
                                     4 * kGByteSize,
                                     16 * kGByteSize,
                                     26 * kGByteSize};

static bool BlockComparator(const Block *left, const Block *right) {
  if (left->size != right->size) {
    return left->size < right->size;
  }
  return reinterpret_cast<uintptr_t>(left->ptr) < reinterpret_cast<uintptr_t>(right->ptr);
}

bool CanMerge(Block *block) {
  if (block == nullptr || block->allocated || !block->IsSplit()) {
    return false;
  }
  return true;
}

size_t GetBinIndex(size_t size) {
  size_t index = 0;
  for (auto range : bin_ranges) {
    if (size <= range) {
      break;
    }
    ++index;
  }
  if (index > kNumBins - 1) {
    index = kNumBins - 1;
  }
  return index;
}

size_t GetAllocationSize(size_t size) {
  size_t index = GetBinIndex(size);
  return bin_ranges[index];
}

///
/// @ingroup ge_graph
/// @brief block size based on alignment
/// @param [in] original malloc size
/// @return allocation size
///
size_t GetBlockSize(size_t size) {
  if (size == 0) {
    return kRoundBlockSize;
  }
  return kRoundBlockSize * ((size + kRoundBlockSize - 1) / kRoundBlockSize);
}

bool ShouldSplit(const Block *block, size_t size) {
  return static_cast<double>(size) <= (static_cast<double>(block->size) * kSplitThreshold);
}

CachingAllocator::CachingAllocator(rtMemType_t memory_type) : memory_type_(memory_type), memory_allocator_(nullptr) {
  for (uint32_t i = 0; i < kNumBins; ++i) {
    free_block_bins_[i] = nullptr;
  }
}

Status CachingAllocator::Initialize(uint32_t device_id) {
  GELOGI("Device id %u", device_id);
  // when redo Initialize free old memory
  FreeBlocks();
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (uint32_t i = 0; i < kNumBins; ++i) {
    if (free_block_bins_[i] != nullptr) {
      continue;
    }
    auto bin_ptr = new (std::nothrow) BlockBin(BlockComparator);
    if (bin_ptr == nullptr) {
      GELOGE(ge::FAILED, "Alloc BlockBin failed.");
      return ge::FAILED;
    }
    free_block_bins_[i] = bin_ptr;
  }
  memory_allocator_ = MemManager::Instance(memory_type_);
  if (memory_allocator_ == nullptr) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

void CachingAllocator::Finalize(uint32_t device_id) {
  GELOGI("Device id %u", device_id);
  FreeBlocks();
  FreeBlockBins();
}

uint8_t *CachingAllocator::Malloc(size_t size, uint8_t *org_ptr, uint32_t device_id) {
  uint8_t *ptr = nullptr;
  size = GetBlockSize(size);
  Block *block = FindFreeBlock(size, org_ptr, device_id);
  if (block != nullptr) {
    ptr = block->ptr;
  } else {
    if (ge::SUCCESS == TryExtendCache(size, device_id)) {
      block = FindFreeBlock(size, org_ptr, device_id);
      if (block != nullptr) {
        ptr = block->ptr;
      }
    }
  }
  if (ptr == nullptr) {
    GELOGE(FAILED, "Malloc failed device id = %u, size= %zu", device_id, size);
  } else {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    block->allocated = true;
    allocated_blocks_[block->ptr] = block;
    GELOGI("Malloc device id = %u, size= %zu", device_id, size);
  }
  return ptr;
}

Status CachingAllocator::Free(uint8_t *ptr, uint32_t device_id) {
  GELOGI("Free device id = %u", device_id);
  if (ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Invalid memory pointer");
    return ge::PARAM_INVALID;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = allocated_blocks_.find(ptr);
  if (it == allocated_blocks_.end()) {
    GELOGE(PARAM_INVALID, "Invalid memory pointer");
    return ge::PARAM_INVALID;
  }
  Block *block = it->second;
  allocated_blocks_.erase(it);
  FreeBlock(block);
  return ge::SUCCESS;
}

void CachingAllocator::FreeBlock(Block *block) {
  if (block == nullptr || !block->allocated) {
    return;
  }
  GELOGI("Free block size = %zu", block->size);

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  block->allocated = false;
  auto &bin = *block->bin;
  Block *merge_blocks[] = {block->prev, block->next};
  for (Block *merge_block : merge_blocks) {
    MergeBlocks(block, merge_block, bin);
  }
  bin.insert(block);
}

void CachingAllocator::MergeBlocks(Block *dst, Block *src, BlockBin &bin) {
  if (!CanMerge(dst) || !CanMerge(src)) {
    return;
  }

  if (dst->prev == src) {
    dst->ptr = src->ptr;
    dst->prev = src->prev;
    if (dst->prev != nullptr) {
      dst->prev->next = dst;
    }
  } else {
    dst->next = src->next;
    if (dst->next != nullptr) {
      dst->next->prev = dst;
    }
  }

  dst->size += src->size;
  bin.erase(src);
  delete src;
}

BlockBin *CachingAllocator::GetBlockBin(size_t size) {
  size_t index = GetBinIndex(size);
  return free_block_bins_[index];
}

Block *CachingAllocator::FindFreeBlock(size_t size, uint8_t *org_ptr, uint32_t device_id) {
  // org_ptr - 1, try to find ptr same as org_ptr
  Block key(device_id, size, (org_ptr == nullptr ? nullptr : org_ptr - 1));
  BlockBin *bin = GetBlockBin(size);
  if (bin == nullptr) {
    GELOGE(ge::FAILED, "Get block bin failed size = %zu", size);
    return nullptr;
  }
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = bin->lower_bound(&key);
  if (it != bin->end()) {
    Block *block = *it;
    bin->erase(it);
    if (block != nullptr) {
      GELOGI("Find block size = %zu", block->size);
      if (ShouldSplit(block, size)) {
        return SplitBlock(block, size, *bin, device_id);
      }
    }
    return block;
  }
  return nullptr;
}

Block *CachingAllocator::SplitBlock(Block *block, size_t size, BlockBin &bin, uint32_t device_id) {
  // block has been checked, should not be nullptr
  Block *remaining = block;
  Block *new_block = new (std::nothrow) Block(device_id, size, &bin, block->ptr);
  if (new_block == nullptr) {
    GELOGE(ge::FAILED, "Alloc block failed size = %zu", size);
    return block;
  }
  new_block->prev = remaining->prev;
  if (new_block->prev != nullptr) {
    new_block->prev->next = new_block;
  }
  new_block->next = remaining;
  remaining->prev = new_block;
  remaining->ptr = remaining->ptr + size;
  remaining->size -= size;
  bin.insert(remaining);
  return new_block;
}

Status CachingAllocator::TryExtendCache(size_t size, uint32_t device_id) {
  auto memory_size = GetAllocationSize(size);
  const std::string purpose = "Memory for caching.";
  auto memory_addr = memory_allocator_->MallocMemory(purpose, memory_size, device_id);
  // try to free caches and malloc again when malloc memory failed
  if (memory_addr == nullptr) {
    FreeCachedBlocks();
    memory_addr = memory_allocator_->MallocMemory(purpose, memory_size, device_id);
    if (memory_addr == nullptr) {
      GELOGE(ge::FAILED, "TryExtendCache failed, no enough memory for size = %zu, device_id = %u", memory_size,
             device_id);
      return ge::FAILED;
    }
  }
  if (AddToBlockBin(memory_addr, memory_size, device_id) != ge::SUCCESS) {
    (void)memory_allocator_->FreeMemory(memory_addr);
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

Status CachingAllocator::AddToBlockBin(uint8_t *ptr, size_t size, uint32_t device_id) {
  BlockBin *bin = GetBlockBin(size);
  if (bin == nullptr) {
    GELOGE(ge::FAILED, "Get block bin failed size = %zu", size);
    return ge::FAILED;
  }
  Block *block = new (std::nothrow) Block(device_id, size, bin, nullptr);
  if (block == nullptr) {
    GELOGE(ge::FAILED, "Alloc block failed size = %zu", size);
    return ge::FAILED;
  }

  GELOGI("Block size = %zu", size);
  block->ptr = ptr;
  block->size = size;

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  bin->insert(block);
  return ge::SUCCESS;
}

void CachingAllocator::FreeCachedBlocks() {
  GELOGI("Free cached blocks");
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (uint32_t i = 0; i < kNumBins; ++i) {
    auto pool = free_block_bins_[i];
    if (pool == nullptr) {
      continue;
    }
    for (auto it = pool->begin(); it != pool->end();) {
      Block *block = *it;
      // free block memory that has not been split
      if ((block != nullptr) && (block->ptr != nullptr) && (block->prev == nullptr) && (block->next == nullptr) &&
          (memory_allocator_->FreeMemory(block->ptr) == ge::SUCCESS)) {
        pool->erase(it++);
        delete block;
        continue;
      }
      ++it;
    }
  }
}

void CachingAllocator::FreeBlocks() {
  GELOGI("Free blocks");
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // free allocated blocks and put to cache
  for (auto &it : allocated_blocks_) {
    FreeBlock(it.second);
  }
  allocated_blocks_.clear();

  FreeCachedBlocks();
}

void CachingAllocator::FreeBlockBins() {
  GELOGI("Free block bins");
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (uint32_t i = 0; i < kNumBins; ++i) {
    if (free_block_bins_[i] != nullptr) {
      delete free_block_bins_[i];
      free_block_bins_[i] = nullptr;
    }
  }
}
}  // namespace ge
