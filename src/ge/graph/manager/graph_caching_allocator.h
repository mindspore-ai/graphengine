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

#ifndef GE_GRAPH_MANAGER_GRAPH_CACHING_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_GRAPH_CACHING_ALLOCATOR_H_

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "runtime/mem.h"

namespace ge {

constexpr size_t kRoundBlockSize = 512;   // all block sizes are rounded to at least 512 bytes
constexpr double kSplitThreshold = 0.75;  // split when malloc size <= small block size * kSpliThreshold
constexpr size_t kKByteSize = 1024;
constexpr size_t kMByteSize = 1024 * 1024;
constexpr size_t kGByteSize = 1024 * 1024 * 1024;

struct Block;
typedef bool (*Comparison)(const Block *, const Block *);
using BlockBin = std::set<Block *, Comparison>;
static const uint32_t kNumBins = 8;

struct Block {
  uint32_t device_id;  // npu device id
  size_t size;         // block size in bytes
  BlockBin *bin;       // owning block bin
  uint8_t *ptr;        // memory address
  bool allocated;      // in-use flag
  Block *prev;         // prev block if split from a larger allocation
  Block *next;         // next block if split from a larger allocation

  Block(uint32_t device, size_t size, BlockBin *bin, uint8_t *ptr)
      : device_id(device), size(size), bin(bin), ptr(ptr), allocated(0), prev(nullptr), next(nullptr) {}

  // constructor for search key
  Block(uint32_t device, size_t size, uint8_t *ptr)
      : device_id(device), size(size), bin(nullptr), ptr(ptr), allocated(0), prev(nullptr), next(nullptr) {}

  bool IsSplit() const { return (prev != nullptr) || (next != nullptr); }
};

class MemoryAllocator;

class CachingAllocator {
 public:
  explicit CachingAllocator(rtMemType_t memory_type);

  virtual ~CachingAllocator() = default;

  ///
  /// @ingroup ge_graph
  /// @brief caching allocator init
  /// @param [in] device id
  /// @return Status of init
  ///
  Status Initialize(uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief memory allocator finalize, release cached memory
  /// @return void
  ///
  void Finalize(uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] size memory size
  /// @param [in] try to reuse the same memory
  /// @param [in] device id
  /// @return  memory address
  ///
  uint8_t *Malloc(size_t size, uint8_t *org_ptr = nullptr, uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] device_id device id
  /// @param [out] memory_ptr memory address ptr
  /// @return Status result of function
  ///
  Status Free(uint8_t *memory_addr, uint32_t device_id = 0);

 private:
  ///
  /// @ingroup ge_graph
  /// @brief extend cache by size
  /// @param [in] memory size
  /// @param [in] device id
  /// @return Status result of function
  ///
  Status TryExtendCache(size_t size, uint32_t device_id);

  ///
  /// @ingroup ge_graph
  /// @brief find free block by size
  /// @param [in] memory size
  /// @param [in] device_id device id
  /// @return block ptr
  ///
  Block *FindFreeBlock(size_t size, uint8_t *org_ptr, uint32_t device_id);

  ///
  /// @ingroup ge_graph
  /// @brief get the right bin based on size
  /// @param [in] original malloc size
  /// @return block bin
  ///
  BlockBin *GetBlockBin(size_t size);

  ///
  /// @ingroup ge_graph
  /// @brief add memory to right bin based on size
  /// @param [in] memory ptr
  /// @param [in] memory size
  /// @return Status result of function
  ///
  Status AddToBlockBin(uint8_t *ptr, size_t size);

  ///
  /// @ingroup ge_graph
  /// @brief free block to right bin
  /// @param [in] block ptr
  /// @return void
  ///
  void FreeBlock(Block *block);

  ///
  /// @ingroup ge_graph
  /// @brief free all cached blocks to right bin and release the memory when memory is not enough
  /// @return void
  ///
  void FreeCachedBlocks();

  ///
  /// @ingroup ge_graph
  /// @brief free allocated and cached blocks and release the memory when process exit
  /// @return void
  ///
  void FreeBlocks();

  ///
  /// @ingroup ge_graph
  /// @brief free block bins when process exit
  /// @return void
  ///
  void FreeBlockBins();

  ///
  /// @ingroup ge_graph
  /// @brief If a split block is freed, try merging with the original block
  /// @param [inout] dest block ptr
  /// @param [in] src block ptr
  /// @param [out] block bin
  /// @return void
  ///
  void MergeBlocks(Block *dst, Block *src, BlockBin &bin);

  ///
  /// @ingroup ge_graph
  /// @brief If the allocated memory size is too much smaller than the memory block, try to split the memory block
  /// @param [in] original block ptr
  /// @param [in] allocated memory size
  /// @param [in] block bin
  /// @param [in] device id
  /// @return splited block ptr
  ///
  Block *SplitBlock(Block *block, size_t size, BlockBin &bin, uint32_t device_id);

 private:
  rtMemType_t memory_type_;

  // device memory allocator
  MemoryAllocator *memory_allocator_;

  // lock around all operations
  mutable std::recursive_mutex mutex_;

  // allocated blocks by memory pointer
  std::unordered_map<uint8_t *, Block *> allocated_blocks_;

  // block bins by different block size
  BlockBin *free_block_bins_[kNumBins];
};

};  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_CACHING_ALLOCATOR_H_
