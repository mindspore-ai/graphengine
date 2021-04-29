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

#ifndef GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/manager/host_mem_allocator.h"
#include "graph/node.h"
#include "runtime/mem.h"

namespace ge {
class MemoryInfo {
 public:
  MemoryInfo() : memory_addr_(nullptr), memory_size_(0), memory_used_num_(0) {}

  MemoryInfo(uint8_t *memory_addr, size_t memory_size)
      : memory_addr_(memory_addr), memory_size_(memory_size), memory_used_num_(0) {}

  MemoryInfo &operator=(const MemoryInfo &op) {
    if (&op == this) {
      return *this;
    }

    this->memory_addr_ = op.memory_addr_;
    this->memory_size_ = op.memory_size_;
    this->memory_used_num_ = op.memory_used_num_;
    return *this;
  }

  MemoryInfo(const MemoryInfo &op) {
    this->memory_addr_ = op.memory_addr_;
    this->memory_size_ = op.memory_size_;
    this->memory_used_num_ = op.memory_used_num_;
  }
  virtual ~MemoryInfo() = default;

  uint8_t *memory_addr_;
  uint64_t memory_size_;
  int32_t memory_used_num_;
};

class MemoryAllocator {
 public:
  explicit MemoryAllocator(rtMemType_t memory_type) : memory_type_(memory_type), mem_malloced_(false) {}

  virtual ~MemoryAllocator() = default;

  ///
  /// @ingroup ge_graph
  /// @brief memory allocator init
  /// @param [in] options user config params
  /// @return void
  ///
  void Initialize(uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief memory allocator finalize
  /// @return void
  ///
  void Finalize(uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] purpose memory usage
  /// @param [in] size memory size
  /// @param [in] device_id device id
  /// @return  memory address
  ///
  uint8_t *MallocMemory(const string &purpose, size_t memory_size, uint32_t device_id = 0) const;

  ///
  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] device_id device id
  /// @param [out] memory_ptr memory address ptr
  /// @return Status result of function
  ///
  Status FreeMemory(uint8_t *memory_addr, uint32_t device_id = 0) const;

  ///
  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] purpose memory usage
  /// @param [in] memory_key memory key
  /// @param [in] size memory size
  /// @param [in] device_id device id
  /// @return memory address
  ///
  uint8_t *MallocMemory(const string &purpose, const string &memory_key, size_t memory_size,
                        uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] memory_key memory key
  /// @param [in] device_id device id
  /// @return Status result of function
  ///
  Status FreeMemory(const string &memory_key, uint32_t device_id = 0);

  ///
  /// @ingroup ge_graph
  /// @brief get memory address
  /// @param [in] memory_key memory key
  /// @param [in] device_id device id
  /// @return memory address (must not free memory by it)
  ///
  uint8_t *GetMemoryAddr(const string &memory_key, uint32_t device_id = 0);

 private:
  rtMemType_t memory_type_;
  bool mem_malloced_;
  map<string, MemoryInfo> memory_base_map_;
};

using MemoryAllocatorPtr = std::shared_ptr<MemoryAllocator>;
class CachingAllocator;
class RdmaPoolAllocator;
class MemManager {
 public:
  MemManager();
  virtual ~MemManager();
  static MemManager &Instance();
  static MemoryAllocator *Instance(rtMemType_t memory_type);
  CachingAllocator &CachingInstance(rtMemType_t memory_type);
  RdmaPoolAllocator &RdmaPoolInstance(rtMemType_t memory_type);
  HostMemAllocator &HostMemInstance(rtMemType_t memory_type);
  MemManager(const MemManager &) = delete;
  MemManager &operator=(const MemManager &) = delete;
  ///
  /// @ingroup ge_graph
  /// @brief memory allocator manager init
  /// @param [in] options user config params
  /// @return Status result of function
  ///
  Status Initialize(const std::vector<rtMemType_t> &memory_type);

  ///
  /// @ingroup ge_graph
  /// @brief memory allocator finalize
  /// @return void
  ///
  void Finalize() noexcept;

 private:
  ///
  /// @ingroup ge_graph
  /// @brief ge memory allocator
  /// @param [in] memory_type memory type
  /// @return MemoryAllocator ptr
  ///
  MemoryAllocator *GetMemoryAllocator(rtMemType_t memory_type);

  ///
  /// @ingroup ge_graph
  /// @param [in] memory_type memory type
  /// @param [in] allocate_map memory allocator map
  /// @return Status result of function
  ///
  template <typename T>
  Status InitAllocator(const std::vector<rtMemType_t> &memory_type, std::map<rtMemType_t, T *> &allocate_map) {
    T *allocator = nullptr;
    for (unsigned int index : memory_type) {
      auto it = allocate_map.find(index);
      if (it == allocate_map.end()) {
        allocator = new (std::nothrow) T(index);
        if (allocator != nullptr) {
          allocate_map[index] = allocator;
          GELOGI("Create Allocator memory type[%u] success.", index);
        } else {
          GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Alloc Allocator failed.");
        }
      } else {
        allocator = it->second;
      }

      if (allocator == nullptr) {
        GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Create Allocator failed.");
        return ACL_ERROR_GE_MEMORY_ALLOCATION;
      } else {
        if (allocator->Initialize() != SUCCESS) {
          return ACL_ERROR_GE_INTERNAL_ERROR;
        }
      }
    }
    return SUCCESS;
  }
  ///
  /// @ingroup ge_graph
  /// @param [in] memory_type memory type
  /// @param [in] allocate_map memory allocator map
  /// @return Allocator ptr
  ///
  template <typename T>
  T &GetAllocator(rtMemType_t memory_type, std::map<rtMemType_t, T *> allocate_map) {
    std::lock_guard<std::recursive_mutex> lock(allocator_mutex_);
    T *allocator = nullptr;
    auto it = allocate_map.find(memory_type);
    if (it != allocate_map.end()) {
      allocator = it->second;
    }

    // Usually impossible
    if (allocator == nullptr) {
      GELOGW("Get allocator failed, memory type is %u.", memory_type);
      static T default_allocator(RT_MEMORY_RESERVED);
      return default_allocator;
    }
    return *allocator;
  }

  std::map<rtMemType_t, MemoryAllocator *> memory_allocator_map_;
  std::map<rtMemType_t, CachingAllocator *> caching_allocator_map_;
  std::map<rtMemType_t, RdmaPoolAllocator *> rdma_allocator_map_;
  std::map<rtMemType_t, HostMemAllocator *> host_allocator_map_;
  std::recursive_mutex allocator_mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_
