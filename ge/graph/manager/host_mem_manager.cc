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

#include "graph/manager/host_mem_manager.h"

#include <sstream>

#include "graph/utils/tensor_utils.h"

namespace ge {
Status HostMemoryAllocator::Allocate(std::size_t memory_size, uint8_t *memory_addr) {
  GELOGI("HostMemoryAllocator::MallocMemory size= %zu.", memory_size);
  return SUCCESS;
}

Status HostMemoryAllocator::DeAllocate(uint8_t *memory_addr) {
  if (rtFreeHost(memory_addr) != RT_ERROR_NONE) {
    GELOGE(GE_GRAPH_FREE_FAILED, "MemoryAllocator::Free memory failed.");
    return GE_GRAPH_FREE_FAILED;
  }
  memory_addr = nullptr;
  return ge::SUCCESS;
}

HostMemManager &HostMemManager::Instance() {
  static HostMemManager mem_manager;
  return mem_manager;
}

Status HostMemManager::Initialize() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  allocator_ = std::unique_ptr<HostMemoryAllocator>(new (std::nothrow) HostMemoryAllocator());
  if (allocator_ == nullptr) {
    GELOGE(GE_GRAPH_MALLOC_FAILED, "Host mem allocator init failed!");
    return GE_GRAPH_MALLOC_FAILED;
  }
  return SUCCESS;
}

void HostMemManager::Finalize() noexcept {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  for (const auto &it : var_memory_base_map_) {
    if (allocator_->DeAllocate(it.second.address) != SUCCESS) {
      GELOGW("Host %s mem deAllocator failed!", it.first.c_str());
    }
  }
  var_memory_base_map_.clear();
}

Status HostMemManager::MallocMemoryForHostVar(const string &op_name, uint64_t tensor_size, uint8_t *&var_addr) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_memory_base_map_.find(op_name) != var_memory_base_map_.end()) {
    GELOGI("Host mem for variable %s has been malloced", op_name.c_str());
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(allocator_);
  GE_CHK_STATUS(allocator_->Allocate(tensor_size, var_addr));
  HostMemInfo info(var_addr, tensor_size);
  var_memory_base_map_[op_name] = info;
  return SUCCESS;
}

Status HostMemManager::QueryVarMemInfo(const string &op_name, uint64_t &base_addr, uint64_t &data_size) {
  if (var_memory_base_map_.find(op_name) == var_memory_base_map_.end()) {
    GELOGE(INTERNAL_ERROR, "Find host base base_addr failed,node name:%s!", op_name.c_str());
    return INTERNAL_ERROR;
  }
  base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(var_memory_base_map_[op_name].address));
  data_size = var_memory_base_map_[op_name].data_size;
  return SUCCESS;
}
}  // namespace ge
