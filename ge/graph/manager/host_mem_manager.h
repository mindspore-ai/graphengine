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

#ifndef GE_GRAPH_MANAGER_HOST_VAR_MANAGER_H_
#define GE_GRAPH_MANAGER_HOST_VAR_MANAGER_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "framework/common/l2_cache_optimize.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/tensor.h"
#include "runtime/mem.h"

namespace ge {
class HostMemoryAllocator {
 public:
  ~HostMemoryAllocator() = default;

  Status Allocate(std::size_t size, uint8_t *memory_addr);
  Status DeAllocate(uint8_t *memory_addr);
};

struct HostMemInfo {
  uint8_t *address;
  uint64_t data_size;
  HostMemInfo() : address(nullptr), data_size(0) {}
  HostMemInfo(uint8_t *addr, uint64_t size) : address(addr), data_size(size) {}
};

class HostMemManager {
 public:
  HostMemManager() = default;
  ~HostMemManager() { Finalize(); }
  HostMemManager(const HostMemManager &) = delete;
  HostMemManager &operator=(const HostMemManager &) = delete;

  static HostMemManager &Instance();
  Status Initialize();
  void Finalize() noexcept;
  Status MallocMemoryForHostVar(const string &op_name, uint64_t tensor_size, uint8_t *&var_addr);
  Status QueryVarMemInfo(const string &op_name, uint64_t &base_addr, uint64_t &data_size);

 private:
  std::unordered_map<std::string, HostMemInfo> var_memory_base_map_;
  std::unique_ptr<HostMemoryAllocator> allocator_;
  mutable std::recursive_mutex mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_HOST_VAR_MANAGER_H_
