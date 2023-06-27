/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef GE_GRAPH_COMMON_GRAPH_MANAGER_MEMORY_MANAGER_H_
#define GE_GRAPH_COMMON_GRAPH_MANAGER_MEMORY_MANAGER_H_

#include <string>
#include "external/ge/ge_api_types.h"
#include "runtime/mem.h"

namespace ge {
class MemoryManager {
 public:
  MemoryManager() = default;
  virtual ~MemoryManager() = default;

  MemoryManager(const MemoryManager &) = delete;
  MemoryManager &operator=(const MemoryManager &) & = delete;

  virtual uint8_t *MallocMemory(rtMemType_t memory_type, const std::string &purpose, const std::string &memory_key,
                                size_t memory_size, uint32_t device_id) = 0;

  virtual Status FreeMemory(rtMemType_t memory_type, const std::string &memory_key, uint32_t device_id) = 0;

  virtual uint8_t *GetMemoryBase(rtMemType_t memory_type, const std::string &memory_key, uint32_t device_id) = 0;

  virtual uint8_t *GetMemoryAddr(rtMemType_t memory_type, const std::string &memory_key, uint32_t device_id) = 0;

  virtual uint8_t *MallocMemory(rtMemType_t memory_type, const std::string &purpose,
                                size_t memory_size, uint32_t device_id) = 0;

  virtual Status FreeMemory(rtMemType_t memory_type, void *const memory_addr, uint32_t device_id) = 0;

  virtual uint8_t *GetRdmaPoolMemory(rtMemType_t memory_type, size_t memory_size, uint32_t device_id) = 0;

  virtual uint8_t *GetHostPoolMemory(const rtMemType_t memory_type, const size_t memory_size) = 0;
};
}  // namespace ge

#endif  // GE_GRAPH_COMMON_GRAPH_MANAGER_MEMORY_MANAGER_H_
