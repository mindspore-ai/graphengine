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

#include "single_op/stream_resource.h"

#include "common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "runtime/rt.h"

namespace ge {
StreamResource::~StreamResource() {
  for (auto it : op_map_) {
    // it's safe to delete a nullptr
    delete it.second;
    it.second = nullptr;
  }

  for (auto mem : memory_list_) {
    if (mem != nullptr) {
      auto rt_ret = rtFree(mem);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "rtFree failed"));
    }
  }

  for (auto weight : weight_list_) {
    if (weight != nullptr) {
      auto rt_ret = rtFree(weight);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "rtFree failed"));
    }
  }
}

void StreamResource::CacheOperator(const void *key, SingleOp *single_op) { op_map_[key] = single_op; }

SingleOp *StreamResource::GetOperator(const void *key) {
  auto it = op_map_.find(key);
  if (it == op_map_.end()) {
    return nullptr;
  }

  return it->second;
}

uint8_t *StreamResource::DoMallocMemory(size_t size, size_t &max_allocated, std::vector<uint8_t *> &allocated) {
  if (size <= max_allocated && !allocated.empty()) {
    GELOGD("reuse last memory");
    return allocated.back();
  }

  uint8_t *buffer = nullptr;
  auto ret = rtMalloc(reinterpret_cast<void **>(&buffer), size, RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMalloc failed, size = %zu, ret = %d", size, ret);
    return nullptr;
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "malloc function.", size)

  ret = rtMemset(buffer, size, 0U, size);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMemset failed, ret = %d", ret);
    auto rt_ret = rtFree(buffer);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "rtFree failed"));
    return nullptr;
  }

  GELOGD("Malloc new memory succeeded. size = %zu", size);
  max_allocated = size;
  allocated.emplace_back(buffer);
  return buffer;
}

uint8_t *StreamResource::MallocMemory(size_t size) {
  GELOGD("To Malloc memory, size = %zu", size);
  uint8_t *buffer = DoMallocMemory(size, max_memory_size_, memory_list_);
  return buffer;
}

uint8_t *StreamResource::MallocWeight(size_t size) {
  GELOGD("To Malloc weight, size = %zu", size);
  uint8_t *buffer = DoMallocMemory(size, max_weight_size_, weight_list_);
  return buffer;
}
}  // namespace ge
