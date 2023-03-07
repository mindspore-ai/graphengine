/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "common/overflow/overflow_manager.h"

#include "runtime/rt.h"
#include "framework/common/debug/log.h"

namespace ge {
namespace {
constexpr size_t kValidOverflowAddrSize = 1U;
}
void *OverflowManager::overflow_addr_ = nullptr;
size_t OverflowManager::globalworkspace_overflow_size_ = 0U;
std::mutex OverflowManager::overflow_manager_mutex_;

OverflowManager::OverflowManager() {}

OverflowManager::~OverflowManager() {}

OverflowManager &OverflowManager::GetInstance() {
  static OverflowManager inst;
  return inst;
}

Status OverflowManager::MallocOverflowAddr(const std::string &purpose, const size_t size) {
  const std::lock_guard<std::mutex> lock(overflow_manager_mutex_);
  if (overflow_addr_ == nullptr) {
    GE_CHECK_GE(size, kValidOverflowAddrSize);
    GELOGD("To Malloc overflow memory, size = %ld", size);
    GE_CHK_RT_RET(rtMalloc(&overflow_addr_, size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
    GE_PRINT_DYNAMIC_MEMORY(rtMalloc, purpose.c_str(), size);
    globalworkspace_overflow_size_ = size;
  }
  return SUCCESS;
}

void OverflowManager::Finalize() {
  // Free mem for overflow detection
  const std::lock_guard<std::mutex> lock(overflow_manager_mutex_);
  GE_FREE_RT_LOG(overflow_addr_);
}
}  // namespace ge

