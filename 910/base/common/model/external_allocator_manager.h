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
#ifndef GE_COMMON_EXTERNAL_ALLOCATOR_MANAGER_H
#define GE_COMMON_EXTERNAL_ALLOCATOR_MANAGER_H

#include <mutex>
#include <map>
#include "ge/ge_allocator.h"

namespace ge {
class ExternalAllocatorManager {
 public:
  static void SetExternalAllocator(const void *const stream, AllocatorPtr allocator);
  static void DeleteExternalAllocator(const void *const stream);
  static AllocatorPtr GetExternalAllocator(const void *const stream);
 private:
  static std::mutex stream_to_external_allocator_Mutex_;
  static std::map<const void *const, AllocatorPtr> stream_to_external_allocator_;
};
}  // namespace ge
#endif // GE_COMMON_EXECUTOR_H