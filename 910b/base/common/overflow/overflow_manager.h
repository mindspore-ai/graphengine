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

#ifndef GE_GRAPH_COMMON_OVERFLOW_MANAGER_H_
#define GE_GRAPH_COMMON_OVERFLOW_MANAGER_H_

#include <mutex>
#include <string>
#include "framework/common/ge_types.h"

namespace ge {
class OverflowManager {
 public:
  static OverflowManager &GetInstance();
  static Status MallocOverflowAddr(const std::string &purpose, const size_t size);
  static void *GetOverflowAddr() { return overflow_addr_; }
  static size_t GetGlobalWorkspaceOverflowSize() { return globalworkspace_overflow_size_; }
  void Finalize();

 private:
  OverflowManager() = default;
  ~OverflowManager() = default;
  static void *overflow_addr_;
  static size_t globalworkspace_overflow_size_;
  static std::mutex overflow_manager_mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_COMMON_OVERFLOW_MANAGER_H_

