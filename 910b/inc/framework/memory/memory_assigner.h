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

#ifndef INC_FRAMEWORK_MEMORY_MEMORY_ASSIGNER_H_
#define INC_FRAMEWORK_MEMORY_MEMORY_ASSIGNER_H_

#include <utility>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"

namespace ge {
const int64_t MEM_ALIGN_SIZE = 512;
class GE_FUNC_VISIBILITY MemoryAssigner {
 public:
  explicit MemoryAssigner(ge::ComputeGraphPtr compute_graph) : compute_graph_(std::move(compute_graph)) {}
  virtual ~MemoryAssigner() = default;

  MemoryAssigner(const MemoryAssigner &) = delete;

  MemoryAssigner &operator=(const MemoryAssigner &) = delete;

  Status AssignMemory(std::map<uint64_t, size_t> &mem_offset, size_t &zero_copy_mem_size,
                      const bool has_assigned_var_mem = false);

 private:
  ge::ComputeGraphPtr compute_graph_;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_MEMORY_MEMORY_ASSIGNER_H_
