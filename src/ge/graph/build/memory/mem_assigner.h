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

#ifndef GE_GRAPH_BUILD_MEMORY_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_MEM_ASSIGNER_H_

#include "common/ge_inner_error_codes.h"
#include "memory/memory_assigner.h"

namespace ge {
static const int64_t kInvalidOffset = -1;

class MemAssigner {
 public:
  MemAssigner() = default;

  MemAssigner(const MemAssigner &) = delete;

  MemAssigner &operator=(const MemAssigner &) = delete;

  virtual ~MemAssigner() = default;

  virtual Status Assign() = 0;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_MEM_ASSIGNER_H_
