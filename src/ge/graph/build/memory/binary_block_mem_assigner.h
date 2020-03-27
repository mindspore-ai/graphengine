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

#ifndef GE_GRAPH_BUILD_MEMORY_BINARY_BLOCK_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_BINARY_BLOCK_MEM_ASSIGNER_H_

#include <utility>
#include <vector>

#include "graph/build/memory/block_mem_assigner.h"

namespace ge {
class BinaryBlockMemAssigner : public BlockMemAssigner {
 public:
  explicit BinaryBlockMemAssigner(ge::ComputeGraphPtr compute_graph) : BlockMemAssigner(std::move(compute_graph)) {}

  BinaryBlockMemAssigner(const BinaryBlockMemAssigner &) = delete;

  BinaryBlockMemAssigner &operator=(const BinaryBlockMemAssigner &) = delete;

  ~BinaryBlockMemAssigner() override = default;

  Status GetMemoryRanges(std::vector<int64_t> &ranges) override;

 private:
  void PlanRanges(size_t range_number_limit, std::vector<std::vector<int64_t>> &ranges);
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_BINARY_BLOCK_MEM_ASSIGNER_H_
