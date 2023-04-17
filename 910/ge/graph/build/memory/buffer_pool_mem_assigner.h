/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef GE_GRAPH_BUILD_MEMORY_BUFFER_POOL_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_BUFFER_POOL_MEM_ASSIGNER_H_

#include <vector>
#include <map>
#include <unordered_map>
#include "graph/build/memory/mem_assigner.h"
#include "runtime/mem.h"

namespace ge {
class BufferPoolMemAssigner : public MemAssigner {
 public:
  BufferPoolMemAssigner(ComputeGraphPtr compute_graph, const std::map<int64_t, size_t> &mem_type_to_offset)
      : MemAssigner(), compute_graph_(compute_graph),
        mem_type_(0),
        mem_offset_(0),
        mem_offset_base_(0),
        init_offset_base_(false),
        mem_type_to_offset_(mem_type_to_offset) {}

  BufferPoolMemAssigner(const BufferPoolMemAssigner &) = delete;

  BufferPoolMemAssigner &operator=(const BufferPoolMemAssigner &) = delete;

  ~BufferPoolMemAssigner() override = default;

  Status Assign() override;

  size_t GetMemOffset() const { return mem_offset_; }

  int64_t GetMemType() const { return mem_type_; }

 private:
  static Status GetOutputMemoryType(const NodePtr &node, size_t idx,  int64_t &memory_type);

  Status InitAssigner(const ComputeGraphPtr &graph);

  Status InitMemOffsetBase(const NodePtr &node);

  Status AssignOutput();

  Status AssignOutputInOneBufferPool(const std::string &batch_label,
                                     int64_t output_offset_base,
                                     const std::vector<NodePtr> &buffer_pool_nodes);

  ComputeGraphPtr compute_graph_;

  int64_t mem_type_;

  size_t mem_offset_;

  int64_t mem_offset_base_;

  bool init_offset_base_;

  std::map<int64_t, size_t> mem_type_to_offset_;

  // Use map to ensure that each visit is in the order of pool id
  std::unordered_map<std::string, std::map<int64_t, std::vector<NodePtr>>> buffer_pool_nodes_;

  // Use map to ensure that each visit is in the order of pool id
  std::unordered_map<std::string, std::map<int64_t, int64_t>> buffer_pool_size_;

  std::unordered_map<std::string, std::unordered_map<int64_t, int64_t>> buffer_pool_offset_base_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_BUFFER_POOL_MEM_ASSIGNER_H_
