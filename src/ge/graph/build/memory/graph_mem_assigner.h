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

#ifndef GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "runtime/mem.h"

namespace ge {
struct MemoryOffset {
  MemoryOffset(rtMemType_t mem_type, size_t mem_offset) : mem_type_(mem_type), mem_offset_(mem_offset) {}

 public:
  rtMemType_t mem_type_;
  size_t mem_offset_;
};

using MemoryOffsetList = vector<MemoryOffset>;

class VariableMemoryAssigner {
 public:
  explicit VariableMemoryAssigner(ge::ComputeGraphPtr compute_graph) : compute_graph_(std::move(compute_graph)) {}

  VariableMemoryAssigner(const VariableMemoryAssigner &) = delete;

  VariableMemoryAssigner &operator=(const VariableMemoryAssigner &) = delete;

  virtual ~VariableMemoryAssigner() = default;

  ///
  /// @ingroup ge_graph
  /// @brief assign memory offset
  /// @return Status result of function
  ///
  ge::Status Assign();

  ///
  /// @ingroup ge_graph
  /// @brief assign variable attr to nodes
  /// @return Status result of function
  ///
  ge::Status AssignVarAttr2Nodes();

 private:
  ge::ComputeGraphPtr compute_graph_;
};

using VariableMemoryAssignerPtr = std::shared_ptr<VariableMemoryAssigner>;

class GraphMemoryAssigner {
 public:
  explicit GraphMemoryAssigner(ge::ComputeGraphPtr compute_graph) : compute_graph_(std::move(compute_graph)) {}

  GraphMemoryAssigner(const GraphMemoryAssigner &) = delete;

  GraphMemoryAssigner &operator=(const GraphMemoryAssigner &) = delete;

  virtual ~GraphMemoryAssigner() = default;

  ///
  /// @ingroup ge_graph
  /// @brief assign memory offset
  /// @return Status result of function
  ///
  ge::Status AssignMemory();

  ///
  /// @ingroup ge_graph
  /// @brief assign variable attr to nodes,
  /// must be called after all memory assigned.
  /// @return Status result of function
  ///
  ge::Status AssignVarAttr2Nodes();

  ge::Status AssignSubgraphInputsMemory();

  ge::Status AssignSubgraphOutputsMemory();

  ge::Status ReAssignMemory(bool is_loop_graph, size_t &mem_offset);

  ge::Status SetInputOffset();

  ge::Status UpdateOpInputOffset(const NodePtr &node) const;

  ge::Status CheckOffset();

 private:
  ///
  /// @ingroup ge_graph
  /// @brief assign memory offset
  /// @return Status result of function
  ///
  ge::Status ReAssignContinuousMemory(bool is_loop_graph);

  ge::Status ReAssignReuseAndNoPaddingContinuousInputMemory();

  ge::Status ReAssignReuseAndNoPaddingContinuousOutputMemory();

  ge::Status CalculateTensorRealSizeAndOutSize(const ge::ConstGeTensorDescPtr &output_desc, int64_t dim_index,
                                               int64_t &output_mem_size, int64_t &batch_dim_num, int64_t &out_size);

  ge::Status ReAssignMergeMemory();

  ge::Status ReAssignAtomicMemory(bool is_loop_graph);

  ge::Status AssignContinuousInputMemory(const ge::NodePtr &node);

  ge::Status AssignContinuousOutputMemory(const ge::NodePtr &node);

  ge::Status AssignReferenceMemory(const ge::NodePtr &node);

  ///
  /// @brief check the input of node whether support atomic attr
  /// @param node
  /// @return true:supported; false:not supported
  ///
  bool CheckInputIsSupportAtomic(const ge::NodePtr &node);

  ge::Status AssignAtomicOutputMemory(const ge::NodePtr &node);

  ge::Status AssignOrdinaryAtomicWorkspaceMemory(const ge::OpDescPtr &op_desc,
                                                 std::map<std::string, std::map<int64_t, int64_t>> &workspace_info);

  ge::Status AssignFusionAtomicWorkspaceMemory(const ge::OpDescPtr &op_desc,
                                               std::map<std::string, std::map<int64_t, int64_t>> &workspace_info);

  ///
  /// @brief set loop graph atomic attr
  /// @param node
  /// @param atomic_mem_start: atomic op memory start address
  ///
  ge::Status SetLoopGraphAtomicAttr(const ge::NodePtr &node, int64_t atomic_mem_start);

  ge::Status SetAtomicCleanAttr(const ge::NodePtr &n, int64_t atomic_mem_start, int64_t atomic_mem_size);

  void AlignMemOffset(const int64_t &mem_align_size);

  ge::Status UpdateOpInputOffset(const NodePtr &node, vector<int64_t> &input_list) const;

  MemoryOffsetList memory_offset_;
  ge::ComputeGraphPtr compute_graph_;
};
}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_
