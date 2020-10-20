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
#include "graph/build/memory/hybrid_mem_assigner.h"

namespace ge {
struct MemoryOffset {
  MemoryOffset(rtMemType_t mem_type, size_t mem_offset) : mem_type_(mem_type), mem_offset_(mem_offset) {}

 public:
  rtMemType_t mem_type_;
  size_t mem_offset_;
};

using MemoryOffsetMap = std::map<int64_t, MemoryOffset>;

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
using BlockMemAssignerPtr = std::shared_ptr<BlockMemAssigner>;
using HybridMemAssignerPtr = std::shared_ptr<HybridMemAssigner>;


class GraphMemoryAssigner {
 public:
  explicit GraphMemoryAssigner(ge::ComputeGraphPtr compute_graph)
      : compute_graph_(std::move(compute_graph)),
        mem_assigner_(nullptr) {}

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

  ge::Status ReAssignMemory(bool is_loop_graph, map<int64_t, size_t> &mem_type_to_offset);

  ge::Status AssignZeroCopyMemory(map<int64_t, size_t> &mem_offset, size_t &zero_mem_copy_size);

  ge::Status SetInputOffset();

  ge::Status UpdateOpInputOffset(const NodePtr &node) const;

  ge::Status CheckOffset();

  ge::Status AssignReferenceMemory();

 private:
  ///
  /// @ingroup ge_graph
  /// @brief assign memory offset
  /// @return Status result of function
  ///
  ge::Status ReAssignContinuousMemory(bool is_loop_graph);

  ge::Status ReAssignReuseAndNoPaddingContinuousInputMemory();

  ge::Status ReAssignReuseAndNoPaddingContinuousOutputMemory();

  ge::Status ReAssignVirtualInputNodeMemory(NodePtr node, size_t &mem_offset_reuse);

  ge::Status ReAssignVirtualOutputNodeMemory(NodePtr node, size_t &mem_offset_reuse);

  ge::Status ReAssignVirtualNodesMemory(map<string, vector<NodePtr>> &mem_reuse_nodes_map, int32_t mem_reuse_model);

  ge::Status GetMaxBatchLabel(const map<string, vector<NodePtr>> &mem_reuse_virtual_nodes_map,
                              int32_t mem_reuse_model, string &max_batch_label);

  ge::Status CalculateTensorRealSizeAndOutSize(const ge::ConstGeTensorDescPtr &output_desc, int64_t dim_index,
                                               int64_t &output_mem_size, int64_t &batch_dim_num, int64_t &out_size);

  ge::Status ReAssignAtomicMemory(bool is_loop_graph);
  
  ge::Status FilterAtomicNodesForMemoryAssign(std::map<NodePtr, vector<NodePtr>> &normal_atomic_nodes_map,
                                              std::vector<NodePtr> &connecting_output_atomic_nodes);

  ge::Status AssignContinuousInputMemory(const ge::NodePtr &node, int64_t &continuous_mem_start,
                                         int64_t &continuous_mem_size, int64_t memory_type);

  ge::Status AssignContinuousOutputMemory(const ge::NodePtr &node);

  ///
  /// @brief check the input of node whether support atomic attr
  /// @param node
  /// @return true:supported; false:not supported
  ///
  bool CheckInputIsSupportAtomic(const ge::NodePtr &node);

  ge::Status GetMemoryAssignmentStatus(const ge::NodePtr &node, int64_t output_index, bool &is_mem_assigned);

  ge::Status AssignAtomicOutputMemory(const ge::NodePtr &node, std::vector<int64_t> &mem_offset_end);

  ge::Status AssignOrdinaryAtomicWorkspaceMemory(const ge::OpDescPtr &op_desc,
                                                 std::map<std::string, std::map<int64_t, int64_t>> &workspace_info,
                                                 std::vector<int64_t> &mem_offset_end);

  ge::Status AssignFusionAtomicWorkspaceMemory(const ge::OpDescPtr &op_desc,
                                               std::map<std::string, std::map<int64_t, int64_t>> &workspace_info,
                                               std::vector<int64_t> &mem_offset_end);

  ge::Status AssignAtomicOutputAndWorkspaceMemory(const ge::NodePtr &node, std::vector<int64_t> &mem_offset_end);

  ge::Status AssignConnectNetOutputAtomicMemory(vector<NodePtr> &connect_netoutput_nodes);

  ge::Status SetIndependentAtomicAttr(const ge::NodePtr &node, int64_t atomic_mem_start,
                                      const std::vector<int64_t> &mem_offset_end);

  ge::Status SetAtomicCleanAttr(const ge::NodePtr &node, const std::vector<int64_t> &atomic_mem_start,
                                const std::vector<int64_t> &atomic_mem_size);

  ge::Status IsIndependentAtomicClean(const ge::NodePtr &node, bool &is_independent_atomic_clean_node);

  void AlignMemOffset(const int64_t &mem_align_size, int64_t memory_type);

  ge::Status UpdateOpInputOffset(const NodePtr &node, vector<int64_t> &input_list) const;

  ge::Status UpdateConstArgsOffset(const NodePtr &node, vector<int64_t> &input_list) const;

  NodePtr GetKnownInputNode(const NodePtr &node) const;

  ge::Status GetNodeMemoryType(const NodePtr &node, int64_t &memory_type, string input_or_output);
  ge::Status GetNodeListMemoryType(const vector<NodePtr> &nodes, int32_t mem_reuse_model, int64_t &memory_type);

  bool CheckContinuousMemType(vector<int64_t> mem_type_list);

  void PrintMemoryOffset();

  MemoryOffsetMap memory_offset_;
  ge::ComputeGraphPtr compute_graph_;
  HybridMemAssignerPtr mem_assigner_;
};
}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_
