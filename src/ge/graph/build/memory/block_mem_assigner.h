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

#ifndef GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/build/memory/mem_assigner.h"
#include "graph/compute_graph.h"

namespace ge {
enum MemoryType { kOutput, kWorkspace };

struct NodeTypeIndex {
  NodeTypeIndex(ge::NodePtr node, MemoryType mem_type, uint32_t index)
      : node_(std::move(node)), mem_type_(mem_type), index_(index) {}

  ge::NodePtr node_ = nullptr;
  MemoryType mem_type_ = kOutput;
  uint32_t index_ = 0;
};

class MemoryBlock {
 public:
  explicit MemoryBlock(size_t block_size)
      : ref_count_(0),
        stream_id_(0),
        deleted_block_(false),
        block_size_(block_size),
        head_offset_(0),
        tail_offset_(0) {}

  MemoryBlock(const MemoryBlock &) = delete;

  MemoryBlock &operator=(const MemoryBlock &) = delete;

  ~MemoryBlock() { node_type_index_list_.clear(); }

  void Init(size_t real_size, MemoryType type, const ge::NodePtr &node, uint32_t out_index) {
    real_size_list_.emplace_back(real_size);
    node_type_index_list_.emplace_back(node, type, out_index);
  }
  size_t Size() const { return block_size_; }

  void SetHeadOffset(size_t offset) { head_offset_ = offset; }

  void SetTailOffset(size_t offset) { tail_offset_ = offset; }

  size_t HeadOffset() const { return head_offset_; }

  size_t TailOffset() const { return tail_offset_; }

  void AddNodeTypeIndex(const NodeTypeIndex &node_type_index, size_t real_size) {
    node_type_index_list_.emplace_back(node_type_index);
    real_size_list_.emplace_back(real_size);
  }

  const std::vector<NodeTypeIndex> &NodeTypeIndexList() const { return node_type_index_list_; }
  const std::vector<size_t> &RealSizeList() const { return real_size_list_; }

  void Resize();

  std::string String();

  bool IsSameLabel(std::string &first_batch_label);

  int ref_count_;
  int64_t stream_id_;
  bool deleted_block_;

 private:
  size_t block_size_;
  std::vector<size_t> real_size_list_;
  size_t head_offset_;
  size_t tail_offset_;
  std::vector<NodeTypeIndex> node_type_index_list_;
};

class BlockMemAssigner : public MemAssigner {
 public:
  explicit BlockMemAssigner(ge::ComputeGraphPtr compute_graph);

  BlockMemAssigner(const BlockMemAssigner &) = delete;

  BlockMemAssigner &operator=(const BlockMemAssigner &) = delete;

  ~BlockMemAssigner() override;

  Status Assign() override;

  size_t GetMemOffset() const { return mem_offset_; }

  ///
  /// @ingroup domi
  /// @brief   memory size fixed for reuse. get memory range
  /// @param [out] ranges return memory range
  /// @return Status result
  ///
  virtual Status GetMemoryRanges(std::vector<int64_t> &ranges) = 0;
  ///
  /// @ingroup domi
  /// @brief traverse all nodes' outputs and needed workspace mem, apply memory, consider reuse memory
  /// @param [in] ranges memory range provided
  /// @author
  ///
  void AssignMemoryWithReuse(std::vector<int64_t> &ranges);

  void SetOpMemOffset();

 protected:
  ///
  /// @ingroup domi
  /// @brief traverse all memory size, resize, and calculate offset
  /// @param [in&out] memory_blocks memory size, resize and calculate memory address after offset
  ///
  void ResizeMemoryBlocks();

  void GetOutAndWorkSpaceMem(std::vector<int64_t> &all_memory_size);

  void GetNodeWorkSpaceSize(const ge::NodePtr &node, std::vector<int64_t> &workspace_memory);

  ///
  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to find the reuse relationship between streams
  /// @param [in] reusable_stream_map map to save stream_id and its reusable stream_ids
  /// @return void
  /// @author
  ///
  void InitReusableStreamMap();

  ///
  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to find the first and last nodeptr of a stream.
  /// @param [in] stream_head_tail_node_map map to save stream_id and its first and last nodeptr.
  /// @param [in] stream_mem_map map to save stream_id and its memory capacity.
  /// @return void
  /// @author
  ///
  void FindHeadAndTailNodesForStream(std::map<int64_t, std::pair<NodePtr, NodePtr>> &stream_head_tail_node_map,
                                     std::unordered_map<int64_t, int64_t> &stream_mem_map);

  ///
  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to find the reuse relationship between streams.
  /// @param [in] stream_head_tail_node_map map to save stream_id and its first and last nodeptr.
  /// @param [in] stream_dependency_map map to save stream_id and stream_ids depends on it.
  /// @return void
  /// @author
  ///
  void FindDependantStream(std::map<int64_t, std::pair<NodePtr, NodePtr>> &stream_head_tail_node_map,
                           std::map<int64_t, std::unordered_set<int64_t>> &stream_dependency_map);

  ///
  /// @ingroup GE
  /// @brief Determine whether it is the type of zero memory node.
  /// @param [in] node type.
  /// @return bool true: is zero memory node; false: is not zero memory node
  /// @author
  ///
  bool CheckIsZeroMemNodeType(const std::string &node_type) const;

  size_t mem_offset_;

  ge::ComputeGraphPtr compute_graph_;

  std::vector<MemoryBlock *> memory_blocks_;

  std::vector<NodeTypeIndex> zero_memory_list_;

 private:
  ///
  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for output memory while considering reuse
  /// @param [in] n node in compute_graph_
  /// @param [in] index output node index
  /// @param [in] ranges available memory specifications
  /// @return MemoryBlock*
  /// @author
  ///
  MemoryBlock *ApplyOutMemory(const ge::NodePtr &n, uint32_t index, const std::vector<int64_t> &ranges);

  ///
  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for memory while considering reuse
  /// @param [in] block_size applied memory block size
  /// @param [in] real_size actual memory size required
  /// @param [in] type output or workspace
  /// @param [in] n node in compute_graph_
  /// @param [in] out_index output node index
  /// @param [in] workspace_reuse_flag reuse flag for workspace
  /// @return MemoryBlock*
  /// @author
  ///
  MemoryBlock *ApplyMemory(size_t block_size, size_t real_size, MemoryType mem_type, const ge::NodePtr &n,
                           uint32_t out_index, const std::vector<bool> &workspace_reuse_flag);

  ///
  /// @ingroup GE
  /// @brief Release memory block to reusable list
  /// @param [in] to_release memory block to be released
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  ///
  void ReleaseMemory(MemoryBlock *to_release, vector<MemoryBlock *> &reusable_memory);

  ///
  /// @ingroup GE
  /// @brief Release memory blocks to reusable list
  /// @param [in] to_releases memory blocks to be released
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  ///
  void ReleaseMemorys(const vector<MemoryBlock *> &to_releases, vector<MemoryBlock *> &reusable_memory);

  ///
  /// @ingroup GE
  /// @brief Release memory block to reusable list
  /// @param [in] n node in compute_graph_
  /// @param [in] node_out_blocks output memory blocks for ops
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  ///
  void ReleaseInputNodeOutMemory(const ge::NodePtr &n,
                                 const std::unordered_map<string, vector<MemoryBlock *>> &node_out_blocks,
                                 vector<MemoryBlock *> &reusable_memory);

  ///
  /// @ingroup GE
  /// @brief Merge memory blocks between different batchs
  /// @return void
  /// @author
  ///
  void MergeDynamicBatchBlocks();

  std::vector<MemoryBlock *> reusable_blocks_;

  std::map<std::string, uint64_t> reusable_block_counts_;

  std::unordered_map<int64_t, std::vector<MemoryBlock *>> stream_workspace_blocks_;

  std::unordered_map<std::string, std::vector<MemoryBlock *>> node_out_blocks_;

  // save stream_id and reusable stream_ids
  std::unordered_map<int64_t, std::unordered_set<int64_t>> reusable_streams_map_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_
