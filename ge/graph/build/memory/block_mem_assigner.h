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

#ifndef GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <list>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/build/memory/mem_assigner.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"

namespace ge {
const size_t kMaxLifeTime = 0xffffffff;
const int32_t kInvalidThreadScopeId = -1;
const uint64_t kSessionScopeMemory = 0x100000000;
const uint64_t kMemoryTypeMask = 0xffffffff;

enum MemoryNoReuseScope { kReuse, kSessionNoReuse, kGraphNoReuse };

using DependStreamLife = std::map<int64_t, std::map<int64_t, size_t>>;

enum OpMemoryType { kOutput, kWorkspace };

struct NodeTypeIndex {
  NodeTypeIndex(ge::NodePtr node, OpMemoryType mem_type, uint32_t index, bool ref_input = false, size_t begin = 0,
                int32_t thread_scope_id = kInvalidThreadScopeId)
      : node(std::move(node)), mem_type(mem_type), index(index), ref_input(ref_input), life_time_begin(begin),
        thread_scope_id(thread_scope_id) {}

  ge::NodePtr node = nullptr;
  OpMemoryType mem_type = kOutput;
  uint32_t index = 0;
  bool ref_input = false;
  size_t life_time_begin = 0;
  size_t life_time_end = kMaxLifeTime;
  int32_t thread_scope_id = kInvalidThreadScopeId;
  const string GetMemType() const {
    if (mem_type == kOutput) {
      return "output";
    } else if (mem_type == kWorkspace) {
      return "workspace";
    }
    return "unknown";
  }

  size_t GetLifeBegin() const {
    if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
      return 0;
    }

    if ((life_time_begin > 0) && (life_time_begin < static_cast<size_t>(node->GetOpDesc()->GetId()))) {
      return life_time_begin;
    } else {
      return node->GetOpDesc()->GetId();
    }
  }

  std::string GetLifeBeginDesc() const {
    if (node == nullptr) {
      return "";
    }
    auto node_op_desc = node->GetOpDesc();
    if (node_op_desc != nullptr) {
      auto life_begin = GetLifeBegin();
      if (life_begin != static_cast<size_t>(node_op_desc->GetId())) {
        return std::to_string(life_begin) + "-" + std::to_string(node_op_desc->GetId());
      } else {
        return std::to_string(node_op_desc->GetId());
      }
    }
    return "";
  }
};

class MemoryBlock {
 public:
  explicit MemoryBlock(size_t block_size, int64_t stream_id = 0, bool reuse_mem = true,
                       int64_t memory_type = RT_MEMORY_HBM)
      : ref_count_(0),
        stream_id_(stream_id),
        deleted_block_(false),
        reuse_mem_(reuse_mem),
        same_stream_(true),
        input_index_(0),
        continuous_block_(false),
        first_continuous_block_(false),
        last_continuous_block_(false),
        is_zero_copy_(false),
        memory_type_(memory_type),
        block_size_(block_size),
        head_offset_(0),
        tail_offset_(0),
        child_offset_(0) {}

  MemoryBlock(const MemoryBlock &) = delete;

  MemoryBlock &operator=(const MemoryBlock &) = delete;

  ~MemoryBlock() {
    node_type_index_list_.clear();
    symbol_list_.clear();
  }

  size_t Size() const { return block_size_; }

  void SetSize(size_t size) {
    if (size > block_size_) {
      block_size_ = size;
    }
  }

  size_t AlignSize() const;

  void SetHeadOffset(size_t offset);

  void SetTailOffset(size_t offset);

  size_t HeadOffset() const { return head_offset_; }

  size_t TailOffset() const { return tail_offset_; }

  void AddNodeTypeIndex(const NodeTypeIndex &node_type_index, size_t real_size, size_t no_align_size) {
    node_type_index_list_.emplace_back(node_type_index);
    real_size_list_.emplace_back(real_size);
    no_align_size_list_.emplace_back(no_align_size);
    if ((node_type_index.node != nullptr) && (node_type_index.node->GetOpDesc() != nullptr)) {
      auto stream_id = node_type_index.node->GetOpDesc()->GetStreamId();
      if (stream_id != stream_id_) {
        same_stream_ = false;
      }
    }
    if (node_type_index.thread_scope_id != kInvalidThreadScopeId) {
      thread_scope_id_.insert(node_type_index.thread_scope_id);
    }
  }

  void AddSymbol(const std::string &symbol) {
    symbol_list_.emplace_back(symbol);
  }

  const std::vector<NodeTypeIndex> &NodeTypeIndexList() const { return node_type_index_list_; }
  const std::vector<std::string> &SymbolList() const { return symbol_list_; }
  const std::vector<size_t> &RealSizeList() const { return real_size_list_; }
  const std::vector<MemoryBlock *> &ChildBlockList() const { return child_blocks_; }
  const std::vector<size_t> &NoAlignSizeList() const { return no_align_size_list_; }
  const std::set<int32_t> &ThreadScopeId() const { return thread_scope_id_; }

  void Resize();

  std::string String();

  bool IsSameBatchLabel();

  void AddContinuousLifeReuseBlock(MemoryBlock *block, DependStreamLife &total_node_depend_stream_life);

  void AddLifeReuseBlock(MemoryBlock *block, DependStreamLife &node_depend_stream_life);

  void SetLifeTimeEnd(size_t time);

  size_t GetLifeBegin();

  size_t GetLifeEnd() const;

  void AddDependLifeBegin(DependStreamLife &node_depend_stream_life);

  size_t GetDependLifeBegin(int64_t stream_id, DependStreamLife &node_depend_stream_life);

  bool CanReuse(int32_t thread_scope_id) const;

  int ref_count_;
  int64_t stream_id_;
  bool deleted_block_;
  bool reuse_mem_;
  bool same_stream_;
  uint32_t input_index_;
  bool continuous_block_;
  bool first_continuous_block_;
  bool last_continuous_block_;
  bool is_zero_copy_;
  std::map<int64_t, size_t> depend_stream_life_;
  int64_t memory_type_;
  std::string batch_label_;
 private:
  size_t block_size_;
  std::vector<size_t> real_size_list_;
  std::vector<size_t> no_align_size_list_;
  size_t head_offset_;
  size_t tail_offset_;
  size_t child_offset_;
  std::vector<NodeTypeIndex> node_type_index_list_;
  std::vector<std::string> symbol_list_;
  std::vector<MemoryBlock *> child_blocks_;
  std::set<int32_t> thread_scope_id_;
};

class BlockMemAssigner : public MemAssigner {
 public:
  BlockMemAssigner(ComputeGraphPtr compute_graph, const std::map<std::string, std::string> &anchor_to_symbol,
                   const std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors);

  BlockMemAssigner(const BlockMemAssigner &) = delete;

  BlockMemAssigner &operator=(const BlockMemAssigner &) = delete;

  ~BlockMemAssigner() override;

  Status Assign() override;

  const std::map<uint64_t, size_t> &GetMemOffsets() const { return mem_offsets_; }

  int64_t GetAtomicAddrCleanId() const { return atomic_addr_clean_id_; }

  std::vector<MemoryBlock *> GetMemoryBlocks() const { return memory_blocks_; }

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

  void SetOpMemOffset(bool is_zero_copy);

  std::string GetMaxBatchLabel() const { return max_batch_label_; }
 protected:
  ///
  /// @ingroup domi
  /// @brief traverse all memory size, resize, and calculate offset
  /// @param [in&out] memory_blocks memory size, resize and calculate memory address after offset
  ///
  void ResizeMemoryBlocks();

  void GetOutAndWorkSpaceMem(std::vector<int64_t> &all_memory_size);

  void GetNodeWorkSpaceSize(const ge::NodePtr &node, std::vector<int64_t> &workspace_memory, int64_t &total_size);

  ///
  /// @ingroup GE
  /// @brief Determine whether it is the type of zero memory node.
  /// @param [in] node type.
  /// @return bool true: is zero memory node; false: is not zero memory node
  /// @author
  ///
  bool CheckIsZeroMemNodeType(const std::string &node_type) const;

  ///
  /// @ingroup GE
  /// @brief Check pre_reuse flag & post_reuse glag for each symbol
  /// @return void
  ///
  void InitReuseFlag();

  ///
  /// @ingroup GE
  /// @brief get pre_reuse flag
  /// @param [in] node
  /// @param [in] out_index
  /// @return bool
  ///
  bool IsPreReuse(const NodePtr &node, uint32_t out_index) const;

  ///
  /// @ingroup GE
  /// @brief get post_reuse flag
  /// @param [in] mem_block
  /// @return bool
  ///
  bool IsPostReuse(const MemoryBlock *mem_block) const;

  ///
  /// @ingroup GE
  /// @brief check if symbol of cur node_index_io has block
  /// @param [in] node_index_io
  /// @param [out] symbol
  /// @return bool
  ///
  bool IsSymbolExist(const NodeIndexIO &node_index_io, std::string &symbol);

  ///
  /// @ingroup GE
  /// @brief Print symbol
  /// @return void
  ///
  void PrintSymbolMap();

  ///
  /// @ingroup GE
  /// @brief Get the memory type corresponding to the current symbol.
  /// @param [in] node_index_io_list
  /// @param [out] memory_type
  /// @return void
  ///
  void GetSymbolMemType(std::list<NodeIndexIO> node_index_io_list, int64_t &memory_type);

  ///
  /// @ingroup GE
  /// @brief Update input tensor or output tensor of op to new memory type attr.
  /// @param [in] node_index_io_list
  /// @param [in] memory_type
  /// @return void
  ///
  void UpdateOpTensorMemType(std::list<NodeIndexIO> node_index_io_list, int64_t memory_type);

  std::map<uint64_t, size_t> mem_offsets_;
  ge::ComputeGraphPtr compute_graph_;
  std::vector<MemoryBlock *> memory_blocks_;
  std::vector<MemoryBlock *> blocks_store_;
  std::vector<NodeTypeIndex> zero_memory_list_;

  // ref mapping
  const std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors_;
  const std::map<std::string, std::string> &anchor_to_symbol_;
  std::map<std::string, bool> pre_reuse_flag_;
  std::map<std::string, bool> post_reuse_flag_;
  std::map<std::string, size_t> symbol_size_;
  std::map<std::string, int64_t> symbol_to_mem_type_;

 private:
  ///
  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for output memory while considering reuse
  /// @param [in] n: node in compute_graph_
  /// @param [in] index: output node index
  /// @param [in] ranges: available memory specifications
  /// @param [in] is_op_reuse_mem: Whether the op reuses the memory, true: reuse; false: not reuse
  /// @param [in] continuous: Whether the op uses continuous memory
  /// @return MemoryBlock*
  /// @author
  ///
  MemoryBlock *ApplyOutMemory(const ge::NodePtr &n, uint32_t index, const std::vector<int64_t> &ranges,
                              const bool is_op_reuse_mem, const bool continuous);

  Status AssignOutputMemoryWithReuse(const NodePtr &node, vector<int64_t> &ranges);
  ///
  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for memory while considering reuse
  /// @param [in] block_size applied memory block size
  /// @param [in] real_size actual memory size required
  /// @param [in] type output or workspace
  /// @param [in] n node in compute_graph_
  /// @param [in] out_index output node index
  /// @param [in] workspace_reuse_flag reuse flag for workspace
  /// @param [in] is_op_reuse_mem whether the op reuses memory
  /// @param [in] continuous whether the memory of op is continuous
  /// @param [in] memory_type device memory type
  /// @return MemoryBlock*
  /// @author
  ///
  MemoryBlock *ApplyMemory(size_t block_size, size_t real_size, size_t no_align_size, OpMemoryType mem_type,
                           const ge::NodePtr &n, uint32_t out_index, const std::vector<bool> &workspace_reuse_flag,
                           const bool is_op_reuse_mem, const bool continuous, uint64_t memory_type);

  ///
  /// @ingroup GE
  /// @brief check workspace_reuse_flag to judge if add workspace block wait reuse
  /// @param [in] workspace_reuse_flag mark out index if support resue
  /// @param [in] index out index
  /// @param [in] stream_id which stream op in
  /// @param [in] mem_block node workspace mem_block
  /// @param [in] memory_type workspace memory type
  /// @return void
  /// @author
  ///
  void CheckWorkspaceReuse(const vector<bool> &workspace_reuse_flag, uint32_t index, int64_t stream_id,
                           MemoryBlock *mem_block, uint64_t memory_type);

  ///
  /// @ingroup GE
  /// @brief Release memory block to reusable list
  /// @param [in] to_release memory block to be released
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  ///
  void ReleaseMemory(MemoryBlock *to_release, vector<MemoryBlock *> &reusable_memory, bool same_stream = true);

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
  void ReleaseInputNodeOutMemory(const std::unordered_map<string, vector<MemoryBlock *>> &node_out_blocks,
                                 vector<MemoryBlock *> &reusable_memory, ge::NodePtr &n);

  ///
  /// @ingroup GE
  /// @brief Resize memory blocks for each batchs
  /// @return merge or not
  /// @author
  ///
  void ResizeDynamicBatchBlocks();

  void AssignContinuousBlocks();

  bool IsZeroCopyBlock(const NodePtr &node, bool continuous);

  bool IsOutNodeSetContinuousInput(const NodePtr &n, uint32_t out_index, std::string &peer_name,
                                   uint32_t &peer_input_index, bool &no_need_assign_memory, bool &reset_zero_copy_flag);

  bool IsContinuousMemoryReuse(const NodePtr &n, const NodePtr &peer_node, uint32_t out_index);
  ///
  /// @ingroup GE
  /// @|+++++++++block1++++++++|                               |+++++++++block1++++++++|
  /// @|+++++++++block1++++++++||++block2++|                   |+++++++++block1++++++++||++block2++|
  /// @                         |++block2++||++block3++|  ==>  |++block3++|             |++block2++|
  /// @                                     |++block3++|       |++block3++|
  /// @return void
  /// @author
  ///
  void ReuseBlocksByLifeTime(size_t range_size);

  bool IsContinuousOutput(const NodePtr &n);

  bool GetWorkSpaceMemoryType(const NodePtr &node, size_t index, uint64_t &memory_type,
                              vector<bool> &workspace_reuse_flag);

  void ContinuousOutRefCheck(bool &isAllOutputRef, bool &isOutputHasRef, const NodePtr &n);

  Status ApplyContinuousMemory(const NodePtr &n, const vector<int64_t> &ranges, const bool is_op_reuse_mem);

  void MarkContinuousAllocedForOneInputFromVariable(const NodePtr &node);

  void CheckAndReleaseSuspendedBlock(const NodePtr &node, uint32_t idx, MemoryBlock *block);

  std::unordered_map<int64_t, std::unordered_map<int64_t, std::vector<MemoryBlock *>>> reusable_blocks_;

  std::unordered_map<int64_t, std::unordered_map<int64_t, std::vector<MemoryBlock *>>> stream_workspace_blocks_;

  std::unordered_map<std::string, std::vector<MemoryBlock *>> node_out_blocks_;

  std::unordered_map<std::string, MemoryBlock *> symbol_blocks_;

  std::unordered_map<std::string, std::unordered_map<uint32_t, MemoryBlock *>> node_continuous_input_blocks_;

  std::map<std::string, uint32_t> node_continuous_input_counts_;

  // reuse memory
  vector<string> op_no_reuse_mem_vec_;

  bool op_reuse_env_valid_ = false;

  std::string ge_disable_reuse_mem_env_ = "0";

  bool is_op_reuse_mem_ = true;

  size_t life_time_;

  int64_t atomic_addr_clean_id_ = 0;

  size_t theory_min_memory_size_ = 0;

  size_t theory_memory_size_ = 0;

  std::string max_batch_label_;

  size_t continuous_life_begin_ = 0;
  ///
  /// @          [stream1][nodeid]
  /// @[nodeid]  [stream2][nodeid]
  /// @          [stream2][nodeid]
  ///
  DependStreamLife total_node_depend_stream_life_;

  bool root_unknown_shape_flag_ = false;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_
