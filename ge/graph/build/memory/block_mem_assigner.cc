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

#include "graph/build/memory/block_mem_assigner.h"
#include <algorithm>
#include <sstream>

#include "external/ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "graph/buffer.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_context.h"
#include "graph/types.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

#include "graph/debug/ge_attr_define.h"

#include "graph/common/local_context.h"
#include "graph/optimize/common/params.h"
#include "omg/omg_inner_types.h"
#include "runtime/mem.h"

using std::map;
using std::set;
using std::list;
using std::pair;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::unordered_set;
using std::vector;

namespace {
const char *const kAttrNameWorkspaceReuseFlag = "workspace_reuse_flag";
const char *const kL2FusionDynamicConvergeOp = "l2fusion_dynamic_converge_op";
const char *const kOpNoReuseMem = "no_reuse_mem_flag";
const char *const OP_NO_REUSE_MEM = "OP_NO_REUSE_MEM";
const int kReuseMaxOpNum = 10;
const int kReuseMaxCharNum = 2000;
}  // namespace

namespace ge {
void AlignMemOffset(size_t &mem_align_size) {
  if (mem_align_size <= 0) {
    return;
  }
  mem_align_size = (mem_align_size + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
}

static bool CompareLifeTime(const NodeTypeIndex &left, const NodeTypeIndex &right) {
  if (left.GetLifeBegin() < right.GetLifeBegin()) {
    return true;
  }
  return false;
}

void GetLifeList(const MemoryBlock &block, std::vector<NodeTypeIndex> &life_list, bool child) {
  for (auto &node : block.NodeTypeIndexList()) {
    life_list.emplace_back(node);
  }

  if (child) {
    for (auto child_block : block.ChildBlockList()) {
      if (child_block == nullptr) {
        continue;
      }
      if (block.stream_id_ != child_block->stream_id_ || !block.same_stream_ || !child_block->same_stream_) {
        life_list.clear();
        return;
      }
      GetLifeList(*child_block, life_list, child);
    }
  }
}

bool CrossLifeTime(const NodeTypeIndex &left, const NodeTypeIndex &right) {
  if ((left.node == nullptr) || (right.node == nullptr)) {
    return true;
  }
  auto left_node_op_desc = left.node->GetOpDesc();
  auto right_node_op_desc = right.node->GetOpDesc();
  if ((left_node_op_desc != nullptr) && (right_node_op_desc != nullptr)) {
    if (left.GetLifeBegin() < right.GetLifeBegin()) {
      if (left.life_time_end >= right.GetLifeBegin()) {
        return true;
      }
    } else if (left.GetLifeBegin() == right.GetLifeBegin()) {
      return true;
    } else {
      if (right.life_time_end >= left.GetLifeBegin()) {
        return true;
      }
    }
  }
  return false;
}

///
/// When child block's life time are not cross with parent block, they can be reused(only same stream).
/// |-----------------------------parent block---------------------|
/// |------child block1--------------||------child block2------|
/// |--child block1-1-|
///
bool CanIntervalLifeReuse(MemoryBlock &parent_block, MemoryBlock &child_block) {
  // judge by interval life time, only same stream can be judged by interval life time
  if (parent_block.stream_id_ != child_block.stream_id_ || !parent_block.same_stream_ || !child_block.same_stream_
      || parent_block.NodeTypeIndexList().empty() || child_block.NodeTypeIndexList().empty()) {
    return false;
  }

  // quick judge by front and back node
  if (CrossLifeTime(parent_block.NodeTypeIndexList().front(), child_block.NodeTypeIndexList().front())) {
    return false;
  }
  if (CrossLifeTime(parent_block.NodeTypeIndexList().back(), child_block.NodeTypeIndexList().back())) {
    return false;
  }

  std::vector<NodeTypeIndex> life_list;
  GetLifeList(parent_block, life_list, false);
  GetLifeList(child_block, life_list, true);
  if (life_list.empty()) {
    return false;
  }
  std::sort(life_list.begin(), life_list.end(), CompareLifeTime);
  size_t pre_life_end = 0;
  for (auto &node : life_list) {
    auto node_op_desc = node.node->GetOpDesc();
    if (node_op_desc != nullptr && pre_life_end >= static_cast<size_t>(node_op_desc->GetId())) {
      // life time cross
      return false;
    }
    pre_life_end = node.life_time_end;
  }
  GELOGI("Block size[%zu, %zu] life time are not cross.", parent_block.Size(), child_block.Size());
  return true;
}

void MemoryBlock::SetHeadOffset(size_t offset) {
  head_offset_ = offset;
  size_t child_offset = head_offset_;
  for (auto block : child_blocks_) {
    if (block != nullptr) {
      block->SetHeadOffset(child_offset);
      child_offset += block->Size();
    }
  }
}

void MemoryBlock::SetTailOffset(size_t offset) {
  tail_offset_ = offset;
  size_t child_offset = head_offset_;
  for (auto block : child_blocks_) {
    if (block != nullptr) {
      child_offset += block->Size();
      block->SetTailOffset(child_offset - 1);
    }
  }
}

void MemoryBlock::Resize() {
  size_t child_block_size = 0;
  for (auto block : child_blocks_) {
    if (block != nullptr) {
      block->Resize();
      child_block_size += block->Size();
    }
  }
  auto iter = std::max_element(real_size_list_.begin(), real_size_list_.end());
  if (iter == real_size_list_.end()) {
    GELOGW("real_size_list_ is empty");
    return;
  } else {
    size_t block_size = (child_block_size > *iter) ? child_block_size : *iter;
    if ((block_size > 0) && (block_size % MEM_ALIGN_SIZE != 0)) {
      AlignMemOffset(block_size);
    }
    block_size_ = block_size;
    if (last_continuous_block_) {
      block_size_ += MEM_ALIGN_SIZE;
    }
  }
}

size_t MemoryBlock::AlignSize() const {
  size_t align_block_size = 0;
  auto iter = std::max_element(real_size_list_.begin(), real_size_list_.end());
  if (iter == real_size_list_.end()) {
    GELOGW("real_size_list_ is empty");
  } else {
    align_block_size = *iter;
    if ((align_block_size > 0) && (align_block_size % MEM_ALIGN_SIZE != 0)) {
      AlignMemOffset(align_block_size);
    }
  }
  return align_block_size;
}

bool MemoryBlock::IsSameBatchLabel() {
  // only same batch label can reuse
  if (batch_label_.empty() || node_type_index_list_.empty()) {
    return false;
  }

  bool all_same_label = true;
  for (size_t index = 1; index < node_type_index_list_.size(); ++index) {
    if (node_type_index_list_[index].node == nullptr) {
      continue;
    }
    std::string batch_label;
    auto index_op_desc = node_type_index_list_[index].node->GetOpDesc();
    GE_IF_BOOL_EXEC(index_op_desc == nullptr, continue);
    // not all op has ATTR_NAME_BATCH_LABEL, no need check return value, only check out parameter
    (void)ge::AttrUtils::GetStr(index_op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
    if (batch_label_ != batch_label) {
      all_same_label = false;
      break;
    }
  }
  return all_same_label;
}

bool MemoryBlock::CanReuse(int32_t thread_scope_id) const {
  return (thread_scope_id_.find(thread_scope_id) == thread_scope_id_.end());
}

bool CanNotLifeReuse(MemoryBlock *block) {
  if ((block == nullptr) || !block->reuse_mem_ || block->deleted_block_) {
    return true;
  }
  return false;
}

void MemoryBlock::AddContinuousLifeReuseBlock(MemoryBlock *block, DependStreamLife &total_node_depend_stream_life) {
  // continuous memory case:only real_size is maximum can be reused and only one continuous memory in one block
  auto it_block = std::max_element(std::begin(block->NoAlignSizeList()), std::end(block->NoAlignSizeList()));
  auto it_this = std::max_element(std::begin(NoAlignSizeList()), std::end(NoAlignSizeList()));
  if (it_block != std::end(block->NoAlignSizeList()) && it_this != std::end(NoAlignSizeList())) {
    if ((continuous_block_ && block->continuous_block_) ||
        (continuous_block_ && (*it_this < *it_block)) || (block->continuous_block_ && (*it_this > *it_block))) {
      GELOGD("Conflict current block size:%zu continuous:%d, reuse block max size:%zu continuous:%d",
          *it_this, continuous_block_, *it_block, block->continuous_block_);
      return;
    }
  }

  MemoryBlock *parent = nullptr;
  MemoryBlock *child = nullptr;
  // merge small block to large block
  if (block->GetDependLifeBegin(stream_id_, total_node_depend_stream_life) > GetLifeEnd()) {
    if ((block->child_offset_ + AlignSize()) <= *it_block) {
      parent = block;
      child = this;
    }
  }
  if ((parent != nullptr) && (child != nullptr) && child->child_blocks_.empty()) {
    parent->child_blocks_.emplace_back(child);
    parent->child_offset_ += child->AlignSize();
    child->deleted_block_ = true;
    GELOGI("Add continuous block[%p size:%zu, stream id:%ld life time[begin:%zu, end:%zu]] to"
           " block[%p size:%zu, stream id:%ld, life time[begin:%zu, end:%zu]]", child, child->block_size_,
           child->stream_id_, child->GetLifeBegin(), child->GetLifeEnd(), parent, parent->block_size_,
           parent->stream_id_, parent->GetLifeBegin(), parent->GetLifeEnd());
  }
}

void MemoryBlock::AddLifeReuseBlock(MemoryBlock *block, DependStreamLife &total_node_depend_stream_life) {
  if (CanNotLifeReuse(this) || CanNotLifeReuse(block) || (batch_label_ != block->batch_label_)) {
    return;
  }

  // not same thread scode id can reuse
  for (auto thread_scope_id : ThreadScopeId()) {
    if (!block->CanReuse(thread_scope_id)) {
      return;
    }
  }

  if (block->continuous_block_) {
    AddContinuousLifeReuseBlock(block, total_node_depend_stream_life);
    return;
  }
  MemoryBlock *parent = nullptr;
  MemoryBlock *child = nullptr;
  // merge small block to large block
  // noalign size         802816 + 802816 = 1605632       can reuse
  // after 32 align size  802848 + 802848 > 1605664       can't reuse
  // after 512 align size 803328 + 803328 > 1606144       can't reuse
  // so                   803328 + 803328 = 1606144 + 512 can reuse
  if ((child_offset_ + block->AlignSize()) <= (AlignSize() + MEM_ALIGN_SIZE)) {
    parent = this;
    child = block;
  } else if ((block->child_offset_ + AlignSize()) <= (block->AlignSize() + MEM_ALIGN_SIZE)) {
    parent = block;
    child = this;
  }

  if ((parent != nullptr) && (child != nullptr)) {
    // Different streams must use stream dependency to judge the life cycle
    // In case same stream if it has child block, can judge all the child block's life time in CanIntervalLifeReuse
    bool can_block_life_reuse = (child->child_blocks_.empty()
        && (block->GetDependLifeBegin(stream_id_, total_node_depend_stream_life) > GetLifeEnd()));
    if (!can_block_life_reuse && !CanIntervalLifeReuse(*parent, *child)) {
      return;
    }

    parent->child_blocks_.emplace_back(child);
    parent->child_offset_ += child->AlignSize();
    child->deleted_block_ = true;
    GELOGI("Add block[%p size:%zu, stream id:%ld life time[begin:%zu, end:%zu]] to"
           " block[%p size:%zu, stream id:%ld, life time[begin:%zu, end:%zu]]", child, child->block_size_,
           child->stream_id_, child->GetLifeBegin(), child->GetLifeEnd(), parent, parent->block_size_,
           parent->stream_id_, parent->GetLifeBegin(), parent->GetLifeEnd());
  }
}

size_t MemoryBlock::GetLifeBegin() {
  size_t life_time = 0;
  if (!node_type_index_list_.empty()) {
      life_time = node_type_index_list_.front().GetLifeBegin();
  }
  return life_time;
}

/// |-stream 1-|   |-stream 2-|
/// |--block1--|   |--block---|
/// |--block2--|   |--block---|
/// |--block3--|\  |--block---|
/// |--block---| \ |--block---|
/// |--block---|  \|--block---|
/// |--block---|   |--block7--|
/// |--block---|   |--block---|
/// block7's first node's input node's life begin > block2's life end, block7 can reuse block1~block2
size_t MemoryBlock::GetDependLifeBegin(int64_t stream_id, DependStreamLife &total_node_depend_stream_life) {
  AddDependLifeBegin(total_node_depend_stream_life);
  auto it = depend_stream_life_.find(stream_id);
  if (it == depend_stream_life_.end()) {
    return 0;
  }
  return it->second;
}

void AddDependLife(const ge::NodePtr &org_node, const ge::NodePtr &node, int64_t stream_id,
                   std::map<int64_t, size_t> &depend_stream_life, DependStreamLife &total_node_depend_stream_life) {
  GE_CHECK_NOTNULL_EXEC(node, return);
  GE_CHECK_NOTNULL_EXEC(org_node, return);
  auto node_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(node_desc, return);
  auto node_id = node_desc->GetId();
  auto stream_life = total_node_depend_stream_life.find(node_id);
  if (stream_life != total_node_depend_stream_life.end()) {
    for (auto &it : stream_life->second) {
      if (depend_stream_life.find(it.first) == depend_stream_life.end()) {
        depend_stream_life[it.first] = it.second;
      }
    }
    return;
  }

  for (const auto &in_anchor : node->GetAllInAnchors()) {
    GE_CHECK_NOTNULL_EXEC(in_anchor, continue);
    for (auto peer_out_anchor : in_anchor->GetPeerAnchors()) {
      GE_CHECK_NOTNULL_EXEC(peer_out_anchor, continue);
      auto peer_node = peer_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL_EXEC(peer_node, continue);
      auto peer_node_desc = peer_node->GetOpDesc();
      GE_CHECK_NOTNULL_EXEC(peer_node_desc, continue);
      auto peer_node_stream_id = peer_node_desc->GetStreamId();
      if (peer_node_stream_id < 0) {
        continue;
      }
      size_t peer_node_life_time = peer_node_desc->GetId();
      auto it = depend_stream_life.find(peer_node_stream_id);
      if (it == depend_stream_life.end() || peer_node_life_time > it->second) {
        depend_stream_life[peer_node_stream_id] = peer_node_life_time;
        if (peer_node_stream_id != stream_id) {
          GELOGI("Node:%s stream id:%ld depend node:%s stream id:%ld index[%d] life time[%zu].",
                 org_node->GetName().c_str(), stream_id, peer_node_desc->GetName().c_str(),
                 peer_node_stream_id, peer_out_anchor->GetIdx(), peer_node_life_time);
        }
        AddDependLife(org_node, peer_node, stream_id, depend_stream_life, total_node_depend_stream_life);
      }
    }
  }

  // save on node to save next calculation
  for (auto &it : depend_stream_life) {
    if (total_node_depend_stream_life[node_id].find(it.first) == total_node_depend_stream_life[node_id].end()) {
      total_node_depend_stream_life[node_id][it.first] = it.second;
    }
  }
}

void MemoryBlock::AddDependLifeBegin(DependStreamLife &total_node_depend_stream_life) {
  if (!depend_stream_life_.empty()) {
    return;
  }
  if (!node_type_index_list_.empty()) {
    auto node = node_type_index_list_.front().node;
    if (node != nullptr) {
      AddDependLife(node, node, stream_id_, depend_stream_life_, total_node_depend_stream_life);
    }
  }
  depend_stream_life_[stream_id_] = GetLifeBegin();
}

size_t MemoryBlock::GetLifeEnd() const {
  if (!node_type_index_list_.empty()) {
    return node_type_index_list_.back().life_time_end;
  }
  return kMaxLifeTime;
}

void MemoryBlock::SetLifeTimeEnd(size_t time) {
  if (!node_type_index_list_.empty()) {
    node_type_index_list_.back().life_time_end = time;
  }
}

void SetLastUsedInputMemAttr(NodePtr &node, int input_index) {
  if (node == nullptr) {
    return;
  }
  auto node_op_desc = node->GetOpDesc();
  if (node_op_desc != nullptr) {
    auto input_desc = node_op_desc->MutableInputDesc(input_index);
    if (!ge::AttrUtils::SetBool(*input_desc, ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true)) {
      GELOGW("Set %s input[%d] ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE to true failed.", node_op_desc->GetName().c_str(),
             input_index);
      return;
    }
    GELOGD("Set %s input[%d] ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE to true success.", node_op_desc->GetName().c_str(),
           input_index);
  }
}

Status GetNoAlignSize(const ge::OpDesc &desc, uint32_t index, size_t &size) {
  // calculate tensor real size
  auto output_op_desc = desc.GetOutputDescPtr(index);
  if (output_op_desc == nullptr) {
    GELOGI("GetNoAlignSize failed. OpName: %s, OpType: %s, index: %d",
           desc.GetName().c_str(), desc.GetType().c_str(), index);
    return  FAILED;
  }
  int64_t tensor_size = 0;
  GeShape shape = output_op_desc->GetShape();
  Format format = output_op_desc->GetFormat();
  DataType data_type = output_op_desc->GetDataType();
  graphStatus graph_status = TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(graph_status, "[Calculate][TensorSize]shape:%s, format:%s, data_type:%s, op:%s, out_index:%u",
           shape.ToString().c_str(),
           TypeUtils::FormatToSerialString(format).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str(),
           desc.GetName().c_str(), index);
    REPORT_CALL_ERROR("E19999", "CalcTensorMemSize fail, shape:%s, format:%s, data_type:%s, op:%s, out_index:%u",
                      shape.ToString().c_str(),
                      TypeUtils::FormatToSerialString(format).c_str(),
                      TypeUtils::DataTypeToSerialString(data_type).c_str(),
                      desc.GetName().c_str(), index);
    return FAILED;
  }
  size = static_cast<size_t>(tensor_size);
  return SUCCESS;
}

string ToString(ge::NodeTypeIndex &x) {
  stringstream ss;
  ss << "[" << x.node->GetName() << "(" << x.node->GetType() << "), ";
  if (x.mem_type == kOutput) {
    ss << "Output, ";
  } else {
    ss << "Workspace, ";
  }
  ss << x.index << "]";
  return ss.str();
}

string MemoryBlock::String() {
  stringstream ss;
  ss << "Block size: " << Size() << " from " << HeadOffset() << " to " << TailOffset() << " ";
  ss << "real_size_list: " << ToString(real_size_list_) << " ";
  ss << "ref_count: " << ref_count_ << " ";
  ss << "members: ";
  for (auto x : NodeTypeIndexList()) {
    ss << "__node: " << ToString(x) << " ";
  }
  for (const auto& symbol : SymbolList()) {
    ss << "__symbol: " << symbol << " ";
  }
  ss << "memory_type: " << memory_type_ << " ";
  return ss.str();
}

BlockMemAssigner::BlockMemAssigner(ComputeGraphPtr compute_graph, const map<string, string> &anchor_to_symbol,
                                   const map<string, list<NodeIndexIO>> &symbol_to_anchors)
    : mem_offset_(0), p2p_mem_offset_(0), compute_graph_(std::move(compute_graph)),
      symbol_to_anchors_(symbol_to_anchors), anchor_to_symbol_(anchor_to_symbol), life_time_(0) {}

BlockMemAssigner::~BlockMemAssigner() {
  GELOGD("[Destruct][BlockMemAssigner]blocks_store_ size : %lu", blocks_store_.size());
  for (MemoryBlock *memory_block : blocks_store_) {
    GE_DELETE_NEW_SINGLE(memory_block);
  }
}

void GetMaxBatchAllMemorySize(std::map<std::string, vector<int64_t>> &batch_all_memory_size,
                              std::map<std::string, int64_t> batch_total_size, vector<int64_t> &all_memory_size,
                              std::string &max_batch_label) {
  // use max batch all memory size for reuse range
  int64_t max_batch_size = 0;
  for (const auto &it : batch_total_size) {
    GELOGI("Batch[%s] total memory size[%ld]", it.first.c_str(), it.second);
    // no batch label
    if (it.first.empty()) {
      continue;
    }
    if (it.second > max_batch_size) {
      max_batch_size = it.second;
      max_batch_label = it.first;
    }
  }
  GELOGI("Max batch[%s] total memory size[%ld]", max_batch_label.c_str(), max_batch_size);

  for (const auto &it : batch_all_memory_size) {
    if (it.first.empty() || (it.first == max_batch_label)) {
      all_memory_size.insert(all_memory_size.end(), it.second.begin(), it.second.end());
    }
  }
  // all_memory_size can't be empty
  if (all_memory_size.empty()) {
    all_memory_size.emplace_back(MEM_ALIGN_SIZE);
  }
  sort(all_memory_size.begin(), all_memory_size.end());
  GELOGD("All memory size: %s", ToString(all_memory_size).c_str());

  for (auto iter = all_memory_size.begin(); iter != all_memory_size.end();) {
    if (*iter == 0) {
      iter = all_memory_size.erase(iter);
    } else {
      ++iter;
    }
  }
}

void BlockMemAssigner::MarkContinuousAllocedForOneInputFromVariable(const NodePtr &node) {
  auto node_op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(node_op_desc == nullptr, return);
  // if input size just one and from variable, no need to reassign continuous memory
  bool is_input_continuous = false;
  (void)ge::AttrUtils::GetBool(node_op_desc, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
  if (is_input_continuous && (node_op_desc->GetInputsSize() == 1)) {
    auto peer_out_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, return);
    auto in_node = peer_out_anchor->GetOwnerNode();
    GE_IF_BOOL_EXEC(in_node == nullptr, return);
    if (in_node->GetType() == VARIABLE || in_node->GetType() == CONSTANT) {
      GELOGI("node only one input and from variable, set continuous alloced. node_name:%s", node->GetName().c_str());
      (void)ge::AttrUtils::SetBool(node_op_desc, ATTR_NAME_CONTINUOUS_INPUT_ALLOC, true);
    }
  }
}

void BlockMemAssigner::GetOutAndWorkSpaceMem(vector<int64_t> &all_memory_size) {
  vector<int64_t> temp;
  std::map<std::string, vector<int64_t>> batch_all_memory_size;
  std::map<std::string, int64_t> batch_total_size;
  for (const NodePtr &n : compute_graph_->GetAllNodes()) {
    MarkContinuousAllocedForOneInputFromVariable(n);

    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);

    if (CheckIsZeroMemNodeType(node_op_desc->GetType())) {
      continue;
    }

    std::string batch_label;
    (void)ge::AttrUtils::GetStr(node_op_desc, ATTR_NAME_BATCH_LABEL, batch_label);

    if (node_op_desc->GetType() == ATOMICADDRCLEAN) {
      atomic_addr_clean_id_ = node_op_desc->GetId();
    }

    for (auto &out_anchor : n->GetAllOutDataAnchors()) {
      auto output_desc = node_op_desc->GetOutputDescPtr(out_anchor->GetIdx());
      int64_t size = 0;
      GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(*output_desc, size) != SUCCESS, GELOGI("Get size failed"));
      GE_IF_BOOL_EXEC(size < 0,
                      GELOGE(FAILED, "[Check][TensorSize]tensor_size:%ld is invalid, "
                             "maybe it is unknown shape node, Node_name:%s",
                             size, node_op_desc->GetName().c_str());
                      REPORT_INNER_ERROR("E19999", "tensor_size:%ld is invalid, "
                                         "maybe it is unknown shape node, Node_name:%s",
                                         size, node_op_desc->GetName().c_str());
                      return;);
      batch_all_memory_size[batch_label].emplace_back(size);
      if (batch_total_size.find(batch_label) == batch_total_size.end()) {
        batch_total_size[batch_label] = size;
      } else {
        batch_total_size[batch_label] += size;
      }

      if (!anchor_to_symbol_.empty()) {
        auto iter1 = anchor_to_symbol_.find(NodeIndexIO(n, out_anchor->GetIdx(), kOut).ToString());
        if (iter1 == anchor_to_symbol_.end()) {
          continue;
        }
        const std::string &symbol = iter1->second;
        auto iter2 = symbol_size_.find(symbol);
        if (iter2 == symbol_size_.end()) {
          symbol_size_[symbol] = size;
        } else if (size > static_cast<int64_t>(iter2->second)) {
          iter2->second = size;
        }
      }
    }
    temp.clear();
    GetNodeWorkSpaceSize(n, temp, batch_total_size[batch_label]);
    batch_all_memory_size[batch_label].insert(batch_all_memory_size[batch_label].end(), temp.begin(), temp.end());
  }
  GELOGI("The last atomic_addr_clean node id: %ld", atomic_addr_clean_id_);
  GetMaxBatchAllMemorySize(batch_all_memory_size, batch_total_size, all_memory_size, max_batch_label_);
  InitReuseFlag();
  PrintSymbolMap();
}

///
/// @ingroup domi
/// @brief decide memory size based on actual input memory size
/// @param [in] size actual memory size in need
/// @param [in] ranges memory size provided
/// @return size_t memory size to apply
///
size_t GetBlockSize(size_t size, const vector<int64_t> &ranges) {
  for (int64_t x : ranges) {
    auto x_temp = static_cast<size_t>(x);
    if (size <= x_temp) {
      return x_temp;
    }
  }

  GELOGW("Memory needed size:%zu is beyond the biggest block in memory ranges.", size);
  return size;
}

bool IsDirectOutputNode(const NodePtr &node, int idx) {
  if ((node != nullptr) && (node->GetOpDesc() != nullptr) && (node->GetOpDesc()->GetType() == NETOUTPUT)) {
    GELOGD("This is netoutput node, the input node mem can not be reused");
    return true;
  }
  return false;
}

bool CanReuseBlock(int32_t thread_scope_id, size_t continuous_life_begin, const MemoryBlock &reusable_block,
                   size_t block_size) {
  if (!reusable_block.CanReuse(thread_scope_id)) {
    return false;
  }

  bool can_reuse = false;
  if (reusable_block.Size() == block_size) {
    // in some continuous input case, continuous first input node's is not same as topo first node.
    if (continuous_life_begin > 0) {
      if (continuous_life_begin > reusable_block.GetLifeEnd()) {
        can_reuse = true;
      }
    } else {
      can_reuse = true;
    }
  }
  return can_reuse;
}

bool BlockMemAssigner::IsOutNodeSetContinuousInput(const NodePtr &n, uint32_t out_index, std::string &peer_name,
                                                   uint32_t &peer_input_index,
                                                   bool &no_need_assign_memory, bool &reset_zero_copy_flag) {
  if (n == nullptr || n->GetAllOutDataAnchors().size() <= 0) {
    return false;
  }
  auto node_desc = n->GetOpDesc();
  GE_IF_BOOL_EXEC(node_desc == nullptr, GELOGE(FAILED, "Node[%s] nodedesc is null.", n->GetName().c_str());
                  return false;);
  std::vector<int64_t> offsets_for_fusion = {};
  bool has_lx_fusion_attr =
      AttrUtils::GetListInt(node_desc, ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);

  if (static_cast<size_t>(out_index) < n->GetAllOutDataAnchors().size()) {
    auto out_anchor = n->GetOutDataAnchor(out_index);
    GE_IF_BOOL_EXEC(out_anchor == nullptr,
                    GELOGE(FAILED, "[Check][Anchor]Node[%s] output[%u] anchor is null.",
                           n->GetName().c_str(), out_index);
                    REPORT_INNER_ERROR("E19999", "output anchor is null, node_name: %s output_index: %u.",
                                       n->GetName().c_str(), out_index);
                    return false;);
    for (auto const &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(peer_in_anchor == nullptr,
                      GELOGE(FAILED, "[Check][Anchor]Node[%s] output[%u] peer_in_anchor 0 is null.",
                             n->GetName().c_str(), out_index);
                      REPORT_INNER_ERROR("E19999", "output anchor peer is null, node_name: %s output_index: %u.",
                                         n->GetName().c_str(), out_index);
                      return false;);
      auto peer_node = peer_in_anchor->GetOwnerNode();
      GE_IF_BOOL_EXEC(peer_node == nullptr,
                      GELOGE(FAILED, "[Check][Node]Node[%s] output[%u] peer node is null.",
                             n->GetName().c_str(), out_index);
                      REPORT_INNER_ERROR("E19999", "output anchor peer node is null, node_name: %s output_index: %u.",
                                         n->GetName().c_str(), out_index);
                      return false;);

      // Get the continuous input type of the node, default is false
      bool is_input_continuous = false;
      auto peer_in_node_desc = peer_node->GetOpDesc();
      GE_IF_BOOL_EXEC(peer_in_node_desc == nullptr,
                      GELOGE(FAILED, "[Check][OpDesc]Node[%s] output[%u] nodedesc is null.",
                             n->GetName().c_str(), out_index);
                      REPORT_INNER_ERROR("E19999", "output anchor peer op_desc is null, node_name:%s output_index:%u.",
                                         n->GetName().c_str(), out_index);
                      return false;);

      // If GetBool fail, is_input_continuous is false.
      (void)ge::AttrUtils::GetBool(peer_in_node_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_input_continuous);
      if (is_input_continuous) {
        reset_zero_copy_flag = true;
        has_lx_fusion_attr = true;
      } else {
        (void)ge::AttrUtils::GetBool(peer_in_node_desc, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
      }

      // lx_fusion memory only assign first input, broadcast's input some are variable some are not, reassign later
      GE_IF_BOOL_EXEC(is_input_continuous &&
          (CheckIsZeroMemNodeType(peer_node->GetType()) || (has_lx_fusion_attr && (peer_in_anchor->GetIdx() != 0))),
                      GELOGI("Node[%s] output[%u] no_need_assign_memory.", n->GetName().c_str(), out_index);
                      no_need_assign_memory = true;
                      return false;);

      if (is_input_continuous) {
        if (n->GetOwnerComputeGraph() != nullptr) {
          string graph_name = n->GetOwnerComputeGraph()->GetName();
          GELOGI("%s name[%s] output[%u] node[%s] set input[%d] continuous, input size[%u].", graph_name.c_str(),
                 n->GetName().c_str(), out_index, peer_in_node_desc->GetName().c_str(), peer_in_anchor->GetIdx(),
                 peer_node->GetAllInDataAnchorsSize());
          // Only set attr one times.
          if (node_continuous_input_blocks_[peer_in_node_desc->GetName()].size() == 0) {
            (void)ge::AttrUtils::SetBool(peer_in_node_desc, ATTR_NAME_CONTINUOUS_INPUT_ALLOC, true);
            // lx fusion case assign max size for first block, so reuse as none continuous
            GE_IF_BOOL_EXEC(has_lx_fusion_attr,
                            is_op_reuse_mem_ = IsContinuousMemoryReuse(n, peer_node, out_index);
                            return false;);
            node_continuous_input_counts_[peer_in_node_desc->GetName()] = peer_node->GetAllInDataAnchorsSize();
          }
          peer_input_index = peer_in_anchor->GetIdx();
          peer_name = peer_in_node_desc->GetName();
          return true;
        }
      }
    }
  }
  return false;
}

bool IsContinuousInputNodeMaxLife(const NodePtr &n, uint32_t out_index) {
  if (n == nullptr) {
    return false;
  }

  int64_t max_node_life_time = 0;
  int64_t continuous_input_node_life_time = 0;
  if (static_cast<size_t>(out_index) < n->GetAllOutDataAnchors().size()) {
    auto out_anchor = n->GetOutDataAnchor(out_index);
    if(out_anchor == nullptr) {
      return false;
    }

    // continuous input node's life time should be max
    for (auto const &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if ((peer_in_anchor == nullptr) || (peer_in_anchor->GetOwnerNode() == nullptr)){
        return false;
      }
      auto peer_in_node_desc = peer_in_anchor->GetOwnerNode()->GetOpDesc();
      GE_IF_BOOL_EXEC(peer_in_node_desc == nullptr,
                      GELOGE(FAILED, "Node[%s] output[%u] peer in node desc is null.", n->GetName().c_str(), out_index);
      return false;);

      if(peer_in_node_desc->GetId() > max_node_life_time) {
        max_node_life_time = peer_in_node_desc->GetId();
      }

      // If GetBool fail, is_input_continuous is false.
      bool is_input_continuous = false;
      (void)ge::AttrUtils::GetBool(peer_in_node_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_input_continuous);
      if (!is_input_continuous) {
        (void)ge::AttrUtils::GetBool(peer_in_node_desc, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
      }
      if (is_input_continuous) {
        continuous_input_node_life_time = peer_in_node_desc->GetId();
      }
    }
  }
  return ((max_node_life_time != 0) && (continuous_input_node_life_time == max_node_life_time)) ;
}

///
/// @ingroup GE
/// @brief Check continuous memory reuseable
/// @return void
///
bool BlockMemAssigner::IsContinuousMemoryReuse(const NodePtr &n, const NodePtr &peer_node, uint32_t out_index) {
  // n,peer_node_desc have been checked
  auto node_desc = n->GetOpDesc();
  auto peer_node_desc = peer_node->GetOpDesc();
  continuous_life_begin_ = static_cast<size_t>(node_desc->GetId());
  // lx fusion case check all continuous input node, firt input node's life time should be min
  for (const auto &in_anchor : peer_node->GetAllInDataAnchors()) {
    if ((in_anchor == nullptr) || (in_anchor->GetPeerOutAnchor() == nullptr) ||
        (in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr) ||
        (in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc() == nullptr)) {
      GELOGE(FAILED, "[Check][OpDesc]Node[%s] output[%u] peer input node desc is null.",
             n->GetName().c_str(), out_index);
      REPORT_INNER_ERROR("E19999", "get output anchor peer op_desc fail, node_name: %s output_index: %u.",
                         n->GetName().c_str(), out_index);
      return false;
    }
    auto peer_out_node_desc = in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc();
    ///
    ///  node2 node1  node3
    ///      |   /   / |
    ///      node5    node6
    /// firt input node's life time is not min
    /// when node5's first input node2's life time is not min(node2 > node1), use node1's life time to reuse
    ///
    if (static_cast<size_t>(peer_out_node_desc->GetId()) < continuous_life_begin_) {
      continuous_life_begin_ = static_cast<size_t>(peer_out_node_desc->GetId());
      GELOGI(
        "Node[%s] life[%ld] output[%u] is not continuous input node[%s] life[%ld]'s min life time,"
        "min is node[%s] life[%zu]",
        n->GetName().c_str(), node_desc->GetId(), out_index, peer_node_desc->GetName().c_str(),
        peer_node_desc->GetId(), peer_out_node_desc->GetName().c_str(), continuous_life_begin_);
    }
    // when node3's output node5's life time is not max(node6 > node5), not reuse
    if (!IsContinuousInputNodeMaxLife(in_anchor->GetPeerOutAnchor()->GetOwnerNode(),
                                      in_anchor->GetPeerOutAnchor()->GetIdx())) {
      GELOGI(
        "Node[%s] life[%ld] output[%u]'s continuous input node[%s] life[%ld]'s is not node[%s] output[%d]'s "
        "max life node",
        n->GetName().c_str(), node_desc->GetId(), out_index, peer_node_desc->GetName().c_str(),
        peer_node_desc->GetId(), peer_out_node_desc->GetName().c_str(), in_anchor->GetPeerOutAnchor()->GetIdx());
      return false;
    }
  }
  return true;
}

///
/// @ingroup GE
/// @brief Check pre_reuse flag & post_reuse glag for each symbol
/// @return void
///
void BlockMemAssigner::InitReuseFlag() {
  static const std::set<std::string> kPreReuseTypes = { ge::DATA_TYPE, ge::AIPP_DATA_TYPE, ge::ANN_DATA_TYPE,
                                                        ge::NETOUTPUT, ge::PROPOSAL, ge::ZEROSLIKE,
                                                        ge::CONSTANT, ge::CONSTANTOP };
  static const std::set<std::string> kPostReuseTypes = { ge::DATA_TYPE, ge::AIPP_DATA_TYPE, ge::ENTER, ge::REFENTER,
                                                         ge::NEXTITERATION, ge::REFNEXTITERATION };
  for (const auto &pair : symbol_to_anchors_) {
    std::string symbol = pair.first;
    bool pre_reuse_flag = true;
    bool post_reuse_flag = true;
    // default memory type
    int64_t mem_type = RT_MEMORY_HBM;
    GetSymbolMemType(pair.second, mem_type);
    GELOGD("The memory type of symbol[%s] is [%ld]].", symbol.c_str(), mem_type);
    if (mem_type == RT_MEMORY_P2P_DDR) {
      UpdateOpTensorMemType(pair.second, mem_type);
    }
    // Only the memory with special requirements is processed. The HBM uses the default processing mode.
    if (mem_type == RT_MEMORY_P2P_DDR) {
      symbol_to_mem_type_[symbol] = mem_type;
    }

    for (const auto &node_index_io : pair.second) {
      if (node_index_io.io_type_ == kIn) {
        continue;
      }

      OutDataAnchorPtr out_anchor = node_index_io.node_->GetOutDataAnchor(node_index_io.index_);
      if (out_anchor == nullptr) {
        continue;
      }

      bool out_flg = false;
      if (node_index_io.node_->GetOutDataNodes().empty()) {
        out_flg = true;
      }
      for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
        if (IsDirectOutputNode(in_anchor->GetOwnerNode(), in_anchor->GetIdx())) {
          out_flg = true;
          break;
        }
      }
      const std::string &type = out_anchor->GetOwnerNode()->GetType();
      pre_reuse_flag = pre_reuse_flag && !out_flg && (kPreReuseTypes.count(type) == 0);
      post_reuse_flag = post_reuse_flag && (kPostReuseTypes.count(type) == 0);
      if (!pre_reuse_flag && !post_reuse_flag) {
        break;
      }
    }
    pre_reuse_flag_[symbol] = pre_reuse_flag;
    post_reuse_flag_[symbol] = post_reuse_flag;
  }
}

///
/// @ingroup GE
/// @brief get pre_reuse flag
/// @param [in] node
/// @param [in] out_index
/// @return bool
///
bool BlockMemAssigner::IsPreReuse(const NodePtr &node, uint32_t out_index) const {
  OutDataAnchorPtr out_data_anchor = nullptr;
  if (static_cast<size_t>(out_index) < node->GetAllOutDataAnchors().size()) {
    out_data_anchor = node->GetOutDataAnchor(out_index);
  }
  if (out_data_anchor == nullptr) {
    return false;
  }
  NodeIndexIO cur_node_index_io(out_data_anchor->GetOwnerNode(), out_data_anchor->GetIdx(), kOut);
  auto iter1 = anchor_to_symbol_.find(cur_node_index_io.ToString());
  if (iter1 == anchor_to_symbol_.end()) {
    return false;
  }

  const std::string &symbol = iter1->second;
  auto iter2 = pre_reuse_flag_.find(symbol);
  if (iter2 == pre_reuse_flag_.end()) {
    return false;
  }
  return iter2->second;
}

///
/// @ingroup GE
/// @brief get post_reuse flag
/// @param [in] mem_block
/// @return bool
///
bool BlockMemAssigner::IsPostReuse(const MemoryBlock *mem_block) const {
  if (mem_block == nullptr) {
    return false;
  }
  for (const auto &symbol : mem_block->SymbolList()) {
    auto iter = post_reuse_flag_.find(symbol);
    if (iter == post_reuse_flag_.end()) {
      continue;
    }
    if (!iter->second) {
      return false;
    }
  }
  return true;
}

///
/// @ingroup GE
/// @brief check if symbol of cur node_index_io has block
/// @param [in] node_index_io
/// @param [out] symbol
/// @return bool
///
bool BlockMemAssigner::IsSymbolExist(const NodeIndexIO &node_index_io, string &symbol) {
  auto iter = anchor_to_symbol_.find(node_index_io.ToString());
  if (iter == anchor_to_symbol_.end()) {
    return false;
  }

  symbol = iter->second;
  return symbol_blocks_.find(iter->second) != symbol_blocks_.end();
}

///
/// @ingroup GE
/// @brief Print symbol
/// @return void
///
void BlockMemAssigner::PrintSymbolMap() {
  for (const auto &pair : symbol_to_anchors_) {
    GELOGD("symbol=%s, max_size=%zu, pre_reuse=%s, post_reuse=%s", pair.first.c_str(), symbol_size_[pair.first],
           pre_reuse_flag_[pair.first] ? "true" : "false", post_reuse_flag_[pair.first] ? "true" : "false");
    for (const auto &node_index_io : pair.second) {
      GELOGD("anchor:%s", node_index_io.ToString().c_str());
    }
  }
}

void BlockMemAssigner::GetSymbolMemType(std::list<NodeIndexIO> node_index_io_list, int64_t &memory_type) {
  memory_type = RT_MEMORY_HBM;
  vector<int64_t> memory_types;
  for (auto &node_index_io : node_index_io_list) {
    auto op_desc = node_index_io.node_->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGW("Node[%s] op desc is null.", node_index_io.node_->GetName().c_str());
      return;
    }

    if (node_index_io.io_type_ == kIn) {
      vector<int64_t> input_memory_types;
      (void) ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, input_memory_types);
      if (!input_memory_types.empty() && node_index_io.index_ < input_memory_types.size()) {
        int64_t input_memory_type = input_memory_types[node_index_io.index_];
        GELOGD("Node[%s]: the memory type of input index [%u] is [%ld]].", op_desc->GetName().c_str(),
               node_index_io.index_, input_memory_type);
        memory_types.emplace_back(input_memory_type);
      }
    }
    if (node_index_io.io_type_ == kOut) {
      vector<int64_t> output_memory_types;
      (void) ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_memory_types);
      if (!output_memory_types.empty() && node_index_io.index_ < output_memory_types.size()) {
        int64_t output_memory_type = output_memory_types[node_index_io.index_];
        GELOGD("Node[%s]: the memory type of output index [%u] is [%ld]].", op_desc->GetName().c_str(),
               node_index_io.index_, output_memory_type);
        memory_types.emplace_back(output_memory_type);
      }
    }
  }

  // memory priority
  for (auto node_memory_type : memory_types) {
    if (node_memory_type > memory_type) {
      memory_type = node_memory_type;
    }
  }
}

void BlockMemAssigner::UpdateOpTensorMemType(std::list<NodeIndexIO> node_index_io_list, int64_t memory_type) {
  for (auto &node_index_io : node_index_io_list) {
    auto op_desc = node_index_io.node_->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGW("Node[%s] op desc is null.", node_index_io.node_->GetName().c_str());
      return;
    }

    if (node_index_io.io_type_ == kIn) {
      auto input_desc = op_desc->MutableInputDesc(node_index_io.index_);
      (void) AttrUtils::SetInt(input_desc, ATTR_NAME_TENSOR_MEM_TYPE, memory_type);
    }

    if (node_index_io.io_type_ == kOut) {
      auto output_desc = op_desc->MutableOutputDesc(node_index_io.index_);
      (void) AttrUtils::SetInt(output_desc, ATTR_NAME_TENSOR_MEM_TYPE, memory_type);
    }
  }
}

bool BlockMemAssigner::IsContinuousOutput(const NodePtr &n) {
  if (n == nullptr) {
    GELOGE(FAILED, "Node is null.");
    return false;
  }

  // Get the continuous output type of the node, default is false
  bool is_output_continuous = false;
  auto node_desc = n->GetOpDesc();
  if (node_desc == nullptr) {
    GELOGE(FAILED, "Node[%s] nodedesc is null.", n->GetName().c_str());
    return false;
  }

  // If GetBool fail, is_output_continuous is false.
  (void)ge::AttrUtils::GetBool(node_desc, ATTR_NAME_CONTINUOUS_OUTPUT, is_output_continuous);
  if (is_output_continuous) {
    if (n->GetOwnerComputeGraph() != nullptr) {
      string graph_name = n->GetOwnerComputeGraph()->GetName();
      GELOGI("%s name[%s] set continuous, output size[%u].", graph_name.c_str(),
          n->GetName().c_str(), n->GetAllOutDataAnchorsSize());
      return true;
    }
  }

  return false;
}

bool BlockMemAssigner::IsZeroCopyBlock(const NodePtr &node, bool continuous) {
  if (NodeUtils::IsDynamicShape(node)) {
    return ((node->GetType() == DATA_TYPE) && !continuous) || (node->GetType() == NETOUTPUT);
  }

  if ((node->GetType() == DATA_TYPE) && !continuous) {
    return !node->GetOpDesc()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX);
  }

  if (node->GetType() == NETOUTPUT) {
    const auto &owner = node->GetOwnerComputeGraph();
    return owner->GetParentGraph() == nullptr;
  }

  return false;
}

MemoryBlock *BlockMemAssigner::ApplyMemory(size_t block_size, size_t real_size, size_t no_align_size,
                                           OpMemoryType mem_type, const NodePtr &n, uint32_t out_index,
                                           const vector<bool> &workspace_reuse_flag, const bool is_op_reuse_mem,
                                           const bool continuous, int64_t memory_type) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      n == nullptr,
      REPORT_INNER_ERROR("E19999", "Input parameter n(type:node_ptr) is null, apply memory failed");
      return nullptr, "[Check][Param]Input parameter n(type:node_ptr) is null.");
  auto node_op_desc = n->GetOpDesc();
  GE_IF_BOOL_EXEC(node_op_desc == nullptr, return nullptr);
  std::string batch_label;
  (void)ge::AttrUtils::GetStr(node_op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
  if (batch_label.empty() || (batch_label == max_batch_label_)) {
    size_t align_size = real_size;
    AlignMemOffset(align_size);
    theory_memory_size_ += align_size;
    if (theory_memory_size_ > theory_min_memory_size_) {
      theory_min_memory_size_ = theory_memory_size_;
    }
  }

  bool is_reuse_memory = false;
  int32_t thread_scope_id = kInvalidThreadScopeId;
  (void)ge::AttrUtils::GetInt(node_op_desc, ATTR_NAME_THREAD_SCOPE_ID, thread_scope_id);
  if (ge_disable_reuse_mem_env_ != "1") {
    bool reuse_mem_flag = (mem_type == kOutput) ? IsPreReuse(n, out_index) :
                          !((workspace_reuse_flag.size() > out_index) && !workspace_reuse_flag[out_index]);
    is_reuse_memory = !node_op_desc->HasAttr(kL2FusionDynamicConvergeOp) &&
                      !node_op_desc->HasAttr(kOpNoReuseMem) && reuse_mem_flag && is_op_reuse_mem;
    bool do_reuse = is_reuse_memory && !continuous && !reusable_blocks_[memory_type].empty();
    if (do_reuse) {
      auto stream_id = node_op_desc->GetStreamId();
      for (auto it = reusable_blocks_[memory_type][stream_id].rbegin();
           it != reusable_blocks_[memory_type][stream_id].rend(); ++it) {
        MemoryBlock *reusable_block = *it;
        if (!IsPostReuse(reusable_block)) {
          reusable_block->reuse_mem_ = false;
          GELOGI("Unreusable block.");
          continue;
        }
        GE_IF_BOOL_EXEC(reusable_block->batch_label_ != batch_label, continue);

        // A node can reuse blocks of the same stream and preorder streams
        if (CanReuseBlock(thread_scope_id, continuous_life_begin_, *reusable_block, block_size)) {
          reusable_block->AddNodeTypeIndex({n, mem_type, out_index, false, continuous_life_begin_, thread_scope_id},
                                           real_size, no_align_size);
          if (mem_type == kOutput) {
            auto iter = anchor_to_symbol_.find(NodeIndexIO(n, out_index, kOut).ToString());
            if (iter != anchor_to_symbol_.end()) {
              reusable_block->AddSymbol(iter->second);
            }
          }
          reusable_block->continuous_block_ = continuous;
          reusable_blocks_[memory_type][stream_id].erase((++it).base());
          return reusable_block;
        }
      }
    }
  }

  auto block = new (std::nothrow) MemoryBlock(block_size, node_op_desc->GetStreamId(), is_reuse_memory, memory_type);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      block == nullptr,
      REPORT_INNER_ERROR("E19999", "new a memoryblock object failed. node_name:%s out_index:%u",
                         n->GetName().c_str(), out_index);
      return nullptr,
      "[New][Object]new MemoryBlock failed, node_name:%s out_index:%u", n->GetName().c_str(), out_index);

  // Data and netoutput need zero copy block
  block->is_zero_copy_ = IsZeroCopyBlock(n, continuous);
  block->AddNodeTypeIndex({n, mem_type, out_index, false, continuous_life_begin_, thread_scope_id},
                          real_size, no_align_size);
  block->stream_id_ = node_op_desc->GetStreamId();
  block->continuous_block_ = continuous;
  block->batch_label_ = batch_label;
  if (mem_type == kOutput) {
    auto iter = anchor_to_symbol_.find(NodeIndexIO(n, out_index, kOut).ToString());
    if (iter != anchor_to_symbol_.end()) {
      block->AddSymbol(iter->second);
    }
  }
  memory_blocks_.emplace_back(block);
  // cause memory_blocks_ may reduce when swap after,
  // create blocks_store_ to assure blocks deleted finally
  blocks_store_.emplace_back(block);
  return block;
}

bool IsOutputIndexRef(const OpDescPtr &op_desc, uint32_t index) {
  auto output_tensor = op_desc->GetOutputDescPtr(index);
  bool dst_reuse_input = false;
  (void)ge::TensorUtils::GetReuseInput(*output_tensor, dst_reuse_input);
  if (dst_reuse_input) {
    return true;
  }

  bool is_ref = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (is_ref) {
    string output_name = op_desc->GetOutputNameByIndex(index);
    for (const auto &input_name : op_desc->GetAllInputNames()) {
      if (output_name == input_name) {
        return true;;
      }
    }
  }
  return false;
}

void BlockMemAssigner::ContinuousOutRefCheck(bool &isAllOutputRef, bool &isOutputHasRef,
                                             const NodePtr &n) {
  const auto node_op_desc = n->GetOpDesc();
  for (uint32_t index = 0; index < static_cast<uint32_t>(node_op_desc->GetOutputsSize()); index++) {
    if (!IsOutputIndexRef(node_op_desc, index)) {
      isAllOutputRef = false;
      break;
    } else {
      zero_memory_list_.emplace_back(n, kOutput, index);
      isOutputHasRef = true;
    }
  }
}


Status BlockMemAssigner::ApplyContinuousMemory(const NodePtr &n, const vector<int64_t> &ranges,
                                               const bool is_op_reuse_mem) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      n == nullptr,
      REPORT_INNER_ERROR("E19999", "Input parameter n(type:node_ptr) is null");
      return INTERNAL_ERROR, "[check][param]Input parameter n(type:NodePtr) is null.");
  auto node_op_desc = n->GetOpDesc();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      node_op_desc == nullptr,
      REPORT_INNER_ERROR("E19999", "Input parameter n(type:OpDescPtr) is null");
      return INTERNAL_ERROR, "[Check][Param]Input parameter n(type:OpDescPtr) is null");

  // continuous output support ref only when all output ref input
  bool isAllOutputRef = true;
  bool isOutputHasRef = false;

  ContinuousOutRefCheck(isAllOutputRef, isOutputHasRef, n);

  if (isAllOutputRef) {
    GELOGI("continuous output node ref all input, skip continuous alloc, node_name:%s", n->GetName().c_str());
    return SUCCESS;
  }

  if (!isAllOutputRef && isOutputHasRef) {
    REPORT_INNER_ERROR("E19999", "continuous output node ref part input, not support now. node_name:%s",
                       n->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][OutRefStatus]continuous output node ref part input, not support, node_name:%s",
           n->GetName().c_str());
    return INTERNAL_ERROR;
  }

  MemoryBlock *block = nullptr;
  int64_t total_size = 0;
  int64_t memory_type = RT_MEMORY_HBM;
  for (uint32_t index = 0; index < static_cast<uint32_t>(node_op_desc->GetOutputsSize()); index++) {
    auto output_op_desc = node_op_desc->GetOutputDescPtr(index);
    if (output_op_desc == nullptr) {
      REPORT_INNER_ERROR("E19999", "get output_desc failed, node_name:%s, output_index:%u",
                         n->GetName().c_str(), index);
      GELOGE(INTERNAL_ERROR, "[Get][OutputDesc]node_name:%s, output_index:%u", n->GetName().c_str(), index);
      return INTERNAL_ERROR;
    }

    if (CheckIsZeroMemNodeType(n->GetType())) {
      zero_memory_list_.emplace_back(n, kOutput, index);
      continue;
    }

    int64_t size = 0;
    if (ge::TensorUtils::GetSize(*output_op_desc, size) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "get tensor_size failed, node_name:%s, output_index:%u",
                        n->GetName().c_str(), index);
      GELOGE(INTERNAL_ERROR, "[Get][TensorSize]node_name:%s, output_index:%u", n->GetName().c_str(), index);
      return INTERNAL_ERROR;
    }
    size_t align_size = static_cast<size_t>(size);
    AlignMemOffset(align_size);
    total_size += align_size;

    // only apply total size in first block
    if (index != 0) {
      zero_memory_list_.emplace_back(n, kOutput, index);
    } else {
      NodeIndexIO node_index_io(n, index, kOut);
      auto iter = anchor_to_symbol_.find(node_index_io.ToString());
      if (iter != anchor_to_symbol_.end()) {
        string symbol = iter->second;
        if (symbol_to_mem_type_.find(symbol) != symbol_to_mem_type_.end()) {
          memory_type = symbol_to_mem_type_[symbol];
          GELOGD("Continuous out memory symbol is [%s], memory type is [%ld]", symbol.c_str(), memory_type);
        }
      }
    }
  }

  if (total_size == 0) {
    return SUCCESS;
  }

  auto block_size = GetBlockSize(total_size, ranges);
  GELOGI("Node[%s] continuous out memory size[%ld] block size[%zu]", node_op_desc->GetName().c_str(),
         total_size, block_size);

  vector<bool> workspace_reuse_flag;
  block = ApplyMemory(block_size, total_size, total_size, kOutput, n, 0, workspace_reuse_flag, is_op_reuse_mem, true,
                      memory_type);
  if (block != nullptr) {
    // hccl task need align header and tail
    block->first_continuous_block_ = true;
    block->last_continuous_block_ = true;
    ++(block->ref_count_);
  } else {
    REPORT_CALL_ERROR("E19999", "apply continuousMemory failed, node_name:%s, total_size:%ld",
                      n->GetName().c_str(), total_size);
    GELOGE(INTERNAL_ERROR, "[Apply][ContinuousMemory]node_name:%s, total_size:%ld", n->GetName().c_str(), total_size);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

MemoryBlock *BlockMemAssigner::ApplyOutMemory(const NodePtr &n, uint32_t index, const vector<int64_t> &ranges,
                                              const bool is_op_reuse_mem, const bool continuous) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      n == nullptr,
      REPORT_INNER_ERROR("E19999", "Input parameter n(type:NodePtr) is null");
      return nullptr, "[Check][Param]Input parameter n(type:NodePtr) is null");
  auto node_op_desc = n->GetOpDesc();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      node_op_desc == nullptr,
      REPORT_INNER_ERROR("E19999", "Input parameter n(type:OpDescPtr) is null");
      return nullptr, "[Check][Param]Input parameter n(type:OpDescPtr) is null");
  MemoryBlock *block = nullptr;
  NodeIndexIO node_index_io(n, index, kOut);
  int64_t size = 0;
  auto output_op_desc = node_op_desc->GetOutputDescPtr(index);
  GE_IF_BOOL_EXEC(
      output_op_desc == nullptr,
      REPORT_INNER_ERROR("E19999", "get output_desc failed, node_name:%s, output_index:%u",
                         n->GetName().c_str(), index);
      GELOGE(FAILED, "[Get][OutputDesc]node_name:%s, output_index:%u", n->GetName().c_str(), index);
      return nullptr);
  GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(*output_op_desc, size) != SUCCESS, GELOGI("Get size failed"));
  size_t no_align_size = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      GetNoAlignSize(*node_op_desc, index, no_align_size) != SUCCESS,
      REPORT_CALL_ERROR("E19999", "Get no align size failed, node_name:%s, output_index:%u",
                        n->GetName().c_str(), index);
      return nullptr,
      "[Get][TensorSize]Get no align size, node_name:%s, output_index:%u", n->GetName().c_str(), index);

  std::string symbol;
  bool reuse_input = false;
  if (IsSymbolExist(node_index_io, symbol)) {
    block = symbol_blocks_[symbol];
    GE_IF_BOOL_EXEC(block == nullptr,
      REPORT_INNER_ERROR("E19999", "get ref block failed, node_name:%s, symbol:%s",
                         node_op_desc->GetName().c_str(), node_index_io.ToString().c_str());
      GELOGE(FAILED, "[Get][RefBlock]node_name:%s, symbol:%s",
             node_op_desc->GetName().c_str(), node_index_io.ToString().c_str());
      return nullptr);
    // reduce old size
    size_t align_size = block->Size();
    AlignMemOffset(align_size);
    theory_memory_size_ -= align_size;

    auto block_size = GetBlockSize(size, ranges);
    block->SetSize(block_size);
    block->SetLifeTimeEnd(life_time_);
    block->AddNodeTypeIndex({n, kOutput, index, true, continuous_life_begin_}, size, no_align_size);
    block->ref_count_++;
    reuse_input = true;

    // add new size
    align_size = block_size;
    AlignMemOffset(align_size);
    theory_memory_size_ += align_size;
  } else {
    // if ref input is variable, can not find symbol, must judge alone
    if (IsOutputIndexRef(node_op_desc, index)) {
      zero_memory_list_.emplace_back(n, kOutput, index, false);
      GELOGI("ref mode skip out block assign. node_name: %s, index:%d", n->GetName().c_str(), index);
      return nullptr;
    }

    int64_t max_size = size;
    int64_t memory_type = RT_MEMORY_HBM;
    auto iter1 = anchor_to_symbol_.find(node_index_io.ToString());
    if (iter1 != anchor_to_symbol_.end()) {
      auto iter2 = symbol_size_.find(iter1->second);
      if (iter2 != symbol_size_.end()) {
        max_size = iter2->second;
      }
      auto iter3 = symbol_to_mem_type_.find(iter1->second);
      if (iter3 != symbol_to_mem_type_.end()) {
        memory_type = iter3->second;
      }
    }

    auto block_size = GetBlockSize(max_size, ranges);
    vector<bool> workspace_reuse_flag;
    block = ApplyMemory(block_size, size, no_align_size, kOutput, n, index,
                        workspace_reuse_flag, is_op_reuse_mem, continuous, memory_type);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        block == nullptr,
        REPORT_CALL_ERROR("E19999", "apply out Memory failed, node_name:%s, block_size:%ld, out_index:%u",
                          n->GetName().c_str(), block_size, index);
        return nullptr,
        "[Apply][Memory]node_name:%s, block_size:%ld, out_index:%u",
        n->GetName().c_str(), block_size, index);
  }
  int out_count = 0;
  GE_IF_BOOL_EXEC(
      index >= n->GetAllOutDataAnchors().size(),
      REPORT_INNER_ERROR("E19999", "out index:%u exceed out_size:%lu, node_name:%s",
                         index, n->GetAllOutDataAnchors().size(), n->GetName().c_str());
      GELOGE(FAILED, "[Check][OutIndex]index:%u exceed out_size:%lu, node_name:%s",
             index, n->GetAllOutDataAnchors().size(), n->GetName().c_str());
      return nullptr);
  auto out_data_anchor = n->GetOutDataAnchor(index);
  GE_IF_BOOL_EXEC(
      out_data_anchor == nullptr,
      REPORT_INNER_ERROR("E19999", "out anchor is null, index:%u, node_name:%s", index, n->GetName().c_str());
      GELOGE(FAILED, "[Check][OutAnchor]is null, index:%u, node_name:%s", index, n->GetName().c_str());
      return nullptr);
  for (const auto &in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    auto owner_node = in_anchor->GetOwnerNode();
    auto op_desc = owner_node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
    Params *instance = Params::Instance();
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(instance == nullptr, return nullptr, "Params instance is nullptr.");
    if (!((instance->GetTarget() == TARGET_TYPE_TINY) && (op_desc->GetType() == NETOUTPUT))) {
      out_count++;
    }
  }
  block->ref_count_ = (reuse_input && out_count != 0) ? (block->ref_count_ + out_count - 1)
                                                      : (block->ref_count_ + out_count);
  return block;
}

bool IsOutputBlock(const ge::InDataAnchorPtr &in_data_anchor) {
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, GELOGE(FAILED, "Peer out anchor is nullptr."); return false);
  auto src = peer_out_anchor->GetOwnerNode();
  int32_t index = peer_out_anchor->GetIdx();
  auto iter = GetLocalOmgContext().out_nodes_map.find(src->GetName());
  if (iter != GetLocalOmgContext().out_nodes_map.end()) {
    for (auto id : iter->second) {
      if (index == id) {
        return true;
      }
    }
  }
  return false;
}

// atomic out memory will be reassigned
bool IsAtomicOutputMemory(const ge::NodePtr &node, uint32_t output_index, bool is_atomic,
                          bool out_node_set_continuous_input) {
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  vector<int64_t> atomic_output_index;
  // If GetListInt fail, atomic_output_index is empty.
  (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  if (!out_node_set_continuous_input && is_atomic) {
    for (auto &index : atomic_output_index) {
      if (static_cast<uint32_t>(index) == output_index) {
        if (node->GetOwnerComputeGraph() != nullptr) {
          string graph_name = node->GetOwnerComputeGraph()->GetName();
          GELOGD("Atomic no assign %s name[%s] output[%ld] streamid[%ld].", graph_name.c_str(),
                 op_desc->GetName().c_str(), index, op_desc->GetStreamId());
        }
        return true;
      }
    }
  }
  return false;
}

bool IsKnownSubgraphData(const NodePtr &node) {
  if (NodeUtils::IsDynamicShape(node)) {
    return false;
  }

  return node->GetOpDesc()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX);
}

void BlockMemAssigner::ReleaseMemory(MemoryBlock *to_release, vector<MemoryBlock *> &reusable_memory,
                                     bool same_stream) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(to_release == nullptr, return, "Input parameter to_release is null.");
  GE_CHK_TRUE_EXEC_INFO(to_release->ref_count_ <= 0, return, "Release memory");
  --to_release->ref_count_;
  GE_CHK_TRUE_EXEC_INFO(!to_release->reuse_mem_, return, "doesn't reuse memory");
  if (!same_stream) {
    to_release->same_stream_ = false;
  }
  if (to_release->ref_count_ == 0) {
    if (to_release->reuse_mem_ && !to_release->RealSizeList().empty()) {
      if (to_release->batch_label_.empty() || (to_release->batch_label_ == max_batch_label_)) {
        size_t align_size = to_release->RealSizeList().back();
        AlignMemOffset(align_size);
        theory_memory_size_ -= align_size;
      }
    }
    if (to_release->same_stream_) {
      to_release->SetLifeTimeEnd(life_time_);
      reusable_memory.emplace_back(to_release);
    }
  }
}

void BlockMemAssigner::ReleaseMemorys(const vector<MemoryBlock *> &to_releases,
                                      vector<MemoryBlock *> &reusable_memory) {
  for (auto mem_block : to_releases) {
    ReleaseMemory(mem_block, reusable_memory);
  }
}

void BlockMemAssigner::ReleaseInputNodeOutMemory(const unordered_map<string, vector<MemoryBlock *>> &node_out_blocks,
                                                 vector<MemoryBlock *> &reusable_memory, NodePtr &node) {
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    if ((in_anchor->GetPeerOutAnchor() == nullptr) ||
        (in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc() == nullptr) || (node->GetOpDesc() == nullptr)) {
      return;
    }
    GE_IF_BOOL_EXEC(IsOutputBlock(in_anchor), continue);

    auto node_name = in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName();

    GE_IF_BOOL_EXEC((in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetType() == CONSTANT) ||
                      (in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetType() == FASTRCNNPREDICTIONS) ||
                      (in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetType() == CONSTANTOP),
                    continue);

    auto it = node_out_blocks.find(node_name);
    if (it == node_out_blocks.end()) {
      continue;
    }
    for (auto block : it->second) {
      const vector<NodeTypeIndex> &node_type_indexs = block->NodeTypeIndexList();
      if (node_type_indexs.empty()) {
        continue;
      }
      GELOGD("node_type_indexs: %d, %s", node_type_indexs.back().index,
             node_type_indexs.back().node->GetName().c_str());

      bool is_block_matched = false;
      for (auto &node_type_index : node_type_indexs) {
        is_block_matched = (node_type_index.node == in_anchor->GetPeerOutAnchor()->GetOwnerNode()) &&
                           (node_type_index.index == static_cast<uint32_t>(in_anchor->GetPeerOutAnchor()->GetIdx()));
        if (is_block_matched) {
          GELOGI("Block of peer out is matched. Peer node:%s, output index:%u, "
                 "current node:%s, input index:%d, block ref_count:%d.",
                 node_type_index.node->GetName().c_str(), node_type_index.index,
                 node->GetName().c_str(), in_anchor->GetIdx(), block->ref_count_);
          break;
        }
      }

      if (is_block_matched) {
        ReleaseMemory(block, reusable_memory, (node->GetOpDesc()->GetStreamId() == block->stream_id_));
        if (block->ref_count_ == 0 && block->same_stream_) {
          SetLastUsedInputMemAttr(node, in_anchor->GetIdx());
        }
        break;
      }
    }
  }
}

void SplitStringByComma(const string &str, vector<string> &sub_str_vec) {
  std::string tmp_string = str + ",";
  std::string::size_type start_pos = 0;
  std::string::size_type cur_pos = tmp_string.find(',', 0);
  while (cur_pos != std::string::npos) {
    std::string sub_str = tmp_string.substr(start_pos, cur_pos - start_pos);
    if (!sub_str.empty()) {
      vector<string>::iterator ret = std::find(sub_str_vec.begin(), sub_str_vec.end(), sub_str);
      if (ret == sub_str_vec.end()) {
        sub_str_vec.push_back(sub_str);
      }
    }
    start_pos = cur_pos + 1;
    cur_pos = tmp_string.find(',', start_pos);
  }
}

void CheckAndGetOpReuseEnv(const string &env, vector<string> &env_vec, bool &op_reuse_env_valid) {
  string env_str;
  env_str = string(env);
  if (env_str.size() > kReuseMaxCharNum) {
    GELOGE(FAILED, "The OP_NO_REUSE_MEM has more than %d characters.", kReuseMaxCharNum);
    return;
  }

  SplitStringByComma(env_str, env_vec);
  if (env_vec.size() > kReuseMaxOpNum) {
    GELOGE(FAILED, "The OP_NO_REUSE_MEM has more than %d nodes.", kReuseMaxOpNum);
    return;
  }

  op_reuse_env_valid = true;
  return;
}

void BlockMemAssigner::CheckAndReleaseSuspendedBlock(const NodePtr &node, uint32_t idx, MemoryBlock *block) {
  if (node == nullptr || node->GetOpDesc() == nullptr || block == nullptr) {
    return;
  }
  int64_t stream_id = node->GetOpDesc()->GetStreamId();
  auto out_data_anchor = node->GetOutDataAnchor(static_cast<int>(idx));
  bool is_suspended = (out_data_anchor != nullptr) && (out_data_anchor->GetPeerInDataNodesSize() == 0);
  if (is_suspended) {
    block->ref_count_ = (block->ref_count_ != 0) ? (block->ref_count_) : (1);
    stream_workspace_blocks_[block->memory_type_][stream_id].emplace_back(block);
    GELOGI("The output is suspended, and will be released in allocation of next node. Name:%s, index:%u, "
           "size:%zu, ref_count:%d.", node->GetName().c_str(), idx, block->Size(), block->ref_count_);
  }
}

Status BlockMemAssigner::AssignOutputMemoryWithReuse(const NodePtr &node, vector<int64_t> &ranges) {
  auto op_desc = node->GetOpDesc();
  int64_t stream_id = op_desc->GetStreamId();
  vector<int64_t> memorys_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memorys_type);
  GELOGD("Assign memory node[%s], output size[%zu], output memory type size[%zu]", op_desc->GetName().c_str(),
         op_desc->GetOutputsSize(), memorys_type.size());
  if (has_mem_type_attr && (memorys_type.size() != op_desc->GetOutputsSize())) {
    REPORT_INNER_ERROR("E19999", "Attr[%s] size:%zu not equal to node output size:%zu, node_name:%s",
                       ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), memorys_type.size(),
                       op_desc->GetOutputsSize(), op_desc->GetName().c_str());
    GELOGE(
        INTERNAL_ERROR,
        "[Check][MemTypeAttr]Attr %s size:%zu not equal to node output size:%zu, node_name:%s",
        ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), memorys_type.size(),
        op_desc->GetOutputsSize(), op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  is_op_reuse_mem_ = true;
  continuous_life_begin_ = 0;
  if (op_reuse_env_valid_ == true) {
    vector<string>::iterator it_name =
      std::find(op_no_reuse_mem_vec_.begin(), op_no_reuse_mem_vec_.end(), op_desc->GetName());
    vector<string>::iterator it_type =
      std::find(op_no_reuse_mem_vec_.begin(), op_no_reuse_mem_vec_.end(), op_desc->GetType());
    GE_IF_BOOL_EXEC(it_name != op_no_reuse_mem_vec_.end() || it_type != op_no_reuse_mem_vec_.end(),
                    is_op_reuse_mem_ = false;);
  }

  bool is_atomic = false;
  // If GetBool fail, is_atomic is false.
  (void)ge::AttrUtils::GetBool(op_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic);
  bool is_buffer_pool_mem_supported = (op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID)) &&
                                      (op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE)) && (!root_unknown_shape_flag_);
  // Allocate memory for the current node and release node memory of the same size in the workspace
  GE_IF_BOOL_EXEC(ge_disable_reuse_mem_env_ != "1",
                  for (auto iter = stream_workspace_blocks_.begin(); iter != stream_workspace_blocks_.end();
                       ++iter) { ReleaseMemorys(iter->second[stream_id], reusable_blocks_[iter->first][stream_id]);
                                 iter->second[stream_id].clear();});
  bool need_apply_continuous_memory = IsContinuousOutput(node) && (!is_buffer_pool_mem_supported);
  if (need_apply_continuous_memory) {
    return ApplyContinuousMemory(node, ranges, is_op_reuse_mem_);
  }
  for (uint32_t i = 0; i < static_cast<uint32_t>(op_desc->GetOutputsSize()); i++) {
    int64_t size = 0;
    auto output_op_desc = op_desc->GetOutputDescPtr(i);
    if (output_op_desc != nullptr) {
      GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(*output_op_desc, size) != SUCCESS, GELOGI("Get size failed"));
    }

    // fusion: other type's size not means malloc HBM memory
    bool l1_flag = has_mem_type_attr && memorys_type[i] == RT_MEMORY_L1;
    if (l1_flag) {
      GELOGI("fusion: node[%s], output[%s], output memory type [%ld]",
             op_desc->GetName().c_str(), op_desc->GetOutputNameByIndex(i).c_str(), memorys_type[i]);
      size = 0;
    }

    int32_t calc_type = 0;
    bool ret = ge::AttrUtils::GetInt(output_op_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    GE_IF_BOOL_EXEC((ret && (calc_type == static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY))), size = 0;);

    std::string peer_name;
    uint32_t peer_input_index = 0;
    bool out_node_set_continuous_input = false;
    bool reset_zero_copy_flag = false;
    bool no_need_assign_memory = ((size == 0) || CheckIsZeroMemNodeType(node->GetType()));
    if (!no_need_assign_memory) {
      out_node_set_continuous_input =
          IsOutNodeSetContinuousInput(node, i, peer_name, peer_input_index,
                                      no_need_assign_memory, reset_zero_copy_flag);
      GE_IF_BOOL_EXEC(!no_need_assign_memory,
          no_need_assign_memory = IsAtomicOutputMemory(node, i, is_atomic, out_node_set_continuous_input););
    }
    no_need_assign_memory = (no_need_assign_memory || IsKnownSubgraphData(node) || is_buffer_pool_mem_supported);
    if (no_need_assign_memory) {
      zero_memory_list_.emplace_back(node, kOutput, i, false);
      continue;
    }
    // atomic can't be reused
    bool need_change = is_op_reuse_mem_ && is_atomic;
    if (need_change) {
      is_op_reuse_mem_ = false;
    }

    MemoryBlock *mem_block = ApplyOutMemory(node, i, ranges, is_op_reuse_mem_, out_node_set_continuous_input);
    if (mem_block != nullptr) {
      GE_IF_BOOL_EXEC(reset_zero_copy_flag,
        mem_block->is_zero_copy_ = false;
        GELOGI("Node[%s] output[%u] need assign memory before reassign.", op_desc->GetName().c_str(), i););
      node_out_blocks_[node->GetName()].emplace_back(mem_block);
      if (out_node_set_continuous_input) {
        node_continuous_input_blocks_[peer_name][peer_input_index] = mem_block;
      }
      NodeIndexIO node_index_io(node, i, kOut);
      auto iter = anchor_to_symbol_.find(node_index_io.ToString());
      if (iter == anchor_to_symbol_.end()) {
        continue;
      }
      symbol_blocks_[iter->second] = mem_block;
      // The output is suspended, and will be released in allocation of next node.
      CheckAndReleaseSuspendedBlock(node, i, mem_block);
    }
  }
  return SUCCESS;
}

///
/// @ingroup domi
/// @brief traverse all nodes outputs and workspace in need, apply memory block considering memory reuse
/// @param [in/out] ranges memory size provided
/// @return Status result
///
void BlockMemAssigner::AssignMemoryWithReuse(vector<int64_t> &ranges) {
  (void)ge::GetContext().GetOption(OPTION_EXEC_DISABLE_REUSED_MEMORY, ge_disable_reuse_mem_env_);
  GEEVENT("Reuse memory %s", ge_disable_reuse_mem_env_ == "1" ? "close" : "open");
  string op_no_reuse_mem_str;
  const char *op_no_reuse_mem = std::getenv(OP_NO_REUSE_MEM);
  GE_IF_BOOL_EXEC(op_no_reuse_mem != nullptr, op_no_reuse_mem_str = string(op_no_reuse_mem);
                  CheckAndGetOpReuseEnv(op_no_reuse_mem_str, op_no_reuse_mem_vec_, op_reuse_env_valid_););
  auto root_graph = GraphUtils::FindRootGraph(compute_graph_);
  if (root_graph == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Check][RootGraph]Root graph is nullptr, graph:%s.", compute_graph_->GetName().c_str());
    REPORT_INNER_ERROR("E19999", "Root graph is nullptr, graph:%s.", compute_graph_->GetName().c_str());
    return;
  }
  root_unknown_shape_flag_ = root_graph->GetGraphUnknownFlag();

  for (NodePtr &n : compute_graph_->GetAllNodes()) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    life_time_ = node_op_desc->GetId();
    int64_t stream_id = node_op_desc->GetStreamId();
    if (AssignOutputMemoryWithReuse(n, ranges) != SUCCESS) {
      return;
    }
    vector<int64_t> temp;
    int64_t tatal_size = 0;
    GetNodeWorkSpaceSize(n, temp, tatal_size);
    vector<int64_t> workspace_bytes;
    vector<int64_t> tvm_workspace_memory_type;
    bool has_tvm_workspace_mem_type_attr =
      ge::AttrUtils::GetListInt(node_op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_memory_type);
    vector<bool> workspace_reuse_flag;
    GE_IF_BOOL_EXEC(!ge::AttrUtils::GetListBool(node_op_desc, kAttrNameWorkspaceReuseFlag, workspace_reuse_flag),
                    GELOGD("OP %s get workspace_reuse_flag attr failed", node_op_desc->GetName().c_str()));
    GELOGD("Assign memory node[%s], size [temp:%zu, memory type size:%zu]", node_op_desc->GetName().c_str(),
           temp.size(), tvm_workspace_memory_type.size());

    if (has_tvm_workspace_mem_type_attr && (temp.size() != tvm_workspace_memory_type.size())) {
      REPORT_INNER_ERROR("E19999", "Attr[%s]size:%zu is not equal to workspace size:%zu, node_name:%s",
                         TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), tvm_workspace_memory_type.size(),
                         temp.size(), n->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Attr]Attr %s size:%zu is not equal to workspace size:%zu, node_name:%s",
             TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), tvm_workspace_memory_type.size(),
             temp.size(), n->GetName().c_str());
      return;
    }
    for (size_t i = 0; i < temp.size(); i++) {
      // fusion: other type's size not means malloc HBM memory
      bool workspace_skip_flag = false;
      if (has_tvm_workspace_mem_type_attr && tvm_workspace_memory_type[i] == RT_MEMORY_L1) {
        GELOGI(
            "fusion:node[%s]workspace index[%zu] is not hbm type, add to zero_memory_list, workspace memory type [%ld]",
            node_op_desc->GetName().c_str(), i, tvm_workspace_memory_type[i]);
        workspace_skip_flag = true;
      }
      if (temp[i] == 0 || workspace_skip_flag) {
        zero_memory_list_.emplace_back(n, kWorkspace, static_cast<uint32_t>(i), false);
        continue;
      }
      int64_t memory_type = RT_MEMORY_HBM;
      if (!GetWorkSpaceMemoryType(n, i, memory_type)) {
        GELOGW("Get workspace memory type failed.");
        return;
      }
      MemoryBlock *mem_block = ApplyMemory(GetBlockSize(static_cast<size_t>(temp[i]), ranges),
                                           static_cast<size_t>(temp[i]), static_cast<size_t>(temp[i]),
                                           kWorkspace, n, static_cast<uint32_t>(i), workspace_reuse_flag,
                                           is_op_reuse_mem_, false, memory_type);
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(mem_block == nullptr, continue, "failed to apply memory block.");
      ++(mem_block->ref_count_);
      CheckWorkspaceReuse(workspace_reuse_flag, i, stream_id, mem_block, memory_type);
    }
    for (auto it = reusable_blocks_.begin(); it != reusable_blocks_.end(); ++it) {
      ReleaseInputNodeOutMemory(node_out_blocks_, it->second[stream_id], n);
    }
  }

  GELOGD("Assigned memory blocks:");
  for (auto mem_block : memory_blocks_) {
    GELOGD("%s", mem_block->String().c_str());
    (void)mem_block;  // Fix warning
  }

  GE_IF_BOOL_EXEC(!(ge_disable_reuse_mem_env_ == "1"), ReuseBlocksByLifeTime(ranges.size()));
  AssignContinuousBlocks();
  ResizeMemoryBlocks();

  GELOGD("Memory blocks after resize:");
  for (auto mem_block : memory_blocks_) {
    GELOGD("%s", mem_block->String().c_str());
    (void)mem_block;  // Fix warning
  }
}

void BlockMemAssigner::CheckWorkspaceReuse(const vector<bool> &workspace_reuse_flag, uint32_t index, int64_t stream_id,
                                           MemoryBlock *mem_block, int64_t memory_type) {
  bool reuse_mem_flag =
      ((workspace_reuse_flag.size() > index) && (workspace_reuse_flag[index] == false)) ? false : true;
  if (reuse_mem_flag) {
    stream_workspace_blocks_[memory_type][stream_id].emplace_back(mem_block);
  }
}

void BlockMemAssigner::GetNodeWorkSpaceSize(const NodePtr &node, vector<int64_t> &workspace_memory,
                                            int64_t &total_size) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(node->GetOpDesc() == nullptr, return, "Op desc is null.");
  vector<int64_t> workspace_byte_nums = node->GetOpDesc()->GetWorkspaceBytes();

  GELOGD("node[%s] size:%zu", node->GetOpDesc()->GetName().c_str(), workspace_byte_nums.size());
  for (int64_t byte_size : workspace_byte_nums) {
    workspace_memory.emplace_back(byte_size);
    total_size += byte_size;
    GELOGD("push back size:%ld", byte_size);
  }
}

// asending order
static bool CompareBlockIndex(MemoryBlock *left, MemoryBlock *right) {
  if (left == nullptr || right == nullptr) {
    return false;
  }
  if (left->input_index_ < right->input_index_) {
    return true;
  }
  return false;
}
///
/// @ingroup domi
/// @brief order blocks by continuous input index
/// @param [in] blocks need be processed
/// @param [in] input blocks need continuous
/// @param [out] blocks after continuous order
/// @param [in/out] blocks ordered
/// @param [in] input or output
///
void ReAssignContinuousBlocks(const std::vector<MemoryBlock *> &org_blocks,
                              const std::map<MemoryBlock *, uint32_t> block_map,
                              std::vector<MemoryBlock *> &dest_blocks, std::vector<MemoryBlock *> &continuous_blocks,
                              const std::string &type) {
  for (auto &memory_block : org_blocks) {
    if (memory_block == nullptr || memory_block->deleted_block_) {
      continue;
    }
    if (block_map.find(memory_block) != block_map.end()) {
      continue;
    }
    dest_blocks.emplace_back(memory_block);
  }

  // add continuous block
  std::sort(continuous_blocks.begin(), continuous_blocks.end(), CompareBlockIndex);
  size_t count = 0;
  for (auto &memory_block : continuous_blocks) {
    GE_IF_BOOL_EXEC(memory_block == nullptr, continue);

    GELOGI("Block continuous %s index:%d", type.c_str(), memory_block->input_index_);
    count++;
    if (count == 1) {
      memory_block->first_continuous_block_ = true;
    }
    if (count == continuous_blocks.size()) {
      memory_block->last_continuous_block_ = true;
    }
    dest_blocks.emplace_back(memory_block);
  }
}

void BlockMemAssigner::AssignContinuousBlocks() {
  for (auto &block_map : node_continuous_input_blocks_) {
    std::vector<MemoryBlock *> dest_memory_blocks;
    std::map<MemoryBlock *, uint32_t> continuous_block_map;
    std::vector<MemoryBlock *> continuous_blocks;
    auto it = node_continuous_input_counts_.find(block_map.first);
    GE_IF_BOOL_EXEC(it == node_continuous_input_counts_.end(), continue);
    GELOGI("Node:%s continuous input block count:%zu input count:%u", block_map.first.c_str(), block_map.second.size(),
           it->second);
    GE_IF_BOOL_EXEC(it->second != block_map.second.size(), continue);

    for (auto &it : block_map.second) {
      if (it.second != nullptr) {
        continuous_block_map[it.second] = it.first;
        it.second->input_index_ = it.first;
        continuous_blocks.emplace_back(it.second);
      }
    }
    if (continuous_block_map.size() != continuous_blocks.size()) {
      GELOGW("Node:%s continuous input map size:%zu vector size:%zu", block_map.first.c_str(),
             continuous_block_map.size(), continuous_blocks.size());
      continue;
    }
    ReAssignContinuousBlocks(memory_blocks_, continuous_block_map, dest_memory_blocks, continuous_blocks, "input");
    memory_blocks_.swap(dest_memory_blocks);
  }
}

void BlockMemAssigner::ReuseBlocksByLifeTime(size_t range_size) {
  // 1 means block size is same so no need to do this
  if (range_size <= 1) {
    return;
  }
  for (size_t i = 0; i < memory_blocks_.size(); ++i) {
    auto parent = memory_blocks_[i];
    if (parent == nullptr || parent->deleted_block_ || parent->continuous_block_) {
      continue;
    }
    if (parent->reuse_mem_ && !IsPostReuse(parent)) {
      parent->reuse_mem_ = false;
    }
    for (size_t j = i + 1; j < memory_blocks_.size(); ++j) {
      auto child = memory_blocks_[j];
      if (child == nullptr) {
        continue;
      }
      // If node is before atomic_addr_clean node, the continus memory can't be reused.
      if (!parent->NodeTypeIndexList().empty() && child->continuous_block_) {
        auto node = parent->NodeTypeIndexList()[0].node;
        if (node == nullptr || node->GetOpDesc() == nullptr || (node->GetOpDesc()->GetId() < GetAtomicAddrCleanId())) {
          continue;
        }
      }
      parent->AddLifeReuseBlock(child, total_node_depend_stream_life_);
    }
  }
}

void AddBlockMemOffset(size_t &mem_offset, size_t &p2p_mem_offset, MemoryBlock &block) {
  if (block.memory_type_ == RT_MEMORY_HBM) {
    if (block.first_continuous_block_) {
      mem_offset += MEM_ALIGN_SIZE;
    }
    block.Resize();
    block.SetHeadOffset(mem_offset);
    mem_offset += block.Size();
    block.SetTailOffset(mem_offset - 1);
  } else if (block.memory_type_ == RT_MEMORY_P2P_DDR) {
    if (block.first_continuous_block_) {
      p2p_mem_offset += MEM_ALIGN_SIZE;
    }
    block.Resize();
    block.SetHeadOffset(p2p_mem_offset);
    p2p_mem_offset += block.Size();
    block.SetTailOffset(p2p_mem_offset - 1);
  }
}

bool DynamicBatchBlockReuse(MemoryBlock &block) {
  return (block.IsSameBatchLabel() && block.reuse_mem_);
}

///
/// @ingroup domi_omg
/// @brief get max batch memory size, others reuse this block memory
/// @param [in&out] memory_blocks_ memory block, after calculating offset
/// |-dynamic batch block batch1|
/// |-dynamic batch block batch2----|
/// |-dynamic batch block batch3--|
///
void BlockMemAssigner::ResizeDynamicBatchBlocks() {
  std::map<std::string, std::vector<MemoryBlock *>> dynamic_batch_blocks;
  for (auto block : memory_blocks_) {
    if (block == nullptr) {
      continue;
    }
    // when memory is not reuseable, it can't be reused by different branch
    if (DynamicBatchBlockReuse(*block)) {
      dynamic_batch_blocks[block->batch_label_].emplace_back(block);
    }
  }

  size_t max_mem_offset = mem_offset_;
  size_t max_p2p_mem_offset = p2p_mem_offset_;
  for (auto &batch_blocks : dynamic_batch_blocks) {
    size_t mem_offset = mem_offset_;
    size_t p2p_mem_offset = p2p_mem_offset_;
    for (auto block : batch_blocks.second) {
      if (block == nullptr || block->deleted_block_ || block->is_zero_copy_) {
        continue;
      }
      AddBlockMemOffset(mem_offset, p2p_mem_offset, *block);
    }
    if (mem_offset > max_mem_offset) {
      max_mem_offset = mem_offset;
    }
    if (p2p_mem_offset > max_p2p_mem_offset) {
      max_p2p_mem_offset = p2p_mem_offset;
    }
    GELOGI("Batch[%s] offset[%zu] p2p_offset[%zu]", batch_blocks.first.c_str(), mem_offset, p2p_mem_offset);
  }
  mem_offset_ = max_mem_offset;
  p2p_mem_offset_ = max_p2p_mem_offset;
}

///
/// @ingroup domi_omg
/// @brief traverse memory size, resize, calculate offset
/// @param [in&out] memory_blocks_ memory block, after calculating offset
/// |-not dynamic batch block-||-dynamic batch block batch1|    |-zero copy block-|
/// |-not dynamic batch block-||-dynamic batch block batch2----||-zero copy block-|
/// |-not dynamic batch block-||-dynamic batch block batch3--|  |-zero copy block-|
///
void BlockMemAssigner::ResizeMemoryBlocks() {
  for (auto &memory_block : memory_blocks_) {
    if (memory_block == nullptr || memory_block->deleted_block_ || memory_block->is_zero_copy_
        || DynamicBatchBlockReuse(*memory_block)) {
      continue;
    }

    AddBlockMemOffset(mem_offset_, p2p_mem_offset_, *memory_block);
  }
  ResizeDynamicBatchBlocks();
  GELOGI("mem_offset_ exclude zero_copy_memory is %zu, p2p_mem_offset_ exclude zero_copy_memory is %zu,"
         "theory_min_memory_size %zu", mem_offset_, p2p_mem_offset_, theory_min_memory_size_);
}

///
/// @ingroup domi
/// @brief given NodeTypeIndex, set offset in Op's OpDef
/// @param [in&out] node_type_index <node, memory type, id>
/// @param [in] offset offset to be set
/// @param [in] size memory size
/// @param [in] real_size memory size in need
/// @return Status result
///
void SetOffsetSize(const NodeTypeIndex &node_type, const MemoryBlock *block,
                   size_t real_size, size_t no_align_size, int32_t child_block_level) {
  ge::OpDescPtr op_desc = node_type.node->GetOpDesc();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(op_desc == nullptr, return, "op_desc is null.");
  auto owner_graph = node_type.node->GetOwnerComputeGraph();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(owner_graph == nullptr, return, "owner_graph is null.");
  string graph_name = owner_graph->GetName();
  if (owner_graph->GetParentGraph() != nullptr) {
    graph_name = owner_graph->GetParentGraph()->GetName();
  }

  vector<int64_t> memorys_type;
  int64_t offset = block->HeadOffset();
  size_t end = node_type.life_time_end;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memorys_type);
  if (node_type.mem_type == kOutput) {
    vector<int64_t> output_list = op_desc->GetOutputOffset();
    for (auto i = static_cast<uint32_t>(output_list.size()); i < node_type.index + 1; i++) {
      output_list.emplace_back(kInvalidOffset);
    }
    if (output_list.empty()) {
      GELOGW("Empty output");
      return;
    }

    static const set<string> kSetOffsetTypes = { DATA_TYPE, AIPP_DATA_TYPE, MULTISHAPE, NETOUTPUT };
    if ((kSetOffsetTypes.count(op_desc->GetType()) > 0) && !IsKnownSubgraphData(node_type.node)) {
      if ((output_list[node_type.index] == kInvalidOffset) || (output_list[node_type.index] < offset)) {
        output_list.at(node_type.index) = offset;
      }
    } else {
      // fusion: keep the original other type offset value from op_desc
      bool set_out_offset = (!has_mem_type_attr) ||
        (memorys_type.size() > node_type.index && memorys_type[node_type.index] != RT_MEMORY_L1);
      if (set_out_offset) {
        output_list.at(node_type.index) = offset;
      }
    }
    op_desc->SetOutputOffset(output_list);
  } else if (node_type.mem_type == kWorkspace) {
    vector<int64_t> workspace_list;
    workspace_list = op_desc->GetWorkspace();
    for (auto i = static_cast<uint32_t>(workspace_list.size()); i < node_type.index + 1; i++) {
      workspace_list.emplace_back(kInvalidOffset);
    }
    vector<int64_t> workspace_mem_type;
    bool has_workspace_mem_type = ge::AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, workspace_mem_type);
    // fusion: keep the original other type offset value from op_desc
    bool set_workspace_offset = (!has_workspace_mem_type) ||
      (workspace_mem_type.size() > node_type.index && workspace_mem_type[node_type.index] != RT_MEMORY_L1);
    if (set_workspace_offset) {
      workspace_list.at(node_type.index) = offset;
    }
    op_desc->SetWorkspace(workspace_list);
  }
  GELOGI("[IMAS]Set %s name[%s] optype[%s] %s[%u] offset to [%ld] streamid[%ld] memtype[%ld] size[%zu] realsize[%zu] "
         "noalignsize[%zu] life time begin[%s] life time end[%zu] child[%d:%d:%d:%d:%d] isref[%d] batch[%s] scope[%d]",
         graph_name.c_str(), op_desc->GetName().c_str(), node_type.node->GetType().c_str(),
         node_type.GetMemType().c_str(), node_type.index, offset, op_desc->GetStreamId(),block->memory_type_,
         block->Size(), real_size, no_align_size, node_type.GetLifeBeginDesc().c_str(), end, child_block_level,
         block->reuse_mem_, block->continuous_block_, block->is_zero_copy_, block->same_stream_, node_type.ref_input,
         block->batch_label_.c_str(), node_type.thread_scope_id);
}

void SetBlockOpMemOffset(MemoryBlock *block, int32_t child_block_level) {
  if (block == nullptr) {
    return;
  }
  size_t index = 0;
  size_t real_size = 0;
  size_t no_align_size = 0;
  auto real_size_list_size = block->RealSizeList().size();
  for (const NodeTypeIndex &node_type_index : block->NodeTypeIndexList()) {
    if (index < real_size_list_size) {
      real_size = block->RealSizeList()[index];
      no_align_size = block->NoAlignSizeList()[index];
    }
    SetOffsetSize(node_type_index, block, real_size, no_align_size, child_block_level);
    index++;
  }

  child_block_level++;
  for (MemoryBlock *child_block : block->ChildBlockList()) {
    SetBlockOpMemOffset(child_block, child_block_level);
  }
}

void BlockMemAssigner::SetOpMemOffset(bool is_zero_copy) {
  for (MemoryBlock *memory_block : memory_blocks_) {
    if (memory_block == nullptr || memory_block->deleted_block_) {
      continue;
    }

    if ((is_zero_copy && !memory_block->is_zero_copy_) || (!is_zero_copy && memory_block->is_zero_copy_)) {
      continue;
    }

    SetBlockOpMemOffset(memory_block, 0);
  }

  if (!is_zero_copy) {
    for (const NodeTypeIndex &node_type_index : zero_memory_list_) {
      MemoryBlock block(0, 0);
      SetOffsetSize(node_type_index, &block, 0, 0, 0);
    }
  }
}

Status BlockMemAssigner::Assign() {
  vector<int64_t> ranges;
  if (GetMemoryRanges(ranges) != SUCCESS) {
    GELOGE(FAILED, "[Get][MemoryRanges] Fail!");
    return FAILED;
  }
  GE_IF_BOOL_EXEC(ranges.empty(), return SUCCESS);
  AssignMemoryWithReuse(ranges);

  SetOpMemOffset(false);

  return SUCCESS;
}

bool BlockMemAssigner::CheckIsZeroMemNodeType(const string &node_type) const {
  return (node_type == VARIABLE) || (node_type == CONSTANT) || (node_type == MULTISHAPE) ||
         (node_type == CONSTANTOP) || (node_type == HVDWAIT);
}

bool BlockMemAssigner::GetWorkSpaceMemoryType(const NodePtr &node, size_t index, int64_t &memory_type) {
  memory_type = RT_MEMORY_HBM;
  vector<int64_t> workspace_memory_type;
  auto op_desc = node->GetOpDesc();
  bool has_workspace_mem_type_attr =
      ge::AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, workspace_memory_type);
  if (has_workspace_mem_type_attr && (workspace_memory_type.size() <= index)) {
    REPORT_INNER_ERROR("E19999", "get workspace mem_type failed, "
                       "index %zu invalid, bigger than attr %s size:%zu, node_name:%s",
                       index, TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(),
                       workspace_memory_type.size(), node->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][WorkspaceMemType]index %zu invalid, bigger than attr %s size:%zu, node_name:%s",
           index, TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), workspace_memory_type.size(), node->GetName().c_str());
    return false;
  }
  memory_type = has_workspace_mem_type_attr ? workspace_memory_type[index] : RT_MEMORY_HBM;
  return true;
}
}  // namespace ge
