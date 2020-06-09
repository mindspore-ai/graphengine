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

#include "graph/build/memory/block_mem_assigner.h"
#include <algorithm>
#include <sstream>

#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "graph/buffer.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_context.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

#include "graph/debug/ge_attr_define.h"

#include "graph/optimize/common/params.h"
#include "omg/omg_inner_types.h"
#include "runtime/mem.h"

namespace {
const char *const kAttrNameWorkspaceReuseFlag = "workspace_reuse_flag";
const char *const kL2FusionDynamicConvergeOp = "l2fusion_dynamic_converge_op";
const char *const kDisableReuseMemory = "ge.exec.disableReuseMemory";
const char *const OP_NO_REUSE_MEM = "OP_NO_REUSE_MEM";
const int kReuseMaxCount = 10;
const int kReuseMaxOpNum = 10;
const int kReuseMaxCharNum = 2000;
}  // namespace

namespace ge {
using std::map;
using std::pair;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::unordered_set;
using std::vector;

void MemoryBlock::Resize() {
  auto iter = std::max_element(real_size_list_.begin(), real_size_list_.end());
  if (iter == real_size_list_.end()) {
    GELOGW("real_size_list_ is empty");
    return;
  } else {
    size_t block_size = *iter;
    if ((block_size > 0) && (block_size % MEM_ALIGN_SIZE != 0)) {
      block_size = (block_size + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
    }
    block_size_ = block_size;
    if (last_continuous_block_) {
      block_size_ += MEM_ALIGN_SIZE;
    }
  }
}

bool MemoryBlock::IsSameLabel(std::string &first_batch_label) {
  if (node_type_index_list_.empty()) {
    return false;
  }

  auto node_op_desc = node_type_index_list_[0].node->GetOpDesc();
  if (node_op_desc == nullptr) {
    return false;
  }
  // not all op has ATTR_NAME_BATCH_LABEL, no need check return value, only check out parameter
  (void)ge::AttrUtils::GetStr(node_op_desc, ATTR_NAME_BATCH_LABEL, first_batch_label);
  if (first_batch_label.empty()) {
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
    (void)ge::AttrUtils::GetStr(index_op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
    if (first_batch_label != batch_label) {
      all_same_label = false;
      break;
    }
  }
  return all_same_label;
}

void SetLastUsedInputMemAttr(NodePtr &node, int input_index) {
  if (node == nullptr) {
    return;
  }
  auto node_op_desc = node->GetOpDesc();
  if (node_op_desc != nullptr) {
    auto input_desc = node_op_desc->GetInputDesc(input_index);
    if (!ge::AttrUtils::SetInt(input_desc, ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true)) {
      GELOGW("Set %s input[%d] ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE to true failed.", node_op_desc->GetName().c_str(),
             input_index);
      return;
    }
    GELOGD("Set %s input[%d] ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE to true success.", node_op_desc->GetName().c_str(),
           input_index);
    if (node_op_desc->UpdateInputDesc(input_index, input_desc) != GRAPH_SUCCESS) {
      GELOGW("Update %s input[%d] desc failed.", node_op_desc->GetName().c_str(), input_index);
    }
  }
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
  ss << "Block size: " << Size() << " from " << HeadOffset() << " to " << TailOffset() << "";
  ss << "real_size_list: " << ToString(real_size_list_) << "";
  ss << "ref_count: " << ref_count_ << "";
  ss << "members: ";
  for (auto x : NodeTypeIndexList()) {
    ss << "__node: " << ToString(x) << "";
  }
  for (const auto &symbol : SymbolList()) {
    ss << "__symbol: " << symbol << "";
  }
  return ss.str();
}

BlockMemAssigner::BlockMemAssigner(ge::ComputeGraphPtr compute_graph)
    : mem_offset_(0), compute_graph_(std::move(compute_graph)) {}

BlockMemAssigner::~BlockMemAssigner() {
  for (MemoryBlock *memory_block : memory_blocks_) {
    GE_DELETE_NEW_SINGLE(memory_block);
  }
}

void BlockMemAssigner::GetOutAndWorkSpaceMem(vector<int64_t> &all_memory_size) {
  if (GraphUtils::GetRefMapping(compute_graph_, symbol_to_anchors_, anchor_to_symbol_) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Get ref-mapping for graph %s failed.", compute_graph_->GetName().c_str());
    return;
  }

  vector<int64_t> temp;
  for (const NodePtr &n : compute_graph_->GetAllNodes()) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);

    if (node_op_desc->GetType() == ATOMICADDRCLEAN) {
      atomic_addr_clean_id_ = node_op_desc->GetId();
    }

    for (auto &out_anchor : n->GetAllOutDataAnchors()) {
      GeTensorDesc output_desc = node_op_desc->GetOutputDesc(out_anchor->GetIdx());
      bool reuse_input = false;
      GE_IF_BOOL_EXEC(ge::TensorUtils::GetReuseInput(output_desc, reuse_input) != SUCCESS,
                      GELOGI("Get reuse_input failed"));

      if (!reuse_input) {
        int64_t size = 0;
        GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(output_desc, size) != SUCCESS, GELOGI("Get size failed"));
        if (anchor_to_symbol_.empty()) {
          all_memory_size.emplace_back(size);
        } else {
          auto iter1 = anchor_to_symbol_.find(NodeIndexIO(n, out_anchor->GetIdx(), kOut).ToString());
          if (iter1 == anchor_to_symbol_.end()) {
            continue;
          }
          std::string symbol = iter1->second;
          auto iter2 = symbol_size_.find(symbol);
          if (iter2 == symbol_size_.end()) {
            symbol_size_[symbol] = size;
          } else if (size > static_cast<int64_t>(iter2->second)) {
            iter2->second = size;
          }
        }
      }
    }
    temp.clear();
    GetNodeWorkSpaceSize(n, temp);
    all_memory_size.insert(all_memory_size.end(), temp.begin(), temp.end());
  }
  GELOGI("The last atomic_addr_clean node id: %ld", atomic_addr_clean_id_);
  for (auto &pair : symbol_size_) {
    all_memory_size.emplace_back(pair.second);
  }
  sort(all_memory_size.begin(), all_memory_size.end());
  GELOGI("All memory size: %s", ToString(all_memory_size).c_str());

  for (auto iter = all_memory_size.begin(); iter != all_memory_size.end();) {
    if (*iter == 0) {
      iter = all_memory_size.erase(iter);
    } else {
      ++iter;
    }
  }

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
  return 0;
}

bool IsDirectOutputNode(const NodePtr &node, int idx) {
  if ((node != nullptr) && (node->GetOpDesc() != nullptr) && (node->GetOpDesc()->GetType() == NETOUTPUT)) {
    GELOGI("This is netoutput node, the input node mem can not be reused");
    return true;
  }
  return false;
}

void AddReusableBlockCount(const MemoryBlock &mem_block, map<string, uint64_t> &reusable_block_counts) {
  string key = std::to_string(mem_block.Size());
  key += "_" + std::to_string(mem_block.stream_id_);
  auto it = reusable_block_counts.find(key);
  if (it != reusable_block_counts.end()) {
    it->second++;
  } else {
    reusable_block_counts[key] = 1;
  }
}

void ReduceReusableBlockCount(const MemoryBlock &mem_block, map<string, uint64_t> &reusable_block_counts) {
  string key = std::to_string(mem_block.Size());
  key += "_" + std::to_string(mem_block.stream_id_);
  auto it = reusable_block_counts.find(key);
  if (it != reusable_block_counts.end()) {
    if (it->second > 0) {
      it->second--;
    }
  }
}

bool CanReuseBySize(const map<string, uint64_t> &reusable_block_counts, const MemoryBlock &reusable_block,
                    size_t block_size, size_t real_size, bool continuous, int64_t atomic_addr_clean_id) {
  bool can_reuse = false;

  // If node is before atomic_addr_clean node, the continus memory can't be reused.
  if (!reusable_block.NodeTypeIndexList().empty()) {
    auto node = reusable_block.NodeTypeIndexList()[0].node;
    if (node != nullptr) {
      auto op_desc = node->GetOpDesc();
      if (op_desc != nullptr) {
        if ((op_desc->GetId() < atomic_addr_clean_id) && continuous) {
          return false;
        }
      }
    }
  }

  // continuous memory case:only real_size is maximum can be reused and only one continuous memory in one block
  if (continuous || reusable_block.continuous_block_) {
    auto it = std::max_element(std::begin(reusable_block.RealSizeList()), std::end(reusable_block.RealSizeList()));
    if (it != std::end(reusable_block.RealSizeList())) {
      GE_IF_BOOL_EXEC((continuous && reusable_block.continuous_block_) || (continuous && (real_size < *it)) ||
                        (reusable_block.continuous_block_ && (real_size > *it)),
                      GELOGD("Conflict current block size:%zu continuous:%d, reuse block max size:%zu continuous:%d",
                             real_size, continuous, *it, reusable_block.continuous_block_);
                      return false;);
    }
  }
  if (reusable_block.Size() == block_size) {
    can_reuse = true;
  } else {
    string key = std::to_string(reusable_block.Size());
    key += "_" + std::to_string(reusable_block.stream_id_);
    auto it = reusable_block_counts.find(key);
    GE_IF_BOOL_EXEC(
      (it != reusable_block_counts.end() && (it->second > kReuseMaxCount)) && (reusable_block.Size() > block_size),
      can_reuse = true;
      GELOGD("Less size mem reuse, reuse block size:%zu, current block size:%zu", reusable_block.Size(), block_size););
  }
  return can_reuse;
}

bool CanReuseByStream(const std::unordered_set<int64_t> &reuse_stream, MemoryBlock &reusable_block) {
  bool can_reuse = false;
  if (reuse_stream.find(reusable_block.stream_id_) != reuse_stream.cend()) {
    can_reuse = true;
  }
  return can_reuse;
}

bool BlockMemAssigner::IsOutNodeSetContinuousInput(const NodePtr &n, uint32_t out_index, std::string &peer_name,
                                                   uint32_t &peer_input_index) {
  if (n == nullptr || n->GetAllOutDataAnchors().size() <= 0) {
    return false;
  }
  if (static_cast<size_t>(out_index) < n->GetAllOutDataAnchors().size()) {
    auto out_anchor = n->GetOutDataAnchor(out_index);
    GE_IF_BOOL_EXEC(out_anchor == nullptr,
                    GELOGE(FAILED, "Node[%s] output[%u] anchor is null.", n->GetName().c_str(), out_index);
                    return false;);
    for (auto const &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(peer_in_anchor == nullptr,
                      GELOGE(FAILED, "Node[%s] output[%u] peer_in_anchor 0 is null.", n->GetName().c_str(), out_index);
                      return false;);
      auto peer_node = peer_in_anchor->GetOwnerNode();
      GE_IF_BOOL_EXEC(peer_node == nullptr,
                      GELOGE(FAILED, "Node[%s] output[%u] node is null.", n->GetName().c_str(), out_index);
                      return false;);

      // Get the continuous input type of the node, default is false
      bool is_input_continuous = false;
      auto peer_in_node_desc = peer_node->GetOpDesc();
      GE_IF_BOOL_EXEC(peer_in_node_desc == nullptr,
                      GELOGE(FAILED, "Node[%s] output[%u] nodedesc is null.", n->GetName().c_str(), out_index);
                      return false;);

      // If GetBool fail, is_input_continuous is false.
      (void)ge::AttrUtils::GetBool(peer_in_node_desc, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
      if (is_input_continuous) {
        if (n->GetOwnerComputeGraph() != nullptr) {
          string graph_name = n->GetOwnerComputeGraph()->GetName();
          GELOGI("%s name[%s] output[%u] node[%s] set input[%d] continuous, input size[%u].", graph_name.c_str(),
                 n->GetName().c_str(), out_index, peer_in_node_desc->GetName().c_str(), peer_in_anchor->GetIdx(),
                 peer_node->GetAllInDataAnchorsSize());
          // Only set attr one times.
          if (node_continuous_input_blocks_[peer_in_node_desc->GetName()].size() == 0) {
            (void)ge::AttrUtils::SetBool(peer_in_node_desc, ATTR_NAME_CONTINUOUS_INPUT_ALLOC, true);
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

///
/// @ingroup GE
/// @brief Check pre_reuse flag & post_reuse glag for each symbol
/// @return void
///
void BlockMemAssigner::InitReuseFlag() {
  static const std::set<std::string> kPreReuseTypes = {ge::DATA_TYPE, ge::AIPP_DATA_TYPE, ge::ANN_DATA_TYPE,
                                                       ge::NETOUTPUT, ge::PROPOSAL,       ge::ZEROSLIKE,
                                                       ge::CONSTANT,  ge::CONSTANTOP};
  static const std::set<std::string> kPostReuseTypes = {ge::DATA_TYPE, ge::AIPP_DATA_TYPE, ge::ENTER,
                                                        ge::REFENTER,  ge::NEXTITERATION,  ge::REFNEXTITERATION};
  for (auto &pair : symbol_to_anchors_) {
    std::string symbol = pair.first;
    bool pre_reuse_flag = true;
    bool post_reuse_flag = true;
    for (auto &node_index_io : pair.second) {
      if (node_index_io.io_type == kIn) {
        continue;
      }

      OutDataAnchorPtr out_anchor = node_index_io.node->GetOutDataAnchor(node_index_io.index);
      if (out_anchor == nullptr) {
        continue;
      }

      bool out_flg = false;
      if (node_index_io.node->GetOutDataNodes().empty()) {
        out_flg = true;
      }
      for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
        if (IsDirectOutputNode(in_anchor->GetOwnerNode(), in_anchor->GetIdx())) {
          out_flg = true;
          break;
        }
      }
      std::string type = out_anchor->GetOwnerNode()->GetType();
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

  std::string symbol = iter1->second;
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
  for (auto &symbol : mem_block->SymbolList()) {
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
/// @return bool
///
bool BlockMemAssigner::IsSymbolExist(const NodeIndexIO &node_index_io) {
  auto iter = anchor_to_symbol_.find(node_index_io.ToString());
  if (iter == anchor_to_symbol_.end()) {
    return false;
  }
  std::string symbol = iter->second;
  return symbol_blocks_.find(symbol) != symbol_blocks_.end();
}

///
/// @ingroup GE
/// @brief Print symbol
/// @return void
///
void BlockMemAssigner::PrintSymbolMap() {
  for (auto &pair : symbol_to_anchors_) {
    GELOGD("symbol=%s, max_size=%zu, pre_reuse=%s, post_reuse=%s", pair.first.c_str(), symbol_size_[pair.first],
           pre_reuse_flag_[pair.first] ? "true" : "false", post_reuse_flag_[pair.first] ? "true" : "false");
    for (auto &node_index_io : pair.second) {
      GELOGD("anchor:%s", node_index_io.ToString().c_str());
    }
  }
}

MemoryBlock *BlockMemAssigner::ApplyMemory(size_t block_size, size_t real_size, MemoryType mem_type, const NodePtr &n,
                                           uint32_t out_index, const vector<bool> &workspace_reuse_flag,
                                           const bool is_op_reuse_mem, const bool continuous) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(n == nullptr, return nullptr, "Input parameter n is null.");
  auto node_op_desc = n->GetOpDesc();
  GE_IF_BOOL_EXEC(node_op_desc == nullptr, return nullptr);

  string ge_disable_reuse_mem_env = "0";
  (void)ge::GetContext().GetOption(kDisableReuseMemory, ge_disable_reuse_mem_env);
  if (ge_disable_reuse_mem_env != "1") {
    bool reuse_mem_flag = !((workspace_reuse_flag.size() > out_index) && !workspace_reuse_flag[out_index]);
    bool is_reuse_memory = !node_op_desc->HasAttr(kL2FusionDynamicConvergeOp) && reuse_mem_flag && is_op_reuse_mem &&
                           (IsPreReuse(n, out_index));
    auto stream_id = node_op_desc->GetStreamId();
    auto map_iter = reusable_streams_map_.find(stream_id);
    if (is_reuse_memory && map_iter != reusable_streams_map_.end()) {
      for (auto it = reusable_blocks_.begin(); it != reusable_blocks_.end(); ++it) {
        MemoryBlock *reusable_block = *it;
        if (!IsPostReuse(reusable_block)) {
          continue;
        }

        // A node can reuse blocks of the same stream and preorder streams
        auto id = GetAtomicAddrCleanId();
        if (CanReuseBySize(reusable_block_counts_, *reusable_block, block_size, real_size, continuous, id) &&
            CanReuseByStream(map_iter->second, *reusable_block)) {
          GELOGD("Cross stream mem reuse, target stream:%ld, current stream:%ld", reusable_block->stream_id_,
                 stream_id);
          reusable_block->AddNodeTypeIndex({n, mem_type, out_index}, real_size);
          if (mem_type == kOutput) {
            auto iter = anchor_to_symbol_.find(NodeIndexIO(n, out_index, kOut).ToString());
            if (iter != anchor_to_symbol_.end()) {
              reusable_block->AddSymbol(iter->second);
            }
          }
          reusable_block->continuous_block_ = continuous;
          reusable_block->ref_count_++;
          ReduceReusableBlockCount(*reusable_block, reusable_block_counts_);
          reusable_blocks_.erase(it);
          return reusable_block;
        }
      }
    }
  }

  auto block = new (std::nothrow) MemoryBlock(block_size, is_op_reuse_mem);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(block == nullptr, return nullptr, "new an object failed.");

  // Data and netoutput need zero copy block
  if ((node_op_desc->GetType() == DATA_TYPE && !continuous) || (node_op_desc->GetType() == NETOUTPUT)) {
    block->is_zero_copy_ = true;
  }

  block->Init(real_size, mem_type, n, out_index);
  block->stream_id_ = node_op_desc->GetStreamId();
  block->ref_count_++;
  block->continuous_block_ = continuous;
  if (mem_type == kOutput) {
    auto iter = anchor_to_symbol_.find(NodeIndexIO(n, out_index, kOut).ToString());
    if (iter != anchor_to_symbol_.end()) {
      block->AddSymbol(iter->second);
    }
  }
  memory_blocks_.emplace_back(block);
  return block;
}

MemoryBlock *BlockMemAssigner::ApplyOutMemory(const NodePtr &n, uint32_t index, const vector<int64_t> &ranges,
                                              const bool is_op_reuse_mem, const bool continuous) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(n == nullptr, return nullptr, "input node is null.");
  auto node_op_desc = n->GetOpDesc();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(node_op_desc == nullptr, return nullptr, "node_op_desc is null.");
  MemoryBlock *block = nullptr;
  NodeIndexIO node_index_io = NodeIndexIO(n, index, kOut);
  int64_t size = 0;
  auto output_op_desc = node_op_desc->GetOutputDescPtr(index);
  if (output_op_desc != nullptr) {
    GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(*output_op_desc, size) != SUCCESS, GELOGI("Get size failed"));
  }

  if (IsSymbolExist(node_index_io)) {
    std::string symbol = anchor_to_symbol_[node_index_io.ToString()];
    block = symbol_blocks_[symbol];
    block->AddNodeTypeIndex({n, kOutput, index}, size);
    block->ref_count_++;
  } else {
    int64_t max_size = size;
    auto iter1 = anchor_to_symbol_.find(node_index_io.ToString());
    if (iter1 != anchor_to_symbol_.end()) {
      auto iter2 = symbol_size_.find(iter1->second);
      if (iter2 != symbol_size_.end()) {
        max_size = iter2->second;
      }
    }
    auto block_size = GetBlockSize(max_size, ranges);
    vector<bool> workspace_reuse_flag;
    block = ApplyMemory(block_size, size, kOutput, n, index, workspace_reuse_flag, is_op_reuse_mem, continuous);
  }
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(block == nullptr, return nullptr, "Block is nullptr.");
  int out_count_reuse_input = block->ref_count_;
  int out_count = 0;
  GE_IF_BOOL_EXEC(index >= n->GetAllOutDataAnchors().size(), GELOGE(FAILED, "index is out of range."); return nullptr);
  auto out_data_anchor = n->GetOutDataAnchor(index);
  GE_IF_BOOL_EXEC(out_data_anchor == nullptr, GELOGE(FAILED, "Out data anchor is nullptr."); return nullptr);
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
  bool reuse_input = false;
  for (const auto &in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    auto owner_node = in_anchor->GetOwnerNode();
    GE_IF_BOOL_EXEC(owner_node == nullptr, continue);
    auto op_desc = owner_node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
    for (uint32_t i = 0; i < static_cast<uint32_t>(op_desc->GetOutputsSize()); i++) {
      bool dst_reuse_input = false;
      uint32_t dst_reuse_input_index = 0;
      auto owner_node_op_desc = op_desc->GetOutputDescPtr(i);
      GE_IF_BOOL_EXEC(owner_node_op_desc == nullptr, continue);
      GE_IF_BOOL_EXEC(ge::TensorUtils::GetReuseInput(*owner_node_op_desc, dst_reuse_input) != SUCCESS,
                      GELOGI("Get dst_reuse_input failed"));
      GE_IF_BOOL_EXEC(ge::TensorUtils::GetReuseInputIndex(*owner_node_op_desc, dst_reuse_input_index) != SUCCESS,
                      GELOGI("Get dst_reuse_input_index failed"));
      if (dst_reuse_input && (dst_reuse_input_index == static_cast<uint32_t>(in_anchor->GetIdx()))) {
        block->AddNodeTypeIndex({owner_node, kOutput, i}, block->Size());
        out_count_reuse_input += 1;
        reuse_input = true;
      }
    }
  }
  block->ref_count_ = reuse_input ? out_count_reuse_input + out_count - 1 : out_count;
  return block;
}

bool IsOutputBlock(const ge::InDataAnchorPtr &in_data_anchor) {
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, GELOGE(FAILED, "Peer out anchor is nullptr."); return false);
  auto src = peer_out_anchor->GetOwnerNode();
  int32_t index = peer_out_anchor->GetIdx();
  auto iter = domi::GetContext().out_nodes_map.find(src->GetName());
  if (iter != domi::GetContext().out_nodes_map.end()) {
    for (auto id : iter->second) {
      if (index == id) {
        return true;
      }
    }
  }
  return false;
}

// current node's output uses previous node's output memory
bool IsReferencePreviousNodeOutputMemory(const ge::NodePtr &node, uint32_t output_index) {
  // Get the reference type of the node, default is false
  bool is_ref = false;
  // If GetBool fail, is_ref is false.
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (!is_ref) {
    return false;
  }
  const string &output_name = op_desc->GetOutputNameByIndex(output_index);
  for (const auto &input_name : op_desc->GetAllInputNames()) {
    if (!input_name.empty() && output_name == input_name) {
      int input_index = op_desc->GetInputIndexByName(input_name);
      GELOGI("Reference memory:name[%s] output[%s][%u] ref to input[%s][%d] ", op_desc->GetName().c_str(),
             output_name.c_str(), output_index, input_name.c_str(), input_index);
      return true;
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
          GELOGD("[IMAS]Atomic no assign %s name[%s] output[%d] streamid[%ld].", graph_name.c_str(),
                 op_desc->GetName().c_str(), index, op_desc->GetStreamId());
        }
        return true;
      }
    }
  }
  return false;
}

void BlockMemAssigner::ReleaseMemory(MemoryBlock *to_release, vector<MemoryBlock *> &reusable_memory) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(to_release == nullptr, return, "Input parameter to_release is null.");
  GE_CHK_TRUE_EXEC_INFO(to_release->ref_count_ <= 0, return, "Release memory");
  GE_CHK_TRUE_EXEC_INFO(!to_release->reuse_mem_, return, "doesn't reuse memory");
  --to_release->ref_count_;
  if (to_release->ref_count_ == 0) {
    reusable_memory.emplace_back(to_release);
    AddReusableBlockCount(*to_release, reusable_block_counts_);
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

      if ((node_type_indexs.back().node == in_anchor->GetPeerOutAnchor()->GetOwnerNode()) &&
          (node_type_indexs.back().index == static_cast<uint32_t>(in_anchor->GetPeerOutAnchor()->GetIdx())) &&
          (node->GetOpDesc()->GetStreamId() == block->stream_id_)) {
        ReleaseMemory(block, reusable_memory);
        if (block->ref_count_ == 0) {
          SetLastUsedInputMemAttr(node, in_anchor->GetIdx());
        }
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

Status BlockMemAssigner::AssignOutputMemoryWithReuse(const NodePtr &node, vector<int64_t> &ranges) {
  auto node_op_desc = node->GetOpDesc();
  int64_t stream_id = node_op_desc->GetStreamId();
  vector<int64_t> memorys_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(node_op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memorys_type);
  GELOGI("Assign memory node[%s], output size[%d], output memory type size[%d]", node_op_desc->GetName().c_str(),
         node_op_desc->GetOutputsSize(), memorys_type.size());
  if (has_mem_type_attr && (memorys_type.size() != node_op_desc->GetOutputsSize())) {
    GELOGE(INTERNAL_ERROR, "fusion: node[%s], output memory size err[outputsize:%zu, memorysize:%zu]",
           node_op_desc->GetName().c_str(), node_op_desc->GetOutputsSize(), memorys_type.size());
    return INTERNAL_ERROR;
  }

  is_op_reuse_mem_ = true;
  if (op_reuse_env_valid_ == true) {
    vector<string>::iterator it_name =
      std::find(op_no_reuse_mem_vec_.begin(), op_no_reuse_mem_vec_.end(), node_op_desc->GetName());
    vector<string>::iterator it_type =
      std::find(op_no_reuse_mem_vec_.begin(), op_no_reuse_mem_vec_.end(), node_op_desc->GetType());
    GE_IF_BOOL_EXEC(it_name != op_no_reuse_mem_vec_.end() || it_type != op_no_reuse_mem_vec_.end(),
                    is_op_reuse_mem_ = false;);
  }

  bool is_atomic = false;
  // If GetBool fail, is_atomic is false.
  (void)ge::AttrUtils::GetBool(node_op_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic);
  // Allocate memory for the current node and release node memory of the same size in the workspace
  GE_IF_BOOL_EXEC(ge_disable_reuse_mem_env_ != "1",
                  ReleaseMemorys(stream_workspace_blocks_[stream_id], reusable_blocks_);)
  for (uint32_t i = 0; i < static_cast<uint32_t>(node_op_desc->GetOutputsSize()); i++) {
    int64_t size = 0;
    auto output_op_desc = node_op_desc->GetOutputDescPtr(i);
    if (output_op_desc != nullptr) {
      GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(*output_op_desc, size) != SUCCESS, GELOGI("Get size failed"));
    }
    // fusion: other type's size not means malloc HBM memory
    bool l1_flag = has_mem_type_attr && memorys_type[i] == RT_MEMORY_L1;
    if (l1_flag) {
      GELOGI("fusion: node[%s], output[%s], output memory type [%d]", node_op_desc->GetName().c_str(),
             node_op_desc->GetOutputNameByIndex(i).c_str(), memorys_type[i]);
      size = 0;
    }
    std::string peer_name;
    uint32_t peer_input_index = 0;
    bool out_node_set_continuous_input = false;
    bool no_need_assign_memory =
      ((size == 0) || CheckIsZeroMemNodeType(node->GetType()) || IsReferencePreviousNodeOutputMemory(node, i));
    if (!no_need_assign_memory) {
      out_node_set_continuous_input = IsOutNodeSetContinuousInput(node, i, peer_name, peer_input_index);
      no_need_assign_memory = IsAtomicOutputMemory(node, i, is_atomic, out_node_set_continuous_input);
    }
    if (no_need_assign_memory) {
      zero_memory_list_.emplace_back(node, kOutput, i);
      continue;
    }
    bool reuse_mem = is_op_reuse_mem_;
    // atomic can't be reused
    if (is_op_reuse_mem_ && out_node_set_continuous_input && is_atomic) {
      reuse_mem = false;
    }
    MemoryBlock *mem_block = ApplyOutMemory(node, i, ranges, reuse_mem, out_node_set_continuous_input);
    if (mem_block != nullptr) {
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
  // Init reusable streams map
  InitReusableStreamMap();

  (void)ge::GetContext().GetOption(kDisableReuseMemory, ge_disable_reuse_mem_env_);
  GEEVENT("Reuse memory %s", ge_disable_reuse_mem_env_ == "1" ? "close" : "open");
  string op_no_reuse_mem_str;
  const char *op_no_reuse_mem = std::getenv(OP_NO_REUSE_MEM);
  GE_IF_BOOL_EXEC(op_no_reuse_mem != nullptr, op_no_reuse_mem_str = string(op_no_reuse_mem);
                  CheckAndGetOpReuseEnv(op_no_reuse_mem_str, op_no_reuse_mem_vec_, op_reuse_env_valid_););

  for (NodePtr &n : compute_graph_->GetAllNodes()) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    int64_t stream_id = node_op_desc->GetStreamId();
    if (AssignOutputMemoryWithReuse(n, ranges) != SUCCESS) {
      return;
    }

    stream_workspace_blocks_[stream_id].clear();
    vector<int64_t> temp;
    GetNodeWorkSpaceSize(n, temp);
    vector<int64_t> workspace_bytes;
    vector<int64_t> workspace_memory_type;
    bool has_workspace_mem_type_attr =
      ge::AttrUtils::GetListInt(node_op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, workspace_memory_type);
    vector<bool> workspace_reuse_flag;
    GE_IF_BOOL_EXEC(!ge::AttrUtils::GetListBool(node_op_desc, kAttrNameWorkspaceReuseFlag, workspace_reuse_flag),
                    GELOGD("OP %s get workspace_reuse_flag attr failed", node_op_desc->GetName().c_str()));
    GELOGI("Assign memory node[%s], size [temp:%zu, memory type size:%zu]", node_op_desc->GetName().c_str(),
           temp.size(), workspace_memory_type.size());

    if (has_workspace_mem_type_attr && (temp.size() != workspace_memory_type.size())) {
      GELOGE(INTERNAL_ERROR, "fusion: node[%s], workspace_memory size err![v_temp:%zu, workspace:%zu]",
             n->GetName().c_str(), temp.size(), workspace_memory_type.size());
      return;
    }
    for (size_t i = 0; i < temp.size(); i++) {
      // fusion: other type's size not means malloc HBM memory
      bool workspace_skip_flag = false;
      if (has_workspace_mem_type_attr && workspace_memory_type[i] == RT_MEMORY_L1) {
        GELOGI(
          "fusion: node[%s]workspace index[%d] is not hbm type, add to zero_memory_list, workspace memory type [%ld]",
          node_op_desc->GetName().c_str(), i, workspace_memory_type[i]);
        workspace_skip_flag = true;
      }
      if (temp[i] == 0 || workspace_skip_flag) {
        zero_memory_list_.emplace_back(n, kWorkspace, static_cast<uint32_t>(i));
        continue;
      }
      MemoryBlock *mem_block =
        ApplyMemory(GetBlockSize(static_cast<size_t>(temp[i]), ranges), static_cast<size_t>(temp[i]), kWorkspace, n,
                    static_cast<uint32_t>(i), workspace_reuse_flag, is_op_reuse_mem_, false);
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(mem_block == nullptr, continue, "failed to apply memory block.");
      CheckWorkspaceReuse(workspace_reuse_flag, i, stream_id, mem_block);
    }
    ReleaseInputNodeOutMemory(node_out_blocks_, reusable_blocks_, n);
  }

  GELOGD("Assigned memory blocks:");
  for (auto mem_block : memory_blocks_) {
    GELOGD("%s", mem_block->String().c_str());
    (void)mem_block;  // Fix warning
  }

  GE_IF_BOOL_EXEC(!(ge_disable_reuse_mem_env_ == "1"), MergeDynamicBatchBlocks();)
  AssignContinuousBlocks();
  ResizeMemoryBlocks();

  GELOGD("Memory blocks after resize:");
  for (auto mem_block : memory_blocks_) {
    GELOGD("%s", mem_block->String().c_str());
    (void)mem_block;  // Fix warning
  }
}

void BlockMemAssigner::CheckWorkspaceReuse(const vector<bool> &workspace_reuse_flag, uint32_t index, int64_t stream_id,
                                           MemoryBlock *mem_block) {
  bool reuse_mem_flag =
    ((workspace_reuse_flag.size() > index) && (workspace_reuse_flag[index] == false)) ? false : true;
  if (reuse_mem_flag) {
    stream_workspace_blocks_[stream_id].emplace_back(mem_block);
  }
}

void BlockMemAssigner::GetNodeWorkSpaceSize(const NodePtr &node, vector<int64_t> &workspace_memory) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(node->GetOpDesc() == nullptr, return, "Op desc is null.");
  vector<int64_t> workspace_byte_nums = node->GetOpDesc()->GetWorkspaceBytes();

  GELOGD("GetNodeWorkSpaceSize: node[%s] size:%zu", node->GetOpDesc()->GetName().c_str(), workspace_byte_nums.size());
  for (int64_t byte_size : workspace_byte_nums) {
    workspace_memory.emplace_back(byte_size);
    GELOGD("GetNodeWorkSpaceSize: push back size:%ld", byte_size);
  }
}

// descending order
static bool CompareBlockMaxSize(MemoryBlock *left, MemoryBlock *right) {
  if (left == nullptr || right == nullptr) {
    return false;
  }
  auto left_max_size = std::max_element(left->RealSizeList().begin(), left->RealSizeList().end());
  if (left_max_size != left->RealSizeList().end()) {
    auto right_max_size = std::max_element(right->RealSizeList().begin(), right->RealSizeList().end());
    if (right_max_size == right->RealSizeList().end() || (*left_max_size > *right_max_size)) {
      return true;
    }
  }
  return false;
}

void MergeBlocks(std::vector<MemoryBlock *> &dest, std::vector<MemoryBlock *> &src) {
  for (size_t i = 0; i < dest.size(); ++i) {
    if (i >= src.size()) {
      return;
    }
    if (dest[i] != nullptr && src[i] != nullptr) {
      for (auto &symbol : src[i]->SymbolList()) {
        dest[i]->AddSymbol(symbol);
      }
      for (size_t j = 0; j < src[i]->NodeTypeIndexList().size(); ++j) {
        dest[i]->AddNodeTypeIndex(src[i]->NodeTypeIndexList()[j], src[i]->RealSizeList()[j]);
        src[i]->deleted_block_ = true;
      }
    }
  }
}

void BlockMemAssigner::MergeDynamicBatchBlocks() {
  std::map<std::string, std::vector<MemoryBlock *>> dynamic_batch_blocks;
  for (auto block : memory_blocks_) {
    if (block == nullptr) {
      continue;
    }
    std::string batch_label;
    if (block->IsSameLabel(batch_label)) {
      dynamic_batch_blocks[batch_label].emplace_back(block);
    }
  }

  auto it = dynamic_batch_blocks.begin();
  auto it_max = it;

  // find max block counts
  for (; it != dynamic_batch_blocks.end(); ++it) {
    if (it->second.size() > it_max->second.size()) {
      it_max = it;
    }
    std::sort(it->second.begin(), it->second.end(), CompareBlockMaxSize);
  }
  if (it_max != dynamic_batch_blocks.end()) {
    GELOGD("MergeDynamicBatch %s block counts %zu", it_max->first.c_str(), it_max->second.size());
  }
  for (it = dynamic_batch_blocks.begin(); it != dynamic_batch_blocks.end(); ++it) {
    if (it != it_max) {
      GELOGD("MergeDynamicBatch from %s to %s", it->first.c_str(), it_max->first.c_str());
      MergeBlocks(it_max->second, it->second);
    }
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
///
void ReAssignContinuousBlocks(const std::vector<MemoryBlock *> &org_blocks,
                              const std::map<MemoryBlock *, uint32_t> block_map,
                              std::vector<MemoryBlock *> &dest_blocks, std::vector<MemoryBlock *> &continuous_blocks) {
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

    GELOGI("Block continuous input index:%d", memory_block->input_index_);
    count++;
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
    ReAssignContinuousBlocks(memory_blocks_, continuous_block_map, dest_memory_blocks, continuous_blocks);
    memory_blocks_.swap(dest_memory_blocks);
  }
}

///
/// @ingroup domi_omg
/// @brief traverse memory size, resize, calculate offset
/// @param [in&out] memory_blocks_ memory block, after calculating offset
///
void BlockMemAssigner::ResizeMemoryBlocks() {
  for (auto &memory_block : memory_blocks_) {
    if (memory_block == nullptr || memory_block->deleted_block_ || memory_block->is_zero_copy_) {
      continue;
    }
    memory_block->Resize();
    memory_block->SetHeadOffset(mem_offset_);
    mem_offset_ += memory_block->Size();
    memory_block->SetTailOffset(mem_offset_ - 1);
    GELOGI("mem_offset_ exclude zero_copy_memory is %zu.", mem_offset_);
  }
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
void SetOffsetSize(const NodeTypeIndex &node_type_index, int64_t offset, size_t size, size_t real_size) {
  ge::OpDescPtr op_desc = node_type_index.node->GetOpDesc();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(op_desc == nullptr, return, "op_desc is null.");
  string graph_name = node_type_index.node->GetOwnerComputeGraph()->GetName();
  vector<int64_t> memorys_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memorys_type);
  if (node_type_index.mem_type == kOutput) {
    vector<int64_t> output_list = op_desc->GetOutputOffset();
    for (auto i = static_cast<uint32_t>(output_list.size()); i < node_type_index.index + 1; i++) {
      output_list.emplace_back(kInvalidOffset);
    }
    if (output_list.empty()) {
      GELOGW("Empty output");
      return;
    }

    if ((op_desc->GetType() == DATA) || (op_desc->GetType() == AIPP_DATA_TYPE) || (op_desc->GetType() == MULTISHAPE) ||
        (op_desc->GetType() == NETOUTPUT)) {
      if ((output_list[node_type_index.index] == kInvalidOffset) || (output_list[node_type_index.index] < offset)) {
        output_list.at(node_type_index.index) = offset;
      }
    } else {
      // fusion: keep the original other type offset value from op_desc
      bool set_out_offset = (!has_mem_type_attr) || (memorys_type[node_type_index.index] != RT_MEMORY_L1);
      if (set_out_offset) {
        output_list.at(node_type_index.index) = offset;
      }
    }
    op_desc->SetOutputOffset(output_list);
    GELOGI("[IMAS]Set %s name[%s] output[%d] offset to [%ld] streamid[%ld] size[%zu] realsize[%zu].",
           graph_name.c_str(), op_desc->GetName().c_str(), node_type_index.index, offset, op_desc->GetStreamId(), size,
           real_size);
  } else if (node_type_index.mem_type == kWorkspace) {
    vector<int64_t> workspace_list;
    workspace_list = op_desc->GetWorkspace();
    for (auto i = static_cast<uint32_t>(workspace_list.size()); i < node_type_index.index + 1; i++) {
      workspace_list.emplace_back(kInvalidOffset);
    }
    vector<int64_t> workspace_memory_type;
    bool has_workspace_mem_type_attr =
      ge::AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, workspace_memory_type);
    // fusion: keep the original other type offset value from op_desc
    bool set_workspace_offset =
      (!has_workspace_mem_type_attr) || (workspace_memory_type[node_type_index.index] != RT_MEMORY_L1);
    if (set_workspace_offset) {
      workspace_list.at(node_type_index.index) = offset;
    }
    op_desc->SetWorkspace(workspace_list);
    GELOGI("[IMAS]Set %s name[%s] workspace[%u] offset to [%ld] streamid[%ld] size[%zu] realsize[%zu].",
           graph_name.c_str(), op_desc->GetName().c_str(), node_type_index.index, offset, op_desc->GetStreamId(), size,
           real_size);
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

    size_t index = 0;
    size_t real_size = 0;
    auto real_size_list_size = memory_block->RealSizeList().size();
    for (const NodeTypeIndex &node_type_index : memory_block->NodeTypeIndexList()) {
      if (index < real_size_list_size) {
        real_size = memory_block->RealSizeList()[index];
      }
      SetOffsetSize(node_type_index, memory_block->HeadOffset(), memory_block->Size(), real_size);
      index++;
    }
  }

  if (!is_zero_copy) {
    for (const NodeTypeIndex &node_type_index : zero_memory_list_) {
      SetOffsetSize(node_type_index, 0, 0, 0);
    }
  }
}

Status BlockMemAssigner::Assign() {
  vector<int64_t> ranges;
  if (GetMemoryRanges(ranges) != SUCCESS) {
    GELOGE(FAILED, "GetMemoryRanges Fail!");
    return FAILED;
  }
  GE_IF_BOOL_EXEC(ranges.empty(), return SUCCESS);
  AssignMemoryWithReuse(ranges);

  SetOpMemOffset(false);

  return SUCCESS;
}

void BlockMemAssigner::InitReusableStreamMap() {
  // save a stream's id and its first Node and last node.
  map<int64_t, pair<NodePtr, NodePtr>> stream_head_tail_node_map;
  // save a stream's id and its directly child stream.
  map<int64_t, unordered_set<int64_t>> stream_dependency_map;
  // save a stream's id and its occupied memory.
  unordered_map<int64_t, int64_t> stream_mem_map;

  // Find streams's first and last node.
  FindHeadAndTailNodesForStream(stream_head_tail_node_map, stream_mem_map);

  // If streamB's first node is the output of streamA's last node, then B depends on A.
  FindDependentStream(stream_head_tail_node_map, stream_dependency_map);

  // If a stream has more than one child stream, select the one that occupies the closest memory
  for (const auto &iter : stream_dependency_map) {
    if (iter.second.empty()) {
      continue;
    }
    int64_t target_size = stream_mem_map[iter.first];
    int64_t min_size_gap = LONG_MAX;
    int64_t target_reuse_stream_id = 0;
    for (auto id : iter.second) {
      if (labs(stream_mem_map[id] - target_size) < min_size_gap) {
        target_reuse_stream_id = id;
        min_size_gap = labs(stream_mem_map[id] - target_size);
      }
    }
    // If b can reuse a, then b should also be able to reuse all blocks that a can reuse.
    reusable_streams_map_[target_reuse_stream_id].insert(reusable_streams_map_[iter.first].begin(),
                                                         reusable_streams_map_[iter.first].end());
  }
}

void BlockMemAssigner::FindHeadAndTailNodesForStream(map<int64_t, pair<NodePtr, NodePtr>> &stream_head_tail_node_map,
                                                     unordered_map<int64_t, int64_t> &stream_mem_map) {
  for (const auto &n : compute_graph_->GetAllNodes()) {
    GE_IF_BOOL_EXEC(n->GetOpDesc() == nullptr, GELOGW("Op desc is nullptr"); continue);
    auto stream_id = n->GetOpDesc()->GetStreamId();
    // traverse to find streams's first and last node.
    if (stream_head_tail_node_map.find(stream_id) == stream_head_tail_node_map.end()) {
      stream_head_tail_node_map[stream_id] = std::make_pair(n, n);
      reusable_streams_map_[stream_id].insert(stream_id);  // a node can reuse blocks from same stream.
    } else {
      stream_head_tail_node_map[stream_id].second = n;
    }

    // Accumulate the output size of the node in the stream.
    for (size_t i = 0; i < n->GetOpDesc()->GetOutputsSize(); i++) {
      int64_t size = 0;
      if (ge::TensorUtils::GetSize(*n->GetOpDesc()->GetOutputDescPtr(static_cast<uint32_t>(i)), size) != SUCCESS) {
        GELOGW("Get output size failed！");
        continue;
      }
      stream_mem_map[stream_id] += size;
    }
    // Accumulate the workspace size of the node in the stream.
    for (auto size : n->GetOpDesc()->GetWorkspaceBytes()) {
      stream_mem_map[stream_id] += size;
    }
  }
}

void BlockMemAssigner::FindDependentStream(map<int64_t, pair<NodePtr, NodePtr>> &stream_head_tail_node_map,
                                           map<int64_t, unordered_set<int64_t>> &stream_dependency_map) {
  for (const auto &it1 : stream_head_tail_node_map) {
    for (const auto &it2 : stream_head_tail_node_map) {
      if (it1 == it2) {
        continue;
      }
      NodePtr pre_node = it1.second.second;
      NodePtr post_node = it2.second.first;
      std::vector<NodePtr> out_nodes;
      // Direct link out_node
      for (const auto &out_node : pre_node->GetOutNodes()) {
        if ((out_node->GetOpDesc() == nullptr) || (post_node->GetOpDesc() == nullptr) ||
            (pre_node->GetOpDesc() == nullptr)) {
          continue;
        }
        out_nodes.emplace_back(out_node);
      }

      FindDependentStreamBetweenGraphs(pre_node, out_nodes);

      for (auto &out_node : out_nodes) {
        if (out_node->GetOpDesc()->GetId() == post_node->GetOpDesc()->GetId()) {
          stream_dependency_map[pre_node->GetOpDesc()->GetStreamId()].insert(post_node->GetOpDesc()->GetStreamId());
        }
      }
    }
  }
}

///
/// @ingroup GE
/// @brief Find dependent link between parent_graph and sub_graph
/// @param [in] pre_node
/// @param [out] out_nodes
/// @return void
/// @author
///
void BlockMemAssigner::FindDependentStreamBetweenGraphs(const NodePtr &pre_node, std::vector<NodePtr> &out_nodes) {
  if ((pre_node == nullptr) || (pre_node->GetOpDesc() == nullptr)) {
    return;
  }

  // FunctionOp & subgraph input
  std::vector<std::string> subgraph_names = pre_node->GetOpDesc()->GetSubgraphInstanceNames();
  for (auto &subgraph_name : subgraph_names) {
    ComputeGraphPtr subgraph = compute_graph_->GetSubgraph(subgraph_name);
    if (subgraph == nullptr) {
      continue;
    }
    for (auto &node : subgraph->GetDirectNode()) {
      OpDescPtr op_desc = node->GetOpDesc();
      if (op_desc == nullptr) {
        continue;
      }
      if (op_desc->HasAttr(ATTR_NAME_PARENT_NODE_INDEX)) {
        out_nodes.emplace_back(node);
      }
    }
  }

  // subgraph output & parent_node output
  if (NodeUtils::IsSubgraphOutput(pre_node)) {
    NodePtr parent_node = pre_node->GetOwnerComputeGraph()->GetParentNode();
    for (const auto &out_node : parent_node->GetOutNodes()) {
      out_nodes.emplace_back(out_node);
    }
  }
}

bool BlockMemAssigner::CheckIsZeroMemNodeType(const string &node_type) const {
  return (node_type == VARIABLE) || (node_type == CONSTANT) || (node_type == MULTISHAPE) ||
         (node_type == HCOMBROADCAST) || (node_type == HCOMALLREDUCE) || (node_type == CONSTANTOP) ||
         (node_type == ASSIGNADD) || (node_type == ASSIGNSUB) || (node_type == ASSIGN);
}
}  // namespace ge
