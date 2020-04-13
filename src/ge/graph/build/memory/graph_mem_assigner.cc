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

#include "graph/build/memory/graph_mem_assigner.h"
#include <cstring>
#include <set>
#include "framework/common/debug/ge_log.h"
#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace {
const int kAllInputAddrIsAtomic = -1;
}  // namespace
namespace ge {
Status VariableMemoryAssigner::Assign() {
  Status result = ge::VarMemAssignUtil::AssignConstantOpMemory(compute_graph_);
  if (result != ge::SUCCESS) {
    return result;
  }

  result = ge::VarMemAssignUtil::AssignVarMemory(compute_graph_);
  if (result != ge::SUCCESS) {
    return result;
  }
  return ge::SUCCESS;
}

Status VariableMemoryAssigner::AssignVarAttr2Nodes() {
  Status result = ge::VarMemAssignUtil::AssignVarAttr2Nodes(compute_graph_);
  if (result != ge::SUCCESS) {
    return result;
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignMemory() {
  auto mem_assigner = std::unique_ptr<ge::HybridMemAssigner>(new (std::nothrow) ge::HybridMemAssigner(compute_graph_));
  if (mem_assigner == nullptr) {
    GELOGE(ge::FAILED, "Alloc HybridMemAssigner failed.");
    return ge::FAILED;
  }
  if (mem_assigner->Assign() != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Memory assigner failed");
    return ge::FAILED;
  }
  MemoryOffset memory_offset(RT_MEMORY_HBM, mem_assigner->GetMemOffset());
  memory_offset_.push_back(memory_offset);

  auto session_id = compute_graph_->GetSessionID();
  int64_t var_size_before_assign = ge::VarManager::Instance(session_id)->GetVarMemSize(RT_MEMORY_HBM);
  auto variable_assigner =
    std::unique_ptr<ge::VariableMemoryAssigner>(new (std::nothrow) ge::VariableMemoryAssigner(compute_graph_));
  if (variable_assigner == nullptr) {
    GELOGE(ge::FAILED, "Alloc VariableMemoryAssigner failed.");
    return ge::FAILED;
  }

  if (variable_assigner->Assign() != ge::SUCCESS) {
    return ge::FAILED;
  }
  int64_t var_size_assign = ge::VarManager::Instance(session_id)->GetVarMemSize(RT_MEMORY_HBM) - var_size_before_assign;
  GELOGI("GraphMemoryAssigner::AssignMemory variable size = %ld", var_size_assign);
  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::AssignVarAttr2Nodes() {
  auto variable_assigner =
    std::unique_ptr<ge::VariableMemoryAssigner>(new (std::nothrow) ge::VariableMemoryAssigner(compute_graph_));
  if (variable_assigner == nullptr) {
    GELOGE(ge::FAILED, "Alloc VariableMemoryAssigner failed.");
    return ge::FAILED;
  }
  if (variable_assigner->AssignVarAttr2Nodes() != ge::SUCCESS) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::ReAssignMemory(bool is_loop_graph, size_t &mem_offset) {
  if (memory_offset_.empty()) {
    GELOGE(FAILED, "memory_offset_ is empty.");
    return ge::FAILED;
  }

  GE_CHK_STATUS_RET(ReAssignContinuousMemory(is_loop_graph), "ReAssignContinuousMemory Failed!");

  GE_CHK_STATUS_RET(ReAssignVirtualConcatMemory(), "ReAssignVirtualConcatMemory Failed!");

  GE_CHK_STATUS_RET(ReAssignMergeMemory(), "ReAssignMergeMemory Failed!");

  GE_CHK_STATUS_RET(ReAssignAtomicMemory(is_loop_graph), "ReAssignAtomicMemory Failed!");

  mem_offset = memory_offset_[0].mem_offset_;

  if (mem_offset > VarManager::Instance(0)->GetGraphMemoryMaxSize()) {
    GELOGE(ge::FAILED, "Current memoffset %zu is greater than memory manager malloc max size %zu", mem_offset,
           VarManager::Instance(0)->GetGraphMemoryMaxSize());
    return ge::FAILED;
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignContinuousMemory(bool is_loop_graph) {
  GELOGI("Begin to reassign continuous memory");
  Status ret;
  for (auto &node : compute_graph_->GetDirectNode()) {
    // Get the continuous input type of the node, default is false
    bool is_input_continuous = false;
    GE_CHECK_NOTNULL(node->GetOpDesc());
    // If GetBool fail, is_input_continuous is false.
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
    int64_t mem_clean_start = memory_offset_[0].mem_offset_;
    // Assign continuous input memory
    if (is_input_continuous) {
      ret = AssignContinuousInputMemory(node);
      if (ret != ge::SUCCESS) {
        GELOGE(ret, "Assign continuous input memory failed!");
        return ret;
      }

      memory_offset_[0].mem_offset_ += kMemAlignSize;

      // Clean up atomic address, eg, hcom node
      vector<int32_t> input_indexes;
      // If GetListInt fail, input_indexes is empty.
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, input_indexes);

      if (!input_indexes.empty() && input_indexes[0] == kAllInputAddrIsAtomic) {
        // check whether there is an atomic conflict between the current node and the peer out node
        if (!CheckInputIsSupportAtomic(node)) {
          GELOGE(ge::FAILED,
                 "There is an atomic conflict between the current node and the peer out node, not supported!");
          return ge::FAILED;
        } else if (is_loop_graph) {
          GE_CHK_STATUS_RET(SetLoopGraphAtomicAttr(node, mem_clean_start));
        } else {
          int64_t mem_clean_size = memory_offset_[0].mem_offset_ - mem_clean_start;
          GE_CHK_STATUS_RET(SetAtomicCleanAttr(nullptr, mem_clean_start, mem_clean_size), "SetAtomicCleanAttr failed.");
        }
      }
    }

    // Get the reference type of the node, default is false
    bool is_ref = false;
    // If GetBool fail, is_ref is false.
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_REFERENCE, is_ref);

    // Get the continuous output type of the node, default is false
    bool is_output_continuous = false;
    // If GetBool fail, is_output_continuous is false.
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, is_output_continuous);

    // If the output is ref type and refers to the ref of an input, the name of the output
    // and the input are the same. Ge encounters ref type, finds matching relationship according
    // to the names of input and output, and allocates the same memory address, eg: HCOMBroadcast
    if (is_ref) {
      ret = AssignReferenceMemory(node);
      if (ret != ge::SUCCESS) {
        GELOGE(ret, "Assign reference memory failed!");
        return ret;
      }
    } else if (is_output_continuous) {  // Assign continuous output memory
      ret = AssignContinuousOutputMemory(node);
      if (ret != ge::SUCCESS) {
        GELOGE(ret, "Assign reference memory failed!");
        return ret;
      }
    }
  }

  GELOGI("After reassign continuous memory, memoffset = %zu.", memory_offset_[0].mem_offset_);
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignContinuousInputMemory(const ge::NodePtr &node) {
  GELOGI("Current node %s needs continuous input.", node->GetName().c_str());
  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();

    if (peer_out_data_anchor == nullptr) {
      continue;
    }
    auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
    GE_IF_BOOL_EXEC(peer_op_desc == nullptr, continue);
    bool is_peer_output_continuous = false;
    // If GetBool fail, is_peer_output_continuous is false.
    (void)ge::AttrUtils::GetBool(peer_op_desc, ATTR_NAME_CONTINUOUS_OUTPUT, is_peer_output_continuous);

    // Get peer node output size, if size == 1(peer node has only one output), continuous input of the node and
    // continuous output of the previous node is the same, we can support it. If size != 1, there may be
    // conflict between the two, we can not support it.
    auto peer_output_size = peer_op_desc->GetOutputsSize();
    if (is_peer_output_continuous && (peer_output_size != 1)) {
      GELOGE(ge::PARAM_INVALID,
             "Current node %s requires continuous input, while the previous node %s requires "
             "continuous output. There may be conflict between the two. This node is not supported now.",
             node->GetOpDesc()->GetName().c_str(), peer_op_desc->GetName().c_str());
      return ge::PARAM_INVALID;
    }

    bool is_peer_reference = false;
    // If GetBool fail, is_peer_reference is false.
    (void)ge::AttrUtils::GetBool(peer_op_desc, ATTR_NAME_REFERENCE, is_peer_reference);

    if (is_peer_reference) {
      GELOGE(ge::PARAM_INVALID,
             "Current node %s requires continuous input, while the previous node %s requires "
             "reference. There may be conflict between the two. This node is not supported now.",
             node->GetOpDesc()->GetName().c_str(), peer_op_desc->GetName().c_str());
      return ge::PARAM_INVALID;
    }

    vector<int64_t> output_list = peer_op_desc->GetOutputOffset();
    if (peer_out_data_anchor->GetIdx() < static_cast<int>(output_list.size())) {
      output_list.at(peer_out_data_anchor->GetIdx()) = memory_offset_[0].mem_offset_;
    } else {
      GELOGE(ge::FAILED, "index : %d is out of range.", peer_out_data_anchor->GetIdx());
      return ge::FAILED;
    }
    peer_op_desc->SetOutputOffset(output_list);
    uint32_t tensor_desc_size = 0;
    if (ge::TensorUtils::GetSize(*(peer_op_desc->GetOutputDescPtr(peer_out_data_anchor->GetIdx())), tensor_desc_size) !=
        ge::SUCCESS) {
      GELOGE(FAILED, "GetSize failed.");
      return FAILED;
    }

    memory_offset_[0].mem_offset_ += tensor_desc_size;
    AlignMemOffset(kMemAlignSize);
  }

  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignContinuousOutputMemory(const ge::NodePtr &node) {
  GELOGI("Current node %s needs continuous output.", node->GetName().c_str());
  auto out_op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(out_op_desc == nullptr, GELOGE(ge::FAILED, "out_op_desc is null."); return ge::FAILED);
  vector<int64_t> output_list = out_op_desc->GetOutputOffset();

  if (out_op_desc->GetOutputsSize() > output_list.size()) {
    GELOGE(ge::FAILED, "The size %zu of node output desc is more than output_list's size %zu.",
           out_op_desc->GetOutputsSize(), output_list.size());
    return ge::FAILED;
  }

  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    output_list[out_data_anchor->GetIdx()] = memory_offset_[0].mem_offset_;
    uint32_t tensor_desc_size = 0;
    if (ge::TensorUtils::GetSize(*(out_op_desc->GetOutputDescPtr(out_data_anchor->GetIdx())), tensor_desc_size) !=
        ge::SUCCESS) {
      GELOGE(FAILED, "GetSize failed.");
      return FAILED;
    }
    memory_offset_[0].mem_offset_ += tensor_desc_size;

    AlignMemOffset(kMemAlignSize);
  }

  out_op_desc->SetOutputOffset(output_list);
  memory_offset_[0].mem_offset_ += kMemAlignSize;
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::ReAssignVirtualConcatMemory() {
  for (const auto &n : compute_graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(n->GetOpDesc());
    if (n->GetOpDesc()->GetType() == CONCAT) {
      int64_t is_node_virtual;
      GE_IF_BOOL_EXEC(
        !(ge::AttrUtils::GetInt(n->GetOpDesc(), "fusion_virtual_op", is_node_virtual)),  // Need  to  change
        continue;);
      vector<int64_t> output_list = n->GetOpDesc()->GetOutputOffset();
      if (output_list.empty()) {
        GELOGE(FAILED, "Outputoffset is empty node name:%s", n->GetName().c_str());
        return FAILED;
      }
      output_list.at(0) = memory_offset_[0].mem_offset_;
      n->GetOpDesc()->SetOutputOffset(output_list);
      GELOGD("Set Concat %s output offset to %zu.", n->GetOpDesc()->GetName().c_str(), memory_offset_[0].mem_offset_);

      size_t extra_memory_size = 0;
      for (const auto &in_data_anchor : n->GetAllInDataAnchors()) {
        auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
        if (peer_out_data_anchor != nullptr) {
          for (const auto &next_in_data_anchor : peer_out_data_anchor->GetPeerInDataAnchors()) {
            if (in_data_anchor->GetOwnerNode()->GetName() == next_in_data_anchor->GetOwnerNode()->GetName()) {
              auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
              GE_CHECK_NOTNULL(peer_op_desc);
              vector<int64_t> output_offsets = peer_op_desc->GetOutputOffset();
              if (peer_out_data_anchor->GetIdx() < static_cast<int>(output_offsets.size())) {
                output_offsets.at(peer_out_data_anchor->GetIdx()) = memory_offset_[0].mem_offset_;
              } else {
                GELOGE(ge::FAILED, "index : %d is out of range.", peer_out_data_anchor->GetIdx());
                return ge::FAILED;
              }
              peer_op_desc->SetOutputOffset(output_offsets);
              ge::ConstGeTensorDescPtr output_desc = peer_op_desc->GetOutputDescPtr(peer_out_data_anchor->GetIdx());
              GE_CHECK_NOTNULL(output_desc);
              int64_t output_mem_size = 0;

              // calculate tensor real size
              GeShape output_shape = output_desc->GetShape();
              Format format = output_desc->GetFormat();
              DataType data_type = output_desc->GetDataType();
              graphStatus graph_status =
                TensorUtils::CalcTensorMemSize(output_shape, format, data_type, output_mem_size);
              if (graph_status != GRAPH_SUCCESS) {
                GELOGE(graph_status, "CalcTensorMemSize failed!");
                return FAILED;
              }

              if ((output_mem_size > UINT32_MAX) || (output_mem_size < 0)) {
                GELOGE(FAILED,
                       "After calc virtual concat tensor memory size, output_mem_size = %ld, "
                       "out of data range [0, %u]",
                       output_mem_size, UINT32_MAX);
                return FAILED;
              }

              uint32_t size = static_cast<uint32_t>(output_mem_size);
              memory_offset_[0].mem_offset_ += size;
              uint32_t out_size = 0;
              if (ge::TensorUtils::GetSize(*(peer_op_desc->GetOutputDescPtr(peer_out_data_anchor->GetIdx())),
                                           out_size) != ge::SUCCESS) {
                GELOGE(FAILED, "GetSize failed.");
                return FAILED;
              }
              extra_memory_size = extra_memory_size + out_size - size;
            }
          }
        }
      }
      memory_offset_[0].mem_offset_ += extra_memory_size;
    }
  }

  GELOGI("After reassign virtual concat memory, memoffset = %zu.", memory_offset_[0].mem_offset_);
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignMergeMemory() {
  for (const ge::NodePtr &n : compute_graph_->GetDirectNode()) {
    GE_IF_BOOL_EXEC(n->GetOpDesc() == nullptr, continue);
    string node_type;
    GE_CHK_STATUS_RET(GetOriginalType(n, node_type), "Get node type fail.");
    if (node_type != STREAMMERGE) {
      continue;
    }

    vector<std::pair<int, NodePtr>> input_node_list;
    for (const auto &in_anchor : n->GetAllInDataAnchors()) {
      ge::OutDataAnchorPtr out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor == nullptr) {
        std::string in_name;
        GE_IF_BOOL_EXEC(ge::AttrUtils::GetStr(n->GetOpDesc(), ATTR_NAME_NEXT_ITERATION, in_name) && !in_name.empty(), {
          ge::NodePtr in_node = compute_graph_->FindNode(in_name);
          GE_CHECK_NOTNULL(in_node);
          input_node_list.emplace_back(std::make_pair(0, in_node));
        });
        continue;
      }
      ge::NodePtr src_node = out_anchor->GetOwnerNode();
      input_node_list.emplace_back(std::make_pair(out_anchor->GetIdx(), src_node));
    }

    int64_t data_output_offset = -1;
    int64_t max_output_size = -1;
    for (auto &iter : input_node_list) {
      int index = iter.first;
      NodePtr src_node = iter.second;
      GE_CHECK_NOTNULL(src_node->GetOpDesc());
      int64_t tmp_output_size = src_node->GetOpDesc()->GetOutputDesc(index).GetShape().GetShapeSize();
      if ((data_output_offset == -1) || (tmp_output_size > max_output_size)) {
        vector<int64_t> output_list = src_node->GetOpDesc()->GetOutputOffset();
        int output_size = static_cast<int>(output_list.size());
        if (index >= output_size) {
          GELOGE(INTERNAL_ERROR, "out_anchor[%d] >= output_list[%d]", index, output_size);
          return INTERNAL_ERROR;
        }

        data_output_offset = output_list[index];
        max_output_size = tmp_output_size;
      }
      GELOGD("merge=%s, input=%s, size=%ld, offset=%ld, max_size=%ld", n->GetName().c_str(),
             src_node->GetName().c_str(), tmp_output_size, data_output_offset, max_output_size);
    }

    vector<int64_t> input_list;
    for (auto &iter : input_node_list) {
      int index = iter.first;
      NodePtr src_node = iter.second;
      GE_CHECK_NOTNULL(src_node->GetOpDesc());
      vector<int64_t> output_list = src_node->GetOpDesc()->GetOutputOffset();
      int output_size = static_cast<int>(output_list.size());
      if (index >= output_size) {
        GELOGE(INTERNAL_ERROR, "out_anchor[%d] >= output_list[%d]", index, output_size);
        return INTERNAL_ERROR;
      }

      output_list[index] = data_output_offset;
      src_node->GetOpDesc()->SetOutputOffset(output_list);
      input_list.emplace_back(data_output_offset);
    }

    n->GetOpDesc()->SetInputOffset(input_list);
  }
  GELOGI("After reassign merge memory, memoffset = %zu.", memory_offset_[0].mem_offset_);
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignAtomicMemory(bool is_loop_graph) {
  if (compute_graph_ == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Graph must not be null.");
    return ge::PARAM_INVALID;
  }
  // Atomic op memory start addr
  int64_t atomic_mem_start = static_cast<int64_t>(memory_offset_[0].mem_offset_);
  GELOGI("Begin to reAssign atomic memory, atomic initial address mem_offset = %zu!", memory_offset_[0].mem_offset_);

  for (auto &node : compute_graph_->GetDirectNode()) {
    auto node_op_desc = node->GetOpDesc();
    if (node_op_desc == nullptr) {
      continue;
    }

    bool is_atomic = false;
    // If GetBool fail, is_atomic is false.
    (void)ge::AttrUtils::GetBool(node_op_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic);
    if (!is_atomic) {
      continue;
    }

    bool is_ref = false;
    // If GetBool fail, is_ref is false.
    (void)ge::AttrUtils::GetBool(node_op_desc, ATTR_NAME_REFERENCE, is_ref);
    if (is_ref) {
      GELOGE(ge::PARAM_INVALID, "The node %s cannot have both atomic and ref attribute.",
             node_op_desc->GetName().c_str());
      return ge::PARAM_INVALID;
    }

    // Atomic op memory start addr of loop graph
    int64_t loop_graph_atomic_mem_start = static_cast<int64_t>(memory_offset_[0].mem_offset_);

    // Reassign atomic node output memory
    Status ret = AssignAtomicOutputMemory(node);
    if (ret != SUCCESS) {
      GELOGE(ret, "Assign atomic output memory failed, node is %s.", node_op_desc->GetName().c_str());
      return ret;
    }

    // Check atomic workspace
    map<string, map<int64_t, int64_t>> sub_node_workspace_info;
    sub_node_workspace_info = node_op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, sub_node_workspace_info);
    if (!sub_node_workspace_info.empty()) {
      bool is_fusion_node = false;
      // If GetBool fail, is_fusion_node is false.
      (void)ge::AttrUtils::GetBool(node_op_desc, ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);

      if (is_fusion_node) {
        // Assign fusion atomic node workspace memory
        ret = AssignFusionAtomicWorkspaceMemory(node_op_desc, sub_node_workspace_info);
      } else {
        // Assign single ordinary atomic node workspace memory, not include fusion node
        ret = AssignOrdinaryAtomicWorkspaceMemory(node_op_desc, sub_node_workspace_info);
      }

      if (ret != SUCCESS) {
        GELOGE(ret, "Assign atomic workspace memory failed, node is %s.", node_op_desc->GetName().c_str());
        return ret;
      }
    }

    /// In networks with loop op, atomic op uses atomic_addr_clean op independently,
    /// so we need to set the attr separately.
    if (is_loop_graph) {
      GE_CHK_STATUS_RET(SetLoopGraphAtomicAttr(node, loop_graph_atomic_mem_start));
    }
  }

  // In networks without loop op, the same atomic addr clean op is used for atomic op
  if (!is_loop_graph) {
    // Set the address attr of atomic clean operator
    int64_t atomic_mem_size = memory_offset_[0].mem_offset_ - atomic_mem_start;
    if (atomic_mem_size != 0) {
      GE_CHK_STATUS_RET(SetAtomicCleanAttr(nullptr, atomic_mem_start, atomic_mem_size), "SetAtomicCleanAttr failed.");
    }
  }

  return SUCCESS;
}

Status GraphMemoryAssigner::AssignReferenceMemory(const ge::NodePtr &node) {
  auto out_op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(out_op_desc == nullptr, GELOGE(ge::FAILED, "out_op_desc is null."); return ge::FAILED);
  vector<int64_t> output_list = out_op_desc->GetOutputOffset();

  if (out_op_desc->GetOutputsSize() > output_list.size()) {
    GELOGE(ge::FAILED, "The size %zu of node output desc is more than output_list's size %zu.",
           out_op_desc->GetOutputsSize(), output_list.size());
    return ge::FAILED;
  }

  map<string, int> input_name_index;
  for (const auto &input_name : out_op_desc->GetAllInputNames()) {
    int index = out_op_desc->GetInputIndexByName(input_name);
    input_name_index.emplace(input_name, index);
  }

  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    string out_data_anchor_name = out_op_desc->GetOutputNameByIndex(out_data_anchor->GetIdx());
    auto iter = input_name_index.find(out_data_anchor_name);
    if (iter != input_name_index.end()) {
      int index = iter->second;
      GE_CHECK_NOTNULL(node->GetInDataAnchor(index));
      auto peer_out_anchor = node->GetInDataAnchor(index)->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
      int peer_out_anchor_index = peer_out_anchor->GetIdx();
      auto peer_out_node = peer_out_anchor->GetOwnerNode();
      auto peer_out_op_desc = peer_out_node->GetOpDesc();
      GE_CHECK_NOTNULL(peer_out_op_desc);
      output_list[out_data_anchor->GetIdx()] = peer_out_op_desc->GetOutputOffset()[peer_out_anchor_index];
    } else {
      GELOGD("Reference output : origin %s name[%s] output[%d] offset is [%ld] stream_id[%ld]",
             node->GetOwnerComputeGraph()->GetName().c_str(), out_op_desc->GetName().c_str(), out_data_anchor->GetIdx(),
             output_list[out_data_anchor->GetIdx()], out_op_desc->GetStreamId());
    }
  }

  out_op_desc->SetOutputOffset(output_list);

  return ge::SUCCESS;
}

bool GraphMemoryAssigner::CheckInputIsSupportAtomic(const ge::NodePtr &node) {
  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_data_anchor == nullptr) {
      continue;
    }
    auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
    if (peer_op_desc == nullptr) {
      continue;
    }
    if ((peer_op_desc->GetType() == CONSTANTOP) || (peer_op_desc->GetType() == AIPP_DATA_TYPE) ||
        (peer_op_desc->GetType() == VARIABLE)) {
      GELOGE(ge::FAILED,
             "The current node is %s, and the peer out node is %s. Currently, this scenario is not supported",
             node->GetName().c_str(), peer_op_desc->GetName().c_str());
      return false;
    }
  }
  return true;
}

Status GraphMemoryAssigner::AssignAtomicOutputMemory(const ge::NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(ge::FAILED, "op_desc is null."); return ge::FAILED);
  GELOGD("Begin to assign atomic output memory, node = %s.", op_desc->GetName().c_str());

  vector<int64_t> atomic_output_index;
  // If GetListInt fail, atomic_output_index is empty.
  (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  // Check atomic output
  vector<int64_t> output_list = op_desc->GetOutputOffset();
  if (atomic_output_index.size() > output_list.size()) {
    GELOGE(ge::FAILED, "The size of atomic_output_index is more than the size of output_list");
    return ge::FAILED;
  }
  auto output_list_size = static_cast<int64_t>(output_list.size());
  for (auto &output_index : atomic_output_index) {
    if (output_index >= output_list_size) {
      GELOGE(ge::PARAM_INVALID, "The output index %ld is more than the size %ld of output_list.", output_index,
             output_list_size);
      return ge::PARAM_INVALID;
    }

    // If the input of the cascade op needs to clear the atomic addr, there is no need to clear it separately here
    bool is_assigned_mem = false;
    if (static_cast<size_t>(output_index) >= node->GetAllOutDataAnchors().size()) {
      GELOGE(ge::PARAM_INVALID, "Output index %ld is more than the size of node's AllOutDataAnchors.", output_index);
      return ge::PARAM_INVALID;
    }
    auto out_data_anchor = node->GetAllOutDataAnchors().at(output_index);
    GE_CHECK_NOTNULL(out_data_anchor);
    auto input_anchors = out_data_anchor->GetPeerInDataAnchors();
    for (auto &input_anchor : input_anchors) {
      auto output_node = input_anchor->GetOwnerNode();

      /// Get input atomic attr of peer output op, if atomic_input_index[0] = -1, indicates that the atomic address
      /// has been assigned
      vector<int64_t> atomic_input_index;
      (void)ge::AttrUtils::GetListInt(output_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, atomic_input_index);
      if (!atomic_input_index.empty() && (atomic_input_index[0] == kAllInputAddrIsAtomic)) {
        is_assigned_mem = true;
        break;
      }
    }

    // If you have already assigned an atomic address, skip it, and you don't need to reassign it.
    if (is_assigned_mem) {
      GELOGD(
        "[IMAS]Atomic output : we have assigned atomic memory as the input of next node in "
        "ReAssignContinuousMemory function.");
      continue;
    }

    auto output_desc = op_desc->GetAllOutputsDescPtr().at(output_index);
    uint32_t size = 0;
    if (ge::TensorUtils::GetSize(*output_desc, size) != SUCCESS) {
      GELOGI("Get size failed");
    }

    output_list[output_index] = memory_offset_[0].mem_offset_;

    memory_offset_[0].mem_offset_ += size;
    AlignMemOffset(kMemAlignSize);
  }

  op_desc->SetOutputOffset(output_list);

  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignOrdinaryAtomicWorkspaceMemory(const ge::OpDescPtr &op_desc,
                                                                map<string, map<int64_t, int64_t>> &workspace_info) {
  GELOGI("Begin to reassign normal atomic memory, node = %s.", op_desc->GetName().c_str());
  vector<int64_t> workspace_vector = op_desc->GetWorkspace();

  for (auto iter = workspace_info.begin(); iter != workspace_info.end(); ++iter) {
    if (op_desc->GetName() != iter->first) {
      GELOGE(ge::PARAM_INVALID, "The node name %s and the node name %s in workspace info are inconsistent.",
             op_desc->GetName().c_str(), iter->first.c_str());
      return ge::PARAM_INVALID;
    }

    if (iter->second.empty()) {
      continue;
    }

    for (auto &info_iter : iter->second) {
      auto workspace_index = static_cast<uint64_t>(info_iter.first);
      auto workspace_size = info_iter.second;
      if (workspace_index >= workspace_vector.size()) {
        GELOGE(ge::PARAM_INVALID, "The workspace index %lu is more than the size %zu of workspace vector.",
               workspace_index, workspace_vector.size());
        return ge::PARAM_INVALID;
      }

      workspace_vector[workspace_index] = memory_offset_[0].mem_offset_;

      memory_offset_[0].mem_offset_ += workspace_size;
    }
  }
  op_desc->SetWorkspace(workspace_vector);

  return SUCCESS;
}

Status GraphMemoryAssigner::AssignFusionAtomicWorkspaceMemory(const ge::OpDescPtr &op_desc,
                                                              map<string, map<int64_t, int64_t>> &workspace_info) {
  GELOGI("Begin to reassign fusion atomic memory, node = %s.", op_desc->GetName().c_str());
  map<string, map<int64_t, int64_t>> sub_node_workspace_offset;

  for (auto &iter : workspace_info) {
    if (iter.second.empty()) {
      continue;
    }

    map<int64_t, int64_t> index_offset;
    for (auto &info_iter : iter.second) {
      auto workspace_index = static_cast<uint64_t>(info_iter.first);
      auto workspace_size = info_iter.second;

      size_t workspace_offset = memory_offset_[0].mem_offset_;

      memory_offset_[0].mem_offset_ += workspace_size;
      index_offset.insert(std::make_pair(workspace_index, workspace_offset));
    }
    sub_node_workspace_offset.insert(std::make_pair(iter.first, index_offset));
  }
  if (!(op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_OFFSET, sub_node_workspace_offset))) {
    GELOGE(FAILED, "Set EXT_ATTR_ATOMIC_WORKSPACE_OFFSET failed, op name:%s.", op_desc->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status GraphMemoryAssigner::CheckOffset() {
  for (const ge::NodePtr &node : compute_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    vector<int64_t> input_list = node->GetOpDesc()->GetInputOffset();
    for (auto input : input_list) {
      if (input == ge::kInvalidOffset) {
        GELOGE(FAILED, "Invalid offset in node: %s input: %ld.", node->GetName().c_str(), ge::kInvalidOffset);
        return FAILED;
      }
    }
    vector<int64_t> output_list = node->GetOpDesc()->GetOutputOffset();
    for (auto output : output_list) {
      if (output == ge::kInvalidOffset) {
        GELOGE(FAILED, "Invalid offset in node: %s output: %ld.", node->GetName().c_str(), ge::kInvalidOffset);
        return FAILED;
      }
    }
    vector<int64_t> workspace_list = node->GetOpDesc()->GetWorkspace();
    for (auto workspace : workspace_list) {
      if (workspace == ge::kInvalidOffset) {
        GELOGE(FAILED, "Invalid offset in node: %s workspace: %ld.", node->GetName().c_str(), ge::kInvalidOffset);
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::SetInputOffset() {
  if (memory_offset_.empty()) {
    GELOGE(FAILED, "memory_offset_ is empty.");
    return FAILED;
  }
  GEEVENT("[IMAS]AfterAssignMemory : %s", compute_graph_->GetName().c_str());
  for (const ge::NodePtr &node : compute_graph_->GetDirectNode()) {
    if (UpdateOpInputOffset(node) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "Update op input offset failed");
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::UpdateOpInputOffset(const NodePtr &node) const {
  vector<int64_t> input_list;
  if (node->GetType() == HCOMBROADCAST) {
    for (const auto &anchor : node->GetAllInDataAnchors()) {
      vector<int64_t> output_list;
      auto peer_out_anchor = anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }

      auto last_peer_out_node = peer_out_anchor->GetOwnerNode();
      // If the current node is broadcast and the preceding node is variable, because InputOffset has been set
      // in function:AssignVarAttr2Nodes, then the InputOffset of the broadcast node is taken to update the input_list.
      // Otherwise, the OutputOffset of the previous node is used to update the input_list.
      if (last_peer_out_node->GetType() != VARIABLE) {
        auto last_peer_out_op_desc = last_peer_out_node->GetOpDesc();
        GE_CHECK_NOTNULL(last_peer_out_op_desc);
        output_list = last_peer_out_op_desc->GetOutputOffset();
        if (output_list.size() > static_cast<size_t>(peer_out_anchor->GetIdx())) {
          input_list.emplace_back(output_list.at(peer_out_anchor->GetIdx()));
        }
      } else {
        vector<int64_t> cur_node_input_list;
        auto cur_node_op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(cur_node_op_desc);
        cur_node_input_list = cur_node_op_desc->GetInputOffset();
        if (cur_node_input_list.size() > static_cast<size_t>(anchor->GetIdx())) {
          input_list.emplace_back(cur_node_input_list.at(anchor->GetIdx()));
        }
      }
    }
  } else {
    for (const auto &anchor : node->GetAllInDataAnchors()) {
      vector<int64_t> output_list;
      auto peer_out_anchor = anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }

      // If the current node not broadcast, the OutputOffset of the previous node is used to update the input_list
      auto last_peer_out_node = peer_out_anchor->GetOwnerNode();
      auto last_peer_out_op_desc = last_peer_out_node->GetOpDesc();
      GE_CHECK_NOTNULL(last_peer_out_op_desc);
      output_list = last_peer_out_op_desc->GetOutputOffset();
      if (output_list.size() > static_cast<size_t>(peer_out_anchor->GetIdx())) {
        input_list.emplace_back(output_list.at(peer_out_anchor->GetIdx()));
      }
    }
  }
  GE_CHECK_NOTNULL(node->GetOpDesc());
  node->GetOpDesc()->SetInputOffset(input_list);
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::SetLoopGraphAtomicAttr(const ge::NodePtr &node, int64_t atomic_mem_start) {
  // set the address attr of atomic clean operator for loop graph
  int64_t atomic_mem_size = memory_offset_[0].mem_offset_ - atomic_mem_start;
  GELOGI("SetLoopGraphAtomicAttr beign, atomic_addr_clean start size is %ld, mem_size is %ld, mem_offset is %zu.",
         atomic_mem_start, atomic_mem_size, memory_offset_[0].mem_offset_);
  const auto &in_control_anchor = node->GetInControlAnchor();
  if (atomic_mem_size != 0 && in_control_anchor != nullptr) {
    for (auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
      if (peer_out_control_anchor == nullptr) {
        continue;
      }
      auto peer_out_node = peer_out_control_anchor->GetOwnerNode();
      auto peer_out_node_desc = peer_out_node->GetOpDesc();
      if (peer_out_node_desc == nullptr) {
        continue;
      }

      GELOGD("SetLoopGraphAtomicAttr,  node is %s, op type is %s.", peer_out_node_desc->GetName().c_str(),
             peer_out_node_desc->GetType().c_str());

      if (peer_out_node_desc->GetType() == ATOMICADDRCLEAN) {
        GE_CHK_STATUS_EXEC(SetAtomicCleanAttr(peer_out_node, atomic_mem_start, atomic_mem_size),
                           GELOGE(FAILED, "SetAtomicCleanAttr failed.");
                           return FAILED);
      }
    }
  }
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::SetAtomicCleanAttr(const NodePtr &n, int64_t atomic_mem_start,
                                                   int64_t atomic_mem_size) {
  for (ge::NodePtr &node : compute_graph_->GetDirectNode()) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);

    if (((n != nullptr) && (node->GetName() == n->GetName())) ||
        ((n == nullptr) && (node_op_desc->GetType() == ATOMICADDRCLEAN))) {
      vector<int64_t> workspace_vector = node_op_desc->GetWorkspace();
      vector<int64_t> workspace_byte_vector = node_op_desc->GetWorkspaceBytes();
      workspace_vector.emplace_back(atomic_mem_start);
      workspace_byte_vector.emplace_back(atomic_mem_size);
      node_op_desc->SetWorkspace(workspace_vector);
      node_op_desc->SetWorkspaceBytes(workspace_byte_vector);

      std::vector<int64_t> mem_start_vector;
      // If GetListInt fail, mem_start_vector is empty.
      (void)ge::AttrUtils::GetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_START, mem_start_vector);
      mem_start_vector.emplace_back(atomic_mem_start);
      GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_START, mem_start_vector),
                       GELOGE(FAILED, "SetListInt failed.");
                       return FAILED);

      std::vector<int64_t> mem_size_vector;
      // If GetListInt fail, mem_size_vector is empty.
      (void)ge::AttrUtils::GetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_MEM_SIZE, mem_size_vector);
      mem_size_vector.emplace_back(atomic_mem_size);
      GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_MEM_SIZE, mem_size_vector),
                       GELOGE(FAILED, "SetListInt failed.");
                       return FAILED);
    }
  }
  return SUCCESS;
}

void GraphMemoryAssigner::AlignMemOffset(const int64_t &mem_align_size) {
  if (mem_align_size <= 0) {
    return;
  }
  memory_offset_[0].mem_offset_ =
    (memory_offset_[0].mem_offset_ + mem_align_size - 1) / mem_align_size * mem_align_size;
}
}  // namespace ge
