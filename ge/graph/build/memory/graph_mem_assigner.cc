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

#include "graph/build/memory/graph_mem_assigner.h"
#include <cstring>
#include <set>
#include "common/math/math_util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace {
const int kAllInputAddrIsAtomic = -1;
const int kVirtualInputNodeMemoryReuse = 0;
const int kVirtualOutputNodeMemoryReuse = 1;
const size_t kVirtualInputNodeOutputSize = 1;
const size_t kVirtualOutputNodeInputSize = 1;
const size_t kVirtualNodeDataIndex = 0;
const char *const kMbatchNodeNameFlag = "_ascend_mbatch_batch_";
int64_t GetSymbolOutputOffset(const std::map<std::string, std::string> &anchor_to_symbol,
                              const std::map<std::string, std::list<ge::NodeIndexIO>> &symbol_to_anchors,
                              const ge::NodePtr &node, const uint32_t i) {
  ge::NodeIndexIO cur_node_index_io(node, i, ge::kOut);
  auto iter1 = anchor_to_symbol.find(cur_node_index_io.ToString());
  if (iter1 == anchor_to_symbol.end()) {
    return ge::kInvalidOffset;
  }
  auto out_symbol = iter1->second;
  auto iter2 = symbol_to_anchors.find(out_symbol);
  if (iter2 == symbol_to_anchors.end()) {
    return ge::kInvalidOffset;
  }
  for (const auto &node_index_io : iter2->second) {
    if (node_index_io.value_ == out_symbol) {
      vector<int64_t> output_list = node->GetOpDesc()->GetOutputOffset();
      vector<int64_t> symbol_output_list = node_index_io.node_->GetOpDesc()->GetOutputOffset();
      if (node_index_io.index_ >= symbol_output_list.size()) {
        return ge::kInvalidOffset;
      }
      GELOGD("Node %s %uth output offset is %ld, Symbol %s output offset is %ld.", node->GetName().c_str(), i,
             output_list[i], iter2->first.c_str(), symbol_output_list.at(node_index_io.index_));
      return symbol_output_list.at(node_index_io.index_);
    }
  }
  return ge::kInvalidOffset;
}
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
  ge::HybridMemAssignerPtr mem_assigner(new(std::nothrow) HybridMemAssigner(compute_graph_));
  if (mem_assigner->Assign() != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Memory assigner failed");
    return ge::FAILED;
  }
  MemoryOffset memory_offset(RT_MEMORY_HBM, mem_assigner->GetMemOffset());
  memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);

  if (mem_assigner->GetP2PMemOffset() >= 0) {
    MemoryOffset p2p_memory_offset(RT_MEMORY_P2P_DDR, mem_assigner->GetP2PMemOffset());
    memory_offset_.emplace(RT_MEMORY_P2P_DDR, p2p_memory_offset);
  }

  auto session_id = compute_graph_->GetSessionID();
  int64_t var_size_before_assign = ge::VarManager::Instance(session_id)->GetVarMemSize(RT_MEMORY_HBM);
  auto variable_assigner =
      std::unique_ptr<ge::VariableMemoryAssigner>(new(std::nothrow) ge::VariableMemoryAssigner(compute_graph_));
  if (variable_assigner == nullptr) {
    GELOGE(ge::FAILED, "Alloc VariableMemoryAssigner failed.");
    return ge::FAILED;
  }

  if (variable_assigner->Assign() != ge::SUCCESS) {
    return ge::FAILED;
  }
  int64_t var_size_assign = ge::VarManager::Instance(session_id)->GetVarMemSize(RT_MEMORY_HBM) - var_size_before_assign;
  GELOGD("GraphMemoryAssigner::AssignMemory variable size = %ld", var_size_assign);

  mem_assigner_ = std::move(mem_assigner);

  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::AssignVarAttr2Nodes() {
  auto variable_assigner =
      std::unique_ptr<ge::VariableMemoryAssigner>(new(std::nothrow) ge::VariableMemoryAssigner(compute_graph_));
  if (variable_assigner == nullptr) {
    GELOGE(ge::FAILED, "Alloc VariableMemoryAssigner failed.");
    return ge::FAILED;
  }
  if (variable_assigner->AssignVarAttr2Nodes() != ge::SUCCESS) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::CalculateTensorRealSizeAndOutSize(const ge::ConstGeTensorDescPtr &output_desc,
                                                                  int64_t dim_index, int64_t &output_mem_size,
                                                                  int64_t &batch_dim_num, int64_t &out_size) {
  graphStatus graph_status = ge::TensorUtils::GetSize(*output_desc, out_size);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Opdesc GetSize failed!");
    return FAILED;
  }

  GeShape output_shape = output_desc->GetShape();
  std::vector<int64_t> output_dims = output_shape.GetDims();
  if (dim_index >= static_cast<int64_t>(output_dims.size())) {
    std::string error = "Invaild value" + FmtToStr(dim_index) +
        " of attr _reuse_input_on_dim_index, which is out of data range [0,"
        + std::to_string(output_dims.size()) + ")";
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }

  for (int64_t index = 0; index < dim_index; index++) {
    FMK_INT64_MULCHECK(batch_dim_num, output_dims[index]);
    batch_dim_num *= output_dims[index];
    output_dims[index] = 1;
  }

  output_shape = GeShape(output_dims);
  Format out_format = output_desc->GetFormat();
  DataType data_type = output_desc->GetDataType();

  graph_status = ge::TensorUtils::CalcTensorMemSize(output_shape, out_format, data_type, output_mem_size);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(graph_status, "Opdesc CalcTensorMemSize failed!");
    return FAILED;
  }

  if (output_mem_size < 0) {
    std::string error = "After calculating tensor memory size, output_mem_size" + FmtToStr(output_mem_size) +
        " is out of data range [0," + std::to_string(INT64_MAX) + "]";
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status GraphMemoryAssigner::GetMaxBatchLabel(const map<string, vector<NodePtr>> &mem_reuse_virtual_nodes_map,
                                             int32_t mem_reuse_model, string &max_batch_label) {
  for (auto &i_map : mem_reuse_virtual_nodes_map) {
    vector<NodePtr> virtual_nodes_list = i_map.second;
    vector<int64_t> max_shape_dims;
    size_t max_batch_dim = 0;
    bool max_batch_dim_find = false;
    for (size_t i = 0; i < virtual_nodes_list.size(); ++i) {
      GE_CHECK_NOTNULL(virtual_nodes_list[i]);
      OpDescPtr op_desc = virtual_nodes_list[i]->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);

      ge::ConstGeTensorDescPtr input_output_desc;
      if (mem_reuse_model == kVirtualInputNodeMemoryReuse) {
        input_output_desc = op_desc->GetOutputDescPtr(kVirtualNodeDataIndex);
      } else if (mem_reuse_model == kVirtualOutputNodeMemoryReuse) {
        input_output_desc = op_desc->GetInputDescPtr(kVirtualNodeDataIndex);
      } else {
        std::string error = "Invalid parameter memory reuse model, which is " + FmtToStr(mem_reuse_model);
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
      GE_CHECK_NOTNULL(input_output_desc);

      if (i == 0) {
        // All ops must have ATTR_NAME_BATCH_LABEL, no need to check return value.
        (void) ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, max_batch_label);
        max_shape_dims = input_output_desc->GetShape().GetDims();
      } else {
        vector<int64_t> current_shape_dims = input_output_desc->GetShape().GetDims();
        if (current_shape_dims.size() != max_shape_dims.size()) {
          std::string error = "The shape of several nodes between multiple batches does not match.";
          GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
          return FAILED;
        }
        for (size_t j = 0; j < current_shape_dims.size(); ++j) {
          if (current_shape_dims[j] == max_shape_dims[j]) {
            continue;
          }
          if (max_batch_dim_find && max_batch_dim != j) {
            std::string error = "The shape of several nodes between multiple batches does not match.";
            GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
            return FAILED;
          }
          max_batch_dim_find = true;
          max_batch_dim = j;
          if (current_shape_dims[j] > max_shape_dims[j]) {
            max_shape_dims[j] = current_shape_dims[j];
            // All ops must have ATTR_NAME_BATCH_LABEL, no need to check return value.
            (void) ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, max_batch_label);
          }
          // Only compare the first different dim in shape.
          break;
        }
      }
    }
    // In every element of virtual_input_nodes_map, the label of the max batch node is the same.
    break;
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignMemory(bool is_loop_graph, map<int64_t, size_t> &mem_type_to_offset) {
  if (memory_offset_.empty()) {
    GELOGE(FAILED, "memory_offset_ is empty.");
    return ge::FAILED;
  }

  GE_CHK_STATUS_RET(ReAssignContinuousMemory(is_loop_graph), "ReAssignContinuousMemory Failed!");

  GE_CHK_STATUS_RET(ReAssignReuseAndNoPaddingContinuousInputMemory(),
                    "ReAssignReuseAndNoPaddingContinuousInputMemory Failed!");

  GE_CHK_STATUS_RET(ReAssignReuseAndNoPaddingContinuousOutputMemory(),
                    "ReAssignReuseAndNoPaddingContinuousOutputMemory Failed!");

  GE_CHK_STATUS_RET(ReAssignAtomicMemory(is_loop_graph), "ReAssignAtomicMemory Failed!");

  size_t total_mem_offset = 0;
  for (auto pair : memory_offset_) {
    mem_type_to_offset[pair.first] = pair.second.mem_offset_;
    total_mem_offset += pair.second.mem_offset_;
  }

  auto session_id = compute_graph_->GetSessionID();
  if (total_mem_offset > VarManager::Instance(session_id)->GetGraphMemoryMaxSize()) {
    GELOGE(ge::FAILED, "Current memoffset %zu is greater than memory manager malloc max size %zu", total_mem_offset,
           VarManager::Instance(session_id)->GetGraphMemoryMaxSize());
    for (auto iter : mem_type_to_offset) {
      ErrorManager::GetInstance().ATCReportErrMessage("E19022", {"memType", "size", "item", "maxsize"},
        {std::to_string(iter.first), std::to_string(iter.second), "featuremap",
         std::to_string(VarManager::Instance(session_id)->GetGraphMemoryMaxSize())});
    }
    return ge::FAILED;
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::AssignZeroCopyMemory(map<int64_t, size_t> &mem_offset, size_t &zero_mem_copy_size) {
  BlockMemAssignerPtr priority_assigner = std::move(mem_assigner_->GetPriorityAssinger());
  GE_IF_BOOL_EXEC(priority_assigner == nullptr, GELOGE(FAILED, "Get priority_assigner failed."); return ge::FAILED;);

  size_t mem_offset_tmp = mem_offset[RT_MEMORY_HBM];

  // set offset for zero copy block
  for (auto &memory_block : priority_assigner->GetMemoryBlocks()) {
    if (memory_block == nullptr || memory_block->deleted_block_ || !memory_block->is_zero_copy_) {
      continue;
    }
    memory_block->Resize();
    memory_block->SetHeadOffset(mem_offset[RT_MEMORY_HBM]);
    mem_offset[RT_MEMORY_HBM] += memory_block->Size();
    memory_block->SetTailOffset(mem_offset[RT_MEMORY_HBM] - 1);
  }

  // set offset for zero copy nodes
  priority_assigner->SetOpMemOffset(true);
  zero_mem_copy_size = mem_offset[RT_MEMORY_HBM] - mem_offset_tmp;
  auto iter = memory_offset_.find(RT_MEMORY_HBM);
  if (iter == memory_offset_.end()) {
    std::string error = "Memory offset does not have memory type[HBM]";
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }
  iter->second.mem_offset_ = mem_offset[RT_MEMORY_HBM];

  GELOGD("max_mem_offset:%zu, mem_offset:%zu, zero_mem_copy_size:%zu.", mem_offset[RT_MEMORY_HBM], mem_offset_tmp,
         zero_mem_copy_size);

  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignContinuousMemory(bool is_loop_graph) {
  Status ret;
  for (auto &node : compute_graph_->GetAllNodes()) {
    // Get the continuous input type of the node, default is false
    bool is_input_continuous = false;
    GE_CHECK_NOTNULL(node->GetOpDesc());
    // If GetBool fail, is_input_continuous is false.
    (void) ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);

    // Assign continuous input memory
    if (is_input_continuous) {
      int64_t memory_type = RT_MEMORY_HBM;
      GE_CHK_STATUS_RET(GetNodeMemoryType(node, memory_type, "input"), "Get node memory type failed.");
      int64_t mem_clean_start = 0;
      int64_t mem_clean_size = 0;
      ret = AssignContinuousInputMemory(node, mem_clean_start, mem_clean_size, memory_type);
      if (ret != ge::SUCCESS) {
        GELOGE(ret, "Assign continuous input memory failed!");
        return ret;
      }

      // Clean up atomic address, eg, hcom node
      vector<int32_t> input_indexes;
      // If GetListInt fail, input_indexes is empty.
      (void) ge::AttrUtils::GetListInt(node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, input_indexes);

      if (!input_indexes.empty() && input_indexes[0] == kAllInputAddrIsAtomic) {
        // check whether there is an atomic conflict between the current node and the peer out node
        if (!CheckInputIsSupportAtomic(node)) {
          GELOGE(ge::FAILED,
                 "There is an atomic conflict between the current node and the peer out node, not supported!");
          return ge::FAILED;
        }

        const auto &in_control_anchor = node->GetInControlAnchor();
        GE_CHECK_NOTNULL(in_control_anchor);
        for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
          auto peer_out_node = peer_out_control_anchor->GetOwnerNode();
          if (peer_out_node->GetType() == ATOMICADDRCLEAN) {
            ret = SetAtomicCleanAttr(peer_out_node, {mem_clean_start}, {mem_clean_size});
            if (ret != SUCCESS) {
              GELOGE(ret, "Failed to set attr for atomic addr clean node %s.", peer_out_node->GetName().c_str());
              return ret;
            }
          }
        }
      }
    }

    // Get the reference type of the node, default is false
    bool is_ref = false;
    // If GetBool fail, is_ref is false.
    (void) ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_REFERENCE, is_ref);

    // Get the continuous output type of the node, default is false
    bool is_output_continuous = false;
    // If GetBool fail, is_output_continuous is false.
    (void) ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, is_output_continuous);

    // If the output is ref type and refers to the ref of an input, the name of the output
    // and the input are the same. Ge encounters ref type, finds matching relationship according
    // to the names of input and output, and allocates the same memory address, eg: HCOMBroadcast
    if (!is_ref && is_output_continuous) {  // Assign continuous output memory
      ret = AssignContinuousOutputMemory(node);
      if (ret != ge::SUCCESS) {
        GELOGE(ret, "Assign reference memory failed!");
        return ret;
      }
    }
  }
  for (auto pair : memory_offset_) {
    GELOGD("After reassign continuous memory, memory type = %ld, memoffset = %zu.", pair.first,
           pair.second.mem_offset_);
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignContinuousInputMemory(const ge::NodePtr &node, int64_t &continuous_mem_start,
                                                        int64_t &continuous_mem_size, int64_t memory_type) {
  GELOGI("Current node %s needs continuous input.", node->GetName().c_str());
  bool continuous_input_alloc = false;
  (void) ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT_ALLOC, continuous_input_alloc);
  auto iter = memory_offset_.find(memory_type);
  if (iter == memory_offset_.end()) {
    std::string error = "Memory offset does not have memory type" + FmtToStr(memory_type);
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }
  continuous_mem_start = iter->second.mem_offset_;
  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_data_anchor == nullptr, continue);

    auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
    GE_IF_BOOL_EXEC(peer_op_desc == nullptr, continue);
    bool is_peer_output_continuous = false;
    // If GetBool fail, is_peer_output_continuous is false.
    (void) ge::AttrUtils::GetBool(peer_op_desc, ATTR_NAME_CONTINUOUS_OUTPUT, is_peer_output_continuous);

    // Get peer node output size, if size == 1(peer node has only one output), continuous input of the node and
    // continuous output of the previous node is the same, we can support it. If size != 1, there may be
    // conflict between the two, we can not support it.
    auto peer_output_size = peer_op_desc->GetOutputsSize();
    GE_IF_BOOL_EXEC(is_peer_output_continuous && (peer_output_size != 1),
                    std::string error = "Current op" + FmtToStr(node->GetOpDesc()->GetName()) +
                        " requires continuous input, while the previous op" + FmtToStr(peer_op_desc->GetName()) +
                        " requires continuous output. There may be conflict between the two." +
                        "This node is not supported now.";
                    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
                    return PARAM_INVALID;);

    bool is_peer_reference = false;
    // If GetBool fail, is_peer_reference is false.
    (void) AttrUtils::GetBool(peer_op_desc, ATTR_NAME_REFERENCE, is_peer_reference);
    GE_IF_BOOL_EXEC(is_peer_reference,
                    std::string error = "Current op" + FmtToStr(node->GetOpDesc()->GetName()) +
                        " requires continuous input, while the previous op" + FmtToStr(peer_op_desc->GetName()) +
                        " requires continuous output. There may be conflict between the two." +
                        "This node is not supported now.";
                    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
                    return PARAM_INVALID;);

    vector<int64_t> output_list = peer_op_desc->GetOutputOffset();
    std::vector<int64_t> offsets_for_fusion = {};
    bool has_offset_attr =
        AttrUtils::GetListInt(peer_op_desc, ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);
    if (peer_out_data_anchor->GetIdx() < static_cast<int>(output_list.size())) {
      if (continuous_input_alloc && !has_offset_attr) {
        if (in_data_anchor->GetIdx() == 0) {
          continuous_mem_start = output_list.at(peer_out_data_anchor->GetIdx());
        }
        // can not use else if, incase only one input
        if (in_data_anchor->GetIdx() == static_cast<int>(node->GetAllInDataAnchors().size()) - 1) {
          int64_t tensor_desc_size = 0;
          Status ret = ge::TensorUtils::GetSize(*(peer_op_desc->GetOutputDescPtr(peer_out_data_anchor->GetIdx())),
                                                tensor_desc_size);
          GE_IF_BOOL_EXEC(ret != ge::SUCCESS, GELOGE(FAILED, "GetSize failed."); return FAILED;);

          tensor_desc_size = (tensor_desc_size + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
          continuous_mem_size =
              output_list.at(peer_out_data_anchor->GetIdx()) - continuous_mem_start + tensor_desc_size + MEM_ALIGN_SIZE;
        }
        GELOGI(
            "[IMAS]Check Continuous input : Set %s name[%s] output[%d] offset to [%ld] stream_id[%ld] size[%u] "
            "real_size[%u].",
            node->GetOwnerComputeGraph()->GetName().c_str(), peer_op_desc->GetName().c_str(),
            peer_out_data_anchor->GetIdx(), output_list.at(peer_out_data_anchor->GetIdx()), peer_op_desc->GetStreamId(),
            0, 0);
        continue;
      }

      output_list.at(peer_out_data_anchor->GetIdx()) = iter->second.mem_offset_;
    } else {
      std::string error = "index" + FmtToStr(peer_out_data_anchor->GetIdx()) + " is out of range.";
      GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
      GELOGE(FAILED, "index : %d is out of range.", peer_out_data_anchor->GetIdx());
      return FAILED;
    }
    peer_op_desc->SetOutputOffset(output_list);
    size_t pre_mem_offset = iter->second.mem_offset_;

    int64_t tensor_desc_size = 0;
    if (has_offset_attr) {
      if (peer_out_data_anchor->GetIdx() < static_cast<int>(offsets_for_fusion.size())) {
        auto offset_for_fusion = offsets_for_fusion[peer_out_data_anchor->GetIdx()];
        iter->second.mem_offset_ += offset_for_fusion;
      } else {
        std::string error = "fusion: peer node" + FmtToStr(peer_op_desc->GetName()) +
            " index" + FmtToStr(peer_out_data_anchor->GetIdx()) + " is out of range.";
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
    } else {
      Status ret =
          TensorUtils::GetSize(*(peer_op_desc->GetOutputDescPtr(peer_out_data_anchor->GetIdx())), tensor_desc_size);
      GE_IF_BOOL_EXEC(ret != ge::SUCCESS, GELOGE(FAILED, "GetSize failed."); return FAILED;);

      iter->second.mem_offset_ += tensor_desc_size;
    }

    // If set tensor_actual_size, Memory alignment is not required.
    int32_t is_tensor_actual_size = 0;
    ge::AttrUtils::GetInt(peer_op_desc, ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, is_tensor_actual_size);
    if (is_tensor_actual_size == 0) {
      AlignMemOffset(MEM_ALIGN_SIZE, memory_type);
    }
    GELOGI(
        "[IMAS]Continuous input : Set %s name[%s] output[%d] offset to [%zu] stream_id[%ld] size[%zu] "
        "real_size[%ld].", node->GetOwnerComputeGraph()->GetName().c_str(), peer_op_desc->GetName().c_str(),
        peer_out_data_anchor->GetIdx(), pre_mem_offset, peer_op_desc->GetStreamId(),
        (iter->second.mem_offset_ - pre_mem_offset), tensor_desc_size);
  }

  iter->second.mem_offset_ += MEM_ALIGN_SIZE;
  if (!continuous_input_alloc) {
    continuous_mem_size = iter->second.mem_offset_ - continuous_mem_start;
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::AssignContinuousOutputMemory(const ge::NodePtr &node) {
  GELOGI("Current node %s needs continuous output.", node->GetName().c_str());
  auto out_op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(out_op_desc == nullptr, GELOGE(ge::FAILED, "out_op_desc is null."); return ge::FAILED);
  vector<int64_t> output_list = out_op_desc->GetOutputOffset();

  if ((out_op_desc->GetOutputsSize() > output_list.size()) || (output_list.size() == 0)) {
    GELOGE(ge::FAILED, "The size %zu of node output desc is more than output_list's size %zu.",
           out_op_desc->GetOutputsSize(), output_list.size());
    return ge::FAILED;
  }

  size_t mem_offset = output_list[0];
  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    output_list[out_data_anchor->GetIdx()] = mem_offset;
    int64_t tensor_desc_size = 0;
    if (ge::TensorUtils::GetSize(*(out_op_desc->GetOutputDescPtr(out_data_anchor->GetIdx())), tensor_desc_size) !=
        ge::SUCCESS) {
      GELOGE(FAILED, "GetSize failed.");
      return FAILED;
    }
    mem_offset += tensor_desc_size;
    if (mem_offset <= 0) {
      return FAILED;
    }
    mem_offset = (mem_offset + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
    GELOGI(
        "[IMAS]Continuous output : Set %s name[%s] output[%d] offset to [%zu] stream_id[%ld] size[%ld] "
        "real_size[%ld].",
        node->GetOwnerComputeGraph()->GetName().c_str(), out_op_desc->GetName().c_str(), out_data_anchor->GetIdx(),
        output_list[out_data_anchor->GetIdx()], out_op_desc->GetStreamId(), tensor_desc_size, tensor_desc_size);
  }
  out_op_desc->SetOutputOffset(output_list);
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::ReAssignVirtualInputNodeMemory(NodePtr node, size_t &mem_offset_reuse) {
  OpDescPtr op_desc = node->GetOpDesc();
  vector<int64_t> output_list = op_desc->GetOutputOffset();
  if (output_list.empty()) {
    GELOGE(FAILED, "Outputoffset is empty node name:%s", node->GetName().c_str());
    return FAILED;
  }
  output_list.at(0) = mem_offset_reuse;
  op_desc->SetOutputOffset(output_list);
  GELOGI("Set virtual input node %s output offset to %zu.", op_desc->GetName().c_str(), mem_offset_reuse);

  int64_t attr_dim_index;
  bool get_attr_dim_flag = ge::AttrUtils::GetInt(op_desc, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, attr_dim_index);
  if (!get_attr_dim_flag) {
    GELOGE(FAILED, "Get attr _reuse_input_on_dim_index failed.");
    return FAILED;
  }

  size_t extra_memory_size = 0;
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_data_anchor);
    auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
    GE_CHECK_NOTNULL(peer_op_desc);
    vector<int64_t> output_offsets = peer_op_desc->GetOutputOffset();
    if (peer_out_data_anchor->GetIdx() >= static_cast<int>(output_offsets.size())) {
      GELOGE(ge::FAILED, "Index : %d is out of range.", peer_out_data_anchor->GetIdx());
      return ge::FAILED;
    }
    output_offsets.at(peer_out_data_anchor->GetIdx()) = mem_offset_reuse;
    peer_op_desc->SetOutputOffset(output_offsets);
    size_t pre_mem_offset = mem_offset_reuse;

    // Calculate tensor real size of each piece of data and out size of complete data
    ge::ConstGeTensorDescPtr output_desc = peer_op_desc->GetOutputDescPtr(peer_out_data_anchor->GetIdx());
    GE_CHECK_NOTNULL(output_desc);
    int64_t output_mem_size;
    int64_t batch_dim_num = 1;
    int64_t out_size;
    if (CalculateTensorRealSizeAndOutSize(output_desc, attr_dim_index, output_mem_size, batch_dim_num, out_size) !=
        SUCCESS) {
      GELOGE(FAILED, "CalculateTensorRealSizeAndOutSize failed for node %s output [%d].",
             peer_op_desc->GetName().c_str(), peer_out_data_anchor->GetIdx());
      return FAILED;
    }

    mem_offset_reuse += output_mem_size;
    extra_memory_size = extra_memory_size + out_size - output_mem_size;

    GELOGI("[IMAS]Virtual node optimize: set %s name[%s] output[%d] offset to [%zu] stream_id[%ld] size[%ld] "
           "real_size[%ld].",
           node->GetOwnerComputeGraph()->GetName().c_str(), peer_op_desc->GetName().c_str(),
           peer_out_data_anchor->GetIdx(), pre_mem_offset, peer_op_desc->GetStreamId(), out_size,
           output_mem_size);
  }
  mem_offset_reuse += extra_memory_size;
  size_t after_mem_offset = mem_offset_reuse;
  GELOGI("After reassign virtual input node[name: %s, type: %s] memory, memory offset = %zu.",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), after_mem_offset);
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignReuseAndNoPaddingContinuousInputMemory() {
  map<string, vector<NodePtr>> mem_reuse_virtual_input_nodes_map;
  int64_t memory_type = RT_MEMORY_HBM;
  for (const auto &n : compute_graph_->GetAllNodes()) {
    OpDescPtr op_desc = n->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    bool attr_continuous = false;
    bool get_continuous_flag = ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, attr_continuous);
    GE_IF_BOOL_EXEC(!get_continuous_flag, continue);
    bool attr_reuse = false;
    bool get_reuse_flag = ge::AttrUtils::GetBool(op_desc, ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
    GE_IF_BOOL_EXEC(!get_reuse_flag, continue);
    if (attr_reuse && attr_continuous) {
      if (op_desc->GetOutputsSize() != kVirtualInputNodeOutputSize) {
        // When current virtual node has several outputs, can't directly determine which input is the tensor for reuse.
        std::string error = "Only one output is supported, current virtual node" + FmtToStr(n->GetName()) +
            " has " + FmtToStr(op_desc->GetOutputsSize()) + " outputs.";
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
      GE_CHK_STATUS_RET(GetNodeMemoryType(n, memory_type, "input"), "Get node memory type failed.");
      auto iter = memory_offset_.find(memory_type);
      if (iter == memory_offset_.end()) {
        std::string error = "Memory offset does not have memory type" + FmtToStr(memory_type);
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
      GELOGD("Start to reassign memory for virtual input node, memory offset = %zu, memory type = %ld.",
             iter->second.mem_offset_, memory_type);
      string batch_label_string;
      // Not all ops have ATTR_NAME_BATCH_LABEL, no need to check return value, only check out parameter
      (void) ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label_string);
      if (batch_label_string.empty()) {
        size_t node_mem_offset = iter->second.mem_offset_;
        // No ATTR_NAME_BATCH_LABEL, no need to reuse memory.
        Status status = ReAssignVirtualInputNodeMemory(n, node_mem_offset);
        if (status != SUCCESS) {
          GELOGE(FAILED, "Reassign memory of virtual input node failed, node name: %s.", n->GetName().c_str());
          return FAILED;
        }

        iter->second.mem_offset_ = node_mem_offset;
        AlignMemOffset(MEM_ALIGN_SIZE, memory_type);
        GELOGD("After reassign memory for virtual input node, align memory = %zu, memory type = %ld.",
               iter->second.mem_offset_, memory_type);
      } else {
        // Has ATTR_NAME_BATCH_LABEL, for dynamic multi-batch node, need to reuse memory.
        string current_node_full_name = op_desc->GetName();
        size_t pos = current_node_full_name.find(kMbatchNodeNameFlag);
        if (pos == string::npos) {
          GELOGE(FAILED, "Cannot find key string [%s] of multi-batch in name of virtual input node, node name: %s.",
                 kMbatchNodeNameFlag, n->GetName().c_str());
          return FAILED;
        }
        string fixed_name = current_node_full_name.substr(0, pos);
        vector<NodePtr> parallel_virtual_input_nodes;
        if (mem_reuse_virtual_input_nodes_map.count(fixed_name) != 0) {
          parallel_virtual_input_nodes = mem_reuse_virtual_input_nodes_map[fixed_name];
        }
        parallel_virtual_input_nodes.emplace_back(n);
        mem_reuse_virtual_input_nodes_map[fixed_name] = parallel_virtual_input_nodes;
      }
    }
  }

  int32_t mem_reuse_model = 0;
  if (ReAssignVirtualNodesMemory(mem_reuse_virtual_input_nodes_map, mem_reuse_model) != SUCCESS) {
    GELOGE(FAILED, "Reassign memory of virtual input nodes failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignVirtualOutputNodeMemory(NodePtr node, size_t &mem_offset_reuse) {
  OpDescPtr op_desc = node->GetOpDesc();

  // 1. set memory of to be reused input tensor
  auto in_data_anchor_list = node->GetAllInDataAnchors();
  auto peer_out_data_anchor = in_data_anchor_list.at(0)->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_data_anchor);
  auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHECK_NOTNULL(peer_op_desc);
  vector<int64_t> in_node_output_offsets = peer_op_desc->GetOutputOffset();
  if (peer_out_data_anchor->GetIdx() >= static_cast<int>(in_node_output_offsets.size())) {
    GELOGE(FAILED, "Index : %d is out of range.", peer_out_data_anchor->GetIdx());
    return FAILED;
  }
  in_node_output_offsets.at(peer_out_data_anchor->GetIdx()) = mem_offset_reuse;
  peer_op_desc->SetOutputOffset(in_node_output_offsets);
  GELOGI("Set virtual output node %s input data offset to %zu.", op_desc->GetName().c_str(), mem_offset_reuse);

  // 2. set memory of output tensor
  vector<int64_t> output_list = op_desc->GetOutputOffset();
  if (output_list.empty()) {
    GELOGE(FAILED, "Outputoffset is empty, node name: %s", node->GetName().c_str());
    return FAILED;
  }
  if (op_desc->GetOutputsSize() > output_list.size()) {
    GELOGE(FAILED, "The size %zu of op_desc is more than output_list's size %zu.", op_desc->GetOutputsSize(),
           output_list.size());
    return FAILED;
  }
  int64_t attr_dim_index;
  bool get_attr_dim_flag = ge::AttrUtils::GetInt(op_desc, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, attr_dim_index);
  if (!get_attr_dim_flag) {
    GELOGE(FAILED, "Get attr _reuse_input_on_dim_index failed.");
    return FAILED;
  }

  size_t extra_memory_size = 0;
  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    output_list[out_data_anchor->GetIdx()] = mem_offset_reuse;
    size_t pre_mem_offset = mem_offset_reuse;

    // calculate tensor real size of each piece of data and out size of complete data
    ge::ConstGeTensorDescPtr output_desc = op_desc->GetOutputDescPtr(out_data_anchor->GetIdx());
    GE_CHECK_NOTNULL(output_desc);
    int64_t output_mem_size;
    int64_t batch_dim_num = 1;
    int64_t out_size;
    if (CalculateTensorRealSizeAndOutSize(output_desc, attr_dim_index, output_mem_size, batch_dim_num, out_size) !=
        SUCCESS) {
      GELOGE(FAILED, "CalculateTensorRealSizeAndOutSize failed for node %s output [%d].",
             op_desc->GetName().c_str(), out_data_anchor->GetIdx());
      return FAILED;
    }

    mem_offset_reuse += output_mem_size;
    extra_memory_size = extra_memory_size + out_size - output_mem_size;

    GELOGI("[IMAS]Virtual node optimize: set %s name[%s] output[%d] offset to [%zu], size[%ld], real_size[%ld].",
           node->GetOwnerComputeGraph()->GetName().c_str(), op_desc->GetName().c_str(), out_data_anchor->GetIdx(),
           pre_mem_offset, out_size, output_mem_size);
  }
  op_desc->SetOutputOffset(output_list);
  mem_offset_reuse += extra_memory_size;
  size_t after_mem_offset = mem_offset_reuse;
  GELOGI("After reassign virtual output node[name: %s, type: %s] memory, memory offset = %zu.",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), after_mem_offset);
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignReuseAndNoPaddingContinuousOutputMemory() {
  map<string, vector<NodePtr>> mem_reuse_virtual_output_nodes_map;
  int64_t memory_type = RT_MEMORY_HBM;
  for (const auto &n : compute_graph_->GetAllNodes()) {
    OpDescPtr op_desc = n->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    bool attr_continuous = false;
    bool get_continuous_flag = ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, attr_continuous);
    GE_IF_BOOL_EXEC(!get_continuous_flag, continue);
    bool attr_reuse = false;
    bool get_reuse_flag = ge::AttrUtils::GetBool(op_desc, ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
    GE_IF_BOOL_EXEC(!get_reuse_flag, continue);

    if (attr_reuse && attr_continuous) {
      auto in_data_anchor_list = n->GetAllInDataAnchors();
      if (in_data_anchor_list.size() != kVirtualOutputNodeInputSize) {
        // When current virtual node has several inputs, can't directly determine which input is the tensor for reuse.
        std::string error = "Only one input is supported, current virtual node" + FmtToStr(n->GetName()) +
            " has " + FmtToStr(in_data_anchor_list.size()) + " inputs.";
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
      GE_CHK_STATUS_RET(GetNodeMemoryType(n, memory_type, "output"), "Get node memory type failed.");
      auto iter = memory_offset_.find(memory_type);
      if (iter == memory_offset_.end()) {
        std::string error = "Memory offset does not have memory type" + FmtToStr(RT_MEMORY_HBM);
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
      GELOGD("Start to reassign memory for virtual output node, memory offset = %zu, memory type = %ld.",
             iter->second.mem_offset_, memory_type);
      string batch_label_string;
      // Not all ops have ATTR_NAME_BATCH_LABEL, no need to check return value, only check out parameter
      (void) ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label_string);
      if (batch_label_string.empty()) {
        size_t node_mem_offset = iter->second.mem_offset_;
        // No ATTR_NAME_BATCH_LABEL, no need to reuse memory.
        Status status = ReAssignVirtualOutputNodeMemory(n, node_mem_offset);
        if (status != SUCCESS) {
          GELOGE(FAILED, "Reassign memory of virtual output node failed, node name: %s.", n->GetName().c_str());
          return FAILED;
        }
        iter->second.mem_offset_ = node_mem_offset;
        AlignMemOffset(MEM_ALIGN_SIZE, memory_type);
        GELOGD("After reassign memory for virtual output node, align memory = %zu, memory type = %ld.",
               iter->second.mem_offset_, memory_type);
      } else {
        // Has ATTR_NAME_BATCH_LABEL, for dynamic multi-batch node, need to reuse memory.
        string current_node_full_name = op_desc->GetName();
        size_t pos = current_node_full_name.find(kMbatchNodeNameFlag);
        if (pos == string::npos) {
          std::string error = "Cannot find key string" + FmtToStr(kMbatchNodeNameFlag) +
          " of multi-batch in name of virtual output node, the node name is " + FmtToStr(n->GetName());
          GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
          return FAILED;
        }
        string fixed_name = current_node_full_name.substr(0, pos);
        vector<NodePtr> parallel_virtual_output_nodes;
        if (mem_reuse_virtual_output_nodes_map.count(fixed_name) != 0) {
          parallel_virtual_output_nodes = mem_reuse_virtual_output_nodes_map[fixed_name];
        }
        parallel_virtual_output_nodes.emplace_back(n);
        mem_reuse_virtual_output_nodes_map[fixed_name] = parallel_virtual_output_nodes;
      }
    }
  }

  int32_t mem_reuse_model = 1;
  if (ReAssignVirtualNodesMemory(mem_reuse_virtual_output_nodes_map, mem_reuse_model) != SUCCESS) {
    GELOGE(FAILED, "Reassign memory of virtual output nodes failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignVirtualNodesMemory(map<string, vector<NodePtr>> &mem_reuse_nodes_map,
                                                       int32_t mem_reuse_model) {
  // Find max batch label value
  string max_batch_label;
  GE_CHK_STATUS_RET(GetMaxBatchLabel(mem_reuse_nodes_map, mem_reuse_model, max_batch_label),
                    "Get max batch label failed.");
  PrintMemoryOffset();
  vector<size_t> nodes_mem_offset_list;
  for (auto &i_map : mem_reuse_nodes_map) {
    vector<NodePtr> virtual_nodes_list = i_map.second;
    int64_t memory_type = RT_MEMORY_HBM;
    GE_CHK_STATUS_RET(GetNodeListMemoryType(virtual_nodes_list, mem_reuse_model, memory_type),
                      "Get node list memory type failed.");
    auto iter = memory_offset_.find(memory_type);
    if (iter == memory_offset_.end()) {
      std::string error = "Memory offset does not have memory type" + FmtToStr(RT_MEMORY_HBM);
      GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
      return FAILED;
    }
    size_t max_batch_node_mem_offset = iter->second.mem_offset_;
    nodes_mem_offset_list.emplace_back(max_batch_node_mem_offset);
    for (auto &i_node : virtual_nodes_list) {
      // Op_desc is not nullptr, it has been checked.
      OpDescPtr op_desc = i_node->GetOpDesc();
      string batch_label_string;
      // All ops must have ATTR_NAME_BATCH_LABEL, no need to check return value.
      (void) ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label_string);
      if (batch_label_string == max_batch_label) {
        Status status = SUCCESS;
        if (mem_reuse_model == kVirtualInputNodeMemoryReuse) {
          status = ReAssignVirtualInputNodeMemory(i_node, max_batch_node_mem_offset);
        } else if (mem_reuse_model == kVirtualOutputNodeMemoryReuse) {
          status = ReAssignVirtualOutputNodeMemory(i_node, max_batch_node_mem_offset);
        } else {
          std::string error = "Invalid parameter memory reuse model, which is " + FmtToStr(mem_reuse_model);
          GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
          return FAILED;
        }

        if (status != SUCCESS) {
          GELOGE(FAILED, "Reassign memory of virtual node failed, node name: %s.", i_node->GetName().c_str());
          return FAILED;
        }
        iter->second.mem_offset_ = max_batch_node_mem_offset;
        AlignMemOffset(MEM_ALIGN_SIZE, memory_type);
        GELOGD("After reassign memory for virtual node, align memory = %zu, memory type = %ld.",
               iter->second.mem_offset_, memory_type);
        // Only assign memory of max batch nodes.
        break;
      }
    }
  }
  PrintMemoryOffset();
  size_t memory_reuse_index = 0;
  for (auto &i_map : mem_reuse_nodes_map) {
    vector<NodePtr> virtual_nodes_list = i_map.second;
    for (auto &i_node : virtual_nodes_list) {
      size_t remaining_batch_node_mem_offset = nodes_mem_offset_list[memory_reuse_index];
      Status status = SUCCESS;
      if (mem_reuse_model == kVirtualInputNodeMemoryReuse) {
        status = ReAssignVirtualInputNodeMemory(i_node, remaining_batch_node_mem_offset);
      } else if (mem_reuse_model == kVirtualOutputNodeMemoryReuse) {
        status = ReAssignVirtualOutputNodeMemory(i_node, remaining_batch_node_mem_offset);
      } else {
        std::string error = "Invalid parameter memory reuse model, which is " + FmtToStr(mem_reuse_model);
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }

      if (status != SUCCESS) {
        GELOGE(FAILED, "Reassign memory of virtual node failed, node name: %s.", i_node->GetName().c_str());
        return FAILED;
      }
    }
    memory_reuse_index++;
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignAtomicMemory(bool is_loop_graph) {
  map<NodePtr, vector<NodePtr>> normal_atomic_and_clean_nodes_map;
  vector<NodePtr> connecting_output_atomic_nodes;
  Status status = FilterAtomicNodesForMemoryAssign(normal_atomic_and_clean_nodes_map, connecting_output_atomic_nodes);
  if (status != SUCCESS) {
    GELOGE(status, "Failed to filter atomic nodes for memory assignment.");
    return status;
  }

  auto mem_iter = memory_offset_.find(RT_MEMORY_HBM);
  if (mem_iter == memory_offset_.end()) {
    std::string error = "Memory offset does not have memory type" + FmtToStr(RT_MEMORY_HBM);
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }

  for (auto &iter : normal_atomic_and_clean_nodes_map) {
    int64_t atomic_mem_start = static_cast<int64_t>(mem_iter->second.mem_offset_);
    GELOGD("Begin to reAssign atomic memory, atomic address memory start = %ld", atomic_mem_start);

    for (auto &atomic_node : iter.second) {
      vector<int64_t> mem_offset_end;
      status = AssignAtomicOutputAndWorkspaceMemory(atomic_node, mem_offset_end);
      if (status != SUCCESS) {
        GELOGE(status, "Assign atomic output and workspace memory failed, node name is %s.",
               atomic_node->GetName().c_str());
        return status;
      }
    }

    int64_t atomic_mem_size = static_cast<int64_t>(mem_iter->second.mem_offset_) - atomic_mem_start;
    if (atomic_mem_size != 0) {
      GE_CHK_STATUS_RET(SetAtomicCleanAttr(iter.first, {atomic_mem_start}, {atomic_mem_size}),
                        "Failed to set attr for atomic addr clean node %s.", iter.first->GetName().c_str());
    }
  }

  if (AssignConnectNetOutputAtomicMemory(connecting_output_atomic_nodes) != SUCCESS) {
    GELOGE(FAILED, "Failed to assign memory of nodes that connect to netoutput.");
    return FAILED;
  }

  return SUCCESS;
}

Status GraphMemoryAssigner::FilterAtomicNodesForMemoryAssign(map<NodePtr, vector<NodePtr>> &normal_atomic_nodes_map,
                                                             vector<NodePtr> &connecting_output_atomic_nodes) {
  GE_CHECK_NOTNULL(compute_graph_);
  for (const auto &node : compute_graph_->GetAllNodes()) {
    if (node->GetType() == ATOMICADDRCLEAN) {
      vector<NodePtr> tmp_normal_atomic_nodes;
      const auto &out_control_anchor = node->GetOutControlAnchor();
      GE_CHECK_NOTNULL(out_control_anchor);
      for (const auto &peer_in_control_anchor : out_control_anchor->GetPeerInControlAnchors()) {
        if (peer_in_control_anchor != nullptr) {
          auto peer_in_node = peer_in_control_anchor->GetOwnerNode();
          auto peer_in_node_desc = peer_in_node->GetOpDesc();
          if (peer_in_node_desc != nullptr) {
            bool is_atomic_node = false;
            // If GetBool fail, is_atomic_node is false.
            (void) ge::AttrUtils::GetBool(peer_in_node_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic_node);
            if (is_atomic_node) {
              bool is_reference = false;
              // If GetBool fail, is_reference is false.
              (void) ge::AttrUtils::GetBool(peer_in_node_desc, ATTR_NAME_REFERENCE, is_reference);
              if (is_reference) {
                std::string error = "Op" + FmtToStr(peer_in_node_desc->GetName()) +
                    " cannot have both atomic and is_reference attribute.";
                GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
                return ge::PARAM_INVALID;
              }

              vector<int> is_connecting_output;
              // If GetBool fail, attr is_connecting_output is an empty vector.
              (void) ge::AttrUtils::GetListInt(peer_in_node_desc, ATTR_NAME_NODE_CONNECT_OUTPUT, is_connecting_output);
              if (is_connecting_output.empty()) {
                tmp_normal_atomic_nodes.emplace_back(peer_in_node);
                continue;
              }
              connecting_output_atomic_nodes.emplace_back(peer_in_node);
              tmp_normal_atomic_nodes.clear();
              break;
            }
          }
        }
      }

      if (!tmp_normal_atomic_nodes.empty()) {
        normal_atomic_nodes_map[node] = tmp_normal_atomic_nodes;
      }
    }
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::AssignAtomicOutputAndWorkspaceMemory(const ge::NodePtr &node,
                                                                 vector<int64_t> &mem_offset_end) {
  auto node_op_desc = node->GetOpDesc();
  // Assign atomic node output memory
  Status ret = AssignAtomicOutputMemory(node, mem_offset_end);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to assign atomic output memory, node is %s.", node_op_desc->GetName().c_str());
    return ret;
  }

  // Check and assign atomic node workspace memory
  map<string, map<int64_t, int64_t>> atomic_workspace_info;
  atomic_workspace_info = node_op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);
  if (!atomic_workspace_info.empty()) {
    bool is_fusion_node = false;
    // If GetBool fail, is_fusion_node is false.
    (void) ge::AttrUtils::GetBool(node_op_desc, ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);

    if (is_fusion_node) {
      // Assign fusion atomic node workspace memory
      ret = AssignFusionAtomicWorkspaceMemory(node_op_desc, atomic_workspace_info, mem_offset_end);
    } else {
      // Assign single ordinary atomic node workspace memory, not include fusion node
      ret = AssignOrdinaryAtomicWorkspaceMemory(node_op_desc, atomic_workspace_info, mem_offset_end);
    }
    if (ret != SUCCESS) {
      GELOGE(ret, "Assign atomic workspace memory failed, node is %s.", node_op_desc->GetName().c_str());
      return ret;
    }
  } else {
    GELOGW("Current atomic node %s does not have attr ATOMIC_WORKSPACE_INFO.", node->GetName().c_str());
  }

  return SUCCESS;
}

Status GraphMemoryAssigner::AssignConnectNetOutputAtomicMemory(vector<NodePtr> &connect_netoutput_nodes) {
  auto iter = memory_offset_.find(RT_MEMORY_HBM);
  if (iter == memory_offset_.end()) {
    std::string error = "Memory offset does not have memory type" + FmtToStr(RT_MEMORY_HBM);
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }
  for (auto &node : connect_netoutput_nodes) {
    GE_CHECK_NOTNULL(node);
    if (node->GetOpDesc() == nullptr) {
      GELOGW("Current node %s op desc is nullptr, memory assignment is skipped.", node->GetName().c_str());
      continue;
    }

    // Atomic memory start addr
    int64_t original_atomic_mem_start = static_cast<int64_t>(iter->second.mem_offset_);
    GELOGD("Start to assign memory of atomic node, node name: %s, node type: %s, mem_offset: %ld.",
           node->GetName().c_str(), node->GetOpDesc()->GetType().c_str(), original_atomic_mem_start);
    vector<int64_t> mem_offset_end;
    if (AssignAtomicOutputAndWorkspaceMemory(node, mem_offset_end) != SUCCESS) {
      GELOGE(FAILED, "Assign atomic output and workspace memory failed, node is %s.", node->GetName().c_str());
      return FAILED;
    }

    // All atomic nodes use atomic_addr_clean op independently, so we need to set the attr separately.
    if (SetIndependentAtomicAttr(node, original_atomic_mem_start, mem_offset_end) != SUCCESS) {
      GELOGE(FAILED, "Failed to set atomic attr separately.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::AssignReferenceMemory() {
  for (auto &node : compute_graph_->GetDirectNode()) {
    // Get the reference type of the node, default is false
    bool is_ref = false;
    // If GetBool fail, is_ref is false.
    (void) ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_REFERENCE, is_ref);
    if (!is_ref) {
      continue;
    }

    GELOGI("Current node %s needs to support the reference relationship between output and input.",
           node->GetName().c_str());

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
        GELOGI("Reference memory: input anchor index = %d, input anchor name = %s, output anchor name = %s.", index,
               iter->first.c_str(), out_data_anchor_name.c_str());
        GE_CHECK_NOTNULL(node->GetInDataAnchor(index));
        auto peer_out_anchor = node->GetInDataAnchor(index)->GetPeerOutAnchor();
        GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
        int peer_out_anchor_index = peer_out_anchor->GetIdx();
        auto peer_out_node = peer_out_anchor->GetOwnerNode();
        auto peer_out_op_desc = peer_out_node->GetOpDesc();
        GE_CHECK_NOTNULL(peer_out_op_desc);
        output_list[out_data_anchor->GetIdx()] = peer_out_op_desc->GetOutputOffset()[peer_out_anchor_index];
        GELOGI("Reference output : Set %s name[%s] output[%d] offset to [%ld] stream_id[%ld]",
               node->GetOwnerComputeGraph()->GetName().c_str(), peer_out_op_desc->GetName().c_str(),
               out_data_anchor->GetIdx(), output_list[out_data_anchor->GetIdx()], peer_out_op_desc->GetStreamId());
      } else {
        GELOGI("Reference output : origin %s name[%s] output[%d] offset is [%ld] stream_id[%ld]",
               node->GetOwnerComputeGraph()->GetName().c_str(), out_op_desc->GetName().c_str(),
               out_data_anchor->GetIdx(), output_list[out_data_anchor->GetIdx()], out_op_desc->GetStreamId());
      }
    }

    out_op_desc->SetOutputOffset(output_list);
  }

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
      std::string error = "Op" + FmtToStr(node->GetName()) + "'s peer out node" +
          FmtToStr(peer_op_desc->GetName()) + " is invalid, only support Constant/AippData/Variable";
      GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
      return false;
    }
  }
  return true;
}

Status GraphMemoryAssigner::AssignAtomicOutputMemory(const ge::NodePtr &node, vector<int64_t> &mem_offset_end) {
  auto op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(ge::FAILED, "op_desc is null."); return ge::FAILED);
  mem_offset_end.clear();
  GELOGD("Begin to assign atomic output memory, node = %s.", op_desc->GetName().c_str());

  vector<int64_t> atomic_output_index;
  // If GetListInt fail, atomic_output_index is empty.
  (void) ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  // Check atomic output
  vector<int64_t> output_list = op_desc->GetOutputOffset();
  if (atomic_output_index.size() > output_list.size()) {
    std::string error = "Op" + FmtToStr(node->GetName()) +
        "'s size of atomic_output_index is more than the size of output_list";
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return ge::FAILED;
  }
  auto output_list_size = static_cast<int64_t>(output_list.size());
  auto iter = memory_offset_.find(RT_MEMORY_HBM);
  if (iter == memory_offset_.end()) {
    std::string error = "Memory offset does not have memory type" + FmtToStr(RT_MEMORY_HBM);
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }
  for (auto &output_index : atomic_output_index) {
    if (output_index >= output_list_size) {
      std::string error = "Op" + FmtToStr(node->GetName()) + "'s output index" + FmtToStr(output_index) +
          " is more than the size" + FmtToStr(output_list_size) + " of output_list.";
      GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
      return ge::PARAM_INVALID;
    }

    // If the input of the cascade op needs to clear the atomic addr, there is no need to clear it separately here
    bool is_assigned_mem = false;
    if (GetMemoryAssignmentStatus(node, output_index, is_assigned_mem) != SUCCESS) {
      GELOGE(ge::FAILED, "Failed to get memory assignment of node %s.", node->GetName().c_str());
      return ge::FAILED;
    }

    // If you have already assigned an atomic address, skip it, and you don't need to reassign it.
    if (is_assigned_mem) {
      GELOGI(
          "Node %s atomic output : we have assigned atomic memory as the input of next node in "
          "ReAssignContinuousMemory function.",
          op_desc->GetName().c_str());
      continue;
    }

    auto output_desc = op_desc->GetAllOutputsDescPtr().at(output_index);
    int64_t size = 0;
    if (ge::TensorUtils::GetSize(*output_desc, size) != SUCCESS) {
      GELOGI("Get size failed");
    }

    output_list[output_index] = iter->second.mem_offset_;
    GELOGI("[IMAS]Atomic output : Set %s name[%s] output[%ld] offset to [%zu] stream_id[%ld] size[%ld] real_size[%ld].",
           compute_graph_->GetName().c_str(), op_desc->GetName().c_str(), output_index,
           iter->second.mem_offset_, op_desc->GetStreamId(), size, size);

    iter->second.mem_offset_ += size;
    AlignMemOffset(MEM_ALIGN_SIZE, RT_MEMORY_HBM);
    mem_offset_end.emplace_back(iter->second.mem_offset_);
  }

  op_desc->SetOutputOffset(output_list);

  return ge::SUCCESS;
}

Status GraphMemoryAssigner::GetMemoryAssignmentStatus(const ge::NodePtr &node, int64_t output_index,
                                                      bool &is_mem_assigned) {
  if (static_cast<size_t>(output_index) >= node->GetAllOutDataAnchors().size()) {
    std::string error = "Op" + FmtToStr(node->GetName()) + "'s output index" + FmtToStr(output_index) +
        " is more than the size of node's AllOutDataAnchors.";
    GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
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
    (void) ge::AttrUtils::GetListInt(output_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, atomic_input_index);
    if (!atomic_input_index.empty() && (atomic_input_index[0] == kAllInputAddrIsAtomic)) {
      is_mem_assigned = true;
      break;
    }
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::AssignOrdinaryAtomicWorkspaceMemory(const ge::OpDescPtr &op_desc,
                                                                map<string, map<int64_t, int64_t>> &workspace_info,
                                                                vector<int64_t> &mem_offset_end) {
  GELOGI("Begin to reassign normal atomic memory, node = %s.", op_desc->GetName().c_str());
  auto mem_type_iter = memory_offset_.find(RT_MEMORY_HBM);
  if (mem_type_iter == memory_offset_.end()) {
    std::string error = "Memory offset does not have memory type" + FmtToStr(RT_MEMORY_HBM);
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }
  vector<int64_t> workspace_vector = op_desc->GetWorkspace();

  for (auto iter = workspace_info.begin(); iter != workspace_info.end(); ++iter) {
    if (op_desc->GetName() != iter->first) {
      std::string error = "The node name" + FmtToStr(op_desc->GetName()) +
          " and the node name" + FmtToStr(iter->first) + " in workspace info are inconsistent.";
      GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
      return ge::PARAM_INVALID;
    }

    if (iter->second.empty()) {
      continue;
    }

    for (auto &info_iter : iter->second) {
      auto workspace_index = static_cast<uint64_t>(info_iter.first);
      auto workspace_size = info_iter.second;
      if (workspace_index >= workspace_vector.size()) {
        std::string error = "The workspace index" + FmtToStr(workspace_index) +
            " is more than the size" + FmtToStr(workspace_vector.size()) + " of workspace vector.";
        GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
        return ge::PARAM_INVALID;
      }

      workspace_vector[workspace_index] = mem_type_iter->second.mem_offset_;
      GELOGI(
          "[IMAS]Atomic ordinary workspace : Set %s name[%s] workspace[%lu] offset to [%zu] stream_id[%ld] "
          "size[%ld] real_size[%ld].",
          compute_graph_->GetName().c_str(), op_desc->GetName().c_str(), workspace_index,
          mem_type_iter->second.mem_offset_, op_desc->GetStreamId(), workspace_size, workspace_size);

      mem_type_iter->second.mem_offset_ += workspace_size;
      mem_offset_end.emplace_back(mem_type_iter->second.mem_offset_);
    }
  }
  op_desc->SetWorkspace(workspace_vector);

  return SUCCESS;
}

Status GraphMemoryAssigner::AssignFusionAtomicWorkspaceMemory(const ge::OpDescPtr &op_desc,
                                                              map<string, map<int64_t, int64_t>> &workspace_info,
                                                              vector<int64_t> &mem_offset_end) {
  GELOGI("Begin to reassign fusion atomic memory, node = %s.", op_desc->GetName().c_str());
  auto mem_type_iter = memory_offset_.find(RT_MEMORY_HBM);
  if (mem_type_iter == memory_offset_.end()) {
    std::string error = "Memory offset does not have memory type" + FmtToStr(RT_MEMORY_HBM);
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }
  map<string, map<int64_t, int64_t>> sub_node_workspace_offset;

  for (auto &iter : workspace_info) {
    if (iter.second.empty()) {
      continue;
    }

    map<int64_t, int64_t> index_offset;
    for (auto &info_iter : iter.second) {
      auto workspace_index = static_cast<uint64_t>(info_iter.first);
      auto workspace_size = info_iter.second;

      size_t workspace_offset = mem_type_iter->second.mem_offset_;
      GELOGI(
          "[IMAS]Atomic fusion workspace : Set %s name[%s] workspace[%lu] offset to [%zu] stream_id[%ld] size[%ld] "
          "real_size[%ld].", compute_graph_->GetName().c_str(), op_desc->GetName().c_str(), workspace_index,
          mem_type_iter->second.mem_offset_, op_desc->GetStreamId(), workspace_size, workspace_size);

      mem_type_iter->second.mem_offset_ += workspace_size;
      mem_offset_end.emplace_back(mem_type_iter->second.mem_offset_);
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
  std::map<std::string, std::string> anchor_to_symbol;
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  if (GraphUtils::GetRefMapping(compute_graph_, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Get ref-mapping for graph %s failed.", compute_graph_->GetName().c_str());
    return FAILED;
  }
  for (const ge::NodePtr &node : compute_graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    vector<int64_t> input_list = node->GetOpDesc()->GetInputOffset();
    for (auto input : input_list) {
      if (input == ge::kInvalidOffset) {
        std::string error = "Invalid input offset" + FmtToStr(ge::kInvalidOffset) +
            + " in node" + FmtToStr(node->GetName());
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
    }

    bool need_update_output = false;
    vector<int64_t> output_list = node->GetOpDesc()->GetOutputOffset();
    for (uint32_t i = 0; i < output_list.size(); ++i) {
      if (output_list[i] == ge::kInvalidOffset) {
        std::string error = "Invalid output offset" + FmtToStr(ge::kInvalidOffset) +
            + " in node" + FmtToStr(node->GetName());
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
      if (node->GetType() == IDENTITY || node->GetType() == READVARIABLEOP) {
        auto symbol_offset = GetSymbolOutputOffset(anchor_to_symbol, symbol_to_anchors, node, i);
        if (symbol_offset != ge::kInvalidOffset && output_list[i] != symbol_offset) {
          output_list[i] = symbol_offset;
          need_update_output = true;
        }
      }
    }
    if (need_update_output) {
      node->GetOpDesc()->SetOutputOffset(output_list);
    }

    vector<int64_t> workspace_list = node->GetOpDesc()->GetWorkspace();
    for (auto workspace : workspace_list) {
      if (workspace == ge::kInvalidOffset) {
        std::string error = "Invalid workspace" + FmtToStr(ge::kInvalidOffset) +
            + " in node" + FmtToStr(node->GetName());
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        GELOGE(FAILED, "Invalid workspace in node: %s workspace: %ld.", node->GetName().c_str(), ge::kInvalidOffset);
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
  for (auto pair : memory_offset_) {
    GEEVENT("[IMAS]AfterAssignMemory : %s memoffset[%zu], memory type[%ld]", compute_graph_->GetName().c_str(),
            pair.second.mem_offset_, pair.first);
  }

  for (const ge::NodePtr &node : compute_graph_->GetAllNodes()) {
    if (UpdateOpInputOffset(node) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "Update op input offset failed");
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}

NodePtr GraphMemoryAssigner::GetKnownInputNode(const NodePtr &node) const {
  if (!node->GetOpDesc()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX)) {
    return node;
  }

  if (NodeUtils::IsDynamicShape(node)) {
    return node;
  }

  return NodeUtils::GetParentInput(node);
}

ge::Status GraphMemoryAssigner::UpdateConstArgsOffset(const NodePtr &node, vector<int64_t> &input_list) const {
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return SUCCESS;
  }

  // Subgraph Data Node, check for constant input.
  std::string op_type;
  const auto &in_node = NodeUtils::GetParentInput(node);
  if (NodeUtils::GetConstOpType(in_node, op_type)) {
    input_list = in_node->GetOpDesc()->GetOutputOffset();
    node->GetOpDesc()->SetOutputOffset(input_list);     // Set Data output same as const output.
    return SUCCESS;   // Constant input.
  }

  // Memory allocated for dynamic shape subgraph Data.
  if (NodeUtils::IsDynamicShape(node)) {
    return SUCCESS;
  }

  const auto &owner = node->GetOwnerComputeGraph();
  const auto &parent_desc = owner->GetParentNode()->GetOpDesc();
  const auto parent_inputs = parent_desc->GetInputOffset();
  if (parent_inputs.size() <= parent_index) {
    std::string error = "Get Parent input offset failed, node is " + FmtToStr(node->GetName()) +
        + ", input_size is " + FmtToStr(parent_inputs.size()) + ", parent index is " +
        FmtToStr(parent_index);
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }

  input_list = {parent_inputs[parent_index]};
  node->GetOpDesc()->SetOutputOffset(input_list);   // Set Data output same as parent input.
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::UpdateOpInputOffset(const NodePtr &node, vector<int64_t> &input_list) const {
  vector<int64_t> origin_input_list;
  vector<int64_t> memory_type;
  auto tmp_op_desc = node->GetOpDesc();
  origin_input_list = tmp_op_desc->GetInputOffset();
  int64_t valid_input_index = 0;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(tmp_op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, memory_type);
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
    auto out_index = static_cast<unsigned long>(peer_out_anchor->GetIdx());
    if (output_list.size() > static_cast<size_t>(out_index)) {
      int64_t input_offset = output_list.at(out_index);
      if (has_mem_type_attr && !origin_input_list.empty()) {
        auto input_size = tmp_op_desc->GetInputsSize();
        auto ori_input_offset_list_size = origin_input_list.size();
        auto mem_type_size = memory_type.size();
        if ((input_size != mem_type_size) || (input_size != ori_input_offset_list_size)) {
            std::string error = "fusion: node" + FmtToStr(tmp_op_desc->GetName()) +
                + " input_size" + FmtToStr(input_size) + " diff from memory_type_size" +
                FmtToStr(mem_type_size) + " from ori_input_offset_list_size" +
                FmtToStr(ori_input_offset_list_size);
            GE_ERRORLOG_AND_ERRORMSG(ge::FAILED, error.c_str());
          return ge::FAILED;
        }
        // not hbm keep orignal inputoffest
        // hbm inputoffset = original inputoffset + outputoffset
        input_offset = (memory_type[valid_input_index] == RT_MEMORY_L1 ? origin_input_list[valid_input_index]
                       : origin_input_list[valid_input_index] + output_list.at(out_index));
      }
      const auto &in_node = GetKnownInputNode(peer_out_anchor->GetOwnerNode());
      if (in_node->GetType() == CONSTANT) {
        GeTensorDesc tensor_desc = tmp_op_desc->GetInputDesc(static_cast<uint32_t>(anchor->GetIdx()));
        GE_CHK_STATUS(TensorUtils::GetDataOffset(tensor_desc, input_offset));
      }

      GELOGD("%s node[%s] input[%d] is set from node[%s] out index[%lu] offset[%ld]",
             has_mem_type_attr == true ? "Fusion" : "",
             tmp_op_desc->GetName().c_str(),
             valid_input_index,
             peer_out_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
             out_index,
             input_offset);
      input_list.emplace_back(input_offset);
      valid_input_index++;
    }
  }
  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::UpdateOpInputOffset(const NodePtr &node) const {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  vector<int64_t> input_list;
  if (node->GetType() == HCOMBROADCAST || node->GetType() == HVDCALLBACKBROADCAST) {
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
  } else if (node->GetType() == DATA_TYPE) {
    if (UpdateConstArgsOffset(node, input_list) != SUCCESS) {
      GELOGE(FAILED, "Update data: %s args offset failed.", node->GetName().c_str());
      return FAILED;
    }
  } else {
    if (UpdateOpInputOffset(node, input_list) != SUCCESS) {
      GELOGE(FAILED, "Update node: %s input offset failed.", node->GetName().c_str());
      return FAILED;
    }
  }

  node->GetOpDesc()->SetInputOffset(input_list);
  return SUCCESS;
}

Status GraphMemoryAssigner::SetIndependentAtomicAttr(const ge::NodePtr &node, int64_t atomic_mem_start,
                                                     const vector<int64_t> &mem_offset_end) {
  GELOGD("Start to set independent atomic attr, atomic_addr_clean memory offset start is %ld", atomic_mem_start);

  // Parsing offset and size vectors
  vector<int64_t> memory_offset_start;
  vector<int64_t> memory_offset_size;
  memory_offset_start.emplace_back(atomic_mem_start);
  for (size_t i = 0; i < mem_offset_end.size(); ++i) {
    memory_offset_start.emplace_back(mem_offset_end[i]);
    // Number 1 means element index
    auto size = memory_offset_start[i + 1] - memory_offset_start[i];
    memory_offset_size.emplace_back(size);
  }
  memory_offset_start.pop_back();
  const auto &in_control_anchor = node->GetInControlAnchor();
  if (!memory_offset_size.empty() && in_control_anchor != nullptr) {
    for (auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
      if (peer_out_control_anchor == nullptr) {
        continue;
      }
      auto peer_out_node = peer_out_control_anchor->GetOwnerNode();
      auto peer_out_node_desc = peer_out_node->GetOpDesc();
      if (peer_out_node_desc == nullptr) {
        continue;
      }

      GELOGD("Current node memory_offset vector size is %zu, node name %s, node type is %s.", memory_offset_size.size(),
             peer_out_node_desc->GetName().c_str(), peer_out_node_desc->GetType().c_str());
      if (peer_out_node_desc->GetType() == ATOMICADDRCLEAN) {
        if (SetAtomicCleanAttr(peer_out_node, memory_offset_start, memory_offset_size) != SUCCESS) {
          GELOGE(FAILED, "Set atomic clean attr failed.");
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::SetAtomicCleanAttr(const NodePtr &node, const vector<int64_t> &atomic_mem_start,
                                                   const vector<int64_t> &atomic_mem_size) {
  auto node_op_desc = node->GetOpDesc();
  if (node_op_desc != nullptr) {
    GELOGD("Node %s, set atomic clean attr start.", node->GetName().c_str());
    vector<int64_t> workspace_vector = node_op_desc->GetWorkspace();
    vector<int64_t> workspace_byte_vector = node_op_desc->GetWorkspaceBytes();
    workspace_vector.insert(workspace_vector.end(), atomic_mem_start.begin(), atomic_mem_start.end());
    workspace_byte_vector.insert(workspace_byte_vector.end(), atomic_mem_size.begin(), atomic_mem_size.end());
    node_op_desc->SetWorkspace(workspace_vector);
    node_op_desc->SetWorkspaceBytes(workspace_byte_vector);

    std::vector<int64_t> mem_start_vector;
    // If GetListInt fail, mem_start_vector is empty.
    (void) ge::AttrUtils::GetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_START, mem_start_vector);
    mem_start_vector.insert(mem_start_vector.end(), atomic_mem_start.begin(), atomic_mem_start.end());
    GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_START, mem_start_vector),
                     GELOGE(FAILED, "SetListInt failed.");
                     return FAILED);

    std::vector<int64_t> mem_size_vector;
    // If GetListInt fail, mem_size_vector is empty.
    (void) ge::AttrUtils::GetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_MEM_SIZE, mem_size_vector);
    mem_size_vector.insert(mem_size_vector.end(), atomic_mem_size.begin(), atomic_mem_size.end());
    GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_MEM_SIZE, mem_size_vector),
                     GELOGE(FAILED, "SetListInt failed.");
                     return FAILED);

    std::stringstream ss;
    for (auto iter : atomic_mem_start) {
      ss << iter << " ";
    }
    string atomic_mem_start_str = ss.str();
    ss.clear();
    ss.str("");
    for (auto iter : atomic_mem_size) {
      ss << iter << " ";
    }
    string atomic_mem_size_str = ss.str();

    GELOGI("[IMAS]SetAtomicCleanAttr : Set %s atomic_node name[%s] output[0] offset to [%s] streamid[%ld] size[%s]",
           node->GetOwnerComputeGraph()->GetName().c_str(), node_op_desc->GetName().c_str(),
           atomic_mem_start_str.c_str(), node->GetOpDesc()->GetStreamId(), atomic_mem_size_str.c_str());
  }
  return SUCCESS;
}

void GraphMemoryAssigner::AlignMemOffset(const int64_t &mem_align_size, int64_t memory_type) {
  if (mem_align_size <= 0) {
    return;
  }
  auto iter = memory_offset_.find(memory_type);
  if (iter == memory_offset_.end()) {
    GELOGW("Memory offset don't have memory type[%ld].", memory_type);
    return;
  }
  iter->second.mem_offset_ =
      (iter->second.mem_offset_ + mem_align_size - 1) / mem_align_size * mem_align_size;
}

ge::Status GraphMemoryAssigner::GetNodeListMemoryType(const vector<NodePtr> &nodes, int32_t mem_reuse_model,
                                                      int64_t &memory_type) {
  memory_type = RT_MEMORY_HBM;
  // In the dynamic batch scenario, the memory attributes of nodes are the same.
  for (auto &n : nodes) {
    if (mem_reuse_model == kVirtualInputNodeMemoryReuse) {
      GE_CHK_STATUS_RET(GetNodeMemoryType(n, memory_type, "input"), "Get node memory type failed.")
      break;
    }

    if (mem_reuse_model == kVirtualOutputNodeMemoryReuse) {
      GE_CHK_STATUS_RET(GetNodeMemoryType(n, memory_type, "output"), "Get node memory type failed.");
      break;
    }
  }
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::GetNodeMemoryType(const NodePtr &node, int64_t &memory_type, string input_or_output) {
  memory_type = RT_MEMORY_HBM;
  vector<int64_t> mem_type_list;
  if (input_or_output == "input") {
    (void) ge::AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type_list);
  }
  if (input_or_output == "output") {
    (void) ge::AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type_list);
  }
  if (mem_type_list.empty()) {
    if (memory_offset_.find(memory_type) == memory_offset_.end()) {
      std::string error = "Memory offset map does not have memory type" + FmtToStr(memory_type) +
          + ", opname is " + FmtToStr(node->GetName()) + ", optype is " + FmtToStr(node->GetType());
      GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
      return FAILED;
    }
    return SUCCESS;
  }

  if (mem_type_list.size() != node->GetAllInDataAnchorsSize()) {
    std::string error = "The size" + FmtToStr(mem_type_list.size()) +
        " of mem type list is not equal to the size of in data anchor" +
        FmtToStr(node->GetAllInDataAnchorsSize()) + ", opname is " +
        FmtToStr(node->GetName()) + ", optype is "  + FmtToStr(node->GetType());
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }

  if (!CheckContinuousMemType(mem_type_list)) {
    GELOGE(FAILED, "Check continuous memory type failed.");
    return FAILED;
  }
  // It is continuous memory and memory type is the same, so use the first memory.
  memory_type = mem_type_list[0];
  return SUCCESS;
}

bool GraphMemoryAssigner::CheckContinuousMemType(vector<int64_t> mem_type_list) {
  if (mem_type_list.size() == 0) {
    return true;
  }
  int64_t mem_type_tmp = mem_type_list[0];
  for (auto mem_type : mem_type_list) {
    if (mem_type != mem_type_tmp) {
      std::string error = "The memory is continuous, but the type of the input memory is inconsistent. They are " +
          FmtToStr(mem_type_tmp) + " and " + FmtToStr(mem_type);
      ErrorManager::GetInstance().ATCReportErrMessage("E10043", {"reason"}, {error});
      GELOGW("The memory is continuous, but the type of the input memory is inconsistent. They are [%ld] and [%ld].",
             mem_type_tmp, mem_type);
      return false;
    }
  }
  if (memory_offset_.find(mem_type_tmp) == memory_offset_.end()) {
    std::string error = "Memory offset map does not have memory type" + FmtToStr(mem_type_tmp);
    ErrorManager::GetInstance().ATCReportErrMessage("E10043", {"reason"}, {error});
    GELOGW("Memory offset map does not have memory type[%ld].", mem_type_tmp);
    return false;
  }
  return true;
}

void GraphMemoryAssigner::PrintMemoryOffset() {
  for (auto pair : memory_offset_) {
    // Assign memory of max batch nodes that have the same batch label.
    GELOGD("Reassign memory for max batch virtual nodes, memory type = %ld, memory offset = %zu.",
           pair.first, pair.second.mem_offset_);
  }
}
}  // namespace ge
