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
#include "graph/build/memory/buffer_pool_mem_assigner.h"

namespace {
const int kAllInputAddrIsAtomic = -1;
const int kVirtualInputNodeMemoryReuse = 0;
const int kVirtualOutputNodeMemoryReuse = 1;
// One state per bit cannot be repeated
enum ContinuousType { kTypeInput = 1, kTypeInputNoPadding = 2, kTypeOutput = 4, kTypeOutputNoPadding = 8 };

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

Status VariableMemoryAssigner::AssignMemory2HasRefAttrNode() {
  Status result = ge::VarMemAssignUtil::AssignMemory2HasRefAttrNode(compute_graph_);
  if (result != ge::SUCCESS) {
    return result;
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignMemory() {
  ge::HybridMemAssignerPtr mem_assigner(new(std::nothrow) HybridMemAssigner(compute_graph_));
  if (mem_assigner->Assign() != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Assign][GraphMem]graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
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
    GELOGE(ge::FAILED, "[New][Object:VariableMemoryAssigner]graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "New Object:VariableMemoryAssigner failed when assign graph memory, "
                      "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
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
    GELOGE(ge::FAILED, "[New][Object:VariableMemoryAssigner]graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "New Object:VariableMemoryAssigner failed when assign graph memory, "
                      "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return ge::FAILED;
  }
  if (variable_assigner->AssignVarAttr2Nodes() != ge::SUCCESS) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::AssignMemory2HasRefAttrNode() {
  auto variable_assigner =
      std::unique_ptr<ge::VariableMemoryAssigner>(new(std::nothrow) ge::VariableMemoryAssigner(compute_graph_));
  if (variable_assigner == nullptr) {
    GELOGE(ge::FAILED, "[New][Object:VariableMemoryAssigner]graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "New Object:VariableMemoryAssigner failed when assign graph memory, "
                      "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
  }
  if (variable_assigner->AssignMemory2HasRefAttrNode() != ge::SUCCESS) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status CalculateTensorRealSizeAndOutSize(const ge::ConstGeTensorDescPtr &output_desc,
                                             int64_t dim_index, int64_t &output_mem_size,
                                             int64_t &batch_dim_num, int64_t &out_size) {
  graphStatus graph_status = ge::TensorUtils::GetSize(*output_desc, out_size);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(FAILED, "[Get][TensorSize]");
    REPORT_INNER_ERROR("E19999", "Get tensor size failed when %s", __FUNCTION__);
    return FAILED;
  }

  GeShape output_shape = output_desc->GetShape();
  std::vector<int64_t> output_dims = output_shape.GetDims();
  if (dim_index >= static_cast<int64_t>(output_dims.size())) {
    REPORT_INNER_ERROR("E19999", "Inner param dim_index value:%ld invalid, bigger than dim size:%lu in shape:%s",
                       dim_index, output_dims.size(), output_shape.ToString().c_str());
    GELOGE(FAILED, "[Check][Param:dim_index]value:%ld invalid, bigger than dim size:%lu in shape:%s",
           dim_index, output_dims.size(), output_shape.ToString().c_str());
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
    GELOGE(graph_status, "[Calc][TensorSize]");
    return FAILED;
  }

  if (output_mem_size < 0) {
    REPORT_INNER_ERROR("E19999", "After calculating, tensor memory size:%ld invalid, less than 0. "
                       "shape:%s, format:%s, dtype:%s, maybe has dynamic shape",
                       output_mem_size,
                       output_shape.ToString().c_str(),
                       TypeUtils::FormatToSerialString(out_format).c_str(),
                       TypeUtils::DataTypeToSerialString(data_type).c_str());
    GELOGE(FAILED, "[Check][TensorSize]value:%ld invalid after calc, less than 0. shape:%s, format:%s, dtype:%s, "
           "maybe has dynamic shape",
           output_mem_size,
           output_shape.ToString().c_str(),
           TypeUtils::FormatToSerialString(out_format).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignMemory(bool is_loop_graph, map<int64_t, size_t> &mem_type_to_offset) {
  if (memory_offset_.empty()) {
    REPORT_INNER_ERROR("E19999", "InnerData memory_offset_ empty, not expected when ReAssignMemory, "
                       "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData:memory_offset_]empty is not expected, "
           "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  GE_CHK_STATUS_RET(ReAssignContinuousMemory(is_loop_graph), "ReAssignContinuousMemory Failed!");
  GE_CHK_STATUS_RET(ReAssignAtomicMemory(is_loop_graph), "ReAssignAtomicMemory Failed!");
  GE_CHK_STATUS_RET(AssignBufferPoolMemory(), "AssignBufferPoolMemory Failed!");

  size_t total_mem_offset = 0;
  for (auto pair : memory_offset_) {
    mem_type_to_offset[pair.first] = pair.second.mem_offset_;
    total_mem_offset += pair.second.mem_offset_;
  }

  auto session_id = compute_graph_->GetSessionID();
  if (total_mem_offset > VarManager::Instance(session_id)->GetGraphMemoryMaxSize()) {
    GELOGE(ge::FAILED, "[Check][TotalMemOffset] %zu is greater than memory manager malloc max size %zu, "
           "graph_id:%u, graph_name:%s, reduce your batchsize or scale your model may solve problem",
           total_mem_offset, VarManager::Instance(session_id)->GetGraphMemoryMaxSize(),
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    for (auto iter : mem_type_to_offset) {
      ErrorManager::GetInstance().ATCReportErrMessage("E19022", {"memType", "size", "item", "maxsize"},
        {std::to_string(iter.first), std::to_string(iter.second), "featuremap",
         std::to_string(VarManager::Instance(session_id)->GetGraphMemoryMaxSize())});
      GEEVENT("[IMAS]AfterAssignMemory : %s memoffset[%zu], memtype[%ld]", compute_graph_->GetName().c_str(),
              iter.second, iter.first);
    }
    return ge::FAILED;
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::AssignZeroCopyMemory(map<int64_t, size_t> &mem_offset, size_t &zero_mem_copy_size) {
  BlockMemAssignerPtr priority_assigner = std::move(mem_assigner_->GetPriorityAssinger());
  if (priority_assigner == nullptr) {
    REPORT_INNER_ERROR("E19999", "InnerData priority_assigner nullptr, not expected when AssignZeroCopyMemory, "
                       "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData:priority_assigner]nullptr is invalid, "
           "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

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
    REPORT_INNER_ERROR("E19999", "InnerData memory_offset_ does not have type[HBM], "
                       "not expected when AssignZeroCopyMemory, "
                       "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData]memory_offset_ does not have memory type[HBM]"
           "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return FAILED;
  }
  iter->second.mem_offset_ = mem_offset[RT_MEMORY_HBM];

  GELOGD("max_mem_offset:%zu, mem_offset:%zu, zero_mem_copy_size:%zu.", mem_offset[RT_MEMORY_HBM], mem_offset_tmp,
         zero_mem_copy_size);

  return SUCCESS;
}

uint32_t GetContinuousMemoryType(const OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    return 0;
  };

  bool is_continuous = false;
  uint32_t continuous_type = 0;
  // If GetBool fail, is_continuous is false.
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_INPUT, is_continuous);
  if (is_continuous) {
    continuous_type |= kTypeInput;
  } else {
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_continuous);
    if (is_continuous) {
      bool attr_reuse = false;
      (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
      if (attr_reuse) {
        continuous_type |= kTypeInputNoPadding;
      }
    }
  }

  is_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_OUTPUT, is_continuous);
  if (is_continuous) {
    continuous_type |= kTypeOutput;
  } else {
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_continuous);
    if (is_continuous) {
      bool attr_reuse = false;
      (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
      if (attr_reuse) {
        continuous_type |= kTypeOutputNoPadding;
      }
    }
  }

  if (continuous_type != 0) {
    GELOGI("[Get][MemType:Continuous]Current node %s, value is %d", op_desc->GetName().c_str(), continuous_type);
  }
  return continuous_type;
}

Status GetMemorySize(const OpDescPtr &op_desc, const ge::ConstGeTensorDescPtr &output_desc, uint32_t continuous_type,
                     int64_t &tensor_size, int64_t &nopadding_size) {
  if ((op_desc == nullptr) || (output_desc == nullptr)) {
    REPORT_INNER_ERROR("E19999", "InnerData param op_desc or output_desc is nullptr, "
                       "not expected when GetMemorySize");
    GELOGE(FAILED, "[Check][Param]op_desc or output_desc is nullptr");
  }
  tensor_size = 0;
  nopadding_size = 0;
  bool is_nopadding = ((continuous_type & kTypeInputNoPadding) != 0) || ((continuous_type & kTypeOutputNoPadding) != 0);
  if (is_nopadding) {
    int64_t attr_dim_index;
    bool get_attr_dim_flag = ge::AttrUtils::GetInt(op_desc, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, attr_dim_index);
    if (!get_attr_dim_flag) {
      REPORT_INNER_ERROR("E19999", "Get Attr:%s failed when GetMemorySize, op_name:%s",
                         ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX.c_str(), op_desc->GetName().c_str());
      GELOGE(FAILED, "[Get][Attr:%s]fail for op_name:%s",
             ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX.c_str(), op_desc->GetName().c_str());
      return FAILED;
    }

    // Calculate tensor real size of each piece of data and out size of complete data
    int64_t batch_dim_num = 1;
    if (CalculateTensorRealSizeAndOutSize(output_desc, attr_dim_index, nopadding_size, batch_dim_num, tensor_size) !=
        SUCCESS) {
      REPORT_CALL_ERROR("E19999", "CalculateTensorRealSizeAndOutSize failed, attr_dim_index:%ld, op_name:%s",
                        attr_dim_index, op_desc->GetName().c_str());
      GELOGE(FAILED, "[Calculate][NopaddingSize]failed for node %s, attr_dim_index:%ld",
             op_desc->GetName().c_str(), attr_dim_index);
      return FAILED;
    }
  } else {
    if (ge::TensorUtils::GetSize(*output_desc, tensor_size) != ge::SUCCESS) {
      REPORT_INNER_ERROR("E19999", "Get Tensor Size failed, op_name:%s", op_desc->GetName().c_str());
      GELOGE(FAILED, "[Get][TensorSize]failed in padding case, op_name:%s", op_desc->GetName().c_str());
      return FAILED;
    }
  }
  if ((tensor_size < 0) || (nopadding_size < 0)) {
    REPORT_INNER_ERROR("E19999", "GetMemorySize fail, "
                       "tensor_size:%ld or nopadding_size:%ld less than 0, invalid, op_name:%s",
                       tensor_size, nopadding_size, op_desc->GetName().c_str());
    GELOGE(FAILED, "[Get][MemorySize]tensor_size:%ld or nopadding_size:%ld less than 0, invalid, op_name:%s",
           tensor_size, nopadding_size, op_desc->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

void AlignMemOffset(int64_t &mem_align_size) {
  if (mem_align_size <= 0) {
    return;
  }
  mem_align_size = (mem_align_size + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
}

bool IsContinuousInputConflict(const ge::NodePtr &node, const OpDescPtr &peer_op_desc) {
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
                  return true;);

  bool is_peer_reference = false;
  // If GetBool fail, is_peer_reference is false.
  (void) AttrUtils::GetBool(peer_op_desc, ATTR_NAME_REFERENCE, is_peer_reference);
  GE_IF_BOOL_EXEC(is_peer_reference,
                  std::string warning = "[Check][Continuous]Current op" + FmtToStr(node->GetOpDesc()->GetName()) +
                      " requires continuous input, while the previous op" + FmtToStr(peer_op_desc->GetName()) +
                      " is ref. There may be conflict between the two.";
                  GELOGW("%s", warning.c_str());
                  return false;);
  return false;
}

Status GraphMemoryAssigner::ReAssignContinuousMemory(bool is_loop_graph) {
  Status ret;
  // Stored nodes which need assign continuous input memory in `reverse topo order`
  std::vector<NodePtr> nodes_stack;
  std::map<NodePtr, uint32_t> node_2_continuous_type;

  // Traverse nodes
  for (auto &node : compute_graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    uint32_t continuous_type;
    auto iter = node_2_continuous_type.find(node);
    if (iter == node_2_continuous_type.end()) {
      continuous_type = GetContinuousMemoryType(node->GetOpDesc());
      node_2_continuous_type.emplace(node, continuous_type);
    } else {
      continuous_type = iter->second;
    }
    // Assign continuous input memory
    bool continuous_input = ((continuous_type & kTypeInput) != 0) || ((continuous_type & kTypeInputNoPadding) != 0);
    if (continuous_input) {
      if (AssignContinuousInputMemoryWithAtomicProcessDirectly(node, node_2_continuous_type)) {
        GE_CHK_STATUS_RET(AssignContinuousInputMemoryWithAtomicProcess(node, continuous_type),
                          "[Assign][Memory:Continuous:Input]fail for node:%s", node->GetName().c_str())
      } else {
        nodes_stack.push_back(node);
      }
    }
    // Assign continuous output memory
    int64_t memory_type = RT_MEMORY_HBM;
    bool continuous_output = ((continuous_type & kTypeOutput) != 0) || ((continuous_type & kTypeOutputNoPadding) != 0);
    if (continuous_output) {
      GE_CHK_STATUS_RET(GetNodeMemoryType(node, memory_type, "output"),
                        "[Get][MemType]fail for node:%s", node->GetName().c_str());
      ret = AssignContinuousOutputMemory(node, memory_type, continuous_type);
      if (ret != ge::SUCCESS) {
        GELOGE(ret, "[Assign][Memory:Continuous:Ouput]fail for node:%s", node->GetName().c_str());
        return ret;
      }
    }
  }
  // Assign continuous input memory in `reverse topo order` which stored before
  while (!nodes_stack.empty()){
    auto node = nodes_stack.back();
    nodes_stack.pop_back();
    auto iter = node_2_continuous_type.find(node);
    if (iter == node_2_continuous_type.end()) {
      REPORT_INNER_ERROR("E19999", "Inner data error when process continuous memory alloc for node:%s, "
                         "but has no continuous type", node->GetName().c_str());
      GELOGE(FAILED, "[Get][ContinuousType] find fail for node:%s", node->GetName().c_str());
      return FAILED;
    }
    GE_CHK_STATUS_RET(AssignContinuousInputMemoryWithAtomicProcess(node, iter->second, true),
                      "[Assign][Memory:Continuous:Input]fail for node:%s.", node->GetName().c_str())
  }
  for (auto pair : memory_offset_) {
    GELOGD("[Reassign][Memory:Continuous]At last, memory type = %ld, mem offset = %zu", pair.first,
           pair.second.mem_offset_);
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignContinuousInputMemory(const ge::NodePtr &node, int64_t &continuous_mem_start,
    int64_t &continuous_mem_size, int64_t memory_type, uint32_t continuous_type, bool reverse_refresh) {
  GELOGI("[Assign][Memory:Input:Continuous]start for Current node %s", node->GetName().c_str());
  auto iter = memory_offset_.find(memory_type);
  if (iter == memory_offset_.end()) {
    REPORT_INNER_ERROR("E19999", "find memory offset fail for mem_type:%ld, "
                       "when assign continuous input memory for node:%s, ", memory_type, node->GetName().c_str());
    GELOGE(FAILED, "[Find][MemOffset]fail for mem_type:%ld, when AssignContinuousInputMemory for node:%s",
           memory_type, node->GetName().c_str());
    return FAILED;
  }
  // The head and tail of hcom continuous input should be added 512
  iter->second.mem_offset_ += MEM_ALIGN_SIZE;
  continuous_mem_start = iter->second.mem_offset_;
  int64_t mem_offset = iter->second.mem_offset_;
  int64_t extra_memory_size = 0;
  bool is_continuous_input_allocated = false;
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  vector<int64_t> output_list_this = op_desc->GetOutputOffset();
  if (output_list_this.empty()) {
    REPORT_INNER_ERROR("E19999", "No output offset in node :%s, not expected when assign continuous input memory",
                       node->GetName().c_str());
    GELOGE(FAILED, "[Get][OutputOffset] empty is invalid, node:%s", node->GetName().c_str());
    return FAILED;
  }
  (void) ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_INPUT_ALLOC, is_continuous_input_allocated);
  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    GE_IF_BOOL_EXEC(in_data_anchor == nullptr, continue);
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_data_anchor == nullptr, continue);
    auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
    GE_IF_BOOL_EXEC(peer_op_desc == nullptr, continue);
    GE_IF_BOOL_EXEC(IsContinuousInputConflict(node, peer_op_desc), return PARAM_INVALID;);

    int64_t tensor_desc_size = 0;
    int64_t nopadding_size = 0;
    int64_t real_size = 0;
    std::vector<int64_t> offsets_of_fusion = {};
    bool lx_fusion = AttrUtils::GetListInt(peer_op_desc, ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
    lx_fusion = lx_fusion && !offsets_of_fusion.empty();
    if (lx_fusion) {
      if (peer_out_data_anchor->GetIdx() >= static_cast<int>(offsets_of_fusion.size())) {
        std::string error = "fusion: peer node:" + FmtToStr(peer_op_desc->GetName()) +
            " anchor_index:" + FmtToStr(peer_out_data_anchor->GetIdx()) +
            " is out of range:" + FmtToStr(offsets_of_fusion.size());
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
      nopadding_size = offsets_of_fusion[peer_out_data_anchor->GetIdx()];
      tensor_desc_size = nopadding_size;
    } else {
      if (GetMemorySize(node->GetOpDesc(), peer_op_desc->GetOutputDescPtr(peer_out_data_anchor->GetIdx()),
                        continuous_type, tensor_desc_size, nopadding_size) != ge::SUCCESS) {
        return FAILED;
      }
    }

    bool is_nopadding = ((continuous_type & kTypeInputNoPadding) != 0) || lx_fusion;
    vector<int64_t> output_list = peer_op_desc->GetOutputOffset();
    if (peer_out_data_anchor->GetIdx() >= static_cast<int>(output_list.size())) {
      std::string error = "peer node:" + FmtToStr(peer_op_desc->GetName()) +
          " anchor_index:" + FmtToStr(peer_out_data_anchor->GetIdx()) +
          " is out of range:" + FmtToStr(output_list.size());
      GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
      return FAILED;
    }

    // when continuous input has been allocated first input is beginning offset
    bool is_allocated_first_input = is_continuous_input_allocated && (in_data_anchor->GetIdx() == 0);
    if (is_allocated_first_input) {
      std::map<int32_t, int32_t> out2ins;
      GE_CHK_STATUS_RET(GetAllRef(node, out2ins), "[Get][AllRef]fail for node: %s", node->GetName().c_str());
      // output is beginning offset, set offset for input; only support this case now
      if ((out2ins.size() == 1) && (out2ins.begin()->second == 0) && (reverse_refresh)) {
        auto peer_output_offset = output_list.at(peer_out_data_anchor->GetIdx());
        output_list.at(peer_out_data_anchor->GetIdx()) = output_list_this.at(out2ins.begin()->first);
        peer_op_desc->SetOutputOffset(output_list);
        GELOGI("[Update][Offset]Node %s out %d ref in %d input node %s, use output offset %ld update %ld",
               node->GetName().c_str(), out2ins.begin()->first, out2ins.begin()->second,
               peer_op_desc->GetName().c_str(), output_list_this.at(out2ins.begin()->first), peer_output_offset);
      } else {
        GELOGD("Node %s out %d ref in %d input node %s with total ref numbers %zu.", node->GetName().c_str(),
               out2ins.begin()->first, out2ins.begin()->second, peer_op_desc->GetName().c_str(), out2ins.size());
      }
      // first input is beginning offset
      mem_offset = output_list.at(peer_out_data_anchor->GetIdx());
      continuous_mem_start = output_list.at(peer_out_data_anchor->GetIdx());
    } else {
      // set offset for input
      output_list.at(peer_out_data_anchor->GetIdx()) = mem_offset;
      peer_op_desc->SetOutputOffset(output_list);
    }

    int64_t align_size = tensor_desc_size;
    if (is_nopadding) {
      mem_offset += nopadding_size;
      extra_memory_size += (tensor_desc_size - nopadding_size);
      real_size = nopadding_size;
    } else {
      ge::AlignMemOffset(align_size);
      mem_offset += align_size;
      // The head and tail of hcom continuous input should be added 512
      extra_memory_size = MEM_ALIGN_SIZE;
      real_size = tensor_desc_size;
    }

    GELOGI("[IMAS]Continuous input : Set %s name[%s] optype[%s] output[%d] offset to [%zu] stream_id[%ld] memtype[%ld] "
        "size[%zu] realsize[%ld] nopadding size[%d]", node->GetOwnerComputeGraph()->GetName().c_str(),
        peer_op_desc->GetName().c_str(), node->GetType().c_str(), peer_out_data_anchor->GetIdx(),
        output_list.at(peer_out_data_anchor->GetIdx()), peer_op_desc->GetStreamId(), memory_type,
        is_continuous_input_allocated ? 0UL : align_size, real_size, is_nopadding);
  }

  mem_offset += extra_memory_size;
  ge::AlignMemOffset(mem_offset);
  continuous_mem_size = mem_offset - continuous_mem_start;
  if (is_continuous_input_allocated) {
    // not allocate memory here, so no need add 512 in header
    iter->second.mem_offset_ -= MEM_ALIGN_SIZE;
  } else {
    iter->second.mem_offset_ = mem_offset;
  }
  return SUCCESS;
}

Status GetFirstInputPeerOutOutputOffset(const ge::NodePtr &node, int64_t &mem_offset) {
  auto in_data_anchor_list = node->GetAllInDataAnchors();
  if (in_data_anchor_list.empty()) {
    REPORT_INNER_ERROR("E19999", "InAnchor list empty in node:%s, not expect when GetFirstInputPeerOutOutputOffset",
                       node->GetName().c_str());
    GELOGE(FAILED, "[Get][InAnchor]empty is invalid, node:%s", node->GetName().c_str());
    return FAILED;
  }
  auto peer_out_data_anchor = in_data_anchor_list.at(0)->GetPeerOutAnchor();
  GE_IF_BOOL_EXEC(peer_out_data_anchor == nullptr,
                  REPORT_INNER_ERROR("E19999", "PeerAcnhor is null, "
                                     "not expect when GetFirstInputPeerOutOutputOffset for node:%s",
                                     node->GetName().c_str());
                  GELOGE(ge::FAILED, "[Check][PeerAnchor]null is invalid, node:%s", node->GetName().c_str());
                  return ge::FAILED);
  auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
  GE_IF_BOOL_EXEC(peer_op_desc == nullptr,
                  REPORT_INNER_ERROR("E19999", "PeerOpDesc is null, "
                                     "not expect when GetFirstInputPeerOutOutputOffset for node:%s",
                                     node->GetName().c_str());
                  GELOGE(ge::FAILED, "[Check][PeerOpDesc]null is invalid, node:%s",  node->GetName().c_str());
                  return ge::FAILED);
  vector<int64_t> in_node_output_offsets = peer_op_desc->GetOutputOffset();
  if (peer_out_data_anchor->GetIdx() >= static_cast<int>(in_node_output_offsets.size())) {
    REPORT_INNER_ERROR("E19999", "PeerAnchorIndex:%d bigger than in_offset size:%lu, "
                       "judge invalid when GetFirstInputPeerOutOutputOffset for node:%s",
                       peer_out_data_anchor->GetIdx(), in_node_output_offsets.size(), node->GetName().c_str());
    GELOGE(FAILED, "[Check][Index:PeerOutDataAnchor]PeerIndex:%d bigger than in_offset size:%lu, node:%s",
           peer_out_data_anchor->GetIdx(), in_node_output_offsets.size(), node->GetName().c_str());
    return FAILED;
  }
  mem_offset = in_node_output_offsets.at(peer_out_data_anchor->GetIdx());
  return SUCCESS;
}

Status GraphMemoryAssigner::AssignContinuousOutputMemory(const ge::NodePtr &node, int64_t memory_type,
                                                         uint32_t continuous_type) {
  GELOGI("Current node %s needs continuous output.", node->GetName().c_str());
  auto out_op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(out_op_desc == nullptr,
                  REPORT_INNER_ERROR("E19999", "OpDesc is null, "
                                     "not expect when AssignContinuousOutputMemory for node:%s",
                                     node->GetName().c_str());
                  GELOGE(ge::FAILED, "[Check][OpDesc]null is invalid, node:%s",  node->GetName().c_str()));
  vector<int64_t> output_list = out_op_desc->GetOutputOffset();
  if ((out_op_desc->GetOutputsSize() > output_list.size()) || (output_list.size() == 0)) {
    REPORT_INNER_ERROR("E19999", "Output size:%zu more than output offset size:%zu, invalid in node:%s, "
                       "when AssignContinuousOutputMemory",
                       out_op_desc->GetOutputsSize(), output_list.size(), node->GetName().c_str());
    GELOGE(ge::FAILED, "[Check][InnerData]Output size:%zu more than output offset size:%zu, invalid in node:%s",
           out_op_desc->GetOutputsSize(), output_list.size(), node->GetName().c_str());
    return ge::FAILED;
  }

  int64_t mem_offset = 0;
  bool is_nopadding = ((continuous_type & kTypeOutputNoPadding) != 0);
  if (is_nopadding) {
    // out tensor memory must be reused input tensor memory
    if (GetFirstInputPeerOutOutputOffset(node, mem_offset) != SUCCESS) {
      return ge::FAILED;
    }
  } else {
    // Get the reference type of the node, default is false
    bool is_ref = false;
    // If GetBool fail, is_ref is false.
    (void) ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_REFERENCE, is_ref);

    // If the output is ref type and refers to the ref of an input, the name of the output
    // and the input are the same. Ge encounters ref type, finds matching relationship according
    // to the names of input and output, and allocates the same memory address, eg: HCOMBroadcast
    if (is_ref) {
      GELOGI("Current node %s no needs assign continuous output because reference input by name.",
             node->GetName().c_str());
      return SUCCESS;
    }
    mem_offset = output_list[0];
  }

  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    output_list[out_data_anchor->GetIdx()] = mem_offset;
    int64_t tensor_desc_size = 0;
    int64_t nopadding_size = 0;
    if (GetMemorySize(out_op_desc, out_op_desc->GetOutputDescPtr(out_data_anchor->GetIdx()), continuous_type,
                      tensor_desc_size, nopadding_size) != ge::SUCCESS) {
      return FAILED;
    }

    if (is_nopadding) {
      mem_offset += nopadding_size;
    } else {
      mem_offset += tensor_desc_size;
      ge::AlignMemOffset(mem_offset);
    }
    GELOGI("[IMAS]Continuous output : Set %s name[%s] optype[%s] output[%d] offset to [%zu] stream_id[%ld] memtype[%ld]"
           " size[%zu] realsize[%ld] nopadding[%d].", node->GetOwnerComputeGraph()->GetName().c_str(),
           out_op_desc->GetName().c_str(), node->GetType().c_str(), out_data_anchor->GetIdx(),
           output_list[out_data_anchor->GetIdx()], out_op_desc->GetStreamId(), memory_type, 0UL,
           is_nopadding ? nopadding_size : tensor_desc_size, is_nopadding);
  }
  out_op_desc->SetOutputOffset(output_list);
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::ReAssignAtomicMemory(bool is_loop_graph) {
  // key:dynamic batch, batch name
  map<string, map<NodePtr, vector<NodePtr>>> normal_atomic_and_clean_nodes_map;
  map<string, vector<NodePtr>> connecting_output_atomic_nodes;
  Status status = FilterAtomicNodesForMemoryAssign(normal_atomic_and_clean_nodes_map, connecting_output_atomic_nodes);
  if (status != SUCCESS) {
    GELOGE(status, "[Filter][AtomicNode]failed in graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return status;
  }

  auto mem_iter = memory_offset_.find(RT_MEMORY_HBM);
  if (mem_iter == memory_offset_.end()) {
    REPORT_INNER_ERROR("E19999", "InnerData memory_offset_ does not have type[HBM], "
                       "not expected when ReAssignAtomicMemory, "
                       "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData]memory_offset_ does not have memory type[HBM]"
           "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return FAILED;
  }

  int64_t batch_atomic_mem_start = static_cast<int64_t>(mem_iter->second.mem_offset_);
  int64_t batch_max_mem_offset = batch_atomic_mem_start;
  for (auto &iter_batch : normal_atomic_and_clean_nodes_map) {
    mem_iter->second.mem_offset_ = batch_atomic_mem_start;
    for (auto &iter : iter_batch.second) {
      int64_t atomic_mem_start = static_cast<int64_t>(mem_iter->second.mem_offset_);
      GELOGD("Begin to reAssign atomic memory, atomic address memory start = %ld", atomic_mem_start);

      for (auto &atomic_node : iter.second) {
        vector<int64_t> mem_offset_end;
        status = AssignAtomicOutputAndWorkspaceMemory(atomic_node, mem_offset_end);
        if (status != SUCCESS) {
          GELOGE(status, "[Assign][Memory]output atomic mem and workspace mem, fail for node name is %s.",
                 atomic_node->GetName().c_str());
          return status;
        }
      }

      int64_t atomic_mem_size = static_cast<int64_t>(mem_iter->second.mem_offset_) - atomic_mem_start;
      if (atomic_mem_size != 0) {
        GE_CHK_STATUS_RET(SetAtomicCleanAttr(iter.first, {atomic_mem_start}, {atomic_mem_size}, RT_MEMORY_HBM),
                          "[Set][Attr]fail for atomic addr clean node %s.", iter.first->GetName().c_str());
      }
    }
    batch_max_mem_offset = std::max(batch_max_mem_offset, static_cast<int64_t>(mem_iter->second.mem_offset_));
  }

  mem_iter->second.mem_offset_ = static_cast<size_t>(batch_max_mem_offset);
  batch_atomic_mem_start = batch_max_mem_offset;
  for (auto &iter_batch : connecting_output_atomic_nodes) {
    mem_iter->second.mem_offset_ = batch_atomic_mem_start;
    if (AssignConnectNetOutputAtomicMemory(iter_batch.second) != SUCCESS) {
      GELOGE(FAILED, "[Assign][Memory]for nodes that connect to netoutput failed."
             "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
      return FAILED;
    }
    batch_max_mem_offset = std::max(batch_max_mem_offset, static_cast<int64_t>(mem_iter->second.mem_offset_));
  }
  mem_iter->second.mem_offset_ = static_cast<size_t>(batch_max_mem_offset);
  return SUCCESS;
}

Status GraphMemoryAssigner::FilterAtomicNodesForMemoryAssign(
    map<string, map<NodePtr, vector<NodePtr>>> &normal_atomic_nodes_map,
    map<string, vector<NodePtr>> &connecting_output_atomic_nodes) {
  GE_CHECK_NOTNULL(compute_graph_);
  for (const auto &node : compute_graph_->GetAllNodes()) {
    if (node->GetType() == ATOMICADDRCLEAN) {
      map<string, vector<NodePtr>> tmp_normal_atomic_nodes;
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
                REPORT_INNER_ERROR("E19999", "Op:%s cannot have both atomic and is_reference attribute, "
                                   "not support now", peer_in_node_desc->GetName().c_str());
                GELOGE(FAILED, "[Check][Attr]Op:%s cannot have both atomic and is_reference attribute, "
                               "not support now", peer_in_node_desc->GetName().c_str());
                return ge::PARAM_INVALID;
              }

              std::string batch_label;
              (void)ge::AttrUtils::GetStr(peer_in_node_desc, ATTR_NAME_BATCH_LABEL, batch_label);

              vector<int> is_connecting_output;
              // If GetBool fail, attr is_connecting_output is an empty vector.
              (void) ge::AttrUtils::GetListInt(peer_in_node_desc, ATTR_NAME_NODE_CONNECT_OUTPUT, is_connecting_output);
              if (is_connecting_output.empty()) {
                tmp_normal_atomic_nodes[batch_label].emplace_back(peer_in_node);
                continue;
              }
              connecting_output_atomic_nodes[batch_label].emplace_back(peer_in_node);
              tmp_normal_atomic_nodes[batch_label].clear();
              break;
            }
          }
        }
      }

      for (auto &it_atomic_node : tmp_normal_atomic_nodes) {
        if (!it_atomic_node.second.empty()) {
          normal_atomic_nodes_map[it_atomic_node.first][node] = it_atomic_node.second;
        }
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
    GELOGE(ret, "[Assign][Memory:Ouput:Atomic]Failed for node:%s.", node_op_desc->GetName().c_str());
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
      GELOGE(ret, "[Assign][Memory:Atomic:Workspace]fail for node:%s.", node_op_desc->GetName().c_str());
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
    REPORT_INNER_ERROR("E19999", "InnerData memory_offset_ does not have type[HBM], "
                       "not expected when AssignConnectNetOutputAtomicMemory, "
                       "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData]memory_offset_ does not have memory type[HBM]"
           "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
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
      GELOGE(FAILED, "[Assign][Memory]output atomic mem and workspace mem, fail for node name is %s.",
             node->GetName().c_str());
      return FAILED;
    }

    // All atomic nodes use atomic_addr_clean op independently, so we need to set the attr separately.
    if (SetIndependentAtomicAttr(node, original_atomic_mem_start, mem_offset_end, RT_MEMORY_HBM) != SUCCESS) {
      GELOGE(FAILED, "[Set][Attr:IndependentAtomic]fail for node:%s", node->GetName().c_str());
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
      REPORT_INNER_ERROR("E19999", "Output size:%zu more than output offset size:%zu, judge invalid in node:%s "
                         "when AssignReferenceMemory",
                         out_op_desc->GetOutputsSize(), output_list.size(), node->GetName().c_str());
      GELOGE(ge::FAILED, "[Check][InnerData]Output size:%zu more than output offset size:%zu, invalid in node:%s",
             out_op_desc->GetOutputsSize(), output_list.size(), node->GetName().c_str());
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
      REPORT_INNER_ERROR("E19999", "node(type:%s, name:%s) link to atomic node(name:%s), "
                         "this situation not supported now",
                         peer_op_desc->GetType().c_str(), peer_op_desc->GetName().c_str(), node->GetName().c_str());
      GELOGE(ge::FAILED, "[Check][Link]node(type:%s, name:%s) link to atomic node(name:%s), "
             "this situation not supported now",
             peer_op_desc->GetType().c_str(), peer_op_desc->GetName().c_str(), node->GetName().c_str());
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
    std::string error =
        "Op:" + FmtToStr(node->GetName()) + "'s size:" + FmtToStr(atomic_output_index.size()) +
        " of atomic_output_index is more than the size:" + FmtToStr(output_list.size()) + " of output_list";
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return ge::FAILED;
  }
  auto output_list_size = static_cast<int64_t>(output_list.size());
  auto iter = memory_offset_.find(RT_MEMORY_HBM);
  if (iter == memory_offset_.end()) {
    REPORT_INNER_ERROR("E19999", "InnerData memory_offset_ does not have type[HBM], "
                       "not expected when AssignAtomicOutputMemory, "
                       "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData]memory_offset_ does not have memory type[HBM]"
           "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return FAILED;
  }
  for (auto &output_index : atomic_output_index) {
    if (output_index >= output_list_size) {
      std::string error =
          "Op:" + FmtToStr(node->GetName()) + "'s atomic_output index:" + FmtToStr(output_index) +
          " is more than the size:" + FmtToStr(output_list_size) + " of output_list.";
      GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
      return ge::PARAM_INVALID;
    }

    // If the input of the cascade op needs to clear the atomic addr, there is no need to clear it separately here
    bool is_assigned_mem = false;
    if (GetMemoryAssignmentStatus(node, output_index, is_assigned_mem) != SUCCESS) {
      GELOGE(ge::FAILED, "[Get][MemoryAssignmentStatus]fail for node %s, out_index:%ld",
             node->GetName().c_str(), output_index);
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
    std::string batch_label;
    (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
    GELOGI("[IMAS]Atomic output : Set %s name[%s] optype[%s] output[%ld] offset to [%zu] stream_id[%ld] memtype[%u] "
           "size[%ld] real_size[%ld] batch[%s].", compute_graph_->GetName().c_str(), op_desc->GetName().c_str(),
           node->GetType().c_str(), output_index, iter->second.mem_offset_, op_desc->GetStreamId(), RT_MEMORY_HBM,
           size, size, batch_label.c_str());

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
    std::string error =
        "Op:" + FmtToStr(node->GetName()) + "'s output index:" + FmtToStr(output_index) +
        " is more than the size:" + FmtToStr(node->GetAllOutDataAnchors().size()) + " of node's AllOutDataAnchors.";
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
    REPORT_INNER_ERROR("E19999", "InnerData memory_offset_ does not have type[HBM], "
                       "not expected when AssignOrdinaryAtomicWorkspaceMemory, "
                       "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData]memory_offset_ does not have memory type[HBM]"
           "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
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
        std::string error = "The workspace index:" + FmtToStr(workspace_index) +
            " is more than the size:" + FmtToStr(workspace_vector.size()) + " of workspace vector in op:" +
            op_desc->GetName().c_str();
        GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
        return ge::PARAM_INVALID;
      }

      workspace_vector[workspace_index] = mem_type_iter->second.mem_offset_;
      std::string batch_label;
      (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
      GELOGI(
          "[IMAS]Atomic ordinary workspace : Set %s name[%s] optype[%s] workspace[%lu] offset to [%zu] stream_id[%ld] "
          "memtype[%u] size[%ld] real_size[%ld] batch[%s].",
          compute_graph_->GetName().c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), workspace_index,
          mem_type_iter->second.mem_offset_, op_desc->GetStreamId(), RT_MEMORY_HBM, workspace_size, workspace_size,
          batch_label.c_str());

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
    REPORT_INNER_ERROR("E19999", "InnerData memory_offset_ does not have type[HBM], "
                       "not expected when AssignFusionAtomicWorkspaceMemory, "
                       "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData]memory_offset_ does not have memory type[HBM]"
           "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
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
      std::string batch_label;
      (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
      GELOGI(
          "[IMAS]Atomic fusion workspace : Set %s name[%s] optype[%s] workspace[%lu] offset to [%zu] stream_id[%ld] "
          "memtype[%u] ssize[%ld] real_size[%ld] batch[%s].", compute_graph_->GetName().c_str(),
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), workspace_index, mem_type_iter->second.mem_offset_,
          op_desc->GetStreamId(), RT_MEMORY_HBM, workspace_size, workspace_size, batch_label.c_str());

      mem_type_iter->second.mem_offset_ += workspace_size;
      mem_offset_end.emplace_back(mem_type_iter->second.mem_offset_);
      index_offset.insert(std::make_pair(workspace_index, workspace_offset));
    }
    sub_node_workspace_offset.insert(std::make_pair(iter.first, index_offset));
  }
  if (!(op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_OFFSET, sub_node_workspace_offset))) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for node:%s when AssignFusionAtomicWorkspaceMemory",
                       EXT_ATTR_ATOMIC_WORKSPACE_OFFSET.c_str(), op_desc->GetName().c_str());
    GELOGE(FAILED, "[Set][Attr:%s]fail for node:%s.",
           EXT_ATTR_ATOMIC_WORKSPACE_OFFSET.c_str(), op_desc->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status GraphMemoryAssigner::CheckOffset() {
  std::map<std::string, std::string> anchor_to_symbol;
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  if (GraphUtils::GetRefMapping(compute_graph_, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get ref-mapping for graph %s failed", compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Get][RefMapping]fail for graph %s", compute_graph_->GetName().c_str());
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
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::SetInputOffset() {
  if (memory_offset_.empty()) {
    REPORT_INNER_ERROR("E19999", "InnerData memory_offset_ empty, not expected when SetInputOffset, "
                       "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData:memory_offset_]empty is not expected, "
           "graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
  }
  for (auto pair : memory_offset_) {
    GEEVENT("[IMAS]AfterAssignMemory : %s memoffset[%zu], memtype[%ld]", compute_graph_->GetName().c_str(),
            pair.second.mem_offset_, pair.first);
  }

  for (const ge::NodePtr &node : compute_graph_->GetAllNodes()) {
    if (UpdateOpInputOffset(node) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "[Update][Offset:Input]fail for op:%s", node->GetName().c_str());
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

      GELOGD("%s node[%s] input[%ld] is set from node[%s] out index[%lu] offset[%ld]",
             has_mem_type_attr ? "Fusion" : "",
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
      GELOGE(FAILED, "[Update][Offset:Input:Const]fail for node:%s ", node->GetName().c_str());
      return FAILED;
    }
  } else {
    if (UpdateOpInputOffset(node, input_list) != SUCCESS) {
      GELOGE(FAILED, "[Update][Offset:Input]fail for node:%s", node->GetName().c_str());
      return FAILED;
    }
  }

  node->GetOpDesc()->SetInputOffset(input_list);
  return SUCCESS;
}

Status GraphMemoryAssigner::SetIndependentAtomicAttr(const ge::NodePtr &node, int64_t atomic_mem_start,
                                                     const vector<int64_t> &mem_offset_end, int64_t memory_type) {
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
        if (SetAtomicCleanAttr(peer_out_node, memory_offset_start, memory_offset_size, memory_type) != SUCCESS) {
          GELOGE(FAILED, "[Set][AtomicCleanAttr]fail for node:%s", peer_out_node->GetName().c_str());
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::SetAtomicCleanAttr(const NodePtr &node, const vector<int64_t> &atomic_mem_start,
                                                   const vector<int64_t> &atomic_mem_size, int64_t memory_type) {
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
                     REPORT_INNER_ERROR("E19999", "Set Attr:%s failed when SetAtomicCleanAttr, op_name:%s",
                                        ATTR_NAME_AUTOMIC_ADD_START.c_str(), node_op_desc->GetName().c_str());
                     GELOGE(FAILED, "[Set][Attr:%s]fail for op_name:%s",
                            ATTR_NAME_AUTOMIC_ADD_START.c_str(), node_op_desc->GetName().c_str());
                     return FAILED);

    std::vector<int64_t> mem_size_vector;
    // If GetListInt fail, mem_size_vector is empty.
    (void) ge::AttrUtils::GetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_MEM_SIZE, mem_size_vector);
    mem_size_vector.insert(mem_size_vector.end(), atomic_mem_size.begin(), atomic_mem_size.end());
    GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(node_op_desc, ATTR_NAME_AUTOMIC_ADD_MEM_SIZE, mem_size_vector),
                     REPORT_INNER_ERROR("E19999", "Set Attr:%s failed when SetAtomicCleanAttr, op_name:%s",
                                        ATTR_NAME_AUTOMIC_ADD_MEM_SIZE.c_str(), node_op_desc->GetName().c_str());
                     GELOGE(FAILED, "[Set][Attr:%s]fail for op_name:%s",
                            ATTR_NAME_AUTOMIC_ADD_MEM_SIZE.c_str(), node_op_desc->GetName().c_str());
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

    GELOGI("[IMAS]SetAtomicCleanAttr : Set %s atomic_node name[%s] optype[%s] output[0] offset to [%s] streamid[%ld]"
           " memtype[%ld] size[%s]",node->GetOwnerComputeGraph()->GetName().c_str(), node_op_desc->GetName().c_str(),
           node->GetType().c_str(), atomic_mem_start_str.c_str(), node->GetOpDesc()->GetStreamId(), memory_type,
           atomic_mem_size_str.c_str());
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
      GE_CHK_STATUS_RET(GetNodeMemoryType(n, memory_type, "input"),
                        "[Get][MemType:input]fail for node:%s", n->GetName().c_str())
      break;
    }

    if (mem_reuse_model == kVirtualOutputNodeMemoryReuse) {
      GE_CHK_STATUS_RET(GetNodeMemoryType(n, memory_type, "output"),
                        "[Get][MemType:output]fail for node:%s", n->GetName().c_str())
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
    GELOGE(FAILED, "[Check][MemType:Continuous]fail for node:%s", node->GetName().c_str());
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

ge::Status GraphMemoryAssigner::GetAllRef(const NodePtr &node, map<int32_t, int32_t> &out2ins) {
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    int32_t reuse_in_index = -1;
    bool reuse_input_flag = GraphUtils::IsRefFromInput(out_data_anchor, reuse_in_index);
    if (reuse_input_flag) {
      if (node->GetInDataAnchor(reuse_in_index) != nullptr) {
        out2ins.emplace(out_data_anchor->GetIdx(), reuse_in_index);
      } else {
        REPORT_INNER_ERROR("E19999", "Invalid reuse_input value %d on output %d of node %s, "
                           "please check attr reuse_input",
                           reuse_in_index, out_data_anchor->GetIdx(), node->GetName().c_str());
        GELOGE(FAILED, "[Check][Attr]Invalid reuse_input value %d on output %d of node %s, "
               "please check attr reuse_input",
               reuse_in_index, out_data_anchor->GetIdx(), node->GetName().c_str());
        return FAILED;
      }
    }
  }

  return ge::SUCCESS;
}

bool GraphMemoryAssigner::AssignContinuousInputMemoryWithAtomicProcessDirectly(
  const NodePtr &input_continuous_node, map<NodePtr, uint32_t> &node_2_continuous_type) {
  for (const auto &in_node : input_continuous_node->GetInDataNodes()) {
    if (in_node->GetType() == VARIABLE) {
      GELOGI("node %s 's precursor node %s is variable, do not store.", input_continuous_node->GetName().c_str(),
             in_node->GetName().c_str());
      return true;
    }
    auto iter = node_2_continuous_type.find(in_node);
    // In node's topo order in the front, so function can not be exception
    auto continuous_type = iter->second;
    bool continuous_input = ((continuous_type & kTypeInput) != 0) || ((continuous_type & kTypeInputNoPadding) != 0);
    if (continuous_input) {
      GELOGI("[Store][Node] of %s cause it's precursor node %s need assign continuous input memory",
             input_continuous_node->GetName().c_str(), in_node->GetName().c_str());
      return false;
    }
  }
  for (const auto &out_node : input_continuous_node->GetOutDataNodes()) {
    auto continuous_type = GetContinuousMemoryType(out_node->GetOpDesc());
    node_2_continuous_type.emplace(out_node, continuous_type);
    bool continuous_input = ((continuous_type & kTypeInput) != 0) || ((continuous_type & kTypeInputNoPadding) != 0);
    if (continuous_input) {
      GELOGI("[Store][Node] of %s cause it's succeed node %s need assign continuous input memory",
             input_continuous_node->GetName().c_str(), out_node->GetName().c_str());
      return false;
    }
  }

  return true;
}

ge::Status GraphMemoryAssigner::AssignContinuousInputMemoryWithAtomicProcess(const NodePtr &input_continuous_node,
                                                                             uint32_t continuous_type,
                                                                             bool reverse_refresh) {
  int64_t mem_clean_start = 0;
  int64_t mem_clean_size = 0;
  int64_t memory_type = RT_MEMORY_HBM;

  GE_CHK_STATUS_RET(GetNodeMemoryType(input_continuous_node, memory_type, "input"),
                    "[Get][MemType]fail for node:%s", input_continuous_node->GetName().c_str());
  auto ret = AssignContinuousInputMemory(input_continuous_node, mem_clean_start, mem_clean_size, memory_type,
                                         continuous_type, reverse_refresh);
  if (ret != ge::SUCCESS) {
    GELOGE(ret, "[Assign][Memory:Input:continuous]fail for node:%s", input_continuous_node->GetName().c_str());
    return ret;
  }

  // Clean up atomic address, eg, hcom node
  vector<int32_t> input_indexes;
  // If GetListInt fail, input_indexes is empty.
  (void)ge::AttrUtils::GetListInt(input_continuous_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, input_indexes);
  if (!input_indexes.empty() && input_indexes[0] == kAllInputAddrIsAtomic) {
    // check whether there is an atomic conflict between the current node and the peer out node
    if (!CheckInputIsSupportAtomic(input_continuous_node)) {
      return ge::FAILED;
    }

    const auto &in_control_anchor = input_continuous_node->GetInControlAnchor();
    GE_CHECK_NOTNULL(in_control_anchor);
    for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
      GE_CHECK_NOTNULL(peer_out_control_anchor);
      auto peer_out_node = peer_out_control_anchor->GetOwnerNode();
      if (peer_out_node->GetType() == ATOMICADDRCLEAN) {
        ret = SetAtomicCleanAttr(peer_out_node, {mem_clean_start}, {mem_clean_size}, memory_type);
        if (ret != SUCCESS) {
          GELOGE(ret, "[Set][AtomicCleanAttr]fail for node:%s", peer_out_node->GetName().c_str());
          return ret;
        }
      }
    }
  }

  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignBufferPoolMemory() {
  auto is_buffer_pool_mem_enable = [] (const ComputeGraphPtr &graph) -> bool {
    for (NodePtr &node : graph->GetAllNodes()) {
      auto op_desc = node->GetOpDesc();
      if (op_desc == nullptr) {
        continue;
      }
      bool has_attrs = op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID) && op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE);
      if (has_attrs) {
        return true;
      }
    }
    return false;
  };
  auto root_graph = GraphUtils::FindRootGraph(compute_graph_);
  GE_CHECK_NOTNULL(root_graph);
  if (root_graph->GetGraphUnknownFlag()) {
    GELOGI("[Check][Enable]Unknown root graph does not support buffer pool memory, graph:%s.",
           compute_graph_->GetName().c_str());
    return SUCCESS;
  }
  if (!is_buffer_pool_mem_enable(compute_graph_)) {
    GELOGD("[Check][Enable]Buffer pool memory is not enable, graph:%s.", compute_graph_->GetName().c_str());
    return SUCCESS;
  }
  map<int64_t, size_t> mem_type_to_offset;
  for (const auto &pair : memory_offset_) {
    mem_type_to_offset[pair.first] = pair.second.mem_offset_;
  }
  BufferPoolMemAssigner buffer_pool_mem_assigner(compute_graph_, mem_type_to_offset);
  Status status = buffer_pool_mem_assigner.Assign();
  if (status != SUCCESS) {
    GELOGE(status, "[Assign][BufferPoolMem]Graph:%s.", compute_graph_->GetName().c_str());
    REPORT_INNER_ERROR("E19999", "Failed to assign buffer pool memory, graph:%s.", compute_graph_->GetName().c_str());
    return status;
  }
  int64_t mem_type = buffer_pool_mem_assigner.GetMemType();
  auto iter = memory_offset_.find(mem_type);
  if (iter == memory_offset_.end()) {
    GELOGE(FAILED, "[Check][MemType]Memory type is not supported, graph:%s, mem type:%ld.",
           compute_graph_->GetName().c_str(), mem_type);
    REPORT_INNER_ERROR("E19999", "Memory type is not supported, graph:%s, mem type:%ld.",
                       compute_graph_->GetName().c_str(), mem_type);
    return FAILED;
  }
  iter->second.mem_offset_ = buffer_pool_mem_assigner.GetMemOffset();
  GELOGI("[Assign][BufferPoolMem]Assign buffer pool memory successfully, graph:%s, mem type:%ld, mem offset:%zu.",
         compute_graph_->GetName().c_str(), mem_type, buffer_pool_mem_assigner.GetMemOffset());
  return SUCCESS;
}
}  // namespace ge
