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

#include "graph/build/memory/var_mem_assign_util.h"
#include <vector>
#include "common/types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/common/transop_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"

using std::string;
using std::vector;

namespace ge {
Status VarMemAssignUtil::AssignVarMemory(ge::ComputeGraphPtr &compute_graph) {
  return AssignMemory2VariableNode(compute_graph);
}

Status VarMemAssignUtil::AssignConstantOpMemory(ge::ComputeGraphPtr &compute_graph) {
  return AssignStaticMemory2Node(compute_graph);
}

Status VarMemAssignUtil::AssignMemory2VariableNode(ge::ComputeGraphPtr &compute_graph) {
  return AssignStaticMemory2Node(compute_graph);
}

Status VarMemAssignUtil::AssignStaticMemory2Node(ge::ComputeGraphPtr &compute_graph) {
  GE_IF_BOOL_EXEC(compute_graph == nullptr, return FAILED);
  for (const ge::NodePtr &n : compute_graph->GetAllNodes()) {
    GE_IF_BOOL_EXEC((n->GetType() != VARIABLE) && (n->GetType() != CONSTANTOP), continue);
    string ref_var_src_var_name;
    GE_CHECK_NOTNULL(n->GetOpDesc());
    GE_IF_BOOL_EXEC(ge::AttrUtils::GetStr(n->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name), continue);
    string node_name = n->GetName();
    GE_IF_BOOL_EXEC(n->GetOpDesc()->GetAllOutputsDesc().empty(),
                    GELOGE(FAILED, "node:%s has no OutputDesc.", n->GetName().c_str());
                    return FAILED);
    ge::ConstGeTensorDescPtr tensor_desc = n->GetOpDesc()->GetOutputDescPtr(0);
    GE_CHECK_NOTNULL(tensor_desc);
    rtMemType_t memory_type = RT_MEMORY_HBM;
    uint32_t mem_type = 0;
    if (AttrUtils::GetInt(n->GetOpDesc(), ATTR_OUTPUT_MEMORY_TYPE, mem_type) && (mem_type == 1)) {
      memory_type = RT_MEMORY_RDMA_HBM;
    }
    if (!VarManager::Instance(compute_graph->GetSessionID())->IsVarExist(node_name, *tensor_desc)) {
      GE_CHK_STATUS_RET(
          VarManager::Instance(compute_graph->GetSessionID())->AssignVarMem(node_name, *tensor_desc, memory_type));
      GE_IF_BOOL_EXEC(n->GetType() == VARIABLE,
                      GE_CHK_STATUS_RET(AssignData2Fp32Var(n, compute_graph->GetSessionID())));
      GE_CHK_STATUS_RET(VarManager::Instance(compute_graph->GetSessionID())
                            ->SetAllocatedGraphId(node_name, compute_graph->GetGraphID()));
    }

    uint8_t *dev_ptr = nullptr;
    GE_CHK_STATUS_RET(VarManager::Instance(compute_graph->GetSessionID())
                          ->GetVarAddr(node_name, *tensor_desc, &dev_ptr, memory_type));
    vector<int64_t> output_list = n->GetOpDesc()->GetOutputOffset();
    GE_IF_BOOL_EXEC(output_list.empty(), return FAILED);
    output_list[0] = static_cast<int64_t>(reinterpret_cast<intptr_t>(dev_ptr));
    n->GetOpDesc()->SetOutputOffset(output_list);
  }
  return SUCCESS;
}

Status VarMemAssignUtil::AssignData2Fp32Var(const ge::NodePtr &node, uint64_t session_id) {
  string src_var_name;
  GE_CHECK_NOTNULL(node->GetOpDesc());
  if (ge::AttrUtils::GetStr(node->GetOpDesc(), VAR_ATTR_SRC_VAR_NAME, src_var_name)) {
    ge::GeTensorDesc cur_tensor_desc;
    uint8_t *dev_ptr = nullptr;
    rtMemType_t memory_type = RT_MEMORY_HBM;
    GE_CHK_STATUS_RET(VarManager::Instance(session_id)->GetCurVarDesc(src_var_name, cur_tensor_desc));
    GE_CHK_STATUS_RET(
        VarManager::Instance(session_id)->GetVarAddr(src_var_name, cur_tensor_desc, &dev_ptr, memory_type));
    GE_CHK_STATUS_RET(
        VarManager::Instance(session_id)->SetVarAddr(node->GetName(), cur_tensor_desc, dev_ptr, memory_type));
  }
  return SUCCESS;
}

Status VarMemAssignUtil::AssignVarAttr2Nodes(ge::ComputeGraphPtr &compute_graph) {
  for (const ge::NodePtr &node : compute_graph->GetAllNodes()) {
    GE_IF_BOOL_EXEC(node->GetType() != VARIABLE, continue);
    string ref_var_src_var_name;
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GE_IF_BOOL_EXEC(ge::AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name), continue);
    GE_CHK_STATUS_RET(DealVariableNode(compute_graph->GetGraphID(), node, compute_graph->GetSessionID()));
  }
  return SUCCESS;
}

Status VarMemAssignUtil::SetOutVariableAttr(const ge::NodePtr &node, const ge::NodePtr &var_node, int index,
                                            uint64_t session_id) {
  vector<int64_t> output_list;
  uint8_t *dev_ptr = nullptr;
  GE_CHECK_NOTNULL(node->GetOpDesc());
  output_list = node->GetOpDesc()->GetOutputOffset();
  if (output_list.empty()) {
    GELOGE(PARAM_INVALID, "Output_list is empty");
    return PARAM_INVALID;
  }
  GE_CHECK_NOTNULL(var_node->GetOpDesc());
  GeTensorDesc var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0);
  rtMemType_t memory_type = RT_MEMORY_HBM;
  GE_CHK_STATUS_RET(
      VarManager::Instance(session_id)->GetVarAddr(var_node->GetName(), var_tensor_desc, &dev_ptr, memory_type));

  int out_list_size = static_cast<int>(output_list.size());
  GE_CHK_BOOL_RET_STATUS(index < out_list_size, FAILED, "index %d >= output_list.size() %d", index, out_list_size);

  output_list[index] = static_cast<int64_t>(reinterpret_cast<intptr_t>(dev_ptr));
  GELOGI("Assign node outputOffset[index] is: %ld", output_list[index]);
  node->GetOpDesc()->SetOutputOffset(output_list);

  return SUCCESS;
}

Status VarMemAssignUtil::DealExportVariableNode(const ge::NodePtr &node, const ge::NodePtr &var_node,
                                                uint64_t session_id) {
  ge::OutDataAnchorPtr var_out_anchor = node->GetOutDataAnchor(0);
  GE_IF_BOOL_EXEC(var_out_anchor == nullptr, return FAILED);
  for (const ge::InDataAnchorPtr &dst_in_var_anchor : var_out_anchor->GetPeerInDataAnchors()) {
    ge::NodePtr dst_node = dst_in_var_anchor->GetOwnerNode();
    if ((dst_node->GetType() == ASSIGN) || (dst_node->GetType() == ASSIGNADD) || (dst_node->GetType() == ASSIGNSUB)) {
      if (dst_in_var_anchor == dst_node->GetInDataAnchor(0)) {
        GE_CHK_STATUS_RET(DealExportVariableNode(dst_node, var_node, session_id));
      }
    }
  }
  GE_CHK_STATUS_RET(SetOutVariableAttr(node, var_node, 0, session_id));
  return SUCCESS;
}

Status VarMemAssignUtil::DealBroadCastNode(uint32_t graph_id, const ge::NodePtr &node,
                                           const ge::InDataAnchorPtr &in_data_anchor, const ge::NodePtr &var_node,
                                           uint64_t session_id) {
  VarBroadCastInfo broad_cast_info;
  broad_cast_info.idx = in_data_anchor->GetIdx();
  broad_cast_info.var_name = var_node->GetName();
  broad_cast_info.broadcast_name = node->GetName();

  auto op_desc = node->GetOpDesc();
  GE_CHK_BOOL_RET_STATUS(op_desc != nullptr, FAILED, "Get broadcast op %s desc is nullptr", node->GetName().c_str());

  GE_IF_BOOL_EXEC(broad_cast_info.idx < 0,
                  GELOGI("Broadcast input index must be positive, actual %d", broad_cast_info.idx);
                  return INTERNAL_ERROR);

  auto broad_cast_index = static_cast<size_t>(broad_cast_info.idx);
  auto input_tensor_desc_ptr_vistor = op_desc->GetAllInputsDescPtr();
  GE_CHK_BOOL_RET_STATUS(input_tensor_desc_ptr_vistor.size() > broad_cast_index, FAILED,
                         "Get broadcast op %s input tensor desc size [%zu] < idx [%d]", node->GetName().c_str(),
                         input_tensor_desc_ptr_vistor.size(), broad_cast_info.idx);
  const ge::GeTensorDescPtr input_tensor_desc =
      input_tensor_desc_ptr_vistor.at(static_cast<size_t>(broad_cast_info.idx));
  int64_t input_size = 0;
  GE_CHK_STATUS(TensorUtils::GetSize(*input_tensor_desc, input_size), "get input size failed.");
  broad_cast_info.input_size = input_size;

  vector<int64_t> output_list = op_desc->GetOutputOffset();
  GE_CHK_BOOL_RET_STATUS(output_list.size() > broad_cast_index, FAILED,
                         "Get broadcast op %s output_list size [%zu] < idx [%d]", node->GetName().c_str(),
                         output_list.size(), broad_cast_info.idx);
  broad_cast_info.input_offset = output_list[broad_cast_info.idx];
  broad_cast_info.output_offset = output_list[broad_cast_info.idx];

  op_desc->SetInputOffset(output_list);

  auto output_tensor_desc_ptr_vistor = op_desc->GetAllOutputsDescPtr();
  GE_CHK_BOOL_RET_STATUS(output_tensor_desc_ptr_vistor.size() > broad_cast_index, FAILED,
                         "Get broadcast op %s output tensor desc size [%zu] < idx [%d]", node->GetName().c_str(),
                         output_tensor_desc_ptr_vistor.size(), broad_cast_info.idx);
  const ge::GeTensorDescPtr output_tensor_desc =
      output_tensor_desc_ptr_vistor.at(static_cast<size_t>(broad_cast_info.idx));
  int64_t output_size = 0;
  GE_CHK_STATUS(TensorUtils::GetSize(*output_tensor_desc, output_size), "get input size failed.");
  broad_cast_info.output_size = output_size;
  GE_CHK_BOOL_RET_STATUS(broad_cast_info.output_size == broad_cast_info.input_size, FAILED,
                         "Broadcast op input size[%lu] is not equal output size[%lu]", broad_cast_info.input_size,
                         broad_cast_info.output_size);

  GE_CHK_STATUS_RET(VarManager::Instance(session_id)->SaveBroadCastInfo(graph_id, broad_cast_info));
  return SUCCESS;
}

Status VarMemAssignUtil::DealVariableNode(uint32_t graph_id, const ge::NodePtr &node, uint64_t session_id) {
  GE_CHK_STATUS_RET(SetOutVariableAttr(node, node, 0, session_id));

  for (const ge::OutDataAnchorPtr &var_out_data_anchor : node->GetAllOutDataAnchors()) {
    for (const ge::InDataAnchorPtr &dst_in_data_anchor : var_out_data_anchor->GetPeerInDataAnchors()) {
      ge::NodePtr dst_node = dst_in_data_anchor->GetOwnerNode();
      if (dst_node->GetType() == HCOMBROADCAST || dst_node->GetType() == HVDCALLBACKBROADCAST) {
        GE_CHK_STATUS_RET(DealBroadCastNode(graph_id, dst_node, dst_in_data_anchor, node, session_id));
        continue;
      }

      if ((dst_node->GetType() == ASSIGN) || (dst_node->GetType() == ASSIGNADD) || (dst_node->GetType() == ASSIGNSUB)) {
        if (dst_in_data_anchor == dst_node->GetInDataAnchor(0)) {
          GE_CHK_STATUS_RET(DealExportVariableNode(dst_node, node, session_id));
        }
      }
      auto dst_type = dst_node->GetType();
      bool is_trans_node =
          (dst_type == TRANSDATA) || (dst_type == CAST) || (dst_type == TRANSPOSE) || (dst_type == PERMUTE);
      if (is_trans_node) {
        NodePtr final_trans_node = GetFinalTransNode(dst_node);
        GE_CHK_STATUS_RET(DealTransNode(final_trans_node));
      }
    }
  }
  return SUCCESS;
}

ge::NodePtr VarMemAssignUtil::GetFinalTransNode(const ge::NodePtr &trans_node) {
  NodePtr final_ref_node = trans_node;
  OutDataAnchorPtr trans_out_data_anchor = trans_node->GetOutDataAnchor(0);
  GE_IF_BOOL_EXEC(trans_out_data_anchor == nullptr, return final_ref_node);
  for (const auto &dst_in_anchor : trans_out_data_anchor->GetPeerInDataAnchors()) {
    NodePtr dst_node = dst_in_anchor->GetOwnerNode();
    auto dst_type = dst_node->GetType();
    bool is_trans_node =
        (dst_type == TRANSDATA) || (dst_type == CAST) || (dst_type == TRANSPOSE) || (dst_type == PERMUTE);
    if (is_trans_node && (dst_in_anchor->GetIdx() == 0)) {
      final_ref_node = GetFinalTransNode(dst_node);
    }
  }
  GELOGI("Final writable node is %s", final_ref_node->GetName().c_str());
  return final_ref_node;
}

Status VarMemAssignUtil::DealTransNode(const ge::NodePtr &final_trans_node) {
  ge::OutDataAnchorPtr final_trans_out_anchor = final_trans_node->GetOutDataAnchor(0);
  GE_IF_BOOL_EXEC(final_trans_out_anchor == nullptr, return SUCCESS);
  for (const ge::InDataAnchorPtr &dst_in_var_anchor : final_trans_out_anchor->GetPeerInDataAnchors()) {
    ge::NodePtr dst_node = dst_in_var_anchor->GetOwnerNode();
    if ((dst_node->GetType() == ASSIGN) || (dst_node->GetType() == ASSIGNADD) || (dst_node->GetType() == ASSIGNSUB)) {
      GE_CHK_STATUS_RET(DealExportTransNode(dst_node, final_trans_node));
    }
  }
  return SUCCESS;
}

Status VarMemAssignUtil::DealExportTransNode(const ge::NodePtr &node, const ge::NodePtr &final_trans_node) {
  ge::OutDataAnchorPtr node_out_anchor = node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(node_out_anchor);
  for (const ge::InDataAnchorPtr &dst_in_var_anchor : node_out_anchor->GetPeerInDataAnchors()) {
    ge::NodePtr dst_node = dst_in_var_anchor->GetOwnerNode();
    if ((dst_node->GetType() == ASSIGN) || (dst_node->GetType() == ASSIGNADD) || (dst_node->GetType() == ASSIGNSUB)) {
      GE_CHK_STATUS_RET(DealExportTransNode(dst_node, final_trans_node));
    }
  }
  GE_CHK_STATUS_RET(SetOutTransNodeToAssign(node, final_trans_node, 0));
  return SUCCESS;
}

Status VarMemAssignUtil::SetOutTransNodeToAssign(const ge::NodePtr &node, const ge::NodePtr &final_trans_node,
                                                 size_t index) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  GE_CHECK_NOTNULL(final_trans_node->GetOpDesc());
  // get final_trans_node outputOffset
  vector<int64_t> final_trans_output_list = final_trans_node->GetOpDesc()->GetOutputOffset();
  GE_CHECK_SIZE(final_trans_output_list.size());

  // get assign_node outputOffset
  vector<int64_t> output_list = node->GetOpDesc()->GetOutputOffset();
  auto out_list_size = output_list.size();
  GE_CHECK_SIZE(out_list_size);
  GE_CHK_BOOL_RET_STATUS(index < out_list_size, FAILED, "index %zu >= output_list.size() %zu", index, out_list_size);

  // final_trans_node outputOffset[0] to assign_node outputOffset[0]
  GELOGI("final_trans_node outputOffset[0] is: %ld", final_trans_output_list[0]);

  output_list[index] = final_trans_output_list[0];
  GELOGI("Assign node outputOffset[0] is: %ld", output_list[index]);
  node->GetOpDesc()->SetOutputOffset(output_list);

  return SUCCESS;
}

Status VarMemAssignUtil::AssignMemory2HasRefAttrNode(ge::ComputeGraphPtr &compute_graph) {
  for (const ge::NodePtr &n : compute_graph->GetAllNodes()) {
    string ref_var_src_var_name;
    auto op_desc = n->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (uint32_t idx = 0; idx < op_desc->GetOutputsSize(); idx += 1) {
      const auto out_desc = op_desc->MutableOutputDesc(idx);
      if (ge::AttrUtils::GetStr(out_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name)) {
        GE_CHK_STATUS_RET(AssignData2VarRef(n, ref_var_src_var_name, compute_graph->GetSessionID(), idx));
      }
    }
  }
  return SUCCESS;
}

Status VarMemAssignUtil::AssignData2VarRef(const ge::NodePtr &has_ref_attr_node, const string &src_var_name,
                                           uint64_t session_id, uint32_t out_index) {
  // Get ref_var_src_var address
  auto root_graph = GraphUtils::FindRootGraph(has_ref_attr_node->GetOwnerComputeGraph());
  GE_CHECK_NOTNULL(root_graph);
  ge::NodePtr var_ref_src_var = root_graph->FindNode(src_var_name);
  if (var_ref_src_var == nullptr) {
    for (auto sub_graph : root_graph->GetAllSubgraphs()) {
        auto node_ptr = sub_graph->FindNode(src_var_name);
        if (node_ptr != nullptr) {
          var_ref_src_var = node_ptr;
          break;
        }
    }
  }
  GE_IF_BOOL_EXEC(var_ref_src_var == nullptr || var_ref_src_var->GetOpDesc() == nullptr, return FAILED);
  GeTensorDesc src_tensor_desc = var_ref_src_var->GetOpDesc()->GetOutputDesc(0);
  uint8_t *dev_ptr = nullptr;
  GE_CHK_STATUS_RET(VarManager::Instance(session_id)->GetVarAddr(src_var_name, src_tensor_desc, &dev_ptr));
  GE_CHECK_NOTNULL(has_ref_attr_node->GetOpDesc());
  vector<int64_t> ref_attr_node_output_list = has_ref_attr_node->GetOpDesc()->GetOutputOffset();
  GE_CHECK_SIZE(ref_attr_node_output_list.size());

  GE_CHK_BOOL_RET_STATUS(out_index < ref_attr_node_output_list.size(), FAILED,
                         "out_index %u >= ref_attr_node_output_list.size() %zu", out_index,
                         ref_attr_node_output_list.size());

  ref_attr_node_output_list[out_index] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(dev_ptr));
  has_ref_attr_node->GetOpDesc()->SetOutputOffset(ref_attr_node_output_list);
  GELOGI("Refresh address successfully, ref node: [%s], addr: [%ld]", has_ref_attr_node->GetName().c_str(),
         ref_attr_node_output_list[out_index]);
  return SUCCESS;
}
}  // namespace ge
