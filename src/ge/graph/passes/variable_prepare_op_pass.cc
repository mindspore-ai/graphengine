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

#include "graph/passes/variable_prepare_op_pass.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include "common/ge/ge_util.h"
#include "external/graph/graph.h"
#include "framework/common/debug/ge_log.h"
#include "graph/common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
std::map<std::string, std::map<int, int>> VariablePrepareOpPass::ref_node_without_prototype_map_{
  {REFSWITCH, {{0, 0}, {0, 1}}}};

Status VariablePrepareOpPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetDirectNode()) {
    auto iter = ref_input_output_map_.find(node->GetType());
    if (iter == ref_input_output_map_.end()) {
      GenerateRefTypeAndInputOutputMap(node);
    }
  }

  if (ref_input_output_map_.empty()) {
    GELOGI("No need to add variable_ref.");
    return SUCCESS;
  }

  for (auto &node : graph->GetDirectNode()) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    if (node->GetOpDesc()->GetType() == VARIABLE) {
      Status ret = DealVariableNode(node);
      if (ret != SUCCESS) {
        GELOGE(ret, "variable add back edge failed");
        return FAILED;
      }
    }
  }

  for (auto iter = ref_input_output_map_.begin(); iter != ref_input_output_map_.end(); ++iter) {
    GELOGI("ref type:[ %s ]", iter->first.c_str());
    auto index_map = iter->second;
    for (auto index_iter = index_map.begin(); index_iter != index_map.end(); ++index_iter) {
      GELOGI("{ %d : %d }", index_iter->first, index_iter->second);
    }
  }
  return SUCCESS;
}

Status VariablePrepareOpPass::DealVariableNode(NodePtr &var_node) {
  GE_CHECK_NOTNULL(var_node);
  for (auto &dst_node_and_inanchor : var_node->GetOutDataNodesAndAnchors()) {
    NodePtr dst_node = dst_node_and_inanchor.first;
    GE_CHECK_NOTNULL(dst_node);
    InDataAnchorPtr dst_in_data_anchor = dst_node_and_inanchor.second;
    GE_CHECK_NOTNULL(dst_in_data_anchor);
    auto input_index = dst_in_data_anchor->GetIdx();
    int out_index = GetWritableNodeOutIndex(dst_node, input_index);
    if (out_index >= 0) {
      Status ret = DealWritableNode(dst_node, input_index, var_node);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "Deal writable node[%s] failed, input index: %d, var: %s.", dst_node->GetName().c_str(),
               input_index, var_node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status VariablePrepareOpPass::DealWritableNode(const ge::NodePtr &writable_node, int input_index,
                                               const ge::NodePtr &var_node) {
  // Find the last ref node:
  // If the ref input has corresponding output, add variable ref after it.
  // If the ref input has no corresponding output, insert RefIdentity and variable ref before it.
  // If ref node with control output was found while finding the last ref node, add variable ref after it.
  std::stack<pair<NodePtr, int>> nodes_to_check;
  nodes_to_check.push({writable_node, input_index});
  while (!nodes_to_check.empty()) {
    auto node_index = nodes_to_check.top();
    nodes_to_check.pop();
    auto cur_node = node_index.first;
    int cur_input_index = node_index.second;
    // Collect ref node after cur node
    const auto nodes_size = nodes_to_check.size();
    // Add peer ref output node of current node to stack
    CHECK_FALSE_EXEC(GetPeerNodeOfRefInput(cur_node, cur_input_index, nodes_to_check) == SUCCESS,
                     GELOGE(FAILED, "GetPeerNodeOfRefInput for node[%s] failed.", cur_node->GetName().c_str());
                     return FAILED);
    auto output_index = GetWritableNodeOutIndex(cur_node, cur_input_index);
    CHECK_FALSE_EXEC(output_index >= 0,
                     GELOGE(FAILED, "Get writable node[%s] ref input[%d]'s corresponding out index failed: %d.",
                            cur_node->GetName().c_str(), cur_input_index, output_index);
                     return FAILED);
    if (nodes_size == nodes_to_check.size()) {
      const auto &op_desc = cur_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      // No need to add variable_ref for frameworkop
      if (op_desc->GetType() == FRAMEWORKOP) {
        GELOGD("No need to add variable_ref for frameworkop");
        continue;
      }
      if (static_cast<uint32_t>(output_index) < op_desc->GetOutputsSize()) {
        // Add variable ref node after ref output for final ref node
        CHECK_FALSE_EXEC(AddVariableRef(cur_node, var_node, output_index) == SUCCESS,
                         GELOGE(FAILED, "Add variable ref failed");
                         return FAILED);
      } else {
        // Insert variable ref node before ref input without corresponding ref output
        CHECK_FALSE_EXEC(InsertVariableRef(cur_node, cur_input_index, var_node) == SUCCESS,
                         GELOGE(FAILED, "Insert variable ref and ref identity failed");
                         return FAILED);
      }
      continue;
    }
    if (HasControlOut(cur_node)) {
      // Add variable ref node after ref output for ref node has control output.
      CHECK_FALSE_EXEC(AddVariableRef(cur_node, var_node, output_index) == SUCCESS,
                       GELOGE(FAILED, "Add variable ref failed");
                       return FAILED);
    }
  }
  return SUCCESS;
}

Status VariablePrepareOpPass::GetPeerNodeOfRefInput(const ge::NodePtr &node, int input_index,
                                                    std::stack<pair<NodePtr, int>> &nodes) {
  auto output_index = GetWritableNodeOutIndex(node, input_index);
  if (output_index == -1) {
    GELOGE(PARAM_INVALID, "Node[%s] is not a ref node.", node->GetName().c_str());
    return PARAM_INVALID;
  }
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (static_cast<uint32_t>(output_index) == op_desc->GetOutputsSize()) {
    return SUCCESS;
  }
  if (output_index >= static_cast<int>(node->GetAllOutDataAnchorsSize())) {
    GELOGW("Can not get %d th output anchor of %s", output_index, node->GetName().c_str());
    return SUCCESS;
  }
  const auto &out_anchor = node->GetOutDataAnchor(output_index);
  GE_CHECK_NOTNULL(out_anchor);
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    auto peer_node = peer_in_anchor->GetOwnerNode();
    if (peer_node == nullptr) {
      continue;
    }
    const int peer_in_index = peer_in_anchor->GetIdx();
    if (GetWritableNodeOutIndex(peer_node, peer_in_index) != -1) {
      nodes.push({peer_node, peer_in_index});
    }
  }
  return SUCCESS;
}

Status VariablePrepareOpPass::AddVariableRef(ge::NodePtr &final_writable_node, const ge::NodePtr &var_node, int index) {
  GE_CHECK_NOTNULL(final_writable_node);
  GE_CHECK_NOTNULL(var_node);
  if (index >= static_cast<int>(final_writable_node->GetAllOutDataAnchorsSize())) {
    GELOGW("Can not get %d th output anchor of %s", index, final_writable_node->GetName().c_str());
    return SUCCESS;
  }
  // Check for duplicate creation
  OutDataAnchorPtr out_anchor = final_writable_node->GetOutDataAnchor(index);
  GE_CHECK_NOTNULL(out_anchor);
  for (const auto &peer_anchor : out_anchor->GetPeerAnchors()) {
    NodePtr peer_node = peer_anchor->GetOwnerNode();
    OpDescPtr peer_opdesc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(peer_opdesc);
    string src_var_name;
    (void)ge::AttrUtils::GetStr(peer_opdesc, REF_VAR_SRC_VAR_NAME, src_var_name);
    if (peer_node->GetType() == VARIABLE && var_node->GetName() == src_var_name) {
      GELOGI("The corresponding variable_ref has been added to this connection.");
      return SUCCESS;
    }
  }
  // creat variable_ref
  std::stringstream variable_ref_name;
  variable_ref_name << "_TO_" << final_writable_node->GetName() << "_REF_" << index;
  NodePtr variable_ref_node = CreateVariableRef(var_node->GetName() + variable_ref_name.str(), var_node);
  GE_CHECK_NOTNULL(variable_ref_node);
  Status ret_check = CheckStreamLabel(variable_ref_node, final_writable_node);
  if (ret_check != SUCCESS) {
    GELOGE(FAILED, "check stream lable failed");
    return FAILED;
  }

  GELOGI("Add variable_ref between [%s] and [%s]", var_node->GetName().c_str(), variable_ref_node->GetName().c_str());
  // add control anchor between variable_ref and final peer node
  // variable_ref_node need to execute before other nodes
  CHECK_FALSE_EXEC(AddControlEdge(final_writable_node, variable_ref_node) == SUCCESS,
                   GELOGE(FAILED, "Add control edges between variable ref node and output nodes of ref node failed");
                   return FAILED);

  graphStatus ret = ge::GraphUtils::AddEdge(out_anchor, variable_ref_node->GetInDataAnchor(0));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "add data anchor between  variable_ref and final_writable peer node failed");
    return FAILED;
  }
  return SUCCESS;
}

Status VariablePrepareOpPass::InsertVariableRef(ge::NodePtr &node, int in_index, const ge::NodePtr &var_node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(var_node);
  // Check connection between two nodes
  const auto in_anchor = node->GetInDataAnchor(in_index);
  GE_CHECK_NOTNULL(in_anchor);
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);
  auto peer_in_node = peer_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(peer_in_node);

  // Create ref_identity
  std::stringstream ref_identity_name;
  ref_identity_name << "RefIdentity_" << peer_in_node->GetName() << "_" << peer_out_anchor->GetIdx() << "_TO_"
                    << node->GetName() << "_" << in_index;
  NodePtr ref_identity_node = CreateRefIdentity(ref_identity_name.str(), node, static_cast<uint32_t>(in_index));
  GE_CHECK_NOTNULL(ref_identity_node);

  // Create variable_ref
  std::stringstream variable_ref_name;
  variable_ref_name << "_TO_" << node->GetName() << "_REF_" << in_index;
  NodePtr variable_ref_node = CreateVariableRef(var_node->GetName() + variable_ref_name.str(), var_node);
  GE_CHECK_NOTNULL(variable_ref_node);
  Status ret_check = CheckStreamLabel(variable_ref_node, node);
  if (ret_check != SUCCESS) {
    GELOGE(FAILED, "check stream lable failed");
    return FAILED;
  }

  GELOGI("Insert variable_ref of [%s] between [%s] and [%s]", var_node->GetName().c_str(),
         peer_in_node->GetName().c_str(), node->GetName().c_str());
  // add control anchor between variable_ref and node
  // variable_ref_node need to execute before other nodes
  CHECK_FALSE_EXEC(AddControlEdge(node, variable_ref_node) == SUCCESS,
                   GELOGE(FAILED, "Add control edges between variable ref node and output nodes of ref node failed");
                   return FAILED);

  // Insert variable ref node between two nodes and remove the original edge.
  CHECK_FALSE_EXEC(ge::GraphUtils::RemoveEdge(peer_out_anchor, in_anchor) == SUCCESS,
                   GELOGE(FAILED, "Remove edge between ref node and its peer node failed");
                   return FAILED);
  CHECK_FALSE_EXEC(ge::GraphUtils::AddEdge(peer_out_anchor, ref_identity_node->GetInDataAnchor(0)) == SUCCESS,
                   GELOGE(FAILED, "Add data edge between pre node and ref_identity failed");
                   return FAILED);
  CHECK_FALSE_EXEC(ge::GraphUtils::AddEdge(ref_identity_node->GetOutDataAnchor(0), in_anchor) == SUCCESS,
                   GELOGE(FAILED, "Add data edge between ref_identity and ref node failed");
                   return FAILED);

  // Add edge from ref identity node to variable ref node.
  CHECK_FALSE_EXEC(
    ge::GraphUtils::AddEdge(ref_identity_node->GetOutDataAnchor(0), variable_ref_node->GetInDataAnchor(0)) == SUCCESS,
    GELOGE(FAILED, "Add data edge between ref_identity and variable_ref failed");
    return FAILED);
  CHECK_FALSE_EXEC(
    ge::GraphUtils::AddEdge(node->GetOutControlAnchor(), variable_ref_node->GetInControlAnchor()) == SUCCESS,
    GELOGE(FAILED, "Add control edge between ref_identity and variable_ref failed");
    return FAILED);
  return SUCCESS;
}

Status VariablePrepareOpPass::AddControlEdge(const ge::NodePtr &node, const ge::NodePtr &variable_ref_node) {
  auto out_anchors = node->GetAllOutAnchors();
  for (auto &out_anchor : out_anchors) {
    GE_CHECK_NOTNULL(out_anchor);
    for (auto &peer_in_anchor : out_anchor->GetPeerAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      NodePtr peer_node = peer_in_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);
      CHECK_FALSE_EXEC(
        ge::GraphUtils::AddEdge(variable_ref_node->GetOutControlAnchor(), peer_node->GetInControlAnchor()) == SUCCESS,
        GELOGE(FAILED, "Add control edge between variable_ref and ref node's peer node failed");
        return FAILED);
    }
  }
  return SUCCESS;
}

ge::NodePtr VariablePrepareOpPass::CreateRefIdentity(const std::string &ref_identity_name, const ge::NodePtr &node,
                                                     uint32_t input_index) {
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "opdesc is nullptr");
    return nullptr;
  }

  OpDescPtr ref_identity_op_desc = MakeShared<OpDesc>(ref_identity_name.c_str(), REFIDENTITY);
  if (ref_identity_op_desc == nullptr) {
    GELOGE(FAILED, "ref_identity op desc is nullptr");
    return nullptr;
  }

  GE_IF_BOOL_EXEC(ref_identity_op_desc->AddOutputDesc(op_desc->GetInputDesc(input_index)) != SUCCESS,
                  GELOGW("add output desc edge failed");
                  return nullptr);
  GE_IF_BOOL_EXEC(ref_identity_op_desc->AddInputDesc(op_desc->GetInputDesc(input_index)) != SUCCESS,
                  GELOGW("add input desc edge failed");
                  return nullptr);
  NodePtr ref_identity_node = node->GetOwnerComputeGraph()->AddNode(ref_identity_op_desc);
  GE_IF_BOOL_EXEC(ref_identity_node == nullptr, GELOGW("ref_identity_node is null"); return nullptr);
  return ref_identity_node;
}

ge::NodePtr VariablePrepareOpPass::CreateVariableRef(const std::string &variable_ref_name,
                                                     const ge::NodePtr &var_node) {
  OpDescPtr var_op_desc = var_node->GetOpDesc();
  if (var_op_desc == nullptr) {
    GELOGE(FAILED, "get var opdesc is nullptr");
    return nullptr;
  }

  OpDescPtr var_ref_op_desc = MakeShared<OpDesc>(variable_ref_name.c_str(), var_op_desc->GetType());
  if (var_ref_op_desc == nullptr) {
    GELOGE(FAILED, "var_ref opdesc is nullptr");
    return nullptr;
  }

  GE_IF_BOOL_EXEC(var_ref_op_desc->AddOutputDesc(var_op_desc->GetOutputDesc(0)) != SUCCESS,
                  GELOGW("add output desc edge failed");
                  return nullptr);
  GE_IF_BOOL_EXEC(var_ref_op_desc->AddInputDesc(var_op_desc->GetOutputDesc(0)) != SUCCESS,
                  GELOGW("add input desc edge failed");
                  return nullptr);
  NodePtr variable_ref_node = var_node->GetOwnerComputeGraph()->AddNode(var_ref_op_desc);
  GE_IF_BOOL_EXEC(variable_ref_node == nullptr, GELOGW("variable_ref_node is null"); return nullptr);

  bool is_set_str = ge::AttrUtils::SetStr(var_ref_op_desc, REF_VAR_SRC_VAR_NAME, var_op_desc->GetName());
  if (is_set_str) {
    GELOGD("Set node [%s] REF_VAR_SRC_VAR_NAME [%s]", variable_ref_node->GetName().c_str(),
           var_op_desc->GetName().c_str());
  }
  return variable_ref_node;
}

int VariablePrepareOpPass::GetWritableNodeOutIndex(const NodePtr &node, int input_index) {
  if (node == nullptr) {
    return -1;
  }
  GELOGD("get writable node and input index %s:%d", node->GetName().c_str(), input_index);
  auto node_type = node->GetType();
  if (node_type == FRAMEWORKOP) {
    std::string original_type;
    GE_IF_BOOL_EXEC(GetOriginalType(node, original_type) != SUCCESS, GELOGW("Get node original type fail"));
    GELOGD("find frameworkop: [%s], original type is %s", node->GetName().c_str(), original_type.c_str());
    return FindRefOutIndex(original_type, input_index, ref_node_without_prototype_map_);
  }
  return FindRefOutIndex(node_type, input_index, ref_input_output_map_);
}

void VariablePrepareOpPass::GenerateRefTypeAndInputOutputMap(const NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGW("op_desc in null, please check node:[%s]", node->GetName().c_str());
    return;
  }
  for (const auto &name_index : op_desc->GetAllInputName()) {
    // Record the index of output with the same name as input, thinking of them as a pair of ref input and output.
    const int out_index = op_desc->GetOutputIndexByName(name_index.first);
    if (out_index != -1) {
      ref_input_output_map_[node->GetType()][name_index.second] = out_index;
      continue;
    }
    // Record the ref input without corresponding output.
    const auto &input_desc = op_desc->GetInputDesc(name_index.second);
    if (!input_desc.GetRefPortIndex().empty()) {
      ref_input_output_map_[node->GetType()][name_index.second] = static_cast<int>(op_desc->GetOutputsSize());
    }
  }
}

int VariablePrepareOpPass::FindRefOutIndex(const std::string &node_type, int input_index,
                                           const std::map<std::string, std::map<int, int>> &ref_map) {
  auto node_iter = ref_map.find(node_type);
  if (node_iter == ref_map.end()) {
    return -1;
  }

  auto index_iter = node_iter->second.find(input_index);
  if (index_iter == node_iter->second.end()) {
    return -1;
  }
  return index_iter->second;
}

Status VariablePrepareOpPass::CheckStreamLabel(const ge::NodePtr &var_ref_node,
                                               const ge::NodePtr &final_writable_node) {
  // Solve the problem that the writable node is not in the same stream as the subsequent node.
  // Causes the stream to not trigger properly.
  // The label of node should be handled uniformly.
  OpDescPtr writable_desc = final_writable_node->GetOpDesc();
  GE_CHECK_NOTNULL(writable_desc);
  std::string stream_label;
  (void)AttrUtils::GetStr(writable_desc, ATTR_NAME_STREAM_LABEL, stream_label);
  if (!stream_label.empty()) {
    GE_CHK_STATUS_RET(SetStreamLabel(var_ref_node, stream_label), "set stream label failed");
  }
  return SUCCESS;
}

bool VariablePrepareOpPass::HasControlOut(const ge::NodePtr &node) {
  const auto &out_control_anchor = node->GetOutControlAnchor();
  for (const auto &peer_in_control_anchor : out_control_anchor->GetPeerInControlAnchors()) {
    if (peer_in_control_anchor == nullptr || peer_in_control_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    return true;
  }
  return false;
}
}  // namespace ge
