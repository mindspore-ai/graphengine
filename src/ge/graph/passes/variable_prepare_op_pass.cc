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
      GELOGI("{ %d:%d }", index_iter->first, index_iter->second);
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
    int out_index = GetWritableNodeOutIndex(dst_node, dst_in_data_anchor->GetIdx());
    if (out_index >= 0) {
      Status ret = DealWritableNode(dst_node, var_node, out_index);
      if (ret != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status VariablePrepareOpPass::DealWritableNode(ge::NodePtr &writable_node, ge::NodePtr &var_node, int out_index) {
  GE_CHECK_NOTNULL(writable_node);
  GE_CHECK_NOTNULL(var_node);
  NodePtr final_writable_node = writable_node;
  bool is_have_peer_node = false;
  for (auto &dst_node_and_inanchor : writable_node->GetOutDataNodesAndAnchors()) {
    NodePtr dst_node = dst_node_and_inanchor.first;
    GE_CHECK_NOTNULL(dst_node);
    InDataAnchorPtr dst_in_data_anchor = dst_node_and_inanchor.second;
    GE_CHECK_NOTNULL(dst_in_data_anchor);
    is_have_peer_node = true;
    int current_out_index = GetWritableNodeOutIndex(dst_node, dst_in_data_anchor->GetIdx());
    if (current_out_index >= 0) {
      final_writable_node = GetFinalWritableNode(dst_node, current_out_index);
      out_index = current_out_index;
    }

    GE_CHECK_NOTNULL(final_writable_node);
    Status ret = AddVariableRef(final_writable_node, var_node, out_index);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "add variable ref failed");
      return FAILED;
    }
  }
  if (final_writable_node->GetName() == writable_node->GetName() && !is_have_peer_node) {
    Status ret = AddVariableRef(final_writable_node, var_node, out_index);
    if (ret != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

NodePtr VariablePrepareOpPass::GetFinalWritableNode(ge::NodePtr &writable_node, int &out_index) {
  NodePtr current_node = writable_node;
  std::unordered_set<Node *> seen_node;
  while (true) {
    if (seen_node.count(current_node.get())) {
      GELOGE(FAILED, "There is a ring structure in the graph");
      return nullptr;
    }
    seen_node.insert(current_node.get());
    OutDataAnchorPtr out_anchor = current_node->GetOutDataAnchor(out_index);
    if (out_anchor == nullptr) {
      GELOGE(FAILED, "Failed to get data anchor by index %d", out_index);
      return nullptr;
    }
    bool found_writeable_node = false;
    auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
    for (auto &peer_in_anchor : peer_in_anchors) {
      if (peer_in_anchor == nullptr) {
        GELOGE(FAILED, "peer in data anchor is nullptr, node %s:%s", current_node->GetType().c_str(),
               current_node->GetName().c_str());
        continue;
      }

      NodePtr peer_node = peer_in_anchor->GetOwnerNode();
      int current_out_index = GetWritableNodeOutIndex(peer_node, peer_in_anchor->GetIdx());
      if (current_out_index >= 0) {
        current_node = peer_node;
        out_index = current_out_index;
        found_writeable_node = true;
        break;
      }
    }
    if (!found_writeable_node) {
      GELOGD("final writable node is %s", current_node->GetName().c_str());
      return current_node;
    }
  }
}

Status VariablePrepareOpPass::AddVariableRef(ge::NodePtr &final_writable_node, ge::NodePtr &var_node, int index) {
  GE_CHECK_NOTNULL(final_writable_node);
  GE_CHECK_NOTNULL(var_node);

  if (final_writable_node->GetType() == FRAMEWORKOP) {
    GELOGD("No need to add variable_ref for frameworkop");
    return SUCCESS;
  }
  std::stringstream variable_ref_name;
  variable_ref_name << "_TO_" << final_writable_node->GetName() << "_REF_" << index;
  ge::NodePtr find_node = var_node->GetOwnerComputeGraph()->FindNode(var_node->GetName() + variable_ref_name.str());
  if (find_node != nullptr) {
    GELOGD("The corresponding variable_ref [%s] has been added to this connection.", find_node->GetName().c_str());
    return SUCCESS;
  }
  NodePtr variable_ref_node = CreatVariableRef(var_node->GetName() + variable_ref_name.str(), var_node);

  GELOGI("Add variable_ref between [%s] and [%s]", var_node->GetName().c_str(), variable_ref_node->GetName().c_str());
  GE_CHECK_NOTNULL(variable_ref_node);
  // add  control anchor between  variable_ref and final peer node
  // variable_ref_node need to execute before other nodes
  auto final_writable_outAnchors = final_writable_node->GetAllOutAnchors();
  for (auto &final_writable_outAnchor : final_writable_outAnchors) {
    GE_CHECK_NOTNULL(final_writable_outAnchor);
    for (auto &final_writable_peerAnchor : final_writable_outAnchor->GetPeerAnchors()) {
      GE_CHECK_NOTNULL(final_writable_peerAnchor);
      NodePtr peer_node = final_writable_peerAnchor->GetOwnerNode();
      graphStatus ret =
        ge::GraphUtils::AddEdge(variable_ref_node->GetOutControlAnchor(), peer_node->GetInControlAnchor());
      if (ret != GRAPH_SUCCESS) {
        GELOGE(FAILED, "add control anchor between  variable_ref and final_writable peer node failed");
        return FAILED;
      }
    }
  }
  graphStatus ret =
    ge::GraphUtils::AddEdge(final_writable_node->GetOutDataAnchor(index), variable_ref_node->GetInDataAnchor(0));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "add data anchor between  variable_ref and final_writable peer node failed");
    return FAILED;
  }
  return SUCCESS;
}

ge::NodePtr VariablePrepareOpPass::CreatVariableRef(const std::string &variable_ref_name, ge::NodePtr &var_node) {
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
  if (node_type == ASSIGN) {
    if (UpdateAssignOpDesc(node) != SUCCESS) {
      return -1;
    }
  }

  if (node_type == FRAMEWORKOP) {
    std::string original_type;
    GE_IF_BOOL_EXEC(GetOriginalType(node, original_type) != SUCCESS, GELOGW("Get node original type fail"));
    GELOGI("find frameworkop: [%s], original type is %s", node->GetName().c_str(), original_type.c_str());
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
  for (const auto &out_ancohor : node->GetAllOutDataAnchors()) {
    int output_index = out_ancohor->GetIdx();
    string output_name = op_desc->GetOutputNameByIndex(output_index);
    GELOGD("output name:[%s]", output_name.c_str());

    int input_index = op_desc->GetInputIndexByName(output_name);
    if (input_index == -1) {
      continue;
    }
    auto ref_type_and_input_output_iter = ref_input_output_map_.find(node->GetType());
    if (ref_type_and_input_output_iter != ref_input_output_map_.end()) {
      auto input_output_index_map = ref_type_and_input_output_iter->second;
      if (input_output_index_map.find(input_index) == input_output_index_map.end()) {
        input_output_index_map.emplace(input_index, output_index);
        GELOGD("Add RefInputOutputMap %s:{ %d, %d }", node->GetType().c_str(), input_index, output_index);
      }
    } else {
      ref_input_output_map_.insert({node->GetType(), {{input_index, output_index}}});
      GELOGD("Create RefInputOutputMap { %s:{ %d, %d } }", node->GetType().c_str(), input_index, output_index);
    }
  }
}

Status VariablePrepareOpPass::UpdateAssignOpDesc(const ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  ge::InDataAnchorPtr var_anchor = node->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(var_anchor);
  GE_CHECK_NOTNULL(var_anchor->GetPeerOutAnchor());
  ge::NodePtr var_node = var_anchor->GetPeerOutAnchor()->GetOwnerNode();
  ge::OpDescPtr var_op_desc = var_node->GetOpDesc();
  GE_CHECK_NOTNULL(var_op_desc);
  ge::GeTensorDesc var_tensor_desc = var_op_desc->GetOutputDesc(0);

  ge::OpDescPtr assign_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(assign_op_desc);
  Status update_input_desc_ret = assign_op_desc->UpdateInputDesc(0, var_tensor_desc);
  Status update_output_desc_ret = assign_op_desc->UpdateOutputDesc(0, var_tensor_desc);
  if (update_input_desc_ret != GRAPH_SUCCESS || update_output_desc_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "update input or output desc success");
    return FAILED;
  }
  return SUCCESS;
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
}  // namespace ge
