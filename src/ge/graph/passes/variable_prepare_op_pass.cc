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

#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"
#include "external/graph/graph.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
Status VariablePrepareOpPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &node : graph->GetDirectNode()) {
    GELOGD("before VariablePrepareOpPass, graph has node: %s, and node name: %s", node->GetType().c_str(),
           node->GetName().c_str());
  }

  for (const auto &node : graph->GetDirectNode()) {
    GenerateRefTypeAndInputOutputMap(node);
  }

  if (ref_input_output_map_.empty()) {
    GELOGI("No need to add variable_ref.");
    return SUCCESS;
  }

  for (auto &node : graph->GetDirectNode()) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    GE_IF_BOOL_EXEC(node->GetOpDesc()->GetType() != VARIABLE, continue);
    Status ret = DealVariableNode(node);
    if (ret != SUCCESS) {
      GELOGE(ret, "variable add back edge failed");
      return FAILED;
    }
  }

  for (auto &node : graph->GetDirectNode()) {
    GELOGD("after VariablePrepareOpPass, graph has node: %s, and node name: %s", node->GetType().c_str(),
           node->GetName().c_str());
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
      GELOGI("final writable node is %s", current_node->GetName().c_str());
      return current_node;
    }
  }
}

Status VariablePrepareOpPass::AddVariableRef(ge::NodePtr &final_writable_node, ge::NodePtr &var_node, int index) {
  GE_CHECK_NOTNULL(final_writable_node);
  GE_CHECK_NOTNULL(var_node);

  NodePtr var_ref_node = CreatVariableRef(final_writable_node, var_node);
  GE_CHECK_NOTNULL(var_ref_node);
  // add  control anchor between var_ref_node and final peer node
  // var_ref_node need to execute before other nodes
  auto final_writable_outAnchors = final_writable_node->GetAllOutAnchors();
  for (auto &final_writable_outAnchor : final_writable_outAnchors) {
    GE_CHECK_NOTNULL(final_writable_outAnchor);
    for (auto &final_writable_peerAnchor : final_writable_outAnchor->GetPeerAnchors()) {
      GE_CHECK_NOTNULL(final_writable_peerAnchor);
      NodePtr peer_node = final_writable_peerAnchor->GetOwnerNode();
      graphStatus ret = ge::GraphUtils::AddEdge(var_ref_node->GetOutControlAnchor(), peer_node->GetInControlAnchor());
      if (ret != GRAPH_SUCCESS) {
        GELOGE(FAILED, "add  control anchor between var_ref_node and final_writable peer_node failed");
        return FAILED;
      }
    }
  }
  // add edge final node:index ---> var_ref_node:0
  graphStatus ret =
    ge::GraphUtils::AddEdge(final_writable_node->GetOutDataAnchor(index), var_ref_node->GetInDataAnchor(0));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "add  data anchor between var_ref_node and final_writable peer_node failed");
    return FAILED;
  }
  return SUCCESS;
}

ge::NodePtr VariablePrepareOpPass::CreatVariableRef(ge::NodePtr &final_writable_node, ge::NodePtr &var_node) {
  if ((final_writable_node == nullptr) || (var_node == nullptr) || (var_node->GetOwnerComputeGraph() == nullptr)) {
    GELOGE(FAILED, "parameter ptr is null.");
    return nullptr;
  }
  GELOGI("Create VarRef Op: final_writable_node: [%s] var_node: [%s]>>>>", final_writable_node->GetName().c_str(),
         var_node->GetName().c_str());

  static uint32_t var_ref_count = 0;
  std::stringstream var_ref_name;
  var_ref_name << "_to_" << final_writable_node->GetName() << "_REF_" << var_ref_count++;

  OpDescPtr var_op_desc = var_node->GetOpDesc();
  if (var_op_desc == nullptr) {
    GELOGE(FAILED, "get var opdesc is nullptr");
    return nullptr;
  }

  OpDescPtr var_ref_op_desc = MakeShared<OpDesc>(var_node->GetName() + var_ref_name.str(), var_op_desc->GetType());
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
  NodePtr var_ref_node = var_node->GetOwnerComputeGraph()->AddNode(var_ref_op_desc);
  GE_IF_BOOL_EXEC(var_ref_node == nullptr, GELOGW("var_ref_node is null"); return nullptr);

  bool is_set_str = ge::AttrUtils::SetStr(var_ref_op_desc, REF_VAR_SRC_VAR_NAME, var_op_desc->GetName());
  if (is_set_str) {
    GELOGI("Set node [%s] REF_VAR_SRC_VAR_NAME [%s]", var_ref_node->GetName().c_str(), var_op_desc->GetName().c_str());
  }
  return var_ref_node;
}

int VariablePrepareOpPass::GetWritableNodeOutIndex(const NodePtr &node, int input_index) {
  if (node == nullptr) {
    return -1;
  }
  GELOGI("get writable node and input index %s:%d", node->GetName().c_str(), input_index);
  auto node_type = node->GetType();
  if (node_type == ASSIGN) {
    if (UpdateAssignOpDesc(node) != SUCCESS) {
      return -1;
    }
  }

  auto node_iter = ref_input_output_map_.find(node_type);
  if (node_iter == ref_input_output_map_.end()) {
    return -1;
  }

  auto index_iter = node_iter->second.find(input_index);
  if (index_iter == node_iter->second.end()) {
    return -1;
  }
  return index_iter->second;
}

void VariablePrepareOpPass::GenerateRefTypeAndInputOutputMap(const NodePtr &node) {
  auto out_op_desc = node->GetOpDesc();
  map<string, int> input_name_index;
  for (const auto &input_name : out_op_desc->GetAllInputNames()) {
    int index = out_op_desc->GetInputIndexByName(input_name);
    input_name_index.emplace(input_name, index);
  }

  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    string out_data_anchor_name = out_op_desc->GetOutputNameByIndex(out_data_anchor->GetIdx());
    auto iter = input_name_index.find(out_data_anchor_name);
    if (iter != input_name_index.end()) {
      GELOGD("From input_name_index_map find corresponding output name and out index : [ %s : %d]",
             out_data_anchor_name.c_str(), out_data_anchor->GetIdx());
      auto ref_type_iter = ref_input_output_map_.find(node->GetType());
      if (ref_type_iter != ref_input_output_map_.end()) {
        GELOGD("From ref_input_output_map_ find already existed ref_type_iter. Type : [%s]",
               ref_type_iter->first.c_str());
        auto input_output_iter = ref_type_iter->second.find(iter->second);
        if (input_output_iter != ref_type_iter->second.end()) {
          ref_type_iter->second.emplace(iter->second, out_data_anchor->GetIdx());
          GELOGI("Add RefInputOutputMap  [ %s ] : {%d, %d}", node->GetType().c_str(), iter->second,
                 out_data_anchor->GetIdx());
        }
      } else {
        ref_input_output_map_.insert({node->GetType(), {{iter->second, out_data_anchor->GetIdx()}}});
        GELOGI("Create RefInputOutputMap { %s : {%d, %d}}", node->GetType().c_str(), iter->second,
               out_data_anchor->GetIdx());
      }
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
}  // namespace ge
