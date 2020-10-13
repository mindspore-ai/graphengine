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

#include "graph/passes/variable_ref_delete_op_pass.h"
#include <string>

namespace ge {
Status VariableRefDeleteOpPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  std::set<std::string> all_var_names;
  auto root_graph = GraphUtils::FindRootGraph(graph);
  GE_CHECK_NOTNULL(root_graph);
  for (const auto &n : root_graph->GetAllNodes()) {
    all_var_names.insert(n->GetName());
  }
  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string ref_var_src_var_name;
    bool is_variable_ref = (node->GetOpDesc()->GetType() == VARIABLE) &&
                           (ge::AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name));
    if (!is_variable_ref) {
      continue;
    }
    if (all_var_names.count(ref_var_src_var_name) == 0) {
      GELOGE(FAILED, "Can not find source variable[%s] of variable ref[%s]", ref_var_src_var_name.c_str(),
             node->GetName().c_str());
      return FAILED;
    }
    Status ret = DealVariableRef(graph, node, ref_var_src_var_name);
    if (ret != SUCCESS) {
      GELOGE(ret, "variable ref [%s] delete failed", node->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status VariableRefDeleteOpPass::DealVariableRef(ge::ComputeGraphPtr &graph, ge::NodePtr &variable_ref,
                                                const std::string &ref_var_src_var_name) {
  GE_CHECK_NOTNULL(variable_ref);
  auto inAnchor0 = variable_ref->GetInDataAnchor(0);
  if (inAnchor0 == nullptr) {
    GELOGE(FAILED, "variable_ref [%s] no input", variable_ref->GetName().c_str());
    return FAILED;
  }
  GE_CHECK_NOTNULL(inAnchor0->GetPeerOutAnchor());
  // get the output index of the previous node connected to the variable_ref
  // prepare for refreshing address in build phase
  int index = inAnchor0->GetPeerOutAnchor()->GetIdx();

  // get previous node of variable_ref
  NodePtr peer_node = inAnchor0->GetPeerOutAnchor()->GetOwnerNode();

  // add attr [REF_VAR_SRC_VAR_NAME] to the previous op output desc of the variable_ref
  auto op_desc = peer_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto out_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
  bool is_set_str = ge::AttrUtils::SetStr(out_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);
  if (is_set_str) {
    GELOGI("[%s-%d]: add attr [REF_VAR_SRC_VAR_NAME: %s ] ", peer_node->GetName().c_str(), index,
           ref_var_src_var_name.c_str());
  } else {
    GELOGE(FAILED, "[%s-%d]: add attr [REF_VAR_SRC_VAR_NAME: %s ] failed", peer_node->GetName().c_str(), index,
           ref_var_src_var_name.c_str());
    return FAILED;
  }
  // remove variable_ref
  if (GraphUtils::IsolateNode(variable_ref, {0}) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Isolate removed node: %s, type: %s failed", variable_ref->GetName().c_str(),
           variable_ref->GetType().c_str());
    return FAILED;
  }
  if (GraphUtils::RemoveNodeWithoutRelink(graph, variable_ref) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Remove node: %s, type: %s without relink failed", variable_ref->GetName().c_str(),
           variable_ref->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace ge
