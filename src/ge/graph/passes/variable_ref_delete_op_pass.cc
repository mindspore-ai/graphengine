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
#include "framework/common/debug/ge_log.h"

namespace ge {
Status VariableRefDeleteOpPass::Run(ge::ComputeGraphPtr graph) {
  GE_TIMESTAMP_START(VariableRefDeleteOpPass);
  GE_CHECK_NOTNULL(graph);

  for (auto &node : graph->GetDirectNode()) {
    GELOGD("before VariableRefDeleteOpPass, graph has node: %s, and node name: %s", node->GetType().c_str(),
           node->GetName().c_str());
  }

  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string ref_var_src_var_name;
    bool is_variable_ref = (node->GetOpDesc()->GetType() == VARIABLE) &&
                           (ge::AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name));
    if (!is_variable_ref) {
      continue;
    }
    Status ret = DealVariableRef(graph, node, ref_var_src_var_name);
    if (ret != SUCCESS) {
      GELOGE(ret, "variable ref [%s] delete failed", node->GetName().c_str());
      return FAILED;
    }
  }

  for (auto &node : graph->GetDirectNode()) {
    GELOGD("after VariableRefDeleteOpPass, graph has node: %s, and node name: %s", node->GetType().c_str(),
           node->GetName().c_str());
  }
  GE_TIMESTAMP_END(VariableRefDeleteOpPass, "GraphManager::VariableRefDeleteOpPass");

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

  // add attr [REF_VAR_SRC_VAR_NAME] to the previous node of the variable_ref
  GE_CHECK_NOTNULL(peer_node->GetOpDesc());
  bool is_set_str = ge::AttrUtils::SetStr(peer_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);

  ge::NodePtr ref_var_src_var = graph->FindNode(ref_var_src_var_name);
  if (ref_var_src_var == nullptr) {
    GELOGE(FAILED, "get ref_var_src_var failed");
    return FAILED;
  }

  GE_CHECK_NOTNULL(ref_var_src_var->GetOpDesc());
  bool is_set_index = ge::AttrUtils::SetInt(ref_var_src_var->GetOpDesc(), REF_VAR_PRE_PEER_OUT_INDEX, index);
  if (is_set_str && is_set_index) {
    GELOGI("[%s]: add attr [REF_VAR_SRC_VAR_NAME: %s ] ", peer_node->GetName().c_str(), ref_var_src_var_name.c_str());
    GELOGI("[%s]: add attr [REF_VAR_PRE_PEER_OUT_INDEX: %d]", ref_var_src_var->GetName().c_str(), index);
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
