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

#include "graph/passes/assign_pass.h"
#include "framework/common/debug/log.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace {
const uint32_t kValidInputNodeOutputNum = 1;
const int32_t kAssignRefInputIndex = 0;
const int32_t kAssignValueInputIndex = 1;
}

namespace ge {
#if (ENABLE_OPEN_SRC != True)
Status AssignPass::Run(NodePtr &node) {
  GELOGD("AssignPass running");

  if (TransformAttr(node) != SUCCESS) {
    GELOGE(FAILED, "Transform assign_var_name attr failed, node=%s", node->GetName().c_str());
    return FAILED;
  }

  if (node->GetType() == ASSIGN) {
    if (OptimizedAssignNode(node) != SUCCESS) {
      GELOGE(FAILED, "Optimize for assign_node %s failed", node->GetName().c_str());
      return FAILED;
    }
  }

  GELOGD("AssignPass success");
  return SUCCESS;
}

///
/// @brief Optimize for assign_node
/// @param [in] assign_node
/// @return Status
///
Status AssignPass::OptimizedAssignNode(NodePtr &assign_node) {
  const auto &ref_in_anchor = assign_node->GetInDataAnchor(kAssignRefInputIndex);
  const auto &value_in_anchor = assign_node->GetInDataAnchor(kAssignValueInputIndex);
  if ((ref_in_anchor == nullptr) || (value_in_anchor == nullptr)) {
    GELOGE(FAILED, "In data anchor is null, node:%s", assign_node->GetName().c_str());
    return FAILED;
  }
  const auto &ref_peer_anchor = ref_in_anchor->GetPeerOutAnchor();
  const auto &value_peer_anchor = value_in_anchor->GetPeerOutAnchor();
  if ((ref_peer_anchor == nullptr) || (value_peer_anchor == nullptr)) {
    GELOGE(FAILED, "Peer data anchor is null, node:%s", assign_node->GetName().c_str());
    return FAILED;
  }

  if (IsCondMatch(assign_node, ref_peer_anchor, value_peer_anchor)) {
    ///
    ///    variable  not-const               not-const
    ///         \     /                          |
    ///          \   /                           |
    ///         Assign           ---->        variable
    ///           |                              |
    ///           |                              |
    ///         node                           node
    ///
    GELOGD("Optimization for assign_node %s start", assign_node->GetName().c_str());
    if (IsolateAndDeleteNode(assign_node, {kAssignRefInputIndex}) != SUCCESS) {
      GELOGE(FAILED, "Isolate and delete assign_node %s failed.", assign_node->GetName().c_str());
      return FAILED;
    }
    AddNodeDeleted(assign_node);

    const auto &ref_input = ref_peer_anchor->GetOwnerNode()->GetOpDesc();
    const auto &value_input = value_peer_anchor->GetOwnerNode()->GetOpDesc();
    if ((ref_input == nullptr) || (value_input == nullptr)) {
      GELOGE(FAILED, "value input is null");
      return FAILED;
    }

    // variable has and only has one input
    if (ref_input->UpdateInputDesc(0, value_input->GetOutputDesc(value_peer_anchor->GetIdx())) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Update input_desc for variable %s failed.", ref_input->GetName().c_str());
      return FAILED;
    }
    if (GraphUtils::AddEdge(value_peer_anchor, ref_peer_anchor->GetOwnerNode()->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add data edge %s->%s failed", value_input->GetName().c_str(), ref_input->GetName().c_str());
      return FAILED;
    }

    GELOGD("add attr ASSIGN_VAR_NAME on node %s, var_name=%s",
           value_input->GetName().c_str(), ref_input->GetName().c_str());
    if (!AttrUtils::SetStr(value_input->MutableOutputDesc(value_peer_anchor->GetIdx()), ASSIGN_VAR_NAME,
                           ref_input->GetName())) {
      GELOGE(FAILED, "Set attr ASSIGN_VAR_NAME failed.");
      return FAILED;
    }
    auto value_node = value_peer_anchor->GetOwnerNode();
    AddRePassNode(value_node);
  }
  return SUCCESS;
}

///
/// @brief Transform assign_var_name attr
/// @param [in] node
/// @return Status
///
Status AssignPass::TransformAttr(NodePtr &node) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  for (const auto &output_desc : node->GetOpDesc()->GetAllOutputsDesc()) {
    int32_t inplace_input_idx = -1;
    std::string assign_var_name;
    if (AttrUtils::GetInt(output_desc, INPLACE_SUPPORT_INPUT_INDEX, inplace_input_idx) &&
        AttrUtils::GetStr(output_desc, ASSIGN_VAR_NAME, assign_var_name)) {
      GELOGD("Transform attr ASSIGN_VAR_NAME on node %s, assign_var_name=%s, inplace_input_idx=%d, ",
             node->GetName().c_str(), assign_var_name.c_str(), inplace_input_idx);
      const auto &in_data_anchor = node->GetInDataAnchor(inplace_input_idx);
      GE_CHECK_NOTNULL(in_data_anchor);
      const auto &peer_data_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_data_anchor);
      auto in_node = peer_data_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(in_node->GetOpDesc());
      GELOGD("add attr ASSIGN_VAR_NAME on node %s, var_name=%s", in_node->GetName().c_str(), assign_var_name.c_str());
      if (!AttrUtils::SetStr(in_node->GetOpDesc()->MutableOutputDesc(peer_data_anchor->GetIdx()),
                             ASSIGN_VAR_NAME, assign_var_name)) {
        GELOGE(FAILED, "Set attr ASSIGN_VAR_NAME failed.");
        return FAILED;
      }
      AddRePassNode(in_node);
    }
  }
  return SUCCESS;
}
#else
Status AssignPass::Run(NodePtr &node) {
  GELOGD("AssignPass running");
  if (node->GetType() != ASSIGN) {
    GELOGD("No need run AssignPass on [%s, %s].", node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  const auto &ref_in_anchor = node->GetInDataAnchor(kAssignRefInputIndex);
  const auto &value_in_anchor = node->GetInDataAnchor(kAssignValueInputIndex);
  if ((ref_in_anchor == nullptr) || (value_in_anchor == nullptr)) {
    GELOGE(FAILED, "In data anchor is null, node:%s", node->GetName().c_str());
    return FAILED;
  }
  const auto &ref_peer_anchor = ref_in_anchor->GetPeerOutAnchor();
  const auto &value_peer_anchor = value_in_anchor->GetPeerOutAnchor();
  if ((ref_peer_anchor == nullptr) || (value_peer_anchor == nullptr)) {
    GELOGE(FAILED, "Peer data anchor is null, node:%s", node->GetName().c_str());
    return FAILED;
  }

  if (IsCondMatch(node, ref_peer_anchor, value_peer_anchor)) {
    ///
    ///    variable  not-const               not-const
    ///         \     /                          |
    ///          \   /                           |
    ///         Assign           ---->        variable
    ///           |                              |
    ///           |                              |
    ///         node                           node
    ///
    GELOGI("Optimization for assign_node %s start", node->GetName().c_str());
    if (IsolateAndDeleteNode(node, {kAssignRefInputIndex}) != SUCCESS) {
      GELOGE(FAILED, "Isolate and delete assign_node %s failed.", node->GetName().c_str());
      return FAILED;
    }
    AddNodeDeleted(node);

    const auto &ref_input = ref_peer_anchor->GetOwnerNode()->GetOpDesc();
    const auto &value_input = value_peer_anchor->GetOwnerNode()->GetOpDesc();
    if ((ref_input == nullptr) || (value_input == nullptr)) {
      GELOGE(FAILED, "value input is null");
      return FAILED;
    }
    if (!AttrUtils::SetStr(value_input->MutableOutputDesc(value_peer_anchor->GetIdx()), ASSIGN_VAR_NAME,
                           ref_input->GetName())) {
      GELOGE(FAILED, "Set attr ASSIGN_VAR_NAME failed.");
      return FAILED;
    }

    // variable has and only has one input
    if (ref_input->UpdateInputDesc(0, value_input->GetOutputDesc(value_peer_anchor->GetIdx())) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Update input_desc for variable %s failed.", ref_input->GetName().c_str());
      return FAILED;
    }
    if (GraphUtils::AddEdge(value_peer_anchor, ref_peer_anchor->GetOwnerNode()->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add data edge %s->%s failed", value_input->GetName().c_str(), ref_input->GetName().c_str());
      return FAILED;
    }
  }

  GELOGD("AssignPass success");
  return SUCCESS;
}
#endif
///
/// @brief Check if need optimize for assign_node
/// @param [in] assign_node
/// @param [in] peer_data_anchor for ref_input of assign_node
/// @param [in] peer_data_anchor for value_input of assign_node
/// @return Status
///
bool AssignPass::IsCondMatch(const NodePtr &node, const OutDataAnchorPtr &ref_peer_anchor,
                             const OutDataAnchorPtr &value_peer_anchor) {
  GELOGD("Check if assign_node %s match optimization condition, ref_input: %s, value_input: %s",
         node->GetName().c_str(), ref_peer_anchor->GetOwnerNode()->GetName().c_str(),
         value_peer_anchor->GetOwnerNode()->GetName().c_str());

  const std::string &value_type = value_peer_anchor->GetOwnerNode()->GetType();
  if ((value_type == CONSTANTOP) || (value_type == CONSTANT)) {
    GELOGD("value input is const");
    return false;
  }

  const std::string &ref_type = ref_peer_anchor->GetOwnerNode()->GetType();
  if ((ref_type != VARIABLE) && (ref_type != VARIABLEV2)) {
    GELOGD("ref input is not var");
    return false;
  }
  if (!ref_peer_anchor->GetOwnerNode()->GetInDataNodes().empty()) {
    GELOGD("ref input has data input");
    return false;
  }

  if ((ref_peer_anchor->GetPeerInDataNodesSize() != kValidInputNodeOutputNum) ||
      (value_peer_anchor->GetPeerInDataNodesSize() != kValidInputNodeOutputNum)) {
    GELOGD("ref / value input has other output(s)");
    return false;
  }

  GELOGD("Optimization condition matches, assign_node: %s", node->GetName().c_str());
  return true;
}
}  // namespace ge
