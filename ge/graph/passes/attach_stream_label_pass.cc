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

#include "graph/passes/attach_stream_label_pass.h"
#include "ge/ge_api_types.h"
#include "graph/common/omg_util.h"

namespace ge {
Status AttachStreamLabelPass::Run(ComputeGraphPtr graph) {
  GELOGD("AttachStreamLabelPass Enter.");

  FindNodes(graph);
  for (const auto &node : need_label_nodes_) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (!op_desc->HasAttr(ATTR_NAME_STREAM_LABEL)) {
      GE_CHK_STATUS_RET(UpdateCondBranch(node), "Update cond branch failed, start node:%s.", node->GetName().c_str());
    }
  }
  GE_CHK_STATUS_RET(UpdateEnterNode(), "UpdateEnterNode failed.");

  GELOGD("AttachStreamLabelPass Leave.");
  return SUCCESS;
}

///
/// @brief Clear Status, used for subgraph pass
/// @return
///
Status AttachStreamLabelPass::ClearStatus() {
  stream_switch_nodes_.clear();
  need_label_nodes_.clear();
  enter_nodes_.clear();
  branch_head_nodes_.clear();
  return SUCCESS;
}

///
/// @brief Find StreamSwitch / StreamMerge / Enter node
/// @param [in] graph
/// @return void
///
void AttachStreamLabelPass::FindNodes(const ComputeGraphPtr &graph) {
  for (const NodePtr &node : graph->GetDirectNode()) {
    const std::string &type = node->GetType();
    if (type == STREAMSWITCH) {
      stream_switch_nodes_.emplace_back(node);
    } else if (type == STREAMMERGE) {
      if ((node->GetOpDesc() != nullptr) && !node->GetOpDesc()->HasAttr(ATTR_NAME_NEXT_ITERATION)) {
        need_label_nodes_.emplace_back(node);
      }
    } else if ((type == ENTER) || (type == REFENTER)) {
      enter_nodes_.emplace_back(node);
    }
  }

  for (const auto &node : stream_switch_nodes_) {
    for (const auto &out_ctrl_node : node->GetOutControlNodes()) {
      GELOGD("branch_head_node %s of stream_switch %s.", out_ctrl_node->GetName().c_str(), node->GetName().c_str());
      branch_head_nodes_[out_ctrl_node] = node;
    }
    need_label_nodes_.emplace_back(node);
  }
}

///
/// @brief update cond branch
/// @param [in] node
/// @return Status
///
Status AttachStreamLabelPass::UpdateCondBranch(const NodePtr &node) {
  std::string stream_label;
  std::unordered_set<NodePtr> branch_nodes;
  std::unordered_set<NodePtr> visited;
  std::stack<NodePtr> nodes;
  nodes.push(node);

  static const std::set<std::string> end_type_set = {STREAMSWITCH, STREAMMERGE, MERGE};
  bool merge_flag = false;
  bool exit_flag = false;
  bool net_output_flag = false;
  while (!nodes.empty()) {
    NodePtr cur_node = nodes.top();
    nodes.pop();
    if (visited.count(cur_node) > 0) {
      continue;
    }
    if (AttachFlag(cur_node, stream_label, merge_flag, exit_flag, net_output_flag) != SUCCESS) {
      GELOGE(FAILED, "Attach flag for node %s failed.", cur_node->GetName().c_str());
      return FAILED;
    }

    const std::string &type = cur_node->GetType();
    for (const auto &out_node : cur_node->GetOutAllNodes()) {
      const std::string &out_type = out_node->GetType();
      bool stop_flag = (end_type_set.count(out_type) > 0) ||
                       ((branch_head_nodes_.count(out_node) > 0) && (branch_head_nodes_[out_node] != node)) ||
                       (((type == ENTER) || (type == REFENTER)) && (out_type != STREAMACTIVE));
      if (!stop_flag) {
        nodes.push(out_node);
        GELOGD("Insert branch node %s.", out_node->GetName().c_str());
        branch_nodes.insert(out_node);
      }
    }
    visited.insert(cur_node);
  }

  if (node->GetType() == STREAMSWITCH) {
    GE_CHK_STATUS_RET(SetActiveLabelList(node, {stream_label}), "set active_label_list failed.");
  }

  bool attach_flag = (merge_flag || exit_flag) && net_output_flag;
  if (attach_flag) {
    GELOGI("No need to keep on attaching label.");
    return SUCCESS;
  }

  for (const NodePtr &tmp_node : branch_nodes) {
    GELOGD("Attach label %s to node: %s.", stream_label.c_str(), tmp_node->GetName().c_str());
    GE_CHK_STATUS_RET(SetStreamLabel(tmp_node, stream_label), "Set stream label failed.");
  }

  return SUCCESS;
}

///
/// @brief attach flag
/// @param [in] node
/// @param [out] stream_label
/// @param [out] merge_flag
/// @param [out] exit_flag
/// @param [out] net_output_flag
/// @return Status
///
Status AttachStreamLabelPass::AttachFlag(const NodePtr &node, std::string &stream_label, bool &merge_flag,
                                         bool &exit_flag, bool &net_output_flag) {
  const std::string &type = node->GetType();
  if (type == STREAMSWITCH) {
    if (node->GetInDataNodes().empty()) {
      GELOGE(INTERNAL_ERROR, "node %s has no input_data_node.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    stream_label = node->GetInDataNodes().at(0)->GetName();
    GE_CHK_STATUS_RET(SetStreamLabel(node, stream_label), "Set stream label failed.");
    bool value = false;
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHK_BOOL_EXEC(AttrUtils::GetBool(op_desc, ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, value), return FAILED,
                     "StreamSwitch get attr TRUE_BRANCH_STREAM failed.");
    stream_label += (value ? "_t" : "_f");
  } else if (type == STREAMMERGE) {
    stream_label = node->GetName();
    GE_CHK_STATUS_RET(SetStreamLabel(node, stream_label), "Set stream label failed.");
    merge_flag = true;
  } else if ((type == EXIT) || (type == REFEXIT)) {
    GE_CHK_STATUS_RET(SetStreamLabel(node, stream_label), "Set stream label failed.");
    exit_flag = true;
  } else if (type == NETOUTPUT) {
    net_output_flag = true;
  }

  return SUCCESS;
}

///
/// @brief Update stream_label start with enter nodes
/// @return Status
///
Status AttachStreamLabelPass::UpdateEnterNode() {
  std::unordered_map<NodePtr, std::vector<NodePtr>> enter_active_map;
  for (const auto &enter_node : enter_nodes_) {
    for (const auto &out_ctrl_node : enter_node->GetOutControlNodes()) {
      if (out_ctrl_node->GetType() == STREAMACTIVE) {
        if (enter_active_map.find(out_ctrl_node) == enter_active_map.end()) {
          enter_active_map[out_ctrl_node] = {enter_node};
        } else {
          enter_active_map[out_ctrl_node].emplace_back(enter_node);
        }
      }
    }
  }

  for (const auto &pair : enter_active_map) {
    if (SetEnterLabel(pair.second, pair.first) != SUCCESS) {
      GELOGE(FAILED, "Set stream_label for enter_nodes failed.");
      return FAILED;
    }

    NodePtr active_node = pair.first;
    GE_CHECK_NOTNULL(active_node);
    std::vector<std::string> active_label_list;
    bool get_attr = AttrUtils::GetListStr(active_node->GetOpDesc(), ATTR_NAME_ACTIVE_LABEL_LIST, active_label_list) &&
                    (active_label_list.size() == 1) && !active_label_list[0].empty();
    if (!get_attr) {
      GELOGE(INTERNAL_ERROR, "Get attr ATTR_NAME_ACTIVE_LABEL_LIST failed, node: %s.", active_node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    std::stack<NodePtr> enter_nodes;
    std::string batch_label;
    for (const auto &enter_node : pair.second) {
      enter_nodes.emplace(enter_node);
      std::string tmp_label;
      (void)AttrUtils::GetStr(enter_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, tmp_label);
      if (!tmp_label.empty()) {
        if (batch_label.empty()) {
          batch_label = tmp_label;
        } else if (batch_label != tmp_label) {
          GELOGE(FAILED, "multi batch_label exist, label1=%s, label2=%s.", batch_label.c_str(), tmp_label.c_str());
          return FAILED;
        }
      }
    }
    if (UpdateLoopBranch(enter_nodes, active_label_list[0], batch_label) != SUCCESS) {
      GELOGE(FAILED, "Update stream_label for loop_branch failed.");
      return FAILED;
    }
  }

  return SUCCESS;
}

///
/// @brief Set stream_label for enter_nodes
/// @param [in] enter_nodes
/// @param [in] active_node
/// @return Status
///
Status AttachStreamLabelPass::SetEnterLabel(const std::vector<NodePtr> &enter_nodes, const NodePtr &active_node) {
  std::string stream_label;
  GE_CHECK_NOTNULL(active_node);
  (void)AttrUtils::GetStr(active_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label);

  if (stream_label.empty()) {
    GELOGW("stream_label of enter_active & enter_nodes is empty.");
    return SUCCESS;
  }

  for (const auto &enter_node : enter_nodes) {
    GE_CHECK_NOTNULL(enter_node->GetOpDesc());
    if (enter_node->GetOpDesc()->HasAttr(ATTR_NAME_STREAM_LABEL)) {
      GE_CHK_STATUS_RET(SetStreamLabel(enter_node, stream_label), "Set stream label failed.");
    }
  }
  GE_CHK_STATUS_RET(SetStreamLabel(active_node, stream_label), "Set stream label failed.");
  return SUCCESS;
}

///
/// @brief Update stream_label for loop_branch
/// @param [in] enter_nodes
/// @param [in] stream_label
/// @param [in] batch_label
/// @return Status
///
Status AttachStreamLabelPass::UpdateLoopBranch(const std::stack<NodePtr> &enter_nodes, const std::string &stream_label,
                                               const std::string &batch_label) {
  std::stack<NodePtr> nodes(enter_nodes);
  NodePtr cur_node = nullptr;
  while (!nodes.empty()) {
    cur_node = nodes.top();
    nodes.pop();
    for (const NodePtr &out_node : cur_node->GetOutAllNodes()) {
      OpDescPtr out_desc = out_node->GetOpDesc();
      GE_CHECK_NOTNULL(out_desc);
      std::string tmp_label;
      (void)AttrUtils::GetStr(out_desc, ATTR_NAME_BATCH_LABEL, tmp_label);
      if (!tmp_label.empty() && (tmp_label != batch_label)) {
        continue;
      }
      std::string out_type = out_desc->GetType();
      bool need_skip =
          out_desc->HasAttr(ATTR_NAME_STREAM_LABEL) || (out_type == ENTER) || (out_type == REFENTER) ||
          (((cur_node->GetType() == ENTER) || (cur_node->GetType() == REFENTER)) && (out_type == STREAMACTIVE));
      if (need_skip) {
        continue;
      }
      GELOGD("Attach label %s to node: %s.", stream_label.c_str(), out_node->GetName().c_str());
      GE_CHK_STATUS_RET(SetStreamLabel(out_node, stream_label), "Set stream label failed.");
      nodes.push(out_node);
    }
  }
  return SUCCESS;
}
}  // namespace ge
