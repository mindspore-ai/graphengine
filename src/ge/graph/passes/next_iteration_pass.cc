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

#include "graph/passes/next_iteration_pass.h"

#include <algorithm>
#include <memory>
#include <stack>
#include <string>
#include <utility>

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/common/omg_util.h"
#include "graph/debug/ge_attr_define.h"


namespace ge {
Status NextIterationPass::Run(ComputeGraphPtr graph) {
  GELOGD("NextIterationPass Enter");
  /// Enter-----------+
  ///                 +-> Merge -> Switch <- LoopCond <- Cond
  /// NextIteration---+
  for (auto &node : graph->GetDirectNode()) {
    const std::string type = node->GetType();
    if ((type != ENTER) && (type != REFENTER)) {
      continue;
    }
    if (HandleEnterNode(node) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "HandleEnterNode for node %s fail.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }

  if (FindWhileGroups() != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "FindWhileGroups fail");
    return INTERNAL_ERROR;
  }

  if (!VerifyWhileGroup()) {
    GELOGE(INTERNAL_ERROR, "VerifyWhileGroup fail");
    return INTERNAL_ERROR;
  }

  if (HandleWhileGroup(graph) != SUCCESS) {
    GELOGE(FAILED, "HandleWhileGroup fail");
    return FAILED;
  }

  GELOGD("NextIterationPass Leave");
  return SUCCESS;
}

///
/// @brief Handle Enter node
/// @param [in] enter_node
/// @return Status
///
Status NextIterationPass::HandleEnterNode(const NodePtr &enter_node) {
  OpDescPtr enter_desc = enter_node->GetOpDesc();
  GE_CHECK_NOTNULL(enter_desc);
  std::string frame_name;
  if (!ge::AttrUtils::GetStr(enter_desc, ENTER_ATTR_FRAME_NAME, frame_name) || frame_name.empty()) {
    GELOGE(FAILED, "Get attr ENTER_ATTR_FRAME_NAME fail, node: %s", enter_desc->GetName().c_str());
    return FAILED;
  }

  auto iter = loop_group_map_.find(frame_name);
  if (iter == loop_group_map_.end()) {
    LoopCondGroupPtr loop_group = MakeShared<LoopCondGroup>();
    if (loop_group == nullptr) {
      GELOGE(FAILED, "MakeShared for LoopCondGroup fail.");
      return FAILED;
    }
    loop_group->enter_nodes.emplace_back(enter_node);
    loop_group_map_[frame_name] = loop_group;
  } else {
    iter->second->enter_nodes.emplace_back(enter_node);
  }

  return SUCCESS;
}

///
/// @brief Find while groups
/// @return Status
///
Status NextIterationPass::FindWhileGroups() {
  for (auto &loop_group_iter : loop_group_map_) {
    const std::string frame_name = loop_group_iter.first;
    for (auto &enter_node : loop_group_iter.second->enter_nodes) {
      for (auto &out_node : enter_node->GetOutAllNodes()) {
        const std::string type = out_node->GetType();
        if ((type != MERGE) && (type != REFMERGE)) {
          continue;
        }

        NodePtr next_node = nullptr;
        if (FindTargetNode(out_node, NEXTITERATION, true, next_node) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "Get NextIteration node fail, frame_name: %s.", frame_name.c_str());
          return INTERNAL_ERROR;
        }

        NodePtr switch_node = nullptr;
        if (FindTargetNode(out_node, SWITCH, false, switch_node) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "Get Switch node fail, frame_name: %s.", frame_name.c_str());
          return INTERNAL_ERROR;
        }

        NodePtr loop_cond = nullptr;
        if (FindTargetNode(switch_node, LOOPCOND, true, loop_cond) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "Get LoopCond node fail, frame_name: %s.", frame_name.c_str());
          return INTERNAL_ERROR;
        }

        if (loop_group_iter.second->loop_cond == nullptr) {
          loop_group_iter.second->loop_cond = loop_cond;
        } else if (loop_group_iter.second->loop_cond != loop_cond) {
          GELOGE(FAILED, "Multi LoopCond nodes exist, frame_name: %s.", frame_name.c_str());
          return FAILED;
        }
        loop_group_iter.second->merge_next_pairs.emplace_back(std::make_pair(out_node, next_node));
      }
    }
  }

  return SUCCESS;
}

///
/// @brief Verify if valid
/// @return bool
///
bool NextIterationPass::VerifyWhileGroup() {
  // map<frame_name, LoopCondGroup>
  for (auto &loop_group_iter : loop_group_map_) {
    const std::string frame_name = loop_group_iter.first;
    if (frame_name.empty()) {
      GELOGE(INTERNAL_ERROR, "VerifyWhileGroup fail, frame_name is empty.");
      return false;
    }

    if (loop_group_iter.second->loop_cond == nullptr) {
      GELOGE(INTERNAL_ERROR, "VerifyWhileGroup fail, LoopCond is null, frame_name: %s.", frame_name.c_str());
      return false;
    }

    for (auto &pair_iter : loop_group_iter.second->merge_next_pairs) {
      if ((pair_iter.first == nullptr) || (pair_iter.second == nullptr)) {
        GELOGE(INTERNAL_ERROR, "VerifyWhileGroup fail, merge_node/next_node is null, frame_name: %s.",
               frame_name.c_str());
        return false;
      }
    }
  }

  return true;
}

///
/// @brief Handle while group
/// @param [in] graph
/// @return Status
///
Status NextIterationPass::HandleWhileGroup(ComputeGraphPtr &graph) {
  for (auto &loop_cond_iter : loop_group_map_) {
    std::string cond_name = loop_cond_iter.second->loop_cond->GetName();
    GELOGI("HandleWhileGroup, LoopCond node: %s.", cond_name.c_str());

    // Create Active node, Enter->Active->Merge, NextItaration->Active->Merge
    NodePtr enter_active = CreateActiveNode(graph, cond_name + "_Enter_" + STREAMACTIVE);
    NodePtr next_active = CreateActiveNode(graph, cond_name + "_Next_" + STREAMACTIVE);
    if ((enter_active == nullptr) || (next_active == nullptr)) {
      GELOGE(INTERNAL_ERROR, "CreateActiveNode fail, cond_name: %s.", cond_name.c_str());
      return INTERNAL_ERROR;
    }

    for (auto &enter_node : loop_cond_iter.second->enter_nodes) {
      // Enter --> Active
      if (GraphUtils::AddEdge(enter_node->GetOutControlAnchor(), enter_active->GetInControlAnchor()) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Add control edge fail");
        return INTERNAL_ERROR;
      }
    }

    for (auto &pair : loop_cond_iter.second->merge_next_pairs) {
      NodePtr merge_node = pair.first;
      NodePtr next_node = pair.second;
      // Active --> Merge
      if (GraphUtils::AddEdge(enter_active->GetOutControlAnchor(), merge_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Add control edge fail");
        return INTERNAL_ERROR;
      }

      // NextIteration --> Active
      if (GraphUtils::AddEdge(next_node->GetOutControlAnchor(), next_active->GetInControlAnchor()) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Add control edge fail");
        return INTERNAL_ERROR;
      }

      // break link between NextIteration and Merge
      if (BreakNextIteration(next_node, merge_node) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "BreakNextIteration failed");
        return INTERNAL_ERROR;
      }
    }

    if ((SetActiveLabelList(enter_active, {cond_name}) != SUCCESS) ||
        (SetActiveLabelList(next_active, {cond_name}) != SUCCESS)) {
      GELOGE(INTERNAL_ERROR, "SetActiveLabelList failed");
      return INTERNAL_ERROR;
    }
  }

  return SUCCESS;
}

///
/// @brief Create Active Node
/// @param [in] graph
/// @param [in] name
/// @return ge::NodePtr
///
NodePtr NextIterationPass::CreateActiveNode(ComputeGraphPtr &graph, const std::string &name) {
  OpDescPtr op_desc = MakeShared<OpDesc>(name, STREAMACTIVE);
  if (op_desc == nullptr) {
    return nullptr;
  }

  GELOGI("Create StreamActive op:%s.", op_desc->GetName().c_str());
  NodePtr active_node = graph->AddNode(op_desc);
  if (active_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "Create node[%s] fail.", name.c_str());
    return nullptr;
  }

  if (SetSwitchBranchNodeLabel(active_node, name) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "SetSwitchBranchNodeLabel for node: %s failed.", active_node->GetName().c_str());
    return nullptr;
  }

  return active_node;
}

///
/// @brief Break NextIteration Link & add name to merge attr
/// @param [in] next_node
/// @param [in] merge_node
/// @return Status
///
Status NextIterationPass::BreakNextIteration(const NodePtr &next_node, NodePtr &merge_node) {
  if ((merge_node == nullptr) || (next_node == nullptr)) {
    GELOGE(PARAM_INVALID, "merge node or next node is null.");
    return PARAM_INVALID;
  }
  for (auto &in_anchor : merge_node->GetAllInDataAnchors()) {
    OutDataAnchorPtr out_anchor = in_anchor->GetPeerOutAnchor();
    if ((out_anchor == nullptr) || (out_anchor->GetOwnerNode() != next_node)) {
      continue;
    }
    if (GraphUtils::RemoveEdge(out_anchor, in_anchor) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Remove data edge fail, %s->%s.", next_node->GetName().c_str(),
             merge_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    if (SetNextIteration(merge_node, next_node->GetName()) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "SetNextIteration for node %s fail.", merge_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

///
/// @brief find target node
/// @param [in] node
/// @param [in] target_type
/// @param [in] is_input
/// @param [out] target_node
/// @return Status
///
Status NextIterationPass::FindTargetNode(const NodePtr &node, const std::string &target_type, bool is_input,
                                         NodePtr &target_node) {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "node is null.");
    return PARAM_INVALID;
  }
  std::vector<NodePtr> nodes;
  if (is_input) {
    for (auto &tmp_node : node->GetInDataNodes()) {
      nodes.emplace_back(tmp_node);
    }
  } else {
    for (auto &tmp_node : node->GetOutDataNodes()) {
      nodes.emplace_back(tmp_node);
    }
  }

  for (auto &tmp_node : nodes) {
    const std::string type = tmp_node->GetType();
    if ((target_type == LOOPCOND) && (type == target_type)) {
      target_node = tmp_node;
      break;
    } else if ((type == target_type) || (type == "Ref" + target_type)) {
      target_node = tmp_node;
      break;
    }
  }

  if (target_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "Find node %s fail", target_type.c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}
}  // namespace ge
