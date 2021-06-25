/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "mark_force_unknown_for_cond_pass.h"

#include "graph/utils/node_utils.h"
#include "graph/common/omg_util.h"

namespace ge {
namespace {
inline bool IsMergeInLoop(const NodePtr &node) {
  const static std::set<std::string> kLoopMergeInputs{ ENTER, REFENTER, NEXTITERATION, REFNEXTITERATION };

  return kLoopMergeInputs.count(NodeUtils::GetNodeType(node)) > 0;
}
}

Status MarkForceUnknownForCondPass::Run(ComputeGraphPtr graph) {
  GELOGD("MarkForceUnknownForCondPass Enter");
  std::map<NodePtr, std::vector<NodePtr>> switch_groups;
  for (const auto &node : graph->GetDirectNode()) {
    if (kMergeOpTypes.count(NodeUtils::GetNodeType(node)) == 0) {
      continue;
    }

    const auto &all_in_nodes = node->GetInDataNodes();
    if (std::any_of(all_in_nodes.begin(), all_in_nodes.end(), IsMergeInLoop)) {
      continue;  // LoopCond marked in NextIterationPass.
    }

    MarkUnknownForSwitch(node, switch_groups[node]);
  }

  MarkUnknownForSwitch(switch_groups);
  GELOGD("MarkForceUnknownForCondPass Leave");
  return SUCCESS;
}

///
/// @brief Deal with Switch node for LoopCond
/// @param [in] Switch node
/// @param [in] dest span
/// @param [out] Search queue
/// @return true: Switch In while loop / false: Not in while Loop.
///
bool MarkForceUnknownForCondPass::DealWithLoopSwitch(const NodePtr &node, uint32_t dst_span,
                                                     std::queue<std::pair<NodePtr, uint32_t>> search_queue) {
  ///                 LoopCond --->\.
  ///                               \.
  /// Enter-----------+              \.
  ///                 +--> Merge --> Switch --> Exit
  /// NextIteration---+
  const auto is_loop_op = [](const NodePtr &n) {
    return NodeUtils::GetNodeType(n) == LOOPCOND;
  };
  const auto is_exit_op = [](const NodePtr &n) {
    return kExitOpTypes.count(NodeUtils::GetNodeType(n)) > 0;
  };

  const auto src_nodes = node->GetInAllNodes();
  const auto dst_nodes = node->GetOutAllNodes();
  if (std::none_of(src_nodes.begin(), src_nodes.end(), is_loop_op) &&
      std::none_of(dst_nodes.begin(), dst_nodes.end(), is_exit_op)) {
    return false;
  }

  for (const auto &m : src_nodes) {
    if (kMergeOpTypes.count(NodeUtils::GetNodeType(m)) > 0) {
      for (const auto &n : m->GetInAllNodes()) {
        if (kNextIterationOpTypes.count(NodeUtils::GetNodeType(n)) > 0) {
          continue;
        }

        search_queue.push({n, dst_span});
        GELOGD("Travel in Loop: %s <-- %s <-- %s, span is: %u", node->GetName().c_str(), m->GetName().c_str(),
               n->GetName().c_str(), dst_span);
      }
    }
  }

  return true;
}

///
/// @brief Mark force unknown shape for Switch node
/// @param [in] merge node
/// @param [out] switch group
/// @return
///
void MarkForceUnknownForCondPass::MarkUnknownForSwitch(const NodePtr &node, std::vector<NodePtr> &switch_group) {
  // Switch --> {Switch --> Merge} --> Merge
  GELOGD("Search Switch node for Merge: %s", node->GetName().c_str());
  std::unordered_set<NodePtr> nodes_seen;
  std::queue<std::pair<NodePtr, uint32_t>> search_queue({{node, 0}});
  while (!search_queue.empty()) {
    const auto dst_node = search_queue.front().first;
    const auto dst_span = search_queue.front().second;
    search_queue.pop();

    for (const auto &in_node : dst_node->GetInAllNodes()) {
      if (nodes_seen.count(in_node) > 0) {
        GELOGD("Travel node: %s, Skip already seen node: %s", dst_node->GetName().c_str(), in_node->GetName().c_str());
        continue;
      }
      nodes_seen.insert(in_node);

      const std::string node_type = NodeUtils::GetNodeType(in_node);
      GELOGD("Travel node: %s, %s node: %s, span is: %u", dst_node->GetName().c_str(), node_type.c_str(),
             in_node->GetName().c_str(), dst_span);
      if (kSwitchOpTypes.count(node_type) > 0) { // Switch input node.
        if (DealWithLoopSwitch(in_node, dst_span, search_queue)) {
          continue;
        }

        if (dst_span > 0) {
          search_queue.push({in_node, dst_span - 1});
        } else {
          switch_group.emplace_back(in_node);
        }
      } else if (kMergeOpTypes.count(node_type) > 0) { // Merge input node.
        search_queue.push({in_node, dst_span + 1});
      } else {
        search_queue.push({in_node, dst_span});
      }
    }
  }
}

///
/// @brief Mark force unknown shape for Switch node
/// @param [in] switch groups
/// @return
///
void MarkForceUnknownForCondPass::MarkUnknownForSwitch(const std::map<NodePtr, std::vector<NodePtr>> &switch_groups) {
  for (auto it = switch_groups.begin(); it != switch_groups.end(); ++it) {
    const auto &op_node = it->first;
    const auto &op_desc = op_node->GetOpDesc();
    if (op_desc->HasAttr(ATTR_NAME_CONTROL_FLOW_GROUP)) {
      continue;
    }

    int64_t group_index = op_desc->GetId();
    SetControlFlowGroup(op_node, group_index);
    for (const auto &n : it->second) {
      SetControlFlowGroup(n, group_index);
    }
  }
}
} // namespace ge
