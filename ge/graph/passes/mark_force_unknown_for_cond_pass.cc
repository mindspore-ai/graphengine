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

#include <queue>

#include "graph/utils/node_utils.h"
#include "graph/common/omg_util.h"

namespace ge {
namespace {
inline bool IsMergeInLoop(const NodePtr &node) {
  const static std::set<std::string> kLoopMergeInputs{ ENTER, REFENTER, NEXTITERATION, REFNEXTITERATION };

  std::string node_type;
  (void)GetOriginalType(node, node_type);
  return kLoopMergeInputs.count(node_type) > 0;
}

inline bool IsSwitchInLoop(const NodePtr &node) {
  const static std::set<std::string> kLoopSwitchInputs{ MERGE, REFMERGE, LOOPCOND };

  std::string node_type;
  (void)GetOriginalType(node, node_type);
  return kLoopSwitchInputs.count(node_type) > 0;
}
}

Status MarkForceUnknownForCondPass::Run(ComputeGraphPtr graph) {
  GELOGD("MarkForceUnknownForCondPass Enter");
  std::map<NodePtr, std::vector<NodePtr>> switch_groups;
  for (const auto &node : graph->GetDirectNode()) {
    std::string node_type;
    GE_CHK_STATUS_RET(GetOriginalType(node, node_type), "Get original type failed.");
    if (kMergeOpTypes.count(node_type) == 0) {
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
/// @brief Mark force unknown shape for Switch node
/// @param [in] merge node
/// @param [out] switch group
/// @return
///
void MarkForceUnknownForCondPass::MarkUnknownForSwitch(const NodePtr &node, std::vector<NodePtr> &switch_group) {
  // Switch --> {Switch --> Merge} --> Merge
  std::unordered_set<NodePtr> nodes_seen;
  std::queue<std::pair<NodePtr, uint32_t>> search_queue({{node, 0}});
  while (!search_queue.empty()) {
    const auto dst_node = search_queue.front().first;
    const auto dst_span = search_queue.front().second;
    search_queue.pop();

    // Switch --> Identity --> Constant
    for (const auto &in_node : dst_node->GetInControlNodes()) {
      if (nodes_seen.count(in_node) > 0) {
        GELOGD("Travel node: %s, Skip already seen node: %s", dst_node->GetName().c_str(), in_node->GetName().c_str());
        continue;
      }
      nodes_seen.insert(in_node);

      if (in_node->GetType() == IDENTITY) {
        GELOGD("Travel node: %s, In control: %s, span is: %u", dst_node->GetName().c_str(),
               in_node->GetName().c_str(), dst_span);
        search_queue.push({in_node, dst_span});
      }
    }

    for (const auto &in_node : dst_node->GetInDataNodes()) {
      if (nodes_seen.count(in_node) > 0) {
        GELOGD("Travel node: %s, Skip already seen node: %s", dst_node->GetName().c_str(), in_node->GetName().c_str());
        continue;
      }
      nodes_seen.insert(in_node);

      std::string node_type;
      (void)GetOriginalType(in_node, node_type);
      GELOGD("Travel node: %s, %s node: %s, span is: %u", dst_node->GetName().c_str(), node_type.c_str(),
             in_node->GetName().c_str(), dst_span);
      if (kSwitchOpTypes.count(node_type) > 0) { // Switch input node.
        if (dst_span > 0) {
          search_queue.push({in_node, dst_span - 1});
        } else {
          const auto &all_in_nodes = in_node->GetInDataNodes();
          if (std::any_of(all_in_nodes.begin(), all_in_nodes.end(), IsSwitchInLoop)) {
            GELOGW("Travel node: %s, %s node: %s, Skip LoopCond switch", dst_node->GetName().c_str(), node_type.c_str(),
                   in_node->GetName().c_str());
          } else {
            switch_group.emplace_back(in_node);
          }
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
  std::function<bool(const NodePtr &)> callback = [](const NodePtr &n) {
    return n->GetOpDesc()->HasAttr(ATTR_NAME_CONTROL_FLOW_GROUP);
  };

  for (auto it1 = switch_groups.begin(); it1 != switch_groups.end(); ++it1) {
    const auto &op_node1 = it1->first;
    const auto &op_desc1 = op_node1->GetOpDesc();
    if (op_desc1->HasAttr(ATTR_NAME_CONTROL_FLOW_GROUP)) {
      continue;
    }

    if (IsUnknownShapeTensor(op_desc1->GetOutputDesc(0))) {
      int64_t group_index = op_desc1->GetId();
      GELOGI("Mark %s as unknown shape control flow, group index: %ld", op_desc1->GetName().c_str(), group_index);
      MarkForceUnknownShape(op_node1, true, group_index);
      for (const auto &n : it1->second) {
        MarkForceUnknownShape(n, true, group_index);
      }

      for (auto it2 = switch_groups.begin(); it2 != switch_groups.end(); ++it2) {
        const auto &op_node2 = it2->first;
        const auto &op_desc2 = op_node2->GetOpDesc();
        if (op_desc2->HasAttr(ATTR_NAME_CONTROL_FLOW_GROUP)) {
          continue;
        }

        if (std::any_of(it2->second.begin(), it2->second.end(), callback)) {
          MarkForceUnknownShape(op_node2, true, group_index);
          for (const auto &n : it2->second) {
            MarkForceUnknownShape(n, true, group_index);
          }
        }
      }
    }
  }
}
} // namespace ge
