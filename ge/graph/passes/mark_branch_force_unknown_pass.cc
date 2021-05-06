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

#include "mark_branch_force_unknown_pass.h"

#include <queue>

#include "graph/common/omg_util.h"

namespace ge {
namespace {
const std::set<std::string> kMergeOpTypes{ MERGE, REFMERGE };

const std::set<std::string> kSwitchOpTypes{ SWITCH, REFSWITCH };

const std::set<std::string> kLoopMergeInputs{ ENTER, REFENTER, NEXTITERATION, REFNEXTITERATION };

inline bool IsMergeInLoop(const NodePtr &node) {
  std::string node_type;
  (void)GetOriginalType(node, node_type);
  return kLoopMergeInputs.count(node_type) > 0;
}
}

Status MarkBranchForceUnknownPass::Run(ComputeGraphPtr graph) {
  GELOGD("MarkBranchForceUnknownPass Enter");
  for (const auto &node : graph->GetDirectNode()) {
    std::string node_type;
    GE_CHK_STATUS_RET(GetOriginalType(node, node_type), "Get original type failed.");
    if (kMergeOpTypes.count(node_type) == 0) {
      continue;
    }

    const auto op_desc = node->GetOpDesc();
    if (!op_desc->HasAttr(ATTR_NAME_FORCE_UNKNOWN_SHAPE) && !IsUnknownShapeTensor(op_desc->GetOutputDesc(0))) {
      GELOGI("Merge[%s] has known shape, no need check switch", node->GetName().c_str());
      continue;
    }

    const auto &all_in_nodes = node->GetInDataNodes();
    if (std::any_of(all_in_nodes.begin(), all_in_nodes.end(), IsMergeInLoop)) {
      continue;  // LoopCond marked in NextIterationPass.
    }

    MarkUnknownForSwitch(node);
  }

  GELOGD("MarkBranchForceUnknownPass Leave");
  return SUCCESS;
}

///
/// @brief Mark force unknown shape for Switch node
/// @param [in] merge node
/// @return
///
void MarkBranchForceUnknownPass::MarkUnknownForSwitch(const NodePtr &node) {
  // Switch --> {Switch --> Merge} --> Merge
  std::vector<NodePtr> switch_group;
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
          switch_group.emplace_back(in_node);
        }
      } else if (kMergeOpTypes.count(node_type) > 0) { // Merge input node.
        search_queue.push({in_node, dst_span + 1});
      } else {
        search_queue.push({in_node, dst_span});
      }
    }
  }

  for (const auto &n : switch_group) {
    MarkForceUnknownShape(n, true);
  }
}
} // namespace ge
