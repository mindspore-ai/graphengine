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

#ifndef GE_GRAPH_PASSES_SWITCH_FUSION_PASS_H_
#define GE_GRAPH_PASSES_SWITCH_FUSION_PASS_H_

#include <set>
#include "graph/passes/base_pass.h"
namespace ge {
class SwitchFusionPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  Status FuseSwitchGroup();
  Status RemoveSwitchBetweenTwoNode(const int switch_out_anchor_idx, const NodePtr &switch_node);
  Status FuseSwitchNodesToOne(NodePtr &remain_switch, const std::set<NodePtr> switch_nodes_set);
  const string GetFusionRoadId(const string branch_id, const NodePtr &switch_node);
  NodePtr InsertIdentityNode(const NodePtr &remain_switch, const int out_data_anchor_idx);
  map<string, std::set<NodePtr>> switch_group_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_SWITCH_FUSION_PASS_H_
