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

#ifndef GE_GRAPH_PASSES_NEXT_ITERATION_PASS_H_
#define GE_GRAPH_PASSES_NEXT_ITERATION_PASS_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "inc/graph_pass.h"

struct LoopCondGroup {
  LoopCondGroup() : loop_cond(nullptr) {}
  ge::NodePtr loop_cond;                                              // LoopCond node
  std::vector<ge::NodePtr> enter_nodes;                               // Enter nodes
  std::vector<std::pair<ge::NodePtr, ge::NodePtr>> merge_next_pairs;  // <Merge, NextIteration>
};
using LoopCondGroupPtr = std::shared_ptr<LoopCondGroup>;

namespace ge {
class NextIterationPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);
  Status ClearStatus() override;

 private:
  Status HandleEnterNode(const NodePtr &enter_node);
  Status FindWhileGroups();
  bool VerifyWhileGroup();
  Status HandleWhileGroup(ComputeGraphPtr &graph);
  NodePtr CreateActiveNode(ComputeGraphPtr &graph, const std::string &name);
  Status BreakNextIteration(const NodePtr &next_node, NodePtr &merge_node);
  Status FindTargetNode(const NodePtr &node, const std::string &target_type, bool is_input, NodePtr &target_node);

  // map<frame_name, LoopCondGroup>
  std::unordered_map<std::string, LoopCondGroupPtr> loop_group_map_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_NEXT_ITERATION_PASS_H_
