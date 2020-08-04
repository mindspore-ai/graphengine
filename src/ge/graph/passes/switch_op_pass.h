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

#ifndef GE_GRAPH_PASSES_SWITCH_OP_PASS_H_
#define GE_GRAPH_PASSES_SWITCH_OP_PASS_H_

#include <list>
#include <set>
#include <string>
#include <stack>
#include <unordered_map>
#include <vector>
#include "inc/graph_pass.h"

namespace ge {
/* Variable Initialize Flow, take as FrameworkOp
                  +-----------+
                  |   Merge   |
                  +-----------+
                  /           \
                0/             \x
                /               \
     +-----------+             +-----------+
     |  Switch   |             |  Switch   |
     +-----------+             +-----------+
      |         |F             T|         |
     0|         |               |        x|
      |         |               |         |
      |     +-----------------------+     |
      |     | IsVariableInitialized |     |
      |     +-----------------------+     |
      |                         |         |
      |                         |         |
      |                         |         |
  +-----------+                +-----------+
  |   Const   |                | VariableV2|
  +-----------+                +-----------+
*/

/* Switch branch op optimize, Switches in same case merge to one StreamSwitch, update following nodes' input

                                            +-----------+
                                          / |   task2   | \
                                        T/  +-----------+  \
        +-----------+     +-----------+ /                   \ +-----------+     +-----------+
        |   task1   | --> |  Switch   |                       |   task4   | --> |   noop    |
        +-----------+     +-----------+ \                   / +-----------+     +-----------+
                                        F\  +-----------+  /
                                          \ |   task3   | /
                                            +-----------+

                cond(x < y, lambda: add(x, z), lambda: square(y))

                    +-----------+                                                 +-----------+
                    |   Merge   |                                    +------------|StreamMerge|----------+
                    +-----------+                                    |            +-----------+          |
                    /           \                                    |                 |                 |
                   /             \                                   |c                |                 |c
                  /               \                             +----------+      -----------      +----------+
        +-----------+           +-----------+                   | Active_f |     /           \     | Active_t |
        |  Square   |           |    Add    |                   +----------+    /             \    +----------+
        +-----------+           +-----------+                         \        /               \       /
              /                  /         \                           \c     /                 \     /c
            y/                 x/           \z                        +-----------+         +-----------+
            /                  /             \                        |  Square   |         |    Add    |
   +-----------+     +-----------+        +-----------+               +-----------+         +-----------+
   |  Switch   |     |  Switch   |        |  Switch   |  ====>            /   |               /   |   \
   +-----------+     +-----------+        +-----------+                  /    |              /    |    \
    y|       |F       T|       |x          T|       |z            +--------+  |       +--------+  |  +--------+
     |       |         |       |            |       |             | y/read |  |       | x/read |  |  | z/read |
     |      +-----------+      |            |       |             +--------+  |       +--------+  |  +--------+
     |      |   Less    |-------------------+       |                         |c                  |c
     |      +-----------+      |                    |               +----------------+     +----------------+
     |                         |                    |               | StreamSwitch_f |     | StreamSwitch_t |
     |                         |                    |               +----------------+     +----------------+
 +-----------+         +-----------+      +-----------+                    |                      |
 |  y/read   |         |  x/read   |      |  z/read   |                    |     +-----------+    |
 +-----------+         +-----------+      +-----------+                    +-----|   Less    |----+
                                                                                 +-----------+
*/
class SwitchOpPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);
  Status ClearStatus() override;

 private:
  Status ReplaceSwitchNode(ComputeGraphPtr &graph, NodePtr &switch_node);

  Status ReplaceMergeNode(ComputeGraphPtr &graph, NodePtr &merge_node);

  NodePtr CreateStreamSwitchNode(ComputeGraphPtr &graph, const NodePtr &switch_node, const std::string &suffix,
                                 OutDataAnchorPtr &peer_cond_anchor);

  NodePtr CreateMemcpyAsyncNode(ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor, bool multi_batch_flag);

  Status CombineSwitchNode(ComputeGraphPtr &graph);

  NodePtr CreateActiveNode(ComputeGraphPtr &graph, NodePtr &node);

  Status AddMemcpyAsyncNodes(ComputeGraphPtr &graph, NodePtr &stream_merge_node, bool multi_batch_flag);

  Status BypassSwitchNode(NodePtr &switch_node, OutDataAnchorPtr &peer_data_anchor, OutDataAnchorPtr &peer_cond_anchor);

  Status FindSwitchCondInput(bool pass_switch_flag, OutDataAnchorPtr &peer_cond_anchor);

  Status MarkBranchs(OutDataAnchorPtr &peer_cond_anchor, NodePtr &stream_switch_node, bool true_branch_flag);

  NodePtr CreateCastOp(ComputeGraphPtr &graph, OutDataAnchorPtr &peer_cond_anchor);

  Status AddConstNode(ComputeGraphPtr &graph, NodePtr &stream_switch_node);

  Status UpdateCondBranch(NodePtr &node);

  Status UpdateAttachFlag(const NodePtr &node, std::string &stream_label, bool &merge_flag, bool &exit_flag,
                          bool &net_output_flag);

  Status UpdateLoopBranch(const std::stack<NodePtr> &enter_nodes, const std::string &stream_label);

  Status UpdateEnterNode();

  std::string CheckDuplicateName(const std::string &node_name);

  Status CheckCycleDependence(ComputeGraphPtr &graph);

  void MarkCycleDependence(const std::unordered_map<NodePtr, std::vector<NodePtr>> &cond_switch_map);

  Status ModifySwitchInCtlEdges(NodePtr &switch_node, NodePtr &cast_node, const std::set<NodePtr> &same_cond_switch);

  Status ModifySwitchOutCtlEdges(NodePtr &switch_node, NodePtr &stream_switch, NodePtr &active_node);

  void CopyControlEdges(NodePtr &old_node, NodePtr &new_node, bool input_check_flag = false);

  void RemoveControlEdges(NodePtr &node);

  void ReplaceControlEdges(NodePtr &old_node, NodePtr &new_node);

  int64_t GetGroupId(const NodePtr &node);

  void MarkHeadNodes(const NodePtr &node, const NodePtr &stream_switch);

  std::vector<NodePtr> switch_nodes_;
  std::vector<NodePtr> merge_nodes_;
  std::vector<NodePtr> enter_nodes_;
  std::unordered_map<NodePtr, std::set<std::string>> switch_cyclic_map_;

  std::set<NodePtr> bypass_nodes_;
  std::unordered_map<NodePtr, NodePtr> branch_head_nodes_;
  std::vector<NodePtr> stream_switch_nodes_;
  std::vector<NodePtr> need_label_nodes_;
  std::unordered_map<OutDataAnchorPtr, std::map<int64_t, std::vector<std::list<NodePtr>>>> cond_node_map_;
  std::unordered_map<NodePtr, std::set<std::string>> switch_node_map_;
  std::unordered_map<std::string, uint32_t> node_num_map_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_SWITCH_OP_PASS_H_
