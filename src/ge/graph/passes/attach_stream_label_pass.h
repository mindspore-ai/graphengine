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

#ifndef GE_GRAPH_PASSES_ATTACH_STREAM_LABEL_PASS_H_
#define GE_GRAPH_PASSES_ATTACH_STREAM_LABEL_PASS_H_

#include <stack>
#include "inc/graph_pass.h"

namespace ge {
class AttachStreamLabelPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

  ///
  /// @brief Clear Status, used for subgraph pass
  /// @return
  ///
  Status ClearStatus() override;

 private:
  ///
  /// @brief Find StreamSwitch / StreamMerge / Enter node
  /// @param [in] graph
  /// @return void
  ///
  void FindNodes(const ComputeGraphPtr &graph);

  ///
  /// @brief update cond branch
  /// @param [in] node
  /// @return Status
  ///
  Status UpdateCondBranch(const NodePtr &node);

  ///
  /// @brief attach flag
  /// @param [in] node
  /// @param [out] stream_label
  /// @param [out] merge_flag
  /// @param [out] exit_flag
  /// @param [out] net_output_flag
  /// @return Status
  ///
  static Status AttachFlag(const NodePtr &node, std::string &stream_label, bool &merge_flag, bool &exit_flag,
                           bool &net_output_flag);

  ///
  /// @brief Update stream_label for loop_branch
  /// @param [in] enter_nodes
  /// @param [in] stream_label
  /// @return Status
  ///
  static Status UpdateLoopBranch(const std::stack<NodePtr> &enter_nodes, const std::string &stream_label);

  ///
  /// @brief Update stream_label start with enter nodes
  /// @return Status
  ///
  Status UpdateEnterNode();

  ///
  /// @brief Set stream_label for enter_nodes
  /// @param [in] enter_nodes
  /// @param [in] active_node
  /// @return Status
  ///
  static Status SetEnterLabel(const std::vector<NodePtr> &enter_nodes, const NodePtr &active_node);

  std::vector<NodePtr> stream_switch_nodes_;
  std::vector<NodePtr> need_label_nodes_;
  std::vector<NodePtr> enter_nodes_;
  std::unordered_map<NodePtr, NodePtr> branch_head_nodes_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_ATTACH_STREAM_LABEL_PASS_H_
