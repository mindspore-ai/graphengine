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

#ifndef GE_GRAPH_PASSES_ATOMIC_ADDR_CLEAN_PASS_H_
#define GE_GRAPH_PASSES_ATOMIC_ADDR_CLEAN_PASS_H_

#include <vector>

#include "graph/graph.h"
#include "inc/graph_pass.h"

namespace ge {
class AtomicAddrCleanPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  ///
  /// HandleLoopGraph
  /// @param graph
  /// @return
  ///
  Status HandleLoopGraph(ComputeGraphPtr &graph, const vector<NodePtr> &atomic_node_vec);
  ///
  /// HandleNormalGraph
  /// @param graph
  /// @return
  ///
  Status HandleNormalGraph(ComputeGraphPtr &graph, const vector<NodePtr> &atomic_node_vec);
  ///
  /// Insert atomic clean node to graph
  /// @param graph
  /// @return
  ///
  NodePtr InsertAtomicAddrCleanNode(ComputeGraphPtr &graph);

  ///
  /// Link control anchor from atomic clean node to atomic node
  /// @param atomic_node
  /// @param atomic_clean_node
  /// @return
  ///
  Status LinkToAtomicNode(const NodePtr &atomic_node, NodePtr &atomic_clean_node);

  ///
  /// Check if this node is atomic op.
  /// @param node
  /// @return
  ///
  bool IsAtomicOp(const NodePtr &node);

  vector<NodePtr> hcom_node_vec_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_ATOMIC_ADDR_CLEAN_PASS_H_
