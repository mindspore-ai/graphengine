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

#ifndef GE_COMMON_SUBGRAPH_CONST_MIGRATION_H_
#define GE_COMMON_SUBGRAPH_CONST_MIGRATION_H_

#include "graph/types.h"
#include "inc/graph_pass.h"

#include <map>
#include <set>
#include <vector>
#include <string>

using std::set;
using std::map;

namespace ge {
class SubgraphConstMigrationPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  ///
  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_desc: functional OpDesc of Case.
  /// @param [out] graph_datas: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status ClassifyGraphNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                            map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                            map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes);

  ///
  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] node: Const node of subgraph.
  /// @param [in] func_desc: functional OpDesc of Case.
  /// @param [out] graph_nodes: Data groups of subgraph.
  /// @return true: SUCCESS / false: FAILED
  ///
  bool GetAssociatedNodes(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes,
                          const NodePtr &const_node, uint32_t &parent_index);

  ///
  /// @ingroup ge
  /// @brief Get all Data nodes for all subgraph.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] data_base: Data Node for migration.
  /// @param [in] data_idx: Data groups of subgraph.
  /// @param [in] data_idx: Data groups of subgraph.
  /// @return true: Same / false: not same
  ///
  bool IsParallelNodeSame(const map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                          const NodePtr &const_node, const string &node_key);

  ///
  /// @ingroup ge
  /// @brief Migration subgraph Node to Root
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] data_base: Data Node for migration.
  /// @param [in] node_key: Data groups of subgraph.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status GraphNodeMigration(const ComputeGraphPtr &graph, const NodePtr &func_node,
                            const map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                            map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes,
                            const NodePtr &const_node, const string &node_key);

  ///
  /// @ingroup ge
  /// @brief Move node to Parent graph.
  /// @param [in] graph: Root compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] anchor_idx: anchor index of move Node.
  /// @param [in] inputs: Parent index of Node input.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status MoveNodeToParent(const ComputeGraphPtr &graph, const NodePtr &func_node,
                          const map<ComputeGraphPtr, map<string, NodePtr>> &all_const_nodes,
                          const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes,
                          const string &node_key, uint32_t parent_index);

  ///
  /// @ingroup ge
  /// @brief Append Input Tensor for functional node.
  /// @param [in] graph_nodes: Const groups of subgraph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status AppendParallelNode(const NodePtr &func_node, uint32_t &parent_index,
                            map<ComputeGraphPtr, map<uint32_t, NodePtr>> &all_data_nodes);

  ///
  /// @ingroup ge
  /// @brief Delete Node from all subgraph.
  /// @param [in] graph_nodes: Data groups of subgraph.
  /// @param [in] detach: Node will move to parent.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status DetachParallelNode(const ComputeGraphPtr &graph, const map<string, NodePtr> &const_nodes,
                            const NodePtr &const_node, const NodePtr &data_node);

  ///
  /// @ingroup ge
  /// @brief Move Node to Parent Graph.
  /// @param [in] graph: Parent compute graph.
  /// @param [in] func_node: functional Node of Case.
  /// @param [in] attach: Node will move to parent.
  /// @param [in] inputs: Parent index of Node input.
  /// @param [in] outputs: Parent index of Node output.
  /// @return 0: SUCCESS / others: FAILED
  ///
  Status AttachParallelNode(const ComputeGraphPtr &graph, const NodePtr &func_node,
                            const NodePtr &const_node, uint32_t parent_index);
};
}  // namespace ge
#endif  // GE_COMMON_SUBGRAPH_CONST_MIGRATION_H_