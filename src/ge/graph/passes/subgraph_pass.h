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

#ifndef GE_GRAPH_PASSES_SUBGRAPH_PASS_H_
#define GE_GRAPH_PASSES_SUBGRAPH_PASS_H_

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "graph/types.h"
#include "inc/graph_pass.h"

namespace ge {
class SubgraphPass : public GraphPass {
 public:
  /**
   * @ingroup ge
   * @brief Subgraph optimizer.
   * @param [in] graph: Input ComputeGraph
   * @return: 0 for success / others for fail
   */
  Status Run(ComputeGraphPtr graph) override;

 private:
  /**
   * @ingroup ge
   * @brief Check Subgraph Data node.
   * @param [in] graph: ComputeGraph.
   * @param [in] node: NetOutput node in Subgraph.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status SubgraphInputNode(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Check Subgraph NetOutput node.
   * @param [in] graph: ComputeGraph.
   * @param [in] node: NetOutput node in Subgraph.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status SubgraphOutputNode(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Check is Input->While and Input link to other nodes
   * @param [in] graph: ComputeGraph.
   * @param [in] node: While node.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status WhileInputNodes(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Check body subgraph of While op
   * @param [in] graph: ComputeGraph.
   * @param [in] node: While node.
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status WhileBodySubgraph(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Get body subgraph of While op
   * @param [in] graph: ComputeGraph.
   * @param [in] node: While node.
   * @return: body subgraph
   */
  ComputeGraphPtr GetWhileBodySubgraph(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Mark output parent_node_index
   * @param [in] peer_out_anchor: peer_out_anchor of NetOutput
   * @param [in] index: parent_node_index of NetOutput
   * @param [out] node_to_attr_index: key for node in subgraph, value for parent_node_index
   * @return: void
   */
  void MarkOutputIndex(const OutDataAnchorPtr &peer_out_anchor, uint32_t index,
                       std::unordered_map<NodePtr, std::vector<uint32_t>> &node_to_attr_index);

  /**
   * @ingroup ge
   * @brief Get data_nodes / input_indexes of netoutput if need insert memcpy
   * @param [in] node_to_attr_index: key for node in subgraph, value for parent_node_index
   * @param [out] data_nodes: data_nodes need insert memcpy
   * @param [out] netoutput_input_indexes: input_indexes of netoutput need insert memcpy
   * @return: void
   */
  void GetExchangeInOut(const std::unordered_map<NodePtr, std::vector<uint32_t>> &node_to_attr_index,
                        std::set<NodePtr> &data_nodes, std::set<uint32_t> &netoutput_input_indexes);

  /**
   * @ingroup ge
   * @brief Insert memcpy node in while_body
   * @param [in] graph: while_body
   * @param [in] data_nodes: data_nodes need insert memcpy
   * @param [in] output_node: NetOutput in while_body
   * @param [in] netoutput_input_indexes: input_indexes of netoutput need insert memcpy
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status InsertMemcpyInWhileBody(const ComputeGraphPtr &graph, const std::set<NodePtr> &data_nodes,
                                 const NodePtr &output_node, const std::set<uint32_t> &netoutput_input_indexes);

  /**
   * @ingroup ge
   * @brief Insert NoOp node between memcpy_nodes and loop_body_nodes
   * @param [in] graph: while_body
   * @param [in] memcpy_nodes
   * @param [in] loop_body_nodes
   * @return: 0 for SUCCESS / others for FAILED
   */
  Status InsertNoOp(const ComputeGraphPtr &graph, const std::set<NodePtr> &memcpy_nodes,
                    const std::set<NodePtr> &loop_body_nodes);

  /**
   * @ingroup ge
   * @brief Check is Data->NetOutput in while body
   * @param [in] in_data_anchor
   * @return: true for Data->NetOutput in while body / false for others
   */
  bool IsWhileBodyOutput(const InDataAnchorPtr &in_data_anchor);

  /**
   * @ingroup ge
   * @brief Check is AtomicOp->NetOutput
   * @param [in] node
   * @param [in] out_index
   * @return: true for AtomicOp->NetOutput / false for others
   */
  bool IsAtomicRequired(const NodePtr &node, int64_t out_index);

  /**
   * @ingroup ge
   * @brief Check is OutputContinuesRequiredOp->NetOutput
   * @param [in] node
   * @return: true for OutputContinuesRequiredOp->NetOutput / false for others
   */
  bool IsOutputContinuesRequired(const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Check is InputContinuesRequiredOp->NetOutput
   * @param [in] node
   * @return: true for InputContinuesRequiredOp->NetOutput / false for others
   */
  bool IsInputContinuesRequired(const NodePtr &node);

  /**
   * @ingroup ge
   * @brief Insert memcpy node
   * @param [in] graph
   * @param [in] out_anchor
   * @param [in] in_anchors
   * @param [in] name
   * @return: 0 for success / others for fail
   */
  Status InsertMemcpyNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                          const std::vector<InDataAnchorPtr> &in_anchors, const std::string &name);

  // Append index for new memcpy node.
  uint32_t memcpy_num_{0};
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_SUBGRAPH_PASS_H_
