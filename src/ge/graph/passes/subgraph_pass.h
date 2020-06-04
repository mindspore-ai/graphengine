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
   * @brief Check is data->netoutput in while body
   * @param [in] in_data_anchor
   * @return: true for data->netoutput in while body / for false for others
   */
  bool IsWhileBodyOutput(const InDataAnchorPtr &in_data_anchor);

  /**
   * @ingroup ge
   * @brief Insert memcpy node
   * @param [in] graph
   * @param [in] out_anchor
   * @param [in] in_anchor
   * @param [in] name
   * @return: 0 for success / others for fail
   */
  Status InsertMemcpyNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                          const InDataAnchorPtr &in_anchor, const std::string &name);

  // Append index for new memcpy node.
  uint32_t memcpy_num_{0};
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_SUBGRAPH_PASS_H_
