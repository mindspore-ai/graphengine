/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#ifndef INC_39BC87BB1C574BBB96A3716A54964F5F_H
#define INC_39BC87BB1C574BBB96A3716A54964F5F_H

#include <functional>
#include "ge/ge_api_types.h"
#include "graph/compute_graph.h"

namespace gert {
class GraphUnfolder {
 public:
  static ge::Status UnfoldSubgraphs(const ge::ComputeGraphPtr &root_graph, ge::ComputeGraphPtr &merged_graph);

  static bool IsGraphNeedUnfold(const ge::ComputeGraphPtr &root_graph);
  static bool IsGraphDynamicCompiled(const ge::ComputeGraphPtr &graph);

  static bool IsDirectInputNode(const ge::NodePtr &node, const ge::ComputeGraphPtr &graph);

 private:
    static ge::Status DoUnlinkDataAnchors(const ge::OutDataAnchorPtr &out_data_anchor,
                                          const ge::InDataAnchorPtr &in_data_anchor);
    static ge::Status DoLinkDataAnchors(const ge::OutDataAnchorPtr &out_data_anchor,
                                        const ge::InDataAnchorPtr &in_data_anchor);
    static ge::Status MergeInputNodes(ge::ComputeGraph &compute_graph);
    static ge::Status MergeInputInData(const ge::NodePtr &node, const ge::NodePtr &wrapped_node,
                                       std::set<ge::NodePtr> &root_nodes);
    static ge::Status MergeNetOutputNode(ge::ComputeGraph &compute_graph);
    static ge::Status MergeNetOutputInData(const ge::NodePtr &parent_node, const ge::OpDescPtr &net_output_desc,
                                           const ge::InDataAnchorPtr &in_data_anchor);
    static ge::Status UnfoldSubgraph(const ge::ComputeGraphPtr &root_graph,
                                     const ge::ComputeGraphPtr &origin_sub_graph,
                                     ge::ComputeGraphPtr &merged_graph,
                                     const uint32_t depth = 0U);
    static ge::Status UnfoldPartitionedCallSubgraph(const ge::ComputeGraphPtr &sub_graph,
                                                    ge::ComputeGraphPtr &merged_graph,
                                                    const ge::ComputeGraphPtr &root_graph,
                                                    const ge::NodePtr &node,
                                                    const uint32_t &depth);
    static ge::Status UnfoldControlNodeSubgraph(const std::vector<ge::ComputeGraphPtr> &subgraphs,
                                                const ge::ComputeGraphPtr &root_graph,
                                                const ge::NodePtr &node,
                                                const uint32_t &depth);
    static ge::Status MarkGraphNodeIndex(const ge::ComputeGraphPtr &merged_graph);
};
}

#endif
