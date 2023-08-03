/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef ROOT_GRAPH_BUILDER__H
#define ROOT_GRAPH_BUILDER__H
#include <string>
#include <map>
#include <vector>
#include "graph/compute_graph.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
class RootGraphBuilder {
 public:
  explicit RootGraphBuilder(std::vector<ComputeGraphPtr> subgraphs);
  ~RootGraphBuilder() = default;
  Status BuildRootGraph();

  const ComputeGraphPtr GetComputeGraphPtr() const {return root_graph_;}
 private:
  Status MakePartitionedCallNode(const ComputeGraphPtr &subgraph);
  Status ConnectPartionedCallNodesWithRelation();
  Status ConnectPartionedCallNodesWithoutRelation();
  Status ConnectInputEdge(const std::map<std::string, std::map<uint32_t, uint32_t>> &input_name_to_indices,
                          const NodePtr curr_partitioned_call);
  Status ConnectInputEdge();
  Status ConnectOutputEdge(const std::map<std::string, std::map<uint32_t, uint32_t>> &output_name_to_indices,
                           const NodePtr curr_partitioned_call);
  Status CreateDataNode(const GeTensorDesc &input_desc, NodePtr &data_node, const uint32_t data_index);
  Status CreatNetoutput();

  ComputeGraphPtr root_graph_;
  NodePtr netoutput_node_;
  std::vector<ComputeGraphPtr> subgraphs_;
  std::map<std::string, NodePtr> src_model_name_to_partitioned_nodes_;
  std::map<int32_t, ge::GeTensorDesc> idx_to_output_desc_;
  std::map<int32_t, DataAnchorPtr> idx_to_output_anchor_;
  // for data parallel
  std::vector<std::vector<DataAnchorPtr>> all_input_anchors_;
  std::vector<std::vector<GeTensorDesc>> all_input_desc_;
  std::vector<std::vector<DataAnchorPtr>> all_output_anchors_;
  std::vector<std::vector<GeTensorDesc>> all_output_desc_;
};
}
#endif  // ROOT_GRAPH_BUILDER__H