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

#ifndef D_BASE_GRAPH_PARTITION_GRAPH_PARTITIONER_H
#define D_BASE_GRAPH_PARTITION_GRAPH_PARTITIONER_H

#include "graph/compute_graph.h"
#include "graph/partition/base_merge_policy.h"
#include "graph/partition/base_subgraph_builder.h"
#include "graph/partition/base_subgraph_build_policy.h"

namespace ge {
// who need to partition subgraphs
class GraphPartitioner {
 public:
  GraphPartitioner(const std::string &partitioner_name,
                   const ComputeGraphPtr &root_graph,
                   std::unique_ptr<BaseSubgraphBuildPolicy> &builder_policy,
                   std::vector<std::unique_ptr<BaseMergePolicy>> &merge_policies) :
      root_graph_(root_graph),
      cluster_dict_(partitioner_name) {
    build_policy_ = std::move(builder_policy);
    for (auto &policy : merge_policies) {
      merge_policies_.emplace_back(std::move(policy));
    }
  };
  // Partition including InitClusters, InvokeOnTurnBeginPolicy, MergeClusters, SortClusters, BuildSubgraphs, five parts
  graphStatus Partition();
 private:
  void DebugMergeLog() const;
  graphStatus InitClusters();
  graphStatus InvokeOnTurnBeginPolicy();
  graphStatus MergeClusters();
  graphStatus SortClustersForBuildSubgraph();
  graphStatus BuildSubgraphs();
  ComputeGraphPtr root_graph_;
  // builder for subgraph of root_graph, and modify root_graph
  std::unique_ptr<BaseSubgraphBuilder> subgraph_builder_;
  // support multi merge policies
  std::vector<std::unique_ptr<BaseMergePolicy>> merge_policies_;
  // support only one build policy for build subgraph
  std::unique_ptr<BaseSubgraphBuildPolicy> build_policy_;
  // dictionary for node->cluster
  GraphPartitionerClusterDict cluster_dict_;
  size_t turn_ = 0UL;
};
} // namespace ge
#endif  // D_BASE_GRAPH_PARTITION_GRAPH_PARTITIONER_H
