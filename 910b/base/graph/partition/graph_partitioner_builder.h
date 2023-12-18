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

#ifndef D_BASE_GRAPH_PARTITION_GRAPH_PARTITIONER_BUILDER_H
#define D_BASE_GRAPH_PARTITION_GRAPH_PARTITIONER_BUILDER_H

#include "graph/compute_graph.h"
#include "graph/partition/base_merge_policy.h"
#include "graph/partition/base_subgraph_build_policy.h"
#include "graph/partition/graph_partitioner.h"

namespace ge {
// producer of partitioner
class GraphPartitionerBuilder {
 public:
  explicit GraphPartitionerBuilder(const ComputeGraphPtr &root_graph) : root_graph_(root_graph) {};
  GraphPartitionerBuilder &AppendMergePolicy(std::unique_ptr<BaseMergePolicy> &merge_policy);
  GraphPartitionerBuilder &SetSubgraphBuilderPolicy(std::unique_ptr<BaseSubgraphBuildPolicy> &build_policy);
  // create partitioner
  std::unique_ptr<GraphPartitioner> Build(const std::string &partitioner_name);
 private:
  ComputeGraphPtr root_graph_;
  // temporary own, when call Build ownership transfer
  std::vector<std::unique_ptr<BaseMergePolicy>> merge_policies_;
  std::unique_ptr<BaseSubgraphBuildPolicy> build_policy_;
};
} // namespace ge
#endif  // D_BASE_GRAPH_PARTITION_GRAPH_PARTITIONER_BUILDER_H
