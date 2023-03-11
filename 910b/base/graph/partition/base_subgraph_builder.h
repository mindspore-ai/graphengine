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

#ifndef D_BASE_GRAPH_PARTITION_SUBGRAPH_BUILDER_H
#define D_BASE_GRAPH_PARTITION_SUBGRAPH_BUILDER_H

#include "graph/compute_graph.h"
#include "graph/partition/base_merge_policy.h"
#include "graph/partition/base_subgraph_build_policy.h"

namespace ge {
// who build subgraph
class BaseSubgraphBuilder {
 public:
  BaseSubgraphBuilder() = default;
  virtual ~BaseSubgraphBuilder() = default;
  virtual Status BuildSubgraph() = 0;
  // Set all members
  BaseSubgraphBuilder *SetRootGraph(const ComputeGraphPtr &root_graph) {
    root_graph_ = root_graph;
    return this;
  };
  BaseSubgraphBuilder *SetClustersDict(const GraphPartitionerClusterDict *clusters_dict) {
    clusters_dict_ = clusters_dict;
    return this;
  }
  BaseSubgraphBuilder *SetSubgraphBuilderPolicy(const BaseSubgraphBuildPolicy *build_policy) {
    build_policy_ = build_policy;
    return this;
  }
  BaseSubgraphBuilder *SetNodesCluster(const NodesClusterPtr &cluster) {
    cluster_ = cluster;
    return this;
  }
  const NodesClusterPtr &GetCluster() const {
    return cluster_;
  }
  const ComputeGraphPtr &GetRootGraph() const {
    return root_graph_;
  }
  NodesClusterPtr GetNodeCluster(const NodePtr &node) const {
    if (clusters_dict_ != nullptr) {
      return clusters_dict_->GetNodeCluster(node);
    }
    return nullptr;
  }
  const std::string &GetPartitionerName() const {
    return clusters_dict_->GetPartitionerName();
  }
  bool NeedMergeInputs() const {
    if (build_policy_ != nullptr) {
      return build_policy_->NeedMergeInputs();
    }
    return false;
  }
  bool IsNeedBuildSubgraph(const NodesClusterPtr &cluster) const {
    if (build_policy_ != nullptr) {
      return build_policy_->IsNeedBuildSubgraph(cluster);
    }
    return false;
  }
 private:
  NodesClusterPtr cluster_{nullptr};
  ComputeGraphPtr root_graph_{nullptr};
  // not own, own by GraphPartitioner
  const BaseSubgraphBuildPolicy *build_policy_{nullptr};
  // not own, need partitioner_->GetCluster(node), when build relationship of clusters
  const GraphPartitionerClusterDict *clusters_dict_{nullptr};
};
}  // namespace ge
#endif  // D_BASE_SUBGRAPH_BUILDER_H
