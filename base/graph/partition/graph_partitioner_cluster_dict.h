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

#ifndef D_BASE_GRAPH_PARTITION_GRAPH_PARTITIONER_CLUSTER_DICT_H
#define D_BASE_GRAPH_PARTITION_GRAPH_PARTITIONER_CLUSTER_DICT_H

#include "graph/node.h"
#include "graph/partition/nodes_cluster.h"

namespace ge {
// who manage partitioner clusters map
class GraphPartitionerClusterDict {
 public:
  explicit GraphPartitionerClusterDict(const std::string &partitioner_name) : partitioner_name_(partitioner_name) {};
  ~GraphPartitionerClusterDict() = default;
  void AddCluster(const NodesClusterPtr &cluster);
  void SetNodeClusterPair(const NodePtr &node, const NodesClusterPtr &cluster);
  NodesClusterPtr GetNodeCluster(const NodePtr &node) const;
  const std::vector<NodesClusterPtr> &GetAllClusters() const {
    return clusters_;
  }
  void SwapClusters(std::vector<NodesClusterPtr> &swap_clusters) {
    clusters_.swap(swap_clusters);
  }
  const std::string &GetPartitionerName() const {
    return partitioner_name_;
  }
 private:
  // defined for clusters merge
  std::vector<NodesClusterPtr> clusters_;
  std::unordered_map<NodePtr, NodesClusterPtr> nodes_2_cluster_;
  std::string partitioner_name_;
};
} // namespace ge
#endif  // D_BASE_GRAPH_PARTITION_GRAPH_PARTITIONER_CLUSTER_DICT_H
