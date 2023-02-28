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

#include "graph/partition/graph_partitioner_cluster_dict.h"
namespace ge {
void GraphPartitionerClusterDict::AddCluster(const NodesClusterPtr &cluster) {
  clusters_.emplace_back(cluster);
}

void GraphPartitionerClusterDict::SetNodeClusterPair(const NodePtr &node, const NodesClusterPtr &cluster) {
  nodes_2_cluster_[node] = cluster;
}

NodesClusterPtr GraphPartitionerClusterDict::GetNodeCluster(const NodePtr &node) const {
  const auto &pair = nodes_2_cluster_.find(node);
  if (pair == nodes_2_cluster_.cend()) {
    return nullptr;
  }
  return pair->second;
}
} // namespace ge