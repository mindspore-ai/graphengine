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

#include "graph/partition/merge_adjacent_policy.h"
#include <queue>
namespace ge {
graphStatus MergeAdjacentPolicy::OnTurnBegin(const GraphPartitionerClusterDict &cluster_dict) {
  for (const auto &cluster : cluster_dict.GetAllClusters()) {
    cluster_to_min_.emplace(cluster.get(), cluster->Id());
    cluster_to_max_.emplace(cluster.get(), cluster->Id());
  }
  return GRAPH_SUCCESS;
}
// CanMerge + OnMergeCluster = TryMerge
bool MergeAdjacentPolicy::TryMerge(const NodesCluster &src, const NodesCluster &dst) {
  std::queue<const NodesCluster *> forward_reached;
  forward_reached.push(&src);
  size_t dst_max = cluster_to_max_[&dst];
  // Try merge other cluster to this cluster, ONLY if will not leads to a ring
  while (!forward_reached.empty()) {
    auto current_cluster = forward_reached.front();
    forward_reached.pop();
    for (const auto &cluster : current_cluster->Outputs()) {
      if (cluster_to_max_[cluster] == dst_max && current_cluster != &src) {
        return false;
      } else if (cluster_to_min_[cluster] < dst_max) {
        forward_reached.push(cluster);
      }
    }
  }
  cluster_to_max_[&src] = src.Id() > dst.Id() ? src.Id() : dst.Id();
  cluster_to_min_[&src] = src.Id() < dst.Id() ? src.Id() : dst.Id();
  return true;
}
}