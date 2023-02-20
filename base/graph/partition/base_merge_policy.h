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

#ifndef D_BASE_GRAPH_PARTITION_MERGE_POLICY_H
#define D_BASE_GRAPH_PARTITION_MERGE_POLICY_H

#include "graph/compute_graph.h"
#include "graph/partition/graph_partitioner_cluster_dict.h"

namespace ge {
// policy of merge clusters
class BaseMergePolicy {
 public:
  BaseMergePolicy() = default;
  virtual ~BaseMergePolicy() = default;
  // define function on the begin of this merge turn
  virtual graphStatus OnTurnBegin(const GraphPartitionerClusterDict &cluster_dict) {
    (void) cluster_dict;
    return SUCCESS;
  }
  // CanMerge + OnMergeCluster = TryMerge
  virtual bool TryMerge(const NodesCluster &src, const NodesCluster &dst) = 0;
  virtual std::string PolicyDebugString(const NodesCluster &cluster) {
    (void) cluster;
    return "";
  }
};
} // namespace ge

#endif  // D_BASE_GRAPH_PARTITION_MERGE_POLICY_H
