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
#ifndef D_BASE_GRAPH_PARTITION_MERGE_ADJACENT_POLICY_H
#define D_BASE_GRAPH_PARTITION_MERGE_ADJACENT_POLICY_H

#include "graph/partition/base_merge_policy.h"
namespace ge {
class MergeAdjacentPolicy : public BaseMergePolicy {
 public:
  MergeAdjacentPolicy() = default;
  virtual ~MergeAdjacentPolicy() = default;
  // on the begin, record cluster min and max to rank
  graphStatus OnTurnBegin(const GraphPartitionerClusterDict &cluster_dict) override;
  // merge adjacent cluster
  bool TryMerge(const NodesCluster &src, const NodesCluster &dst) override;
  std::string PolicyDebugString(const NodesCluster &cluster) override {
    (void) cluster;
    return "";
  }
 private:
  std::unordered_map<const NodesCluster*, size_t> cluster_to_min_;
  std::unordered_map<const NodesCluster*, size_t> cluster_to_max_;
};
} // namespace ge
#endif  // D_BASE_GRAPH_PARTITION_MERGE_ADJACENT_POLICY_H
