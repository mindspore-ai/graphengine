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

#ifndef D_BASE_GRAPH_PARTITION_SUBGRAPH_BUILDER_POLICY_H
#define D_BASE_GRAPH_PARTITION_SUBGRAPH_BUILDER_POLICY_H

#include "graph/partition/nodes_cluster.h"

namespace ge {
enum class PartitionSubgraphType {
  kDefaultSubgraph = 0,
  kSubgraphForOptimization,
  kUnsupportSubgraphType
};
// policy of build subgraphs
class BaseSubgraphBuildPolicy {
 public:
  BaseSubgraphBuildPolicy(const PartitionSubgraphType subgraph_type, const bool need_merge_inputs)
      : subgraph_type_(subgraph_type),
        need_merge_inputs_(need_merge_inputs) {}
  virtual ~BaseSubgraphBuildPolicy() = default;
  virtual bool IsNeedBuildSubgraph(const NodesClusterPtr &cluster) const {
    (void) cluster;
    return true;
  }
  // 生成子图名的策略
  virtual std::string GetSubgraphName(const NodesClusterPtr &cluster) const = 0;
  bool NeedMergeInputs() const {
    return need_merge_inputs_;
  }
  PartitionSubgraphType GetSubgraphType() const {
    return subgraph_type_;
  }
 private:
  PartitionSubgraphType subgraph_type_;
  bool need_merge_inputs_;
};
}  // namespace ge

#endif  // D_BASE_SUBGRAPH_BUILDER_POLICY_H
