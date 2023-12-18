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

#ifndef D_BASE_GRAPH_PARTITION_NODES_CLUSTER_H
#define D_BASE_GRAPH_PARTITION_NODES_CLUSTER_H

#include "graph/node.h"
#include "ge/ge_api_types.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
// abstract of subgraph
class NodesCluster {
 public:
  struct NodesClusterCmp {
    bool operator()(const NodesCluster *a, const NodesCluster *b) const {
      // AddInput & AddOutput has ensure a and b are not nullptr.
      return a->Id() < b->Id();
    }
  };
  NodesCluster(const NodePtr &node, const size_t id) : nodes_{node}, id_(id) {}
  virtual ~NodesCluster() = default;
  void MergeFrom(NodesCluster &from);
  void AddInput(NodesCluster &input);
  void AddOutput(NodesCluster &output);
  void RemoveInput(NodesCluster &input);
  void RemoveOutput(NodesCluster &output);
  const std::list<NodePtr> &Nodes() const;
  size_t Id() const {
    return id_;
  }
  const std::set<NodesCluster *, NodesClusterCmp> &Inputs() const {
    return inputs_;
  }
  const std::set<NodesCluster *, NodesClusterCmp> &Outputs() const {
    return outputs_;
  }
  std::string DebugString() const;
 private:
  std::list<NodePtr> nodes_; // including all nodes of this cluster
  std::set<NodesCluster *, NodesClusterCmp> inputs_; // including all inputs cluster of this cluster
  std::set<NodesCluster *, NodesClusterCmp> outputs_; // including all outputs cluster of this cluster
  size_t id_; // unique id of cluster
};
using NodesClusterPtr = std::shared_ptr<NodesCluster>;
} // namespace ge
#endif  // D_BASE_GRAPH_PARTITION_NODES_CLUSTER_H
