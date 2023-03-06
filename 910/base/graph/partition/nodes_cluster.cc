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

#include "graph/partition/nodes_cluster.h"
#include <queue>
#include <sstream>
namespace ge {
void NodesCluster::AddInput(NodesCluster &input) {
  if (std::find(inputs_.cbegin(), inputs_.cend(), &input) != inputs_.cend()) {
    return;
  }
  inputs_.insert(&input);
  if (std::find(input.outputs_.cbegin(), input.outputs_.cend(), this) != input.outputs_.cend()) {
    return;
  }
  input.outputs_.insert(this);
}

void NodesCluster::AddOutput(NodesCluster &output) {
  if (std::find(outputs_.cbegin(), outputs_.cend(), &output) != outputs_.cend()) {
    return;
  }
  outputs_.insert(&output);
  if (std::find(output.inputs_.cbegin(), output.inputs_.cend(), this) != output.inputs_.cend()) {
    return;
  }
  output.inputs_.insert(this);
}

void NodesCluster::MergeFrom(NodesCluster &from) {
  nodes_.insert(nodes_.cend(), from.nodes_.cbegin(), from.nodes_.cend());
  from.inputs_.erase(this);
  from.outputs_.erase(this);
  inputs_.erase(&from);
  outputs_.erase(&from);
  auto in_clusters = from.inputs_;
  for (const auto &cluster : in_clusters) {
    cluster->RemoveOutput(from);
    cluster->AddOutput(*this);
  }
  auto out_clusters = from.outputs_;
  for (const auto &cluster : out_clusters) {
    cluster->RemoveInput(from);
    cluster->AddInput(*this);
  }
}

void NodesCluster::RemoveInput(NodesCluster &input) {
  inputs_.erase(&input);
  input.outputs_.erase(this);
}

void NodesCluster::RemoveOutput(NodesCluster &output) {
  outputs_.erase(&output);
  output.inputs_.erase(this);
}

const std::list<NodePtr> &NodesCluster::Nodes() const {
  return nodes_;
}

std::string NodesCluster::DebugString() const {
  std::stringstream ss;
  ss << "id:" << id_ << ", node_size:" << nodes_.size() << ",";
  ss << " inputs:[";
  for (const auto &cluster : inputs_) {
    ss << cluster->id_ << ",";
  }
  ss << "] outputs:[";
  for (const auto &cluster : outputs_) {
    ss << cluster->id_ << ",";
  }
  ss << "] nodes:|";
  for (const auto &node : nodes_) {
    ss << (node->GetName() + "|");
  }
  return ss.str();
}
} // namespace ge