/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "graph/partition/dynamic_shape_partition.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

#define REQUIRE(cond, ...)                                     \
  do {                                                         \
    if (!(cond)) {                                             \
      GELOGE(FAILED, "[Dynamic shape partition]" __VA_ARGS__); \
      return FAILED;                                           \
    }                                                          \
  } while (0)

#define REQUIRE_NOT_NULL(cond, ...) REQUIRE(((cond) != nullptr), __VA_ARGS__)
#define REQUIRE_SUCCESS(cond, ...) REQUIRE(((cond) == SUCCESS), __VA_ARGS__)
#define REQUIRE_GRAPH_SUCCESS(cond, ...) REQUIRE(((cond) == GRAPH_SUCCESS), __VA_ARGS__)

bool IsExperimental() {
  const static bool kIsExperimental = (std::getenv("EXPERIMENTAL_DYNAMIC_PARTITION") != nullptr);
  return kIsExperimental;
}

namespace ge {
using Cluster = DynamicShapePartitioner::Cluster;
using ClusterPtr = std::shared_ptr<Cluster>;

Status DynamicShapePartitioner::Partition() {
  REQUIRE_NOT_NULL(root_graph_, "Graph is nullptr.");
  if (!IsExperimental()) {
    GELOGD("Skip dynamic shape partition as not in experimental mode.");
    REQUIRE(AttrUtils::SetBool(*root_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, false),
            "Failed set dynamic shape partitioned flag on root graph.");
    return SUCCESS;
  }

  GELOGD("Start dynamic shape partition graph %s.", root_graph_->GetName().c_str());
  REQUIRE_SUCCESS(MarkUnknownShapeNodes(), "Failed mark unknown shape nodes, root grah name:%s.",
                  root_graph_->GetName().c_str());
  if (unknown_shape_nodes_.empty()) {
    GELOGD("Skip dynamic shape partition of graph %s as all nodes are known shape.", root_graph_->GetName().c_str());
    REQUIRE(AttrUtils::SetBool(*root_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, false),
            "Failed set dynamic shape partitioned flag on root graph %s.", root_graph_->GetName().c_str());
    return SUCCESS;
  }
  REQUIRE(AttrUtils::SetBool(*root_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true),
          "Failed set dynamic shape partitioned flag on root graph %s.", root_graph_->GetName().c_str());

  DumpGraph("_Before_DSP");
  auto status = PartitionImpl();
  GELOGD("%s.", DebugString().c_str());
  if (status != SUCCESS) {
    GELOGE(status, "Failed dynamic shape partition graph: %s, status:\n %s", root_graph_->GetName().c_str(),
           DebugString().c_str());
  }
  DumpGraph("_After_DSP");
  GELOGD("Finish dynamic shape partition graph %s.", root_graph_->GetName().c_str());
  ClearResource();
  return status;
}

Status DynamicShapePartitioner::PartitionImpl() {
  REQUIRE_SUCCESS(root_graph_->TopologicalSorting(), "Graph topological sort failed.");
  REQUIRE_SUCCESS(InitClusters(), "Failed init cluster nodes.");
  REQUIRE_SUCCESS(MergeClusters(), "Failed merge clusters.");
  PruneUniqueClusters();
  REQUIRE_SUCCESS(BuildPartitionFrame(), "Failed build cluster partition frame.");
  REQUIRE_SUCCESS(CombinePartitionFrame(), "Failed combine cluster partition frame.");
  REQUIRE_SUCCESS(BuildPartitionSubgraph(), "Failed build cluster partition subgraph.");
  return SUCCESS;
}

void DynamicShapePartitioner::PruneUniqueClusters() {
  for (auto &node : root_graph_->GetDirectNode()) {
    auto cluster = node_2_cluster_[node];
    if (unique_clusters_.count(cluster) != 0) {
      continue;
    }
    unique_clusters_.insert(cluster);
  }
}

Status DynamicShapePartitioner::BuildPartitionFrame() {
  for (const auto &cluster : unique_clusters_) {
    REQUIRE_SUCCESS(cluster->BuildFrame(), "Failed build frame of cluster[%lu].", cluster->Id());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::CombinePartitionFrame() {
  for (const auto &cluster : unique_clusters_) {
    REQUIRE_SUCCESS(cluster->CombinePartitionFrame(), "Failed combine frame of cluster[%lu].", cluster->Id());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::BuildPartitionSubgraph() {
  for (const auto &cluster : unique_clusters_) {
    REQUIRE_SUCCESS(cluster->BuildPartitionSubgraph(), "Failed build subgraph of cluster[%lu].", cluster->Id());
  }
  return SUCCESS;
}

std::string DynamicShapePartitioner::DebugString() const {
  size_t unknown = 0;
  size_t known = 0;
  size_t data = 0;
  size_t netoutput = 0;
  std::stringstream ss;
  ss << "All unknown shape nodes:" << std::endl;
  for (const auto &node : unknown_shape_nodes_) {
    ss << "  [" << node->GetName() << "](" << node->GetType() << ")" << std::endl;
  }
  for (const auto &cluster : unique_clusters_) {
    if (cluster->IsUnknownShape()) {
      unknown++;
    } else if (cluster->IsKnownShape()) {
      known++;
    } else if (cluster->IsData()) {
      data++;
    } else if (cluster->IsNetOutput()) {
      netoutput++;
    }
  }
  ss << "All clusters:" << unique_clusters_.size() << ", data:" << data << ", known:" << known
     << ", unknown:" << unknown << ", netoutput:" << netoutput << std::endl;
  for (const auto &cluster : unique_clusters_) {
    ss << "  " << cluster->DebugString() << std::endl;
  }
  return ss.str();
}

void DynamicShapePartitioner::DumpGraph(const std::string &suffix) {
  GraphUtils::DumpGEGraphToOnnx(*root_graph_, root_graph_->GetName() + suffix);
  for (const auto &sub_graph : root_graph_->GetAllSubgraphs()) {
    GraphUtils::DumpGEGraphToOnnx(*sub_graph, sub_graph->GetName() + suffix);
  }
}

void DynamicShapePartitioner::ClearResource() {
  for (const auto &cluster : unique_clusters_) {
    cluster->Clear();
  }
  node_2_cluster_.clear();
  ordered_cluster_.clear();
  unique_clusters_.clear();
  unknown_shape_nodes_.clear();
  root_graph_.reset();
}

Status DynamicShapePartitioner::MarkUnknownShapeNodes() {
  for (auto &node : root_graph_->GetDirectNode()) {
    REQUIRE_SUCCESS(CollectSpreadUnknownShapeNodes(node), "Failed collect spread unknown shape nodes %s.",
                    node->GetName().c_str());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::InitClusters() {
  auto graph = root_graph_;
  size_t rank = 0;
  for (const auto &node : graph->GetDirectNode()) {
    Cluster::Type type = Cluster::DATA;
    if (node->GetType() == DATA) {
      type = Cluster::DATA;
    } else if (node->GetType() == NETOUTPUT) {
      type = Cluster::NETOUTPUT;
    } else if (unknown_shape_nodes_.count(node) > 0) {
      type = Cluster::UNKNOWN_SHAPE;
    } else {
      type = Cluster::KNOWN_SHAPE;
    }
    auto cluster = MakeShared<Cluster>(rank++, type, node, this);
    REQUIRE_NOT_NULL(cluster, "Failed new memory for cluster.");
    node_2_cluster_[node] = cluster;
    if (cluster->IsUnknownShape()) {
      ordered_cluster_.push_back(cluster);
    }
    // Already sorted topologically, so access to the parent cluster is safe
    for (const auto &parent : node->GetInAllNodes()) {
      cluster->AddInput(node_2_cluster_[parent]);
    }
  }
  for (const auto &node : graph->GetDirectNode()) {
    GELOGD("Make cluster for node %s : %s.", node->GetName().c_str(), node_2_cluster_[node]->DebugString().c_str());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::TopologicalSortClusters() {
  ordered_cluster_.clear();
  // BFS topological sort clusters for known shape cluster
  std::queue<ClusterPtr> ready_clusters;
  std::unordered_map<ClusterPtr, size_t> cluster_pending_count;
  std::unordered_set<ClusterPtr> seen_clusters;
  for (auto &iter : node_2_cluster_) {
    auto cluster = iter.second;
    if (seen_clusters.count(cluster) != 0) {
      continue;
    }
    seen_clusters.insert(cluster);
    auto pending_count = cluster->Inputs().size();
    if (pending_count == 0) {
      ready_clusters.push(cluster);
    } else {
      cluster_pending_count[cluster] = pending_count;
    }
  }

  size_t rank = 0;
  while (!ready_clusters.empty()) {
    auto cluster = ready_clusters.front();
    ready_clusters.pop();
    cluster->UpdateRank(rank++);
    if (cluster->IsKnownShape()) {
      ordered_cluster_.push_back(cluster);
    }
    for (const auto &out_cluster : cluster->Outputs()) {
      if (cluster_pending_count[out_cluster] > 0 && --cluster_pending_count[out_cluster] == 0) {
        ready_clusters.push(out_cluster);
      }
    }
  }
  if (rank != seen_clusters.size()) {
    return FAILED;
  }
  return SUCCESS;
}

namespace {
static std::string ToString(const std::vector<ClusterPtr> &clusters) {
  if (clusters.empty()) {
    return "()";
  }
  std::stringstream ss;
  ss << "(";
  auto iter = clusters.begin();
  for (size_t i = 0; i < clusters.size() - 1; i++) {
    ss << (*iter)->Id() << ",";
    iter++;
  }
  ss << (*iter)->Id() << ").";
  return ss.str();
}
}  // namespace

Status DynamicShapePartitioner::MergeClusters() {
  // Merge unknown shape clusters
  for (const auto &cluster : ordered_cluster_) {
    for (const auto &in_cluster : cluster->Inputs()) {
      if (!in_cluster->IsUnknownShape()) {
        continue;
      }
      auto merged_clusters = cluster->MergeAllPathFrom(in_cluster);
      GELOGD("Merge all path cluster from %lu to %lu %s.", in_cluster->Id(), cluster->Id(),
             ToString(merged_clusters).c_str());
      for (const auto &merged_cluster : merged_clusters) {
        for (const auto &node : merged_cluster->Nodes()) {
          node_2_cluster_[node] = cluster;
        }
      }
    }
  }

  REQUIRE_SUCCESS(TopologicalSortClusters(), "Failed topological sort clusters after merge unknown shape clusters.");
  // Merge known shape clusters
  for (const auto &cluster : ordered_cluster_) {
    if (cluster->IsRefVariable() && cluster->Inputs().size() == 1) {
      auto in_cluster = *(cluster->Inputs().begin());
      in_cluster->Merge(cluster);
      node_2_cluster_[*(cluster->Nodes().begin())] = in_cluster;
      continue;
    }

    for (const auto &in_cluster : cluster->Inputs()) {
      if (!in_cluster->IsKnownShape()) {
        continue;
      }
      if (cluster->TryMerge(in_cluster)) {
        GELOGD("Success merge known shape cluster from %lu to %lu.", in_cluster->Id(), cluster->Id());
        for (const auto &node : in_cluster->Nodes()) {
          node_2_cluster_[node] = cluster;
        }
      }
    }
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::CollectSpreadUnknownShapeNodes(NodePtr node) {
  if (unknown_shape_nodes_.count(node) > 0) {
    return SUCCESS;
  }
  auto opdesc = node->GetOpDesc();
  // One can set 'ATTR_NAME_IS_UNKNOWN_SHAPE=true' on node so as to forcing the node flow into the unknown subgraph,
  // ignore the actual shape.
  bool is_forced_unknown = false;
  if (AttrUtils::GetBool(opdesc, ATTR_NAME_IS_UNKNOWN_SHAPE, is_forced_unknown) && is_forced_unknown) {
    GELOGD("Collect node %s as unknown as it was marked unknown forcibly.", node->GetName().c_str());
    unknown_shape_nodes_.insert(node);
    return SUCCESS;
  }
  size_t anchor_index = 0;
  bool is_unknown = false;
  for (auto &out_tensor : opdesc->GetAllOutputsDesc()) {
    if (IsUnknownShapeTensor(out_tensor)) {
      GELOGD("Collect node %s as unknown as output %lu is unknown.", node->GetName().c_str(), anchor_index);
      is_unknown = true;
      auto anchor = node->GetOutDataAnchor(static_cast<int>(anchor_index));
      for (const auto peer_anchor : anchor->GetPeerInDataAnchors()) {
        if (peer_anchor != nullptr) {
          GELOGD("Collect node %s as has unknown input from %s:%lu.", peer_anchor->GetOwnerNode()->GetName().c_str(),
                 node->GetName().c_str(), anchor_index);
          unknown_shape_nodes_.insert(peer_anchor->GetOwnerNode());
        }
      }
    }
    anchor_index++;
  }
  anchor_index = 0;
  for (auto &in_tensor : opdesc->GetAllInputsDesc()) {
    if (IsUnknownShapeTensor(in_tensor)) {
      GELOGD("Collect node %s as unknown as input %lu is unknown.", node->GetName().c_str(), anchor_index);
      is_unknown = true;
      auto anchor = node->GetInDataAnchor(static_cast<int>(anchor_index));
      const auto peer_anchor = anchor->GetPeerOutAnchor();
      if (peer_anchor != nullptr) {
        GELOGD("Collect node %s as has unknown output to %s:%lu.", peer_anchor->GetOwnerNode()->GetName().c_str(),
               node->GetName().c_str(), anchor_index);
        unknown_shape_nodes_.insert(peer_anchor->GetOwnerNode());
      }
    }
    anchor_index++;
  }
  if (is_unknown) {
    unknown_shape_nodes_.insert(node);
  } else {
    auto graph = root_graph_;
    for (const auto &subgraph_name : opdesc->GetSubgraphInstanceNames()) {
      auto subgraph = graph->GetSubgraph(subgraph_name);
      REQUIRE_NOT_NULL(subgraph, "Failed get subgraph %s of node %s on root graph.", subgraph_name.c_str(),
                       node->GetName().c_str());
      bool is_graph_unknow = false;
      REQUIRE_SUCCESS(IsUnknownShapeGraph(subgraph, is_graph_unknow), "Failed check subgraph %s shape of node %s.",
                      subgraph_name.c_str(), node->GetName().c_str());
      if (is_graph_unknow) {
        GELOGD("Collect node %s as its subgraph %s is unknown.", node->GetName().c_str(), subgraph->GetName().c_str());
        unknown_shape_nodes_.insert(node);
        break;
      }
    }
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::IsUnknownShapeNode(NodePtr node, bool &is_unknown) {
  auto opdesc = node->GetOpDesc();
  auto graph = root_graph_;
  for (auto &out_tensor : opdesc->GetAllOutputsDesc()) {
    if (IsUnknownShapeTensor(out_tensor)) {
      GELOGD("Mark node %s unknown as unknown output.", node->GetName().c_str());
      is_unknown = true;
      return SUCCESS;
    }
  }
  for (auto &in_tensor : opdesc->GetAllInputsDesc()) {
    if (IsUnknownShapeTensor(in_tensor)) {
      GELOGD("Mark node %s unknown as unknown intput.", node->GetName().c_str());
      is_unknown = true;
      return SUCCESS;
    }
  }
  for (auto &subgraph_name : opdesc->GetSubgraphInstanceNames()) {
    auto subgraph = graph->GetSubgraph(subgraph_name);
    REQUIRE_NOT_NULL(subgraph, "Failed get subgraph %s of node %s on root graph.", subgraph_name.c_str(),
                     node->GetName().c_str());
    REQUIRE_SUCCESS(IsUnknownShapeGraph(subgraph, is_unknown), "Failed check subgraph %s shape of node %s.",
                    subgraph_name.c_str(), node->GetName().c_str());
    if (is_unknown) {
      GELOGD("Mark node %s unknown as unknown subgraph.", node->GetName().c_str());
      return SUCCESS;
    }
  }
  is_unknown = false;
  return SUCCESS;
}

Status DynamicShapePartitioner::IsUnknownShapeGraph(ComputeGraphPtr graph, bool &is_unknown) {
  for (auto &node : graph->GetDirectNode()) {
    REQUIRE_SUCCESS(IsUnknownShapeNode(node, is_unknown), "Failed check node %s shape on graph %s.",
                    node->GetName().c_str(), graph->GetName().c_str());
    if (is_unknown) {
      GELOGD("Mark graph %s unknown as contains unknown node %s.", graph->GetName().c_str(), node->GetName().c_str());
      return SUCCESS;
    }
  }
  return SUCCESS;
}

bool DynamicShapePartitioner::IsUnknownShapeTensor(const GeTensorDesc &tensor) {
  const static int kUnknowShape = -1;
  const static int kUnknowRank = -2;
  for (auto dim_size : tensor.GetShape().GetDims()) {
    if (dim_size == kUnknowShape || dim_size == kUnknowRank) {
      return true;
    }
  }
  return false;
}

std::string Cluster::DebugString() const {
  std::stringstream ss;
  switch (type_) {
    case DATA:
      ss << "DATA";
      break;
    case NETOUTPUT:
      ss << "NETOUTPUT";
      break;
    case UNKNOWN_SHAPE:
      ss << "UNKNOW";
      break;
    case KNOWN_SHAPE:
      ss << "KNOW";
      break;
  }
  ss << "[" << id_ << "](size:" << nodes_.size() << ")";
  ss << "(" << min_ << "," << max_ << ")(";
  for (const auto &cluster : in_clusters_) {
    ss << cluster->id_ << ",";
  }
  ss << ")->(";
  for (const auto &cluster : out_clusters_) {
    ss << cluster->id_ << ",";
  }
  ss << ")|";
  for (const auto &node : nodes_) {
    ss << (node->GetName() + "|");
  }
  return ss.str();
}

size_t Cluster::Id() const { return id_; }
void Cluster::UpdateRank(size_t rank) {
  max_ = rank;
  min_ = rank;
};
bool Cluster::IsData() const { return type_ == DATA; };
bool Cluster::IsKnownShape() const { return type_ == KNOWN_SHAPE; };
bool Cluster::IsUnknownShape() const { return type_ == UNKNOWN_SHAPE; };
bool Cluster::IsNetOutput() const { return type_ == NETOUTPUT; };
bool Cluster::IsRefVariable() const {
  if ((nodes_.size() == 1) && ((nodes_[0]->GetType() == VARIABLE) || (nodes_[0]->GetType() == VARIABLEV2))) {
    std::string ref_variable_name;
    return (AttrUtils::GetStr(nodes_[0]->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_variable_name) &&
            !ref_variable_name.empty());
  }
  return false;
}
void Cluster::AddInput(ClusterPtr in) {
  in_clusters_.insert(in);
  in->out_clusters_.insert(shared_from_this());
};
void Cluster::RemoveInput(ClusterPtr in) {
  in_clusters_.erase(in);
  in->out_clusters_.erase(shared_from_this());
};
void Cluster::AddOutput(ClusterPtr out) {
  out_clusters_.insert(out);
  out->in_clusters_.insert(shared_from_this());
};
void Cluster::RemoveOutput(ClusterPtr out) {
  out_clusters_.erase(out);
  out->in_clusters_.erase(shared_from_this());
};
void Cluster::Merge(ClusterPtr other) {
  nodes_.insert(nodes_.end(), other->nodes_.begin(), other->nodes_.end());
  other->in_clusters_.erase(shared_from_this());
  other->out_clusters_.erase(shared_from_this());
  in_clusters_.erase(other);
  out_clusters_.erase(other);
  auto in_clusters = other->in_clusters_;
  for (const auto &cluster : in_clusters) {
    cluster->RemoveOutput(other);
    cluster->AddOutput(shared_from_this());
  }
  auto out_clusters = other->out_clusters_;
  for (const auto &cluster : out_clusters) {
    cluster->RemoveInput(other);
    cluster->AddInput(shared_from_this());
  }
  if (other->max_ > max_) {
    max_ = other->max_;
  }
  if (other->min_ < min_) {
    min_ = other->min_;
  }
};
bool Cluster::TryMerge(ClusterPtr other) {
  std::queue<ClusterPtr> forward_reached;
  forward_reached.push(other);
  while (!forward_reached.empty()) {
    auto current_cluster = forward_reached.front();
    forward_reached.pop();
    for (const auto &cluster : current_cluster->out_clusters_) {
      if (cluster->max_ == max_ && current_cluster != other) {
        return false;
      } else if (cluster->min_ < max_) {
        forward_reached.push(cluster);
      }
    }
  }
  Merge(other);
  return true;
};
std::vector<ClusterPtr> Cluster::MergeAllPathFrom(ClusterPtr other) {
  std::queue<ClusterPtr> forward_reached_queue;
  std::queue<ClusterPtr> backward_reached_queue;

  std::unordered_set<ClusterPtr> forward_reached_clusters;
  std::unordered_set<ClusterPtr> backward_reached_clusters;
  std::vector<ClusterPtr> path_clusters;

  if (other->out_clusters_.count(shared_from_this()) == 0) {
    return path_clusters;
  }
  path_clusters.push_back(other);
  forward_reached_queue.push(other);
  backward_reached_queue.push(shared_from_this());
  while (!forward_reached_queue.empty()) {
    auto current_cluster = forward_reached_queue.front();
    forward_reached_queue.pop();
    for (const auto &cluster : current_cluster->out_clusters_) {
      if (cluster->min_ < max_ && cluster->max_ != max_ && forward_reached_clusters.count(cluster) == 0) {
        forward_reached_clusters.insert(cluster);
        forward_reached_queue.push(cluster);
      }
    }
  }
  while (!backward_reached_queue.empty()) {
    auto current_cluster = backward_reached_queue.front();
    backward_reached_queue.pop();
    for (const auto &cluster : current_cluster->in_clusters_) {
      if (cluster->max_ > other->min_ && cluster->max_ != other->max_ &&
          backward_reached_clusters.count(cluster) == 0) {
        backward_reached_clusters.insert(cluster);
        backward_reached_queue.push(cluster);
        if (forward_reached_clusters.count(cluster) != 0) {
          path_clusters.push_back(cluster);
        }
      }
    }
  }
  for (const auto &cluster : path_clusters) {
    Merge(cluster);
  }
  return path_clusters;
}
std::unordered_set<ClusterPtr> Cluster::Inputs() const { return in_clusters_; };
std::unordered_set<ClusterPtr> Cluster::Outputs() const { return out_clusters_; };
std::vector<NodePtr> Cluster::Nodes() const { return nodes_; };

void Cluster::AddFrameInput(InDataAnchorPtr anchor) {
  inputs_index_[anchor] = inputs_.size();
  inputs_.push_back(anchor);
};

void Cluster::AddFrameOutput(OutDataAnchorPtr anchor) {
  outputs_index_[anchor] = outputs_.size();
  outputs_.push_back(anchor);
};

InDataAnchorPtr Cluster::GetFrameInDataAnchor(InDataAnchorPtr anchor) {
  return partition_node_->GetInDataAnchor(static_cast<int>(inputs_index_[anchor]));
};

OutDataAnchorPtr Cluster::GetFrameOutDataAnchor(OutDataAnchorPtr anchor) {
  return partition_node_->GetOutDataAnchor(static_cast<int>(outputs_index_[anchor]));
};

InControlAnchorPtr Cluster::GetFrameInControlAnchor() { return partition_node_->GetInControlAnchor(); };

OutControlAnchorPtr Cluster::GetFrameOutControlAnchor() { return partition_node_->GetOutControlAnchor(); };

Status Cluster::BuildFrame() {
  if (IsUnknownShape() || IsKnownShape()) {
    return BuildPartitionFrame();
  } else {
    auto node = nodes_.front();
    auto in_control_anchor = node->GetInControlAnchor();
    if (in_control_anchor != nullptr) {
      for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
        auto src_cluster = partitioner_->node_2_cluster_[peer_out_control_anchor->GetOwnerNode()];
        if (src_cluster->id_ != id_) {
          REQUIRE_GRAPH_SUCCESS(
            GraphUtils::RemoveEdge(peer_out_control_anchor, in_control_anchor),
            "Failed remove edge from node %s index %d to node %s index %d.",
            peer_out_control_anchor->GetOwnerNode()->GetName().c_str(), AnchorUtils::GetIdx(peer_out_control_anchor),
            in_control_anchor->GetOwnerNode()->GetName().c_str(), AnchorUtils::GetIdx(in_control_anchor));
          control_inputs_.insert(src_cluster);
          src_cluster->control_outputs_.insert(peer_out_control_anchor);
        }
      }
    }
    if (IsData()) {
      for (const auto &anchor : node->GetAllOutDataAnchors()) {
        AddFrameOutput(anchor);
      }
    } else {
      for (const auto &anchor : node->GetAllInDataAnchors()) {
        AddFrameInput(anchor);
      }
    }
    partition_node_ = node;
  }
  return SUCCESS;
}

Status Cluster::BuildPartitionFrame() {
  auto graph = partitioner_->root_graph_;
  bool is_unknown_shape = IsUnknownShape();
  std::string sub_graph_name =
    graph->GetName() + "_sub_" + std::to_string(unique_id_) + (is_unknown_shape ? "_unknow" : "_know");
  subgraph_ = MakeShared<ComputeGraph>(sub_graph_name);
  REQUIRE_NOT_NULL(subgraph_, "Failed new memory for subgraph.");
  auto partition_op = MakeShared<OpDesc>("PartitionedCall_" + std::to_string(unique_id_++), "PartitionedCall");
  REQUIRE_NOT_NULL(partition_op, "Failed new memory for partition op.");
  REQUIRE(AttrUtils::SetBool(partition_op, ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape),
          "Failed set _is_unknown_shape flag on partitioned op %s.", partition_op->GetName().c_str());
  REQUIRE_GRAPH_SUCCESS(partition_op->AddSubgraphName(subgraph_->GetName()), "Failed add subgraph name.");
  REQUIRE_GRAPH_SUCCESS(partition_op->SetSubgraphInstanceName(0, subgraph_->GetName()),
                        "Failed set subgraph instance name.");
  for (auto &node : nodes_) {
    REQUIRE_NOT_NULL(subgraph_->AddNode(node), "Failed add node to subgraph.");
    REQUIRE(AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape),
            "Failed set shape flag.");
    REQUIRE_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, node), "Failed remove root graph node.");
    REQUIRE_GRAPH_SUCCESS(node->SetOwnerComputeGraph(subgraph_), "Failed set owner graph.");
    for (const auto &anchor : node->GetAllInDataAnchors()) {
      auto peer_out_anchor = anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;  // Skip overhang input.
      }
      auto src_cluster = partitioner_->node_2_cluster_[peer_out_anchor->GetOwnerNode()];
      if (src_cluster->id_ != id_) {
        AddFrameInput(anchor);
        REQUIRE_GRAPH_SUCCESS(partition_op->AddInputDesc(node->GetOpDesc()->GetInputDesc(anchor->GetIdx())),
                              "Failed add input desc.");
      }
    }
    auto in_control_anchor = node->GetInControlAnchor();
    if (in_control_anchor != nullptr) {
      for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
        if (peer_out_control_anchor == nullptr) {
          continue;
        }
        auto src_cluster = partitioner_->node_2_cluster_[peer_out_control_anchor->GetOwnerNode()];
        if (src_cluster->id_ != id_) {
          REQUIRE_GRAPH_SUCCESS(
            GraphUtils::RemoveEdge(peer_out_control_anchor, in_control_anchor),
            "Failed remove edge from %s:%d to %s:%d.", peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
            peer_out_control_anchor->GetIdx(), node->GetName().c_str(), in_control_anchor->GetIdx());
          control_inputs_.insert(src_cluster);
          src_cluster->control_outputs_.insert(peer_out_control_anchor);
        }
      }
    }
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      auto peer_in_anchors = anchor->GetPeerInDataAnchors();
      for (const auto &peer_in_anchor : peer_in_anchors) {
        auto src_cluster = partitioner_->node_2_cluster_[peer_in_anchor->GetOwnerNode()];
        if (src_cluster->id_ != id_) {
          AddFrameOutput(anchor);
          REQUIRE_GRAPH_SUCCESS(partition_op->AddOutputDesc(node->GetOpDesc()->GetOutputDesc(anchor->GetIdx())),
                                "Failed add output desc.");
          break;
        }
      }
    }
  }
  partition_node_ = graph->AddNode(partition_op);
  REQUIRE_NOT_NULL(partition_node_, "Failed add partition node.");
  REQUIRE_GRAPH_SUCCESS(partition_node_->SetOwnerComputeGraph(graph), "Failed set owner graph.");
  subgraph_->SetParentNode(partition_node_);
  subgraph_->SetParentGraph(graph);
  REQUIRE_GRAPH_SUCCESS(graph->AddSubgraph(subgraph_), "Failed add subgraph to root graph.");
  std::string session_graph_id;
  REQUIRE(AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
          "Failed get ATTR_NAME_SESSION_GRAPH_ID on root graph.");
  REQUIRE(AttrUtils::SetStr(*subgraph_, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
          "Failed set ATTR_NAME_SESSION_GRAPH_ID on subgraph.");
  return SUCCESS;
}

Status Cluster::CombinePartitionFrame() {
  for (const auto &anchor : inputs_) {
    auto peer_out_anchor = anchor->GetPeerOutAnchor();
    auto src_cluster = partitioner_->node_2_cluster_[peer_out_anchor->GetOwnerNode()];
    auto src_anchor = src_cluster->GetFrameOutDataAnchor(peer_out_anchor);
    auto dst_anchor = GetFrameInDataAnchor(anchor);
    REQUIRE_GRAPH_SUCCESS(GraphUtils::RemoveEdge(peer_out_anchor, anchor), "Failed remove edge from %s:%d to %s:%d.",
                          peer_out_anchor->GetOwnerNode()->GetName().c_str(), peer_out_anchor->GetIdx(),
                          anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx());
    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(src_anchor, dst_anchor), "Failed add edge from %s:%d to %s:%d.",
                          src_anchor->GetOwnerNode()->GetName().c_str(), src_anchor->GetIdx(),
                          dst_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetIdx());
  }
  for (const auto &src_cluster : control_inputs_) {
    auto src_anchor = src_cluster->GetFrameOutControlAnchor();
    auto dst_anchor = GetFrameInControlAnchor();
    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(src_anchor, dst_anchor), "Failed add edge from %s:%d to %s:%d.",
                          src_anchor->GetOwnerNode()->GetName().c_str(), src_anchor->GetIdx(),
                          dst_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetIdx());
  }
  return SUCCESS;
}

Status Cluster::BuildPartitionSubgraph() {
  if (IsData() || IsNetOutput()) {
    return SUCCESS;
  }
  int64_t parent_node_index = 0;
  for (auto anchor : inputs_) {
    auto data_op =
      MakeShared<OpDesc>(subgraph_->GetName() + std::string("Data_") + std::to_string(parent_node_index), ge::DATA);
    REQUIRE_NOT_NULL(data_op, "Failed new memory for data op.");
    auto input_desc = anchor->GetOwnerNode()->GetOpDesc()->GetInputDesc(anchor->GetIdx());
    REQUIRE_GRAPH_SUCCESS(data_op->AddInputDesc(input_desc), "Failed add input desc.");
    REQUIRE_GRAPH_SUCCESS(data_op->AddOutputDesc(input_desc), "Failed add output desc.");
    REQUIRE(AttrUtils::SetInt(data_op, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index),
            "Failed set parent_node_index on subgraph data node.");
    auto data_node = subgraph_->AddNode(data_op);
    REQUIRE_NOT_NULL(data_node, "Failed add data node to subgraph.");
    REQUIRE_GRAPH_SUCCESS(data_node->SetOwnerComputeGraph(subgraph_), "Failed set owner graph of data node.");
    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), anchor),
                          "Faile add data input edge to %s:%d", anchor->GetOwnerNode()->GetName().c_str(),
                          anchor->GetIdx());
    parent_node_index++;
  }
  if (outputs_.empty() && control_outputs_.empty()) {
    return SUCCESS;
  }
  auto net_output_op = MakeShared<OpDesc>(subgraph_->GetName() + "_" + NODE_NAME_NET_OUTPUT, ge::NETOUTPUT);
  REQUIRE_NOT_NULL(net_output_op, "Failed new memory for netoutput op.");
  for (size_t i = 0; i < outputs_.size(); ++i) {
    GeTensorDesc input_desc;
    REQUIRE_GRAPH_SUCCESS(net_output_op->AddInputDesc(input_desc), "Failed add input desc.");
  }
  auto net_output_node = subgraph_->AddNode(net_output_op);
  REQUIRE_NOT_NULL(net_output_node, "Failed add netoutput node to subgraph.");
  REQUIRE_GRAPH_SUCCESS(net_output_node->SetOwnerComputeGraph(subgraph_), "Failed set owner graph of netoutput node.");
  parent_node_index = 0;
  for (const auto &anchor : outputs_) {
    auto output_desc = anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(anchor->GetIdx()));
    REQUIRE(AttrUtils::SetInt(output_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index),
            "Failed set parent_node_index on subgraph netoutput's input.");
    REQUIRE_GRAPH_SUCCESS(net_output_op->UpdateInputDesc(parent_node_index, output_desc),
                          "Failed update input desc of netoutput node.");

    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(anchor, net_output_node->GetInDataAnchor(parent_node_index)),
                          "Faile add edge from %s:%d to netoutput node.", anchor->GetOwnerNode()->GetName().c_str(),
                          anchor->GetIdx());
    parent_node_index++;
  }
  for (const auto &anchor : control_outputs_) {
    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(anchor, net_output_node->GetInControlAnchor()),
                          "Faile add control edge from %s:%d to netoutput node.",
                          anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx());
  }
  return SUCCESS;
}
void Cluster::Clear() {
  in_clusters_.clear();
  out_clusters_.clear();
  nodes_.clear();
  partitioner_ = nullptr;
  inputs_index_.clear();
  outputs_index_.clear();
  inputs_.clear();
  outputs_.clear();
  control_inputs_.clear();
  control_outputs_.clear();
  partition_node_.reset();
  subgraph_.reset();
}

size_t Cluster::unique_id_ = 0;
}  // namespace ge