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

#include "graph/partition/graph_partitioner.h"
#include <sstream>
#include "common/checker.h"
#include "common/plugin/ge_util.h"
#include "common/util/mem_utils.h"
#include "common/types.h"
#include "graph/partition/default_subgraph_builder.h"
#include "graph/partition/subgraph_builder_for_optimization.h"

namespace ge {
namespace {
using BuildSubgraphBuilderFunc = std::unique_ptr<BaseSubgraphBuilder> (*)();
std::unique_ptr<BaseSubgraphBuilder> BuildDefaultSubgraphBuilder() {
  return MakeUnique<DefaultSubgraphBuilder>();
}

std::unique_ptr<BaseSubgraphBuilder> BuildSubgraphBuilderForOptimization() {
  return MakeUnique<SubgraphBuilderForOptimization>();
}

std::unique_ptr<BaseSubgraphBuilder> GetBuildSubgraphBuilder(const PartitionSubgraphType subgraph_type) {
  static std::map<PartitionSubgraphType, BuildSubgraphBuilderFunc> subgraph_type_to_builder_func = {
      {PartitionSubgraphType::kDefaultSubgraph, BuildDefaultSubgraphBuilder},
      {PartitionSubgraphType::kSubgraphForOptimization, BuildSubgraphBuilderForOptimization}
  };
  const std::map<PartitionSubgraphType, BuildSubgraphBuilderFunc>::const_iterator type_2_builder =
      subgraph_type_to_builder_func.find(subgraph_type);
  if (type_2_builder != subgraph_type_to_builder_func.cend()) {
    return type_2_builder->second();
  }
  return nullptr;
}
}
graphStatus GraphPartitioner::Partition() {
  GE_ASSERT_NOTNULL(root_graph_, "[Check][Input] root_graph_ can not be null.");
  GE_ASSERT_GRAPH_SUCCESS(root_graph_->TopologicalSorting(), "[Call][TopologicalSorting] failed, root_graph[%s].",
                          root_graph_->GetName().c_str());
  // 1.InitClusters
  GE_ASSERT_GRAPH_SUCCESS(InitClusters(), "[Call][InitClusters] failed, root_graph[%s].",
                          root_graph_->GetName().c_str());
  for (size_t turn = 0U; turn < merge_policies_.size(); turn++) {
    // 2.OnTurnBegin
    GE_ASSERT_GRAPH_SUCCESS(InvokeOnTurnBeginPolicy(),
                            "[Call][InvokeOnTurnBeginPolicy] on turn[%lu] failed, root_graph[%s].", turn_,
                            root_graph_->GetName().c_str());
    // 3.MergeClusters
    GE_ASSERT_GRAPH_SUCCESS(MergeClusters(), "[Call][MergeClusters] on turn[%lu] failed, root_graph[%s].", turn_,
                            root_graph_->GetName().c_str());
    // 4.SortClusters
    GE_ASSERT_GRAPH_SUCCESS(SortClustersForBuildSubgraph(),
                            "[Call][SortClustersForBuildSubgraph] on turn[%lu] failed, root_graph[%s].", turn_,
                            root_graph_->GetName().c_str());
    DebugMergeLog();
    // 5.BuildSubgraphs
    GE_ASSERT_GRAPH_SUCCESS(BuildSubgraphs(), "[Call][BuildSubgraphs] on turn[%lu] failed, root_graph[%s].", turn_,
                            root_graph_->GetName().c_str());
    turn_++;
  }
  GE_ASSERT_GRAPH_SUCCESS(root_graph_->TopologicalSorting(), "[Call][TopologicalSorting] failed, root_graph[%s].",
                          root_graph_->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus GraphPartitioner::InitClusters() {
  size_t rank = 0UL;
  GELOGI("InitClusters enter, graph_name[%s].", root_graph_->GetName().c_str());
  for (const auto &node : root_graph_->GetDirectNode()) {
    const auto cluster = MakeShared<NodesCluster>(node, rank++);
    GE_ASSERT_NOTNULL(cluster, "[New][Memory] for cluster failed.");
    cluster_dict_.AddCluster(cluster);
    cluster_dict_.SetNodeClusterPair(node, cluster);
    // Already sorted topologically, so access to the parent cluster is safe
    for (const auto &from_node : node->GetInAllNodes()) {
      // cut while v1 loop
      if ((from_node->GetType() == NEXTITERATION) || (from_node->GetType() == REFNEXTITERATION)) {
        continue;
      }
      const auto &from_cluster = cluster_dict_.GetNodeCluster(from_node);
      GE_ASSERT_NOTNULL(from_cluster, "[Check][Input] failed, input cluster is null, from_node:%s, node:%s.",
                        from_node->GetName().c_str(), node->GetName().c_str());
      cluster->AddInput(*(from_cluster.get()));
    }
    GELOGD("Init cluster for node:%s.", node->GetName().c_str());
  }
  for (const auto &debug_node : root_graph_->GetDirectNode()) {
    GELOGD("Make cluster for node %s : %s.", debug_node->GetName().c_str(),
           cluster_dict_.GetNodeCluster(debug_node)->DebugString().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus GraphPartitioner::InvokeOnTurnBeginPolicy() {
  return merge_policies_[turn_]->OnTurnBegin(cluster_dict_);
}

void GraphPartitioner::DebugMergeLog() const {
  if (!IsLogEnable(GE, DLOG_DEBUG)) {
    return;
  }
  for (const auto &cluster : cluster_dict_.GetAllClusters()) {
    std::stringstream ss;
    ss << "[CLUSTER_MERGER]["
       << cluster->DebugString()
       << " policy_info:"
       << merge_policies_[turn_]->PolicyDebugString(*cluster)
       << "]";
    size_t debug_string_size = ss.str().size();
    size_t pos = 0UL;
    for (size_t loop = 0UL; loop < (debug_string_size / 1024UL); loop++) {
      GELOGD("%s", ss.str().c_str() + pos);
      pos += static_cast<size_t>(MSG_LENGTH);
    }
    GELOGD("%s", ss.str().c_str() + pos);
  }
}

graphStatus GraphPartitioner::MergeClusters() {
  // Merge clusters according to the linking relationship
  for (const auto &cluster : cluster_dict_.GetAllClusters()) {
    const auto cluster_inputs = cluster->Inputs();
    for (const auto &in_cluster : cluster_inputs) {
      // partition different engine
      bool need_merge = merge_policies_[turn_]->TryMerge(*in_cluster, *cluster);
      if (need_merge) {
        cluster->MergeFrom(*in_cluster);
        for (const auto &node : in_cluster->Nodes()) {
          cluster_dict_.SetNodeClusterPair(node, cluster);
        }
        GELOGD("merge cluster from %zu to %zu.", in_cluster->Id(), cluster->Id());
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus GraphPartitioner::SortClustersForBuildSubgraph() {
  // Unique clusters left after merged clusters sorted by rank
  std::vector<NodesClusterPtr> sorted_unique_clusters;
  // Unique clusters left after merged clusters
  std::unordered_set<NodesClusterPtr> unique_clusters;
  const auto comp_func = [](const NodesClusterPtr &clu_a,
                            const NodesClusterPtr &clu_b) -> bool {
    return clu_a->Id() < clu_b->Id();
  };
  for (const auto &node : root_graph_->GetDirectNode()) {
    const auto &cluster = cluster_dict_.GetNodeCluster(node);
    if (unique_clusters.count(cluster) != 0U) {
      continue;
    }
    if (unique_clusters.insert(cluster).second) {
      sorted_unique_clusters.emplace_back(cluster);
    }
  }
  std::sort(sorted_unique_clusters.begin(), sorted_unique_clusters.end(), comp_func);
  cluster_dict_.SwapClusters(sorted_unique_clusters);
  return GRAPH_SUCCESS;
}

graphStatus GraphPartitioner::BuildSubgraphs() {
  subgraph_builder_ = GetBuildSubgraphBuilder(build_policy_->GetSubgraphType());
  GE_ASSERT_NOTNULL(subgraph_builder_,
                    "[Check][SubgraphType] by partitioner of graph[%s] failed, subgraph type[%d] should less than %d.",
                    root_graph_->GetName().c_str(), build_policy_->GetSubgraphType(),
                    PartitionSubgraphType::kUnsupportSubgraphType);
  (void)subgraph_builder_->SetRootGraph(root_graph_)
                         ->SetClustersDict(&cluster_dict_)
                         ->SetSubgraphBuilderPolicy(build_policy_.get());
  for (const auto &cluster : cluster_dict_.GetAllClusters()) {
    (void)subgraph_builder_->SetNodesCluster(cluster);
    GE_ASSERT_GRAPH_SUCCESS(subgraph_builder_->BuildSubgraph(), "[Call][BuildSubgraph] failed.");
  }
  return GRAPH_SUCCESS;
}
} // namespace ge
