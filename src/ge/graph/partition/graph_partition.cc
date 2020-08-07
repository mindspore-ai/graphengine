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

#include "graph/partition/graph_partition.h"
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "common/ge/ge_util.h"
#include "common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/common/ge_call_wrapper.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_manager.h"

namespace {
const char *const kEngineDefaultData = "ENGINE_DEFAULT_DATA";
const char *const kEndType = "End";
const char *const kPlaceHolderType = "PlaceHolder";
const int kOneGraph = 1;  // only one graph
const int kRankOne = 1;   // order of graph list is 0,1,2,3..., 1 means second order
const int kRankZero = 0;  // order of graph list is 0,1,2,3..., 0 means first order
}  // namespace
namespace ge {
Status ge::GraphPartitioner::CheckIfEnd2PldEmpty(ge::ComputeGraphPtr &output_merged_compute_graph) {
  // only one condition:no data node, one engine, there is only one graph + input graph
  if (graph_info_.partitions_.size() == kOneGraph) {
    auto partition = (*graph_info_.partitions_.begin());
    if (partition.first == nullptr) {
      GELOGE(GE_GRAPH_EMPTY_PARTITION, "[GraphPartitioner]: partition.first is null, engine name is %s",
             partition.second.c_str());
      return FAILED;
    }
    output_merged_compute_graph = partition.first;
  } else {  // if placeholder to end map is empty, it should be an exception condition
    GELOGE(GE_GRAPH_EMPTY_PARTITION, "[GraphPartitioner]: placeholder to end map is empty, partitions size is not 1.");
    return FAILED;
  }
  return SUCCESS;
}

Status ge::GraphPartitioner::MergeAllSubGraph(ge::ComputeGraphPtr &output_merged_compute_graph,
                                              const std::vector<SubGraphInfoPtr> &sub_graph_list) {
  for (size_t rank = 0; rank < graph_info_.rank_2_partitions_.size(); rank++) {
    string temp_stream;
    // sub_graph_list index is one ahead of rank_2_partitions_list index
    if (rank > 0) {
      temp_stream = sub_graph_list[rank - 1]->GetStreamLabel();
    }
    for (const auto &node : graph_info_.rank_2_partitions_[rank]->GetDirectNode()) {
      if (node == nullptr) {
        continue;
      }
      if ((node->GetType() == kEndType) || (node->GetType() == kPlaceHolderType)) {
        continue;
      }
      if (!temp_stream.empty() && !AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL)) {
        (void)AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, temp_stream);
      }
      if (node->SetOwnerComputeGraph(output_merged_compute_graph) != GRAPH_SUCCESS) {
        GELOGE(GE_GRAPH_PARAM_NULLPTR, "SetownerComputeGraph failed, node %s", node->GetName().c_str());
        return FAILED;
      }
      (void)output_merged_compute_graph->AddNode(node);
    }
  }
  // get session graph id from subgraph
  SetMergedGraphId(output_merged_compute_graph);
  return SUCCESS;
}

void ge::GraphPartitioner::SetMergedGraphId(ge::ComputeGraphPtr &output_merged_compute_graph) {
  string session_graph_id;
  // get session graph id from subgraph
  if (graph_info_.rank_2_partitions_.empty() ||
      !AttrUtils::GetStr(*(graph_info_.rank_2_partitions_[0]), ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Get graph session_graph_id attr failed.");
  }
  // set session graph id into merged subgraph
  if (!session_graph_id.empty()) {
    GELOGI("Set session graph id %s in merged compute graph", session_graph_id.c_str());
    // private function, promise output_merged_compute_graph not null
    GE_IF_BOOL_EXEC(!AttrUtils::SetStr(*output_merged_compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                    GELOGW("SetStr ATTR_NAME_SESSION_GRAPH_ID failed");)
  }
}

Status ge::GraphPartitioner::RemoveNodeAndEdgeBetweenEndPld(ge::ComputeGraphPtr &output_merged_compute_graph,
                                                            const std::vector<SubGraphInfoPtr> &sub_graph_list) {
  if ((output_merged_compute_graph == nullptr) ||
      (MergeAllSubGraph(output_merged_compute_graph, sub_graph_list) != SUCCESS)) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[GraphPartitioner]: MergeAllSubGraph failed.");
    return FAILED;
  }
  for (const auto &it : graph_info_.index_2_end_) {
    auto &end = it.second;
    auto &pld = graph_info_.end_2_pld_[it.second];
    if ((end != nullptr) && (pld != nullptr) && (end->GetInDataAnchor(0) != nullptr) &&
        (pld->GetOutDataAnchor(0) != nullptr)) {
      AnchorPtr end_in_anchor = (end->GetInDataAnchor(0)->GetFirstPeerAnchor() == nullptr)
                                  ? Anchor::DynamicAnchorCast<Anchor>(end->GetInControlAnchor())
                                  : Anchor::DynamicAnchorCast<Anchor>(end->GetInDataAnchor(0));
      AnchorPtr pld_out_anchor = (pld->GetOutDataAnchor(0)->GetFirstPeerAnchor() == nullptr)
                                   ? Anchor::DynamicAnchorCast<Anchor>(pld->GetOutControlAnchor())
                                   : Anchor::DynamicAnchorCast<Anchor>(pld->GetOutDataAnchor(0));
      auto src_anchor = end_in_anchor->GetFirstPeerAnchor();  // src_anchor should be only 1
      if (GraphUtils::RemoveEdge(src_anchor, end_in_anchor) != GRAPH_SUCCESS) {
        GELOGE(GE_GRAPH_PARAM_NULLPTR, "[GraphPartitioner]: RemoveEdge failed. node_name:%s, graph_name:%s",
               end->GetName().c_str(), end->GetOwnerComputeGraph()->GetName().c_str());
        return FAILED;
      }
      GE_CHECK_NOTNULL(pld_out_anchor);
      for (const auto &peer_in_anchor : pld_out_anchor->GetPeerAnchors()) {
        if (GraphUtils::RemoveEdge(pld_out_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
          GELOGE(GE_GRAPH_PARAM_NULLPTR, "[GraphPartitioner]: RemoveEdge failed. node_name:%s, graph_name:%s",
                 pld->GetName().c_str(), pld->GetOwnerComputeGraph()->GetName().c_str());
          return FAILED;
        }
        if (GraphUtils::AddEdge(src_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
          GELOGE(GE_GRAPH_PARAM_NULLPTR, "merge two subgraph fail.");
          return FAILED;
        }
      }
    } else {
      GELOGW("End or pld is nullptr or in data anchor of end is nullptr or out data anchor of pld is nullptr");
    }
  }
  return SUCCESS;
}

Status ge::GraphPartitioner::MergeAfterSubGraphOptimization(ge::ComputeGraphPtr &output_merged_compute_graph,
                                                            const ge::ComputeGraphPtr &original_compute_graph) {
  auto ret = MergeSubGraph(output_merged_compute_graph, original_compute_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph merging Failed");
    return ret;
  }
  // partition sub graph
  for (const auto &sub_graph : original_compute_graph->GetAllSubgraphs()) {
    ComputeGraphPtr merged_sub_graph = nullptr;
    ret = MergeSubGraph(merged_sub_graph, sub_graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "Sub graph merging Failed");
      return ret;
    }
    // add sub graph
    output_merged_compute_graph->SetName(original_compute_graph->GetName());
    merged_sub_graph->SetName(sub_graph->GetName());
    merged_sub_graph->SetInputSize(sub_graph->GetInputSize());
    merged_sub_graph->SetOutputSize(sub_graph->GetOutputSize());
    auto parent_node = sub_graph->GetParentNode();
    GE_IF_BOOL_EXEC(parent_node == nullptr,
                    GELOGE(FAILED, "Parent node is null, graph name is %s", sub_graph->GetName().c_str());
                    return FAILED;)
    auto original_graph = parent_node->GetOwnerComputeGraph();
    GE_IF_BOOL_EXEC(graph_2_graph_partition_info_.find(original_graph) == graph_2_graph_partition_info_.end(),
                    GELOGE(FAILED, "Find graph info failed, graph name is %s", original_graph->GetName().c_str());
                    return FAILED;)
    auto graph_info = graph_2_graph_partition_info_[original_graph];
    GE_IF_BOOL_EXEC(
      graph_info.corresponding_node_in_partitions_.find(parent_node) ==
        graph_info.corresponding_node_in_partitions_.end(),
      GELOGE(FAILED, "Find corresponding node failed, parent node name is %s", parent_node->GetName().c_str());
      return FAILED;)
    auto corresponding_node = graph_info.corresponding_node_in_partitions_[parent_node];
    GE_IF_BOOL_EXEC(corresponding_node == nullptr,
                    GELOGE(FAILED, "Get null node, node name is %s", parent_node->GetName().c_str());
                    return FAILED;);
    merged_sub_graph->SetParentNode(corresponding_node);
    auto subgraph_parent_graph = corresponding_node->GetOwnerComputeGraph();
    merged_sub_graph->SetParentGraph(subgraph_parent_graph);
    ret = output_merged_compute_graph->AddSubgraph(sub_graph->GetName(), merged_sub_graph);
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, return ret;)
  }
  graph_2_graph_partition_info_.clear();
  graph_2_subgraph_list_.clear();
  return SUCCESS;
}

Status ge::GraphPartitioner::MergeSubGraph(ge::ComputeGraphPtr &output_merged_compute_graph,
                                           const ge::ComputeGraphPtr &original_compute_graph) {
  if (original_compute_graph == nullptr) {
    GELOGE(GE_GRAPH_NULL_INPUT, "[GraphPartitioner]: compute_graph is null.");
    return FAILED;
  }
  if ((graph_2_graph_partition_info_.find(original_compute_graph) == graph_2_graph_partition_info_.end()) ||
      (graph_2_subgraph_list_.find(original_compute_graph) == graph_2_subgraph_list_.end())) {
    GELOGE(GE_GRAPH_NULL_INPUT, "[GraphPartitioner]: compute_graph is error.");
    return FAILED;
  }
  GraphPartitionInfo &subgraph_info = graph_2_graph_partition_info_[original_compute_graph];
  const auto &sub_graph_list = graph_2_subgraph_list_[original_compute_graph];
  graph_info_ = subgraph_info;

  if (graph_info_.mode_ != kMerging) {
    GELOGE(GE_GRAPH_UNSUPPORTED, "Cannot call merging in partition mode");
    return FAILED;
  }
  GELOGI("Graph merge starts.");
  // check input param
  for (const auto &it : sub_graph_list) {
    if (it == nullptr) {
      GELOGE(GE_GRAPH_PARAM_NULLPTR, "[GraphPartitioner]: merging sub-graphs failed, sub-graph is null");
      return FAILED;
    }
  }
  bool is_map_empty = graph_info_.end_2_pld_.empty() || graph_info_.pld_2_end_.empty();
  if (is_map_empty) {
    if (CheckIfEnd2PldEmpty(output_merged_compute_graph) != SUCCESS) {
      return FAILED;
    }
  }
  ComputeGraphPtr new_sub_graph = MakeShared<ComputeGraph>(original_compute_graph->GetName());
  GE_CHECK_NOTNULL(new_sub_graph);
  output_merged_compute_graph = new_sub_graph;
  GE_TIMESTAMP_START(MergeSubGraphRemoveNode);
  if (RemoveNodeAndEdgeBetweenEndPld(output_merged_compute_graph, sub_graph_list) != ge::SUCCESS) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[GraphPartitioner]: merging sub-graphs failed");
    return FAILED;
  }
  GE_TIMESTAMP_END(MergeSubGraphRemoveNode, "GraphPartitioner::MergeGraphRemoveNodeAndEdge");
  GE_TIMESTAMP_START(MergeSubGraphTopologicalSorting);
  Status ret = output_merged_compute_graph->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_TOPO_SORT_FAILED, "[GraphPartitioner]: output_merged_compute_graph->TopologicalSorting failed");
    return FAILED;
  }
  GE_TIMESTAMP_END(MergeSubGraphTopologicalSorting, "GraphPartitioner::MergeGraphTopologicalSorting");
  // flush all nodes' engine of merged graph
  GE_TIMESTAMP_START(MergeSubGraphEnginePlacerRun);
  graph_info_.engine_placer_.SetComputeGraph(output_merged_compute_graph);
  if (graph_info_.engine_placer_.Run() != SUCCESS) {
    GELOGE(GE_GRAPH_INIT_FAILED, "[GraphPartitioner]: engine_placer run failed");
    return FAILED;
  }
  GE_TIMESTAMP_END(MergeSubGraphEnginePlacerRun, "GraphPartitioner::MergeGraphEnginePlacerRun");
  GELOGI("Graph merge ends.");
  return SUCCESS;
}

Status ge::GraphPartitioner::UpdatePldOpDesc(const NodePtr &dst_node, int input_index, OpDescPtr &pld_op_desc) {
  if ((dst_node == nullptr) || (pld_op_desc == nullptr) || (dst_node->GetOpDesc() == nullptr)) {
    GELOGE(FAILED, "parameter ptr is null.");
    return FAILED;
  }
  const auto &input_desc = dst_node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(input_index));
  GE_IF_BOOL_EXEC(pld_op_desc->AddOutputDesc(input_desc) != GRAPH_SUCCESS, GELOGE(FAILED, "AddOutputDesc failed");
                  return FAILED;)
  if (pld_op_desc->MutableOutputDesc(0) != nullptr) {
    ge::TensorUtils::SetRealDimCnt(*(pld_op_desc->MutableOutputDesc(0).get()),
                                   static_cast<uint32_t>(input_desc.GetShape().GetDims().size()));
  } else {
    GELOGE(GE_GRAPH_ADD_PLC_END_FAILED, "[GraphPartitioner]: pld_op_desc is null.");
    return FAILED;
  }
  return SUCCESS;
}

Status ge::GraphPartitioner::UpdateEndOpDesc(const NodePtr &src_node, int output_index, OpDescPtr &end_op_desc) {
  if ((src_node == nullptr) || (end_op_desc == nullptr) || (src_node->GetOpDesc() == nullptr)) {
    GELOGE(FAILED, "parameter ptr is null.");
    return FAILED;
  }
  const auto &output_desc = src_node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(output_index));
  GE_IF_BOOL_EXEC(end_op_desc->AddInputDesc(output_desc) != GRAPH_SUCCESS, GELOGE(FAILED, "AddInputDesc failed");
                  return FAILED;)
  if (end_op_desc->MutableInputDesc(0) != nullptr) {
    ge::TensorUtils::SetRealDimCnt(*(end_op_desc->MutableInputDesc(0).get()),
                                   static_cast<uint32_t>(output_desc.GetShape().GetDims().size()));
  } else {
    GELOGE(GE_GRAPH_ADD_PLC_END_FAILED, "[GraphPartitioner]: pld_op_desc is null.");
    return FAILED;
  }
  return SUCCESS;
}

graphStatus ge::GraphPartitioner::AddPlaceHolderEndInSrcDstGraph(const AnchorPtr &out_anchor,
                                                                 const AnchorPtr &peer_in_anchor,
                                                                 const ge::ComputeGraphPtr &pld_graph,
                                                                 const ge::ComputeGraphPtr &end_graph) {
  GE_CHECK_NOTNULL(peer_in_anchor);
  GE_CHECK_NOTNULL(pld_graph);
  GE_CHECK_NOTNULL(out_anchor);
  GE_CHECK_NOTNULL(end_graph);
  const auto &src_node = out_anchor->GetOwnerNode();
  const auto &dst_node = peer_in_anchor->GetOwnerNode();
  // link input -> end
  string end_name = kEndType + std::to_string(graph_info_.num_of_pld_end_);
  auto end_op_desc = MakeShared<OpDesc>(end_graph->GetName() + "_" + end_name, END);
  if (end_op_desc == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "pld_op_desc is nullptr.");
    return FAILED;
  }
  GE_IF_BOOL_EXEC(!AttrUtils::SetInt(end_op_desc, "peerIndex", graph_info_.num_of_pld_end_),
                  GELOGW("SetInt peerIndex failed");)
  GE_IF_BOOL_EXEC(!AttrUtils::SetStr(end_op_desc, "parentOpType", dst_node->GetType()),
                  GELOGW("SetStr parentOpType failed");)
  GE_IF_BOOL_EXEC(!end_op_desc->SetExtAttr("parentNode", dst_node), GELOGW("SetEndExtAttr parentNode failed");)
  OpDescPtr dst_node_op_desc = dst_node->GetOpDesc();
  GE_CHECK_NOTNULL(dst_node_op_desc);
  GE_IF_BOOL_EXEC(
    !AttrUtils::SetStr(end_op_desc, ATTR_NAME_END_REAR_NODE_ENGINE_NAME, dst_node_op_desc->GetOpEngineName()),
    GELOGW("SetStr rearNodeEngineName failed");)
  // replace input_desc of end with owner node's desc
  int output_index = ge::AnchorUtils::GetIdx(out_anchor);
  bool is_need_update_desc = (output_index >= 0) && (graph_info_.mode_ == kPartitioning);
  if (is_need_update_desc) {
    if (UpdateEndOpDesc(src_node, output_index, end_op_desc) != SUCCESS) {
      GELOGE(GRAPH_PARAM_INVALID, "UpdateEndOpDesc failed, input index %d", output_index);
      return FAILED;
    }
  } else {
    GeTensorDesc input_desc;
    if (end_op_desc->AddInputDesc(input_desc) != SUCCESS) {
      GELOGE(GRAPH_PARAM_INVALID, "AddInputDesc failed, input index %d", output_index);
      return FAILED;
    }
  }
  NodePtr new_end_node = end_graph->AddNode(end_op_desc);
  if (new_end_node == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "new_end_node is nullptr.");
    return FAILED;
  }
  GE_IF_BOOL_EXEC(new_end_node->SetOwnerComputeGraph(end_graph) != GRAPH_SUCCESS,
                  GELOGE(GRAPH_PARAM_INVALID, "SetOwnerComputeGraph failed");
                  return FAILED;)
  AnchorPtr end_dst_anchor = GetEndInAnchor(out_anchor, new_end_node);
  if (GraphUtils::AddEdge(out_anchor, end_dst_anchor) != GRAPH_SUCCESS) {
    GELOGE(GE_GRAPH_ADD_PLC_END_FAILED, "add end node :  %s node %dth out-anchor --> end in %s subgraph fail.",
           src_node->GetName().c_str(), AnchorUtils::GetIdx(out_anchor), end_graph->GetName().c_str());
    return FAILED;
  }
  /// For fe, op id has been set in AddNode,
  /// we can take op id of srcNode as the mark of parentId now
  const auto &src_node_opdesc = src_node->GetOpDesc();
  GE_CHECK_NOTNULL(src_node_opdesc);
  int64_t node_id = src_node_opdesc->GetId();
  const string pld_name = kPlaceHolderType + std::to_string(graph_info_.num_of_pld_end_);
  auto pld_op_desc = MakeShared<OpDesc>(pld_graph->GetName() + "_" + pld_name, PLACEHOLDER);
  if (pld_op_desc == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "pld_op_desc is nullptr.");
    return FAILED;
  }
  GE_IF_BOOL_EXEC(!AttrUtils::SetInt(pld_op_desc, "peerIndex", graph_info_.num_of_pld_end_),
                  GELOGW("SetInt peerIndex failed");)
  GE_IF_BOOL_EXEC(!AttrUtils::SetStr(pld_op_desc, "parentOpType", src_node->GetType()),
                  GELOGW("SetStr parentOpType failed");)
  GE_IF_BOOL_EXEC(!AttrUtils::SetStr(pld_op_desc, "parentId", end_graph->GetName() + ":" + std::to_string(node_id)),
                  GELOGW("SetStr parentId failed");)
  GE_IF_BOOL_EXEC(!AttrUtils::SetInt(pld_op_desc, "anchorIndex", AnchorUtils::GetIdx(out_anchor)),
                  GELOGW("SetInt anchorIndex failed");)
  GE_IF_BOOL_EXEC(!pld_op_desc->SetExtAttr("parentNode", src_node), GELOGW("SetPldExtAttr parentNode failed");)
  OpDescPtr src_node_op_desc = src_node->GetOpDesc();
  GE_CHECK_NOTNULL(src_node_op_desc);
  GE_IF_BOOL_EXEC(
    !AttrUtils::SetStr(pld_op_desc, ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, src_node_op_desc->GetOpEngineName()),
    GELOGW("SetStr frontNodeEngineName failed");)
  // do not care over flow
  graph_info_.num_of_pld_end_++;
  // replace output_desc of pld with input node's output desc
  int input_index = ge::AnchorUtils::GetIdx(peer_in_anchor);
  is_need_update_desc = (input_index >= 0) && (graph_info_.mode_ == kPartitioning);
  if (is_need_update_desc) {
    if (UpdatePldOpDesc(dst_node, input_index, pld_op_desc) != SUCCESS) {
      GELOGE(GRAPH_PARAM_INVALID, "UpdateEndOpDesc failed, output index %d", input_index);
      return FAILED;
    }
  } else {
    GeTensorDesc output_desc;
    if (pld_op_desc->AddOutputDesc(output_desc) != SUCCESS) {
      GELOGE(GRAPH_PARAM_INVALID, "AddOutputDesc failed, input index %d", input_index);
      return FAILED;
    }
  }
  NodePtr new_pld_node = pld_graph->AddNode(pld_op_desc);
  if (new_pld_node == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "new_pld_node is nullptr.");
    return FAILED;
  }
  GE_IF_BOOL_EXEC(new_pld_node->SetOwnerComputeGraph(pld_graph) != GRAPH_SUCCESS,
                  GELOGE(GRAPH_PARAM_INVALID, "SetOwnerComputeGraph failed");
                  return FAILED;)
  AnchorPtr pld_src_anchor = GetPldOutAnchor(new_pld_node, peer_in_anchor);
  // link placeHolder -> computeNode
  if (GraphUtils::AddEdge(pld_src_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
    GELOGE(GE_GRAPH_ADD_PLC_END_FAILED,
           "add placeholder node :  placeholder --> %s node %dth in-anchor in %s subgraph fail.",
           dst_node->GetName().c_str(), AnchorUtils::GetIdx(peer_in_anchor), pld_graph->GetName().c_str());
    return FAILED;
  }
  graph_info_.index_2_end_[graph_info_.num_of_pld_end_] = new_end_node;
  graph_info_.pld_2_end_[new_pld_node] = new_end_node;
  graph_info_.end_2_pld_[new_end_node] = new_pld_node;
  return SUCCESS;
}

Status ge::GraphPartitioner::LinkInput2EndRemoveOrginalLink(ge::NodePtr input_node, ge::ComputeGraphPtr src_graph,
                                                            ge::ComputeGraphPtr dst_graph) {
  if ((input_node == nullptr) || (src_graph == nullptr) || (dst_graph == nullptr)) {
    GELOGE(FAILED, "parameter ptr is null.");
    return FAILED;
  }
  // get the original anchors and remove the original link
  for (const auto &out_data_anchor : input_node->GetAllOutAnchors()) {
    for (auto &peer_in_anchor : out_data_anchor->GetPeerAnchors()) {
      if (peer_in_anchor->GetOwnerNode()->GetType() != kEndType) {
        if (GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
          GELOGE(FAILED, "[GraphPartitioner]: RemoveEdge() failed.");
          return FAILED;
        }
        // link input -> end
        auto ret = AddPlaceHolderEndInSrcDstGraph(out_data_anchor, peer_in_anchor, src_graph, dst_graph);
        if (ret != SUCCESS) {
          GELOGE(GE_GRAPH_ADD_PLC_END_FAILED, "[GraphPartitioner]: AddPlaceHolderEndInSrcDstGraph() failed.");
          return ret;
        }
      } else {
        auto end_node = peer_in_anchor->GetOwnerNode();
        if (GraphUtils::RemoveJustNode(src_graph, end_node) != GRAPH_SUCCESS) {
          GELOGE(FAILED, "[GraphPartitioner]: RemoveJustNode() failed.");
          return FAILED;
        }
        if (end_node->SetOwnerComputeGraph(dst_graph) != GRAPH_SUCCESS) {
          GELOGE(FAILED, "[GraphPartitioner]: RemoveJustNode() failed.");
          return FAILED;
        }
        if (dst_graph->AddNode(end_node) == nullptr) {
          GELOGE(FAILED, "[GraphPartitioner]: AddNode() failed.");
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

Status ge::GraphPartitioner::PutInputNodesInSubGraph(const ge::ComputeGraphPtr &src_graph,
                                                     const ge::ComputeGraphPtr &dst_graph) {
  if ((src_graph == nullptr) || (dst_graph == nullptr)) {
    GELOGE(FAILED, "parameter ptr is null.");
    return FAILED;
  }
  for (auto &input_node : src_graph->GetDirectNode()) {
    if (IsDataLike(input_node)) {
      if (input_node->SetOwnerComputeGraph(dst_graph) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "[GraphPartitioner]: SetOwnerComputeGraph failed.");
        return FAILED;
      }
      // remove input node from src_graph
      if (GraphUtils::RemoveJustNode(src_graph, input_node) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "[GraphPartitioner]: RemoveJustNode() failed.");
        return FAILED;
      }
      // add input node to dst_graph
      if (dst_graph->AddNode(input_node) == nullptr) {
        GELOGE(FAILED, "[GraphPartitioner]: AddNode() failed.");
        return FAILED;
      }
      if (LinkInput2EndRemoveOrginalLink(input_node, src_graph, dst_graph) != ge::SUCCESS) {
        GELOGE(FAILED, "[GraphPartitioner]: LinkInput2EndRemoveOrginalLink() failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

void ge::GraphPartitioner::AddNewGraphToPartition(ge::ComputeGraphPtr &input_graph, const std::string &engine_name) {
  if (input_graph == nullptr) {
    GELOGW("[GraphPartitioner]: input_graph is null, engine name is %s", engine_name.c_str());
    return;
  }
  graph_info_.partitions_[input_graph] = engine_name;
}

bool ge::GraphPartitioner::IsDataLike(ge::NodePtr node) {
  return (node->GetType() == CONSTANT) || (node->GetType() == DATA) || (node->GetType() == AIPPDATA) ||
         (node->GetType() == CONSTANTOP) || (node->GetType() == VARIABLE);
}

bool ge::GraphPartitioner::HasNoInput(ge::NodePtr node) {
  if (node == nullptr) {
    GELOGE(FAILED, "node_ptr is null.");
    return true;
  }
  return node->GetInNodes().empty();
}

Status ge::GraphPartitioner::Initialize(ge::ComputeGraphPtr compute_graph) {
  GELOGI("Initialize starts.");
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || compute_graph == nullptr) {
    GELOGE(GE_GRAPH_NOT_INIT, "Graph partitioner initialize failed.");
    return FAILED;
  }
  graph_info_.engine_placer_.SetComputeGraph(compute_graph);
  if (graph_info_.engine_placer_.Run() != SUCCESS) {
    GELOGE(FAILED, "Engine placer run failed.");
    return FAILED;
  }
  const NodeEngineMap *node_engine_map = graph_info_.engine_placer_.GetNodeEngineMap();
  size_t temp_index = 0;
  for (const auto &node : compute_graph->GetDirectNode()) {
    std::string temp_stream;
    // node opdesc has been checked before
    (void)AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, temp_stream);

    ClusterPtr new_cluster;
    // data like node without input should be handle specific
    if (HasNoInput(node) && IsDataLike(node)) {
      ClusterPtr cluster = MakeShared<Cluster>(temp_index, kEngineDefaultData, temp_stream);
      new_cluster = cluster;
    } else {
      ClusterPtr cluster = MakeShared<Cluster>(temp_index, node_engine_map->at(node), temp_stream);
      new_cluster = cluster;
    }
    if (new_cluster == nullptr) {
      GELOGE(FAILED, "[GraphPartitioner]: failed to allocate new_cluster");
      return FAILED;
    }
    new_cluster->nodes_.push_back(node);
    if (!HasNoInput(node)) {
      for (const auto &parent : node->GetInAllNodes()) {
        new_cluster->in_clu_.insert(graph_info_.node_2_cluster_.at(parent)->index_);
        graph_info_.node_2_cluster_.at(parent)->out_clu_.insert(temp_index);
      }
    }
    graph_info_.node_2_cluster_[node] = new_cluster;
    graph_info_.clusters_[temp_index] = new_cluster;
    GELOGD("Node name is %s, engine is %s, cluster index is %zu, stream label is %s", node->GetName().c_str(),
           new_cluster->engine_name_.c_str(), new_cluster->index_, new_cluster->stream_label_.c_str());
    temp_index++;
  }
  GELOGI("Initialize ends.");
  return SUCCESS;
}

Status ge::GraphPartitioner::AddPartitionsToGraphNode(vector<ge::SubGraphInfoPtr> &output_subgraphs,
                                                      ge::ComputeGraphPtr compute_graph) {
  const std::string &input_subgraph_name = "inputNodesSubGraph";
  string session_graph_id;
  if (!AttrUtils::GetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Get graph session_graph_id attr failed.");
    return INTERNAL_ERROR;
  }
  // the output_subgraphs have topological order
  for (const auto &sub_graph : graph_info_.rank_2_partitions_) {
    if (graph_info_.partitions_.find(sub_graph) == graph_info_.partitions_.end()) {
      GELOGE(GE_GRAPH_EMPTY_PARTITION, "[GraphPartitioner]: partition is null.");
      return FAILED;
    }
    auto &engine_name = graph_info_.partitions_.at(sub_graph);
    GE_DUMP(sub_graph, sub_graph->GetName());
    if (!session_graph_id.empty()) {
      GE_IF_BOOL_EXEC(!AttrUtils::SetStr(sub_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                      GELOGW("SetStr ATTR_NAME_SESSION_GRAPH_ID failed");)
    }
    // flush parent node of subgraph
    sub_graph->SetParentNode(compute_graph->GetParentNode());
    (void)AttrUtils::SetStr(*sub_graph, ATTR_NAME_PARENT_GRAPH_NAME, compute_graph->GetName());
    if (engine_name != input_subgraph_name) {  // do not add Data subGraph into SubGraphInfo
      auto sgi = MakeShared<SubGraphInfo>();
      if (sgi == nullptr) {
        GELOGE(GE_GRAPH_PARAM_NULLPTR, "[GraphPartitioner]: MakeShared sub graph info failed.");
        return FAILED;
      }
      // set engine name
      sgi->SetEngineName(engine_name);
      // set stream label
      string sub_graph_stream;
      if (AttrUtils::GetStr(sub_graph->GetDirectNode().at(0)->GetOpDesc(), ATTR_NAME_STREAM_LABEL, sub_graph_stream)) {
        sgi->SetStreamLabel(sub_graph_stream);
      }
      /// for now inputFlag is the same before and after partition. It should
      /// be changed according to the real partition
      std::vector<bool> sub_graph_input(graph_info_.input_size_, true);
      std::vector<bool> sub_graph_output(graph_info_.output_size_, true);
      sgi->SetSubGraph(sub_graph);
      sgi->SetOutputFlag(sub_graph_output);
      sgi->SetInputFlag(sub_graph_input);
      sgi->SetOutputContext(graph_info_.output_name_);
      AddEndPldInformationToSubGraphInfo(sgi);
      GELOGI("[GraphPartitioner]: subGraph engine name is %s, graph name is %s, stream label is %s",
             engine_name.c_str(), sub_graph->GetName().c_str(),
             sgi->GetStreamLabel().empty() ? "null" : sgi->GetStreamLabel().c_str());
      output_subgraphs.push_back(sgi);
    }
  }
  return SUCCESS;
}

// check if two clusters can merge
bool ge::GraphPartitioner::IsMergeable(size_t parent_cluster, size_t child_cluster, size_t upper_bound) {
  if ((graph_info_.clusters_[parent_cluster] == nullptr) || (graph_info_.clusters_[parent_cluster]->nodes_.empty()) ||
      (graph_info_.clusters_[child_cluster] == nullptr) || (graph_info_.clusters_[child_cluster]->nodes_.empty())) {
    return false;
  }
  // Check if parent_cluster,child_cluster has same engine or stream label
  if ((graph_info_.clusters_[parent_cluster]->engine_name_ != graph_info_.clusters_[child_cluster]->engine_name_) ||
      (graph_info_.clusters_[parent_cluster]->stream_label_ != graph_info_.clusters_[child_cluster]->stream_label_)) {
    GELOGD("Parent cluster %zu engine %s stream label %s, child cluster %zu engine %s stream label %s can not merge",
           parent_cluster, graph_info_.clusters_[parent_cluster]->engine_name_.c_str(),
           graph_info_.clusters_[parent_cluster]->stream_label_.c_str(), child_cluster,
           graph_info_.clusters_[child_cluster]->engine_name_.c_str(),
           graph_info_.clusters_[child_cluster]->stream_label_.c_str());
    return false;
  }
  // Check if parent_cluster,child_cluster is reachable
  RemoveEdge(parent_cluster, child_cluster);
  // Check if there is a path between parent and child, if return true, can not merge
  if (HasSecondPath(parent_cluster, child_cluster, upper_bound)) {
    GELOGD("Find second path from %zu to %zu, upper bound is %zu", parent_cluster, child_cluster, upper_bound);
    InsertEdge(parent_cluster, child_cluster);
    return false;
  }
  InsertEdge(parent_cluster, child_cluster);
  return true;
}

void ge::GraphPartitioner::MergeTwoClusters(size_t parent_cluster, size_t &child_cluster) {
  // check which index is bigger
  size_t big_cluster, small_cluster;
  size_t child_cluster_original = child_cluster;
  if (parent_cluster > child_cluster) {
    small_cluster = child_cluster;
    big_cluster = parent_cluster;
  } else {
    big_cluster = child_cluster;
    small_cluster = parent_cluster;
    // flush child_cluster, because it has been modified
    child_cluster = small_cluster;
  }

  // update node_2_cluster_ map
  for (auto &node : graph_info_.clusters_[big_cluster]->nodes_) {
    graph_info_.node_2_cluster_[node] = graph_info_.clusters_[small_cluster];
  }
  // merge nodes
  graph_info_.clusters_[small_cluster]->nodes_.splice(graph_info_.clusters_[small_cluster]->nodes_.end(),
                                                      graph_info_.clusters_[big_cluster]->nodes_);
  // merge all input & output to small cluster
  graph_info_.clusters_[small_cluster]->in_clu_.insert(graph_info_.clusters_[big_cluster]->in_clu_.begin(),
                                                       graph_info_.clusters_[big_cluster]->in_clu_.end());
  graph_info_.clusters_[small_cluster]->out_clu_.insert(graph_info_.clusters_[big_cluster]->out_clu_.begin(),
                                                        graph_info_.clusters_[big_cluster]->out_clu_.end());
  // remove child_cluster's out parent_cluster's in between child_cluster and parent_cluster
  RemoveEdge(parent_cluster, child_cluster_original);
  // update in/out of the cluster with bigger index
  for (auto in_clu : graph_info_.clusters_[big_cluster]->in_clu_) {
    graph_info_.clusters_[in_clu]->out_clu_.insert(small_cluster);
    graph_info_.clusters_[in_clu]->out_clu_.erase(big_cluster);
  }
  for (auto out_clu : graph_info_.clusters_[big_cluster]->out_clu_) {
    graph_info_.clusters_[out_clu]->in_clu_.insert(small_cluster);
    graph_info_.clusters_[out_clu]->in_clu_.erase(big_cluster);
  }
  graph_info_.clusters_[big_cluster] = graph_info_.clusters_[small_cluster];
}

void ge::GraphPartitioner::RemoveEdge(size_t parent_cluster, size_t child_cluster) {
  graph_info_.clusters_[child_cluster]->in_clu_.erase(parent_cluster);
  graph_info_.clusters_[parent_cluster]->out_clu_.erase(child_cluster);
}

void ge::GraphPartitioner::InsertEdge(size_t from, size_t to) {
  if (from == to) {
    return;
  }
  if (!graph_info_.clusters_[from]->out_clu_.insert(to).second) {
    // edge has already exists
    return;
  }
  graph_info_.clusters_[to]->in_clu_.insert(from);
}

void ge::GraphPartitioner::MarkClusters() {
  GELOGI("MarkClusters starts. cluster size is %zu", graph_info_.clusters_.size());
  size_t cluster_size = graph_info_.clusters_.size();
  for (size_t child_cluster = 0; child_cluster < cluster_size; child_cluster++) {
    auto found_child_cluster = graph_info_.clusters_[child_cluster];
    if (found_child_cluster == nullptr) {
      GELOGW("can not found child_cluster is %zu", child_cluster);
      continue;
    }
    auto copy_parents_clusters = found_child_cluster->in_clu_;
    vector<size_t> ordered_cluster;
    for (const auto &parent_cluster : copy_parents_clusters) {
      ordered_cluster.emplace_back(parent_cluster);
    }
    // sort cluster according to it's output amount
    auto comp_func = [this](const size_t &parent_cluster1, const size_t &parent_cluster2) -> bool {
      return graph_info_.clusters_[parent_cluster1]->out_clu_.size() <
             graph_info_.clusters_[parent_cluster2]->out_clu_.size();
    };
    std::sort(ordered_cluster.begin(), ordered_cluster.end(), comp_func);
    auto child_merged = child_cluster;
    for (const auto &parent_cluster : ordered_cluster) {
      if (IsMergeable(parent_cluster, child_merged, child_cluster)) {
        MergeTwoClusters(parent_cluster, child_merged);
        GELOGD("Merging cluster %zu and %zu to %zu", parent_cluster, child_cluster, child_merged);
      }
    }
  }
  GELOGI("MarkClusters ends.");
}

Status ge::GraphPartitioner::SplitSubGraphs(ge::ComputeGraphPtr compute_graph) {
  GELOGI("SplitSubGraphs starts.");
  if (compute_graph == nullptr) {
    GELOGE(FAILED, "parameter ptr is null.");
    return FAILED;
  }
  // Create graphs for all clusters
  std::unordered_set<ClusterPtr> cluster_set;
  // add pld&end
  for (auto &node : compute_graph->GetDirectNode()) {
    GELOGD("Node name is %s.", node->GetName().c_str());
    auto child_cluster = graph_info_.node_2_cluster_[node];
    ge::ComputeGraphPtr corresponding_graph;
    // unordered_set's insert returns a pair, second of pair is bool
    if (!cluster_set.insert(child_cluster).second) {
      GELOGD("Old sub graph, child_cluster is %zu", child_cluster->index_);
      corresponding_graph = graph_info_.cluster_2_partition_.at(child_cluster);
    } else {
      std::string graph_name = "new_sub_graph" + std::to_string(graph_info_.partitions_.size());
      ComputeGraphPtr new_sub_graph = MakeShared<ge::ComputeGraph>(graph_name);
      if (new_sub_graph == nullptr) {
        GELOGE(GE_GRAPH_PARAM_NULLPTR, "[GraphPartitioner]: MakeShared() failed.");
        return FAILED;
      }
      AddNewGraphToPartition(new_sub_graph, child_cluster->engine_name_);
      corresponding_graph = new_sub_graph;
      graph_info_.cluster_2_partition_[child_cluster] = corresponding_graph;
      GELOGD("New sub graph, name is %s", graph_name.c_str());
    }
    // build node to corresponding node map
    NodePtr corresponding_node = corresponding_graph->AddNode(node->GetOpDesc());
    if (corresponding_node == nullptr) {
      GELOGE(GE_GRAPH_PARAM_NULLPTR, "[GraphPartitioner]: AddNode() failed.");
      return FAILED;
    }
    graph_info_.corresponding_node_in_partitions_[node] = corresponding_node;
    GE_CHK_STATUS_RET(corresponding_node->SetOwnerComputeGraph(corresponding_graph))
    for (const auto &in_anchor : node->GetAllInAnchors()) {
      GELOGD("In anchor index is %d", AnchorUtils::GetIdx(in_anchor));
      for (auto &peer_out_anchor : in_anchor->GetPeerAnchors()) {
        GELOGD("Peer out anchor index is %d", AnchorUtils::GetIdx(peer_out_anchor));
        // All nodes have a copy in corresponding_node_in_partitions_, so function at can not be execption
        auto parent_node = graph_info_.corresponding_node_in_partitions_.at(peer_out_anchor->GetOwnerNode());
        GELOGD("Parent node name is %s", parent_node->GetName().c_str());
        // add edge
        auto src_anchor = parent_node->GetOutAnchor(AnchorUtils::GetIdx(peer_out_anchor));
        auto dst_anchor = corresponding_node->GetInAnchor(AnchorUtils::GetIdx(in_anchor));
        // if child and parent's cluster is not same, add plc and end
        auto parent_cluster = graph_info_.node_2_cluster_[peer_out_anchor->GetOwnerNode()];
        if (parent_cluster != child_cluster) {
          GELOGD("Parent cluster is %zu, child_cluster is %zu", parent_cluster->index_, child_cluster->index_);
          if (AddPlaceHolderEnd(peer_out_anchor, in_anchor) != ge::SUCCESS) {
            GELOGE(GE_GRAPH_ADD_PLC_END_FAILED, "[GraphPartitioner]: AddPlaceHolderEndInSrcDstGraph() failed.");
            return FAILED;
          }
        } else {  // parent and child in the same cluster, add edge
          GELOGD("AddEdge from parent cluster %zu to child %zu", parent_cluster->index_, child_cluster->index_);
          if (GraphUtils::AddEdge(src_anchor, dst_anchor) != GRAPH_SUCCESS) {
            GELOGE(GRAPH_FAILED, "AddEdge fail, from %s to %s", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                   in_anchor->GetOwnerNode()->GetName().c_str());
            return FAILED;
          }
        }
      }
    }
  }
  GELOGI("SplitSubGraphs ends.");
  return SUCCESS;
}

/// before calling this function, the direct path between src and dst are already removed.
/// return true if a second path is found
bool ge::GraphPartitioner::HasSecondPath(size_t src, size_t dst, size_t upper_bound) {
  if (graph_info_.clusters_.at(src)->out_clu_.empty() || graph_info_.clusters_.at(dst)->in_clu_.empty()) {
    return false;
  }
  /// Avoid recursion since stack space might be limited.
  /// We instead keep a stack of nodes to visit.
  std::vector<size_t> temp_stack;
  std::unordered_set<size_t> visited;
  temp_stack.push_back(src);
  while (!temp_stack.empty()) {
    size_t cluster = temp_stack.back();
    temp_stack.pop_back();
    ClusterPtr cur_cluster = graph_info_.clusters_[cluster];
    if (!visited.insert(cluster).second) {
      continue;
    }
    for (auto out : cur_cluster->out_clu_) {
      if (out == dst) {
        return true;  // There is cycle
      }
      if (out < upper_bound) {
        temp_stack.push_back(out);
      }
    }
  }
  return false;
}

Status ge::GraphPartitioner::Partition(ge::ComputeGraphPtr compute_graph, Mode mode) {
  graph_2_graph_partition_info_.clear();
  graph_2_subgraph_list_.clear();
  auto ret = PartitionSubGraph(compute_graph, mode);
  if (ret != SUCCESS) {
    GELOGE(ret, "Sub graph partition Failed");
    return ret;
  }
  // partition sub graph
  for (const auto &sub_graph : compute_graph->GetAllSubgraphs()) {
    ret = PartitionSubGraph(sub_graph, mode);
    if (ret != SUCCESS) {
      GELOGE(ret, "Sub graph partition Failed");
      return ret;
    }
  }
  return SUCCESS;
}

Status ge::GraphPartitioner::PartitionSubGraph(ge::ComputeGraphPtr compute_graph, Mode mode) {
  if (compute_graph == nullptr) {
    GELOGE(GE_GRAPH_NULL_INPUT, "[GraphPartitioner]: compute_graph is null.");
    return FAILED;
  }
  // clear graph_info
  graph_info_.ClearAllData(mode);
  graph_info_.output_name_ = compute_graph->GetOutput();
  graph_info_.output_size_ = compute_graph->GetOutputSize();
  graph_info_.input_size_ = compute_graph->GetInputSize();
  if (graph_info_.output_size_ == 0) {
    GELOGE(GE_GRAPH_NULL_INPUT, "The output size need to be greater than 0.");
    return FAILED;
  }
  GELOGI("Graph Partition starts, graph nodes size is %zu", compute_graph->GetDirectNodesSize());
  Status ret = compute_graph->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_TOPO_SORT_FAILED, "[GraphPartitioner]: subGraphPtr->TopologicalSorting failed");
    return FAILED;
  }
  GE_TIMESTAMP_START(PartitionSubGraphInitialize);
  if (Initialize(compute_graph) != SUCCESS) {
    GELOGE(GE_GRAPH_INIT_FAILED, "[GraphPartitioner]: initialize failed");
    return FAILED;
  }
  GE_TIMESTAMP_END(PartitionSubGraphInitialize, "GraphPartitioner::PartitionInitialize");
  GE_TIMESTAMP_START(PartitionSubGraphMarkClusters);
  MarkClusters();
  GE_TIMESTAMP_END(PartitionSubGraphMarkClusters, "GraphPartitioner::PartitionMarkClusters");
  GE_TIMESTAMP_START(PartitionSubGraphSplitSubGraphs);
  if (SplitSubGraphs(compute_graph) != SUCCESS) {
    GELOGE(FAILED, "[GraphPartitioner]: SplitSubGraphs failed");
    return FAILED;
  }
  GE_TIMESTAMP_END(PartitionSubGraphSplitSubGraphs, "GraphPartitioner::PartitionSplitSubGraphs");
  GE_TIMESTAMP_START(PartitionSubGraphSortSubGraphs);
  if (SortSubGraphs(compute_graph) != ge::SUCCESS) {
    GELOGE(GE_GRAPH_TOPO_SORT_FAILED, "Graph Partition SortSubGraphs failed.");
    return ge::FAILED;
  }
  GE_TIMESTAMP_END(PartitionSubGraphSortSubGraphs, "GraphPartitioner::PartitionSortSubGraphs");
  GE_TIMESTAMP_START(PartitionSubGraphAddPartitionsToGraphNode);
  vector<ge::SubGraphInfoPtr> output_subgraphs;
  if (AddPartitionsToGraphNode(output_subgraphs, compute_graph) != ge::SUCCESS) {
    GELOGE(GE_GRAPH_EMPTY_PARTITION, "Graph Partition AddPartitionsToGraphNode failed.");
    return ge::FAILED;
  }
  GE_TIMESTAMP_END(PartitionSubGraphAddPartitionsToGraphNode, "GraphPartitioner::PartitionAddPartitionsToGraphNode");
  GELOGI("Graph Partition ends. Adding partitions to SubGraphInfo, got %zu sub graphs", output_subgraphs.size());
  graph_info_.mode_ = kMerging;
  // do not care over flow
  partition_times_++;
  graph_2_graph_partition_info_[compute_graph] = graph_info_;
  graph_2_subgraph_list_[compute_graph] = output_subgraphs;
  return SUCCESS;
}

// all the inputs are the nodes and anchors in the original graph
Status ge::GraphPartitioner::AddPlaceHolderEnd(const AnchorPtr &out_anchor, const AnchorPtr &in_anchor) {
  if ((out_anchor == nullptr) || (in_anchor == nullptr)) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "src_node or dst_node is null.");
    return FAILED;
  }
  // nodes in original graph
  const auto &src_node = out_anchor->GetOwnerNode();
  const auto &dst_node = in_anchor->GetOwnerNode();
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "src_node or dst_node is null.");
    return FAILED;
  }
  // All nodes have a copy in corresponding_node_in_partitions_, so function at can not be execption
  auto src_anchor =
    graph_info_.corresponding_node_in_partitions_.at(src_node)->GetOutAnchor(AnchorUtils::GetIdx(out_anchor));
  auto dst_anchor =
    graph_info_.corresponding_node_in_partitions_.at(dst_node)->GetInAnchor(AnchorUtils::GetIdx(in_anchor));
  if ((src_anchor == nullptr) || (dst_anchor == nullptr)) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "src_anchor or dst_anchor is null.");
    return FAILED;
  }
  // anchors in subGraph
  const ComputeGraphPtr &src_subgraph = src_anchor->GetOwnerNode()->GetOwnerComputeGraph();
  const ComputeGraphPtr &dst_subgraph = dst_anchor->GetOwnerNode()->GetOwnerComputeGraph();
  // add end and pld node
  auto ret = AddPlaceHolderEndInSrcDstGraph(src_anchor, dst_anchor, dst_subgraph, src_subgraph);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_ADD_PLC_END_FAILED, "[GraphPartitioner]: add placeholder end failed.");
    return ret;
  }
  return SUCCESS;
}

Status ge::GraphPartitioner::SortSubGraphs(const ge::ComputeGraphPtr &compute_graph) {
  uint32_t rank = kRankOne;  // rank 0 for data graph
  ComputeGraphPtr new_input_nodes_sub_graph = MakeShared<ComputeGraph>("inputNodeGraph");
  if ((new_input_nodes_sub_graph == nullptr) || (compute_graph == nullptr)) {
    GELOGE(FAILED, "[GraphPartitioner]: new_input_nodes_sub_graph or compute_graph is null.");
    return FAILED;
  }
  for (const auto &node : compute_graph->GetDirectNode()) {
    // All nodes in original graph have a copy in corresponding_node_in_partitions_, so it can not be null
    auto sub_graph = graph_info_.corresponding_node_in_partitions_.at(node)->GetOwnerComputeGraph();
    if ((graph_info_.partitions_2_rank_.find(sub_graph) == graph_info_.partitions_2_rank_.end()) &&
        (graph_info_.partitions_[sub_graph] != kEngineDefaultData)) {
      graph_info_.partitions_2_rank_[sub_graph] = rank;
      graph_info_.rank_2_partitions_.push_back(sub_graph);
      rank++;
    } else if (graph_info_.partitions_[sub_graph] == kEngineDefaultData) {  // merge data graph
      if (PutInputNodesInSubGraph(sub_graph, new_input_nodes_sub_graph) != SUCCESS) {
        GELOGE(FAILED, "[GraphPartitioner]: putInputNodesInSubGraph failed.");
        return FAILED;
      }
      auto to_be_del = graph_info_.partitions_.find(sub_graph);
      graph_info_.partitions_.erase(to_be_del);
    }
  }
  if (!new_input_nodes_sub_graph->GetDirectNode().empty()) {
    graph_info_.rank_2_partitions_.insert(graph_info_.rank_2_partitions_.begin(), new_input_nodes_sub_graph);
    graph_info_.partitions_2_rank_[new_input_nodes_sub_graph] = 0;
    AddNewGraphToPartition(new_input_nodes_sub_graph, "inputNodesSubGraph");
  }
  // reinit rank
  rank = kRankZero;
  for (const auto &it : graph_info_.rank_2_partitions_) {
    // rename subGraph based on rank
    if (it != nullptr) {
      // rename subGraph based on rank
      string graph_name =
        "partition" + std::to_string(partition_times_) + "_rank" + std::to_string(rank) + "_" + it->GetName();
      it->SetName(graph_name);
    }
    rank++;
  }
  return SUCCESS;
}

AnchorPtr ge::GraphPartitioner::GetEndInAnchor(const AnchorPtr &src_anchor, const NodePtr &end_node) {
  if ((src_anchor == nullptr) || (end_node == nullptr)) {
    GELOGE(FAILED, "parameter ptr is null.");
    return nullptr;
  }
  AnchorPtr end_in_anchor;
  if (Anchor::DynamicAnchorCast<OutDataAnchor>(src_anchor) != nullptr) {
    end_in_anchor = end_node->GetInDataAnchor(0);
  } else {
    end_in_anchor = end_node->GetInControlAnchor();
  }
  return end_in_anchor;
}

AnchorPtr ge::GraphPartitioner::GetPldOutAnchor(const NodePtr &pld_node, const AnchorPtr &dst_anchor) {
  if ((pld_node == nullptr) || (dst_anchor == nullptr)) {
    GELOGE(FAILED, "parameter ptr is null.");
    return nullptr;
  }
  AnchorPtr pld_out_anchor;
  if (Anchor::DynamicAnchorCast<InDataAnchor>(dst_anchor) != nullptr) {
    pld_out_anchor = pld_node->GetOutDataAnchor(0);
  } else {
    pld_out_anchor = pld_node->GetOutControlAnchor();
  }
  return pld_out_anchor;
}

void ge::GraphPartitioner::AddEndPldInformationToSubGraphInfo(ge::SubGraphInfoPtr &subgraph_info) {
  if (subgraph_info == nullptr) {
    GELOGE(FAILED, "parameter ptr is null.");
    return;
  }
  auto subgraph = subgraph_info->GetSubGraph();
  GE_CHECK_NOTNULL_JUST_RETURN(subgraph);
  NodetoNodeMap end_map;
  NodetoNodeMap pld_map;
  for (const auto &node : subgraph->GetDirectNode()) {
    if (node->GetType() == kEndType) {
      end_map[node] = graph_info_.end_2_pld_.at(node);
    }
    if (node->GetType() == kPlaceHolderType) {
      pld_map[node] = graph_info_.pld_2_end_.at(node);
    }
  }
  subgraph_info->SetEnd2PldMap(end_map);
  subgraph_info->SetPld2EndMap(pld_map);
}

const Graph2SubGraphInfoList &ge::GraphPartitioner::GetSubGraphMap() { return graph_2_subgraph_list_; }
}  // namespace ge
