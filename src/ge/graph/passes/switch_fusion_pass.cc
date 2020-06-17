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

#include "switch_fusion_pass.h"
#include <sstream>
#include <string>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
namespace ge {
namespace {
const int kSwitchDataInputIdx = 0;
const int kSwitchCondInputIdx = 1;
const int kSwitchFalseOutIdx = 0;
const int kSwitchTrueOutIdx = 1;
int GetSwitchOutDataIdx(const string fusion_group_id) { return std::stoi(fusion_group_id.substr(0)); }
}  // namespace

Status SwitchFusionPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  if (node->GetOpDesc()->GetType() != SWITCH && node->GetOpDesc()->GetType() != REFSWITCH) {
    return SUCCESS;
  }
  GELOGD("Switch fusion pass in.Current switch node name is %s", node->GetName().c_str());
  // 1. find cond input
  auto switch_in_cond_anchor = node->GetInDataAnchor(kSwitchCondInputIdx);
  if (switch_in_cond_anchor->GetPeerOutAnchor() == nullptr) {
    GELOGI("Switch %s in condition peer out anchor is null.", node->GetName().c_str());
    return FAILED;
  }
  auto switch_cond_in_node = switch_in_cond_anchor->GetPeerOutAnchor()->GetOwnerNode();
  GELOGD("Switch %s cond in data node is %s.", node->GetName().c_str(), switch_cond_in_node->GetName().c_str());
  if (switch_cond_in_node->GetOutDataNodesSize() == 1) {
    GELOGI("This condition only has one switch, no need fusion.");
    return SUCCESS;
  }
  // 2. find other switch with same condition
  for (const auto out_data_node : switch_cond_in_node->GetOutDataNodes()) {
    if (out_data_node->GetType() == SWITCH || out_data_node->GetType() == REFSWITCH) {
      // 2.1 collect switch node can be fused with same cond_in_node
      auto true_out_anchor = out_data_node->GetOutDataAnchor(kSwitchTrueOutIdx);
      auto false_out_anchor = out_data_node->GetOutDataAnchor(kSwitchFalseOutIdx);
      int branch_idx = true_out_anchor == nullptr ? kSwitchFalseOutIdx : kSwitchTrueOutIdx;
      if (out_data_node->GetOutDataAnchor(branch_idx)->GetPeerInDataNodesSize() > 1) {
        GELOGI("Current switch node %s has more than one output, need go to switch split first.",
               out_data_node->GetName().c_str());
        continue;
      }
      string fusion_road_id;
      fusion_road_id = GetFusionRoadId(std::to_string(branch_idx), out_data_node);
      GELOGI("Switch node %s out idx %d, group_id is %s.", out_data_node->GetName().c_str(), branch_idx,
             fusion_road_id.c_str());
      auto iter = switch_group_map_.find(fusion_road_id);
      if (iter == switch_group_map_.end()) {
        switch_group_map_.emplace(std::make_pair(fusion_road_id, std::set<NodePtr>{out_data_node}));
      } else {
        // to avoid one cond node is also as data node
        if (iter->second.count(out_data_node) == 0) {
          iter->second.emplace(out_data_node);
        }
      }
    }
  }
  // 3. fuse switch from different group
  auto ret = FuseSwitchGroup();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Fuse switch nodes with same final output to one failed.");
    return ret;
  }
  return SUCCESS;
}
/*
 *       var1  ALLREDUCE/Cast      var3                 var1   var2    var3     ALLREDUCE/Cast
 *        \  /        \     \    /                       \     |        / \      / \
 *      switch1    switch2   switch3       ======>     AdamApplyOne     /   \--->switch1
 *             \     |          /                             \       /          |
 *           AdamApplyOne      /                                 mul    <--- identity
 *                  \        /
 *                    mul
 */
Status SwitchFusionPass::FuseSwitchGroup() {
  for (auto &key_2_switch_group : switch_group_map_) {
    if (key_2_switch_group.second.size() == 1) {
      break;
    }
    // 1.Insert Identity node
    NodePtr remain_switch = *key_2_switch_group.second.begin();
    auto switch_out_anchor_idx = GetSwitchOutDataIdx(key_2_switch_group.first);
    auto identity_node = InsertIdentityNode(remain_switch, switch_out_anchor_idx);
    if (identity_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "Create Identity op %s fail.", identity_node->GetName().c_str());
      return FAILED;
    }
    // 2. Remove all switch nodes between data anchors.
    string hccl_group_id;
    for (const auto &switch_node : key_2_switch_group.second) {
      GELOGI("Get corresponding SWITCH node is %s.Out data anchor idx is %d.", switch_node->GetName().c_str(),
             switch_out_anchor_idx);
      // get hccl group id for remain switch
      if (AttrUtils::GetStr(switch_node->GetOpDesc(), ATTR_NAME_HCCL_FUSED_GROUP, hccl_group_id)) {
        GELOGI("Get hccl group id %s of switch node %s.", hccl_group_id.c_str(), switch_node->GetName().c_str());
      }
      auto switch_peer_in_data_anchor =
        switch_node->GetOutDataAnchor(switch_out_anchor_idx)->GetPeerInDataAnchors().at(0);
      GE_RETURN_WITH_LOG_IF_ERROR(RemoveSwitchBetweenTwoNode(switch_out_anchor_idx, switch_node));
      GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::AddEdge(identity_node->GetOutControlAnchor(),
                                                      switch_peer_in_data_anchor->GetOwnerNode()->GetInControlAnchor()),
                                  "Link control edge from identity %s to out node %s.",
                                  identity_node->GetName().c_str(),
                                  switch_peer_in_data_anchor->GetOwnerNode()->GetName().c_str());
    }
    GELOGI("Start fusion switch nodes. Switch_nodes_set size is %d", key_2_switch_group.second.size());
    // 3.Fuse all switch to one, first is remain_switch
    GE_RETURN_WITH_LOG_IF_ERROR(FuseSwitchNodesToOne(remain_switch, key_2_switch_group.second));
    if (!hccl_group_id.empty()) {
      AttrUtils::SetStr(remain_switch->GetOpDesc(), ATTR_NAME_HCCL_FUSED_GROUP, hccl_group_id);
      GELOGI("Set attr ATTR_NAME_HCCL_FUSED_GROUP for Stream node %s, value is %s.", remain_switch->GetName().c_str(),
             hccl_group_id.c_str());
    }
    // Link switch to identity
    GraphUtils::AddEdge(remain_switch->GetOutDataAnchor(switch_out_anchor_idx), identity_node->GetInDataAnchor(0));
  }
  return SUCCESS;
}
/*
 *    var1----
 *    cond----
 *    var2----
 */
Status SwitchFusionPass::RemoveSwitchBetweenTwoNode(const int switch_out_anchor_idx, const NodePtr &switch_node) {
  auto switch_in_data_anchor = switch_node->GetInDataAnchor(kSwitchDataInputIdx);
  auto switch_in_cond_anchor = switch_node->GetInDataAnchor(kSwitchCondInputIdx);
  // here we assume after switch split, one switch node only has one data output,so just get first is ok.
  auto switch_peer_in_data_anchor = switch_node->GetOutDataAnchor(switch_out_anchor_idx)->GetPeerInDataAnchors().at(0);
  // 2.1 unlink all data edge from switch to out_node
  GE_RETURN_WITH_LOG_IF_ERROR(
    GraphUtils::RemoveEdge(switch_node->GetOutDataAnchor(switch_out_anchor_idx), switch_peer_in_data_anchor),
    "Remove edge from switch %s to out node %s.", switch_node->GetName().c_str(),
    switch_peer_in_data_anchor->GetOwnerNode()->GetName().c_str());
  // 2.2 replace data edge from switch_data_in_node to switch_data_out_node
  if (switch_in_data_anchor->GetPeerOutAnchor() == nullptr) {
    GELOGI("Switch %s in data peer out anchor is null.", switch_node->GetName().c_str());
    return FAILED;
  }
  auto switch_in_node = switch_in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
  GELOGI("Switch %s in data node is %s.", switch_node->GetName().c_str(), switch_in_node->GetName().c_str());
  GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::ReplaceEdgeDst(switch_in_data_anchor->GetPeerOutAnchor(),
                                                         switch_in_data_anchor, switch_peer_in_data_anchor),
                              "ReplaceEdgeDst from switch_data_in_node %s to switch_out_node %s.",
                              switch_in_node->GetName().c_str(),
                              switch_peer_in_data_anchor->GetOwnerNode()->GetName().c_str());
  // 2.3 link control edge from switch_data_in_node to switch
  GE_RETURN_WITH_LOG_IF_ERROR(
    GraphUtils::AddEdge(switch_in_node->GetOutControlAnchor(), switch_node->GetInControlAnchor()),
    "Link control edge from switch_data_in_node %s to switch node %s failed.", switch_in_node->GetName().c_str(),
    switch_node->GetName().c_str());
  return SUCCESS;
}

Status SwitchFusionPass::FuseSwitchNodesToOne(NodePtr &remain_switch, const std::set<NodePtr> switch_nodes_set) {
  auto iter = ++switch_nodes_set.begin();
  while (iter != switch_nodes_set.end()) {
    GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::CopyInCtrlEdges(*iter, remain_switch),
                                "Copy in control edge from %s to %s failed.", (*iter)->GetName().c_str(),
                                remain_switch->GetName().c_str());
    GE_RETURN_WITH_LOG_IF_ERROR(NodeUtils::MoveOutputEdges(*iter, remain_switch),
                                "Move output edges from %s to %s failed.", (*iter)->GetName().c_str(),
                                remain_switch->GetName().c_str());
    if ((*iter)->GetOutDataNodesSize() == 0) {
      auto ret = IsolateAndDeleteNode(const_cast<NodePtr &>(*iter), {});
      if (ret == SUCCESS) {
        GELOGI("IsolateAndDeleteNode Switch node %s", (*iter)->GetName().c_str());
      }
    } else {
      GELOGI("Switch node %s has more than one out data nodes, keep it.", (*iter)->GetName().c_str());
    }
    iter++;
  }
  // link data input for remain switch
  auto cond_node = remain_switch->GetInDataAnchor(kSwitchCondInputIdx)->GetPeerOutAnchor()->GetOwnerNode();
  GELOGI("Get cond node %s of switch node %s.", cond_node->GetName().c_str(), remain_switch->GetName().c_str());
  GE_RETURN_WITH_LOG_IF_ERROR(
    GraphUtils::AddEdge(cond_node->GetOutDataAnchor(0), remain_switch->GetInDataAnchor(kSwitchDataInputIdx)),
    "Fail to add edge from cond_node %s to remain_switch %s.", cond_node->GetName().c_str(),
    remain_switch->GetName().c_str());
  return SUCCESS;
}

const string SwitchFusionPass::GetFusionRoadId(const string branch_id, const NodePtr &switch_node) {
  std::deque<NodePtr> queue;
  queue.push_back(switch_node);
  std::stringstream group_id;
  group_id << branch_id;

  while (!queue.empty()) {
    NodePtr node = queue.front();
    queue.pop_front();
    if (node->GetOutDataNodesSize() == 0) {
      group_id << "-" << node->GetName();
      GELOGI("Switch node %s, group id is %s", switch_node->GetName().c_str(), group_id.str().c_str());
      return group_id.str();
    }
    for (const auto &out_data_node : node->GetOutDataNodes()) {
      if (out_data_node->GetType() == NETOUTPUT || out_data_node->GetType() == SWITCH ||
          out_data_node->GetType() == SWITCH) {
        // if meet NETOUTPUT, it is the end of current ROAD
        group_id << "-" << node->GetName();
        GELOGI("Switch node %s, group id is %s", switch_node->GetName().c_str(), group_id.str().c_str());
        return group_id.str();
      }
      queue.emplace_back(out_data_node);
    }
  }
  return group_id.str();
}
NodePtr SwitchFusionPass::InsertIdentityNode(const NodePtr &remain_switch, const int out_data_anchor_idx) {
  const std::string identity_name = remain_switch->GetOpDesc()->GetName() + "_" + IDENTITY;
  ComputeGraphPtr graph = remain_switch->GetOwnerComputeGraph();
  auto data_desc = remain_switch->GetOpDesc()->GetOutputDesc(out_data_anchor_idx);
  OpDescPtr op_desc = MakeShared<OpDesc>(identity_name, IDENTITY);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create Identity op %s: create op_desc fail.", identity_name.c_str());
    return nullptr;
  }
  if ((op_desc->AddInputDesc(data_desc) != GRAPH_SUCCESS) || (op_desc->AddOutputDesc(data_desc) != GRAPH_SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "Create Identity op %s: add input/output desc fail.", identity_name.c_str());
    return nullptr;
  }
  GELOGI("Create Identity op:%s.", identity_name.c_str());
  return graph->AddNode(op_desc);
}
}  // namespace ge