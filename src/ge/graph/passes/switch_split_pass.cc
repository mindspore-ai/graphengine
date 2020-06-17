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

#include "switch_split_pass.h"
#include <string>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

using namespace ge;
namespace {
const string output_false = "output_false";
const string output_true = "output_true";
string GetOutputDescName(const int idx) { return idx == 0 ? output_false : output_true; }
graphStatus CopyInDataEdges(const NodePtr &src_node, NodePtr &dst_node) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    GELOGE(GRAPH_FAILED, "Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto src_data_in_nodes = src_node->GetInDataNodes();
  if (src_data_in_nodes.empty()) {
    return GRAPH_SUCCESS;
  }
  for (const auto &in_data_anchor : src_node->GetAllInDataAnchors()) {
    auto input_desc = src_node->GetOpDesc()->GetInputDesc(in_data_anchor->GetIdx());
    auto ret =
      GraphUtils::AddEdge(in_data_anchor->GetPeerOutAnchor(), dst_node->GetInDataAnchor(in_data_anchor->GetIdx()));
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Failed to add data edge from %s to %s when copy in data edge from %s to %s",
             in_data_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(), dst_node->GetName().c_str(),
             src_node->GetName().c_str(), dst_node->GetName().c_str());
      return ret;
    }
  }
  return GRAPH_SUCCESS;
}
NodePtr CreateSwitchFromOld(const int index, const NodePtr &old_switch, const OutDataAnchorPtr &out_data_anchor) {
  auto graph = old_switch->GetOwnerComputeGraph();
  // 1. create new switch op desc
  string new_switch_name = old_switch->GetName() + "_" + std::to_string(index);
  auto new_switch_opdesc = MakeShared<OpDesc>(new_switch_name, old_switch->GetType());
  if (new_switch_opdesc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed to insert switch node, name %s", new_switch_name.c_str());
    return nullptr;
  }
  // 2. add input_desc & output_desc for new switch
  Status ret;
  for (const auto &in_data_anchor : old_switch->GetAllInDataAnchors()) {
    auto input_desc = old_switch->GetOpDesc()->GetInputDesc(in_data_anchor->GetIdx());
    ret = new_switch_opdesc->AddInputDesc(in_data_anchor->GetIdx(), input_desc);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Add Input desc failed for new switch %s.", new_switch_name.c_str());
      return nullptr;
    }
  }
  auto output_desc = old_switch->GetOpDesc()->GetOutputDesc(out_data_anchor->GetIdx());
  // we got out_data_anchor, another out_data_anchor is (1-idx), because idx is 0 or 1.
  auto ret1 = new_switch_opdesc->AddOutputDesc(GetOutputDescName(1 - out_data_anchor->GetIdx()), output_desc);
  auto ret2 = new_switch_opdesc->AddOutputDesc(GetOutputDescName(out_data_anchor->GetIdx()), output_desc);
  if (ret1 != SUCCESS || ret2 != SUCCESS) {
    GELOGE(FAILED, "Add Output desc failed for new switch %s.", new_switch_name.c_str());
    return nullptr;
  }
  GELOGI("Insert new switch node %s.", new_switch_name.c_str());
  return graph->AddNode(new_switch_opdesc);
}
}  // namespace
namespace ge {
Status SwitchSplitPass::Run(NodePtr &node) {
  // To handle one out data anchor with multi peer input data anchor
  GE_CHECK_NOTNULL(node);
  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetType() != SWITCH && op_desc->GetType() != REFSWITCH) {
    return SUCCESS;
  }
  if (op_desc->GetName().find("apply_one_adam") == string::npos) {
    // Currently for bert optimize, will fix later.
    GELOGI("Current switch node name is %s, ignore it.", op_desc->GetName().c_str());
    return SUCCESS;
  }
  GELOGI("Switch split pass in. Current switch node name is %s", op_desc->GetName().c_str());
  int index = 0;
  // 1. find all output
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    if (out_data_anchor->GetPeerInDataNodesSize() < 2) {
      GELOGI("Switch node %s %d th out data anchor only has 1 peer_in_data_anchor.Ignore it.", node->GetName().c_str(),
             out_data_anchor->GetIdx());
      continue;
    }
    for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      NodePtr new_switch = CreateSwitchFromOld(index, node, out_data_anchor);
      if (new_switch == nullptr) {
        GELOGW("Insert switch node failed.");
        return FAILED;
      }
      // 1.3 copy int/out edge from old switch to new switch
      auto ret1 = CopyInDataEdges(node, new_switch);
      auto ret2 = GraphUtils::CopyInCtrlEdges(node, new_switch);
      auto ret3 = GraphUtils::CopyOutCtrlEdges(node, new_switch);
      if (ret1 != GRAPH_SUCCESS || ret2 != GRAPH_SUCCESS || ret3 != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Copy edge from %s to %s failed.", node->GetName().c_str(), new_switch->GetName().c_str());
        return FAILED;
      }
      if (out_data_anchor->Unlink(peer_in_anchor) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Unlink from old switch %s out data anchor %d to peer in anchor failed.",
               node->GetName().c_str(), out_data_anchor->GetIdx());
      }
      auto ret4 = GraphUtils::AddEdge(new_switch->GetOutDataAnchor(out_data_anchor->GetIdx()), peer_in_anchor);
      if (ret4 != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Replace out data edge from old switch %s to new switch %s failed.", node->GetName().c_str(),
               new_switch->GetName().c_str());
        return FAILED;
      }
      AddRePassNode(new_switch);
      index++;
    }
  }
  // 2.isolate switch node with no data output
  if (node->GetOutDataNodesSize() == 0) {
    auto ret = IsolateAndDeleteNode(node, {});
    if (ret != SUCCESS) {
      GELOGE(FAILED, "IsolateAndDelete switch node %s.", node->GetName().c_str());
      return FAILED;
    }
    GELOGI("IsolateAndDelete switch node %s.", node->GetName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge
