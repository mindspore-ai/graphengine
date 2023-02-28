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

#include "graph/partition/default_subgraph_builder.h"
#include "graph/utils/graph_utils.h"
#include "common/types.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/log.h"

namespace ge {
bool DefaultSubgraphBuilder::AddFrameInput(const InDataAnchorPtr &anchor) {
  if ((anchor == nullptr) ||
      (anchor->GetPeerOutAnchor() == nullptr)) {
    return true;
  }
  bool added = true;
  auto index = inputs_.size();
  if (NeedMergeInputs()) {
    GELOGD("Merge inputs is enabled");
    auto src_node = anchor->GetPeerOutAnchor()->GetOwnerNode();
    std::string src_key = src_node->GetName() + ":" + std::to_string(anchor->GetPeerOutAnchor()->GetIdx());
    std::map<std::string, size_t>::const_iterator it = src_key_to_frame_input_index_.find(src_key);
    if (it != src_key_to_frame_input_index_.cend()) {
      index = it->second;
      GELOGD("[%s:%d] Reuse data index: %zu", anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx(),
             it->second);
      added = false;
    } else {
      inputs_.push_back(anchor);
      GELOGD("[%s:%d] Assign data index: %zu", anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx(), index);
      src_key_to_frame_input_index_[src_key] = index;
    }
  } else {
    inputs_.push_back(anchor);
  }
  frame_input_index_to_inputs_[index].emplace_back(anchor);
  return added;
}

void DefaultSubgraphBuilder::AddFrameOutput(const OutDataAnchorPtr &anchor) {
  if (anchor != nullptr) {
    outputs_.push_back(anchor);
  }
}

graphStatus DefaultSubgraphBuilder::AddDataInput(const NodePtr &node) {
  const auto &dst_graph = node->GetOwnerComputeGraph();
  for (const auto &anchor : node->GetAllInDataAnchors()) {
    const auto &peer_out_anchor = anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;  // Skip overhang input.
    }
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    const auto &src_graph = src_node->GetOwnerComputeGraph();
    if (src_graph != dst_graph) {
      if (AddFrameInput(anchor)) {
        GE_ASSERT_GRAPH_SUCCESS(partitioned_call_op_->AddInputDesc(node->GetOpDesc()->GetInputDesc(anchor->GetIdx())),
                                "[Add][InputDesc] to op:%s failed.", partitioned_call_op_->GetName().c_str());
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::AddDataOutput(const NodePtr &node) {
  for (const auto &anchor : node->GetAllOutDataAnchors()) {
    const auto &peer_in_anchors = anchor->GetPeerInDataAnchors();
    for (const auto &peer_in_anchor : peer_in_anchors) {
      NodesClusterPtr dst_cluster = GetNodeCluster(peer_in_anchor->GetOwnerNode());
      if (dst_cluster->Id() != GetCluster()->Id()) {
        AddFrameOutput(anchor);
        GE_ASSERT_GRAPH_SUCCESS(partitioned_call_op_->AddOutputDesc(node->GetOpDesc()->GetOutputDesc(anchor->GetIdx())),
                                "[Add][OutputDesc] to op:%s failed.", partitioned_call_op_->GetName().c_str());
        break; // skip loop peer_in_anchors
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::AddControlInput(const NodePtr &node) {
  const auto &in_control_anchor = node->GetInControlAnchor();
  const auto &dst_graph = node->GetOwnerComputeGraph();
  if (in_control_anchor != nullptr) {
    for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
      if (peer_out_control_anchor == nullptr) {
        continue;
      }
      NodePtr src_node = peer_out_control_anchor->GetOwnerNode();
      const auto &src_graph = src_node->GetOwnerComputeGraph();
      if (src_graph != dst_graph) {
        control_inputs_.insert(in_control_anchor);
        break;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::AddControlOutput(const NodePtr &node) {
  const auto &out_control_anchor = node->GetOutControlAnchor();
  if (out_control_anchor != nullptr) {
    for (const auto &peer_in_control_anchor : out_control_anchor->GetPeerInControlAnchors()) {
      if (peer_in_control_anchor == nullptr) {
        continue;
      }
      const auto &dst_cluster = GetNodeCluster(peer_in_control_anchor->GetOwnerNode());
      GE_ASSERT_NOTNULL(dst_cluster, "[Check][DstCluster] failed, node:%s", node->GetName().c_str());
      if (dst_cluster->Id() != GetCluster()->Id()) {
        control_outputs_.insert(out_control_anchor);
        break;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::BuildPartitionedCallNodeIO() {
  for (const auto &node : GetCluster()->Nodes()) {
    // add data inputs
    GE_ASSERT_GRAPH_SUCCESS(AddDataInput(node), "[Call][AddDataInput] according to node %s failed.",
                            node->GetName().c_str());
    // add data outputs
    GE_ASSERT_GRAPH_SUCCESS(AddDataOutput(node), "[Call][AddDataOutput] according to node %s failed.",
                            node->GetName().c_str());
    // add control_input
    GE_ASSERT_GRAPH_SUCCESS(AddControlInput(node), "[Call][AddControlInput] according to node %s failed.",
                            node->GetName().c_str());
    // add control_output
    GE_ASSERT_GRAPH_SUCCESS(AddControlOutput(node), "[Call][AddControlOutput] according to node %s failed.",
                            node->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::MakePartitionedCallNode() {
  partitioned_call_node_ = GetRootGraph()->AddNode(partitioned_call_op_);
  GE_ASSERT_NOTNULL(partitioned_call_node_, "[Add][Node] %s to graph:%s failed.",
                    partitioned_call_op_->GetName().c_str(), GetRootGraph()->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(partitioned_call_node_->SetOwnerComputeGraph(GetRootGraph()),
                          "[Set][OwnerComputeGraph] %s for node:%s failed.", GetRootGraph()->GetName().c_str(),
                          partitioned_call_node_->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::MakePartitionedCallOp(const std::string &sub_graph_name) {
  partitioned_call_op_ =
      MakeShared<OpDesc>(sub_graph_name + "_PartitionedCall_" + std::to_string(unique_id_), PARTITIONEDCALL);
  GE_ASSERT_NOTNULL(partitioned_call_op_, "[New][Memory] for partition op failed.");
  GE_ASSERT_GRAPH_SUCCESS(partitioned_call_op_->AddSubgraphName(sub_graph_name), "[Add][SubgraphName] %s for op:%s.",
                          sub_graph_name.c_str(), partitioned_call_op_->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(partitioned_call_op_->SetSubgraphInstanceName(0, sub_graph_name),
                          "[Call][SetSubgraphInstanceName] for op:%s failed, index:0, name:%s.",
                          partitioned_call_op_->GetName().c_str(), sub_graph_name.c_str());
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::MakeSubgraph(const std::string &sub_graph_name) {
  subgraph_ = MakeShared<ComputeGraph>(sub_graph_name);
  GE_ASSERT_NOTNULL(subgraph_, "[New][Memory] for subgraph failed, name:%s.", sub_graph_name.c_str());
  subgraph_->SetParentGraph(GetRootGraph());
  for (const auto &node : GetCluster()->Nodes()) {
    GE_ASSERT_NOTNULL(subgraph_->AddNode(node), "[Add][Node] failed.");
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(GetRootGraph(), node),
                            "[Remove][JustNode] failed, graph:%s, node:%s.", GetRootGraph()->GetName().c_str(),
                            node->GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(node->SetOwnerComputeGraph(subgraph_), "[Set][OwnerComputeGraph] %s for node:%s failed.",
                            subgraph_->GetName().c_str(), node->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::BuildSubgraphFrame() {
  std::string sub_graph_name =
      GetRootGraph()->GetName() + "Sub" + GetPartitionerName() + std::to_string(unique_id_);
  GE_ASSERT_GRAPH_SUCCESS(MakePartitionedCallOp(sub_graph_name),
                          "[Call][MakePartitionedCallOp] failed, subgraph name:%s.", sub_graph_name.c_str());
  GE_ASSERT_GRAPH_SUCCESS(MakeSubgraph(sub_graph_name), "[Call][MakeSubgraph] for node%s failed.",
                          partitioned_call_op_->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(BuildPartitionedCallNodeIO(), "[Call][SetSubgraphInstanceName] for node:%s failed, name:%s.",
                          partitioned_call_node_->GetName().c_str(), subgraph_->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(MakePartitionedCallNode(), "[Call][MakePartitionedCallNode] for node:%s failed, name:%s.",
                          partitioned_call_node_->GetName().c_str(), subgraph_->GetName().c_str());
  // all partition add subgraph to root_graph
  subgraph_->SetParentNode(partitioned_call_node_);
  const auto root_graph = GraphUtils::FindRootGraph(GetRootGraph());
  GE_ASSERT_GRAPH_SUCCESS(root_graph->AddSubgraph(subgraph_), "[Add][Subgraph] %s to root graph:%s failed.",
                          subgraph_->GetName().c_str(), GetRootGraph()->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::CombineToPartitionedCallDataInputs() {
  for (auto id = 0U; id < inputs_.size(); id++) {
    const auto &anchor = inputs_[id];
    const auto &src_anchor = anchor->GetPeerOutAnchor();
    const auto &src_node = src_anchor->GetOwnerNode();
    const auto &src_graph = src_node->GetOwnerComputeGraph();
    if (src_graph != subgraph_) {
      const auto &partitioned_node_graph = partitioned_call_node_->GetOwnerComputeGraph();
      GE_ASSERT_TRUE(partitioned_node_graph == src_graph,
                     "[Check][Graph] invalid, src graph[%s] is different from self graph[%s], can not combine.",
                     src_graph->GetName().c_str(), partitioned_node_graph->GetName().c_str());
      for (const auto &node_anchor : frame_input_index_to_inputs_[id]) {
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(src_anchor, node_anchor),
                                "[Remove][Edge] from %s:%d to %s:%d fail.",
                                src_node->GetName().c_str(), src_anchor->GetIdx(),
                                node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx());
      }
      const auto &partitioned_node_in_anchor = partitioned_call_node_->GetInDataAnchor(id);
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(src_anchor, partitioned_node_in_anchor),
                              "[Add][Edge] from %s:%d to %s:%d failed.", src_anchor->GetOwnerNode()->GetName().c_str(),
                              src_anchor->GetIdx(), partitioned_node_in_anchor->GetOwnerNode()->GetName().c_str(),
                              partitioned_node_in_anchor->GetIdx());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::CombineToPartitionedCallDataOutputs() const {
  for (auto id = 0U; id < outputs_.size(); id++) {
    const auto &out_anchor = outputs_[id];
    const auto &dst_anchors = out_anchor->GetPeerAnchors();
    for (const auto &dst_anchor : dst_anchors) {
      const auto &dst_node = dst_anchor->GetOwnerNode();
      const auto &dst_graph = dst_node->GetOwnerComputeGraph();
      if (dst_graph != subgraph_) {
        const auto &partitioned_node_graph = partitioned_call_node_->GetOwnerComputeGraph();
        GE_ASSERT_TRUE(partitioned_node_graph == dst_graph,
                       "[Check][Graph] invalid, src graph[%s] is different from self graph[%s], can not combine.",
                       dst_graph->GetName().c_str(), partitioned_node_graph->GetName().c_str());
        const auto &partitioned_node_out_anchor = partitioned_call_node_->GetOutDataAnchor(id);
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(out_anchor, dst_anchor),
                                "[Remove][Edge] from %s:%d to %s:%d fail.",
                                out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetIdx(),
                                dst_node->GetName().c_str(), dst_anchor->GetIdx());
        GE_ASSERT_GRAPH_SUCCESS(
            GraphUtils::AddEdge(partitioned_node_out_anchor, dst_anchor), "[Add][Edge] from %s:%d to %s:%d failed.",
            dst_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetIdx(),
            partitioned_node_out_anchor->GetOwnerNode()->GetName().c_str(), partitioned_node_out_anchor->GetIdx());
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::CombineToPartitionedCallControlInput() const {
  for (const auto &control_in_anchor : control_inputs_) {
    const auto &src_anchors = control_in_anchor->GetPeerAnchors();
    for (const auto &src_anchor : src_anchors) {
      const auto &src_node = src_anchor->GetOwnerNode();
      const auto &src_graph = src_node->GetOwnerComputeGraph();
      if (src_graph != subgraph_) {
        const auto &partitioned_node_graph = partitioned_call_node_->GetOwnerComputeGraph();
        GE_ASSERT_TRUE(partitioned_node_graph == src_graph,
                       "[Check][Graph] invalid, src graph[%s] is different from self graph[%s], can not combine.",
                       src_graph->GetName().c_str(), partitioned_node_graph->GetName().c_str());
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(src_anchor, control_in_anchor),
                                "[Remove][Edge] from %s:%d to %s:%d fail.",
                                src_node->GetName().c_str(), src_anchor->GetIdx(),
                                control_in_anchor->GetOwnerNode()->GetName().c_str(), control_in_anchor->GetIdx());
        const auto &partitioned_node_in_control_anchor = partitioned_call_node_->GetInControlAnchor();
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(src_anchor, partitioned_node_in_control_anchor),
                                "[Add][Edge] from %s:%d to %s:%d failed.",
                                src_anchor->GetOwnerNode()->GetName().c_str(), src_anchor->GetIdx(),
                                partitioned_node_in_control_anchor->GetOwnerNode()->GetName().c_str(),
                                partitioned_node_in_control_anchor->GetIdx());
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::CombineToPartitionedCallControlOutput() const {
  for (const auto &control_out_anchor : control_outputs_) {
    const auto &dst_anchors = control_out_anchor->GetPeerAnchors();
    for (const auto &dst_anchor : dst_anchors) {
      const auto &dst_node = dst_anchor->GetOwnerNode();
      const auto &dst_graph = dst_node->GetOwnerComputeGraph();
      if (dst_graph != subgraph_) {
        const auto &partitioned_node_graph = partitioned_call_node_->GetOwnerComputeGraph();
        GE_ASSERT_TRUE(partitioned_node_graph == dst_graph,
                       "[Check][Graph] invalid, src graph[%s] is different from self graph[%s], can not combine.",
                       dst_graph->GetName().c_str(), partitioned_node_graph->GetName().c_str());
        const auto &partitioned_node_out_control_anchor = partitioned_call_node_->GetOutControlAnchor();
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(control_out_anchor, dst_anchor),
                                "[Remove][Edge] from %s:%d to %s:%d fail.",
                                control_out_anchor->GetOwnerNode()->GetName().c_str(), control_out_anchor->GetIdx(),
                                dst_node->GetName().c_str(), dst_anchor->GetIdx());
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(partitioned_node_out_control_anchor, dst_anchor),
                                "[Add][Edge] from %s:%d to %s:%d failed.",
                                dst_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetIdx(),
                                partitioned_node_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                                partitioned_node_out_control_anchor->GetIdx());
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::CombinePartitionedCall() {
  GE_ASSERT_GRAPH_SUCCESS(CombineToPartitionedCallDataInputs(), "[Call][CombineToPartitionedCallDataInputs] failed.");
  GE_ASSERT_GRAPH_SUCCESS(CombineToPartitionedCallDataOutputs(), "[Call][CombineToPartitionedCallDataOutputs] failed.");
  GE_ASSERT_GRAPH_SUCCESS(CombineToPartitionedCallControlInput(),
                          "[Call][CombineToPartitionedCallControlInput] failed.");
  GE_ASSERT_GRAPH_SUCCESS(CombineToPartitionedCallControlOutput(),
                          "[Call][CombineToPartitionedCallControlOutput] failed.");
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::MakeDataNode(const size_t parent_node_index,
                                                 const InDataAnchorPtr &data_anchor,
                                                 NodePtr &data_node) {
  // construct data op
  const auto data_op = MakeShared<OpDesc>(
      subgraph_->GetName() + "_" + DATA + "_" + std::to_string(parent_node_index), DATA);
  GE_ASSERT_NOTNULL(data_op, "[New][Memory] for data op failed.");
  const auto &input_desc = data_anchor->GetOwnerNode()->GetOpDesc()->GetInputDesc(data_anchor->GetIdx());
  GE_ASSERT_GRAPH_SUCCESS(data_op->AddInputDesc(input_desc), "[Add][InputDesc] to op:%s failed, graph %s.",
                          data_op->GetName().c_str(), subgraph_->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(data_op->AddOutputDesc(input_desc), "[Add][OutputDesc] to op:%s failed, graph %s.",
                          data_op->GetName().c_str(), subgraph_->GetName().c_str());
  GE_ASSERT_TRUE(AttrUtils::SetInt(data_op, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index),
                 "[Set][Attr] %s on subgraph data node:%s failed.", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                 data_op->GetName().c_str());
  // add new data node to subgraph
  data_node = subgraph_->AddNode(data_op);
  GE_ASSERT_NOTNULL(data_node, "[Add][Node] %s to subgraph:%s failed.", data_node->GetName().c_str(),
                    subgraph_->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(data_node->SetOwnerComputeGraph(subgraph_), "[Set][OwnerGraph] %s of data node:%s failed.",
                          subgraph_->GetName().c_str(), data_node->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::AddSubgraphInputNodes() {
  int64_t parent_node_index = 0;
  for (const auto &anchor : inputs_) {
    GE_CHECK_NOTNULL(subgraph_);
    NodePtr data_node;
    GE_ASSERT_SUCCESS(MakeDataNode(parent_node_index, anchor, data_node),
                      "[Call][MakeDataNode] failed, parent_node_index[%ld], node[%s], index[%d].",
                      parent_node_index, anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx());
    // link data node to inputs
    for (const auto &node_anchor : frame_input_index_to_inputs_[parent_node_index]) {
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_anchor),
                              "[Call][AddEdge] Failed add data input edge to %s:%d",
                              node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx());
    }
    parent_node_index++;
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::MakeNetOutputNode(NodePtr &net_output_node) const {
  // construct new net output node
  const auto net_output_op = MakeShared<OpDesc>(subgraph_->GetName() + "_" + NODE_NAME_NET_OUTPUT, ge::NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output_op, "[New][Memory] for netoutput op failed.");
  for (size_t i = 0; i < outputs_.size(); ++i) {
    GeTensorDesc input_desc;
    GE_ASSERT_GRAPH_SUCCESS(net_output_op->AddInputDesc(input_desc), "[Add][InputDesc] to op:%s failed.",
                            net_output_op->GetName().c_str());
  }
  // add new net output node to subgraph
  net_output_node = subgraph_->AddNode(net_output_op);
  GE_ASSERT_NOTNULL(net_output_node, "[Call][AddNode] Failed add netoutput node:%s to subgraph:%s.",
                    net_output_op->GetName().c_str(), subgraph_->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(net_output_node->SetOwnerComputeGraph(subgraph_),
                          "[Set][OwnerGraph] %s of netoutput node:%s failed.", subgraph_->GetName().c_str(),
                          net_output_node->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::UpdateNetOutputInputTensors(const NodePtr &net_output_node) const {
  // update net output input tensor descs and anchors
  int64_t parent_node_index = 0;
  for (const auto &anchor : outputs_) {
    auto output_desc = anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(anchor->GetIdx()));
    GE_ASSERT_TRUE(AttrUtils::SetInt(output_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index),
                   "[Set][Attr] parent_node_index on subgraph node:%s netoutput's input failed.",
                   anchor->GetOwnerNode()->GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(net_output_node->GetOpDesc()->UpdateInputDesc(parent_node_index, output_desc),
                            "[Update][InputDesc] of netoutput node:%s failed.", net_output_node->GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(anchor, net_output_node->GetInDataAnchor(parent_node_index)),
                            "[Add][Edge] from %s:%d to netoutput node:%s failed.",
                            anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx(),
                            net_output_node->GetName().c_str());
    parent_node_index++;
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::AddSubgraphOutputNode() const {
  if (outputs_.empty() && control_outputs_.empty()) {
    return GRAPH_SUCCESS;
  }
  NodePtr net_output_node;
  GE_ASSERT_GRAPH_SUCCESS(MakeNetOutputNode(net_output_node), "[Call][MakeNetOutputNode] failed.");
  // update net output input tensor descs and anchors
  GE_ASSERT_GRAPH_SUCCESS(UpdateNetOutputInputTensors(net_output_node),
                          "[Update][InputTensors] of node[%s] failed.",
                          net_output_node->GetName().c_str());
  // to check, need link control output to netoutput
  for (const auto &anchor : control_outputs_) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(anchor, net_output_node->GetInControlAnchor()),
                            "[Add][ControlEdge] from %s:%d to netoutput node:%s failed.",
                            anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx(),
                            net_output_node->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::AddSubgraphInputOutputNodes() {
  GE_ASSERT_GRAPH_SUCCESS(AddSubgraphInputNodes(), "[Call][AddSubgraphInputNodes] failed.");
  GE_ASSERT_GRAPH_SUCCESS(AddSubgraphOutputNode(), "[Call][AddSubgraphOutputNode] failed.");
  return GRAPH_SUCCESS;
}

void DefaultSubgraphBuilder::ResetSubgraphFrame() {
  subgraph_ = nullptr;
  partitioned_call_node_ = nullptr;
  inputs_.clear();
  outputs_.clear();
  control_inputs_.clear();
  control_outputs_.clear();
  src_key_to_frame_input_index_.clear();
  frame_input_index_to_inputs_.clear();
}

graphStatus DefaultSubgraphBuilder::CheckInputValid() const {
  GE_ASSERT_NOTNULL(GetRootGraph());
  GE_ASSERT_NOTNULL(GetCluster());
  return GRAPH_SUCCESS;
}

graphStatus DefaultSubgraphBuilder::BuildSubgraph() {
  if (!IsNeedBuildSubgraph(GetCluster())) {
    return GRAPH_SUCCESS;
  }
  ResetSubgraphFrame();
  GE_ASSERT_GRAPH_SUCCESS(CheckInputValid(), "[Check][InputValid] failed.");
  GE_ASSERT_GRAPH_SUCCESS(BuildSubgraphFrame(), "[Call][BuildSubgraphFrame] failed, root_graph[%s].",
                          GetRootGraph()->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(CombinePartitionedCall(), "[Call][BuildSubgraphFrame] failed, root_graph[%s].",
                          GetRootGraph()->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(AddSubgraphInputOutputNodes(), "[Call][BuildSubgraphFrame] failed, root_graph[%s].",
                          GetRootGraph()->GetName().c_str());
  unique_id_++;
  GELOGD("Build subgraph[%s], node[%s], unique_id[%zu], successfully.", subgraph_->GetName().c_str(),
         partitioned_call_node_->GetName().c_str(), unique_id_);
  return GRAPH_SUCCESS;
}
} // namespace ge
