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

#include "graph/compute_graph.h"
#include <deque>
#include "./format_refiner.h"
#include "./ge_context.h"
#include "debug/ge_attr_define.h"
#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "ge/ge_api_types.h"
#include "graph/shape_refiner.h"
#include "proto/ge_ir.pb.h"
#include "utils/ge_ir_utils.h"
#include "utils/graph_utils.h"
#include "utils/node_utils.h"
#include "utils/op_desc_utils.h"
#include "utils/string_utils.h"
#include "utils/tensor_utils.h"

namespace ge {
namespace {
const size_t OUTPUT_PARAM_SIZE = 2;
const std::string alias_name_attr = "_aliasName";
bool IsUseBFS() {
  string run_mode;
  const int base = 10;
  if (ge::GetContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == GRAPH_SUCCESS && !run_mode.empty()) {
    if (GraphRunMode(std::strtol(run_mode.c_str(), nullptr, base)) >= TRAIN) {
      return true;
    }
  } else {
    GELOGW("OPTION_GRAPH_RUN_MODE not set, use BFSTopologicalSorting by default.");
  }
  return false;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::ComputeGraph(const std::string &name)
    : name_(name), nodes_(), input_nodes_(), sub_graph_(), is_valid_flag_(false), need_iteration_(false) {
  attrs_.InitDefault();
}

ComputeGraph::~ComputeGraph() {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY string ComputeGraph::GetName() const { return name_; }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetName(const string &name) { name_ = name; }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t ComputeGraph::GetAllNodesSize() const {
  return GetAllNodes().size();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetAllNodes() const {
  std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
  return AllGraphNodes(subgraphs);
}

ComputeGraph::Vistor<NodePtr> ComputeGraph::AllGraphNodes(std::vector<std::shared_ptr<ComputeGraph>> &subgraphs) const {
  std::vector<NodePtr> all_nodes;
  std::deque<NodePtr> candidates;

  candidates.insert(candidates.begin(), nodes_.begin(), nodes_.end());
  while (!candidates.empty()) {
    NodePtr node = candidates.front();
    all_nodes.emplace_back(node);
    candidates.pop_front();

    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }

    const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
    for (auto name_iter = subgraph_names.rbegin(); name_iter != subgraph_names.rend(); ++name_iter) {
      auto subgraph = GetSubgraph(*name_iter);
      if (subgraph != nullptr) {
        subgraphs.emplace_back(subgraph);
        candidates.insert(candidates.begin(), subgraph->nodes_.begin(), subgraph->nodes_.end());
      }
    }
  }

  return Vistor<NodePtr>(shared_from_this(), all_nodes);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetNodes(
  bool is_unknown_shape) const {
  if (is_unknown_shape) {
    return GetDirectNode();
  } else {
    return GetAllNodes();
  }
}

size_t ComputeGraph::GetDirectNodesSize() const { return nodes_.size(); }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetDirectNode() const {
  return Vistor<NodePtr>(shared_from_this(), nodes_);
}

ComputeGraph::Vistor<NodePtr> ComputeGraph::GetInputNodes() const {
  return Vistor<NodePtr>(shared_from_this(), input_nodes_);
}

ComputeGraph::Vistor<NodePtr> ComputeGraph::GetOutputNodes() const {
  std::vector<NodePtr> result;
  for (auto iter = output_nodes_info_.begin(); iter != output_nodes_info_.end(); ++iter) {
    result.push_back(iter->first);
  }
  return Vistor<NodePtr>(shared_from_this(), result);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::FindNode(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetName() == name) {
      return node;
    }
    std::vector<string> out_alias_name;
    if (AttrUtils::GetListStr(node->GetOpDesc(), alias_name_attr, out_alias_name)) {
      for (const auto &alias_name : out_alias_name) {
        if (alias_name == name) {
          return node;
        }
      }
    }
  }
  return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr
ComputeGraph::FindFirstNodeMatchType(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() == name) {
      return node;
    }
  }
  return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GraphAttrsAreEqual(
  const ComputeGraph &r_graph) const {
  // ProtoMsgOwner <::google::protobuf::Message> is temporarily ignored
  if ((this->attrs_.protoMsg_ != nullptr) && (r_graph.attrs_.protoMsg_ != nullptr)) {
    const auto &proto_attr_map = *(this->attrs_.protoMsg_);
    const auto &r_proto_attr_map = *(r_graph.attrs_.protoMsg_);
    // 1.Verify graph's ProtoAttrMap size
    if (proto_attr_map.size() != r_proto_attr_map.size()) {
      GELOGE(GRAPH_FAILED, "Size of compute graph's ProtoAttrMap verify failed, graph name: %s.",
             this->GetName().c_str());
      return false;
    }
    // 2.Verify graph's ProtoAttrMap key, verify values is temporarily not implemented
    for (const auto &it : proto_attr_map) {
      if (r_proto_attr_map.count(it.first) == 0) {
        GELOGE(GRAPH_FAILED, "Key of compute graph's ProtoAttrMap verify failed, graph name: %s key name: %s.",
               this->GetName().c_str(), it.first.c_str());
        return false;
      }
    }
    return true;
  }
  return ((this->attrs_.protoMsg_ == nullptr) && (r_graph.attrs_.protoMsg_ == nullptr));
}

/// Since there may be different input nodes
/// chosen by user in the same graph, special judgment is needed
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::VectorInputNodePtrIsEqual(
  const std::vector<NodePtr> &left_nodes, const std::vector<NodePtr> &right_nodes) const {
  const auto left_nodes_size = left_nodes.size();
  const auto right_nodes_size = right_nodes.size();
  if (left_nodes_size != right_nodes_size) {
    GELOGE(GRAPH_FAILED,
           "Check failed with graph input_nodes_: "
           "left inputNodes size  %zu is different with right inputNodes size %zu .",
           left_nodes_size, right_nodes_size);
    return false;
  }
  for (size_t j = 0; j < left_nodes_size; j++) {
    if (left_nodes.at(j) == nullptr || right_nodes.at(j) == nullptr) {
      GELOGE(GRAPH_FAILED, "left_nodes.at(%zu) or right_nodes.at(%zu) is nullptr", j, j);
      return false;
    }
    const auto &left_input_name = left_nodes.at(j)->GetName();
    const auto &right_input_name = right_nodes.at(j)->GetName();
    if (left_input_name != right_input_name) {
      GELOGE(GRAPH_FAILED,
             "Check failed with graph input_nodes_: "
             "left inputNode name %s is different with right inputNode name %s at inputNodes index %zu.",
             left_input_name.c_str(), right_input_name.c_str(), j);
      return false;
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GraphMembersAreEqual(
  const ComputeGraph &r_graph) const {
  return (IsEqual(this->sub_graph_.size(), r_graph.sub_graph_.size(), "graph.subgraphs_.size()") &&
          IsEqual(this->nodes_.size(), r_graph.nodes_.size(), "graph.nodes_.size()") &&
          VectorInputNodePtrIsEqual(this->input_nodes_, r_graph.input_nodes_) &&
          IsEqual(this->name_, r_graph.name_, "graph.name_") &&
          IsEqual(this->is_valid_flag_, r_graph.is_valid_flag_, "graph.is_valid_flag_") &&
          IsEqual(this->need_iteration_, r_graph.need_iteration_, "graph.need_iteration_") &&
          IsEqual(this->params_share_map_, r_graph.params_share_map_, "graph.params_share_map_") &&
          IsEqual(this->out_nodes_map_, r_graph.out_nodes_map_, "graph.out_nodes_map_") &&
          IsEqual(this->inputs_order_, r_graph.inputs_order_, "graph.inputs_order_") &&
          IsEqual(this->output_size_, r_graph.output_size_, "graph.output_size_") &&
          IsEqual(this->input_size_, r_graph.input_size_, "graph.input_size_") &&
          IsEqual(this->output_nodes_info_, r_graph.output_nodes_info_, "graph.output_nodes_info_"));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::operator==(const ComputeGraph &r_graph) const {
  // Firstly: Graph's members equal
  if ((!GraphMembersAreEqual(r_graph)) || (!GraphAttrsAreEqual(r_graph))) {
    return false;
  }

  // Secondly: Node equal means the link relationship between node and node itself equal
  for (const auto &left_node : nodes_) {
    if (left_node == nullptr) {
      GELOGE(GRAPH_FAILED, "left_node is nullptr");
      return false;
    }
    const auto &node_name = left_node->GetName();
    // After TopologicalSorting, node order can change, so find node by name
    const auto &right_node = r_graph.FindNode(node_name);
    GE_IF_BOOL_EXEC(right_node == nullptr, GELOGE(GRAPH_FAILED, "right_node is NULL!!!"); return false);
    if (!(*right_node == *left_node)) {
      GELOGE(GRAPH_FAILED, "Compare graph failed, node name: %s.", node_name.c_str());
      return false;
    }
  }

  // Thirdly: Recursively determine whether the sub graphs are equal
  for (size_t i = 0; i < this->sub_graph_.size(); i++) {
    if (!(*((this->sub_graph_)[i]) == *((r_graph.sub_graph_)[i]))) {
      return false;
    }
  }
  return true;
}

NodePtr ComputeGraph::AddNodeFront(NodePtr node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr or op desc should not be null.");
    return nullptr;
  }
  node->SetHostNode(is_valid_flag_);
  node->GetOpDesc()->SetId(nodes_.size());
  if (nodes_.size() > 0 && nodes_[0]->GetType() == DATA) {
    (void)nodes_.insert(nodes_.begin() + 1, node);
  } else {
    (void)nodes_.insert(nodes_.begin(), node);
  }
  return node;
}

NodePtr ComputeGraph::AddNodeFront(const OpDescPtr &op) {
  if (op == nullptr) {
    GELOGE(GRAPH_FAILED, "The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(nodes_.size());
  NodePtr node_ptr = shared_ptr<Node>(new (std::nothrow) Node(op, shared_from_this()));
  GE_IF_BOOL_EXEC(node_ptr == nullptr, GELOGE(GRAPH_FAILED, "node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node_ptr->Init() != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "node init fail."); return nullptr);
  return AddNodeFront(node_ptr);
}

NodePtr ComputeGraph::AddNodeAfter(NodePtr node, const NodePtr &pre_node) {
  if (node == nullptr || node->GetOpDesc() == nullptr || pre_node == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr or op desc should not be null.");
    return nullptr;
  }
  node->SetHostNode(is_valid_flag_);
  node->GetOpDesc()->SetId(nodes_.size());
  auto node_iter = std::find(nodes_.begin(), nodes_.end(), pre_node);
  if (node_iter != nodes_.end()) {
    nodes_.insert(node_iter + 1, node);
  } else {
    GELOGE(GRAPH_FAILED, "Cannot find pre_node in nodes_.");
    return nullptr;
  }

  return node;
}

NodePtr ComputeGraph::AddNodeAfter(OpDescPtr &op, const NodePtr &pre_node) {
  if (op == nullptr) {
    GELOGE(GRAPH_FAILED, "The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(nodes_.size());
  NodePtr node_ptr = shared_ptr<Node>(new (std::nothrow) Node(op, shared_from_this()));
  GE_IF_BOOL_EXEC(node_ptr == nullptr, GELOGE(GRAPH_FAILED, "node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node_ptr->Init() != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "node init failed."); return nullptr);
  return AddNodeAfter(node_ptr, pre_node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::AddNode(NodePtr node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr should not be null.");
    return nullptr;
  }
  node->SetHostNode(is_valid_flag_);
  node->GetOpDesc()->SetId((int64_t)GetDirectNodesSize());
  nodes_.push_back(node);
  return node;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::AddNode(OpDescPtr op) {
  if (op == nullptr) {
    GELOGE(GRAPH_FAILED, "The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(GetDirectNodesSize());
  NodePtr node_ptr = shared_ptr<Node>(new (std::nothrow) Node(op, shared_from_this()));
  GE_IF_BOOL_EXEC(node_ptr == nullptr, GELOGE(GRAPH_FAILED, "node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node_ptr->Init() != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "node init fail."); return nullptr);
  return AddNode(node_ptr);
}

NodePtr ComputeGraph::AddNode(OpDescPtr op, int64_t id) {  // for unserialize.
  if (op == nullptr) {
    GELOGE(GRAPH_FAILED, "The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(id);
  NodePtr node = shared_ptr<Node>(new (std::nothrow) Node(op, shared_from_this()));
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(GRAPH_FAILED, "node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node->Init() != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "node init fail."); return nullptr);
  node->SetHostNode(is_valid_flag_);
  nodes_.push_back(node);
  return node;
}

NodePtr ComputeGraph::AddInputNode(NodePtr node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr should not be null.");
    return nullptr;
  }
  input_nodes_.push_back(node);
  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    GE_CHK_BOOL_EXEC(AddNode(node) != nullptr, return nullptr, "add node failed");
  }
  return node;
}

NodePtr ComputeGraph::AddOutputNode(NodePtr node) { return AddOutputNodeByIndex(node, 0); }

NodePtr ComputeGraph::AddOutputNodeByIndex(NodePtr node, int32_t index) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr or opdesc should not be null.");
    return nullptr;
  }

  bool already_have = false;
  NodePtr result = node;
  // [output_nodes_info_ : should not be null]
  for (const auto &item : output_nodes_info_) {
    if (item.first->GetName() == node->GetName() && item.second == index) {
      already_have = true;
      result = item.first;
      break;
    }
  }

  if (!already_have) {
    output_nodes_info_.emplace_back(std::make_pair(node, index));
    GELOGI("Push back node name:%s, index:%ld, into output_nodes_info_.", node->GetName().c_str(), index);
  }

  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    GE_CHK_BOOL_EXEC(AddNode(node) != nullptr, return nullptr, "add node failed");
  }
  return result;
}

graphStatus ComputeGraph::RemoveConstInput(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr || out_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    if (out_anchor->GetOwnerNode()->GetType() == CONSTANT || out_anchor->GetOwnerNode()->GetType() == CONSTANTOP) {
      GE_CHK_BOOL_RET_STATUS(GraphUtils::RemoveEdge(out_anchor, in_anchor) == GRAPH_SUCCESS, GRAPH_FAILED,
                             "Remove edge from const op failed.");
      if (out_anchor->GetOwnerNode()->GetOutNodes().size() == 0) {
        GELOGI("Remove const op %s.", out_anchor->GetOwnerNode()->GetName().c_str());
        auto iter = find(nodes_.begin(), nodes_.end(), out_anchor->GetOwnerNode());
        if (iter != nodes_.end()) {
          (void)nodes_.erase(iter);
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::RemoveNode(const NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  // delete const op for this node
  (void)RemoveConstInput(node);

  // if the node save as input node, delete it
  (void)RemoveInputNode(node);

  // if the node save as input node, delete it
  (void)RemoveOutputNode(node);

  if (GRAPH_SUCCESS != IsolateNode(node)) {
    GELOGE(GRAPH_FAILED, "Isolate node failed, node name: %s.", node->GetName().c_str());
    return GRAPH_FAILED;
  }

  auto iter = find(nodes_.begin(), nodes_.end(), node);
  if (iter != nodes_.end()) {
    (void)nodes_.erase(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Used in sub_graph scenes
graphStatus ComputeGraph::RemoveInputNode(const NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  auto iter = find(input_nodes_.begin(), input_nodes_.end(), node);
  if (iter != input_nodes_.end()) {
    (void)input_nodes_.erase(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Used in sub_graph scenes
graphStatus ComputeGraph::RemoveOutputNode(const NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  auto iter = output_nodes_info_.begin();
  bool find_node = false;
  // [output_nodes_info_ : should not be null]
  while (iter != output_nodes_info_.end()) {
    if (node->GetName() == iter->first->GetName()) {
      iter = output_nodes_info_.erase(iter);
      find_node = true;
    } else {
      ++iter;
    }
  }
  GE_IF_BOOL_EXEC(find_node == false, return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

std::shared_ptr<ComputeGraph> ComputeGraph::AddSubGraph(std::shared_ptr<ComputeGraph> sub_graph) {
  if (sub_graph == nullptr) {
    GELOGE(GRAPH_FAILED, "The graph ptr should not be null.");
    return nullptr;
  }
  sub_graph_.push_back(sub_graph);
  names_to_subgraph_[sub_graph->GetName()] = sub_graph;
  return sub_graph;
}

graphStatus ComputeGraph::RemoveSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph) {
  if (sub_graph == nullptr) {
    GELOGE(GRAPH_FAILED, "The graph ptr should not be null.");
    return GRAPH_FAILED;
  }

  names_to_subgraph_.erase(sub_graph->GetName());
  auto iter = find(sub_graph_.begin(), sub_graph_.end(), sub_graph);
  if (iter != sub_graph_.end()) {
    (void)sub_graph_.erase(iter);
    return GRAPH_SUCCESS;
  } else {
    GELOGW("find sub_graph failed");
    return GRAPH_SUCCESS;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::AddSubgraph(const std::string &name, const std::shared_ptr<ComputeGraph> &subgraph) {
  if (subgraph == nullptr) {
    GE_LOGE("Try to add a null subgraph, name %s", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  auto parent_graph = subgraph->GetParentGraph();
  if (parent_graph == nullptr) {
    GE_LOGE("Try to add subgraph without parent graph, name %s", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  auto parent_node = subgraph->GetParentNode();
  if (parent_node == nullptr) {
    GE_LOGE("Try to add a subgraph without parent node, name %s", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  if (parent_node->GetOwnerComputeGraph() != parent_graph) {
    GE_LOGE(
      "Try to add a subgraph which parent node's parent graph is not equal to "
      "the subgraph's parent graph, subgraph name %s, parent node name %s",
      subgraph->GetName().c_str(), parent_graph->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  if (!this->parent_graph_.expired()) {
    GELOGW("The subgraphs should only be added to the root graph");
  }
  if (name != subgraph->GetName()) {
    GELOGW("The subgraph name %s is different with input %s", subgraph->GetName().c_str(), name.c_str());
  }
  if (names_to_subgraph_.find(name) != names_to_subgraph_.end()) {
    GE_LOGE("The subgraph %s existed", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  sub_graph_.push_back(subgraph);
  names_to_subgraph_[name] = subgraph;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::AddSubgraph(const std::shared_ptr<ComputeGraph> &subgraph) {
  if (subgraph == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  return AddSubgraph(subgraph->GetName(), subgraph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::RemoveSubgraph(const std::string &name) {
  auto iter = names_to_subgraph_.find(name);
  if (iter == names_to_subgraph_.end()) {
    return;
  }
  for (auto vec_iter = sub_graph_.begin(); vec_iter != sub_graph_.end(); ++vec_iter) {
    if (*vec_iter == iter->second) {
      sub_graph_.erase(vec_iter);
      break;
    }
  }
  names_to_subgraph_.erase(iter);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::RemoveSubgraph(
  const std::shared_ptr<ComputeGraph> &subgraph) {
  if (subgraph != nullptr) {
    RemoveSubgraph(subgraph->GetName());
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::shared_ptr<ComputeGraph> ComputeGraph::GetSubgraph(
  const std::string &name) const {
  std::shared_ptr<ComputeGraph> parent = parent_graph_.lock();
  if (parent == nullptr) {
    auto iter = names_to_subgraph_.find(name);
    return iter == names_to_subgraph_.end() ? nullptr : iter->second;
  } else {
    return parent->GetSubgraph(name);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<std::shared_ptr<ComputeGraph>>
ComputeGraph::GetAllSubgraphs() const {
  return sub_graph_;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY shared_ptr<ComputeGraph> ComputeGraph::GetParentGraph() {
  return parent_graph_.lock();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetParentGraph(
  const shared_ptr<ComputeGraph> &parent) {
  parent_graph_ = parent;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY shared_ptr<Node> ComputeGraph::GetParentNode() {
  return parent_node_.lock();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetParentNode(const shared_ptr<Node> &parent) {
  parent_node_ = parent;
}

///
/// @brief Update input-mapping
/// @param [in] input_mapping : index_of_cur_graph_node_input -> index_of_new_graph_node_input
/// @return graphStatus
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::UpdateInputMapping(const std::map<uint32_t, uint32_t> &input_mapping) {
  for (auto &input : nodes_) {
    if (input->GetType() == DATA) {
      uint32_t cur_index = 0;
      if (!ge::AttrUtils::GetInt(input->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, cur_index)) {
        continue;
      }
      auto iter = input_mapping.find(cur_index);
      if (iter == input_mapping.end()) {
        continue;
      }
      if (!ge::AttrUtils::SetInt(input->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, iter->second)) {
        GE_LOGE("UpdateInputMapping failed: set attr ATTR_NAME_PARENT_NODE_INDEX failed.");
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

///
/// @brief Update output-mapping
/// @param [in] output_mapping : index_of_cur_graph_node_output -> index_of_new_graph_node_output
/// @return graphStatus
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::UpdateOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping) {
  NodePtr net_output = FindFirstNodeMatchType(NETOUTPUT);
  if (net_output == nullptr) {
    GE_LOGE("UpdateOutputMapping failed: node type %s not exist in graph.", NETOUTPUT);
    return GRAPH_FAILED;
  }
  OpDescPtr op_desc = net_output->GetOpDesc();
  if (op_desc == nullptr) {
    GE_LOGE("UpdateOutputMapping failed: op_desc is NULL.");
    return GRAPH_FAILED;
  }

  size_t num = op_desc->GetAllInputsSize();
  for (size_t i = 0; i < num; i++) {
    GeTensorDesc tensor = op_desc->GetInputDesc(i);
    uint32_t cur_index = 0;
    if (!ge::AttrUtils::GetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, cur_index)) {
      continue;
    }
    auto iter = output_mapping.find(cur_index);
    if (iter == output_mapping.end()) {
      continue;
    }
    if (!ge::AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, iter->second)) {
      GE_LOGE("UpdateOutputMapping failed: set attr ATTR_NAME_PARENT_NODE_INDEX failed.");
      return GRAPH_FAILED;
    }
    if (op_desc->UpdateInputDesc(i, tensor) != GRAPH_SUCCESS) {
      GE_LOGE("UpdateOutputMapping failed: update %u input_tensor failed.", i);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::InsertEventNodes() {
  std::vector<NodePtr> node_vec = nodes_;
  for (const auto &node : GetDirectNode()) {
    if (node == nullptr || node->GetOpDesc() == nullptr) {
      GELOGW("node or OpDescPtr is nullptr.");
      continue;
    }
    GE_IF_BOOL_EXEC(node == nullptr, GELOGE(GRAPH_FAILED, "The node should not be null."); return GRAPH_FAILED);
    if (node->GetOpDesc()->GetType() == RECV) {
      auto iter = find(node_vec.begin(), node_vec.end(), node);
      if (iter == node_vec.end()) {
        GELOGW("no node found.");
      } else {
        (void)node_vec.erase(iter);
      }

      auto dst_iter = find(node_vec.begin(), node_vec.end(), node->GetOutControlNodes().at(0));
      (void)node_vec.insert(dst_iter, node);
    }
    if (node->GetOpDesc()->GetType() == SEND) {
      auto iter = find(node_vec.begin(), node_vec.end(), node);
      if (iter == node_vec.end()) {
        GELOGW("no node found.");
      } else {
        (void)node_vec.erase(iter);
      }

      auto src_iter = find(node_vec.begin(), node_vec.end(), node->GetInControlNodes().at(0));
      (void)node_vec.insert(src_iter + 1, node);
    }
  }
  nodes_.clear();
  for (size_t i = 0; i < node_vec.size(); ++i) {
    NodePtr node = node_vec[i];
    if (node == nullptr || node->GetOpDesc() == nullptr) {
      GELOGW("node or OpDescPtr is nullptr.");
    } else {
      node->GetOpDesc()->SetId((int64_t)i);
      nodes_.push_back(node);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraph::DFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                                std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                std::vector<NodePtr> &stack) {
  GELOGI("Runing_Dfs_Sort: %s", name_.c_str());
  // Record the number of non data nodes but no input nodes
  GE_CHK_BOOL_EXEC(SortNodes(stack, map_in_edge_num) == GRAPH_SUCCESS, return GRAPH_FAILED, "sort nodes failed");

  // Only data nodes here
  while (!stack.empty()) {
    NodePtr node = stack.back();
    stack.pop_back();
    node_vec.push_back(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GELOGD("node_vec.push_back %s", node->GetOpDesc()->GetName().c_str());
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(anchor);
      for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
        if (iter != map_in_edge_num.end() && --iter->second == 0) {
          stack.push_back(peer_in_anchor->GetOwnerNode());
        }
      }
      for (const auto &peer_in_anchor : anchor->GetPeerInControlAnchors()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
        if (iter != map_in_edge_num.end() && --iter->second == 0) {
          stack.push_back(peer_in_anchor->GetOwnerNode());
        }
      }
    }
    GE_IF_BOOL_EXEC(
      node->GetOutControlAnchor() != nullptr, for (AnchorPtr peer_in_anchor
                                                   : node->GetOutControlAnchor()->GetPeerAnchors()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
        if (iter != map_in_edge_num.end() && --iter->second == 0) {
          stack.push_back(peer_in_anchor->GetOwnerNode());
        }
      })
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraph::BFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                                std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                std::deque<NodePtr> &stack) {
  GELOGI("Runing_Bfs_Sort: %s", name_.c_str());
  std::vector<NodePtr> stack_input;
  std::map<string, NodePtr> breadth_node_map;
  // Record the number of non data nodes but no input nodes
  GE_CHK_BOOL_EXEC(SortNodes(stack_input, map_in_edge_num) == GRAPH_SUCCESS, return GRAPH_FAILED, "sort nodes failed");

  // Only data nodes here
  while (!stack_input.empty() || !stack.empty()) {
    NodePtr node = nullptr;
    if (!stack.empty()) {
      node = stack.back();
      stack.pop_back();
    } else {
      node = stack_input.back();
      stack_input.pop_back();
    }

    node_vec.push_back(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GELOGD("node_vec.push_back %s", node->GetOpDesc()->GetName().c_str());
    CollectBreadthOutNode(node, map_in_edge_num, breadth_node_map);

    for (const auto &name_node : breadth_node_map) {
      (void)stack.push_front(name_node.second);
    }
    breadth_node_map.clear();
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraph::CollectBreadthOutNode(const NodePtr &node, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                std::map<string, NodePtr> &breadth_node_map) {
  for (const auto &anchor : node->GetAllOutDataAnchors()) {
    for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
      auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
      if (iter != map_in_edge_num.end() && 0 == --iter->second) {
        (void)breadth_node_map.emplace(peer_in_anchor->GetOwnerNode()->GetName(), peer_in_anchor->GetOwnerNode());
      }
    }

    for (const auto &peer_in_anchor : anchor->GetPeerInControlAnchors()) {
      auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
      if (iter != map_in_edge_num.end() && 0 == --iter->second) {
        (void)breadth_node_map.emplace(peer_in_anchor->GetOwnerNode()->GetName(), peer_in_anchor->GetOwnerNode());
      }
    }
  }
  if (node->GetOutControlAnchor() != nullptr) {
    for (AnchorPtr peer_in_anchor : node->GetOutControlAnchor()->GetPeerAnchors()) {
      auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
      if (iter != map_in_edge_num.end() && 0 == --iter->second) {
        (void)breadth_node_map.emplace(peer_in_anchor->GetOwnerNode()->GetName(), peer_in_anchor->GetOwnerNode());
      }
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::TopologicalSorting() {
  auto ret = TopologicalSortingGraph();
  if (ret != SUCCESS) {
    GraphUtils::DumpGEGraphToOnnx(*this, "black_box");
    GELOGE(ret, "Graph [%s] topological sort failed, saved to file black_box", name_.c_str());
    return ret;
  }

  if (sub_graph_.empty()) {
    return SUCCESS;
  }

  // partition sub graph
  for (const auto &sub_graph : sub_graph_) {
    ret = sub_graph->TopologicalSortingGraph();
    if (ret != SUCCESS) {
      GELOGE(ret, "Sub graph topological sort Failed");
      return ret;
    }
  }

  std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
  auto nodes = AllGraphNodes(subgraphs);
  for (size_t i = 0; i < nodes.size(); i++) {
    NodePtr node = nodes.at(i);   // [node: should not be null]
    node->GetOpDesc()->SetId(i);  // [node->GetOpDesc(): should not be null]
  }
  if (sub_graph_.size() != subgraphs.size()) {  // Graph Partition use subgraph, Keep original
    GELOGW("Keep original subgraph for graph size %zu not equal %zu.", sub_graph_.size(), subgraphs.size());
    return SUCCESS;
  }
  sub_graph_.swap(subgraphs);
  return SUCCESS;
}

graphStatus ComputeGraph::TopologicalSortingGraph() {
  std::vector<NodePtr> node_vec;
  std::map<NodePtr, uint32_t> map_in_edge_num;
  bool use_BFS = IsUseBFS();
  if (use_BFS) {
    std::deque<NodePtr> stack;
    if (BFSTopologicalSorting(node_vec, map_in_edge_num, stack) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  } else {
    std::vector<NodePtr> stack;
    if (DFSTopologicalSorting(node_vec, map_in_edge_num, stack) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }

  // If they are not equal, there is a closed loop
  if (node_vec.size() != nodes_.size()) {
    std::set<Node *> itered_nodes_set;
    for (auto &node : node_vec) {
      itered_nodes_set.insert(node.get());
    }
    GE_LOGE("Failed to do topo sorting total %zu, itered %zu, exist closed loop in graph.", nodes_.size(),
            node_vec.size());
    for (auto &node : nodes_) {
      if (itered_nodes_set.count(node.get()) == 0) {
        GE_LOGE("The node %s does not itered when topological sorting", node->GetName().c_str());
      }
    }
    return GRAPH_FAILED;
  }

  nodes_.clear();
  for (size_t i = 0; i < node_vec.size(); i++) {
    NodePtr node = node_vec[i];   // [node: should not be null]
    node->GetOpDesc()->SetId(i);  // [node->GetOpDesc(): should not be null]
    nodes_.push_back(node);
  }

  is_valid_flag_ = true;
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraph::SortNodes(std::vector<NodePtr> &stack, std::map<NodePtr, uint32_t> &map_in_edge_num) {
  // Record the number of non data nodes but no input nodes
  uint32_t spec_node_size = 0;
  bool verify_isolated = false;
  string run_mode;
  const int base = 10;
  // Need verify isolated point in PREDICTION mode.
  if (ge::GetContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == GRAPH_SUCCESS && !run_mode.empty()) {
    if (GraphRunMode(std::strtol(run_mode.c_str(), nullptr, base)) < TRAIN) {
      verify_isolated = true;
    }
  }
  for (const auto &node : GetDirectNode()) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    map_in_edge_num[node] = static_cast<uint32_t>(GetInEdgeSize(node));
    if (map_in_edge_num[node] == 0) {
      if ((node->GetOpDesc()->GetType() != DATA) && (node->GetOpDesc()->GetType() != AIPPDATA) &&
          (node->GetOpDesc()->GetType() != INPUT_TYPE) && (node->GetOpDesc()->GetType() != ANN_DATA)) {
        // At present, can only judge the isolated point without input and output.
        // It is impossible to judge the situation with multiple output nodes.
        if (verify_isolated && GetOutEdgeSize(node) == 0) {
          GELOGE(GRAPH_FAILED, "May has isolated nodes in graph, node name: %s.", node->GetName().c_str());
          return GRAPH_FAILED;
        }
        (void)stack.insert(stack.begin(), node);
        spec_node_size++;
        continue;
      }
      // Need to insert the data nodes in reverse order
      (void)stack.insert(stack.begin() + spec_node_size, node);
    }
  }

  /// Make sure the inputs order matches with user-designated
  /// 1. Get the index of two input nodes in the user-inputs-order(inputs_order_)
  /// 2. Compare two indices, if not match, swap the positions of two inputs
  /// *: Remind: stack is reverse-order
  for (size_t i = 0; i < stack.size(); ++i) {
    // If not found in 'inputs_order_', skip it
    auto it_i = std::find(inputs_order_.begin(), inputs_order_.end(), stack[i]->GetName());
    GE_IF_BOOL_EXEC(it_i == inputs_order_.end(), continue);
    auto inx_i = it_i - inputs_order_.begin();
    for (size_t j = i + 1; j < stack.size(); ++j) {
      // If not found in 'inputs_order_', skip it
      auto it_j = std::find(inputs_order_.begin(), inputs_order_.end(), stack[j]->GetName());
      GE_IF_BOOL_EXEC(it_j == inputs_order_.end(), continue);

      // Compare index, swap them if it should be
      auto inx_j = it_j - inputs_order_.begin();
      GE_IF_BOOL_EXEC(inx_i < inx_j, std::swap(stack[i], stack[j]));
    }
  }

  return GRAPH_SUCCESS;
}

size_t ComputeGraph::GetInEdgeSize(const NodePtr &node) {
  size_t in_edge_size = 0;
  if (node == nullptr) {
    return in_edge_size;
  }
  for (const auto &anchor : node->GetAllInDataAnchors()) {
    in_edge_size = in_edge_size + anchor->GetPeerAnchorsSize();
    // Break flow control data loop.
    OutDataAnchorPtr out_anchor = anchor->GetPeerOutAnchor();
    if ((out_anchor != nullptr) && (out_anchor->GetOwnerNode() != nullptr)) {
      NodePtr out_node = out_anchor->GetOwnerNode();
      if (out_node == nullptr) {
        GELOGW("out node is nullptr");
        continue;
      }
      if ((out_node->GetType() == NEXTITERATION) || (out_node->GetType() == REFNEXTITERATION)) {
        GE_IF_BOOL_EXEC(in_edge_size == 0, GELOGE(GRAPH_FAILED, "If [in_edge_size = 0], the result will be reversed");
                        return in_edge_size);
        in_edge_size -= 1;
      }
    }
  }
  if (node->GetInControlAnchor() != nullptr) {
    in_edge_size = in_edge_size + node->GetInControlAnchor()->GetPeerAnchorsSize();
  }
  return in_edge_size;
}

size_t ComputeGraph::GetOutEdgeSize(const NodePtr &node) {
  size_t out_edge_size = 0;
  if (node == nullptr) {
    return out_edge_size;
  }

  // Break flow control data loop.
  if ((node->GetType() != NEXTITERATION) && (node->GetType() != REFNEXTITERATION)) {
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      if (anchor != nullptr) {
        out_edge_size = out_edge_size + anchor->GetPeerAnchors().size();
      }
    }
  }
  if (node->GetOutControlAnchor() != nullptr) {
    if (out_edge_size > (UINT64_MAX - node->GetOutControlAnchor()->GetPeerAnchors().size())) {
      return 0;
    }
    out_edge_size = out_edge_size + node->GetOutControlAnchor()->GetPeerAnchors().size();
  }
  return out_edge_size;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::IsValid() const { return is_valid_flag_; }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::Dump() const {
  GELOGI("graph name = %s.", GetName().c_str());
  for (const auto &node : GetAllNodes()) {
    GELOGI("node name = %s.", node->GetName().c_str());
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
        GE_IF_BOOL_EXEC(peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNode() != nullptr,
                        GELOGI("node name = %s, out data node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
      for (const auto &peer_in_anchor : anchor->GetPeerInControlAnchors()) {
        GE_IF_BOOL_EXEC(peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNode() != nullptr,
                        GELOGI("node name = %s, out control node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
    }
    auto out_control_anchor = node->GetOutControlAnchor();
    if (out_control_anchor != nullptr) {
      for (const auto &peer_in_anchor : out_control_anchor->GetPeerInControlAnchors()) {
        GE_IF_BOOL_EXEC(peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNode() != nullptr,
                        GELOGI("node name = %s, out control node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
      for (const auto &peer_in_anchor : out_control_anchor->GetPeerInDataAnchors()) {
        GE_IF_BOOL_EXEC(peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNode() != nullptr,
                        GELOGI("node name = %s, out control node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::Swap(ComputeGraph &graph) {
  this->AttrHolder::Swap(graph);

  origGraph_.swap(graph.origGraph_);

  name_.swap(graph.name_);
  std::swap(graph_id_, graph.graph_id_);
  attrs_.Swap(graph.attrs_);
  nodes_.swap(graph.nodes_);
  all_nodes_infos_.swap(graph.all_nodes_infos_);
  target_nodes_info_.swap(graph.target_nodes_info_);

  input_nodes_.swap(graph.input_nodes_);
  inputs_order_.swap(graph.inputs_order_);
  std::swap(input_size_, graph.input_size_);
  out_nodes_map_.swap(graph.out_nodes_map_);
  std::swap(output_size_, graph.output_size_);
  output_nodes_info_.swap(graph.output_nodes_info_);

  sub_graph_.swap(graph.sub_graph_);
  names_to_subgraph_.swap(graph.names_to_subgraph_);
  parent_graph_.swap(graph.parent_graph_);
  parent_node_.swap(graph.parent_node_);

  // the members followed should not in the ComputeGraph class
  std::swap(is_valid_flag_, graph.is_valid_flag_);
  std::swap(is_summary_graph_, graph.is_summary_graph_);
  std::swap(need_iteration_, graph.need_iteration_);
  params_share_map_.swap(graph.params_share_map_);
  op_name_map_.swap(graph.op_name_map_);
  std::swap(session_id_, graph.session_id_);
  std::swap(data_format_, graph.data_format_);
  std::swap(is_unknown_shape_graph_, graph.is_unknown_shape_graph_);

  // Update Node owner.
  SetNodesOwner();
  graph.SetNodesOwner();
}

void ComputeGraph::SetNodesOwner() {
  for (const auto &node : nodes_) {
    if (node == nullptr) {
      continue;
    }
    node->SetOwnerComputeGraph(shared_from_this());
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::IsolateNode(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto next_nodes = node->GetOutAllNodes();
  // If there is input data side
  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); i++) {
    auto in_data_anchor = node->GetInDataAnchor(static_cast<int>(i));
    auto pre_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (pre_out_data_anchor != nullptr) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(pre_out_data_anchor, in_data_anchor) == GRAPH_SUCCESS,
                       return GRAPH_FAILED, "remove edge failed");
      GE_IF_BOOL_EXEC(pre_out_data_anchor->GetOwnerNode()->GetType() == CONSTANT ||
                        pre_out_data_anchor->GetOwnerNode()->GetType() == CONSTANTOP,
                      continue);
      for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
        for (const auto &next_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
          GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_data_anchor) == GRAPH_SUCCESS,
                           return GRAPH_FAILED, "remove edge failed");
          GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_data_anchor, next_in_data_anchor) == GRAPH_SUCCESS,
                           return GRAPH_FAILED, "add edge failed");
        }
        for (const auto &next_in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
          GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                           return GRAPH_FAILED, "remove edge failed");
          GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                           return GRAPH_FAILED, "add edge failed");
        }
      }
      auto out_ctrl_anchor = node->GetOutControlAnchor();
      GE_CHECK_NOTNULL(out_ctrl_anchor);
      auto pre_out_ctrl_anchor = pre_out_data_anchor->GetOwnerNode()->GetOutControlAnchor();
      GE_CHECK_NOTNULL(pre_out_ctrl_anchor);
      for (const auto &next_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
        GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         return GRAPH_FAILED, "remove edge failed");
        GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         return GRAPH_FAILED, "add edge failed");
      }
    }
  }

  // If there is an input control side
  auto in_ctrl_anchor = node->GetInControlAnchor();
  GE_CHECK_NOTNULL(in_ctrl_anchor);
  for (const auto &pre_out_ctrl_anchor : in_ctrl_anchor->GetPeerOutControlAnchors()) {
    GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(pre_out_ctrl_anchor, in_ctrl_anchor) == GRAPH_SUCCESS, return GRAPH_FAILED,
                     "remove edge failed");
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (const auto &next_in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
        GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         return GRAPH_FAILED, "remove edge failed");
        GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         return GRAPH_FAILED, "add edge failed");
      }
    }
    auto out_ctrl_anchor = node->GetOutControlAnchor();
    if (out_ctrl_anchor != nullptr) {
      for (const auto &next_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
        GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         return GRAPH_FAILED, "remove edge failed");
        GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         return GRAPH_FAILED, "add edge failed");
      }
    }
  }

  for (const auto &out_peer_data_anchor : in_ctrl_anchor->GetPeerOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_peer_data_anchor, in_ctrl_anchor) == GRAPH_SUCCESS, return GRAPH_FAILED,
                     "remove edge failed");
    for (const auto &next_node : next_nodes) {
      auto next_in_control_anchor = next_node->GetInControlAnchor();
      GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(out_peer_data_anchor, next_in_control_anchor) == GRAPH_SUCCESS,
                       return GRAPH_FAILED, "add edge failed");
    }
  }

  return RemoveExtraOutEdge(node);
}

graphStatus ComputeGraph::RemoveExtraOutEdge(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  // Remove redundant output edges
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    for (const auto &next_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_data_anchor) == GRAPH_SUCCESS,
                       return GRAPH_FAILED, "remove edge failed");
    }

    for (const auto &next_in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                       return GRAPH_FAILED, "remove edge failed");
    }
  }
  auto out_ctrl_anchor = node->GetOutControlAnchor();
  if (out_ctrl_anchor != nullptr) {
    for (const auto &next_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                       return GRAPH_FAILED, "remove edge failed");
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraph::Verify() {
  bool is_unknown_graph = GetGraphUnknownFlag();
  for (const auto &node_ptr : GetAllNodes()) {
    GE_CHECK_NOTNULL(node_ptr);
    GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
    GE_IF_BOOL_EXEC(is_unknown_graph, continue);
    GE_CHK_BOOL_EXEC(node_ptr->GetOpDesc()->CommonVerify() == GRAPH_SUCCESS, return GRAPH_FAILED,
                     "Verifying %s failed.", node_ptr->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::InferOriginFormat() {
  return ge::FormatRefiner::InferOrigineFormat(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::InferShapeInNeed() {
  GE_CHK_BOOL_ONLY_LOG(TopologicalSorting() == GRAPH_SUCCESS, "Verifying failed.");
  for (const auto &node_ptr : GetAllNodes()) {
    GE_CHECK_NOTNULL(node_ptr);
    auto op_desc = node_ptr->GetOpDesc();
    bool is_need_infer = false;
    (void)ge::AttrUtils::GetBool(op_desc, NEED_INFER, is_need_infer);
    if (is_need_infer) {
      GE_CHK_BOOL_EXEC(node_ptr->Verify() == GRAPH_SUCCESS, return GRAPH_FAILED, "Verifying %s failed.",
                       node_ptr->GetName().c_str());

      graphStatus status = node_ptr->InferShapeAndType();
      GE_CHK_BOOL_EXEC_INFO(node_ptr->GetType() == DATA || GRAPH_PARAM_INVALID != status, break,
                            "Op %s does not have the IMPLEMT_INFERFUNC definition,"
                            " and subsequent operators no longer perform shape inference.",
                            node_ptr->GetName().c_str());
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS, return GRAPH_FAILED, "Inferring %s failed.",
                       node_ptr->GetName().c_str());

      for (const auto &out_anchor : node_ptr->GetAllOutDataAnchors()) {
        GE_CHECK_NOTNULL(out_anchor->GetOwnerNode()->GetOpDesc());
        auto output_tensor = out_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx());
        ge::TensorUtils::SetRealDimCnt(output_tensor, output_tensor.GetShape().GetDims().size());
        (void)out_anchor->GetOwnerNode()->GetOpDesc()->UpdateOutputDesc(out_anchor->GetIdx(), output_tensor);
        for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
          (void)peer_anchor->GetOwnerNode()->GetOpDesc()->UpdateInputDesc(peer_anchor->GetIdx(), output_tensor);
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

ProtoAttrMapHelper ComputeGraph::MutableAttrMap() { return attrs_; }

ConstProtoAttrMapHelper ComputeGraph::GetAttrMap() const {
  return ConstProtoAttrMapHelper(attrs_.GetProtoOwner(), attrs_.GetProtoMsg());
}

const std::map<OperatorImplPtr, NodePtr> &ComputeGraph::GetAllNodesInfo() const { return all_nodes_infos_; }

void ComputeGraph::SetUserDefOutput(const std::string &output_name) {
  if (output_name.empty()) {
    return;
  }

  vector<string> nodes = StringUtils::Split(output_name, ';');
  for (string node : nodes) {
    vector<string> item = StringUtils::Split(node, ':');
    if (item.size() != OUTPUT_PARAM_SIZE) {
      GELOGW("invalid output param!input:%s", output_name.c_str());
      continue;
    }

    int32_t index;
    try {
      index = stoi(StringUtils::Trim(item[1]));
    } catch (const std::out_of_range &) {
      GELOGW("outputname cause out of range execption!output_name:%s", output_name.c_str());
      continue;
    } catch (const std::invalid_argument &) {
      GELOGW("outputname cause invalid argument!output_name:%s", output_name.c_str());
      continue;
    } catch (...) {
      GELOGW("stoi fail! output_name:%s", output_name.c_str());
      continue;
    }
    auto iter = out_nodes_map_.find(item[0]);
    if (iter == out_nodes_map_.end()) {
      out_nodes_map_[item[0]] = std::vector<int32_t>(1, index);
    } else {
      auto idx_iter = std::find(iter->second.begin(), iter->second.end(), index);
      if (idx_iter == iter->second.end()) {
        iter->second.push_back(index);
      }
    }
  }
}

const std::string ComputeGraph::GetOutput() {
  static const int resultDefaultSize = 2048;
  string result;
  result.reserve(resultDefaultSize);
  auto iter = out_nodes_map_.begin();
  while (iter != out_nodes_map_.end()) {
    auto idxes = iter->second;
    for (auto idx : idxes) {
      (void)result.append(iter->first).append(":").append(std::to_string(idx)).append(";");
    }
    ++iter;
  }

  return result.substr(0, result.length() - 1);
}
}  // namespace ge
