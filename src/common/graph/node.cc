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

#include "graph/node.h"
#include <utility>
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "external/graph/operator_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_tensor.h"
#include "graph/operator_factory_impl.h"
#include "graph/shape_refiner.h"
#include "utils/ge_ir_utils.h"
#include "utils/node_utils.h"
#include "utils/op_desc_utils.h"
#include "common/util/error_manager/error_manager.h"

using std::string;
using std::vector;

namespace ge {
Node::Node(const OpDescPtr &op, const ComputeGraphPtr &owner_graph)
    : op_(op),
      owner_graph_(owner_graph),
      in_data_anchors_(),
      out_data_anchors_(),
      in_control_anchor_(nullptr),
      out_control_anchor_(nullptr),
      attrs_(),
      has_init_(false) {
  anchor_status_updated_ = false;
}

Node::~Node() {
  for (const auto &in_data_anchor : in_data_anchors_) {
    if (in_data_anchor != nullptr) {
      in_data_anchor->UnlinkAll();
    }
  }
  for (const auto &out_data_anchor : out_data_anchors_) {
    if (out_data_anchor != nullptr) {
      out_data_anchor->UnlinkAll();
    }
  }
  if (in_control_anchor_ != nullptr) {
    in_control_anchor_->UnlinkAll();
  }
  if (out_control_anchor_ != nullptr) {
    out_control_anchor_->UnlinkAll();
  }
}

graphStatus Node::Init() {
  if (has_init_) {
    return GRAPH_SUCCESS;
  }
  GE_CHK_BOOL_EXEC(op_ != nullptr, return GRAPH_FAILED, "original OpDesc is nullptr");
  size_t size = op_->GetAllInputsSize();
  for (size_t i = 0; i < size; i++) {
    std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(shared_from_this(), i);
    if (anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "Current in_data_anchor is null, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    in_data_anchors_.push_back(anchor);
  }
  size = op_->GetOutputsSize();
  for (size_t i = 0; i < size; i++) {
    std::shared_ptr<OutDataAnchor> anchor = ComGraphMakeShared<OutDataAnchor>(shared_from_this(), i);
    if (anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "Current out_data_anchor is null, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    out_data_anchors_.push_back(anchor);
  }
  in_control_anchor_ = ComGraphMakeShared<InControlAnchor>(shared_from_this(), -1);
  out_control_anchor_ = ComGraphMakeShared<OutControlAnchor>(shared_from_this(), -1);
  if (in_control_anchor_ == nullptr || out_control_anchor_ == nullptr) {
    GELOGE(GRAPH_FAILED, "Current in_control_anchor or out_control_anchor is null, malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  has_init_ = true;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string Node::GetName() const {
  GE_CHK_BOOL_EXEC(op_ != nullptr, return string(), "original OpDesc is nullptr");
  return op_->GetName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string Node::GetType() const {
  GE_CHK_BOOL_EXEC(op_ != nullptr, return string(), "original OpDesc is nullptr");
  return op_->GetType();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeAttrsAreEqual(const Node &r_node) const {
  const auto &attr_map = this->attrs_;
  const auto &r_attr_map = r_node.attrs_;
  // 1.Verify node's map<string, AttrValue> size
  if (attr_map.size() != r_attr_map.size()) {
    GELOGE(GRAPH_FAILED, "Size of node's attr map verify failed, node name: %s.", this->GetName().c_str());
    return false;
  }
  // 2.Verify node's map<string, AttrValue> key, verify values is temporarily not implemented
  for (const auto &it : attr_map) {
    if (r_attr_map.count(it.first) == 0) {
      GELOGE(GRAPH_FAILED, "Key of node's attr map verify failed, node name: %s key name: %s.", this->GetName().c_str(),
             it.first.c_str());
      return false;
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeMembersAreEqual(const Node &r_node) const {
  return ((((this->op_ != nullptr) && (r_node.op_ != nullptr) && (IsEqual(*(this->op_), *(r_node.op_), "node.op_"))) ||
           ((this->op_ == nullptr) && (r_node.op_ == nullptr))) &&
          IsEqual(this->has_init_, r_node.has_init_, "node.has_init_") &&
          IsEqual(this->anchor_status_updated_, r_node.anchor_status_updated_, "node.anchor_status_updated_") &&
          IsEqual(this->send_event_id_list_, r_node.send_event_id_list_, "node.send_event_id_list_") &&
          IsEqual(this->recv_event_id_list_, r_node.recv_event_id_list_, "node.recv_event_id_list_"));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeAnchorIsEqual(const AnchorPtr &left_anchor,
                                                                            const AnchorPtr &right_anchor,
                                                                            size_t i) const {
  GE_IF_BOOL_EXEC(left_anchor == nullptr, GELOGE(GRAPH_FAILED, "left_anchor is null."); return false);
  GE_IF_BOOL_EXEC(right_anchor == nullptr, GELOGE(GRAPH_FAILED, "right_anchor is null."); return false);

  const auto anchor_peer_size = left_anchor->GetPeerAnchors().size();
  const auto right_anchor_peer_size = right_anchor->GetPeerAnchors().size();
  // Firstly, verify anchor's peer anchors size equal or not
  if (anchor_peer_size != right_anchor_peer_size) {
    GELOGE(GRAPH_FAILED,
           "Size of anchor's peer anchors verify failed, node name: %s "
           "anchor_peer_size [%zu]  is different form [%zu] at index [%zu].",
           this->GetName().c_str(), anchor_peer_size, right_anchor_peer_size, i);
    return false;
  }
  // Secondly, verify anchor's peer anchor owner node equal or not
  for (size_t j = 0; j < anchor_peer_size; j++) {
    const auto &peer_node = left_anchor->GetPeerAnchors().at(j)->GetOwnerNode();
    const auto &r_peer_node = right_anchor->GetPeerAnchors().at(j)->GetOwnerNode();
    if (peer_node == nullptr || r_peer_node == nullptr) {
      GELOGE(GRAPH_FAILED, "anchor's peer node is null, node name: %s index[%zu] peer node index[%zu]. ",
             this->GetName().c_str(), i, j);
      return false;
    }
    // Determine the connection relationship by linking the node's name
    if (peer_node->GetName() != r_peer_node->GetName()) {
      GELOGE(GRAPH_FAILED,
             "anchor's peer node name verify failed, node name: %s index[%zu]"
             "peer node name %s is different from %s at index [%zu].",
             this->GetName().c_str(), i, peer_node->GetName().c_str(), r_peer_node->GetName().c_str(), j);
      return false;
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeInConnectsAreEqual(const Node &r_node) const {
  // 1.Verify all in data and control anchors size
  const auto in_data_anchor_size = this->GetAllInDataAnchors().size();
  const auto r_in_data_anchor_size = r_node.GetAllInDataAnchors().size();
  if (in_data_anchor_size != r_in_data_anchor_size) {
    GELOGE(GRAPH_FAILED, "Size of node's in data anchors verify failed, node name: %s.", this->GetName().c_str());
    return false;
  }
  const auto l_in_anchors = this->GetAllInAnchors();
  const auto r_in_anchors = r_node.GetAllInAnchors();
  // Data anchors size equal, all anchors size not equal, means control anchor size not equal
  const auto in_control_anchor_size = l_in_anchors.size() - in_data_anchor_size;
  const auto r_in_control_anchor_size = r_in_anchors.size() - r_in_data_anchor_size;
  if (in_control_anchor_size != r_in_control_anchor_size) {
    GELOGE(GRAPH_FAILED, "Size of node's in control anchors verify failed, node name: %s.", this->GetName().c_str());
    return false;
  }
  // 2.Verify all in data and control anchors connect info
  for (size_t i = 0; i < this->GetAllInAnchors().size(); i++) {
    // Verify data anchors
    if (i < in_data_anchor_size) {
      const auto &in_anchor = l_in_anchors.at(i);
      const auto &r_in_anchor = r_in_anchors.at(i);
      if (!(NodeAnchorIsEqual(in_anchor, r_in_anchor, i))) {
        GELOGE(GRAPH_FAILED, "Node's in data control anchor verify failed, node name: %s.", this->GetName().c_str());
        return false;
      }
    } else {
      // Verify control anchors
      const auto &in_control_anchor = l_in_anchors.at(i);
      const auto &r_in_control_anchor = r_in_anchors.at(i);
      if (!(NodeAnchorIsEqual(in_control_anchor, r_in_control_anchor, i - in_data_anchor_size))) {
        GELOGE(GRAPH_FAILED, "Node's in control anchor verify failed, node name: %s.", this->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeOutConnectsAreEqual(const Node &r_node) const {
  // 1.Verify all out data and control anchors size
  const auto l_out_data_anchors = this->GetAllOutDataAnchors();
  const auto r_out_data_anchors = r_node.GetAllOutDataAnchors();
  const auto out_data_anchor_size = l_out_data_anchors.size();
  const auto r_out_data_anchor_size = r_out_data_anchors.size();
  if (out_data_anchor_size != r_out_data_anchor_size) {
    GELOGE(GRAPH_FAILED, "Size of node's out data anchors verify failed, node name: %s.", this->GetName().c_str());
    return false;
  }
  const auto l_out_anchors = this->GetAllOutAnchors();
  const auto r_out_anchors = r_node.GetAllOutAnchors();
  // Data anchors size equal, all anchors size not equal, means control anchor size not equal
  const auto out_control_anchor_size = l_out_anchors.size() - out_data_anchor_size;
  const auto r_out_control_anchor_size = r_out_anchors.size() - r_out_data_anchor_size;
  if (out_control_anchor_size != r_out_control_anchor_size) {
    GELOGE(GRAPH_FAILED, "Size of node's out control anchors verify failed, node name: %s.", this->GetName().c_str());
    return false;
  }

  // 2.Verify all out data and control anchors connect info
  for (size_t i = 0; i < this->GetAllOutAnchors().size(); i++) {
    // Verify data anchors
    if (i < out_data_anchor_size) {
      const auto &out_anchor = l_out_data_anchors.at(i);
      const auto &r_out_anchor = r_out_data_anchors.at(i);
      if (!(NodeAnchorIsEqual(out_anchor, r_out_anchor, i))) {
        GELOGE(GRAPH_FAILED, "Node's out data control anchor verify failed, node name: %s.", this->GetName().c_str());
        return false;
      }
    } else {
      // Verify control anchors
      const auto &out_control_anchor = l_out_anchors.at(i);
      const auto &r_out_control_anchor = r_out_anchors.at(i);
      if (!(NodeAnchorIsEqual(out_control_anchor, r_out_control_anchor, i - out_data_anchor_size))) {
        GELOGE(GRAPH_FAILED, "Node's out control anchor verify failed, node name: %s.", this->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::operator==(const Node &r_node) const {
  return (NodeMembersAreEqual(r_node) && NodeAttrsAreEqual(r_node) && NodeInConnectsAreEqual(r_node) &&
          NodeOutConnectsAreEqual(r_node));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFrom(const NodePtr &input_node) {
  // This function is deprecated, please use other two overloaded functions
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1) {
    GELOGE(GRAPH_FAILED, "out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }
  GE_CHK_BOOL_EXEC(op_ != nullptr, return GRAPH_FAILED, "original OpDesc is nullptr");
  auto op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  if (op_->AddInputDesc(op_desc->GetOutputDesc(0)) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "add input desc failed.");
    return GRAPH_FAILED;
  }
  std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(shared_from_this(), in_data_anchors_.size());
  if (anchor == nullptr) {
    GELOGE(GRAPH_FAILED, "out_anchor size is:%zu, malloc shared_ptr failed.", out_anchors.size());
    return GRAPH_FAILED;
  }
  in_data_anchors_.push_back(anchor);
  (void)out_anchors.at(0)->LinkTo(in_data_anchors_.back());

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFrom(const uint32_t &index,
                                                                             NodePtr input_node) {
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1) {
    GELOGE(GRAPH_FAILED, "out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  GE_CHECK_NOTNULL(op_);
  auto op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  if (op_->AddInputDesc(index, op_desc->GetOutputDesc(0)) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "add input desc failed.");
    return GRAPH_FAILED;
  }

  if (index < GetAllInDataAnchors().size()) {
    (void)out_anchors.at(0)->LinkTo(in_data_anchors_[index]);
  } else {
    std::shared_ptr<InDataAnchor> anchor =
      ComGraphMakeShared<InDataAnchor>(shared_from_this(), in_data_anchors_.size());
    if (anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "out_anchor size is:%zu, malloc shared_ptr failed.", out_anchors.size());
      return GRAPH_FAILED;
    }
    in_data_anchors_.push_back(anchor);
    (void)out_anchors.at(0)->LinkTo(in_data_anchors_.back());
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFromForParse(const NodePtr &input_node) {
  //  This function is used for ParseWeights.
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1) {
    GELOGE(GRAPH_PARAM_INVALID, "out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(shared_from_this(), in_data_anchors_.size());
  if (anchor == nullptr) {
    GELOGE(GRAPH_FAILED, "out_anchor size is:%zu, make anchor failed", out_anchors.size());
    return GRAPH_FAILED;
  }
  in_data_anchors_.push_back(anchor);
  (void)out_anchors.at(0)->LinkTo(in_data_anchors_.back());

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFrom(const string &name, NodePtr input_node) {
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1) {
    GELOGE(GRAPH_PARAM_INVALID, "out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  GE_CHECK_NOTNULL(op_);
  auto input_op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(input_op_desc);
  auto index = op_->GetInputIndexByName(name);
  if (index != -1) {
    if (index >= static_cast<int>(in_data_anchors_.size())) {
      GELOGE(GRAPH_FAILED, "op %s get input name %s 's index %d is illegal.", op_->GetName().c_str(), name.c_str(),
             index);
      return GRAPH_FAILED;
    }
    (void)out_anchors.at(0)->LinkTo(in_data_anchors_[index]);
  } else {
    std::shared_ptr<InDataAnchor> anchor =
      ComGraphMakeShared<InDataAnchor>(shared_from_this(), in_data_anchors_.size());
    if (anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "in_data_anchors_size is:%zu, malloc shared_ptr failed.", in_data_anchors_.size());
      return GRAPH_FAILED;
    }
    in_data_anchors_.push_back(anchor);
    (void)out_anchors.at(0)->LinkTo(in_data_anchors_.back());
  }
  if (op_->AddInputDesc(name, input_op_desc->GetOutputDesc(0)) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "add input desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr Node::GetOwnerComputeGraph() const {
  return owner_graph_.lock();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::SetOwnerComputeGraph(const ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  owner_graph_ = graph;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<InDataAnchorPtr> Node::GetAllInDataAnchors() const {
  return Vistor<InDataAnchorPtr>(shared_from_this(), in_data_anchors_);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<OutDataAnchorPtr> Node::GetAllOutDataAnchors() const {
  return Vistor<OutDataAnchorPtr>(shared_from_this(), out_data_anchors_);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetAllInDataAnchorsSize() const {
  return in_data_anchors_.size();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetAllOutDataAnchorsSize() const {
  return out_data_anchors_.size();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<AnchorPtr> Node::GetAllInAnchors() const {
  std::vector<AnchorPtr> vec;
  // Push back in_data_anchors_
  for (const auto &in_anchor_iter : Vistor<InDataAnchorPtr>(shared_from_this(), in_data_anchors_)) {
    auto in_anchor = Anchor::DynamicAnchorCast<Anchor>(in_anchor_iter);
    if (in_anchor != nullptr) {
      vec.push_back(in_anchor);
    }
  }
  // Push back in_control_anchor_
  if ((in_control_anchor_->GetPeerOutControlAnchors().size() > 0) ||
      (in_control_anchor_->GetPeerOutDataAnchors().size() > 0)) {
    auto in_anchor = Anchor::DynamicAnchorCast<Anchor>(in_control_anchor_);
    if (in_anchor != nullptr) {
      vec.push_back(in_anchor);
    }
  }
  return Node::Vistor<AnchorPtr>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<AnchorPtr> Node::GetAllOutAnchors() const {
  std::vector<AnchorPtr> vec;
  // Push back out_data_anchors_
  for (const auto &out_anchor_iter : Vistor<OutDataAnchorPtr>(shared_from_this(), out_data_anchors_)) {
    auto out_anchor = Anchor::DynamicAnchorCast<Anchor>(out_anchor_iter);
    if (out_anchor != nullptr) {
      vec.push_back(out_anchor);
    }
  }
  // Push back out_control_anchor_
  if (out_control_anchor_->GetPeerInControlAnchors().size() > 0 ||
      out_control_anchor_->GetPeerInDataAnchors().size() > 0) {
    auto out_anchor = Anchor::DynamicAnchorCast<Anchor>(out_control_anchor_);
    if (out_anchor != nullptr) {
      vec.push_back(out_anchor);
    }
  }
  return Node::Vistor<AnchorPtr>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InDataAnchorPtr Node::GetInDataAnchor(int idx) const {
  if (idx < 0 || idx >= static_cast<int>(in_data_anchors_.size())) {
    ErrorManager::GetInstance().ATCReportErrMessage(
      "E19019", {"opname", "index", "anchorname", "optype"},
      {GetName().c_str(), std::to_string(idx), "in_data_anchor", GetType().c_str()});
    GELOGE(GRAPH_FAILED, "Op[%s] doesn't have index[%d]'s in_data_anchor which optype is %s.", GetName().c_str(), idx,
           GetType().c_str());
    return nullptr;
  } else {
    return in_data_anchors_[idx];
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AnchorPtr Node::GetInAnchor(int idx) const {
  // Idx can't be less than -1 or >= in_data_anchors_.size(), -1 means index of control anchor_
  if (idx < -1 || idx >= static_cast<int>(in_data_anchors_.size())) {
    GELOGW("Op[%s] doesn't have index[%d]'s in_anchor which optype is %s.", GetName().c_str(), idx, GetType().c_str());
    return nullptr;
  } else {
    // Return control anchor
    if (idx == -1) {
      auto in_anchor = Anchor::DynamicAnchorCast<Anchor>(in_control_anchor_);
      return in_anchor;
    }
    // Return data anchor
    return in_data_anchors_[idx];
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AnchorPtr Node::GetOutAnchor(int idx) const {
  // Idx can't be less than -1 or >= out_data_anchors_.size(), -1 means index of control anchor_
  if (idx < -1 || idx >= static_cast<int>(out_data_anchors_.size())) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19019", {"opname", "index", "anchorname", "optype"},
                                                    {
                                                      GetName().c_str(),
                                                      std::to_string(idx),
                                                      "out_anchor",
                                                      GetType().c_str(),
                                                    });
    GELOGE(GRAPH_FAILED, "Op[%s] doesn't have index[%d]'s out_anchor which optype is %s.", GetName().c_str(), idx,
           GetType().c_str());
    return nullptr;
  } else {
    // Return control anchor
    if (idx == -1) {
      auto out_anchor = Anchor::DynamicAnchorCast<Anchor>(out_control_anchor_);
      return out_anchor;
    }
    // Return data anchor
    return out_data_anchors_[idx];
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OutDataAnchorPtr Node::GetOutDataAnchor(int idx) const {
  if (idx < 0 || idx >= static_cast<int>(out_data_anchors_.size())) {
    ErrorManager::GetInstance().ATCReportErrMessage(
      "E19019", {"opname", "index", "anchorname", "optype"},
      {GetName().c_str(), std::to_string(idx), "out_data_anchor", GetType().c_str()});
    GELOGE(GRAPH_FAILED, "Op[%s] doesn't have index[%d]'s out_data_anchor which optype is %s.", GetName().c_str(), idx,
           GetType().c_str());
    return nullptr;
  } else {
    return out_data_anchors_[idx];
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InControlAnchorPtr Node::GetInControlAnchor() const {
  return in_control_anchor_;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OutControlAnchorPtr Node::GetOutControlAnchor() const {
  return out_control_anchor_;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInNodes() const {
  std::vector<NodePtr> vec;
  for (const auto &in_anchor : in_data_anchors_) {
    GE_CHK_BOOL_EXEC((in_anchor != nullptr), continue, "in_data_anchor is nullptr");
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    auto node = out_anchor->GetOwnerNode();
    GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
    vec.push_back(node);
  }
  if (in_control_anchor_ != nullptr) {
    if (in_control_anchor_->IsPeerOutAnchorsEmpty()) {
      return Node::Vistor<NodePtr>(shared_from_this(), vec);
    }

    auto peer_out_anchors = in_control_anchor_->GetPeerOutDataAnchors();
    for (const auto &out_anchor : peer_out_anchors) {
      GE_CHK_BOOL_EXEC(out_anchor != nullptr, continue, "in_control_anchor_ peer out data anchors is nullptr");
      auto node = out_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }

    auto peer_out_control_anchors = in_control_anchor_->GetPeerOutControlAnchors();
    for (const auto &out_control_anchor : peer_out_control_anchors) {
      GE_CHK_BOOL_EXEC(out_control_anchor != nullptr, continue,
                       "in_control_anchor_ peer out control anchors is nullptr");
      auto node = out_control_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::IsAllInNodesSeen(
  std::unordered_set<Node *> &nodes_seen) const {
  for (const auto &in_anchor : in_data_anchors_) {
    GE_CHK_BOOL_EXEC((in_anchor != nullptr), continue, "in_data_anchor is nullptr");
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    auto node = out_anchor->GetOwnerNode();
    GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
    if ((node->GetType() == NEXTITERATION) || (node->GetType() == REFNEXTITERATION)) {
      continue;
    }
    if (nodes_seen.count(node.get()) == 0) {
      return false;
    }
  }

  if (in_control_anchor_ != nullptr) {
    if (in_control_anchor_->IsPeerOutAnchorsEmpty()) {
      return true;
    }
    auto peer_out_control_anchors = in_control_anchor_->GetPeerOutControlAnchors();
    for (const auto &out_control_anchor : peer_out_control_anchors) {
      GE_CHK_BOOL_EXEC(out_control_anchor != nullptr, continue, "out_control_anchor is nullptr");
      auto node = out_control_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      if ((node->GetType() == NEXTITERATION) || (node->GetType() == REFNEXTITERATION)) {
        continue;
      }
      if (nodes_seen.count(node.get()) == 0) {
        return false;
      }
    }
  }

  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInDataNodes() const {
  std::vector<NodePtr> vec;
  for (const auto &in_anchor : in_data_anchors_) {
    GE_CHK_BOOL_EXEC((in_anchor != nullptr), continue, "in_data_anchor is nullptr");
    auto anchor_ptr = in_anchor->GetPeerOutAnchor();
    if (anchor_ptr == nullptr) {
      continue;
    }
    auto node = anchor_ptr->GetOwnerNode();
    GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
    vec.push_back(node);
  }
  return Node::Vistor<NodePtr>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInControlNodes() const {
  std::vector<NodePtr> vec;
  if (in_control_anchor_ != nullptr) {
    for (const auto &in_anchor : in_control_anchor_->GetPeerOutControlAnchors()) {
      GE_CHK_BOOL_EXEC(in_anchor != nullptr, continue, "GetPeerOutControlAnchors is nullptr");
      auto node = in_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutNodes() const {
  std::vector<NodePtr> vec;
  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), continue, "out_data_anchors_ is nullptr");
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC((peer_in_anchor != nullptr), continue, "GetPeerInDataAnchors is nullptr");
      auto node = peer_in_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
  }
  if (out_control_anchor_ != nullptr) {
    auto peer_in_control_anchors = out_control_anchor_->GetPeerInControlAnchors();
    for (const auto &in_control_anchor : peer_in_control_anchors) {
      GE_CHK_BOOL_EXEC(in_control_anchor != nullptr, continue,
                       "out_control_anchor_ peer in control anchors is nullptr");
      auto node = in_control_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInAllNodes() const {
  std::vector<NodePtr> vec;
  for (const auto &in_node : GetInDataNodes()) {
    vec.push_back(in_node);
  }
  for (const auto &in_control_node : GetInControlNodes()) {
    vec.push_back(in_control_node);
  }
  return Node::Vistor<NodePtr>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutDataNodes() const {
  std::vector<NodePtr> vec;
  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), continue, "out_data_anchors_ is nullptr");
    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC((in_anchor != nullptr), continue, "GetPeerInDataAnchors is nullptr");
      auto node = in_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetOutDataNodesSize() const {
  uint32_t out_nums = 0;
  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), continue, "out_data_anchors_ is nullptr");
    out_nums += out_anchor->GetPeerInDataNodesSize();
  }
  return out_nums;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutControlNodes() const {
  std::vector<NodePtr> vec;

  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), continue, "out_data_anchors_ is nullptr");
    for (const auto &in_anchor : out_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC((in_anchor != nullptr), continue, "GetPeerInControlAnchors is nullptr");
      auto node = in_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
  }

  if (out_control_anchor_ != nullptr) {
    for (const auto &in_anchor : out_control_anchor_->GetPeerAnchors()) {
      GE_CHK_BOOL_EXEC(in_anchor != nullptr, continue, "GetPeerInControlAnchors is nullptr");
      auto node = in_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
  }

  return Node::Vistor<NodePtr>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutAllNodes() const {
  std::vector<NodePtr> vec;
  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), { continue; }, "out_data_anchors_ is nullptr");
    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC((in_anchor != nullptr), { continue; }, "GetPeerInDataAnchors is nullptr");
      auto node = in_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
    for (const auto &in_anchor : out_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(in_anchor != nullptr, continue, "GetPeerInControlAnchors is nullptr");
      auto node = in_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
  }

  if (out_control_anchor_ != nullptr) {
    for (const auto &in_anchor : out_control_anchor_->GetPeerAnchors()) {
      GE_CHK_BOOL_EXEC(in_anchor != nullptr, continue, "GetPeerInControlAnchors is nullptr");
      auto node = in_anchor->GetOwnerNode();
      GE_CHK_BOOL_EXEC(node != nullptr, continue, "GetOwnerNode is nullptr");
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(shared_from_this(), vec);
}

graphStatus Node::InferShapeAndType() const {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(shared_from_this());
  graphStatus ret = ShapeRefiner::InferShapeAndType(shared_from_this(), op);
  return ret;
}

graphStatus Node::InferOriginFormat() const {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(shared_from_this());
  // Get infer func and execute
  GE_CHK_BOOL_EXEC(op_ != nullptr, return GRAPH_FAILED, "original OpDesc is nullptr");
  return op_->CallInferFormatFunc(op);
}
graphStatus Node::Verify() const {
  const string data_type = "Data";
  const string aipp_data_type = "AippData";
  const string const_type = "Const";
  const string variable_type = "Variable";
  bool is_unknown_graph = GetOwnerComputeGraph()->GetGraphUnknownFlag();
  GE_CHK_BOOL_EXEC(op_ != nullptr, return GRAPH_FAILED, "original OpDesc is nullptr");

  if (!is_unknown_graph) {
    for (const auto &in_anchor_ptr : GetAllInDataAnchors()) {
      GE_IF_BOOL_EXEC(in_anchor_ptr == nullptr, GELOGW("in anchor ptr is null"); continue);
      bool valid_anchor =
        op_->GetType() == data_type || op_->GetType() == aipp_data_type || op_->GetType() == const_type ||
        op_->GetType() == variable_type || op_->IsOptionalInput(in_anchor_ptr->GetIdx()) ||
        op_->MutableInputDesc(in_anchor_ptr->GetIdx()) == nullptr || in_anchor_ptr->GetPeerAnchors().size() > 0;
      if (!valid_anchor) {
        ErrorManager::GetInstance().ATCReportErrMessage("E11019", {"opname", "index"},
                                                        {GetName(), std::to_string(in_anchor_ptr->GetIdx())});
        GELOGE(GRAPH_FAILED, "operator %s's input %d is not linked.", GetName().c_str(), in_anchor_ptr->GetIdx());
        return GRAPH_FAILED;
      }
    }
  }

  string frameworkop_type = "FrameworkOp";
  bool need_update_name = op_->GetType() != frameworkop_type && !is_unknown_graph;
  if (need_update_name) {
    auto node_op = ge::OperatorFactoryImpl::CreateOperator("node_op", op_->GetType());
    if (node_op.IsEmpty()) {
      GELOGW("get op from OperatorFactory fail. opType: %s", op_->GetType().c_str());
    } else {
      GELOGD("get op from OperatorFactory success. opType: %s", op_->GetType().c_str());
      auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
      if (temp_op_desc == nullptr) {
        GELOGE(GRAPH_FAILED, "temp op desc is null");
        return GRAPH_FAILED;
      }
      if (!op_->UpdateInputName(temp_op_desc->GetAllInputName())) {
        GELOGW("Verify UpdateInputName failed");
      }
      if (!op_->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
        GELOGW("Verify UpdateOutputName failed");
      }
    }
    node_op.BreakConnect();
  }
  GE_IF_BOOL_EXEC(is_unknown_graph, return GRAPH_SUCCESS;);
  if (op_->CommonVerify() == GRAPH_SUCCESS) {
    Operator op_proxy = ge::OpDescUtils::CreateOperatorFromNode(shared_from_this());
    auto verify_func = op_->GetVerifyFunc();
    if (verify_func == nullptr) {
      verify_func = OperatorFactoryImpl::GetVerifyFunc(GetType());
    }
    if (verify_func != nullptr) {
      return (graphStatus)verify_func(op_proxy);
    }
    return GRAPH_SUCCESS;
  } else {
    GELOGE(GRAPH_FAILED, "%s Verify failed.", op_->GetType().c_str());
    return GRAPH_FAILED;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr Node::GetOpDesc() const { return op_; }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::UpdateOpDesc(const OpDescPtr &op_desc) {
  GE_CHK_BOOL_EXEC(op_ != nullptr, return GRAPH_FAILED, "original OpDesc is nullptr");
  GE_CHK_BOOL_EXEC(op_desc != nullptr, return GRAPH_PARAM_INVALID, "Param OpDesc is nullptr");
  GE_CHK_BOOL_EXEC(op_->GetInputsSize() == op_desc->GetInputsSize(), return GRAPH_PARAM_INVALID,
                   "Inputs count expected to be same, orginial OpDesc %zu, Param OpDesc %zu", op_->GetInputsSize(),
                   op_desc->GetInputsSize());

  GE_CHK_BOOL_EXEC(op_->GetOutputsSize() == op_desc->GetOutputsSize(), return GRAPH_PARAM_INVALID,
                   "Outputs count expected to be same, orginial OpDesc %zu, Param OpDesc %zu", op_->GetOutputsSize(),
                   op_desc->GetOutputsSize());
  op_ = op_desc;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>
Node::GetInDataNodesAndAnchors() const {
  std::vector<std::pair<NodePtr, OutDataAnchorPtr>> vec;
  for (const auto &p : in_data_anchors_) {
    if (p == nullptr) {
      GELOGW("indata anchor is nullptr, node %s:%s", GetType().c_str(), GetName().c_str());
      continue;
    }
    auto anchor_ptr = p->GetPeerOutAnchor();
    if (anchor_ptr == nullptr) {
      continue;
    }
    auto node = anchor_ptr->GetOwnerNode();
    if (node == nullptr) {
      GELOGW("src node is nullptr, node %s:%s", GetType().c_str(), GetName().c_str());
      continue;
    }
    vec.push_back(std::make_pair(node, anchor_ptr));
  }
  return Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>(shared_from_this(), vec);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<std::pair<NodePtr, InDataAnchorPtr>>
Node::GetOutDataNodesAndAnchors() const {
  std::vector<std::pair<NodePtr, InDataAnchorPtr>> vec;
  for (const auto &p : out_data_anchors_) {
    if (p == nullptr) {
      GELOGW("out data anchor is nullptr, node %s:%s", GetType().c_str(), GetName().c_str());
      continue;
    }
    for (const auto &in_anchor : p->GetPeerInDataAnchors()) {
      if (in_anchor == nullptr) {
        GELOGW("dst in data anchor is nullptr, node %s:%s", GetType().c_str(), GetName().c_str());
        continue;
      }
      auto node = in_anchor->GetOwnerNode();
      if (node == nullptr) {
        GELOGW("dst node is nullptr, node %s:%s", GetType().c_str(), GetName().c_str());
        continue;
      }
      vec.push_back(std::make_pair(node, in_anchor));
    }
  }
  return Node::Vistor<std::pair<NodePtr, InDataAnchorPtr>>(shared_from_this(), vec);
}
}  // namespace ge
