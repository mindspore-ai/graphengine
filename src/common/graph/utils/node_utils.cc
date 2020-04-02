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

#include "utils/node_utils.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "utils/tensor_utils.h"
#include "utils/type_utils.h"

namespace ge {
std::map<NodePtr, std::vector<uint32_t>> NodeUtils::map_send_info_{};
std::map<NodePtr, std::vector<uint32_t>> NodeUtils::map_recv_info_{};

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::AddSendEventId(const NodePtr &node,
                                                                                     const uint32_t &event_id) {
  GE_CHECK_NOTNULL(node);
  map_send_info_[node].push_back(event_id);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::AddRecvEventId(const NodePtr &node,
                                                                                     const uint32_t &event_id) {
  GE_CHECK_NOTNULL(node);
  map_recv_info_[node].push_back(event_id);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
NodeUtils::GetSendEventIdList(const NodePtr &node, std::vector<uint32_t> &vec_send) {
  GE_CHECK_NOTNULL(node);
  auto find = map_send_info_.find(node);
  if (find == map_send_info_.end()) {
    return GRAPH_FAILED;
  } else {
    vec_send = find->second;
    return GRAPH_SUCCESS;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
NodeUtils::GetRecvEventIdList(const NodePtr &node, std::vector<uint32_t> &vec_recv) {
  GE_CHECK_NOTNULL(node);
  auto find = map_recv_info_.find(node);
  if (find == map_recv_info_.end()) {
    return GRAPH_FAILED;
  } else {
    vec_recv = find->second;
    return GRAPH_SUCCESS;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::ClearSendInfo() {
  map_send_info_.clear();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::ClearRecvInfo() {
  map_recv_info_.clear();
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::GetSingleOutputNodeOfNthLayer(const NodePtr &src, int depth, NodePtr &dst) {
  GE_CHECK_NOTNULL(src);
  NodePtr cur_ptr;
  if (depth < 1) {
    return GRAPH_FAILED;
  }
  for (int i = 0; i < depth; i++) {
    if (src->GetOutDataNodes().size() != 1) {
      return GRAPH_FAILED;
    }
    cur_ptr = src->GetOutDataNodes().at(0);
    GE_CHECK_NOTNULL(cur_ptr);
  }
  dst = cur_ptr;
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::GetDataOutAnchorAndControlInAnchor(const NodePtr &node_ptr, OutDataAnchorPtr &out_data,
                                                          InControlAnchorPtr &in_control) {
  GE_CHECK_NOTNULL(node_ptr);
  for (const auto &p : node_ptr->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC((p != nullptr), continue, "GetAllOutDataAnchors is nullptr");
    for (const auto &p_in : p->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC((p_in != nullptr), continue, "GetPeerInDataAnchors is nullptr");
      out_data = p;
      in_control = p_in;
      return GRAPH_SUCCESS;
    }
  }
  return GRAPH_FAILED;
}

graphStatus NodeUtils::ClearInDataAnchor(const NodePtr &node_ptr, const InDataAnchorPtr &in_data_anchor) {
  GE_CHK_BOOL_EXEC(node_ptr != nullptr && in_data_anchor != nullptr, return GRAPH_FAILED,
                   "node or in_data_anchor is nullptr");
  bool find_flag = false;
  uint32_t index = 0;
  vector<InDataAnchorPtr>::iterator it = node_ptr->in_data_anchors_.end();
  for (const auto &tmp : node_ptr->in_data_anchors_) {
    if (tmp == in_data_anchor) {
      find_flag = true;
      auto iter = node_ptr->in_data_anchors_.begin() + index;
      if (iter != node_ptr->in_data_anchors_.end()) {
        it = node_ptr->in_data_anchors_.erase(iter);
      }
      break;
    }
    index++;
  }
  for (; it != node_ptr->in_data_anchors_.end(); ++it) {
    (*it)->SetIdx(index);
    index++;
  }

  if (!find_flag) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::SetAllAnchorStatus(const NodePtr &node_ptr) {
  GE_CHK_BOOL_EXEC(node_ptr != nullptr, return GRAPH_FAILED, "node is nullptr");
  GE_CHK_BOOL_EXEC(SetAllAnchorStatus(*node_ptr) == GRAPH_SUCCESS, return GRAPH_FAILED, "set all anchor status failed");
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::SetAllAnchorStatus(Node &node) {
  node.anchor_status_updated_ = true;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool NodeUtils::IsAnchorStatusSet(const NodePtr &node_ptr) {
  GE_CHK_BOOL_EXEC(node_ptr != nullptr, return false, "node is nullptr");
  return IsAnchorStatusSet(*node_ptr);
}

bool NodeUtils::IsAnchorStatusSet(const Node &node) { return node.anchor_status_updated_; }

graphStatus NodeUtils::MoveOutputEdges(const NodePtr &origin_node, const NodePtr &new_node) {
  if ((origin_node == nullptr) || (new_node == nullptr)) {
    return GRAPH_FAILED;
  }
  auto origin_out_data_anchors = origin_node->GetAllOutDataAnchors();
  auto new_out_data_anchors = new_node->GetAllOutDataAnchors();
  if (origin_out_data_anchors.size() != new_out_data_anchors.size()) {
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < origin_out_data_anchors.size(); ++i) {
    for (const auto &peer_anchor : origin_out_data_anchors.at(i)->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC(origin_out_data_anchors.at(i)->Unlink(peer_anchor) == GRAPH_SUCCESS, continue,
                       "unlink peer_anchor failed");
      GE_CHK_BOOL_EXEC(new_out_data_anchors.at(i)->LinkTo(peer_anchor) == GRAPH_SUCCESS, continue,
                       "linkto peer_anchor failed");
    }

    for (const auto &peer_anchor : origin_out_data_anchors.at(i)->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(origin_out_data_anchors.at(i)->Unlink(peer_anchor) == GRAPH_SUCCESS, continue,
                       "unlink peer_anchor failed");
      GE_CHK_BOOL_EXEC(new_out_data_anchors.at(i)->LinkTo(peer_anchor) == GRAPH_SUCCESS, continue,
                       "linkto peer_anchor failed");
    }
  }

  auto origin_out_control_anchor = origin_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(origin_out_control_anchor);
  auto new_out_control_anchor = new_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(new_out_control_anchor);
  for (const auto &peer_anchor : origin_out_control_anchor->GetPeerInControlAnchors()) {
    GE_CHK_BOOL_EXEC(new_out_control_anchor->LinkTo(peer_anchor) == GRAPH_SUCCESS, continue,
                     "linkto peer_anchor failed");
  }
  for (const auto &peer_anchor : origin_out_control_anchor->GetPeerInDataAnchors()) {
    GE_CHK_BOOL_EXEC(new_out_control_anchor->LinkTo(peer_anchor) == GRAPH_SUCCESS, continue,
                     "linkto peer_anchor failed");
  }
  origin_out_control_anchor->UnlinkAll();

  return GRAPH_SUCCESS;
}

bool NodeUtils::IsConst(const Node &node) {
  auto src_node_type = node.GetType();
  bool is_const = ((src_node_type == CONSTANT) || (src_node_type == CONSTANTOP));
  return is_const;
}

void NodeUtils::UpdateIsInputConst(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "node is null");
    return;
  }
  UpdateIsInputConst(*node_ptr);
}

///
/// update is_input_const
/// @param node
/// @return void
///
void NodeUtils::UpdateIsInputConst(Node &node) {
  std::vector<bool> is_input_const;
  size_t anchor_num = node.GetAllInDataAnchors().size();
  for (size_t i = 0; i < anchor_num; i++) {
    auto in_anchor = node.GetInDataAnchor(static_cast<int>(i));
    if (in_anchor == nullptr) {
      is_input_const.push_back(false);
      continue;
    }
    auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      is_input_const.push_back(false);
      continue;
    }
    auto src_node = peer_out_anchor->GetOwnerNode();
    if (src_node == nullptr) {
      is_input_const.push_back(false);
      continue;
    }
    if (IsConst(*(src_node))) {
      is_input_const.push_back(true);
    } else {
      is_input_const.push_back(false);
    }
  }
  if (node.GetOpDesc() == nullptr) {
    GELOGE(GRAPH_FAILED, "Node get opdesc is nullptr");
    return;
  }
  node.GetOpDesc()->SetIsInputConst(is_input_const);
}

void NodeUtils::UnlinkAll(const Node &node) {
  for (const auto &anchor : node.GetAllOutAnchors()) {
    anchor->UnlinkAll();
  }
  for (const auto &anchor : node.GetAllInAnchors()) {
    anchor->UnlinkAll();
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::UpdatePeerNodeInputDesc(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "Nodeptr is nullptr");
    return GRAPH_FAILED;
  }
  auto op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    return GRAPH_FAILED;
  }
  for (const auto &out_anchor : node_ptr->GetAllOutDataAnchors()) {
    GeTensorDesc output_tensor = op_desc->GetOutputDesc(out_anchor->GetIdx());
    ge::TensorUtils::SetRealDimCnt(output_tensor, static_cast<uint32_t>(output_tensor.GetShape().GetDims().size()));
    output_tensor.SetOriginShape(output_tensor.GetShape());
    output_tensor.SetOriginDataType(output_tensor.GetDataType());
    GELOGD("node name is %s, origin shape is %ld, origin format is %s, origin data type is %s",
           node_ptr->GetName().c_str(), output_tensor.GetOriginShape().GetShapeSize(),
           TypeUtils::FormatToSerialString(output_tensor.GetOriginFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor.GetOriginDataType()).c_str());
    (void)op_desc->UpdateOutputDesc(out_anchor->GetIdx(), output_tensor);
    for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (peer_anchor->GetOwnerNode()->GetOpDesc() == nullptr) {
        GELOGE(GRAPH_FAILED, "peer_anchor opdesc is null");
        continue;
      }
      auto peer_input_desc = peer_anchor->GetOwnerNode()->GetOpDesc()->GetInputDescPtr(peer_anchor->GetIdx());
      if (peer_input_desc == nullptr) {
        GELOGE(GRAPH_FAILED, "peer_input_desc is nullptr");
        continue;
      }
      output_tensor.SetOriginFormat(peer_input_desc->GetOriginFormat());
      output_tensor.SetFormat(peer_input_desc->GetFormat());
      auto peer_op_desc = peer_anchor->GetOwnerNode()->GetOpDesc();
      GE_IF_BOOL_EXEC(peer_op_desc == nullptr, GELOGE(GRAPH_FAILED, "peer opdesc is null"); continue);
      GE_IF_BOOL_EXEC(peer_op_desc->UpdateInputDesc(peer_anchor->GetIdx(), output_tensor) != GRAPH_SUCCESS,
                      GELOGE(GRAPH_FAILED, "peer opdesc is null");
                      continue);
    }
  }
  return GRAPH_SUCCESS;
}
bool NodeUtils::IsInNodesEmpty(const Node &node) {
  for (const auto &in_anchor : node.in_data_anchors_) {
    if (in_anchor != nullptr) {
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor != nullptr) {
        if (out_anchor->GetOwnerNode() != nullptr) {
          return false;
        }
      }
    }
  }

  if ((node.in_control_anchor_ != nullptr) && (!node.in_control_anchor_->IsPeerOutAnchorsEmpty())) {
    auto peer_out_control_anchors = node.in_control_anchor_->GetPeerOutControlAnchors();
    for (const auto &out_control_anchor : peer_out_control_anchors) {
      if (out_control_anchor != nullptr) {
        if (out_control_anchor->GetOwnerNode() != nullptr) {
          return false;
        }
      }
    }
  }

  return true;
}
GeTensorDesc NodeUtils::GetOutputDesc(const Node &node, uint32_t index) {
  auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GeTensorDesc();
  }
  return desc->GetOutputDesc(index);
}
GeTensorDesc NodeUtils::GetInputDesc(const Node &node, uint32_t index) {
  auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GeTensorDesc();
  }
  return desc->GetInputDesc(index);
}
graphStatus NodeUtils::UpdateOutputShape(const Node &node, uint32_t index, const GeShape &shape) {
  auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  auto output_desc = desc->MutableOutputDesc(index);
  if (output_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  output_desc->SetShape(shape);
  return GRAPH_SUCCESS;
}
graphStatus NodeUtils::UpdateInputShape(const Node &node, uint32_t index, const GeShape &shape) {
  auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  auto input_desc = desc->MutableInputDesc(index);
  if (input_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  input_desc->SetShape(shape);
  return GRAPH_SUCCESS;
}
}  // namespace ge
