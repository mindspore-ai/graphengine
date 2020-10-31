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

#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "external/graph/operator.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/type_utils.h"

namespace ge {
std::map<NodePtr, std::vector<uint32_t>> NodeUtils::map_send_info_{};
std::map<NodePtr, std::vector<uint32_t>> NodeUtils::map_recv_info_{};

const std::set<std::string> kConstOpTypes = {"Const", "Constant"};

const std::set<std::string> kIfOpTypes = {"If", "_If", "StatelessIf"};
const std::set<std::string> kWhileOpTypes = {"While", "_While", "StatelessWhile"};
const std::set<std::string> kCaseOpTypes = {"Case"};
const std::set<std::string> kForOpTypes = {"For"};

bool OpShapeIsUnknown(const OpDescPtr &desc) {
  for (const auto &ptr : desc->GetAllInputsDescPtr()) {
    auto ge_shape = ptr->GetShape();
    for (const auto &dim : ge_shape.GetDims()) {
      if (dim == UNKNOWN_DIM || dim == UNKNOWN_DIM_NUM) {
        return true;
      }
    }
  }
  for (const auto &ptr : desc->GetAllOutputsDescPtr()) {
    auto ge_shape = ptr->GetShape();
    for (const auto &dim : ge_shape.GetDims()) {
      if (dim == UNKNOWN_DIM || dim == UNKNOWN_DIM_NUM) {
        return true;
      }
    }
  }
  return false;
}

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
  bool is_unknown_graph = node_ptr->GetOwnerComputeGraph()->GetGraphUnknownFlag();
  if (is_unknown_graph) {
    return GRAPH_SUCCESS;
  }
  for (const auto &out_anchor : node_ptr->GetAllOutDataAnchors()) {
    auto output_tensor = op_desc->MutableOutputDesc(out_anchor->GetIdx());
    auto out_dims = output_tensor->GetShape().GetDims();
    auto out_dtype = output_tensor->GetDataType();
    ge::TensorUtils::SetRealDimCnt(*output_tensor, static_cast<uint32_t>(output_tensor->GetShape().GetDims().size()));
    output_tensor->SetOriginShape(output_tensor->GetShape());
    output_tensor->SetOriginDataType(output_tensor->GetDataType());

    GELOGD("node name is %s, origin shape is %ld, origin format is %s, origin data type is %s",
           node_ptr->GetName().c_str(), output_tensor->GetOriginShape().GetShapeSize(),
           TypeUtils::FormatToSerialString(output_tensor->GetOriginFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor->GetOriginDataType()).c_str());

    for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (peer_anchor->GetOwnerNode()->GetOpDesc() == nullptr) {
        GELOGE(GRAPH_FAILED, "peer_anchor opdesc is null");
        continue;
      }
      auto peer_input_desc = peer_anchor->GetOwnerNode()->GetOpDesc()->MutableInputDesc(peer_anchor->GetIdx());
      if (peer_input_desc == nullptr) {
        GELOGE(GRAPH_FAILED, "peer_input_desc is nullptr");
        continue;
      }
      // check shape and dtype continuity. do not stop process
      auto peer_input_dims = peer_input_desc->GetShape().GetDims();
      auto peer_input_dtype = peer_input_desc->GetDataType();
      if (out_dtype != peer_input_dtype) {
        GELOGW(
          "current node [%s] [%d]\'th out_dtype is [%s].peer input node [%s] [%d]\'th "
          "input_dtype is [%s].The two dtype should be same! Please check graph and fix it",
          node_ptr->GetName().c_str(), out_anchor->GetIdx(), TypeUtils::DataTypeToSerialString(out_dtype).c_str(),
          peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx(),
          TypeUtils::DataTypeToSerialString(peer_input_dtype).c_str());
      } else if ((!peer_input_dims.empty()) && (out_dims != peer_input_dims)) {
        string out_shape_str, peer_in_shape_str;
        out_shape_str += "[";
        for (int64_t dim : out_dims) {
          out_shape_str += std::to_string(dim) + " ";
        }
        out_shape_str += "]";
        peer_in_shape_str += "[";
        for (int64_t dim : peer_input_dims) {
          peer_in_shape_str += std::to_string(dim) + " ";
        }
        peer_in_shape_str += "]";

        GELOGW(
          "current node [%s] [%d]\'th out_shape is [%s].peer input node [%s] [%d]\'th "
          "input_shape is [%s].The two shape should be same! Please check graph and fix it",
          node_ptr->GetName().c_str(), out_anchor->GetIdx(), out_shape_str.c_str(),
          peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx(), peer_in_shape_str.c_str());
      }
      GELOGI("Peer input opdesc name is %s, need to flush: shape size is %zu, datatype is %d, original datatype is %d",
             peer_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(), output_tensor->GetShape().GetDimNum(),
             output_tensor->GetDataType(), output_tensor->GetOriginDataType());
      peer_input_desc->SetOriginShape(output_tensor->GetOriginShape());
      peer_input_desc->SetShape(output_tensor->GetShape());
      peer_input_desc->SetDataType(output_tensor->GetDataType());
      peer_input_desc->SetOriginDataType(output_tensor->GetOriginDataType());
      std::vector<std::pair<int64_t, int64_t>> shape_range;
      (void)output_tensor->GetShapeRange(shape_range);
      peer_input_desc->SetShapeRange(shape_range);
      ge::TensorUtils::SetRealDimCnt(*peer_input_desc,
                                     static_cast<uint32_t>(output_tensor->GetShape().GetDims().size()));
      GELOGI("Peer input opdesc name is %s, shape size is %zu, datatype is %d, original datatype is %d",
             peer_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(), peer_input_desc->GetShape().GetDimNum(),
             peer_input_desc->GetDataType(), peer_input_desc->GetOriginDataType());
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::AppendInputAnchor(const NodePtr &node,
                                                                                        uint32_t num) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "Input node is null");
    return GRAPH_FAILED;
  }

  GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  const auto &op_desc = node->GetOpDesc();
  for (size_t i = op_desc->GetInputsSize(); i < num; ++i) {
    if (op_desc->AddInputDesc(data_desc) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Add input desc failed");
      return GRAPH_FAILED;
    }

    auto anchor = ComGraphMakeShared<InDataAnchor>(node, i);
    if (anchor == nullptr) {
      GELOGE(OUT_OF_MEMORY, "Current in data anchor is null, make shared_ptr failed.");
      return GRAPH_FAILED;
    }
    node->in_data_anchors_.push_back(anchor);
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::RemoveInputAnchor(const NodePtr &node,
                                                                                        uint32_t num) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "Input node is null");
    return GRAPH_FAILED;
  }

  const auto &op_desc = node->GetOpDesc();
  while (op_desc->GetInputsSize() > num) {
    if (!OpDescUtils::ClearInputDesc(op_desc, num)) {
      return GRAPH_FAILED;
    }
  }

  auto input_names = op_desc->GetAllInputName();
  (void)op_desc->UpdateInputName(input_names);
  auto is_input_const = op_desc->GetIsInputConst();
  is_input_const.resize(num);
  op_desc->SetIsInputConst(is_input_const);

  while (node->in_data_anchors_.size() > num) {
    node->in_data_anchors_.pop_back();
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::AppendOutputAnchor(const NodePtr &node,
                                                                                         uint32_t num) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "Input node is null");
    return GRAPH_FAILED;
  }

  GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  const OpDescPtr &op_desc = node->GetOpDesc();
  for (size_t i = op_desc->GetOutputsSize(); i < num; ++i) {
    if (op_desc->AddOutputDesc(data_desc) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Add output desc failed");
      return GRAPH_FAILED;
    }

    auto anchor = ComGraphMakeShared<OutDataAnchor>(node, i);
    if (anchor == nullptr) {
      GELOGE(OUT_OF_MEMORY, "Current out data anchor is null, make shared_ptr failed.");
      return GRAPH_FAILED;
    }
    node->out_data_anchors_.push_back(anchor);
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::RemoveOutputAnchor(const NodePtr &node,
                                                                                         uint32_t num) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "Input node is null");
    return GRAPH_FAILED;
  }

  const auto &op_desc = node->GetOpDesc();
  auto output_names = op_desc->GetAllOutputName();
  while (op_desc->GetOutputsSize() > num) {
    if (!OpDescUtils::ClearOutputDesc(op_desc, num)) {
      return GRAPH_FAILED;
    }
  }
  (void)op_desc->UpdateOutputName(output_names);

  while (node->out_data_anchors_.size() > num) {
    node->out_data_anchors_.pop_back();
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

graphStatus NodeUtils::GetNodeUnknownShapeStatus(const Node &node, bool &is_unknow) {
  auto desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(desc);
  // check self
  is_unknow = OpShapeIsUnknown(desc);
  if (is_unknow) {
    return GRAPH_SUCCESS;
  }
  auto sub_graph_names = desc->GetSubgraphInstanceNames();
  if (sub_graph_names.empty()) {
    return GRAPH_SUCCESS;
  } else {
    auto owner_graph = node.GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(owner_graph);
    auto root_graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
    if (root_graph == nullptr) {
      GE_LOGE("Node %s gets null root graph", node.GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
    for (auto &sub_graph_name : sub_graph_names) {
      auto sub_graph = root_graph->GetSubgraph(sub_graph_name);
      GE_CHECK_NOTNULL(sub_graph);
      for (const auto &node_ptr : sub_graph->GetDirectNode()) {
        auto status = GetNodeUnknownShapeStatus(*node_ptr, is_unknow);
        if (status != GRAPH_SUCCESS) {
          GE_LOGE("get node unknown shape status failed!");
          return status;
        }
        if (is_unknow) {
          return GRAPH_SUCCESS;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::GetInputConstData(const ConstNodePtr &node_ptr, const string &dst_name, GeTensorPtr &ge_tensor) {
  GE_CHECK_NOTNULL(node_ptr);
  return NodeUtils::GetInputConstData(*node_ptr, dst_name, ge_tensor);
}

graphStatus NodeUtils::GetInputConstData(const Node &node, const string &dst_name, GeTensorPtr &ge_tensor) {
  // For inner compute graph
  auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto index = op_desc->GetInputIndexByName(dst_name);
  auto in_data_anchor = node.GetInDataAnchor(index);
  GE_CHECK_NOTNULL(in_data_anchor);
  auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(out_data_anchor);
  auto peer_node = out_data_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(peer_node);
  auto peer_op_desc = peer_node->GetOpDesc();
  GE_CHECK_NOTNULL(peer_op_desc);
  auto peer_op_type = peer_op_desc->GetType();
  if (peer_op_type == CONSTANTOP || peer_op_type == CONSTANT) {
    if (!AttrUtils::MutableTensor(peer_node->GetOpDesc(), ATTR_NAME_WEIGHTS, ge_tensor)) {
      GELOGW("get attr name %s failed.", ATTR_NAME_WEIGHTS.c_str());
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  } else if (peer_op_type == DATA) {
    auto parent_node = NodeUtils::GetParentInput(peer_node);
    while ((parent_node != nullptr) && (parent_node->GetType() == DATA)) {
      parent_node = NodeUtils::GetParentInput(parent_node);
    }
    if ((parent_node != nullptr) && ((parent_node->GetType() == CONSTANT) || (parent_node->GetType() == CONSTANTOP))) {
      if (!AttrUtils::MutableTensor(parent_node->GetOpDesc(), ATTR_NAME_WEIGHTS, ge_tensor)) {
        GELOGW("get attr name %s failed.", ATTR_NAME_WEIGHTS.c_str());
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }
  }
  // Try get from runtime inference context
  auto session_id = std::to_string(GetContext().SessionId());
  RuntimeInferenceContext *runtime_infer_ctx = nullptr;
  if (RuntimeInferenceContext::GetContext(session_id, &runtime_infer_ctx) == GRAPH_SUCCESS) {
    GELOGD("To get constant from runtime inference context. session_id = %s", session_id.c_str());
    auto ret = runtime_infer_ctx->GetTensor(peer_node->GetOpDesc()->GetId(), out_data_anchor->GetIdx(), ge_tensor);
    if (ret == GRAPH_SUCCESS) {
      return GRAPH_SUCCESS;
    }
  }
  GELOGW("node[%s]'s input[%s]'s peer node is not const", node.GetName().c_str(), dst_name.c_str());
  return GRAPH_FAILED;
}

std::string NodeUtils::GetNodeType(const Node &node) {
  if (node.GetType() != FRAMEWORKOP) {
    return node.GetType();
  }

  std::string type;
  (void)AttrUtils::GetStr(node.GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
  return type;
}

std::string NodeUtils::GetNodeType(const NodePtr &node) { return node == nullptr ? "" : GetNodeType(*node); }

ComputeGraphPtr NodeUtils::GetSubgraph(const Node &node, uint32_t index) {
  auto op_desc = node.GetOpDesc();
  if (op_desc == nullptr) {
    return nullptr;
  }
  auto root_graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    return nullptr;
  }
  return root_graph->GetSubgraph(op_desc->GetSubgraphInstanceName(index));
}

graphStatus NodeUtils::SetSubgraph(Node &node, uint32_t index, const ComputeGraphPtr &subgraph) {
  if (subgraph == nullptr) {
    GE_LOGE("Failed to set subgraph to node %s index %u, null subgraph", node.GetName().c_str(), index);
    return GRAPH_PARAM_INVALID;
  }
  auto op_desc = node.GetOpDesc();
  if (op_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  auto root_graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    GE_LOGE("Failed to add subgraph to node %s, null root graph", node.GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  auto ret = op_desc->SetSubgraphInstanceName(index, subgraph->GetName());
  if (ret != GRAPH_SUCCESS) {
    GE_LOGE("Failed to set subgraph to node %s index %u", node.GetName().c_str(), index);
    return ret;
  }
  subgraph->SetParentNode(node.shared_from_this());
  subgraph->SetParentGraph(node.GetOwnerComputeGraph());
  return root_graph->AddSubgraph(subgraph);
}

///
/// Check if node is input of subgraph
/// @param [in] node
/// @return bool
///
bool NodeUtils::IsSubgraphInput(const NodePtr &node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr) ||
      (node->GetOwnerComputeGraph()->GetParentNode() == nullptr)) {
    return false;
  }

  auto parent_op_desc = node->GetOwnerComputeGraph()->GetParentNode()->GetOpDesc();
  if (parent_op_desc == nullptr) {
    return false;
  }

  // dynamic shape unknown graph false
  // dynamic shape known graph with functional subgraph maybe true
  if (AttrUtils::HasAttr(parent_op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE)) {
    if (node->GetOwnerComputeGraph()->GetParentGraph()->GetGraphUnknownFlag()) {
      return false;
    } else {
      if (node->GetOwnerComputeGraph()->GetParentNode()->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
        return false;
      }
    }
  }

  return node->GetOpDesc()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX);
}

///
/// Check if node is output of subgraph
/// @param [in] node
/// @return bool
///
bool NodeUtils::IsSubgraphOutput(const NodePtr &node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr) ||
      (node->GetOwnerComputeGraph()->GetParentNode() == nullptr) || (node->GetType() != NETOUTPUT)) {
    return false;
  }

  auto parent_op_desc = node->GetOwnerComputeGraph()->GetParentNode()->GetOpDesc();
  if (parent_op_desc == nullptr) {
    return false;
  }

  if (AttrUtils::HasAttr(parent_op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE)) {
    if (node->GetOwnerComputeGraph()->GetParentGraph()->GetGraphUnknownFlag()) {
      return false;
    } else {
      if (node->GetOwnerComputeGraph()->GetParentNode()->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
        return false;
      }
    }
  }

  for (GeTensorDesc &tensor : node->GetOpDesc()->GetAllInputsDesc()) {
    if (AttrUtils::HasAttr(tensor, ATTR_NAME_PARENT_NODE_INDEX)) {
      return true;
    }
  }

  return false;
}

///
/// @brief Get subgraph original input node.
/// @param [in] node
/// @return Node
///
NodePtr NodeUtils::GetParentInput(const Node &node) {
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(node.GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return nullptr;
  }

  // Subgraph Data Node, check for constant input.
  const ComputeGraphPtr &graph = node.GetOwnerComputeGraph();
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  const NodePtr &parent_node = graph->GetParentNode();
  GE_CHECK_NOTNULL_EXEC(parent_node, return nullptr);

  const InDataAnchorPtr &in_anchor = parent_node->GetInDataAnchor(parent_index);
  GE_CHECK_NOTNULL_EXEC(in_anchor, return nullptr);

  const OutDataAnchorPtr &peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL_EXEC(peer_out_anchor, return nullptr);

  return peer_out_anchor->GetOwnerNode();
}

NodePtr NodeUtils::GetParentInput(const NodePtr &node) { return node == nullptr ? node : GetParentInput(*node); }

///
/// @brief Get is dynamic shape graph from node.
/// @param [in] node
/// @return bool
///
bool NodeUtils::IsDynamicShape(const Node &node) {
  const auto graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (graph == nullptr) {
    return false;
  }

  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  return is_dynamic_shape;
}

bool NodeUtils::IsDynamicShape(const NodePtr &node) { return node == nullptr ? false : IsDynamicShape(*node); }

///
/// @brief Check is varying_input for while node
/// @param [in] node: Data node for subgraph
/// @return bool
///
bool NodeUtils::IsWhileVaryingInput(const ge::NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (node->GetType() != DATA) {
    return false;  // not input_node for subgraph
  }

  const NodePtr &parent_node = node->GetOwnerComputeGraph()->GetParentNode();
  if (parent_node == nullptr) {
    return false;  // root graph
  }

  if (kWhileOpTypes.count(parent_node->GetType()) == 0) {
    return false;  // not input_node for while subgraph
  }

  uint32_t index_i = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, index_i)) {
    GELOGW("Node %s has no attr PARENT_NODE_INDEX.", node->GetName().c_str());
    return false;
  }
  bool varying_flag = true;
  for (const auto &item : node->GetOutDataNodesAndAnchors()) {
    if (item.first->GetType() != NETOUTPUT) {
      continue;
    }
    OpDescPtr op_desc = item.first->GetOpDesc();
    uint32_t index_o = 0;
    if ((op_desc == nullptr) ||
        !AttrUtils::GetInt(op_desc->GetInputDesc(item.second->GetIdx()), ATTR_NAME_PARENT_NODE_INDEX, index_o)) {
      continue;  // input for while-cond subgraph
    }
    if (index_i != index_o) {
      continue;  // varying input for while-body subgraph
    }
    varying_flag = false;
    break;
  }
  return varying_flag;
}

///
/// @brief Get subgraph input is constant.
/// @param [in] node
/// @param [out] string
/// @return bool
///
bool NodeUtils::GetConstOpType(const NodePtr &node, std::string &type) {
  if (node == nullptr) {
    return false;
  }

  if ((node->GetType() == CONSTANT) || (node->GetType() == CONSTANTOP)) {
    type = node->GetType();
    return true;
  }

  if (node->GetType() != DATA) {
    return false;  // not subgraph input node
  }

  const auto &parent = GetParentInput(node);
  return GetConstOpType(parent, type);
}

///
/// @brief Remove node-related subgraphs, including subgraphs of nodes in the subgraph.
/// @param [in] node
/// @return return GRAPH_SUCCESS if remove successfully, other for failed.
///
Status NodeUtils::RemoveSubgraphsOnNode(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    return GRAPH_SUCCESS;
  } else {
    auto owner_graph = node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(owner_graph);
    auto root_graph = GraphUtils::FindRootGraph(owner_graph);
    GE_CHECK_NOTNULL(root_graph);

    std::unordered_set<std::string> subgraph_to_remove;
    for (auto &subgraph_name : subgraph_names) {
      std::deque<std::string> queue;
      queue.push_back(subgraph_name);
      subgraph_to_remove.insert(subgraph_name);
      op_desc->RemoveSubgraphInstanceName(subgraph_name);
      while (!queue.empty()) {
        auto graph_name = queue.front();
        queue.pop_front();

        auto subgraph = root_graph->GetSubgraph(graph_name);
        GE_CHECK_NOTNULL(subgraph);
        for (const auto &sub_node : subgraph->GetDirectNode()) {
          auto sub_op_desc = sub_node->GetOpDesc();
          GE_CHECK_NOTNULL(sub_op_desc);
          auto sub_names = sub_op_desc->GetSubgraphInstanceNames();
          // Subgraph and all nodes in it will be removed later,
          // no need to remove 'SubgraphInstanceName' in op desc here.
          for (auto &name : sub_names) {
            if (subgraph_to_remove.insert(name).second) {
              queue.push_back(name);
            }
          }
        }
      }
    }
    // Remove subgraph from root_graph
    for (const auto &name : subgraph_to_remove) {
      GELOGI("Remove subgraph:%s.", name.c_str());
      root_graph->RemoveSubgraph(name);
    }
  }

  return GRAPH_SUCCESS;
}
///
/// @brief Get subgraph input data node by index.
/// @param [in] node
/// @return Node
///
vector<NodePtr> NodeUtils::GetSubgraphDataNodesByIndex(const Node &node, int index) {
  vector<NodePtr> in_data_node_vec;
  auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return in_data_node_vec);
  auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    GELOGW("Node %s is single node without sub graph.", node.GetName().c_str());
    return in_data_node_vec;
  }
  auto compute_graph = node.GetOwnerComputeGraph();
  for (const std::string &instance_name : subgraph_names) {
    auto subgraph = compute_graph->GetSubgraph(instance_name);
    for (const auto &node_in_subgraph : subgraph->GetDirectNode()) {
      int parent_index = -1;
      if (NodeUtils::IsSubgraphInput(node_in_subgraph)) {
        (void)AttrUtils::GetInt(node_in_subgraph->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index);
        if (parent_index == index) {
          in_data_node_vec.emplace_back(node_in_subgraph);
        }
      }
    }
  }
  return in_data_node_vec;
}
///
/// @brief Get subgraph input data node by index.
/// @param [in] node
/// @return Node
///
vector<NodePtr> NodeUtils::GetSubgraphOutputNodes(const Node &node) {
  vector<NodePtr> out_data_node_vec;
  auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return out_data_node_vec);
  auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    GELOGI("Node %s is single node without sub graph.", node.GetName().c_str());
    return out_data_node_vec;
  }
  auto compute_graph = node.GetOwnerComputeGraph();
  for (const std::string &instance_name : subgraph_names) {
    auto subgraph = compute_graph->GetSubgraph(instance_name);
    for (const auto &node_in_subgraph : subgraph->GetDirectNode()) {
      if (NodeUtils::IsSubgraphOutput(node_in_subgraph)) {
        out_data_node_vec.emplace_back(node_in_subgraph);
      }
    }
  }
  return out_data_node_vec;
}

NodePtr NodeUtils::GetInDataNodeByIndex(const Node &node, const int index) {
  if (node.GetInDataAnchor(index) == nullptr) {
    return nullptr;
  }
  if (node.GetInDataAnchor(index)->GetPeerOutAnchor() == nullptr) {
    return nullptr;
  }
  return node.GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
}

vector<pair<InDataAnchorPtr, NodePtr>> NodeUtils::GetOutDataNodesWithAnchorByIndex(const Node &node, const int index) {
  vector<pair<InDataAnchorPtr, NodePtr>> out_data_nodes;
  auto out_data_anchor = node.GetOutDataAnchor(index);
  if (out_data_anchor == nullptr) {
    return out_data_nodes;
  }

  for (const auto peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    if (peer_in_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    out_data_nodes.emplace_back(std::make_pair(peer_in_anchor, peer_in_anchor->GetOwnerNode()));
  }
  return out_data_nodes;
}

ConstNodePtr NodeUtils::GetNodeFromOperator(const Operator &oprt) { return oprt.GetNode(); }
}  // namespace ge
