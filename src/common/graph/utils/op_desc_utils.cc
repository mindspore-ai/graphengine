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

#include "utils/op_desc_utils.h"
#include <algorithm>
#include "debug/ge_attr_define.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "utils/graph_utils.h"
#include "utils/node_utils.h"

using std::vector;

/*lint -e512 -e737 -e752*/
namespace ge {
const char OP_DESC_QUANT_PARAMS[] = "quantize_factor";
static const int CONST_OP_NORMAL_WEIGHT_SIZE = 1;

bool OpDescUtils::ClearInputDesc(const NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, return false, "node is nullptr");
  GE_CHK_BOOL_EXEC(node->GetOpDesc() != nullptr, return false, "opdesc is nullptr");
  vector<int> index_list;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      index_list.push_back(in_anchor->GetIdx());
    }
  }
  std::sort(index_list.begin(), index_list.end());
  // Node's in anchor index need shrink
  for (size_t i = 0; i < index_list.size(); ++i) {
    auto iter = node->GetOpDesc()->inputs_desc_.begin() + index_list[i];
    if (iter < node->GetOpDesc()->inputs_desc_.end()) {
      (void)node->GetOpDesc()->inputs_desc_.erase(iter);
    } else {
      GELOGW("inputs_desc_ iterator out of range.");
    }
  }

  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::ClearInputDesc(OpDescPtr op_desc,
                                                                                const uint32_t index) {
  GE_CHK_BOOL_EXEC(op_desc != nullptr, return false, "op_desc is nullptr");
  GE_CHK_BOOL_EXEC(index < op_desc->inputs_desc_.size(), return false, "index %u is invalid.", index);

  auto iter = op_desc->inputs_desc_.begin() + index;
  if (iter < op_desc->inputs_desc_.end()) {
    (void)op_desc->inputs_desc_.erase(iter);
  } else {
    GELOGW("inputs_desc_ iterator out of range.");
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::HasQuantizeFactorParams(const OpDescPtr &op_desc) {
  GE_CHK_BOOL_EXEC_INFO(op_desc != nullptr, return false, "op_desc is nullptr");
  return op_desc->HasAttr(OP_DESC_QUANT_PARAMS);
}

bool OpDescUtils::ClearOutputDesc(const NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, return false, "node is nullptr");
  GE_CHK_BOOL_EXEC(node->GetOpDesc() != nullptr, return false, "opdesc is nullptr");
  vector<int> index_list;
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor->GetPeerInDataAnchors().empty()) {
      index_list.push_back(out_anchor->GetIdx());
    }
  }
  std::sort(index_list.begin(), index_list.end());
  // Node's out anchor index need shrink
  for (size_t i = 0; i < index_list.size(); ++i) {
    auto iter = node->GetOpDesc()->outputs_desc_.begin() + index_list[i];
    if (iter < node->GetOpDesc()->outputs_desc_.end()) {
      (void)node->GetOpDesc()->outputs_desc_.erase(iter);
    } else {
      GELOGW("outputs_desc_ iterator out of range.");
    }
  }

  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::ClearOutputDesc(const OpDescPtr &op_desc,
                                                                                 uint32_t index) {
  GE_CHK_BOOL_EXEC(op_desc != nullptr, return false, "op_desc is nullptr");
  GE_CHK_BOOL_EXEC(index < op_desc->outputs_desc_.size(), return false, "index %u is invalid.", index);

  auto iter = op_desc->outputs_desc_.begin() + index;
  if (iter < op_desc->outputs_desc_.end()) {
    (void)op_desc->outputs_desc_.erase(iter);
  } else {
    GELOGW("outputs_desc_ iterator out of range.");
  }
  return true;
}

bool OpDescUtils::HasQuantizeFactorParams(const OpDesc &op_desc) { return op_desc.HasAttr(OP_DESC_QUANT_PARAMS); }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::GetQuantizeFactorParams(const OpDescPtr &op_desc, QuantizeFactorParams &quant) {
  GE_CHK_BOOL_EXEC_INFO(op_desc != nullptr, return GRAPH_FAILED, "op_desc is nullptr");
  GeAttrValue attr_value;
  GE_CHK_BOOL_EXEC_INFO(op_desc->GetAttr(OP_DESC_QUANT_PARAMS, attr_value) == GRAPH_SUCCESS, return GRAPH_FAILED,
                        "GetQuantizeFactorParams failed");
  return attr_value.GetValue<QuantizeFactorParams>(quant);
}

graphStatus OpDescUtils::GetQuantizeFactorParams(const OpDesc &op_desc, QuantizeFactorParams &quant) {
  GeAttrValue attr_value;
  GE_CHK_BOOL_EXEC_INFO(op_desc.GetAttr(OP_DESC_QUANT_PARAMS, attr_value) == GRAPH_SUCCESS, return GRAPH_FAILED,
                        "GetQuantizeFactorParams failed");
  return attr_value.GetValue<QuantizeFactorParams>(quant);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetQuantizeFactorParams(const OpDescPtr &op_desc, const QuantizeFactorParams &quant) {
  GE_CHK_BOOL_EXEC_INFO(op_desc != nullptr, return GRAPH_FAILED, "op_desc is nullptr");
  return op_desc->SetAttr(OP_DESC_QUANT_PARAMS, GeAttrValue::CreateFrom<QuantizeFactorParams>(quant));  // lint !e732
}

graphStatus OpDescUtils::SetQuantizeFactorParams(OpDesc &op_desc, const QuantizeFactorParams &quant) {
  return op_desc.SetAttr(OP_DESC_QUANT_PARAMS, GeAttrValue::CreateFrom<QuantizeFactorParams>(quant));  // lint !e732
}

GeTensorPtr OpDescUtils::MutableWeights(OpDesc &op_desc) {
  GeTensorPtr weight = nullptr;
  if (!AttrUtils::MutableTensor(&op_desc, ATTR_NAME_WEIGHTS, weight)) {
    GELOGW("MutableTensor error");
  }

  return weight;
}

GE_FUNC_HOST_VISIBILITY GeTensorPtr OpDescUtils::MutableWeights(OpDescPtr op_desc) {
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "op_desc is null");
    return nullptr;
  }
  return MutableWeights(*op_desc);
}

graphStatus OpDescUtils::SetWeights(OpDesc &op_desc, const GeTensorPtr weight) {
  if (weight == nullptr) {
    GELOGE(GRAPH_FAILED, "weight is null");
    return GRAPH_FAILED;
  }
  return AttrUtils::SetTensor(&op_desc, ATTR_NAME_WEIGHTS, weight) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus OpDescUtils::SetWeights(OpDescPtr op_desc, const GeTensorPtr weight) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(weight);
  return SetWeights(*op_desc, weight);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<ConstGeTensorPtr> OpDescUtils::GetWeights(const ge::Node &node) {
  auto weights = MutableWeights(node);
  vector<ConstGeTensorPtr> ret(weights.size());
  std::copy(weights.begin(), weights.end(), ret.begin());
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<ConstGeTensorPtr> OpDescUtils::GetWeights(
  const ge::ConstNodePtr &node) {
  if (node == nullptr) {
    return vector<ge::ConstGeTensorPtr>();
  }
  return GetWeights(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<ge::NodePtr> OpDescUtils::GetConstInputNode(
  const ge::Node &node) {
  vector<ge::NodePtr> ret;
  auto in_anchors = node.GetAllInDataAnchors();
  for (const auto &in_anchor : in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      // normally out_anchor could be null, this is ok
      GELOGD("node %s' peer_out_anchor is null", node.GetName().c_str());
      continue;
    }
    auto in_node = out_anchor->GetOwnerNode();
    while (true) {
      if (in_node == nullptr) {
        break;
      }
      if ((in_node->GetType() == CONSTANT) || (in_node->GetType() == CONSTANTOP)) {
        ret.push_back(in_node);
        break;
      } else if (in_node->GetType() == DATA) {
        if (NodeUtils::IsWhileVaryingInput(in_node)) {
          break;
        }
        in_node = NodeUtils::GetParentInput(in_node);
      } else if ((in_node->GetType() == ENTER) || (in_node->GetType() == REFENTER)) {
        bool is_constant = false;
        (void)AttrUtils::GetBool(in_node->GetOpDesc(), ENTER_ATTR_CONSTANT_FLAG, is_constant);
        if (!is_constant) {
          break;
        }
        // Enter node has and only has one input
        if (in_node->GetInDataNodes().size() != 1) {
          GELOGW("Check number of input_nodes for Enter node %s failed, size=%zu.", node.GetName().c_str(),
                 in_node->GetInDataNodes().size());
          break;
        }
        in_node = in_node->GetInDataNodes().at(0);
      } else {
        break;
      }
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<ConstGeTensorPtr> OpDescUtils::GetInputData(
  const vector<ge::NodePtr> &input_nodes) {
  vector<ConstGeTensorPtr> ret;

  for (const auto &input_node : input_nodes) {
    auto temp_weight = MutableWeights(input_node->GetOpDesc());
    if (temp_weight == nullptr) {
      GELOGE(GRAPH_FAILED, "const op's weight is null, name: %s", input_node->GetName().c_str());
      return vector<ConstGeTensorPtr>();
    }
    ret.push_back(temp_weight);
  }

  return ret;
}
size_t OpDescUtils::GetNonConstInputsSize(const ge::Node &node) {
  if (NodeUtils::IsAnchorStatusSet(node)) {
    size_t input_num = 0;
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(anchor) == ANCHOR_DATA) {
        input_num++;
        continue;
      }
    }
    return input_num;  // lint !e712
  } else {
    GE_IF_BOOL_EXEC(
      node.GetInDataNodes().size() < GetConstInputs(node).size(),
      GELOGE(GRAPH_FAILED, "%zu is smaller than %zu", node.GetInDataNodes().size(), GetConstInputs(node).size());
      return 0);
    return node.GetInDataNodes().size() - GetConstInputs(node).size();
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDescUtils::GetNonConstInputsSize(const ge::ConstNodePtr node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "Node is nullptr");
    return 0;
  }
  return GetNonConstInputsSize(*node);
}

GeTensorDesc OpDescUtils::GetNonConstInputTensorDesc(const ge::Node &node, size_t index_non_const) {
  GE_CHK_BOOL_EXEC(node.GetOpDesc() != nullptr, return GeTensorDesc(), "node.GetOpDesc() is nullptr!");
  size_t i = 0;
  if (NodeUtils::IsAnchorStatusSet(node)) {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(anchor) == ANCHOR_DATA) {
        if (index_non_const == i) {
          return node.GetOpDesc()->GetInputDesc(static_cast<uint32_t>(anchor->GetIdx()));
        }
        ++i;
      }
    }
  } else {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      auto peer_anchor = anchor->GetPeerOutAnchor();
      if (peer_anchor == nullptr) {
        continue;
      }
      auto owner_node = peer_anchor->GetOwnerNode();
      if (owner_node == nullptr) {
        continue;
      }
      if (owner_node->GetType() == CONSTANT) {
        continue;
      }
      if (index_non_const == i) {
        return node.GetOpDesc()->GetInputDesc(anchor->GetIdx());
      }
      ++i;
    }
  }
  return GeTensorDesc();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDesc
OpDescUtils::GetNonConstInputTensorDesc(const ge::ConstNodePtr &node, size_t index_non_const) {
  CHECK_FALSE_EXEC(node != nullptr, return GeTensorDesc());
  return GetNonConstInputTensorDesc(*node, index_non_const);
}

bool OpDescUtils::GetNonConstInputIndex(const ge::Node &node, const size_t index_non_const, size_t &index) {
  bool ret = false;
  size_t i = 0;
  if (NodeUtils::IsAnchorStatusSet(node)) {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(anchor) == ANCHOR_DATA) {
        if (index_non_const == i) {
          index = static_cast<size_t>(anchor->GetIdx());
          ret = true;
        }
        ++i;
      }
    }
  } else {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      auto peer_anchor = anchor->GetPeerOutAnchor();
      if (peer_anchor == nullptr) {
        continue;
      }
      auto owner_node = peer_anchor->GetOwnerNode();
      if (owner_node == nullptr) {
        continue;
      }
      if (owner_node->GetType() == CONSTANT) {
        continue;
      }
      if (index_non_const == i) {
        index = static_cast<size_t>(anchor->GetIdx());
        ret = true;
      }
      ++i;
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::GetNonConstInputIndex(const ge::ConstNodePtr &node,
                                                                                       size_t index_non_const,
                                                                                       size_t &index) {
  CHECK_FALSE_EXEC(node != nullptr, return false);
  return GetNonConstInputIndex(*node, index_non_const, index);
}

bool OpDescUtils::IsNonConstInput(const ge::Node &node, const size_t index) {
  bool ret = false;
  if (index < node.GetAllInDataAnchors().size()) {
    if (NodeUtils::IsAnchorStatusSet(node)) {
      ret = (ge::AnchorUtils::GetStatus(node.GetInDataAnchor(static_cast<int>(index))) == ANCHOR_DATA);  // lint !e712
    } else {
      for (const auto &anchor : node.GetAllInDataAnchors()) {
        if (anchor->GetIdx() != static_cast<int>(index)) {
          continue;
        }
        auto peer_anchor = anchor->GetPeerOutAnchor();
        if (peer_anchor == nullptr) {
          break;
        }
        auto owner_node = peer_anchor->GetOwnerNode();
        if (owner_node == nullptr) {
          break;
        }
        ret = (owner_node->GetType() != CONSTANT);
      }
    }
  }

  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::IsNonConstInput(const ge::ConstNodePtr &node,
                                                                                 size_t index) {
  CHECK_FALSE_EXEC(node != nullptr, return false);
  return IsNonConstInput(*node, index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<ge::NodePtr> OpDescUtils::GetConstInputs(
  const ge::ConstNodePtr &node) {
  if (node == nullptr) {
    return vector<ge::NodePtr>();
  }
  return GetConstInputs(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<ge::GeTensorDesc> OpDescUtils::GetNonConstTensorDesc(
  const ge::ConstNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return vector<ge::GeTensorDesc>();
  }
  vector<ge::GeTensorDesc> ret;
  if (NodeUtils::IsAnchorStatusSet(*node)) {
    for (const auto &in_anchor : node->GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(in_anchor) == ANCHOR_DATA) {
        ret.push_back(node->GetOpDesc()->GetInputDesc(in_anchor->GetIdx()));
      }
    }
  } else {
    for (const auto &in_anchor : node->GetAllInDataAnchors()) {
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor == nullptr || out_anchor->GetOwnerNode()->GetOpDesc() == nullptr) {
        continue;
      }
      if (out_anchor->GetOwnerNode()->GetOpDesc()->GetType() != CONSTANT) {
        ret.push_back(node->GetOpDesc()->GetInputDesc(in_anchor->GetIdx()));
      }
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<ge::NodePtr> OpDescUtils::GetConstInputs(const ge::Node &node) {
  vector<ge::NodePtr> ret;
  auto in_anchors = node.GetAllInDataAnchors();
  for (const auto &in_anchor : in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) continue;

    auto in_node = out_anchor->GetOwnerNode();
    if (in_node->GetType() == CONSTANT) {
      ret.push_back(in_node);
    } else if (in_node->GetType() == SWITCH && node.GetType() == MATMUL) {
      // const --> switch --> matmul
      auto switch_input = GetConstInputs(*in_node);
      if (switch_input.size() > 0) {
        ret.insert(ret.end(), switch_input.begin(), switch_input.end());
      }
    } else if (in_node->GetType() == DATA) {
      auto parent = NodeUtils::GetParentInput(in_node);
      if ((parent != nullptr) && (parent->GetType() == CONSTANT)) {
        ret.push_back(parent);
      }
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<GeTensorPtr> OpDescUtils::MutableWeights(const ge::Node &node) {
  vector<GeTensorPtr> ret;
  auto op_desc = node.GetOpDesc();
  GE_CHK_BOOL_EXEC(op_desc != nullptr, return ret, "op_desc is nullptr!");
  // Place holder operator, try to get the weight from parent node
  // when parent node is const operator
  if (node.GetType() == PLACEHOLDER) {
    std::string parent_op;
    (void)AttrUtils::GetStr(op_desc, "parentOpType", parent_op);
    // This if judgment is necessary because the current subgraph optimization is multithreaded
    // and the parent node of the PLD operation should be a stable type, such as const
    if (parent_op == CONSTANT || parent_op == CONSTANTOP) {
      NodePtr parent_node = nullptr;
      parent_node = op_desc->TryGetExtAttr("parentNode", parent_node);
      if (parent_node != nullptr) {
        op_desc = parent_node->GetOpDesc();
        GELOGD("pld[%s] get weight from const[%s]", node.GetName().c_str(), op_desc->GetName().c_str());
      }
    }
  }
  // Const operator, take the weight directly
  if (op_desc->GetType() == CONSTANT || (op_desc->GetType() == CONSTANTOP)) {
    auto weight = MutableWeights(op_desc);
    if (weight == nullptr) {
      GELOGI("const op has no weight, op name:%s", node.GetName().c_str());
      return ret;
    }
    ret.push_back(weight);
    return ret;
  }

  if (node.GetType() == DATA) {
    auto parent = NodeUtils::GetParentInput(node);
    if ((parent != nullptr) && NodeUtils::IsConst(*parent)) {
      auto weight = MutableWeights(parent->GetOpDesc());
      if (weight == nullptr) {
        GELOGI("const op has no weight, op name:%s", parent->GetName().c_str());
        return ret;
      }
      ret.push_back(weight);
    }
    return ret;
  }

  // Other operators, get weights from connected constop
  auto input_nodes = GetConstInputs(node);
  for (const auto &input_node : input_nodes) {
    auto temp_weight = MutableWeights(input_node->GetOpDesc());
    if (temp_weight == nullptr) {
      GELOGE(GRAPH_FAILED, "const op's weight is null, name: %s", input_node->GetName().c_str());
      return vector<GeTensorPtr>();
    }
    ret.push_back(temp_weight);
  }

  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<GeTensorPtr> OpDescUtils::MutableWeights(const ge::NodePtr node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "Node is nullptr");
    return vector<ge::GeTensorPtr>();
  }
  return MutableWeights(*node);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(ge::Node &node, const vector<ge::GeTensorPtr> &weights) {
  GE_CHK_BOOL_EXEC(node.GetOpDesc() != nullptr, return GRAPH_PARAM_INVALID, "node.GetOpDesc is nullptr!");
  if (node.GetOpDesc()->GetType() == CONSTANT) {
    if (weights.size() == CONST_OP_NORMAL_WEIGHT_SIZE) {
      return SetWeights(node.GetOpDesc(), weights[0]);
    }
    GELOGI("const op weight size %zu should be 1", weights.size());
    return GRAPH_PARAM_INVALID;
  }

  auto input_nodes = GetConstInputs(node);
  if (weights.size() < input_nodes.size()) {
    GELOGE(GRAPH_FAILED, "weights count can't be less than const input count");
    return GRAPH_PARAM_INVALID;
  }

  ge::GeAttrValue::NAMED_ATTRS named_attrs;
  (void)ge::AttrUtils::SetListTensor(named_attrs, "key", weights);
  vector<ge::GeTensorPtr> copy_weights;
  (void)ge::AttrUtils::MutableListTensor(named_attrs, "key", copy_weights);

  for (size_t i = 0; i < input_nodes.size(); ++i) {
    if (input_nodes[i]->GetOpDesc() != nullptr) {
      SetWeights(input_nodes[i]->GetOpDesc(), copy_weights[i]);
    }
  }

  // If set more weights than constop, need to add constop
  for (size_t i = input_nodes.size(); i < copy_weights.size(); ++i) {
    // Use org weight before SetWeights Overwrite
    auto const_opdesc = CreateConstOp(copy_weights[i]);
    GE_CHECK_NOTNULL(const_opdesc);

    auto owner_graph = node.GetOwnerComputeGraph();
    if (owner_graph == nullptr) {
      GELOGE(GRAPH_FAILED, "node's graph is empty, name: %s", node.GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
    auto const_node = owner_graph->AddNodeFront(const_opdesc);
    GE_CHK_BOOL_EXEC(node.AddLinkFrom(const_node) == GRAPH_SUCCESS, return GRAPH_FAILED, "graph add link failed！");
    std::vector<ge::NodePtr> original_nodes;
    ge::GraphUtils::RecordOriginalNames(original_nodes, const_node);
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(ge::Node &node, const map<int, ge::GeTensorPtr> &weights_map) {
  GE_CHECK_NOTNULL(node.GetOpDesc());
  // 1. node is const
  if (node.GetOpDesc()->GetType() == CONSTANT) {
    if (weights_map.size() == CONST_OP_NORMAL_WEIGHT_SIZE) {
      return SetWeights(node.GetOpDesc(), weights_map.begin()->second);
    }
    GELOGE(GRAPH_PARAM_INVALID, "const op %s weight size %zu should be 1", node.GetName().c_str(), weights_map.size());
    return GRAPH_PARAM_INVALID;
  }
  // 2. node is not const
  for (const auto &pair : weights_map) {
    auto in_data_anchor = node.GetInDataAnchor(pair.first);
    if (in_data_anchor != nullptr && in_data_anchor->GetPeerOutAnchor() != nullptr) {
      // a. update const input node
      auto out_anchor = in_data_anchor->GetPeerOutAnchor();
      auto peer_node = out_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        GELOGE(GRAPH_PARAM_INVALID, "op %s [%d]'s input node is null", node.GetName().c_str(), pair.first);
        return GRAPH_PARAM_INVALID;
      }
      if (peer_node->GetType() != CONSTANT) {
        GELOGE(GRAPH_PARAM_INVALID, " op %s [%d]'s input node should be const, but is %s type:%s ",
               node.GetName().c_str(), pair.first, peer_node->GetName().c_str(), peer_node->GetType().c_str());
      }
      SetWeights(peer_node->GetOpDesc(), pair.second);
    } else {
      // b. create new const input node
      auto const_opdesc = CreateConstOp(pair.second);
      GE_CHECK_NOTNULL(const_opdesc);
      auto owner_graph = node.GetOwnerComputeGraph();
      if (owner_graph == nullptr) {
        GELOGE(GRAPH_PARAM_INVALID, "node's graph is empty, name: %s", node.GetName().c_str());
        return GRAPH_PARAM_INVALID;
      }
      auto const_node = owner_graph->AddNodeFront(const_opdesc);
      if (node.AddLinkFrom(static_cast<uint32_t>(pair.first), const_node) != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "op %s add const to input index[%d] failed", node.GetName().c_str(), pair.first);
        return GRAPH_FAILED;
      }
    }
  }
  NodeUtils::UpdateIsInputConst(node);
  return GRAPH_SUCCESS;
}

OpDescPtr OpDescUtils::CreateConstOp(const GeTensorPtr &tensor_ptr) {
  GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, return nullptr, "tensor_ptr is nullptr!");
  shared_ptr<OpDesc> const_opdesc = ComGraphMakeShared<OpDesc>();
  if (const_opdesc == nullptr) {
    GELOGE(GRAPH_FAILED, "failed to make_shared ");
    return nullptr;
  }

  CHECK_FALSE_EXEC(SetWeights(const_opdesc, tensor_ptr) == ge::GRAPH_SUCCESS, return nullptr);

  const_opdesc->SetType(CONSTANT);

  thread_local int64_t const_count = 0;
  const_opdesc->SetName("dynamic_const_" + std::to_string(GetTid()) + "_" + std::to_string(const_count));
  GELOGI("add const op: %s", const_opdesc->GetName().c_str());
  ++const_count;

  (void)const_opdesc->AddOutputDesc(tensor_ptr->GetTensorDesc());

  GELOGI("after add const op: %s", const_opdesc->GetName().c_str());

  return const_opdesc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::AddConstOpToAnchor(InDataAnchorPtr in_anchor, const GeTensorPtr &tensor_ptr) {
  GE_CHECK_NOTNULL(in_anchor);
  GE_CHECK_NOTNULL(tensor_ptr);
  auto const_opdesc = CreateConstOp(tensor_ptr);
  GE_CHECK_NOTNULL(const_opdesc);
  auto in_node = in_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(in_node);
  auto owner_graph = in_node->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "node's graph is empty, name: %s", in_node->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  auto const_node = in_node->GetOwnerComputeGraph()->AddNodeFront(const_opdesc);
  GE_CHECK_NOTNULL(const_node);
  if (GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), in_anchor) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "Addedge const to node failed.");
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(ge::NodePtr node, const vector<ge::GeTensorPtr> &weights) {
  GE_CHECK_NOTNULL(node);
  return SetWeights(*node, weights);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDescUtils::ClearWeights(const ge::NodePtr node) {
  GE_CHECK_NOTNULL(node);
  auto const_ops = GetConstInputs(node);
  auto graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    GELOGE(GRAPH_FAILED, "Graph is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  for (const auto &const_op : const_ops) {
    GE_CHK_STATUS_RET(GraphUtils::IsolateNode(const_op, {}), "Isolate removed node: %s, type: %s failed",
                      const_op->GetName().c_str(), const_op->GetType().c_str());
    GE_CHK_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(graph, const_op),
                      "Remove node: %s, type: %s without relink failed", const_op->GetName().c_str(),
                      const_op->GetType().c_str());
  }
  return GRAPH_SUCCESS;
}

///
/// @brief Add input
/// @param [in] name
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddInput(const std::string &name) {
  inputs_.emplace_back(std::make_pair(name, GeTensorDesc()));
  return *this;
}

///
/// @brief Add input
/// @param [in] name
/// @param [in] tensor
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddInput(const std::string &name,
                                                                                      const GeTensorDesc &tensor) {
  inputs_.emplace_back(std::make_pair(name, tensor));
  return *this;
}

///
/// @brief Add dynamic input
/// @param [in] name
/// @param [in] num
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddDynamicInput(const std::string &name,
                                                                                             uint32_t num) {
  for (uint32_t i = 0; i < num; i++) {
    inputs_.emplace_back(std::make_pair(name + std::to_string(i), GeTensorDesc()));
  }
  return *this;
}

///
/// @brief Add dynamic input
/// @param [in] name
/// @param [in] num
/// @param [in] tensor
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddDynamicInput(
  const std::string &name, uint32_t num, const GeTensorDesc &tensor) {
  for (uint32_t i = 0; i < num; i++) {
    inputs_.emplace_back(std::make_pair(name + std::to_string(i), tensor));
  }
  return *this;
}

///
/// @brief Add output
/// @param [in] name
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddOutput(const std::string &name) {
  outputs_.emplace_back(std::make_pair(name, GeTensorDesc()));
  return *this;
}

///
/// @brief Add output
/// @param [in] name
/// @param [in] tensor
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddOutput(const std::string &name,
                                                                                       const GeTensorDesc &tensor) {
  outputs_.emplace_back(std::make_pair(name, tensor));
  return *this;
}

///
/// @brief Add dynamic output
/// @param [in] name
/// @param [in] num
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddDynamicOutput(const std::string &name,
                                                                                              uint32_t num) {
  for (uint32_t i = 0; i < num; i++) {
    outputs_.emplace_back(std::make_pair(name + std::to_string(i), GeTensorDesc()));
  }
  return *this;
}

///
/// @brief Add dynamic output
/// @param [in] name
/// @param [in] num
/// @param [in] tensor
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder &OpDescBuilder::AddDynamicOutput(
  const std::string &name, uint32_t num, const GeTensorDesc &tensor) {
  for (uint32_t i = 0; i < num; i++) {
    outputs_.emplace_back(std::make_pair(name + std::to_string(i), tensor));
  }
  return *this;
}

///
/// @brief Build op_desc
/// @return OpDescPtr
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescBuilder::Build() {
  OpDescPtr op_desc = shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name_, type_));
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "OpDesc is nullptr");
    return nullptr;
  }

  for (auto &input : inputs_) {
    if (op_desc->AddInputDesc(input.first, input.second) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Add input_desc failed.");
      return nullptr;
    }
  }

  for (auto &output : outputs_) {
    if (op_desc->AddOutputDesc(output.first, output.second) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Add output_desc failed.");
      return nullptr;
    }
  }

  return op_desc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDescUtils::SetSubgraphInstanceName(
  const std::string &subgraph_name, const std::string &subgraph_instance_name, OpDescPtr &op_desc) {
  const auto &subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  auto iter = subgraph_names_to_index.find(subgraph_name);
  if (iter == subgraph_names_to_index.end()) {
    GELOGE(GRAPH_PARAM_INVALID,
           "Failed to set subgraph instance %s for node %s type %s, the subgraph name %s does not exists",
           subgraph_instance_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(),
           subgraph_name.c_str());
    return GRAPH_PARAM_INVALID;
  }

  return op_desc->SetSubgraphInstanceName(iter->second, subgraph_instance_name);
}
}  // namespace ge
/*lint +e512 +e737 +e752*/
