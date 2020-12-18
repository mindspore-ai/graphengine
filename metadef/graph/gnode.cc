/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "graph/gnode.h"

#include <utility>
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/tensor_adapter.h"
#include <graph/utils/graph_utils.h>
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "utils/node_utils.h"
#include "utils/op_desc_utils.h"

namespace ge {
class NodeImpl {
 public:
  NodeImpl() = default;
  ~NodeImpl() = default;

  NodeImpl(NodeImpl &) = delete;
  NodeImpl &operator=(const NodeImpl &) = delete;

  std::weak_ptr<Node> node_ptr_;
};

NodePtr NodeAdapter::GNode2Node(const ge::GNode &graph_node) {
  if (graph_node.impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GNode2Node: gnode impl is nullptr.");
    return nullptr;
  }

  return graph_node.impl_->node_ptr_.lock();
}

GNode NodeAdapter::Node2GNode(const ge::NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "Node2GNode: node is nullptr");
    return GNode();
  }

  GNode graph_node;
  if (graph_node.impl_ == nullptr) {
    GELOGW("Node2GNode: gnode impl is nullptr, node[%s].", node->GetName().c_str());
    return graph_node;
  }
  graph_node.impl_->node_ptr_ = node;

  return graph_node;
}

GNodePtr NodeAdapter::Node2GNodePtr(const ge::NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "Node2GNodePtr: node is nullptr");
    return nullptr;
  }

  GNodePtr gnode = std::shared_ptr<GNode>(new (std::nothrow) GNode());
  if (gnode == nullptr) {
    GELOGE(GRAPH_FAILED, "Node2GNodePtr: gnode is nullptr, node[%s].", node->GetName().c_str());
    return nullptr;
  }

  if (gnode->impl_ == nullptr) {
    GELOGW("Node2GNode: gnode impl is nullptr, node[%s].", node->GetName().c_str());
    return nullptr;
  }
  gnode->impl_->node_ptr_ = node;

  return gnode;
}

GNode::GNode() { impl_ = ComGraphMakeShared<NodeImpl>(); }


graphStatus GNode::GetType(AscendString &type) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetType: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetType: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }
  std::string node_type = node_ptr->GetType();
  AscendString ascend_type(node_type.c_str());
  type = ascend_type;

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetName(AscendString &name) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetName: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetName: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }
  std::string node_name = node_ptr->GetName();
  AscendString ascend_name(node_name.c_str());
  name = ascend_name;

  return GRAPH_SUCCESS;
}

std::pair<GNodePtr, int32_t> GNode::GetInDataNodesAndPortIndexs(const int32_t index) const {
  pair<GNodePtr, int32_t> gnode_idx = {nullptr, 0xFF};
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "Gnode: node impl is nullptr.");
    return gnode_idx;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "Gnode: the shared ptr is not valid.");
    return gnode_idx;
  }

  auto in_anchor = node_ptr->GetInDataAnchor(index);
  if (in_anchor == nullptr) {
    GELOGE(GRAPH_FAILED, "Failed to get in data node of index[%d] from node[%s], the anchor does not exist",
           index, node_ptr->GetName().c_str());
    return gnode_idx;
  }

  auto out_anchor = in_anchor->GetPeerOutAnchor();
  if (out_anchor == nullptr) {
    GELOGE(GRAPH_FAILED, "Failed to get in data node of index[%d] from node [%s], the data input does not exist",
           index, node_ptr->GetName().c_str());
    return gnode_idx;
  }

  NodePtr peer_node_ptr = out_anchor->GetOwnerNode();
  GNodePtr gnode = NodeAdapter::Node2GNodePtr(peer_node_ptr);
  if (gnode == nullptr) {
    GELOGE(GRAPH_FAILED, "Peer node of node[%s] to gnode faild.", node_ptr->GetName().c_str());
    return gnode_idx;
  }

  return {gnode, out_anchor->GetIdx()};
}

std::vector<GNodePtr> GNode::GetInControlNodes() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "Gnode: node impl is nullptr.");
    return {};
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "Gnode: the shared ptr is not valid.");
    return {};
  }

  std::vector<GNodePtr> gnodes;
  auto in_control_nodes = node_ptr->GetInControlNodes();
  for (auto &in_control_node : in_control_nodes) {
    GNodePtr gnode = NodeAdapter::Node2GNodePtr(in_control_node);
    if (gnode == nullptr) {
      GELOGE(GRAPH_FAILED, "In control_node of node[%s] to gnode faild.", node_ptr->GetName().c_str());
      return {};
    }
    gnodes.emplace_back(gnode);
  }

  return gnodes;
}

std::vector<std::pair<GNodePtr, int32_t>> GNode::GetOutDataNodesAndPortIndexs(const int32_t index) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "Gnode: node impl is nullptr.");
    return {};
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "Gnode: the shared ptr is not valid.");
    return {};
  }

  auto out_anchor = node_ptr->GetOutDataAnchor(index);
  if (out_anchor == nullptr) {
    GELOGE(GRAPH_FAILED, "Failed to get out data node of index %d from node %s, the anchor does not exists",
           index, node_ptr->GetName().c_str());
    return {};
  }

  vector<std::pair<GNodePtr, int32_t>> gnode_index;
  auto in_data_anchors = out_anchor->GetPeerInDataAnchors();
  for (auto &in_data_anchor : in_data_anchors) {
    if (in_data_anchor == nullptr) {
      GELOGE(GRAPH_FAILED, "In data anchor of node[%s] is nullptr.", node_ptr->GetName().c_str());
      return {};
    }
    NodePtr peer_node_ptr = in_data_anchor->GetOwnerNode();
    GNodePtr gnode = NodeAdapter::Node2GNodePtr(peer_node_ptr);
    if (gnode == nullptr) {
      GELOGE(GRAPH_FAILED, "Peer node of node[%s] to gnode faild.", node_ptr->GetName().c_str());
      return {};
    }
    gnode_index.emplace_back(std::pair<GNodePtr, int32_t>(gnode, in_data_anchor->GetIdx()));
  }

  return gnode_index;
}

std::vector<GNodePtr> GNode::GetOutControlNodes() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetOutControlNodes: node impl is nullptr.");
    return {};
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetOutControlNodes: the node shared ptr is not valid.");
    return {};
  }

  std::vector<GNodePtr> gnodes;
  auto out_control_nodes = node_ptr->GetOutControlNodes();
  for (auto &out_control_node : out_control_nodes) {
    GNodePtr gnode = NodeAdapter::Node2GNodePtr(out_control_node);
    if (gnode == nullptr) {
      GELOGE(GRAPH_FAILED, "In control_node of node[%s] to gnode faild.", node_ptr->GetName().c_str());
      return {};
    }
    gnodes.emplace_back(gnode);
  }

  return gnodes;
}

graphStatus GNode::GetInputConstData(const int32_t index, Tensor &data) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetInputConstData: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetInputConstData: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  NodePtr input_data_node = NodeUtils::GetInDataNodeByIndex(*node_ptr, index);
  GE_CHECK_NOTNULL(input_data_node);
  string op_type = input_data_node->GetType();
  if (op_type == CONSTANT || op_type == CONSTANTOP) {
    Operator const_op = OpDescUtils::CreateOperatorFromNode(input_data_node);
    if (const_op.GetAttr(ATTR_NAME_WEIGHTS, data) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Input data node[%s] of node[%s] get data failed.",
             input_data_node->GetName().c_str(), node_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
    return SUCCESS;
  } else if (op_type == DATA) {
    auto parent_node = NodeUtils::GetParentInput(input_data_node);
    while ((parent_node != nullptr) && (parent_node->GetType() == DATA)) {
      parent_node = NodeUtils::GetParentInput(parent_node);
    }
    if ((parent_node != nullptr) &&
        ((parent_node->GetType() == CONSTANT) || (parent_node->GetType() == CONSTANTOP))) {
      Operator const_op =  OpDescUtils::CreateOperatorFromNode(parent_node);
      if (const_op.GetAttr(ATTR_NAME_WEIGHTS, data) != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "Input data node[%s] of node[%s] get data failed.",
               parent_node->GetName().c_str(), node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }
  }

  GELOGE(GRAPH_NODE_WITHOUT_CONST_INPUT, "Node[%s] has no const input.", node_ptr->GetName().c_str());
  return GRAPH_NODE_WITHOUT_CONST_INPUT;
}

graphStatus GNode::GetInputIndexByName(const AscendString &name, int32_t &index) {
  const char* ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "GetInputIndexByName: ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetInputIndexByName: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetInputIndexByName: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  std::string node_name = ascend_name;
  index = op_desc->GetInputIndexByName(node_name);

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetOutputIndexByName(const AscendString &name, int32_t &index) {
  const char* ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "GetOutputIndexByName: ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetOutputIndexByName: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetOutputIndexByName: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  std::string node_name = ascend_name;
  index = op_desc->GetOutputIndexByName(node_name);

  return GRAPH_SUCCESS;
}

size_t GNode::GetInputsSize() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetInputsSize: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetInputsSize: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return op_desc->GetInputsSize();
}

size_t GNode::GetOutputsSize() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetOutputsSize: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetOutputsSize: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return op_desc->GetOutputsSize();
}

graphStatus GNode::GetInputDesc(const int32_t index, TensorDesc &tensor_desc) const {
  if (index < 0) {
    GELOGE(GRAPH_PARAM_INVALID, "GetInputDesc: index[%d] cannot be less than zero.", index);
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetInputDesc: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetInputDesc: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  ConstGeTensorDescPtr ge_tensor_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(index));
  if (ge_tensor_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get tensor desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }
  tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(*ge_tensor_desc);

  return GRAPH_SUCCESS;
}

graphStatus GNode::UpdateInputDesc(const int32_t index, const TensorDesc &tensor_desc) {
  if (index < 0) {
    GELOGE(GRAPH_PARAM_INVALID, "UpdateInputDesc: index[%d] cannot be less than zero.", index);
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "UpdateInputDesc: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "UpdateInputDesc: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  GeTensorDesc ge_tensor_desc = TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc);
  if (op_desc->UpdateInputDesc(static_cast<uint32_t>(index), ge_tensor_desc) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Update input desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetOutputDesc(const int32_t index, TensorDesc &tensor_desc) const {
  if (index < 0) {
    GELOGE(GRAPH_PARAM_INVALID, "GetOutputDesc: index[%d] cannot be less than zero.", index);
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetOutputDesc: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetOutputDesc: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  ConstGeTensorDescPtr ge_tensor_desc = op_desc->GetOutputDescPtr(static_cast<uint32_t>(index));
  if (ge_tensor_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get tensor desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }
  tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(*ge_tensor_desc);

  return GRAPH_SUCCESS;
}

graphStatus GNode::UpdateOutputDesc(const int32_t index, const TensorDesc &tensor_desc) {
  if (index < 0) {
    GELOGE(GRAPH_PARAM_INVALID, "Gnode: index[%d] cannot be less than zero.", index);
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "UpdateOutputDesc: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "UpdateOutputDesc: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  GeTensorDesc ge_tensor_desc = TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc);
  if (op_desc->UpdateOutputDesc(static_cast<uint32_t>(index), ge_tensor_desc) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Update input desc of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

#define NODE_ATTR_GET_IMP(ArgType)                                                                                     \
  graphStatus GNode::GetAttr(const AscendString &name, ArgType &attr_value) const {                                    \
    const char* ascend_name = name.GetString();                                                                        \
    if (ascend_name == nullptr) {                                                                                      \
      GELOGE(GRAPH_PARAM_INVALID, "GetAttr: ascend string error.");                                                    \
      return GRAPH_PARAM_INVALID;                                                                                      \
    }                                                                                                                  \
                                                                                                                       \
    if (impl_ == nullptr) {                                                                                            \
      GELOGE(GRAPH_FAILED, "GetAttr: node impl is nullptr.");                                                          \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();                                                          \
    if (node_ptr == nullptr) {                                                                                         \
      GELOGE(GRAPH_FAILED, "GetAttr: the shared ptr is not valid.");                                                   \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    std::string node_name = ascend_name;                                                                               \
    Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);                                                       \
    if (op.GetAttr(node_name, attr_value) != GRAPH_SUCCESS) {                                                          \
      GELOGE(GRAPH_FAILED, "Get attr of node[%s] failed.", node_ptr->GetName().c_str());                               \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    return GRAPH_SUCCESS;                                                                                              \
  }

#define NODE_ATTR_SET_IMP(ArgType)                                                                                     \
  graphStatus GNode::SetAttr(const AscendString &name, ArgType &attr_value) const {                                    \
    const char* ascend_name = name.GetString();                                                                        \
    if (ascend_name == nullptr) {                                                                                      \
      GELOGE(GRAPH_PARAM_INVALID, "SetAttr: ascend string error.");                                                    \
      return GRAPH_PARAM_INVALID;                                                                                      \
    }                                                                                                                  \
                                                                                                                       \
    if (impl_ == nullptr) {                                                                                            \
      GELOGE(GRAPH_FAILED, "SetAttr: node impl is nullptr.");                                                          \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();                                                          \
    if (node_ptr == nullptr) {                                                                                         \
      GELOGE(GRAPH_FAILED, "SetAttr: the shared ptr is not valid.");                                                   \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    std::string node_name = ascend_name;                                                                               \
    Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);                                                       \
    (void)op.SetAttr(node_name, attr_value);                                                                           \
    return GRAPH_SUCCESS;                                                                                              \
  }

NODE_ATTR_GET_IMP(int64_t)
NODE_ATTR_GET_IMP(int32_t)
NODE_ATTR_GET_IMP(uint32_t)
NODE_ATTR_GET_IMP(float)
NODE_ATTR_GET_IMP(bool)
NODE_ATTR_GET_IMP(Tensor)
NODE_ATTR_GET_IMP(std::vector<int64_t>)
NODE_ATTR_GET_IMP(std::vector<int32_t>)
NODE_ATTR_GET_IMP(std::vector<uint32_t>)
NODE_ATTR_GET_IMP(std::vector<float>)
NODE_ATTR_GET_IMP(std::vector<bool>)
NODE_ATTR_GET_IMP(std::vector<Tensor>)
NODE_ATTR_GET_IMP(OpBytes)
NODE_ATTR_GET_IMP(std::vector<std::vector<int64_t>>)
NODE_ATTR_GET_IMP(std::vector<ge::DataType>)
NODE_ATTR_GET_IMP(ge::DataType)
NODE_ATTR_GET_IMP(AttrValue)

NODE_ATTR_SET_IMP(int64_t)
NODE_ATTR_SET_IMP(int32_t)
NODE_ATTR_SET_IMP(uint32_t)
NODE_ATTR_SET_IMP(float)
NODE_ATTR_SET_IMP(bool)
NODE_ATTR_SET_IMP(Tensor)
NODE_ATTR_SET_IMP(std::vector<int64_t>)
NODE_ATTR_SET_IMP(std::vector<int32_t>)
NODE_ATTR_SET_IMP(std::vector<uint32_t>)
NODE_ATTR_SET_IMP(std::vector<float>)
NODE_ATTR_SET_IMP(std::vector<bool>)
NODE_ATTR_SET_IMP(std::vector<Tensor>)
NODE_ATTR_SET_IMP(OpBytes)
NODE_ATTR_SET_IMP(std::vector<std::vector<int64_t>>)
NODE_ATTR_SET_IMP(std::vector<ge::DataType>)
NODE_ATTR_SET_IMP(ge::DataType)

graphStatus GNode::SetAttr(const AscendString &name, AttrValue &attr_value) const {
  const char* ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "SetAttr: ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "SetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "SetAttr: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  std::string node_name = ascend_name;
  Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  (void)op.SetAttr(node_name, std::move(attr_value));
  return GRAPH_SUCCESS;
}

graphStatus GNode::SetAttr(const AscendString &name, AscendString &attr_value) const {
  const char* ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "SetAttr: name ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  const char* ascend_attr_value = attr_value.GetString();
  if (ascend_attr_value == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "SetAttr: attr value ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "SetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "SetAttr: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }
  std::string node_name = ascend_name;
  std::string node_attr_value = ascend_attr_value;
  Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  (void)op.SetAttr(node_name, node_attr_value);

  return GRAPH_SUCCESS;
}

graphStatus GNode::SetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const {
  const char* ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "SetAttr: name ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  for (auto &attr_val : attr_values) {
    const char* ascend_attr_value = attr_val.GetString();
    if (ascend_attr_value == nullptr) {
      GELOGE(GRAPH_PARAM_INVALID, "SetAttr: attr val error.");
      return GRAPH_PARAM_INVALID;
    }
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "SetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "SetAttr: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }
  vector<std::string> node_attr_vals;
  for (auto attr_val : attr_values) {
    if (attr_val.GetString() != nullptr) {
      std::string node_attr_val = attr_val.GetString();
      node_attr_vals.emplace_back(node_attr_val);
    }
  }
  std::string node_name = ascend_name;
  Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  (void)op.SetAttr(node_name, node_attr_vals);

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetAttr(const AscendString &name, AscendString &attr_value) const {
  const char* ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "GetAttr: name ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetAttr: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  std::string node_name = ascend_name;
  Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  std::string op_name;
  if (op.GetAttr(node_name, op_name) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Get attr of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  AscendString attr_value_get(op_name.c_str());
  attr_value = attr_value_get;

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const {
  const char* ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "GetAttr: name ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetAttr: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  std::string node_name = ascend_name;
  Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  vector<std::string> attr_names;
  if (op.GetAttr(node_name, attr_names) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Get attr of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  for (auto &attr_name : attr_names) {
    AscendString ascend_attr_name(attr_name.c_str());
    attr_values.push_back(ascend_attr_name);
  }

  return GRAPH_SUCCESS;
}

bool GNode::HasAttr(const AscendString &name) {
  const char* ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "HasAttr: ascend string error.");
    return false;
  }

  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "HasAttr: node impl is nullptr.");
    return false;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "HasAttr: the node shared ptr is not valid.");
    return false;
  }

  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    return false;
  }
  std::string attr_name = ascend_name;
  if (!op_desc->HasAttr(attr_name)) {
    GELOGE(GRAPH_FAILED, "Node[%s] has no attr name[%s]", node_ptr->GetName().c_str(), attr_name.c_str());
    return false;
  }

  return true;
}

graphStatus GNode::GetSubgraph(uint32_t index, GraphPtr &graph) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetSubgraph: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetSubgraph: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  ComputeGraphPtr compute_graph_ptr = NodeUtils::GetSubgraph(*node_ptr, index);
  if (compute_graph_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetSubgraph: get subgraph[%u] failed from node[%s].", index, node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  graph = GraphUtils::CreateGraphPtrFromComputeGraph(compute_graph_ptr);
  if (graph == nullptr) {
    GELOGE(GRAPH_FAILED, "GetSubgraph: get subgraph[%u] failed from node[%s].", index, node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetALLSubgraphs(std::vector<GraphPtr> &graph_list) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetALLSubgraphs: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "GetALLSubgraphs: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  std::vector<ComputeGraphPtr> sub_graphs = NodeUtils::GetAllSubgraphs(*node_ptr);
  if (sub_graphs.empty()) {
    GELOGE(GRAPH_FAILED, "GetALLSubgraphs: get all subgraphs failed from node[%s].", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  for (auto &sub_graph : sub_graphs) {
    if (sub_graph == nullptr) {
      GELOGE(GRAPH_FAILED, "Get subgraph failed from node[%s].", node_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
    GraphPtr graph = GraphUtils::CreateGraphPtrFromComputeGraph(sub_graph);
    if (graph == nullptr) {
      GELOGE(GRAPH_FAILED, "Subgraph create compute graph failed from node[%s].", node_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
    graph_list.emplace_back(graph);
  }

  if (graph_list.empty()) {
    GELOGW("Node[%s] has no subgraph.", node_ptr->GetName().c_str());
  }

  return GRAPH_SUCCESS;
}
}  // namespace ge
