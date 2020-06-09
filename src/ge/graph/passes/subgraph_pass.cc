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

#include "graph/passes/subgraph_pass.h"

#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace {
const std::set<std::string> kWhileTypes = {ge::WHILE, ge::_WHILE, ge::STATELESSWHILE};
}

namespace ge {

/**
 * @ingroup ge
 * @brief Subgraph optimizer.
 * @param [in] graph: Input ComputeGraph
 * @return: 0 for success / others for fail
 */
Status SubgraphPass::Run(ComputeGraphPtr graph) {
  const bool is_sub_graph = graph->GetParentNode() != nullptr;
  for (const NodePtr &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());

    if (is_sub_graph && (node->GetType() == DATA)) {
      if (SubgraphInputNode(graph, node) != SUCCESS) {
        return FAILED;
      }
      continue;
    }

    // 2. Const->NetOutput in subgraph
    // 3. Data->NetOutput in subgraph but not while body
    if (is_sub_graph && (node->GetType() == NETOUTPUT)) {
      if (SubgraphOutputNode(graph, node) != SUCCESS) {
        return FAILED;
      }
      continue;
    }

    // 4. Input->While and Input link to other nodes
    if (kWhileTypes.count(node->GetType()) > 0) {
      if (WhileInputNodes(graph, node) != SUCCESS) {
        return FAILED;
      }
      continue;
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check Subgraph NetOutput node
 * @param [in] graph: ComputeGraph.
 * @param [in] node: NetOutput node in Subgraph.
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::SubgraphInputNode(const ComputeGraphPtr &graph, const NodePtr &node) {
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return FAILED;
  }

  // Subgraph Data Node, check for constant input.
  std::string const_type;
  NodePtr in_node = NodeUtils::GetParentInput(node);
  if (!NodeUtils::GetConstOpType(in_node, const_type)) {
    return SUCCESS;
  }

  if (!AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_PARENT_CONST_TYPE, const_type)) {
    return FAILED;
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check Subgraph NetOutput node
 * @param [in] graph: ComputeGraph.
 * @param [in] node: NetOutput node in Subgraph.
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::SubgraphOutputNode(const ComputeGraphPtr &graph, const NodePtr &node) {
  for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(in_node);

    // Need insert memcpy
    //   2. Const->NetOutput in subgraph
    //   3. Data->NetOutput in subgraph but not while body
    std::string op_type;
    bool input_const_flag = NodeUtils::GetConstOpType(in_node, op_type);
    if ((in_node->GetType() == DATA) && !IsWhileBodyOutput(in_data_anchor)) {
      input_const_flag = true;
    }

    if (input_const_flag) {
      GELOGI("Insert MemcpyAsync node between %s and %s.", node->GetName().c_str(), in_node->GetName().c_str());
      std::string name = in_node->GetName() + "_" + MEMCPYASYNC + "_" + std::to_string(memcpy_num_++);
      if (InsertMemcpyNode(graph, peer_out_anchor, in_data_anchor, name) != SUCCESS) {
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check is Input->While and Input link to other nodes
 * @param [in] graph: ComputeGraph.
 * @param [in] node: While node.
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::WhileInputNodes(const ComputeGraphPtr &graph, const NodePtr &node) {
  for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(in_node);

    // Need insert memcpy
    //   4. Input->While and Input link to other nodes
    if (peer_out_anchor->GetPeerInDataAnchors().size() > 1) {
      GELOGI("Insert MemcpyAsync node between %s and %s.", node->GetName().c_str(), in_node->GetName().c_str());
      std::string name = in_node->GetName() + "_" + MEMCPYASYNC + "_" + std::to_string(memcpy_num_++);
      if (InsertMemcpyNode(graph, peer_out_anchor, in_data_anchor, name) != SUCCESS) {
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check is data->netoutput in while body
 * @param [in] in_data_anchor
 * @return: true for data->netoutput in while body / for false for others
 */
bool SubgraphPass::IsWhileBodyOutput(const InDataAnchorPtr &in_data_anchor) {
  // Check is subgraph
  NodePtr parent_node = in_data_anchor->GetOwnerNode()->GetOwnerComputeGraph()->GetParentNode();
  if (parent_node == nullptr) {
    return false;
  }

  // Check if parent_node is While
  if (kWhileTypes.count(parent_node->GetType()) == 0) {
    return false;
  }

  // While cond / body
  OpDescPtr op_desc = in_data_anchor->GetOwnerNode()->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  return AttrUtils::HasAttr(op_desc->GetInputDesc(in_data_anchor->GetIdx()), ATTR_NAME_PARENT_NODE_INDEX);
}

/**
 * @ingroup ge
 * @brief Insert memcpy node
 * @param [in] graph
 * @param [in] out_anchor
 * @param [in] in_anchor
 * @param [in] name
 * @return: 0 for success / others for fail
 */
Status SubgraphPass::InsertMemcpyNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                                      const InDataAnchorPtr &in_anchor, const std::string &name) {
  GE_CHECK_NOTNULL(out_anchor);
  GE_CHECK_NOTNULL(in_anchor);
  NodePtr in_node = out_anchor->GetOwnerNode();
  OpDescBuilder op_desc_builder(name, MEMCPYASYNC);
  OpDescPtr op_desc = op_desc_builder.AddInput("x", in_node->GetOpDesc()->GetOutputDesc(0))
                        .AddOutput("y", in_node->GetOpDesc()->GetOutputDesc(0))
                        .Build();
  if (GraphUtils::InsertNodeBefore(out_anchor, {in_anchor}, graph->AddNode(op_desc)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Insert MemcpyAsync node %s between %s->%s failed.", name.c_str(), in_node->GetName().c_str(),
           in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

}  // namespace ge
