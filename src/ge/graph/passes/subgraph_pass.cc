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
#include <stack>
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

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
    if (is_sub_graph && (node->GetType() == DATA)) {
      if (SubgraphInputNode(graph, node) != SUCCESS) {
        GELOGE(FAILED, "Handle input %s of subgraph failed.", node->GetName().c_str());
        return FAILED;
      }
      continue;
    }

    // NetOutput in subgraph
    if (is_sub_graph && (node->GetType() == NETOUTPUT)) {
      if (SubgraphOutputNode(graph, node) != SUCCESS) {
        GELOGE(FAILED, "Handle output %s of subgraph failed.", node->GetName().c_str());
        return FAILED;
      }
      continue;
    }

    if (kWhileOpTypes.count(node->GetType()) > 0) {
      // Input->While and Input link to other nodes
      if (WhileInputNodes(graph, node) != SUCCESS) {
        GELOGE(FAILED, "Handle input of while_body failed, while:%s.", node->GetName().c_str());
        return FAILED;
      }
      // body subgraph of While op
      if (WhileBodySubgraph(graph, node) != SUCCESS) {
        GELOGE(FAILED, "Handle while_body failed, while:%s.", node->GetName().c_str());
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
  GELOGD("Hadle input_node %s for graph %s.", node->GetName().c_str(), graph->GetName().c_str());
  // Data has and only has one output
  bool input_continues_required_flag = false;
  OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(0);
  std::vector<InDataAnchorPtr> in_anchors;
  for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    input_continues_required_flag =
      input_continues_required_flag || IsInputContinuesRequired(peer_in_anchor->GetOwnerNode());
    in_anchors.emplace_back(peer_in_anchor);
  }
  // Data->InputContinuesRequiredOp in subgraph need memcpy.
  if (input_continues_required_flag) {
    GELOGD("Data %s output_node required continues input.", node->GetName().c_str());
    std::string name = node->GetName() + "_" + MEMCPYASYNC + "_" + std::to_string(memcpy_num_++);
    if (InsertMemcpyNode(graph, out_data_anchor, in_anchors, name) != SUCCESS) {
      GELOGE(FAILED, "Insert memcpy after %s failed.", node->GetName().c_str());
      return FAILED;
    }
  }

  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(FAILED, "Get attr PARENT_NODE_INDEX failed, node:%s.", node->GetName().c_str());
    return FAILED;
  }

  NodePtr in_node = NodeUtils::GetParentInput(node);
  GE_CHECK_NOTNULL(in_node);
  // Subgraph Data Node, check for constant input.
  std::string const_type;
  if (!NodeUtils::GetConstOpType(in_node, const_type)) {
    return SUCCESS;
  }

  const NodePtr &parent_node = graph->GetParentNode();
  if (kWhileOpTypes.count(parent_node->GetType()) == 0) {
    if (!AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_PARENT_CONST_TYPE, const_type)) {
      GELOGE(FAILED, "Set attr PARENT_NODE_INDEX failed, node:%s.", node->GetName().c_str());
      return FAILED;
    }
  } else {
    // Constant input to While need memcpy.
    const ComputeGraphPtr &parent_graph = parent_node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(parent_graph);
    const InDataAnchorPtr &in_data_anchor = parent_node->GetInDataAnchor(parent_index);
    GE_CHECK_NOTNULL(in_data_anchor);
    const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    GELOGD("Constant input %s links to While %s.", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
           parent_node->GetName().c_str());
    std::string name = in_node->GetName() + "_" + MEMCPYASYNC + "_" + std::to_string(memcpy_num_++);
    if (InsertMemcpyNode(parent_graph, peer_out_anchor, {in_data_anchor}, name) != SUCCESS) {
      GELOGE(FAILED, "Insert memcpy between %s and %s failed.", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
             parent_node->GetName().c_str());
      return FAILED;
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
Status SubgraphPass::SubgraphOutputNode(const ComputeGraphPtr &graph, const NodePtr &node) {
  for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(in_node);

    // Need insert memcpy
    //   1. Const->NetOutput in subgraph
    //   2. AtomicOp->NetOutput in subgraph
    //   3. OutputContinuesRequiredOp->NetOutput in subgraph
    //   4. Data->NetOutput in subgraph but not while body
    std::string op_type;
    bool insert_flag = NodeUtils::GetConstOpType(in_node, op_type) ||
                       IsAtomicRequired(in_node, peer_out_anchor->GetIdx()) || IsOutputContinuesRequired(in_node) ||
                       ((in_node->GetType() == DATA) && !IsWhileBodyOutput(in_data_anchor));
    if (insert_flag) {
      GELOGI("Insert MemcpyAsync node between %s and %s.", node->GetName().c_str(), in_node->GetName().c_str());
      std::string name = in_node->GetName() + "_" + MEMCPYASYNC + "_" + std::to_string(memcpy_num_++);
      if (InsertMemcpyNode(graph, peer_out_anchor, {in_data_anchor}, name) != SUCCESS) {
        GELOGE(FAILED, "Insert memcpy between %s and %s failed.", in_node->GetName().c_str(), node->GetName().c_str());
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
    // Input->While and Input link to other nodes need insert memcpy
    if (peer_out_anchor->GetPeerInDataAnchors().size() > 1) {
      GELOGI("Input %s of While %s links to other nodes.", in_node->GetName().c_str(), node->GetName().c_str());
      std::string name = in_node->GetName() + "_" + MEMCPYASYNC + "_" + std::to_string(memcpy_num_++);
      if (InsertMemcpyNode(graph, peer_out_anchor, {in_data_anchor}, name) != SUCCESS) {
        GELOGE(FAILED, "Insert memcpy between %s and %s failed.", in_node->GetName().c_str(), node->GetName().c_str());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check body subgraph of While op
 * @param [in] graph: ComputeGraph.
 * @param [in] node: While node.
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::WhileBodySubgraph(const ComputeGraphPtr &graph, const NodePtr &node) {
  ComputeGraphPtr while_body = GetWhileBodySubgraph(graph, node);
  if (while_body == nullptr) {
    GELOGE(FAILED, "while_body of %s is NULL.", node->GetName().c_str());
    return FAILED;
  }

  NodePtr output_node = while_body->FindFirstNodeMatchType(NETOUTPUT);
  if (output_node == nullptr) {
    GELOGE(FAILED, "net_output_node not exist in graph %s.", while_body->GetName().c_str());
    return FAILED;
  }
  OpDescPtr output_desc = output_node->GetOpDesc();
  GE_CHECK_NOTNULL(output_desc);
  std::unordered_map<NodePtr, std::vector<uint32_t>> node_to_attr_index;
  for (const InDataAnchorPtr &in_data_anchor : output_node->GetAllInDataAnchors()) {
    uint32_t index = 0;
    if (!AttrUtils::GetInt(output_desc->GetInputDesc(in_data_anchor->GetIdx()), ATTR_NAME_PARENT_NODE_INDEX, index)) {
      GELOGE(FAILED, "Get attr PARENT_NODE_INDEX failed, node %s:%u.", output_node->GetName().c_str(),
             in_data_anchor->GetIdx());
      return FAILED;
    }
    MarkOutputIndex(in_data_anchor->GetPeerOutAnchor(), index, node_to_attr_index);
  }

  std::set<NodePtr> data_nodes;
  std::set<uint32_t> netoutput_input_indexes;
  GetExchangeInOut(node_to_attr_index, data_nodes, netoutput_input_indexes);
  return InsertMemcpyInWhileBody(while_body, data_nodes, output_node, netoutput_input_indexes);
}

/**
 * @ingroup ge
 * @brief Get body subgraph of While op
 * @param [in] graph: ComputeGraph.
 * @param [in] node: While node.
 * @return: body subgraph
 */
ComputeGraphPtr SubgraphPass::GetWhileBodySubgraph(const ComputeGraphPtr &graph, const NodePtr &node) {
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "op_desc is NULL.");
    return nullptr;
  }

  const std::vector<std::string> &subgraph_instance_names = op_desc->GetSubgraphInstanceNames();
  std::string body_instance_name;
  for (const std::string &instance_name : subgraph_instance_names) {
    std::string subgraph_name;
    if (op_desc->GetSubgraphNameByInstanceName(instance_name, subgraph_name) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Get subgraph_name by instance_name %s failed, node:%s.", instance_name.c_str(),
             node->GetName().c_str());
      return nullptr;
    }
    if (subgraph_name == ATTR_NAME_WHILE_BODY) {
      body_instance_name = instance_name;
      break;
    }
  }

  ComputeGraphPtr root_graph = GraphUtils::FindRootGraph(graph);
  if (root_graph == nullptr) {
    GELOGE(FAILED, "root_graph is NULL.");
    return nullptr;
  }

  return root_graph->GetSubgraph(body_instance_name);
}

/**
 * @ingroup ge
 * @brief Mark output parent_node_index
 * @param [in] peer_out_anchor: peer_out_anchor of NetOutput
 * @param [in] index: parent_node_index of NetOutput
 * @param [out] node_to_attr_index: key for node in subgraph, value for parent_node_index
 * @return: void
 */
void SubgraphPass::MarkOutputIndex(const OutDataAnchorPtr &peer_out_anchor, uint32_t index,
                                   std::unordered_map<NodePtr, std::vector<uint32_t>> &node_to_attr_index) {
  if (peer_out_anchor == nullptr) {
    return;
  }
  std::set<NodePtr> visited_nodes;
  std::stack<NodePtr> nodes;
  nodes.emplace(peer_out_anchor->GetOwnerNode());
  while (!nodes.empty()) {
    NodePtr cur_node = nodes.top();
    nodes.pop();
    if (visited_nodes.count(cur_node) > 0) {
      continue;
    }
    node_to_attr_index[cur_node].emplace_back(index);
    for (const NodePtr &in_node : cur_node->GetInDataNodes()) {
      nodes.emplace(in_node);
    }
    visited_nodes.emplace(cur_node);
  }
}

/**
 * @ingroup ge
 * @brief Get data_nodes / input_indexes of netoutput if need insert memcpy
 * @param [in] node_to_attr_index: key for node in subgraph, value for parent_node_index
 * @param [out] data_nodes: data_nodes need insert memcpy
 * @param [out] netoutput_input_indexes: input_indexes of netoutput need insert memcpy
 * @return: void
 */
void SubgraphPass::GetExchangeInOut(const std::unordered_map<NodePtr, std::vector<uint32_t>> &node_to_attr_index,
                                    std::set<NodePtr> &data_nodes, std::set<uint32_t> &netoutput_input_indexes) {
  for (const auto &item : node_to_attr_index) {
    NodePtr node = item.first;
    uint32_t input_index = 0;
    if ((node->GetType() != DATA) || !AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, input_index)) {
      continue;
    }
    if (item.second.empty() || ((item.second.size() == 1) && (item.second[0] == input_index))) {
      continue;
    }
    data_nodes.emplace(node);

    // Data node has and only has one output
    OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(0);
    if (out_data_anchor == nullptr) {
      continue;
    }
    for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      NodePtr out_node = peer_in_anchor->GetOwnerNode();
      if ((out_node->GetType() != NETOUTPUT) || (out_node->GetOpDesc() == nullptr)) {
        continue;
      }
      uint32_t output_index = 0;
      GeTensorDesc input_tensor = out_node->GetOpDesc()->GetInputDesc(peer_in_anchor->GetIdx());
      if (!AttrUtils::GetInt(input_tensor, ATTR_NAME_PARENT_NODE_INDEX, output_index)) {
        continue;
      }
      if (input_index != output_index) {
        netoutput_input_indexes.emplace(peer_in_anchor->GetIdx());
      }
    }
  }
}

/**
 * @ingroup ge
 * @brief Insert memcpy node in while_body
 * @param [in] graph: while_body
 * @param [in] data_nodes: data_nodes need insert memcpy
 * @param [in] output_node: NetOutput in while_body
 * @param [in] netoutput_input_indexes: input_indexes of netoutput need insert memcpy
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::InsertMemcpyInWhileBody(const ComputeGraphPtr &graph, const std::set<NodePtr> &data_nodes,
                                             const NodePtr &output_node,
                                             const std::set<uint32_t> &netoutput_input_indexes) {
  for (const NodePtr &data_node : data_nodes) {
    // Data node has and only has one output
    OutDataAnchorPtr out_data_anchor = data_node->GetOutDataAnchor(0);
    std::vector<InDataAnchorPtr> in_anchors;
    for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      in_anchors.emplace_back(peer_in_anchor);
    }
    std::string name = data_node->GetName() + "_" + MEMCPYASYNC + "_" + std::to_string(memcpy_num_++);
    GELOGD("Insert memcpy after while_body %s input_node %s.", graph->GetName().c_str(), data_node->GetName().c_str());
    if (InsertMemcpyNode(graph, out_data_anchor, in_anchors, name) != SUCCESS) {
      GELOGE(FAILED, "Insert MemcpyAsync node %s after %s failed.", name.c_str(), data_node->GetName().c_str());
      return FAILED;
    }
  }

  for (uint32_t index : netoutput_input_indexes) {
    InDataAnchorPtr in_data_anchor = output_node->GetInDataAnchor(index);
    GE_CHECK_NOTNULL(in_data_anchor);
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    std::string name =
      peer_out_anchor->GetOwnerNode()->GetName() + "_" + MEMCPYASYNC + "_" + std::to_string(memcpy_num_++);
    GELOGD("Insert memcpy after while_body %s output %u.", graph->GetName().c_str(), index);
    if (InsertMemcpyNode(graph, peer_out_anchor, {in_data_anchor}, name) != SUCCESS) {
      GELOGE(FAILED, "Insert MemcpyAsync node %s after %s failed.", name.c_str(),
             peer_out_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
  }

  std::set<NodePtr> memcpy_nodes;
  std::set<NodePtr> loop_body_nodes;
  for (const NodePtr &data_node : data_nodes) {
    // data_node has only one output node
    NodePtr memcpy_node = data_node->GetOutDataNodes().at(0);
    GE_CHECK_NOTNULL(memcpy_node);
    memcpy_nodes.emplace(memcpy_node);
    for (const NodePtr &out_node : memcpy_node->GetOutDataNodes()) {
      loop_body_nodes.insert(out_node);
    }
  }
  return InsertNoOp(graph, memcpy_nodes, loop_body_nodes);
}

/**
 * @ingroup ge
 * @brief Insert NoOp node between memcpy_nodes and loop_body_nodes
 * @param [in] graph: while_body
 * @param [in] memcpy_nodes
 * @param [in] loop_body_nodes
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::InsertNoOp(const ComputeGraphPtr &graph, const std::set<NodePtr> &memcpy_nodes,
                                const std::set<NodePtr> &loop_body_nodes) {
  if (memcpy_nodes.empty() || loop_body_nodes.empty()) {
    return SUCCESS;
  }

  OpDescBuilder noop_desc_builder("NoOp_for_Control", NOOP);
  OpDescPtr noop_desc = noop_desc_builder.Build();
  NodePtr noop_node = graph->AddNode(noop_desc);
  GE_CHECK_NOTNULL(noop_node);
  for (const NodePtr &memcpy_node : memcpy_nodes) {
    if (GraphUtils::AddEdge(memcpy_node->GetOutControlAnchor(), noop_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add ctrl edge %s->%s failed.", memcpy_node->GetName().c_str(), noop_node->GetName().c_str());
      return FAILED;
    }
  }
  for (const NodePtr &loop_body_node : loop_body_nodes) {
    if (GraphUtils::AddEdge(noop_node->GetOutControlAnchor(), loop_body_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add ctrl edge %s->%s failed.", noop_node->GetName().c_str(), loop_body_node->GetName().c_str());
      return FAILED;
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
  if (kWhileOpTypes.count(parent_node->GetType()) == 0) {
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
 * @brief Check is AtomicOp->NetOutput
 * @param [in] node
 * @param [in] out_index
 * @return: true for AtomicOp->NetOutput / false for others
 */
bool SubgraphPass::IsAtomicRequired(const NodePtr &node, int64_t out_index) {
  auto op_desc = node->GetOpDesc();
  if (op_desc != nullptr) {
    bool is_atomic = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic);
    if (is_atomic) {
      std::vector<int64_t> atomic_output_index;
      // If GetListInt fail, atomic_output_index is empty.
      (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
      for (int64_t ind : atomic_output_index) {
        if (ind == out_index) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * @ingroup ge
 * @brief Check is OutputContinuesRequiredOp->NetOutput
 * @param [in] node
 * @return: true for OutputContinuesRequiredOp->NetOutput / false for others
 */
bool SubgraphPass::IsOutputContinuesRequired(const NodePtr &node) {
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc != nullptr) {
    bool continuous_output_flag = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_OUTPUT, continuous_output_flag);
    bool no_padding_continuous_output_flag = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, no_padding_continuous_output_flag);
    return continuous_output_flag || no_padding_continuous_output_flag;
  }
  return false;
}

/**
 * @ingroup ge
 * @brief Check is InputContinuesRequiredOp->NetOutput
 * @param [in] node
 * @return: true for InputContinuesRequiredOp->NetOutput / false for others
 */
bool SubgraphPass::IsInputContinuesRequired(const NodePtr &node) {
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc != nullptr) {
    bool continuous_input_flag = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_INPUT, continuous_input_flag);
    bool no_padding_continuous_input_flag = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, no_padding_continuous_input_flag);
    return continuous_input_flag || no_padding_continuous_input_flag;
  }
  return false;
}

/**
 * @ingroup ge
 * @brief Insert memcpy node
 * @param [in] graph
 * @param [in] out_anchor
 * @param [in] in_anchors
 * @param [in] name
 * @return: 0 for success / others for fail
 */
Status SubgraphPass::InsertMemcpyNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                                      const std::vector<InDataAnchorPtr> &in_anchors, const std::string &name) {
  GE_CHECK_NOTNULL(out_anchor);
  NodePtr in_node = out_anchor->GetOwnerNode();
  OpDescBuilder op_desc_builder(name, MEMCPYASYNC);
  OpDescPtr op_desc = op_desc_builder.AddInput("x", in_node->GetOpDesc()->GetOutputDesc(0))
                        .AddOutput("y", in_node->GetOpDesc()->GetOutputDesc(0))
                        .Build();
  if (GraphUtils::InsertNodeBefore(out_anchor, in_anchors, graph->AddNode(op_desc)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Insert MemcpyAsync node %s after %s failed.", name.c_str(), in_node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

}  // namespace ge
