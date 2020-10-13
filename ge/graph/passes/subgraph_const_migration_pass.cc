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

#include "subgraph_const_migration_pass.h"

#include "graph/utils/node_utils.h"
#include "ge_local_engine/engine/host_cpu_engine.h"
#include "graph/passes/folding_pass.h"

namespace ge {
constexpr uint32_t kDataOutIndex = 0;
constexpr uint32_t kCaseInputBase = 1;
constexpr uint32_t kInvalidParent = 0x7fffffffU;

bool IsSameOpNode(const NodePtr &src_node, const NodePtr &dst_node) {
  if ((src_node == nullptr) && (dst_node == nullptr)) {
    return true;
  }

  if ((src_node == nullptr) || (dst_node == nullptr)) {
    return false;
  }

  if (src_node->GetType() != dst_node->GetType()) {
    return false;
  }

  if ((src_node->GetInControlNodes().size() != dst_node->GetInControlNodes().size()) ||
      (src_node->GetOutDataNodesSize() != dst_node->GetOutDataNodesSize())) {
    return false;
  }

  set<uint32_t> related_parent;
  const auto in_nodes = src_node->GetInControlNodes();
  for (uint32_t i = 0; i < in_nodes.size(); ++i) {
    const auto owner_node = in_nodes.at(i);
    uint32_t parent_index = 0;
    if (!AttrUtils::GetInt(owner_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      return false;
    }

    related_parent.insert(parent_index);
  }

  for (const auto &in_node : dst_node->GetInControlNodes()) {
    uint32_t parent_index = 0;
    if (!AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      return false;
    }

    if (related_parent.count(parent_index) == 0) {
      return false;
    }
  }

  return true;
}

/***********************************************************************************************************************
                                                                             +-----------+
                                                                             |   Data    |
                                                                             +-----------+
                                                                                   |
                                                                                   |
                                                                             +-----------+
                                                                             |   Cast    |
                                                                             +-----------+
                                                                                   |
                                                                                   |
                                                                             +-----------+ +-----------+ +-----------+
                                                                             | TransData | |   Data    | |   Data    |
                                                                             +-----------+ +-----------+ +-----------+
                                                                                        \        |        /
                                                                                         \       |       /
                                                                                          \      |      /
                                                                                           \     |     /
 +-----------+ +-----------+ +-----------+ +-----------+ +-----------+    +-----------+    +-----------+
 |   Data    | |   Data    | |   Data    | |   Data    | |   Data    |    |   Data    |    |  Conv2D   |
 +-----------+ +-----------+ +-----------+ +-----------+ +-----------+    +-----------+    +-----------+
        \                 \        |        /                  /                |                |
         \                 \       |       /                  /                 |                |
          \                 \      |      /                  /                  |                |
           \                 \     |     /                  /                   |                |
            \                +-----------+                 /                    |          +-----------+
             +---------------|   Const   |----------------+                     |          |  Pooling  |
                             +-----------+                                      |          +-----------+
                                   \                                            |               /
                                    \                                           |              /
                                     \                                    +-----------+       /
                                      +-----------------------------------|  Conv2D   |------+
                                                                          +-----------+
                                                                                |
                                                                                |
                                                                          +-----------+
                                                                          |   Node    |
                                                                          +-----------+
***********************************************************************************************************************/
Status SubgraphConstMigrationPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (graph->GetParentGraph() != nullptr) {
    GELOGD("Subgraph %s skip the SubgraphConstMigrationPass", graph->GetName().c_str());
    return SUCCESS;
  }

  GELOGD("Begin to run Subgraph Const Migration on graph: %s", graph->GetName().c_str());
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != CASE) {
      continue;
    }

    const auto &func_desc = node->GetOpDesc();
    if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
      GELOGD("Not multi-batch, Skip Case: %s", node->GetName().c_str());
      continue;
    }

    do {
      migration_append_ = false;
      map<ComputeGraphPtr, map<uint32_t, NodePtr>> graph_datas;
      if (ClassifyDataNodes(graph, func_desc, graph_datas) != SUCCESS) {
        return FAILED;
      }

      if (graph_datas.empty()) {
        GELOGW("Graph: %s subgraph is empty", graph->GetName().c_str());
        break;
      }

      // {subgraph0, {{1, Data}, {2, Data}, {3, Data}, {4, Data}, ..., {n, Data}}}
      // {subgraph1, {{1, Data}, {2, Data}, {3, Data}, {4, Data}, ..., {n, Data}}}
      // {subgraph2, {{1, Data}, {2, Data}, {3, Data}, {4, Data}, ..., {n, Data}}}
      const auto base_nodes = graph_datas.begin()->second;  // Need copy.
      for (const auto &node_item : base_nodes) {
        if (GraphNodeMigration(graph, node, graph_datas, node_item.second, node_item.first) != SUCCESS) {
          return FAILED;
        }
      }
    } while (migration_append_);
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get all Data nodes for all subgraph.
/// @param [in] graph: Root compute graph.
/// @param [in] func_desc: functional OpDesc of Case.
/// @param [out] graph_datas: Data groups of subgraph.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::ClassifyDataNodes(const ComputeGraphPtr &graph, const OpDescPtr &func_desc,
                                                     map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_datas) {
  for (const auto &name : func_desc->GetSubgraphInstanceNames()) {
    const auto &subgraph = graph->GetSubgraph(name);
    if (subgraph == nullptr) {
      GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "Subgraph not found, name: %s", name.c_str());
      return GE_GRAPH_EMPTY_SUBGRAPH;
    }

    auto &data_nodes = graph_datas[subgraph];
    for (auto &data : subgraph->GetDirectNode()) {
      if (data->GetType() != DATA) {
        continue;
      }

      uint32_t parent_index = 0;
      if (!AttrUtils::GetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
        GELOGE(FAILED, "Parent index not found, name: %s", data->GetName().c_str());
        return FAILED;
      }

      data_nodes[parent_index] = data;
      GELOGD("%s, Parent index: %u, Data: %s", subgraph->GetName().c_str(), parent_index, data->GetName().c_str());
    }
  }

  auto iter = graph_datas.begin();
  if (iter == graph_datas.end()) {
    return SUCCESS;
  }
  for (const auto &data_nodes : graph_datas) {
    if (data_nodes.second.size() != iter->second.size()) {
      GELOGE(FAILED, "Subgraph %s has invalid Data nodes[%zu != %zu]",
             data_nodes.first->GetName().c_str(), data_nodes.second.size(), iter->second.size());
      return FAILED;
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get all Data nodes for all subgraph.
/// @param [in] node: Const node of subgraph.
/// @param [out] inputs: parent index to Const.
/// @param [out] outputs: Data groups of subgraph.
/// @return true: SUCCESS / false: FAILED
///
bool SubgraphConstMigrationPass::GetAssociatedNodes(const NodePtr &node, map<uint32_t, uint32_t> &inputs,
                                                    map<uint32_t, uint32_t> &outputs) {
  for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
    outputs[i] = kInvalidParent;
  }

  uint32_t out_index = 0;
  const auto in_nodes = node->GetInAllNodes();
  for (size_t i = 0; i < in_nodes.size(); ++i) {
    const auto owner_node = in_nodes.at(i);
    if (owner_node->GetType() != DATA) {
      return false;
    }

    uint32_t parent_index = 0;
    if (!AttrUtils::GetInt(owner_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      return false;
    }

    // Input Data feed other Node, need add new Data.
    inputs[i] = parent_index;
    if ((out_index == outputs.size()) && owner_node->GetOutDataNodes().empty()) {
      outputs[out_index] = parent_index;
      ++out_index;
    }
  }

  return true;
}

///
/// @ingroup ge
/// @brief Get all Data nodes for all subgraph.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] data_base: Data Node for migration.
/// @param [in] data_idx: Data groups of subgraph.
/// @param [in] data_idx: Data groups of subgraph.
/// @return true: Same / false: not same
///
bool SubgraphConstMigrationPass::IsParallelNodeSame(const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_datas,
                                                    const NodePtr &const_node, uint32_t parent_index, size_t index) {
  auto it = graph_datas.begin();
  for (++it; it != graph_datas.end(); ++it) {
    const auto &data_nodes = it->second;
    auto data_it = data_nodes.find(parent_index);
    if (data_it == data_nodes.end()) {
      GELOGE(FAILED, "Data: %s not fount, index: %u", const_node->GetName().c_str(), parent_index);
      return false;
    }

    const auto &work_data = data_it->second;
    const auto &out_anchor = work_data->GetOutControlAnchor();
    const auto &in_anchors = out_anchor->GetPeerInControlAnchors();
    if (in_anchors.size() <= index || in_anchors.at(index) == nullptr) {
      GELOGW("Node anchors not same, Data: %s -> %s anchor size: %zu, index: %zu",
             work_data->GetName().c_str(), const_node->GetName().c_str(), in_anchors.size(), index);
      return false;
    }

    const auto &in_anchor = in_anchors.at(index);
    const auto &work_node = in_anchor->GetOwnerNode();
    if (work_node == nullptr) {
      GELOGE(FAILED, "Data: %s not found, parent: %u, index: %zu", const_node->GetName().c_str(), parent_index, index);
      return false;
    }

    if (!IsSameOpNode(const_node, work_node)) {
      GELOGI("OpDesc not same: %s %s, parent: %u, index: %zu",
             const_node->GetName().c_str(), work_node->GetName().c_str(), parent_index, index);
      return false;
    }
  }

  return true;
}

///
/// @ingroup ge
/// @brief Migration subgraph Node to Root
/// @param [in] graph: Root compute graph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] data_base: Data Node for migration.
/// @param [in] data_idx: Data groups of subgraph.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::GraphNodeMigration(const ComputeGraphPtr &graph, const NodePtr &func_node,
                                                      map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_datas,
                                                      const NodePtr &data_node, uint32_t parent_index) {
  bool can_extrapolation = false;
  do {
    can_extrapolation = false;
    const auto &out_anchor = data_node->GetOutControlAnchor();
    const auto &in_anchors = out_anchor->GetPeerInControlAnchors();
    for (size_t i = in_anchors.size(); i > 0; --i) {
      const auto &in_anchor = in_anchors.at(i - 1);
      const auto &work_node = in_anchor->GetOwnerNode();
      GELOGD("Data: %s, node: %s, parent: %u, index: %zu",
             data_node->GetName().c_str(), work_node->GetName().c_str(), parent_index, i);
      if (work_node->GetType() != CONSTANT) {
        continue;
      }

      // Get associated Data, if Data feed other nodes, need append new Data.
      map<uint32_t, uint32_t> inputs;
      map<uint32_t, uint32_t> outputs;
      if (!GetAssociatedNodes(work_node, inputs, outputs)) {
        continue;
      }

      if (!IsParallelNodeSame(graph_datas, work_node, parent_index, i - 1)) {
        continue;
      }

      GELOGI("Move node: %s, parent: %u, index: %zu", work_node->GetName().c_str(), parent_index, i);
      if (AppendParallelNode(graph_datas, func_node, outputs) != SUCCESS) {
        return FAILED;
      }

      if (MoveNodeToParent(graph, func_node, graph_datas, parent_index, i - 1, inputs, outputs) != SUCCESS) {
        return FAILED;
      }
      can_extrapolation = true;
      break;
    }
  } while (can_extrapolation);

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Append Input Tensor for functional node.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] outputs: Parent index of Node output.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::AppendParallelNode(map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_datas,
                                                      const NodePtr &func_node, map<uint32_t, uint32_t> &outputs) {
  // If outputs index invalid, add Data and Input Tensor.
  for (auto &item : outputs) {
    if (item.second != kInvalidParent) {
      continue;
    }

    // Add Data to subgraph.
    map<ComputeGraphPtr, uint32_t> append_num;
    for (auto &groups : graph_datas) {
      const auto &subgraph = groups.first;
      auto &data_nodes = groups.second;

      item.second = func_node->GetAllInDataAnchorsSize() + append_num[subgraph]; // Update to valid parent index.
      const auto data_name = subgraph->GetName() + "_data_" + std::to_string(item.second);

      OpDescBuilder op_builder(data_name, DATA);
      const OpDescPtr op_desc = op_builder.AddInput("x").AddOutput("y").Build();
      if (op_desc == nullptr) {
        GELOGE(OUT_OF_MEMORY, "Create multi-batch subgraph data desc failed");
        return OUT_OF_MEMORY;
      }

      uint32_t data_index = item.second - kCaseInputBase;
      if (!AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, data_index)) {
        GELOGE(FAILED, "Parent index not found, name: %s", op_desc->GetName().c_str());
        return FAILED;
      }

      if (!AttrUtils::SetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, item.second)) {
        GELOGE(FAILED, "Parent index not found, name: %s", op_desc->GetName().c_str());
        return FAILED;
      }

      append_num[subgraph]++;
      data_nodes[item.second] = subgraph->AddNode(op_desc);
      GELOGI("Add Node: %s, parent index: %u", op_desc->GetName().c_str(), item.second);
    }

    // Add InputTensor to functional Node.
    NodeUtils::AppendInputAnchor(func_node, item.second + 1);
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Delete Node from all subgraph.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] detach: Node will move to parent.
/// @param [in] outputs: Parent index of Node output.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::DetachParallelNode(const map<uint32_t, NodePtr> &graph_datas, const NodePtr &detach,
                                                      const map<uint32_t, uint32_t> &outputs) {
  // Break Data and Move node.
  const auto &in_anchor = detach->GetInControlAnchor();
  const auto &out_anchors = in_anchor->GetPeerOutControlAnchors();
  for (size_t i = out_anchors.size(); i > 0; --i) {
    const auto &out_anchor = out_anchors.at(i - 1);
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, in_anchor), "Remove edge failed");
    const auto &owner_node = out_anchor->GetOwnerNode();
    GELOGI("Remove Edge: %s %s", owner_node->GetName().c_str(), detach->GetName().c_str());
  }

  // Break Move and follow, Link Data and follow.
  for (uint32_t i = 0; i < detach->GetAllOutDataAnchorsSize(); ++i) {
    auto it_idx = outputs.find(i);
    if (it_idx == outputs.end()) {
      GELOGE(FAILED, "Node: %s parent index %u not found", detach->GetName().c_str(), i);
      return FAILED;
    }

    auto it_data = graph_datas.find(it_idx->second);
    if (it_data == graph_datas.end()) {
      GELOGE(FAILED, "Node: %s parent index %u not found", detach->GetName().c_str(), i);
      return FAILED;
    }

    const auto &data_node = it_data->second;
    const auto &out_anchor = detach->GetOutDataAnchor(i);

    const auto &out_desc = detach->GetOpDesc()->GetOutputDesc(i);
    const auto &data_desc = data_node->GetOpDesc();
    (void)data_desc->UpdateInputDesc(kDataOutIndex, out_desc);    // Set Data Input to new connect Node.
    (void)data_desc->UpdateOutputDesc(kDataOutIndex, out_desc);   // Set Data Output to new connect Node.

    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (in_anchor == nullptr) {
          continue;
      }
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, in_anchor), "Remove edge failed");
      const auto &owner_node = in_anchor->GetOwnerNode();
      GELOGI("Remove Edge: %s %s", detach->GetName().c_str(), owner_node->GetName().c_str());

      const auto &data_out_anchor = data_node->GetOutDataAnchor(kDataOutIndex);
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(data_out_anchor, in_anchor), "Add edge failed");
      GELOGI("Add Edge: %s %s", data_node->GetName().c_str(), owner_node->GetName().c_str());
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Move Node to Parent Graph.
/// @param [in] graph: Parent compute graph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] attach: Node will move to parent.
/// @param [in] inputs: Parent index of Node input.
/// @param [in] outputs: Parent index of Node output.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::AttachParallelNode(const ComputeGraphPtr &graph, const NodePtr &func_node,
                                                      const NodePtr &attach, const map<uint32_t, uint32_t> &inputs,
                                                      const map<uint32_t, uint32_t> &outputs) {
  GE_CHECK_NOTNULL(attach);
  for (const auto item : inputs) {
    if (item.second == kInvalidParent) {   // Not connect, Skip.
      continue;
    }

    const auto &in_anchor = func_node->GetInDataAnchor(item.second);
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    const auto &owner_node = out_anchor->GetOwnerNode();
    const auto &in_control = attach->GetInControlAnchor();
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(owner_node->GetOutControlAnchor(), in_control), "Add edge failed");
    GELOGI("Add Edge: %s %s", owner_node->GetName().c_str(), attach->GetName().c_str());
  }

  for (const auto &item : outputs) {
    const auto &func_desc = func_node->GetOpDesc();
    const auto &out_desc = attach->GetOpDesc()->GetOutputDesc(item.second);
    (void)func_desc->UpdateInputDesc(item.second, out_desc);    // Set Data Input to new connect Node.

    const auto &in_anchor = func_node->GetInDataAnchor(item.second);
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor != nullptr) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, in_anchor), "Remove edge failed");
      const auto &owner_node = out_anchor->GetOwnerNode();
      GELOGI("Remove Edge: %s %s", owner_node->GetName().c_str(), func_node->GetName().c_str());
    }
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(attach->GetOutDataAnchor(item.first), in_anchor), "Add edge failed");
    GELOGI("Add Edge: %s %s", attach->GetName().c_str(), func_node->GetName().c_str());
  }

  (void)graph->AddNode(attach);
  (void)attach->SetOwnerComputeGraph(graph);
  GELOGI("Add Node: %s %s", graph->GetName().c_str(), attach->GetName().c_str());
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Move node to Parent graph.
/// @param [in] graph: Root compute graph.
/// @param [in] func_node: functional Node of Case.
/// @param [in] graph_nodes: Data groups of subgraph.
/// @param [in] index: anchor index of move Node.
/// @param [in] inputs: Parent index of Node input.
/// @param [in] outputs: Parent index of Node output.
/// @return 0: SUCCESS / others: FAILED
///
Status SubgraphConstMigrationPass::MoveNodeToParent(const ComputeGraphPtr &graph, const NodePtr &func_node,
                                                    const map<ComputeGraphPtr, map<uint32_t, NodePtr>> &graph_datas,
                                                    uint32_t parent_index, uint32_t index,
                                                    const map<uint32_t, uint32_t> &inputs,
                                                    const map<uint32_t, uint32_t> &outputs) {
  if (inputs.empty()) {
    GELOGE(FAILED, "Graph: %s, inputs is empty", graph->GetName().c_str());
    return FAILED;
  }

  NodePtr move_node;
  for (auto &groups : graph_datas) {
    const auto &subgraph = groups.first;
    const auto &data_nodes = groups.second;
    auto it = data_nodes.find(parent_index);
    if (it == data_nodes.end()) {
      GELOGE(FAILED, "Graph: %s, Data: %u node not found", subgraph->GetName().c_str(), parent_index);
      return FAILED;
    }

    const auto &base_data = it->second;
    const auto &out_anchor = base_data->GetOutControlAnchor();
    const auto &in_anchors = out_anchor->GetPeerInControlAnchors();
    if (in_anchors.size() <= index || in_anchors.at(index) == nullptr) {
      GELOGE(FAILED, "Data: %s, anchor size: %zu, index: %u not found",
             base_data->GetName().c_str(), in_anchors.size(), index);
      return FAILED;
    }

    const auto &in_anchor = in_anchors.at(index);
    move_node = in_anchor->GetOwnerNode();
    if (move_node == nullptr) {
      GELOGE(FAILED, "Data: %s not found, index: %u", base_data->GetName().c_str(), parent_index);
      return FAILED;
    }

    if (DetachParallelNode(data_nodes, move_node, outputs) != SUCCESS) {
      GELOGE(FAILED, "Data: %s not found, index: %u", base_data->GetName().c_str(), parent_index);
      return FAILED;
    }

    GE_CHK_GRAPH_STATUS_RET(subgraph->RemoveNode(move_node), "Remove node failed");
    GELOGI("Remove Node: %s %s", subgraph->GetName().c_str(), move_node->GetName().c_str());
  }

  if (AttachParallelNode(graph, func_node, move_node, inputs, outputs) != SUCCESS) {
    return FAILED;
  }

  migration_append_ = true;
  return SUCCESS;
}
}  // namespace ge
