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

#include "graph/passes/merge_input_memcpy_pass.h"

#include <queue>

#include "common/ge/ge_util.h"
#include "ge/ge_api_types.h"
#include "graph/common/omg_util.h"

namespace ge {
namespace {
const std::set<std::string> kLoopMergeInputs{
    ENTER, REFENTER, NEXTITERATION, REFNEXTITERATION
};
}
Status MergeInputMemcpyPass::Run(ComputeGraphPtr graph) {
  GELOGD("MergeInputMemcpyPass Enter");
  std::unordered_map<NodePtr, std::vector<NodePtr>> switch_groups;
  for (const auto &node : graph->GetDirectNode()) {
    std::string type;
    GE_CHK_STATUS_RET(GetOriginalType(node, type), "Get node type failed.");
    if ((type != MERGE) && (type != REFMERGE)) {
      continue;
    }

    GE_CHECK_NOTNULL(node->GetOpDesc());
    GE_CHK_STATUS_RET(AddMemcpyAsyncNodes(graph, node, node->GetOpDesc()->HasAttr(ATTR_INSERT_BY_MBATCH)),
                      "Merge add memcpy node failed.");
    CollectSwitchGroup(node, switch_groups);
  }

  MarkUnknownForSwitch(switch_groups);
  GELOGD("MergeInputMemcpyPass Leave");
  return SUCCESS;
}

///
/// @brief Add MemcpyAsync Op as Merge in_node
/// @param [in] graph
/// @param [in] node
/// @param [in] multi_batch_flag
/// @return Status
///
Status MergeInputMemcpyPass::AddMemcpyAsyncNodes(const ComputeGraphPtr &graph, const NodePtr &node,
                                                 bool multi_batch_flag) {
  for (const InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    const std::string &type = in_node->GetType();
    // For WhileLoop no need memcpy for merge.
    GE_IF_BOOL_EXEC((type == ENTER) || (type == REFENTER) || (type == NEXTITERATION) || (type == REFNEXTITERATION),
                    continue);

    const std::string &memcpy_name = node->GetName() + "_input_" + std::to_string(in_data_anchor->GetIdx());
    NodePtr memcpy_node = CreateMemcpyAsyncNode(graph, memcpy_name, peer_out_anchor, multi_batch_flag);
    GE_CHK_BOOL_EXEC(memcpy_node != nullptr, return FAILED, "Create MemcpyAsync node failed.");
    GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor),
                  "MemcpyAsync node remove edge failed.");
    GE_CHK_STATUS(GraphUtils::AddEdge(peer_out_anchor, memcpy_node->GetInDataAnchor(0)),
                  "MemcpyAsync node add edge failed.");
    GE_CHK_STATUS(GraphUtils::AddEdge(memcpy_node->GetOutDataAnchor(0), in_data_anchor),
                  "MemcpyAsync node add edge failed.");
  }

  return SUCCESS;
}

///
/// @brief Add MemcpyAsync Node
/// @param [in] graph
/// @param [in] name
/// @param [in] out_data_anchor
/// @param [in] multi_batch_flag
/// @return ge::NodePtr
///
NodePtr MergeInputMemcpyPass::CreateMemcpyAsyncNode(const ComputeGraphPtr &graph, const std::string &name,
                                                    const OutDataAnchorPtr &out_data_anchor, bool multi_batch_flag) {
  OpDescPtr pre_op_desc = out_data_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHK_BOOL_EXEC(pre_op_desc != nullptr, return nullptr, "OpDesc of pre node is invalid.");

  const std::string &memcpy_type = multi_batch_flag ? MEMCPYADDRASYNC : MEMCPYASYNC;
  const std::string &node_name = name + "_" + memcpy_type;
  GELOGI("Create MemcpyAsync op:%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, memcpy_type);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc failed, MemcpyAsync:%s.", node_name.c_str());
    return nullptr;
  }

  GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx())) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add input to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr, "Create MemcpyAsync op: add input desc failed.");
  GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx())) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add output to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr, "Create MemcpyAsync op: add output desc failed.");

  return graph->AddNode(op_desc);
}

///
/// @brief Mark force unknown shape for Switch node
/// @param [in] merge node
/// @param [out] switch_groups
/// @return
///
void MergeInputMemcpyPass::CollectSwitchGroup(const NodePtr &node,
                                              std::unordered_map<NodePtr, std::vector<NodePtr>> &switch_groups) {
  const auto &op_desc = node->GetOpDesc();
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    const auto &src_out_anchor = in_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr) {
      continue;
    }

    std::string node_type;
    GetOriginalType(src_out_anchor->GetOwnerNode(), node_type);
    if (kLoopMergeInputs.count(node_type) > 0) {
      return;
    }
  }

  // Switch --> {Switch --> Merge} --> Merge
  std::queue<std::pair<NodePtr, uint32_t>> search_queue;
  search_queue.push({node, 0});
  std::vector<NodePtr> &switch_group = switch_groups[node];
  while (!search_queue.empty()) {
    const auto dst_node = search_queue.front().first;
    const auto dst_span = search_queue.front().second;
    search_queue.pop();

    // Switch --> Identity --> Constant
    for (const auto &in_ctrl_node : dst_node->GetInControlNodes()) {
      if (in_ctrl_node->GetType() == IDENTITY) {
        GELOGD("Travel node: %s, In control: %s, span is: %u",
               dst_node->GetName().c_str(), in_ctrl_node->GetName().c_str(), dst_span);
        search_queue.push({in_ctrl_node, dst_span});
      }
    }

    for (const auto &in_data_node : dst_node->GetInDataNodes()) {
      std::string node_type;
      GetOriginalType(in_data_node, node_type);
      GELOGD("Travel node: %s, %s node: %s, span is: %u",
             dst_node->GetName().c_str(), node_type.c_str(), in_data_node->GetName().c_str(), dst_span);
      if (node_type == SWITCH || node_type == REFSWITCH) {
        if (dst_span > 0) {
          search_queue.push({in_data_node, dst_span - 1});
        } else {
          switch_group.emplace_back(in_data_node);
        }
      } else if (node_type == MERGE || node_type == REFMERGE) {
        search_queue.push({in_data_node, dst_span + 1});
      } else {
        search_queue.push({in_data_node, dst_span});
      }
    }
  }

  if (IsUnknownShapeTensor(op_desc->GetOutputDesc(0)) || op_desc->HasAttr(ATTR_NAME_FORCE_UNKNOWN_SHAPE)) {
    GELOGI("Mark [%s] as for unknown shape, switch groups: %zu", node->GetName().c_str(), switch_groups.size());
    MarkForceUnknownShape(node, true);
    for (const auto &n : switch_group) {
      MarkForceUnknownShape(n, true);
    }
  }
}

void MergeInputMemcpyPass::MarkUnknownForSwitch(const std::unordered_map<NodePtr, std::vector<NodePtr>> &switch_groups) {
  std::function<bool(const NodePtr &)> callback = [](const NodePtr &n) {
    return n->GetOpDesc()->HasAttr(ATTR_NAME_FORCE_UNKNOWN_SHAPE);
  };

  for (const auto &item : switch_groups) {
    const auto &node = item.first;
    if (node->GetOpDesc()->HasAttr(ATTR_NAME_FORCE_UNKNOWN_SHAPE)) {
      continue;
    }

    const std::vector<NodePtr> &switch_group = item.second;
    if (std::any_of(switch_group.begin(), switch_group.end(), callback)) {
      GELOGI("Mark [%s] as force unknown shape, switch nodes: %zu", node->GetName().c_str(), switch_group.size());
      MarkForceUnknownShape(node, true);
      for (const auto &n : switch_group) {
        MarkForceUnknownShape(n, true);
      }
    }
  }
}
}  // namespace ge
