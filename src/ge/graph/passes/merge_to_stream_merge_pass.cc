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

#include "graph/passes/merge_to_stream_merge_pass.h"
#include "common/ge/ge_util.h"
#include "ge/ge_api_types.h"
#include "graph/common/omg_util.h"

namespace ge {
Status MergeToStreamMergePass::Run(ComputeGraphPtr graph) {
  GELOGD("MergeToStreamMergePass Enter");

  bypass_nodes_.clear();
  for (const auto &node : graph->GetDirectNode()) {
    if ((node->GetType() != MERGE) && (node->GetType() != REFMERGE)) {
      continue;
    }

    OpDescPtr merge_op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(merge_op_desc);
    if (merge_op_desc->HasAttr(ATTR_INSERT_BY_MBATCH)) {
      GE_CHK_STATUS_RET(AddMemcpyAsyncNodes(graph, node, true), "Merge add memcpy node failed.");
      GE_CHK_STATUS_RET(SetStreamLabel(node, node->GetName()), "Set stream label failed");
    } else {
      GE_CHK_STATUS_RET(ReplaceMergeNode(graph, node), "Add StreamMerge node failed.");
    }
  }

  for (const auto &node : bypass_nodes_) {
    GE_CHK_BOOL_EXEC(GraphUtils::RemoveNodeWithoutRelink(graph, node) == GRAPH_SUCCESS, return FAILED,
                     "Remove merge node failed.");
  }

  GELOGD("MergeToStreamMergePass Leave");
  return SUCCESS;
}

///
/// @brief Replace Merge Op
/// @param [in] graph
/// @param [in] merge_node
/// @return Status
///
Status MergeToStreamMergePass::ReplaceMergeNode(const ComputeGraphPtr &graph, const NodePtr &merge_node) {
  OpDescPtr merge_op_desc = merge_node->GetOpDesc();
  GE_CHECK_NOTNULL(merge_op_desc);

  const std::string &node_name = merge_node->GetName();
  GELOGI("Create StreamMerge Op, name=%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, STREAMMERGE);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc failed, StreamMerge:%s.", node_name.c_str());
    return FAILED;
  }

  for (const InDataAnchorPtr &in_anchor : merge_node->GetAllInDataAnchors()) {
    GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(merge_op_desc->GetInputDesc(in_anchor->GetIdx())) == GRAPH_SUCCESS,
                     return FAILED, "Create StreamMerge op: add input desc failed.");
  }

  for (const OutDataAnchorPtr &out_anchor : merge_node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(merge_op_desc->GetOutputDesc(out_anchor->GetIdx())) == GRAPH_SUCCESS,
                     return FAILED, "Create StreamMerge op: add output desc failed.");
  }

  NodePtr stream_merge = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(stream_merge != nullptr, return FAILED, "Insert StreamMerge node failed.");
  GE_CHK_STATUS_RET(MoveEdges(merge_node, stream_merge), "Move edges failed.");
  bypass_nodes_.insert(merge_node);

  if (merge_op_desc->HasAttr(ATTR_NAME_NEXT_ITERATION)) {
    std::string next_iteration_name;
    GE_IF_BOOL_EXEC(!AttrUtils::GetStr(merge_op_desc, ATTR_NAME_NEXT_ITERATION, next_iteration_name),
                    GELOGE(INTERNAL_ERROR, "Get ATTR_NAME_NEXT_ITERATION failed");
                    return INTERNAL_ERROR);
    GE_CHK_STATUS_RET(SetNextIteration(stream_merge, next_iteration_name), "Set next iteration failed");
  }

  return AddMemcpyAsyncNodes(graph, stream_merge, false);
}

///
/// @brief Add MemcpyAsync Op as StreamMerge in_node
/// @param [in] graph
/// @param [in] node
/// @param [in] multi_batch_flag
/// @return Status
///
Status MergeToStreamMergePass::AddMemcpyAsyncNodes(const ComputeGraphPtr &graph, const NodePtr &node,
                                                   bool multi_batch_flag) {
  GE_CHK_BOOL_EXEC(node != nullptr, return FAILED, "Param of pre node is null.");
  for (const InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    const std::string &type = in_node->GetType();
    // For WhileLoop no need memcpy & active for merge.
    GE_IF_BOOL_EXEC((type == ENTER) || (type == REFENTER) || (type == NEXTITERATION) || (type == REFNEXTITERATION),
                    continue);

    const std::string &memcpy_name = node->GetName() + "_input_" + std::to_string(in_data_anchor->GetIdx());
    NodePtr memcpy_node = CreateMemcpyAsyncNode(graph, memcpy_name, peer_out_anchor, multi_batch_flag);
    GE_CHK_BOOL_EXEC(memcpy_node != nullptr, return FAILED, "Create MemcpyAsync node failed.");
    GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor), "MemcpyAsync node remove edge failed.");
    GE_CHK_STATUS(GraphUtils::AddEdge(peer_out_anchor, memcpy_node->GetInDataAnchor(0)),
                  "MemcpyAsync node add edge failed.");
    GE_CHK_STATUS(GraphUtils::AddEdge(memcpy_node->GetOutDataAnchor(0), in_data_anchor),
                  "MemcpyAsync node add edge failed.");

    NodePtr active_node = CreateActiveNode(graph, memcpy_node);
    GE_CHK_BOOL_EXEC(active_node != nullptr, return FAILED, "Create StreamActive node failed.");
    GE_CHK_STATUS(GraphUtils::AddEdge(active_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                  "StreamActive add ctrl edge failed.");
    if (SetActiveLabelList(active_node, {node->GetName()}) != SUCCESS) {
      GELOGE(FAILED, "SetActiveLabelList for node %s failed.", active_node->GetName().c_str());
      return FAILED;
    }
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
NodePtr MergeToStreamMergePass::CreateMemcpyAsyncNode(const ComputeGraphPtr &graph, const std::string &name,
                                                      const OutDataAnchorPtr &out_data_anchor, bool multi_batch_flag) {
  GE_CHK_BOOL_EXEC(out_data_anchor != nullptr, return nullptr, "Param of input node is null.");
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
                   return nullptr, "Create MemcpyAsync op: add input desc failed.");
  GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx())) == GRAPH_SUCCESS,
                   return nullptr, "Create MemcpyAsync op: add output desc failed.");

  return graph->AddNode(op_desc);
}

///
/// @brief Create Active Op
/// @param [in] graph
/// @param [in] node
/// @return ge::NodePtr
///
NodePtr MergeToStreamMergePass::CreateActiveNode(const ComputeGraphPtr &graph, const NodePtr &node) {
  const std::string &node_name = node->GetName() + "_" + STREAMACTIVE;
  GELOGI("Create StreamActive op:%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, STREAMACTIVE);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc failed, StreamActive:%s.", node_name.c_str());
    return nullptr;
  }

  NodePtr active_node = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(active_node != nullptr, return nullptr, "Create StreamActive node failed.");
  GE_IF_BOOL_EXEC(GraphUtils::AddEdge(node->GetOutControlAnchor(), active_node->GetInControlAnchor()) != SUCCESS,
                  GELOGE(INTERNAL_ERROR, "add edge failed");
                  return nullptr);
  GE_IF_BOOL_EXEC(SetSwitchBranchNodeLabel(active_node, node_name) != SUCCESS,
                  GELOGE(INTERNAL_ERROR, "set switch branch node label failed");
                  return nullptr);

  return active_node;
}

///
/// @brief move edges from old_node to new_node
/// @param [in] old_node
/// @param [in] new_node
/// @return Status
///
Status MergeToStreamMergePass::MoveEdges(const NodePtr &old_node, const NodePtr &new_node) {
  for (const InDataAnchorPtr &in_data_anchor : old_node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

    GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor), "Merge remove in data edge failed.");
    GE_CHK_STATUS(GraphUtils::AddEdge(peer_out_anchor, new_node->GetInDataAnchor(in_data_anchor->GetIdx())),
                  "StreamMerge add in data edge failed.");
  }

  for (const OutDataAnchorPtr &out_data_anchor : old_node->GetAllOutDataAnchors()) {
    for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHK_STATUS(GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor), "Merge remove out data edge failed.");
      GE_CHK_STATUS(GraphUtils::AddEdge(new_node->GetOutDataAnchor(out_data_anchor->GetIdx()), peer_in_anchor),
                    "StreamMerge add out data edge failed.");
    }
  }

  for (const NodePtr &in_ctrl_node : old_node->GetInControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(in_ctrl_node->GetOutControlAnchor(), old_node->GetInControlAnchor()),
                  "Merge remove in ctrl edge failed.");
    GE_CHK_STATUS(GraphUtils::AddEdge(in_ctrl_node->GetOutControlAnchor(), new_node->GetInControlAnchor()),
                  "StreamMerge add in ctrl edge failed.");
  }

  for (const NodePtr &out_ctrl_node : old_node->GetOutControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(old_node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor()),
                  "Merge remove out ctrl edge failed.");
    GE_CHK_STATUS(GraphUtils::AddEdge(new_node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor()),
                  "StreamMerge add out ctrl edge failed.");
  }

  return SUCCESS;
}
}  // namespace ge
