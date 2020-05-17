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

#include "graph/passes/iterator_op_pass.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "graph/common/omg_util.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/passes/pass_utils.h"

using domi::MEMCPYASYNC;

namespace ge {
const char *const kGetNext = "GetNext";

Status IteratorOpPass::Run(ge::ComputeGraphPtr graph) {
  GELOGD("GetNextOpPass begin");
  GE_CHECK_NOTNULL(graph);
  if (!PassUtils::IsNeedTrainIteFlowCtrl(graph)) {
    return SUCCESS;
  }
  std::string type;
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    GE_CHK_STATUS_RET(GetOriginalType(node, type));
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const string op_type = op_desc->GetType();
    if (type == "IteratorV2" || type == "Iterator" || op_type == kGetNext) {
      ge::NodePtr memcpy_async_node = InsertMemcpyAsyncNode(node, graph);
      GE_CHECK_NOTNULL(memcpy_async_node);
      GE_CHK_STATUS_RET(SetCycleEvent(memcpy_async_node), "Set cycle event fail, node:%s",
                        memcpy_async_node->GetName().c_str());

      GE_CHK_STATUS_RET(SetStreamLabel(memcpy_async_node, memcpy_async_node->GetName()),
                        "Set stream label fail, node:%s", node->GetName().c_str());

      GE_CHK_STATUS_RET(SetStreamLabel(node, node->GetName()), "Set stream label fail, node:%s",
                        node->GetName().c_str());

      GELOGI("Set independent loop for iterator node success");
    }
  }
  GELOGD("GetNextOpPass end");
  return SUCCESS;
}

///
/// @brief insert memcpy after GetNext
///
/// @param pre_node
/// @param graph
/// @return ge::NodePtr
///
ge::NodePtr IteratorOpPass::InsertMemcpyAsyncNode(const ge::NodePtr &pre_node, const ge::ComputeGraphPtr &graph) {
  GE_CHK_BOOL_EXEC(pre_node != nullptr, GELOGW("Pre node is null."); return nullptr);
  GE_CHK_BOOL_EXEC(graph != nullptr, GELOGW("graph is null."); return nullptr);
  ge::OpDescPtr memcpy_async_op_desc = CreateMemcpyAsyncOp(pre_node);
  GE_CHK_BOOL_EXEC(memcpy_async_op_desc != nullptr, GELOGW("Create memcpyAsync op fail."); return nullptr);
  ge::NodePtr memcpy_async_node = graph->AddNode(memcpy_async_op_desc);
  GE_CHK_BOOL_EXEC(memcpy_async_node != nullptr, return nullptr, "Insert mencpy node fail.");

  // Data out
  for (auto &out_anchor : pre_node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    ge::graphStatus status;
    GELOGI("Graph add memcpyAsync op in edge, index:%d.", out_anchor->GetIdx());
    for (auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(peer_in_anchor == nullptr, GELOGW("peer_in_anchor is nullptr"); return nullptr);
      status = GraphUtils::RemoveEdge(out_anchor, peer_in_anchor);
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS, return nullptr, "Remove edge failed, index:%d.", out_anchor->GetIdx());
      status = GraphUtils::AddEdge(memcpy_async_node->GetOutDataAnchor(out_anchor->GetIdx()), peer_in_anchor);
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS, return nullptr,
                       "Graph add memcpyAsync op out edge fail, src index:%d, dst index:%d, dst node: %s.",
                       out_anchor->GetIdx(), peer_in_anchor->GetIdx(),
                       peer_in_anchor->GetOwnerNode()->GetName().c_str());
      GELOGI("Graph add memcpyAsync op out edge, src index:%d, dst index:%d, dst node: %s.", out_anchor->GetIdx(),
             peer_in_anchor->GetIdx(), peer_in_anchor->GetOwnerNode()->GetName().c_str());
    }
    status = GraphUtils::AddEdge(out_anchor, memcpy_async_node->GetInDataAnchor(out_anchor->GetIdx()));
    GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS, return nullptr, "Graph add memcpyAsync op in edge fail, index:%d.",
                     out_anchor->GetIdx());
  }
  // Control out
  OutControlAnchorPtr out_ctrl_anchor = pre_node->GetOutControlAnchor();
  GE_IF_BOOL_EXEC(
    out_ctrl_anchor != nullptr, for (auto &peer_in_ctrl_anchor
                                     : out_ctrl_anchor->GetPeerInControlAnchors()) {
      ge::graphStatus status = GraphUtils::RemoveEdge(out_ctrl_anchor, peer_in_ctrl_anchor);
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS, return nullptr, "Remove edge failed, dst node: %s.",
                       peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
      status = GraphUtils::AddEdge(memcpy_async_node->GetOutControlAnchor(), peer_in_ctrl_anchor);
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS, return nullptr,
                       "Graph add memcpyAsync op out ctrl edge fail, dst node: %s.",
                       peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
      GELOGI("Graph add memcpyAsync op out ctrl edge, dst node: %s.",
             peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    });
  GELOGI("Insert memcpyAsync op success.");

  return memcpy_async_node;
}

///
/// @brief create memcpy
///
/// @param pre_node
/// @return ge::OpDescPtr
///
ge::OpDescPtr IteratorOpPass::CreateMemcpyAsyncOp(const ge::NodePtr &pre_node) {
  GE_CHK_BOOL_EXEC(pre_node != nullptr, return nullptr, "Input param invalid.");

  string node_name = pre_node->GetName() + "_MemcpyAsync";
  ge::OpDescPtr op_desc = MakeShared<OpDesc>(node_name.c_str(), MEMCPYASYNC);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "MakeShared fail.");
    return op_desc;
  }
  GELOGI("Create memcpyAsync op:%s.", op_desc->GetName().c_str());

  ge::OpDescPtr pre_node_op_desc = pre_node->GetOpDesc();
  GE_CHK_BOOL_EXEC(pre_node_op_desc != nullptr, return nullptr, "OpDesc of pre_node is invalid.");

  size_t out_size = pre_node_op_desc->GetOutputsSize();
  GELOGI("Create memcpyAsync op, pre_node out_size: %zu.", out_size);
  for (size_t i = 0; i < out_size; i++) {
    GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(pre_node_op_desc->GetOutputDesc(i)) == GRAPH_SUCCESS, return nullptr,
                     "Create memcpyAsync op:add input desc fail.");
    GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(pre_node_op_desc->GetOutputDesc(i)) == GRAPH_SUCCESS, return nullptr,
                     "Create memcpyAsync op:add output desc fail.");
  }

  return op_desc;
}
}  // namespace ge
