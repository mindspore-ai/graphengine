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

#include "graph/passes/replace_transshape_pass.h"

#include <string>

#include "common/ge/ge_util.h"
#include "common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/common/omg_util.h"
#include "graph/utils/graph_utils.h"

namespace ge {
Status ReplaceTransShapePass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == TRANSSHAPE) {
      auto ret = ReplaceTransShapeNode(graph, node);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "Trans shape node %s failed", node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status ReplaceTransShapePass::ReplaceTransShapeNode(ComputeGraphPtr &graph, NodePtr &trans_shape_node) {
  std::string op_type;
  auto ret = GetOriginalType(trans_shape_node, op_type);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Get node %s original type failede", trans_shape_node->GetName().c_str());
    return FAILED;
  }
  auto src_op_desc = trans_shape_node->GetOpDesc();
  GE_CHECK_NOTNULL(src_op_desc);

  std::string node_name = trans_shape_node->GetName() + "ToMemcpy";
  auto dst_op_desc = MakeShared<OpDesc>(node_name, MEMCPYASYNC);
  if (dst_op_desc == nullptr) {
    GELOGE(FAILED, "Make node %s opdesc failed", node_name.c_str());
    return FAILED;
  }
  GELOGI("Create memcpy Op, name=%s.", node_name.c_str());
  for (InDataAnchorPtr &in_anchor : trans_shape_node->GetAllInDataAnchors()) {
    auto ret = dst_op_desc->AddInputDesc(src_op_desc->GetInputDesc(in_anchor->GetIdx()));
    if (ret != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add input desc failed");
      return FAILED;
    }
  }
  for (OutDataAnchorPtr &out_anchor : trans_shape_node->GetAllOutDataAnchors()) {
    auto ret = dst_op_desc->AddOutputDesc(src_op_desc->GetOutputDesc(out_anchor->GetIdx()));
    if (ret != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add output desc failed");
      return FAILED;
    }
  }
  NodePtr memcpy_node = graph->AddNode(dst_op_desc);
  GE_CHECK_NOTNULL(memcpy_node);

  for (InDataAnchorPtr &in_data_anchor : trans_shape_node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

    GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor), "Remove Memcpy data input fail.");
    GE_CHK_STATUS(GraphUtils::AddEdge(peer_out_anchor, memcpy_node->GetInDataAnchor(in_data_anchor->GetIdx())),
                  "Memcpy node add edge fail.");
  }

  for (OutDataAnchorPtr &out_data_anchor : trans_shape_node->GetAllOutDataAnchors()) {
    for (InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHK_STATUS(GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor), "Remove Memcpy data output fail.");
      GE_CHK_STATUS(GraphUtils::AddEdge(memcpy_node->GetOutDataAnchor(out_data_anchor->GetIdx()), peer_in_anchor),
                    "Memcpy node add edge fail.");
    }
  }
  ReplaceControlEdges(trans_shape_node, memcpy_node);
  return SUCCESS;
}

void ReplaceTransShapePass::CopyControlEdges(NodePtr &old_node, NodePtr &new_node, bool input_check_flag) {
  GE_CHECK_NOTNULL_JUST_RETURN(old_node);
  GE_CHECK_NOTNULL_JUST_RETURN(new_node);
  GE_IF_BOOL_EXEC(old_node == new_node, return );
  for (NodePtr &node : old_node->GetInControlNodes()) {
    auto out_control_anchor = node->GetOutControlAnchor();
    GE_IF_BOOL_EXEC(!out_control_anchor->IsLinkedWith(new_node->GetInControlAnchor()), {
      GE_CHK_STATUS(GraphUtils::AddEdge(out_control_anchor, new_node->GetInControlAnchor()), "Add in ctl edge fail.");
    });
  }

  for (NodePtr &node : old_node->GetOutControlNodes()) {
    GE_IF_BOOL_EXEC(!new_node->GetOutControlAnchor()->IsLinkedWith(node->GetInControlAnchor()), {
      GE_CHK_STATUS(GraphUtils::AddEdge(new_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                    "Add out ctl edge fail.");
    });
  }
}

void ReplaceTransShapePass::RemoveControlEdges(NodePtr &node) {
  GE_CHECK_NOTNULL_JUST_RETURN(node);
  for (NodePtr &in_node : node->GetInControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(in_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                  "Remove in ctl edge fail.");
  }

  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    for (auto &in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
      GE_CHK_STATUS(GraphUtils::RemoveEdge(out_data_anchor, in_ctrl_anchor), "Remove in ctl edge fail.");
    }
  }

  auto out_control_anchor = node->GetOutControlAnchor();
  GE_CHECK_NOTNULL_JUST_RETURN(out_control_anchor);
  for (auto &peer_anchor : out_control_anchor->GetPeerAnchors()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(out_control_anchor, peer_anchor), "Remove out ctl edge fail.");
  }
}

void ReplaceTransShapePass::ReplaceControlEdges(NodePtr &old_node, NodePtr &new_node) {
  GE_IF_BOOL_EXEC(old_node == new_node, return );
  CopyControlEdges(old_node, new_node);
  RemoveControlEdges(old_node);
}
}  // namespace ge
