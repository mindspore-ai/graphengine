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

#include "graph/passes/enter_pass.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/utils/graph_utils.h"

namespace {
const size_t kOutNodesNum = 1;
}

namespace ge {
Status EnterPass::Run(NodePtr &node) {
  GELOGD("EnterPass running");
  GE_CHECK_NOTNULL(node);

  if ((node->GetType() != ENTER) && (node->GetType() != REFENTER)) {
    return SUCCESS;
  }

  // enter node has only one input
  if (node->GetInDataNodes().empty()) {
    GELOGE(PARAM_INVALID, "enter_node %s has no input", node->GetName().c_str());
    return PARAM_INVALID;
  }
  NodePtr in_node = node->GetInDataNodes().at(0);
  GE_CHECK_NOTNULL(in_node);

  if ((in_node->GetType() != CONSTANT) && (in_node->GetType() != CONSTANTOP)) {
    return SUCCESS;
  }

  bool need_remove_flag = in_node->GetInControlNodes().empty() && node->GetInControlNodes().empty();
  if (!need_remove_flag) {
    return SUCCESS;
  }
  if (node->GetOutDataNodes().empty()) {
    for (auto &out_ctrl_node : node->GetOutControlNodes()) {
      if (out_ctrl_node == nullptr) {
        continue;
      }
      if (GraphUtils::RemoveEdge(node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Remove Enter ctrl output fail, %s->%s", node->GetName().c_str(),
               out_ctrl_node->GetName().c_str());
        return FAILED;
      }
    }
  } else {
    if (OptimizeEnter(node, in_node) != SUCCESS) {
      GELOGE(FAILED, "Optimize enter node[%s] failed.", node->GetName().c_str());
      return FAILED;
    }
  }

  GELOGD("EnterPass success");
  return SUCCESS;
}

Status EnterPass::OptimizeEnter(NodePtr &node, NodePtr &in_node) {
  auto out_nodes_of_in_node = in_node->GetOutAllNodes();
  if (out_nodes_of_in_node.size() != kOutNodesNum) {
    return SUCCESS;
  }

  if (!node->GetOutControlNodes().empty()) {
    return SUCCESS;
  }

  for (const auto &out_node : node->GetOutDataNodes()) {
    GE_CHECK_NOTNULL(out_node);
    if (out_node->GetType() == MERGE) {
      return SUCCESS;
    }
  }

  GE_CHECK_NOTNULL(in_node->GetOutDataAnchor(0));
  GE_CHK_STATUS_RET(in_node->GetOutDataAnchor(0)->Unlink(node->GetInDataAnchor(0)));
  auto out_data_anchor = node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(out_data_anchor);
  for (auto peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_CHK_STATUS_RET(out_data_anchor->Unlink(peer_in_data_anchor));
    GE_CHK_STATUS_RET(in_node->GetOutDataAnchor(0)->LinkTo(peer_in_data_anchor));
  }

  auto graph = node->GetOwnerComputeGraph();
  GE_CHK_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(graph, node))
  AddRePassNodesWithInOut(in_node);

  return SUCCESS;
}
}  // namespace ge
