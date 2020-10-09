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

#include "graph/passes/unused_op_remove_pass.h"
#include <queue>
#include <set>
#include <string>
#include <vector>
#include "common/debug/log.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/pass_manager.h"
#include "graph/passes/isolated_op_remove_pass.h"

using domi::SUCCESS;

namespace ge {
const std::set<std::string> kRemoveOpSet = {DROPOUT, PERMUTE, UNUSEDCONST, ASSERT};
const std::set<std::string> kOtherRemoveOpSet = {DROPOUT};

Status UnusedOpRemovePass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  std::set<std::string> remove_op_set;
  vector<NodePtr> nodes_to_be_deleted;
  if (fmktype_ == TENSORFLOW) {
    remove_op_set = kRemoveOpSet;
  } else {
    remove_op_set = kOtherRemoveOpSet;
  }

  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string op_type_str = node->GetOpDesc()->GetType();
    if (remove_op_set.count(op_type_str)) {
      if (IsExceptions(node)) {
        continue;
      }
      for (auto &out_anchor : node->GetAllOutDataAnchors()) {
        for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
          NodePtr dst_node = in_anchor->GetOwnerNode();
          GE_CHECK_NOTNULL(dst_node->GetOpDesc());
          int dst_index = in_anchor->GetIdx();
          std::vector<bool> list_bool;
          GE_CHECK_NOTNULL(dst_node->GetOpDesc());
          list_bool = dst_node->GetOpDesc()->GetIsInputConst();
          GE_IF_BOOL_EXEC(list_bool.size() == 0, continue);
          list_bool.erase(list_bool.begin() + dst_index);
          dst_node->GetOpDesc()->SetIsInputConst(list_bool);
        }
      }
      if (op_type_str == ASSERT) {
        GE_CHK_STATUS_RET(CollectParentNode(graph, node, nodes_to_be_deleted), "remove node failed");
      } else {
        GE_CHK_STATUS_RET(graph->RemoveNode(node), "remove node failed");
      }
    }
  }
  for (auto &node : nodes_to_be_deleted) {
    for (InDataAnchorPtr &inAnchor : node->GetAllInDataAnchors()) {
      inAnchor->UnlinkAll();
    }
    for (OutDataAnchorPtr &outAnchorPtr : node->GetAllOutDataAnchors()) {
      outAnchorPtr->UnlinkAll();
    }
    if (node->GetOutControlAnchor() != nullptr) {
      node->GetOutControlAnchor()->UnlinkAll();
    }
    GE_CHK_STATUS_RET(graph->RemoveNode(node), "remove node:%s failed", node->GetName().c_str());
  }

  return SUCCESS;
}

Status UnusedOpRemovePass::CollectParentNode(const ComputeGraphPtr &graph, const NodePtr &node,
                                             vector<NodePtr> &node_vec) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(node);
  node_vec.push_back(node);
  std::queue<NodePtr> node_queue;

  for (auto &src_node : node->GetInDataNodes()) {
    if (src_node->GetOutDataNodesSize() == 1) {
      node_queue.push(src_node);
    }
  }

  while (!node_queue.empty()) {
    NodePtr temp = node_queue.front();
    node_queue.pop();

    for (auto &src_node : temp->GetInDataNodes()) {
      if (src_node->GetOutDataNodesSize() == 1) {
        node_queue.push(src_node);
      }
    }
    node_vec.push_back(temp);
  }

  return SUCCESS;
}

bool UnusedOpRemovePass::IsExceptions(const NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, return false, "node is nullptr");
  auto op_def = node->GetOpDesc();
  GE_CHK_BOOL_EXEC(op_def != nullptr, return false, "opdesc is nullptr");
  // permute optimised in permute_pass.cpp
  if (op_def->GetType() == PERMUTE) {
    GE_IF_BOOL_EXEC(
        (node->GetInDataNodes().size() != 0 &&
         (node->GetInDataNodes().at(0) != nullptr && node->GetInDataNodes().at(0)->GetOpDesc() != nullptr &&
          node->GetInDataNodes().at(0)->GetOpDesc()->GetType() == ATTENTIONDECODER)),
        return false);
    return true;
  }
  return false;
}
}  // namespace ge
