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

#include "graph/passes/merge_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge/ge_util.h"
#include "graph/common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/passes/pass_utils.h"

using domi::PARAM_INVALID;
using domi::SUCCESS;

namespace ge {
const int kValueIndexOutputIndex = 1;

bool IsEmptyTensor(const GeShape &shape) {
  const auto &dims = shape.GetDims();
  return std::any_of(dims.begin(), dims.end(), [](int64_t dim) { return dim == 0; });
}

Status MergePass::Run(NodePtr &node) {
  GELOGD("MergePass running");
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "param [node] must not be null.");
    return PARAM_INVALID;
  }

  std::string op_type;
  GE_CHK_STATUS_RET(GetOriginalType(node, op_type), "get original type failed");
  if (op_type != MERGE) {
    return SUCCESS;
  }

  auto out_data_anchors = node->GetAllOutDataAnchors();
  if (out_data_anchors.empty()) {
    GELOGE(PARAM_INVALID, "[%s] Merge node output anchor is empty", node->GetName().c_str());
    return PARAM_INVALID;
  }

  if (OptimizeEmptyTensorInput(node) != SUCCESS) {
    GELOGE(FAILED, "[%s] remove empty_tensor inputs failed.", node->GetName().c_str());
    return FAILED;
  }

  auto in_data_nodes = node->GetInDataNodes();
  switch (in_data_nodes.size()) {
    case 0: {
      /// Case A: input_count = 0, the output of merge node is inactive as well
      /// In which case the output branch can be removed
      /// until another merge node is met
      std::vector<NodePtr> del_nodes;
      std::vector<NodePtr> end_nodes;
      Status ret = PassUtils::RemoveBranch(node, del_nodes, end_nodes);
      for (auto &end_node : end_nodes) {
        AddRePassNode(end_node);
      }
      for (const auto &delete_node : del_nodes) {
        AddNodeDeleted(delete_node);
      }
      return ret;
    }
    case 1: {  // Case B: input_count = 1, the merge node can be optimized out
      std::vector<int> merge_io_map = {PassUtils::GetUniqueInDataAnchorIndex(node), -1};
      if (merge_io_map[0] != -1 && IsNeedChangeIndexToConstant(node)) {
        int index = merge_io_map[0];
        if (ChangeIndexToConstant(node, index) != SUCCESS) {
          GELOGE(FAILED, "[%s] Change value index to be Constant failed.", node->GetName().c_str());
          return FAILED;
        }
      }
      auto in_node = in_data_nodes.at(0);
      if (IsMergeInputNeedOptimized(in_node)) {
        if (IsolateAndDeleteNode(in_node, {0}) != SUCCESS) {
          GELOGE(FAILED, "Isolate and delete node %s failed.", in_node->GetName().c_str());
          return FAILED;
        }
      }
      return IsolateAndDeleteNode(node, merge_io_map);
    }
    default: {
      // Case C: input_count > 1, the merge node can not be optimized
      return SUCCESS;
    }
  }
}

bool MergePass::IsNeedChangeIndexToConstant(NodePtr &node) const {
  /// value_index is the index 1 output of the Merge
  /// value_index link to other node, change it to be Constant
  GE_IF_BOOL_EXEC(node == nullptr, GELOGW("Node is nullptr"); return false);
  auto out_anchor = node->GetOutDataAnchor(kValueIndexOutputIndex);
  GE_IF_BOOL_EXEC(out_anchor == nullptr, GELOGW("Out_anchor is nullptr"); return false);
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNode() != nullptr) {
      GELOGI(
        "[%s] MergePass, value_index link to other node, "
        "change it to be Constant.",
        node->GetName().c_str());
      return true;
    }
  }

  return false;
}

Status MergePass::ChangeIndexToConstant(NodePtr &node, int &value_index) {
  GE_CHECK_NOTNULL(node);
  ComputeGraphPtr graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    GELOGE(FAILED, "[%s] The owner graph must not be null.", node->GetName().c_str());
    return FAILED;
  }

  OpDescPtr constant_op_desc = nullptr;
  if (CreateConstByValue(node, value_index, constant_op_desc) != SUCCESS) {
    return FAILED;
  }
  NodePtr const_node = graph->AddNode(constant_op_desc);
  if (const_node == nullptr) {
    return FAILED;
  }

  // Change peer in anchors from value_index to new Constant node
  if (GraphUtils::ReplaceNodeAnchors(const_node, node, {}, {1}) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "[%s] ReplaceNodeAnchors failed.", node->GetName().c_str());
    return FAILED;
  }
  auto out_control_anchor = node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(out_control_anchor);
  // Add control anchor between Merge and Constant
  if (out_control_anchor->LinkTo(const_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

Status MergePass::CreateConstByValue(NodePtr &node, int value_index, OpDescPtr &op_desc) {
  std::string constant_name = node->GetName() + "_value_index";
  // 1. create Constant OpDesc
  op_desc = MakeShared<OpDesc>(constant_name, CONSTANT);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "[%s] Make shared of Constant op desc failed.", constant_name.c_str());
    return FAILED;
  }

  // 2. get OpDesc of output number one of Merge(value_index)
  OpDescPtr original_op_desc = node->GetOpDesc();
  if (original_op_desc == nullptr) {
    GELOGE(FAILED, "[%s] Op desc must not be null.", constant_name.c_str());
    return FAILED;
  }
  GeTensorDesc original_out_tensor_desc = original_op_desc->GetOutputDesc(1);
  original_out_tensor_desc.SetDataType(DT_INT32);

  // 3. create attr value of Constant, is a tensor
  GeTensorPtr const_tensor_ptr =
    MakeShared<GeTensor>(original_out_tensor_desc, reinterpret_cast<uint8_t *>(&value_index), sizeof(int));
  if (const_tensor_ptr == nullptr) {
    GELOGE(FAILED, "[%s] Make shared of Constant tensor failed.", constant_name.c_str());
    return FAILED;
  }

  GE_IF_BOOL_EXEC(!AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, const_tensor_ptr),
                  GELOGE(FAILED, "get ATTR_NAME_WEIGHTS failed");
                  return FAILED);

  // 4. set Constant output desc
  GE_CHK_STATUS_RET(op_desc->AddOutputDesc(original_out_tensor_desc), "add out put desc failed");
  return SUCCESS;
}

bool MergePass::IsMergeInputNeedOptimized(NodePtr &node) const {
  if (node == nullptr) {
    return false;
  }
  // node is not inserted by MergeInputMemcpyPass
  if ((node->GetType() != MEMCPYASYNC) && (node->GetType() != MEMCPYADDRASYNC)) {
    return false;
  }
  if (node->GetInDataNodes().size() != 1) {
    return false;
  }

  auto in_node = node->GetInDataNodes().at(0);
  if (in_node == nullptr) {
    return false;
  }
  // in_node may be global_step var
  if ((in_node->GetType() == VARIABLE) || (in_node->GetType() == VARIABLEV2)) {
    return false;
  }
  return true;
}

Status MergePass::OptimizeEmptyTensorInput(const NodePtr &node) {
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    const auto &peer_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_data_anchor == nullptr) {
      continue;
    }
    if ((peer_data_anchor->GetOwnerNode() == nullptr) || (peer_data_anchor->GetOwnerNode()->GetOpDesc() == nullptr)) {
      continue;
    }
    const auto &op_desc = peer_data_anchor->GetOwnerNode()->GetOpDesc();
    if (IsEmptyTensor(op_desc->GetOutputDesc(peer_data_anchor->GetIdx()).GetShape())) {
      if (GraphUtils::RemoveEdge(peer_data_anchor, in_data_anchor) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Remove data edge %s:%d->%s:%d failed.", op_desc->GetName().c_str(), peer_data_anchor->GetIdx(),
               node->GetName().c_str(), in_data_anchor->GetIdx());
        return FAILED;
      }
      GELOGD("Remove data edge %s:%d->%s:%d", op_desc->GetName().c_str(), peer_data_anchor->GetIdx(),
             node->GetName().c_str(), in_data_anchor->GetIdx());
    }
  }
  return SUCCESS;
}
}  // namespace ge
