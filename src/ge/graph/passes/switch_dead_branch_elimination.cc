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

#include "graph/passes/switch_dead_branch_elimination.h"

#include <string>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/common/omg_util.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace {
const std::vector<int>::size_type kDataInputIndex = 0;
const std::vector<int>::size_type kPredInputIndex = 1;
const int kDefaultInputIndex = -1;

bool ParsePred(const ConstGeTensorPtr &tensor) {
  if (tensor == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return false;
  }
  const uint8_t *data_ptr = tensor->GetData().data();
  auto type = tensor->GetTensorDesc().GetDataType();
  switch (type) {
    case DT_BOOL:
      return *reinterpret_cast<const bool *>(data_ptr);
    case DT_FLOAT:
      return static_cast<bool>(*reinterpret_cast<const float *>(data_ptr));
    case DT_DOUBLE:
      return static_cast<bool>(*reinterpret_cast<const double *>(data_ptr));
    case DT_INT8:
    case DT_UINT8:
      return static_cast<bool>(*data_ptr);
    case DT_FLOAT16:
    case DT_INT16:
    case DT_UINT16:
      return static_cast<bool>(*reinterpret_cast<const int16_t *>(data_ptr));
    case DT_INT32:
    case DT_UINT32:
      return static_cast<bool>(*reinterpret_cast<const int32_t *>(data_ptr));
    case DT_INT64:
    case DT_UINT64:
      return static_cast<bool>(*reinterpret_cast<const int64_t *>(data_ptr));
    default:
      return static_cast<bool>(*data_ptr);
  }
}

bool ParseOutDataAnchors(const NodePtr &node, const NodePtr &pred_node, OutDataAnchorPtr &active_out_data_anchor,
                         OutDataAnchorPtr &inactive_out_data_anchor) {
  auto tensors = OpDescUtils::MutableWeights(pred_node);
  if (tensors.empty()) {
    return false;
  }

  bool pred_value = ParsePred(tensors[0]);
  int inactive_output_index = pred_value ? 0 : 1;

  if (node == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return false;
  }
  GELOGI("[%s] Inactive output index = %d", node->GetName().c_str(), inactive_output_index);
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor->GetIdx() == inactive_output_index) {
      inactive_out_data_anchor = out_anchor;
    } else {
      active_out_data_anchor = out_anchor;
    }
  }

  return true;
}
}  // namespace

Status SwitchDeadBranchElimination::DeleteSwitchNode(NodePtr &node, NodePtr &pred_node,
                                                     const OutDataAnchorPtr &active_out_data_anchor) {
  if (node == nullptr || active_out_data_anchor == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return FAILED;
  }
  // link pred's in control nodes to switch
  if (GraphUtils::CopyInCtrlEdges(pred_node, node) != GRAPH_SUCCESS) {
    return FAILED;
  }
  // Remove link between pred and switch
  auto in_pred_anchor = node->GetInDataAnchor(kPredInputIndex);
  GE_CHECK_NOTNULL(in_pred_anchor);
  in_pred_anchor->UnlinkAll();

  /// If condition Const is isolate, it will be delete with pruning
  /// Isolate Switch and delete it
  std::vector<int> switch_io_map = {kDefaultInputIndex, kDefaultInputIndex};
  size_t out_index = static_cast<size_t>(active_out_data_anchor->GetIdx());
  if (out_index >= switch_io_map.size()) {
    GELOGE(FAILED, "[%s] out index check failed, out_index:%zu.", node->GetName().c_str(), out_index);
    return FAILED;
  }
  switch_io_map[out_index] = kDataInputIndex;
  return IsolateAndDeleteNode(node, switch_io_map);
}

Status SwitchDeadBranchElimination::Run(NodePtr &node) {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "Param [node] must not be null.");
    return PARAM_INVALID;
  }

  std::string op_type;
  GE_CHK_STATUS_RET(GetOriginalType(node, op_type), "Get original type failed");
  if ((op_type != SWITCH) && (op_type != REFSWITCH)) {
    return SUCCESS;
  }

  if (node->GetOutAllNodes().empty()) {
    return SUCCESS;
  }

  auto pred_node = PassUtils::GetInDataNode(node, kPredInputIndex);
  if (pred_node == nullptr) {
    GELOGD("[%s] Pred input is null.", node->GetName().c_str());
    return SUCCESS;
  }

  // Can be optimized when pred is constant
  if (!PassUtils::IsConstant(pred_node)) {
    GELOGD("[%s] Pred is not constant.", node->GetName().c_str());
    return SUCCESS;
  }

  auto input_node = PassUtils::GetInDataNode(node, kDataInputIndex);
  if (input_node == nullptr) {
    GELOGD("[%s] Data input is null.", node->GetName().c_str());
    return SUCCESS;
  }

  // Get active & inactive output anchors by the value of pred
  OutDataAnchorPtr active_out_data_anchor = nullptr;
  OutDataAnchorPtr inactive_out_data_anchor = nullptr;
  if (!ParseOutDataAnchors(node, pred_node, active_out_data_anchor, inactive_out_data_anchor)) {
    return PARAM_INVALID;
  }

  if (inactive_out_data_anchor != nullptr) {
    GELOGI("[%s] To unlink inactive output %d", node->GetName().c_str(), inactive_out_data_anchor->GetIdx());
    std::vector<NodePtr> del_nodes;
    std::vector<NodePtr> end_nodes;
    Status ret = PassUtils::RemoveInactiveBranchToMerge(inactive_out_data_anchor, del_nodes, end_nodes);
    if (ret != SUCCESS) {
      return ret;
    }

    for (auto &end_node : end_nodes) {
      AddRePassNode(end_node);
    }
    for (const auto &delete_node : del_nodes) {
      AddNodeDeleted(delete_node.get());
    }
  }

  return DeleteSwitchNode(node, pred_node, active_out_data_anchor);
}
}  // namespace ge
