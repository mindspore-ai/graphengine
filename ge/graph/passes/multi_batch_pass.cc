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

#include "graph/passes/multi_batch_pass.h"

#include <stack>
#include <unordered_set>
#include "common/ge/ge_util.h"
#include "graph/common/omg_util.h"
#include "graph/utils/type_utils.h"

using std::string;
using std::vector;

namespace ge {
Status MultiBatchPass::Run(ComputeGraphPtr graph) {
  GELOGD("MultiBatchPass Enter");

  if (graph->GetParentGraph() != nullptr) {
    GELOGI("Subgraph %s skip the MultiBatchPass.", graph->GetName().c_str());
    return SUCCESS;
  }
  OutDataAnchorPtr pred_value = nullptr;
  Status ret = FindPredValue(graph, pred_value);
  if (ret == NOT_CHANGED) {
    GELOGD("SwitchN node not exist, graph not changed.");
    return SUCCESS;
  }
  if (ret != SUCCESS) {
    GELOGE(FAILED, "FindPredValue failed.");
    return FAILED;
  }

  if (GetDynamicType() != SUCCESS) {
    GELOGE(FAILED, "Get dynamic type failed.");
    return FAILED;
  }
  if (GetUserDesignateShape() != SUCCESS) {
    GELOGE(FAILED, "Get user designate shape failed.");
    return FAILED;
  }
  std::vector<std::vector<int64_t>> batch_shape;
  vector<vector<int64_t>> combined_batch;
  if (!CheckSwitchN(batch_shape, combined_batch)) {
    GELOGE(FAILED, "CheckSwitchN failed.");
    return FAILED;
  }

  if (attach_label_only_) {
    return AttachLabelOnly(batch_shape.size());
  }

  if (FindSwitchOutNodes(batch_shape.size()) != SUCCESS) {
    GELOGE(FAILED, "Find SwitchN out nodes failed.");
    return FAILED;
  }

  if (ReplaceSwitchN(graph, pred_value, batch_shape, combined_batch) != SUCCESS) {
    GELOGE(FAILED, "Replace SwitchN nodes failed.");
    return FAILED;
  }

  for (const NodePtr &node : bypass_nodes_) {
    if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Remove SwitchN nodes %s failed.", node->GetName().c_str());
      return FAILED;
    }
  }

  GELOGD("MultiBatchPass Leave");
  return SUCCESS;
}

///
/// @brief Clear Status
/// @return
///
Status MultiBatchPass::ClearStatus() {
  switch_n_nodes_.clear();
  bypass_nodes_.clear();
  batch_head_nodes_.clear();
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Set batch label for Case mode.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @param [in] const NodePtr &case_node: Case Node.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchPass::SetCaseLabel(const ComputeGraphPtr &graph, const NodePtr &case_node) {
  const auto &func_desc = case_node->GetOpDesc();
  if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
    GELOGD("Graph: %s Not multi-batch, Node: %s", graph->GetName().c_str(), case_node->GetName().c_str());
    return SUCCESS;
  }

  const auto &dynamic_branch_names = func_desc->GetSubgraphInstanceNames();
  for (size_t i = 0; i < dynamic_branch_names.size(); ++i) {
    const auto &subgraph = graph->GetSubgraph(dynamic_branch_names[i]);
    GE_CHECK_NOTNULL(subgraph);

    const string batch_label = "Batch_" + std::to_string(i);
    for (const auto &node : subgraph->GetDirectNode()) {
      (void)AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label);
    }
  }

  return SUCCESS;
}

///
/// @brief Replace & Combine SwitchN nodes
/// @param [in] graph
/// @param [out] pred_value
/// @return Status
///
Status MultiBatchPass::FindPredValue(const ComputeGraphPtr &graph, OutDataAnchorPtr &pred_value) {
  for (const NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == CASE) {
      GE_CHK_STATUS_RET(SetCaseLabel(graph, node), "Set batch label failed");
      continue;
    }
    if (node->GetType() != SWITCHN) {
      continue;
    }

    InDataAnchorPtr in_data_anchor = node->GetInDataAnchor(SWITCH_PRED_INPUT);
    if (in_data_anchor == nullptr) {
      GELOGE(FAILED, "FindPredInput failed, in_data_anchor is null, node:%s.", node->GetName().c_str());
      return FAILED;
    }
    OutDataAnchorPtr pred_input = in_data_anchor->GetPeerOutAnchor();
    if (pred_input == nullptr) {
      GELOGE(FAILED, "FindPredInput failed, pred_input is null, node:%s.", node->GetName().c_str());
      return FAILED;
    }

    if (pred_value == nullptr) {
      pred_value = pred_input;
    } else if (pred_value != pred_input) {
      GELOGE(FAILED, "Multi pred_value node exist.");
      return FAILED;
    }
    switch_n_nodes_.emplace_back(node);
  }

  if (switch_n_nodes_.empty()) {
    GELOGD("SwitchN node not exist.");
    return NOT_CHANGED;
  }

  if (pred_value == nullptr) {
    GELOGE(FAILED, "FindPredInput failed, pred_value is null.");
    return FAILED;
  }

  GELOGI("Find pred_value %s.", pred_value->GetOwnerNode()->GetName().c_str());
  return SUCCESS;
}

///
/// @brief Get dynamic type: dynamic batch size: 1, dynamic image size: 2, dynamic dims: 3
/// @return Status
///
Status MultiBatchPass::GetDynamicType() {
  for (const auto &switchn : switch_n_nodes_) {
    auto switchn_desc = switchn->GetOpDesc();
    GE_CHECK_NOTNULL(switchn_desc);
    int32_t dynamic_type = static_cast<int32_t>(FIXED);
    if (!AttrUtils::GetInt(switchn_desc, ATTR_DYNAMIC_TYPE, dynamic_type)) {
      GELOGE(FAILED, "Get attr ATTR_DYNAMIC_TYPE of node: %s failed.", switchn->GetName().c_str());
      return FAILED;
    }
    if (dynamic_type == static_cast<int32_t>(FIXED)) {
      GELOGE(FAILED, "Attr ATTR_DYNAMIC_TYPE shouldn't be 0.");
      return FAILED;
    }
    if (dynamic_type_ != static_cast<int32_t>(FIXED) && dynamic_type_ != dynamic_type) {
      GELOGE(FAILED, "Attr ATTR_DYNAMIC_TYPE of all switchn node should be same, while one is %d and another is %d.",
             dynamic_type, dynamic_type_);
      return FAILED;
    }
    dynamic_type_ = dynamic_type;
  }
  if (dynamic_type_ == static_cast<int32_t>(FIXED)) {
    GELOGE(FAILED, "Attr ATTR_DYNAMIC_TYPE shouldn't be 0.");
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Get user designate shape order. eg{"data","label","mask"}
/// @return Status
///
Status MultiBatchPass::GetUserDesignateShape() {
  data_name_order_.clear();
  bool first_check = true;
  for (const auto &switchn : switch_n_nodes_) {
    auto switchn_desc = switchn->GetOpDesc();
    GE_CHECK_NOTNULL(switchn_desc);
    vector<string> cur_switchn_data_name_order;
    if (!AttrUtils::GetListStr(switchn_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, cur_switchn_data_name_order)) {
      GELOGE(FAILED, "Get attr ATTR_USER_DESIGNEATE_SHAPE_ORDER of node: %s failed.", switchn->GetName().c_str());
      return FAILED;
    }
    if (first_check) {
      data_name_order_ = cur_switchn_data_name_order;
      first_check = false;
    } else {
      if (data_name_order_ != cur_switchn_data_name_order) {
        GELOGE(FAILED, "The ATTR_USER_DESIGNEATE_SHAPE_ORDER of switchN must be same: %s failed.",
               switchn->GetName().c_str());
        return FAILED;
      }
    }
  }
  if (data_name_order_.empty()) {
    GELOGE(FAILED, "user shape order can not be empty");
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Check SwitchN nodes
/// @param [out] batch_shape
/// @param [out] combined_batch
/// @return bool
///
bool MultiBatchPass::CheckSwitchN(vector<vector<int64_t>> &batch_shape, vector<vector<int64_t>> &combined_batch) {
  // Check if output_num of different SwitchN is same
  uint32_t batch_num = 0;
  for (const NodePtr &node : switch_n_nodes_) {
    uint32_t tmp_num = node->GetAllOutDataAnchorsSize();
    if (batch_num == 0) {
      batch_num = tmp_num;
    } else if (batch_num != tmp_num) {
      GELOGE(FAILED, "Output size of SwitchN not equal;");
      return false;
    }
  }

  if (!GetBatchInfo(batch_num, batch_shape, combined_batch)) {
    GELOGE(FAILED, "Get batch info failed.");
    return false;
  }

  if (batch_shape.empty()) {
    GELOGE(FAILED, "batch_shape is empty.");
    return false;
  }
  if (combined_batch.empty()) {
    GELOGE(FAILED, "combined_batch is empty.");
    return false;
  }
  size_t dim_num = batch_shape[0].size();
  size_t combined_dim_num = combined_batch[0].size();
  for (uint32_t i = 1; i < batch_num; i++) {
    size_t tmp_dim_num = batch_shape[i].size();
    if (dim_num != tmp_dim_num) {
      GELOGE(FAILED, "Dim num of batch_shape not equal, batch_0:%zu, batch_%u:%zu.", dim_num, i, tmp_dim_num);
      return false;
    }
    size_t tmp_combined_dim_num = combined_batch[i].size();
    if (combined_dim_num != tmp_combined_dim_num) {
      GELOGE(FAILED, "Dim num of combined_batch not equal, batch_0:%zu, batch_%u:%zu.", dim_num, i, tmp_dim_num);
      return false;
    }
  }

  return true;
}

///
/// @brief Check SwitchN nodes
/// @param [in] batch_num
/// @param [out] batch_shape
/// @param [out] combined_batch
/// @return bool
///
bool MultiBatchPass::GetBatchInfo(uint32_t batch_num, vector<vector<int64_t>> &batch_shape,
                                  vector<vector<int64_t>> &combined_batch) {
  // Check if output_shape of different SwitchN is same
  vector<vector<int64_t>> idx_batch_shape;
  vector<vector<int64_t>> idx_combined_batch;
  for (uint32_t i = 0; i < batch_num; i++) {
    idx_batch_shape.clear();
    idx_combined_batch.clear();
    for (const NodePtr &node : switch_n_nodes_) {
      OpDescPtr op_desc = node->GetOpDesc();
      if (op_desc == nullptr) {
        GELOGE(FAILED, "CheckDims failed, get op_desc failed, node: %s.", node->GetName().c_str());
        return false;
      }
      vector<int64_t> output_dims;
      if (!AttrUtils::GetListInt(op_desc->GetOutputDesc(i), ATTR_NAME_SWITCHN_PRED_VALUE, output_dims)) {
        GELOGE(FAILED, "CheckDims failed, get attr ATTR_NAME_SWITCHN_PRED_VALUE failed, batch_index=%u.", i);
        return false;
      }
      idx_batch_shape.emplace_back(output_dims);
      output_dims.clear();
      if (!AttrUtils::GetListInt(op_desc->GetOutputDesc(i), ATTR_NAME_COMBINED_DYNAMIC_DIMS, output_dims)) {
        GELOGE(FAILED, "CheckDims failed, get attr ATTR_NAME_COMBINED_DYNAMIC_DIMS failed, batch_index=%u.", i);
        return false;
      }
      idx_combined_batch.emplace_back(output_dims);
    }
    if (!CheckDims(idx_batch_shape)) {
      GELOGE(FAILED, "CheckDims failed, batch_index=%u.", i);
      return false;
    }

    batch_shape.emplace_back(idx_batch_shape[0]);
    combined_batch.emplace_back(idx_combined_batch[0]);
  }
  return true;
}

///
/// @brief Find outputs of SwitchN nodes
/// @param [in] batch_num
/// @return void
///
Status MultiBatchPass::FindSwitchOutNodes(uint32_t batch_num) {
  std::vector<NodePtr> output_nodes;
  for (uint32_t i = 0; i < batch_num; i++) {
    output_nodes.clear();
    for (const NodePtr &node : switch_n_nodes_) {
      // idx is promised to be valid
      OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(i);
      GE_CHECK_NOTNULL(out_data_anchor);
      for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        auto out_node = peer_in_anchor->GetOwnerNode();
        if (out_node->GetType() != IDENTITY || !out_node->GetOutDataNodes().empty()) {
          output_nodes.emplace_back(out_node);
          continue;
        }
        bypass_nodes_.emplace_back(out_node);
        if (GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
          GELOGE(FAILED, "Remove SwitchN out_data_edge failed, %s->%s.", node->GetName().c_str(),
                 out_node->GetName().c_str());
          return FAILED;
        }
        for (auto &identity_out_node : out_node->GetOutControlNodes()) {
          output_nodes.emplace_back(identity_out_node);
          if (GraphUtils::RemoveEdge(out_node->GetOutControlAnchor(), identity_out_node->GetInControlAnchor()) !=
              GRAPH_SUCCESS) {
            GELOGE(FAILED, "Remove SwitchN out_data_edge failed, %s->%s.", node->GetName().c_str(),
                   out_node->GetName().c_str());
            return FAILED;
          }
        }
      }
    }
    batch_head_nodes_.emplace_back(output_nodes);
  }

  return SUCCESS;
}

///
/// @brief Replace & Combine SwitchN nodes
/// @param [in] graph
/// @param [in] pred_value
/// @param [in] batch_shape
/// @param [in] combined_batch
/// @return Status
///
Status MultiBatchPass::ReplaceSwitchN(const ComputeGraphPtr &graph, const OutDataAnchorPtr &pred_value,
                                      const vector<vector<int64_t>> &batch_shape,
                                      const vector<vector<int64_t>> &combined_batch) {
  NodePtr pred_value_node = pred_value->GetOwnerNode();
  // Create SwitchCase node
  const std::string &switch_case_name = pred_value_node->GetName() + "_" + STREAMSWITCHN;
  NodePtr switch_case = CreateSwitchCaseNode(graph, switch_case_name, pred_value, batch_shape, combined_batch);
  if (switch_case == nullptr) {
    GELOGE(FAILED, "CreateSwitchCaseNode %s failed.", switch_case_name.c_str());
    return FAILED;
  }

  for (const NodePtr &switch_n_node : switch_n_nodes_) {
    if (BypassSwitchN(switch_n_node, switch_case) != SUCCESS) {
      GELOGE(FAILED, "Bypass SwitchN %s failed.", switch_case_name.c_str());
      return FAILED;
    }
  }

  // Add switchCase input edge
  if (GraphUtils::AddEdge(pred_value, switch_case->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add SwitchCase in_data_edge failed, %s->%s.", pred_value_node->GetName().c_str(),
           switch_case->GetName().c_str());
    return FAILED;
  }

  if (AttachLabel(switch_case) != SUCCESS) {
    GELOGE(FAILED, "AttachLabel failed.");
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Check if output_shape of different SwitchN is same
/// @param [in] output_shape
/// @return bool
///
bool MultiBatchPass::CheckDims(const std::vector<std::vector<int64_t>> &output_shape) const {
  if (output_shape.empty()) {
    GELOGE(FAILED, "CheckDims failed: output_shape is empty.");
    return false;
  }

  size_t num = output_shape.size();
  size_t dim_num = output_shape[0].size();
  for (size_t i = 1; i < num; i++) {
    size_t tmp_dim_num = output_shape[i].size();
    if (dim_num != tmp_dim_num) {
      GELOGE(FAILED, "CheckDims failed: dim_num not equal, output_0:%zu, output_%zu:%zu.", dim_num, i, tmp_dim_num);
      return false;
    }
  }

  if (dim_num == 0) {
    return true;
  }

  for (size_t i = 0; i < dim_num; i++) {
    int64_t dim_value = output_shape[0][i];
    for (size_t j = 1; j < num; j++) {
      int64_t tmp_dim_value = output_shape[j][i];
      if (dim_value != tmp_dim_value) {
        GELOGE(FAILED, "CheckDims failed: dim_value not equal, dim_index=%zu, dim_value_0:%ld, dim_value_%zu:%ld.", i,
               dim_value, j, tmp_dim_value);
        return false;
      }
    }
  }
  return true;
}

///
/// @brief Create StreamSwitchN node
/// @param [in] graph
/// @param [in] name
/// @param [in] pred_value
/// @param [in] batch_shape
/// @param [in] combined_batch
/// @return ge::NodePtr
///
NodePtr MultiBatchPass::CreateSwitchCaseNode(const ComputeGraphPtr &graph, const std::string &name,
                                             const OutDataAnchorPtr &pred_value,
                                             const vector<vector<int64_t>> &batch_shape,
                                             const vector<vector<int64_t>> &combined_batch) {
  OpDescPtr op_desc = MakeShared<OpDesc>(name, STREAMSWITCHN);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc failed, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }

  GELOGI("Create StreamSwitchN op:%s.", name.c_str());
  OpDescPtr pred_desc = pred_value->GetOwnerNode()->GetOpDesc();
  if (pred_desc == nullptr) {
    GELOGE(FAILED, "Get pred_desc failed, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }
  if (op_desc->AddInputDesc(pred_desc->GetOutputDesc(pred_value->GetIdx())) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "AddInputDesc failed, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }

  NodePtr switch_case_node = graph->AddNode(op_desc);
  if (switch_case_node == nullptr) {
    GELOGE(FAILED, "Create node failed, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }

  uint32_t batch_num = static_cast<uint32_t>(batch_shape.size());
  if (!AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGE(FAILED, "set attr ATTR_NAME_BATCH_NUM failed, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }
  if (!AttrUtils::SetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type_)) {
    GELOGE(FAILED, "Set attr ATTR_DYNAMIC_TYPE failed, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }
  if (!AttrUtils::SetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, data_name_order_)) {
    GELOGE(FAILED, "Set attr ATTR_USER_DESIGNEATE_SHAPE_ORDER failed, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }
  for (uint32_t i = 0; i < batch_num; i++) {
    const std::string &attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    if (!AttrUtils::SetListInt(op_desc, attr_name, batch_shape[i])) {
      GELOGE(FAILED, "set attr ATTR_NAME_PRED_VALUE failed, StreamSwitchN:%s.", name.c_str());
      return nullptr;
    }
    const string &attr_combined_batch = ATTR_NAME_COMBINED_BATCH + "_" + std::to_string(i);
    if (!AttrUtils::SetListInt(op_desc, attr_combined_batch, combined_batch[i])) {
      GELOGE(FAILED, "set attr ATTR_NAME_COMBINED_BATCH failed, StreamSwitchN:%s.", name.c_str());
      return nullptr;
    }
  }

  return switch_case_node;
}

///
/// @brief Bypass SwitchN node
/// @param [in] switch_n_node
/// @param [in] switch_case
/// @return Status
///
Status MultiBatchPass::BypassSwitchN(const NodePtr &switch_n_node, const NodePtr &switch_case) {
  InDataAnchorPtr in_data_anchor = switch_n_node->GetInDataAnchor(SWITCH_DATA_INPUT);
  if (in_data_anchor == nullptr) {
    GELOGE(FAILED, "Check in_data_anchor failed, SwitchN:%s.", switch_n_node->GetName().c_str());
    return FAILED;
  }
  OutDataAnchorPtr peer_data_anchor = in_data_anchor->GetPeerOutAnchor();
  if (peer_data_anchor == nullptr) {
    GELOGE(FAILED, "Check peer_data_anchor failed, SwitchN:%s.", switch_n_node->GetName().c_str());
    return FAILED;
  }
  NodePtr data_input = peer_data_anchor->GetOwnerNode();

  // Remove SwitchN data input
  if (GraphUtils::RemoveEdge(peer_data_anchor, in_data_anchor) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Remove SwitchN in_data_edge failed, %s->%s.", data_input->GetName().c_str(),
           switch_n_node->GetName().c_str());
    return FAILED;
  }
  if (GraphUtils::AddEdge(data_input->GetOutControlAnchor(), switch_case->GetInControlAnchor()) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add StreamSwitchN in_control_edge failed, %s->%s.", data_input->GetName().c_str(),
           switch_case->GetName().c_str());
    return FAILED;
  }

  // Add SwitchCase control output
  for (const OutDataAnchorPtr &out_data_anchor : switch_n_node->GetAllOutDataAnchors()) {
    for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      NodePtr data_output = peer_in_anchor->GetOwnerNode();
      if ((GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor) != GRAPH_SUCCESS) ||
          (GraphUtils::AddEdge(peer_data_anchor, peer_in_anchor) != GRAPH_SUCCESS)) {
        GELOGE(FAILED, "Bypass SwitchN data_edge failed, %s->%s->%s.", data_input->GetName().c_str(),
               switch_n_node->GetName().c_str(), data_output->GetName().c_str());
        return FAILED;
      }
      if (GraphUtils::AddEdge(switch_case->GetOutControlAnchor(), data_output->GetInControlAnchor()) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Add SwitchCase out_control_edge failed, %s->%s.", switch_case->GetName().c_str(),
               data_output->GetName().c_str());
        return FAILED;
      }
    }
  }
  GE_CHK_STATUS_RET(MoveCtrlEdges(switch_n_node, switch_case), "Move ctrl edges failed.");

  bypass_nodes_.emplace_back(switch_n_node);
  GELOGI("Bypass SwitchN node %s success.", switch_n_node->GetName().c_str());
  return SUCCESS;
}

///
/// @brief Attach stream_label & batch_label for batch branch
/// @param [in] switch_case_node
/// @return Status
///
Status MultiBatchPass::AttachLabel(const NodePtr &switch_case_node) {
  std::vector<std::string> stream_label_list;
  for (uint32_t i = 0; i < static_cast<uint32_t>(batch_head_nodes_.size()); i++) {
    if (AttachBatchLabel(i) != SUCCESS) {
      GELOGE(FAILED, "AttachBatchLabel failed, batch_idx=%u", i);
      return FAILED;
    }

    const std::string &stream_label = "stream_label_batch_" + std::to_string(i);
    if (AttachStreamLabel(i, stream_label) != SUCCESS) {
      GELOGE(FAILED, "AttachStreamLabel failed, stream_label=%s", stream_label.c_str());
      return FAILED;
    }
    stream_label_list.emplace_back(stream_label);
  }

  return switch_case_node == nullptr ? SUCCESS : SetActiveLabelList(switch_case_node, stream_label_list);
}

///
/// @brief Attach batch_label for batch branch
/// @param [in] batch_idx
/// @return Status
///
Status MultiBatchPass::AttachBatchLabel(uint32_t batch_idx) {
  std::stack<NodePtr> nodes;
  for (const auto &node : batch_head_nodes_[batch_idx]) {
    nodes.push(node);
  }

  const std::string &batch_label = "Batch_" + std::to_string(batch_idx);
  std::unordered_set<NodePtr> handled_nodes;
  while (!nodes.empty()) {
    NodePtr cur_node = nodes.top();
    nodes.pop();
    if (handled_nodes.count(cur_node) > 0) {
      continue;
    }

    OpDescPtr cur_desc = cur_node->GetOpDesc();
    GE_CHECK_NOTNULL(cur_desc);
    if (cur_desc->HasAttr(ATTR_NAME_BATCH_LABEL)) {
      std::string tmp_label;
      if (!AttrUtils::GetStr(cur_desc, ATTR_NAME_BATCH_LABEL, tmp_label)) {
        GELOGE(FAILED, "get attr ATTR_NAME_BATCH_LABEL failed, node: %s.", cur_desc->GetName().c_str());
        return FAILED;
      }
      if (tmp_label != batch_label) {
        GELOGE(FAILED, "Reach other batch_branch, node:%s, cur_label:%s, batch_label:%s.", cur_desc->GetName().c_str(),
               tmp_label.c_str(), batch_label.c_str());
        return FAILED;
      }
    }
    GELOGD("Attach batch_label %s to node %s.", batch_label.c_str(), cur_desc->GetName().c_str());
    if (!AttrUtils::SetStr(cur_desc, ATTR_NAME_BATCH_LABEL, batch_label)) {
      GELOGE(FAILED, "set attr ATTR_NAME_BATCH_LABEL failed, node:%s.", cur_desc->GetName().c_str());
      return FAILED;
    }

    for (const auto &out_node : cur_node->GetOutAllNodes()) {
      OpDescPtr op_desc = out_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const std::string &type = op_desc->GetType();
      if ((type == MERGE) && (op_desc->HasAttr(ATTR_INSERT_BY_MBATCH))) {
        continue;
      }
      if (type == NETOUTPUT) {
        GELOGE(FAILED, "Reach net_output without Merge, cur_node:%s.", cur_node->GetName().c_str());
        return FAILED;
      }
      nodes.push(out_node);
    }
    (void)handled_nodes.insert(cur_node);
  }

  return SUCCESS;
}

///
/// @brief Attach stream_label for batch branch
/// @param [in] batch_idx
/// @param [in] stream_label
/// @return Status
///
Status MultiBatchPass::AttachStreamLabel(uint32_t batch_idx, const std::string &stream_label) {
  std::stack<NodePtr> nodes;
  for (const auto &node : batch_head_nodes_[batch_idx]) {
    nodes.push(node);
  }

  std::unordered_set<NodePtr> handled_nodes;
  while (!nodes.empty()) {
    NodePtr cur_node = nodes.top();
    nodes.pop();

    OpDescPtr cur_desc = cur_node->GetOpDesc();
    GE_CHECK_NOTNULL(cur_desc);
    if ((handled_nodes.count(cur_node) > 0) || (cur_desc->HasAttr(ATTR_NAME_STREAM_LABEL))) {
      continue;
    }

    GELOGD("Attach stream_label %s to node %s.", stream_label.c_str(), cur_desc->GetName().c_str());
    if (SetStreamLabel(cur_node, stream_label) != SUCCESS) {
      GELOGE(FAILED, "Set stream_label failed, node:%s.", cur_node->GetName().c_str());
      return FAILED;
    }

    for (const auto &out_node : cur_node->GetOutAllNodes()) {
      nodes.push(out_node);
    }

    (void)handled_nodes.insert(cur_node);
  }

  return SUCCESS;
}

///
/// @brief move edges from old_node to new_node
/// @param [in] old_node
/// @param [in] new_node
/// @return Status
///
Status MultiBatchPass::MoveCtrlEdges(const NodePtr &old_node, const NodePtr &new_node) {
  if (old_node == new_node) {
    return SUCCESS;
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

///
/// @brief attach stream_label & batch_label without change structure of graph
/// @param [in] batch_num
/// @return void
///
Status MultiBatchPass::AttachLabelOnly(uint32_t batch_num) {
  std::vector<NodePtr> output_nodes;
  for (uint32_t i = 0; i < batch_num; i++) {
    output_nodes.clear();
    for (const NodePtr &node : switch_n_nodes_) {
      // idx is promised to be valid
      OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(i);
      GE_CHECK_NOTNULL(out_data_anchor);
      for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        output_nodes.emplace_back(peer_in_anchor->GetOwnerNode());
      }
    }
    batch_head_nodes_.emplace_back(output_nodes);
  }

  return AttachLabel(nullptr);
}
}  // namespace ge
