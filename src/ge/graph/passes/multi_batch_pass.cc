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

#include "graph/passes/multi_batch_pass.h"

#include <stack>
#include <unordered_set>
#include <utility>

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"

namespace ge {
Status MultiBatchPass::Run(ComputeGraphPtr graph) {
  GELOGD("MultiBatchPass Enter");

  GraphUtils::DumpGEGraph(graph, "BeforeMultiBatchPass");
  GraphUtils::DumpGEGraphToOnnx(*graph, "BeforeMultiBatchPass");

  OutDataAnchorPtr pred_value = nullptr;
  Status ret = FindPredValue(graph, pred_value);
  if (ret == NOT_CHANGED) {
    GELOGI("SwitchN node not exist, graph not changed.");
    return SUCCESS;
  }
  if (ret != SUCCESS) {
    GELOGE(FAILED, "FindPredValue fail.");
    return FAILED;
  }

  std::vector<std::vector<int64_t>> batch_shape;
  if (!CheckSwitchN(batch_shape)) {
    GELOGE(FAILED, "CheckSwitchN fail.");
    return FAILED;
  }

  FindSwitchOutNodes(batch_shape.size());

  if (ReplaceSwitchN(graph, pred_value, batch_shape) != SUCCESS) {
    GELOGE(FAILED, "Replace SwitchN nodes fail.");
    return FAILED;
  }

  for (NodePtr &node : bypass_nodes_) {
    if (graph->RemoveNode(node) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Remove SwitchN nodes %s fail.", node->GetName().c_str());
      return FAILED;
    }
  }

  GraphUtils::DumpGEGraph(graph, "AfterMultiBatchPass");
  GraphUtils::DumpGEGraphToOnnx(*graph, "AfterMultiBatchPass");

  GELOGD("MultiBatchPass Leave");
  return SUCCESS;
}

///
/// @brief Replace & Combine SwitchN nodes
/// @param [in] graph
/// @param [out] pred_value
/// @return Status
///
Status MultiBatchPass::FindPredValue(const ComputeGraphPtr &graph, OutDataAnchorPtr &pred_value) {
  for (NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() != SWITCHN) {
      continue;
    }

    InDataAnchorPtr in_data_anchor = node->GetInDataAnchor(SWITCH_PRED_INPUT);
    if (in_data_anchor == nullptr) {
      GELOGE(FAILED, "FindPredInput fail, in_data_anchor is null, node:%s.", node->GetName().c_str());
      return FAILED;
    }
    OutDataAnchorPtr pred_input = in_data_anchor->GetPeerOutAnchor();
    if (pred_input == nullptr) {
      GELOGE(FAILED, "FindPredInput fail, pred_input is null, node:%s.", node->GetName().c_str());
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
    GELOGI("SwitchN node not exist.");
    return NOT_CHANGED;
  }

  if (pred_value == nullptr) {
    GELOGE(FAILED, "FindPredInput fail, pred_value is null.");
    return FAILED;
  }

  GELOGI("Find pred_value %s.", pred_value->GetOwnerNode()->GetName().c_str());
  return SUCCESS;
}

///
/// @brief Check SwitchN nodes
/// @param [out] batch_shape
/// @return bool
///
bool MultiBatchPass::CheckSwitchN(std::vector<std::vector<int64_t>> &batch_shape) {
  // Check if output_num of different SwitchN is same
  uint32_t batch_num = 0;
  for (NodePtr &node : switch_n_nodes_) {
    uint32_t tmp_num = node->GetAllOutDataAnchorsSize();
    if (batch_num == 0) {
      batch_num = tmp_num;
    } else if (batch_num != tmp_num) {
      GELOGE(FAILED, "Output size of SwitchN not equal;");
      return false;
    }
  }

  // Check if output_shape of different SwitchN is same
  std::vector<std::vector<int64_t>> idx_batch_shape;
  for (uint32_t i = 0; i < batch_num; i++) {
    idx_batch_shape.clear();
    for (NodePtr &node : switch_n_nodes_) {
      std::vector<int64_t> output_dims;
      OpDescPtr op_desc = node->GetOpDesc();
      if (op_desc == nullptr) {
        GELOGE(FAILED, "CheckDims fail, get op_desc fail, node: %s.", node->GetName().c_str());
        return false;
      }
      if (!AttrUtils::GetListInt(op_desc->GetOutputDesc(i), ATTR_NAME_SWITCHN_PRED_VALUE, output_dims)) {
        GELOGE(FAILED, "CheckDims fail, get attr ATTR_NAME_SWITCHN_PRED_VALUE fail, batch_index=%u.", i);
        return false;
      }
      idx_batch_shape.emplace_back(output_dims);
    }
    if (!CheckDims(idx_batch_shape)) {
      GELOGE(FAILED, "CheckDims fail, batch_index=%u.", i);
      return false;
    }

    batch_shape.emplace_back(idx_batch_shape[0]);
  }

  // Check if dim_num of different batch is same
  if (batch_shape.empty()) {
    GELOGE(FAILED, "batch_shape is empty.");
    return false;
  }
  uint32_t dim_num = batch_shape[0].size();
  for (uint32_t i = 1; i < batch_num; i++) {
    uint32_t tmp_dim_num = batch_shape[i].size();
    if (dim_num != tmp_dim_num) {
      GELOGE(FAILED, "dim_num not equal, batch_0:%u, batch_%u:%u.", dim_num, i, tmp_dim_num);
      return false;
    }
  }

  return true;
}

///
/// @brief Find outputs of SwitchN nodes
/// @param [in] batch_num
/// @return void
///
void MultiBatchPass::FindSwitchOutNodes(uint32_t batch_num) {
  std::vector<NodePtr> output_nodes;
  for (uint32_t i = 0; i < batch_num; i++) {
    output_nodes.clear();
    for (NodePtr &node : switch_n_nodes_) {
      // idx is promised to be valid
      OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(i);
      GE_CHECK_NOTNULL_JUST_RETURN(out_data_anchor);
      for (InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        output_nodes.emplace_back(peer_in_anchor->GetOwnerNode());
      }
    }
    batch_head_nodes_.emplace_back(output_nodes);
  }

  return;
}

///
/// @brief Replace & Combine SwitchN nodes
/// @param [in] graph
/// @param [in] pred_value
/// @param [in] batch_shape
/// @return Status
///
Status MultiBatchPass::ReplaceSwitchN(ComputeGraphPtr &graph, OutDataAnchorPtr &pred_value,
                                      const std::vector<std::vector<int64_t>> &batch_shape) {
  NodePtr pred_value_node = pred_value->GetOwnerNode();
  // Create SwitchCase node
  const std::string switch_case_name = pred_value_node->GetName() + "_" + STREAMSWITCHN;
  NodePtr switch_case = CreateSwitchCaseNode(graph, switch_case_name, pred_value, batch_shape);
  if (switch_case == nullptr) {
    GELOGE(FAILED, "CreateSwitchCaseNode %s fail.", switch_case_name.c_str());
    return FAILED;
  }

  for (NodePtr &switch_n_node : switch_n_nodes_) {
    if (BypassSwitchN(switch_n_node, switch_case) != SUCCESS) {
      GELOGE(FAILED, "Bypass SwitchN %s fail.", switch_case_name.c_str());
      return FAILED;
    }
  }

  // Add switchCase input edge
  if (GraphUtils::AddEdge(pred_value, switch_case->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add SwitchCase in_data_edge fail, %s->%s.", pred_value_node->GetName().c_str(),
           switch_case->GetName().c_str());
    return FAILED;
  }

  if (AttachLabel(switch_case) != SUCCESS) {
    GELOGE(FAILED, "AttachLabel fail.");
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
    GELOGE(FAILED, "CheckDims fail: output_shape is empty.");
    return false;
  }

  size_t num = output_shape.size();
  size_t dim_num = output_shape[0].size();
  for (size_t i = 1; i < num; i++) {
    size_t tmp_dim_num = output_shape[i].size();
    if (dim_num != tmp_dim_num) {
      GELOGE(FAILED, "CheckDims fail: dim_num not equal, output_0:%zu, output_%zu:%zu.", dim_num, i, tmp_dim_num);
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
        GELOGE(FAILED, "CheckDims fail: dim_value not equal, dim_index=%zu, dim_value_0:%ld, dim_value_%zu:%ld.", i,
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
/// @return ge::NodePtr
///
NodePtr MultiBatchPass::CreateSwitchCaseNode(ComputeGraphPtr &graph, const std::string &name,
                                             const OutDataAnchorPtr &pred_value,
                                             const std::vector<std::vector<int64_t>> &batch_shape) {
  OpDescPtr op_desc = MakeShared<OpDesc>(name, STREAMSWITCHN);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc fail, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }

  GELOGI("Create StreamSwitchN op:%s.", name.c_str());
  OpDescPtr pred_desc = pred_value->GetOwnerNode()->GetOpDesc();
  if (pred_desc == nullptr) {
    GELOGE(FAILED, "Get pred_desc fail, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }
  if (op_desc->AddInputDesc(pred_desc->GetOutputDesc(pred_value->GetIdx())) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "AddInputDesc fail, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }

  NodePtr switch_case_node = graph->AddNode(op_desc);
  if (switch_case_node == nullptr) {
    GELOGE(FAILED, "Create node fail, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }

  uint32_t batch_num = static_cast<uint32_t>(batch_shape.size());
  if (!AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGE(FAILED, "set attr ATTR_NAME_BATCH_NUM fail, StreamSwitchN:%s.", name.c_str());
    return nullptr;
  }
  for (uint32_t i = 0; i < batch_num; i++) {
    const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    if (!AttrUtils::SetListInt(op_desc, attr_name, batch_shape[i])) {
      GELOGE(FAILED, "set attr ATTR_NAME_PRED_VALUE fail, StreamSwitchN:%s.", name.c_str());
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
Status MultiBatchPass::BypassSwitchN(NodePtr &switch_n_node, NodePtr &switch_case) {
  InDataAnchorPtr in_data_anchor = switch_n_node->GetInDataAnchor(SWITCH_DATA_INPUT);
  if (in_data_anchor == nullptr) {
    GELOGE(FAILED, "Check in_data_anchor fail, SwitchN:%s.", switch_n_node->GetName().c_str());
    return FAILED;
  }
  OutDataAnchorPtr peer_data_anchor = in_data_anchor->GetPeerOutAnchor();
  if (peer_data_anchor == nullptr) {
    GELOGE(FAILED, "Check peer_data_anchor fail, SwitchN:%s.", switch_n_node->GetName().c_str());
    return FAILED;
  }
  NodePtr data_input = peer_data_anchor->GetOwnerNode();

  // Remove SwitchN data input
  if (GraphUtils::RemoveEdge(peer_data_anchor, in_data_anchor) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Remove SwitchN in_data_edge fail, %s->%s.", data_input->GetName().c_str(),
           switch_n_node->GetName().c_str());
    return FAILED;
  }
  if (GraphUtils::AddEdge(data_input->GetOutControlAnchor(), switch_case->GetInControlAnchor()) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add StreamSwitchN in_control_edge fail, %s->%s.", data_input->GetName().c_str(),
           switch_case->GetName().c_str());
    return FAILED;
  }

  // Add SwitchCase control output
  for (OutDataAnchorPtr &out_data_anchor : switch_n_node->GetAllOutDataAnchors()) {
    for (InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      NodePtr data_output = peer_in_anchor->GetOwnerNode();
      if ((GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor) != GRAPH_SUCCESS) ||
          (GraphUtils::AddEdge(peer_data_anchor, peer_in_anchor) != GRAPH_SUCCESS)) {
        GELOGE(FAILED, "Bypass SwitchN data_edge fail, %s->%s->%s.", data_input->GetName().c_str(),
               switch_n_node->GetName().c_str(), data_output->GetName().c_str());
        return FAILED;
      }
      if (GraphUtils::AddEdge(switch_case->GetOutControlAnchor(), data_output->GetInControlAnchor()) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Add SwitchCase out_control_edge fail, %s->%s.", switch_case->GetName().c_str(),
               data_output->GetName().c_str());
        return FAILED;
      }
    }
  }

  bypass_nodes_.emplace_back(switch_n_node);
  GELOGI("Bypass SwitchN node %s success.", switch_n_node->GetName().c_str());
  return SUCCESS;
}

///
/// @brief Attach stream_label & batch_label for batch branch
/// @param [in] switch_case_node
/// @return Status
///
Status MultiBatchPass::AttachLabel(NodePtr &switch_case_node) {
  std::vector<std::string> stream_label_list;
  for (uint32_t i = 0; i < static_cast<uint32_t>(batch_head_nodes_.size()); i++) {
    if (AttachBatchLabel(i) != SUCCESS) {
      GELOGE(FAILED, "AttachBatchLabel fail, batch_idx=%u", i);
      return FAILED;
    }

    const std::string stream_label = "stream_label_batch_" + std::to_string(i);
    if (AttachStreamLabel(i, stream_label) != SUCCESS) {
      GELOGE(FAILED, "AttachStreamLabel fail, stream_label=%s", stream_label.c_str());
      return FAILED;
    }
    stream_label_list.emplace_back(stream_label);
  }

  return SetActiveLabelList(switch_case_node, stream_label_list);
}

///
/// @brief Attach batch_label for batch branch
/// @param [in] batch_idx
/// @return Status
///
Status MultiBatchPass::AttachBatchLabel(uint32_t batch_idx) {
  std::stack<NodePtr> nodes;
  for (auto &node : batch_head_nodes_[batch_idx]) {
    nodes.push(node);
  }

  const std::string batch_label = "Batch_" + std::to_string(batch_idx);
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
        GELOGE(FAILED, "get attr ATTR_NAME_BATCH_LABEL fail, node: %s.", cur_desc->GetName().c_str());
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
      GELOGE(FAILED, "set attr ATTR_NAME_BATCH_LABEL fail, node:%s.", cur_desc->GetName().c_str());
      return FAILED;
    }

    for (auto &out_node : cur_node->GetOutAllNodes()) {
      OpDescPtr op_desc = out_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const std::string type = op_desc->GetType();
      if ((type == STREAMMERGE) && (op_desc->HasAttr(ATTR_INSERT_BY_MBATCH))) {
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
  for (auto &node : batch_head_nodes_[batch_idx]) {
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
      GELOGE(FAILED, "SetStreamLabel fail, node:%s.", cur_node->GetName().c_str());
      return FAILED;
    }

    for (auto &out_node : cur_node->GetOutAllNodes()) {
      nodes.push(out_node);
    }

    (void)handled_nodes.insert(cur_node);
  }

  return SUCCESS;
}
}  // namespace ge
