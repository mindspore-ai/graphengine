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

#include "while_label_maker.h"

#include "common/util.h"
#include "common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace ge {
constexpr uint8_t kCondOutputNum = 1;
constexpr uint8_t kCondOutputIndex = 0;
constexpr uint8_t kCondBranchIndex = 0;
constexpr uint8_t kBodyBranchIndex = 1;

/**
 * @ingroup ge
 * @brief Make label node to functional call.
 * @param [in/out] label_index: serial id for whole graph.
 * @return: 0 for success / others for fail
 */
Status WhileOpLabelMaker::Run(uint32_t &label_index) {
  GE_CHECK_NOTNULL(parent_node_);
  GE_CHECK_NOTNULL(parent_graph_);

  OpDescPtr while_desc = parent_node_->GetOpDesc();
  GE_CHECK_NOTNULL(while_desc);

  std::string cond_name = while_desc->GetSubgraphInstanceName(kCondBranchIndex);
  std::string body_name = while_desc->GetSubgraphInstanceName(kBodyBranchIndex);
  if (cond_name.empty() || body_name.empty()) {
    GELOGE(INTERNAL_ERROR, "Node: %s has invalid subgraph, cond branch: %s, body branch: %s.",
           while_desc->GetName().c_str(), cond_name.c_str(), body_name.c_str());
    return FAILED;
  }

  ComputeGraphPtr cond_graph = parent_graph_->GetSubgraph(cond_name);
  ComputeGraphPtr body_graph = parent_graph_->GetSubgraph(body_name);
  GE_CHECK_NOTNULL(cond_graph);
  GE_CHECK_NOTNULL(body_graph);

  const uint32_t cond_enter_index = label_index++;
  const uint32_t body_enter_index = label_index++;
  const uint32_t body_leave_index = label_index++;
  const std::string cond_enter_name = parent_node_->GetName() + "/CondLabelSet";   // rtLabelSet
  const std::string cond_leave_name = parent_node_->GetName() + "/LabelSwitch";    // rtLabelSwitchByIndex
  const std::string body_enter_name = parent_node_->GetName() + "/EnterLabelSet";  // rtLabelSet
  const std::string goto_leave_name = parent_node_->GetName() + "/LabelGoto";      // rtLabelGoto
  const std::string body_leave_name = parent_node_->GetName() + "/LeaveLabelSet";  // rtLabelSet

  if (AddLabelSetEnter(cond_graph, cond_enter_name, cond_enter_index) == nullptr) {
    GELOGE(INTERNAL_ERROR, "Subgraph: %s add label set failed.", cond_graph->GetName().c_str());
    return FAILED;
  }

  if (AddLabelSetEnter(body_graph, body_enter_name, body_enter_index) == nullptr) {
    GELOGE(INTERNAL_ERROR, "Subgraph: %s add label set failed.", body_graph->GetName().c_str());
    return FAILED;
  }

  if (AddLabelGotoLeave(body_graph, goto_leave_name, cond_enter_index) == nullptr) {
    GELOGE(INTERNAL_ERROR, "Subgraph: %s add label goto failed.", body_graph->GetName().c_str());
    return FAILED;
  }

  if (AddLabelSetLeave(body_graph, body_leave_name, body_leave_index) == nullptr) {
    GELOGE(INTERNAL_ERROR, "Subgraph: %s add label set failed.", body_graph->GetName().c_str());
    return FAILED;
  }

  NodePtr cond_out_node = cond_graph->FindNode(NODE_NAME_NET_OUTPUT);
  GE_CHECK_NOTNULL(cond_out_node);
  OpDescPtr cond_out_desc = cond_out_node->GetOpDesc();
  GE_CHECK_NOTNULL(cond_out_desc);

  GeTensorDesc pred_desc = cond_out_desc->GetInputDesc(kCondOutputIndex);
  GeTensorDesc cond_desc(GeShape(pred_desc.GetShape().GetDims()), pred_desc.GetFormat(), DT_INT32);

  // false ==> 0 ==> switch_labels[0] ==> body_leave_index
  // true  ==> 1 ==> switch_labels[1] ==> body_enter_name
  const std::vector<uint32_t> switch_labels = {body_leave_index, body_enter_index};
  NodePtr switch_node = AddLabelSwitchLeave(cond_graph, cond_leave_name, cond_desc, switch_labels);
  if (switch_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "Subgraph: %s add label switch failed.", cond_graph->GetName().c_str());
    return FAILED;
  }

  // link Data input.
  const auto &all_in_data = cond_out_node->GetAllInDataAnchors();
  if (all_in_data.size() != kCondOutputNum) {
    GELOGE(FAILED, "Node: %s Cond sbugraph output size:%zu should equal size:%u.", switch_node->GetName().c_str(),
           all_in_data.size(), kCondOutputNum);
    return FAILED;
  }

  InDataAnchorPtr in_anchor = all_in_data.at(kCondOutputIndex);
  GE_CHECK_NOTNULL(in_anchor);
  if (GraphUtils::AddEdge(in_anchor->GetPeerOutAnchor(), switch_node->GetInDataAnchor(kCondOutputIndex)) != SUCCESS) {
    GELOGE(FAILED, "Node: %s Add pred data input failed.", switch_node->GetName().c_str());
    return FAILED;
  }

  GELOGI("Node: %s assign label success.", while_desc->GetName().c_str());
  return SUCCESS;
}

REGISTER_LABEL_MAKER(WHILE, WhileOpLabelMaker);
REGISTER_LABEL_MAKER(_WHILE, WhileOpLabelMaker);
REGISTER_LABEL_MAKER(STATELESSWHILE, WhileOpLabelMaker);
}  // namespace ge
