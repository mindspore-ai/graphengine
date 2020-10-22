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

#include "graph/passes/for_pass.h"
#include "common/ge/ge_util.h"
#include "common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace {
  const uint32_t kWhileIInputIndex = 0;
  const uint32_t kWhileAbsDeltaInputIndex = 1;
  const uint32_t kWhileRangeInputIndex = 2;
  const uint32_t kWhileStartInputIndex = 3;
  const uint32_t kWhileDeltaInputIndex = 4;
  const uint32_t kWhileDataInputIndex = 5;
  const uint32_t kSubgraphLoopVarInputIndex = 0;
  const uint32_t kSubgraphInputIndex = 1;
  const uint32_t kWhileOutputIndex = 5;
  const std::string kAbs = "Abs";
}

namespace ge {
Status ForPass::Run(NodePtr &node) {
  if (node->GetType() != FOR) {
    GELOGD("no need for_pass for node %s.", node->GetName().c_str());
    return SUCCESS;
  }

  GELOGI("Begin to transfer for_op to while_op, node:%s.", node->GetName().c_str());
  ComputeGraphPtr graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  ComputeGraphPtr root_graph = GraphUtils::FindRootGraph(graph);
  GE_CHECK_NOTNULL(root_graph);

  ForInfo for_info;
  GE_CHK_STATUS_RET(BuildForInfo(root_graph, node, for_info),
                    "Build ForInfo failed, node:%s.", node->GetName().c_str());

  WhileInfo while_info;
  GE_CHK_STATUS_RET(TranWhileInfo(graph, for_info, while_info),
                    "Transfer WhileInfo from ForInfo failed, node:%s.", node->GetName().c_str());

  ComputeGraphPtr cond_graph = BuildCondGraph(while_info);
  if ((cond_graph == nullptr) || (root_graph->AddSubgraph(cond_graph) != GRAPH_SUCCESS)) {
    GELOGE(FAILED, "Add while_cond_graph failed, node:%s.", node->GetName().c_str());
    return FAILED;
  }

  ComputeGraphPtr body_graph = BuildBodyGraph(while_info);
  if ((body_graph == nullptr) || (root_graph->AddSubgraph(body_graph) != GRAPH_SUCCESS)) {
    GELOGE(FAILED, "Add while_body_graph failed, node:%s.", node->GetName().c_str());
    return FAILED;
  }

  GE_CHK_STATUS_RET(UpdateForBodyInputMapping(while_info),
                    "Update InputMapping for for-body-graph failed, node:%s.", node->GetName().c_str());

  // for node has and only has one subgraph
  GE_CHECK_NOTNULL(node->GetOpDesc());
  node->GetOpDesc()->RemoveSubgraphInstanceName(node->GetOpDesc()->GetSubgraphInstanceName(0));

  GELOGI("Transfer for_op to while_op succ, node:%s.", node->GetName().c_str());
  return IsolateAndDeleteNode(node, std::vector<int>());
}

///
/// @brief Build for_info
/// @param [in] root_graph
/// @param [in] node
/// @param [out] for_info
/// @return Status
///
Status ForPass::BuildForInfo(const ComputeGraphPtr &root_graph, const NodePtr &node, ForInfo &for_info) {
  GELOGI("Begin to build for_info for node %s.", node->GetName().c_str());

  OutDataAnchorPtr start = FindInputWithIndex(node, FOR_START_INPUT);
  OutDataAnchorPtr limit = FindInputWithIndex(node, FOR_LIMIT_INPUT);
  OutDataAnchorPtr delta = FindInputWithIndex(node, FOR_DELTA_INPUT);
  if ((start == nullptr) || (limit == nullptr) || (delta == nullptr)) {
    GELOGE(FAILED, "BuildForInfo for %s failed: start/limit/delta is NULL.", node->GetName().c_str());
    return FAILED;
  }

  std::vector<OutDataAnchorPtr> data_inputs;
  std::vector<std::vector<InDataAnchorPtr>> data_outputs;
  std::vector<OutControlAnchorPtr> ctrl_inputs;
  std::vector<InControlAnchorPtr> ctrl_outputs;
  if (FindInputsAndOutputs(node, data_inputs, data_outputs, ctrl_inputs, ctrl_outputs) != SUCCESS) {
    GELOGE(FAILED, "BuildForInfo for %s failed: find inputs/outputs failed.", node->GetName().c_str());
    return FAILED;
  }
  NodeUtils::UnlinkAll(*node);

  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // For node has and only has one sub_graph
  std::string for_body_name = op_desc->GetSubgraphInstanceName(0);
  if (for_body_name.empty()) {
    GELOGE(FAILED, "BuildForInfo for %s failed: sub_graph_name is empty.", node->GetName().c_str());
    return FAILED;
  }
  ComputeGraphPtr for_body = root_graph->GetSubgraph(for_body_name);
  if (for_body == nullptr) {
    GELOGE(FAILED, "BuildForInfo for %s failed: for_body_graph is NULL.", node->GetName().c_str());
    return FAILED;
  }

  for_info.for_node = node;
  for_info.start = start;
  for_info.limit = limit;
  for_info.delta = delta;
  for_info.body_name = for_body_name;
  for_info.for_body = for_body;
  for_info.data_inputs = std::move(data_inputs);
  for_info.data_outputs = std::move(data_outputs);
  for_info.ctrl_inputs = std::move(ctrl_inputs);
  for_info.ctrl_outputs = std::move(ctrl_outputs);

  GELOGI("Build for_info for node %s succ.", node->GetName().c_str());
  return SUCCESS;
}

///
/// @brief Find input with index for For node
/// @param [in] node
/// @param [in] index
/// @return OutDataAnchorPtr
///
OutDataAnchorPtr ForPass::FindInputWithIndex(const NodePtr &node, uint32_t index) {
  if (node == nullptr) {
    GELOGE(FAILED, "FindInputWithIndex failed: node is NULL.");
    return nullptr;
  }

  InDataAnchorPtr in_data_anchor = node->GetInDataAnchor(index);
  if (in_data_anchor == nullptr) {
    GELOGE(FAILED, "FindInputWithIndex %s:%u failed: in_data_anchor is NULL.", node->GetName().c_str(), index);
    return nullptr;
  }

  OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    GELOGE(FAILED, "FindInputWithIndex %s:%u failed: peer_out_anchor is NULL.", node->GetName().c_str(), index);
    return nullptr;
  }

  return peer_out_anchor;
}

///
/// @brief Find inputs / outputs for for node
/// @param [in] node
/// @param [out] data_inputs
/// @param [out] data_outputs
/// @param [out] ctrl_inputs
/// @param [out] ctrl_outputs
/// @return Status
///
Status ForPass::FindInputsAndOutputs(const NodePtr &node, std::vector<OutDataAnchorPtr> &data_inputs,
                                     std::vector<std::vector<ge::InDataAnchorPtr>> &data_outputs,
                                     std::vector<ge::OutControlAnchorPtr> &ctrl_inputs,
                                     std::vector<ge::InControlAnchorPtr> &ctrl_outputs) {
  GE_CHECK_NOTNULL(node);

  uint32_t input_data_num = node->GetAllInDataAnchorsSize();
  for (uint32_t index = FOR_DATA_INPUT; index < input_data_num; index++) {
    InDataAnchorPtr in_data_anchor = node->GetInDataAnchor(index);
    if (in_data_anchor == nullptr) {
      GELOGE(FAILED, "FindInputWithIndex %s:%u failed: in_data_anchor is NULL.", node->GetName().c_str(), index);
      return FAILED;
    }
    GE_IF_BOOL_EXEC(in_data_anchor->GetPeerOutAnchor() == nullptr,
                    GELOGW("Get null input by index %d from node %s ",
                           in_data_anchor->GetIdx(), node->GetName().c_str());
                    continue);
    data_inputs.emplace_back(in_data_anchor->GetPeerOutAnchor());
  }

  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    std::vector<ge::InDataAnchorPtr> peer_in_data_anchors;
    for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      peer_in_data_anchors.emplace_back(peer_in_data_anchor);
    }
    data_outputs.emplace_back(peer_in_data_anchors);
  }

  InControlAnchorPtr in_ctrl_anchor = node->GetInControlAnchor();
  GE_CHECK_NOTNULL(in_ctrl_anchor);
  for (auto &peer_out_ctrl_anchor : in_ctrl_anchor->GetPeerOutControlAnchors()) {
    ctrl_inputs.emplace_back(peer_out_ctrl_anchor);
  }

  OutControlAnchorPtr out_ctrl_anchor = node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(out_ctrl_anchor);
  for (auto &peer_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
    ctrl_outputs.emplace_back(peer_in_ctrl_anchor);
  }

  return SUCCESS;
}

///
/// @brief Transfer while_info from for_info
/// @param [in] graph
/// @param [in] for_info
/// @param [out] while_info
/// @return Status
///
Status ForPass::TranWhileInfo(const ComputeGraphPtr &graph, const ForInfo &for_info, WhileInfo &while_info) {
  std::string for_name = for_info.for_node->GetName();
  GELOGI("Begin to transfer for_info to while_info, node:%s.", for_name.c_str());

  std::string i_name = for_name + "_i";
  NodePtr i_node = graph->AddNode(CreateConstDesc(i_name, 0));
  if (i_node == nullptr) {
    GELOGE(FAILED, "TranWhileInfo failed: create i_node failed.");
    return FAILED;
  }
  AddRePassNode(i_node);

  std::string identity_name = i_name + "_Identity";
  NodePtr identity_node = graph->AddNode(CreateOpDesc(identity_name, IDENTITY, true));
  // Const node has and only has one output, Identity node has and only has one input
  if ((identity_node == nullptr) ||
      (GraphUtils::AddEdge(i_node->GetOutDataAnchor(0), identity_node->GetInDataAnchor(0)) != GRAPH_SUCCESS)) {
    GELOGE(FAILED, "TranWhileInfo failed: Add data-edge %s:0->%s:0 failed.", i_name.c_str(), identity_name.c_str());
    return FAILED;
  }
  AddRePassNode(identity_node);

  // Identity node has and only has one output
  OutDataAnchorPtr i_input = identity_node->GetOutDataAnchor(0);
  if (i_input == nullptr) {
    GELOGE(FAILED, "TranWhileInfo failed: i_input is NULL.");
    return FAILED;
  }

  OutDataAnchorPtr range_input = nullptr;
  OutDataAnchorPtr abs_delta_input = nullptr;
  if (CreateLoopInput(graph, for_info, range_input, abs_delta_input) != SUCCESS) {
    GELOGE(FAILED, "TranWhileInfo failed: create loop input failed.");
    return FAILED;
  }

  BuildWhileInfo(for_info, i_input, range_input, abs_delta_input, while_info);

  if (InsertWhileNode(graph, for_name + "_While", while_info) != SUCCESS) {
    GELOGE(FAILED, "TranWhileInfo failed: insert while node failed.");
    return FAILED;
  }

  GELOGI("Transfer for_info to while_info succ, for_node:%s, while_node:%s.",
         for_name.c_str(), while_info.while_node->GetName().c_str());
  return SUCCESS;
}

///
/// @brief Create const op_desc
/// @param [in] name
/// @param [in] value
/// @return OpDescPtr
///
OpDescPtr ForPass::CreateConstDesc(const std::string &name, int32_t value) {
  OpDescPtr const_op_desc = MakeShared<OpDesc>(name, CONSTANT);
  if (const_op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc failed, const:%s.", name.c_str());
    return nullptr;
  }

  GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_INT32);
  GeTensorPtr const_value = MakeShared<GeTensor>(data_desc, reinterpret_cast<uint8_t *>(&value), sizeof(int32_t));
  if (const_value == nullptr) {
    GELOGE(FAILED, "Create tensor failed, const:%s.", name.c_str());
    return nullptr;
  }

  if (!AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value)) {
    GELOGE(FAILED, "Set ATTR_NAME_WEIGHTS failed, const:%s.", name.c_str());
    return nullptr;
  }

  if (const_op_desc->AddOutputDesc("y", data_desc) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add output desc failed, const:%s.", name.c_str());
    return nullptr;
  }

  return const_op_desc;
}

///
/// @brief Create loop node
/// @param [in] graph
/// @param [in] for_info
/// @param [out] range_input
/// @param [out] abs_delta_input
/// @return Status
///
Status ForPass::CreateLoopInput(const ComputeGraphPtr &graph, const ForInfo &for_info,
                                OutDataAnchorPtr &range_input, OutDataAnchorPtr &abs_delta_input) {
  std::string for_name = for_info.for_node->GetName();
  GELOGD("Begin to create loop_count input, node:%s", for_name.c_str());

  OutDataAnchorPtr start = for_info.start;
  OutDataAnchorPtr limit = for_info.limit;
  OutDataAnchorPtr delta = for_info.delta;

  std::string sub_name_0 = for_name + "_Sub_0";
  std::string abs_name_0 = for_name + "_Abs_0";
  std::string abs_name_1 = for_name + "_Abs_1";

  // i * |delta| < |limit-start|
  PartialGraphBuilder graph_builder;
  graph_builder.SetOwnerGraph(graph)
               .AddExistNode(for_info.start->GetOwnerNode())
               .AddExistNode(for_info.limit->GetOwnerNode())
               .AddExistNode(for_info.delta->GetOwnerNode())
               .AddNode(CreateOpDesc(sub_name_0, SUB, false))
               .AddNode(CreateOpDesc(abs_name_0, kAbs, true))
               .AddNode(CreateOpDesc(abs_name_1, kAbs, true))
               .AddDataLink(delta->GetOwnerNode()->GetName(), delta->GetIdx(), abs_name_0, 0)
               .AddDataLink(limit->GetOwnerNode()->GetName(), limit->GetIdx(), sub_name_0, 0)
               .AddDataLink(start->GetOwnerNode()->GetName(), start->GetIdx(), sub_name_0, 1)
               .AddDataLink(sub_name_0, 0, abs_name_1, 0);

  graphStatus error_code = GRAPH_SUCCESS;
  std::string error_msg;
  if ((graph_builder.Build(error_code, error_msg) == nullptr) || (error_code != GRAPH_SUCCESS)) {
    GELOGE(FAILED, "Create loop_count node failed: error_code:%u, error_msg:%s.", error_code, error_msg.c_str());
    return FAILED;
  }

  // Add repass_nodes
  for (auto &node : graph_builder.GetAllNodes()) {
    AddRePassNode(node);
  }

  NodePtr abs_delta_node = graph_builder.GetNode(abs_name_0);
  NodePtr loop_count_node = graph_builder.GetNode(abs_name_1);
  if ((abs_delta_node == nullptr) || (loop_count_node == nullptr)) {
    GELOGE(FAILED, "Create loop node failed: node is NULL.");
    return FAILED;
  }

  GELOGD("Create loop_range input succ, node:%s", for_name.c_str());
  // abs_node has and only has one output
  abs_delta_input = abs_delta_node->GetOutDataAnchor(0);
  range_input = loop_count_node->GetOutDataAnchor(0);

  return SUCCESS;
}

///
/// @brief Create op_desc
/// @param [in] name
/// @param [in] type
/// @param [in] io_equal_flag
/// @return OpDescPtr
///
OpDescPtr ForPass::CreateOpDesc(const std::string &name, const std::string &type, bool io_equal_flag) {
  OpDescBuilder op_desc_builder(name, type);
  if (io_equal_flag) {
    op_desc_builder.AddInput("x")
                   .AddOutput("y");
  } else {
    op_desc_builder.AddInput("x1")
                   .AddInput("x2")
                   .AddOutput("y");
  }

  return op_desc_builder.Build();
}

///
/// @brief Build while-info
/// @param [in] for_info
/// @param [in] i_input
/// @param [in] range_input
/// @param [in] abs_delta_input
/// @param [out] while_info
/// @return void
///
void ForPass::BuildWhileInfo(const ForInfo &for_info, const OutDataAnchorPtr &i_input,
                             const OutDataAnchorPtr &range_input, const OutDataAnchorPtr &abs_delta_input,
                             WhileInfo &while_info) {
  while_info.i = i_input;
  while_info.abs_delta = abs_delta_input;
  while_info.range = range_input;
  while_info.start = for_info.start;
  while_info.delta = for_info.delta;
  while_info.for_body_name = for_info.body_name;
  while_info.for_body = for_info.for_body;
  while_info.data_inputs.emplace_back(while_info.i);
  while_info.data_inputs.emplace_back(while_info.abs_delta);
  while_info.data_inputs.emplace_back(while_info.range);
  while_info.data_inputs.emplace_back(while_info.start);
  while_info.data_inputs.emplace_back(while_info.delta);
  for (auto &item : for_info.data_inputs) {
    while_info.data_inputs.emplace_back(item);
  }
  for (auto &item : for_info.data_outputs) {
    while_info.data_outputs.emplace_back(item);
  }
  for (auto &item : for_info.ctrl_inputs) {
    while_info.ctrl_inputs.emplace_back(item);
  }
  for (auto &item : for_info.ctrl_outputs) {
    while_info.ctrl_outputs.emplace_back(item);
  }
}

///
/// @brief Insert while_node
/// @param [in] graph
/// @param [in] name
/// @param [in&out] while_info
/// @return Status
///
Status ForPass::InsertWhileNode(const ComputeGraphPtr &graph, const std::string &name, WhileInfo &while_info) {
  GELOGD("Begin to create while node, name:%s.", name.c_str());

  size_t arg_num = while_info.data_inputs.size();
  OpDescBuilder op_desc_builder(name, WHILE);
  OpDescPtr op_desc = op_desc_builder.AddDynamicInput("input", arg_num).AddDynamicOutput("output", arg_num).Build();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create while op_desc failed, name:%s.", name.c_str());
    return FAILED;
  }
  NodePtr while_node = graph->AddNode(op_desc);
  if (while_node == nullptr) {
    GELOGE(FAILED, "Create while node failed, name:%s.", name.c_str());
    return FAILED;
  }
  AddRePassNode(while_node);

  while_info.while_node = while_node;
  if (BuildWhileLink(while_info) != SUCCESS) {
    GELOGE(FAILED, "Build while link-edge failed, name:%s.", name.c_str());
    return FAILED;
  }

  GELOGD("Create while node succ, name:%s.", name.c_str());
  return SUCCESS;
}

///
/// @brief Build while link-edge
/// @param [in] while_info
/// @return Status
///
Status ForPass::BuildWhileLink(const WhileInfo &while_info) {
  NodePtr while_node = while_info.while_node;
  GE_CHECK_NOTNULL(while_node);

  size_t input_num = while_info.data_inputs.size();
  for (size_t i = 0; i < input_num; i++) {
    InDataAnchorPtr in_data_anchor = while_node->GetInDataAnchor(i);
    GE_CHECK_NOTNULL(in_data_anchor);
    OutDataAnchorPtr peer_out_anchor = while_info.data_inputs[i];
    if (peer_out_anchor == nullptr) {
      continue;
    }
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(peer_out_anchor, in_data_anchor),
                            "Add data-edge %s:%d->%s:%d failed.",
                            peer_out_anchor->GetOwnerNode()->GetName().c_str(), peer_out_anchor->GetIdx(),
                            while_node->GetName().c_str(), i);
  }

  size_t output_num = while_info.data_outputs.size();
  for (size_t i = 0; i < output_num; i++) {
    OutDataAnchorPtr out_data_anchor = while_node->GetOutDataAnchor(static_cast<int>(i + kWhileOutputIndex));
    GE_CHECK_NOTNULL(out_data_anchor);
    for (auto &peer_in_anchor : while_info.data_outputs[i]) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(out_data_anchor, peer_in_anchor),
                              "Add data-edge %s:%d->%s:%d failed.",
                              while_node->GetName().c_str(), i + kWhileOutputIndex,
                              peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
    }
  }

  InControlAnchorPtr in_ctrl_anchor = while_node->GetInControlAnchor();
  GE_CHECK_NOTNULL(in_ctrl_anchor);
  for (auto &peer_out_anchor : while_info.ctrl_inputs) {
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(peer_out_anchor, in_ctrl_anchor),
                            "Add ctrl-edge %s->%s failed.",
                            peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                            in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
  }

  OutControlAnchorPtr out_ctrl_anchor = while_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(out_ctrl_anchor);
  for (auto &peer_in_anchor : while_info.ctrl_outputs) {
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(out_ctrl_anchor, peer_in_anchor),
                            "Add ctrl-edge %s->%s failed.",
                            out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                            peer_in_anchor->GetOwnerNode()->GetName().c_str());
  }

  return SUCCESS;
}

///
/// @brief Build cond_graph for while_node
/// @param [in&out] while_info
/// @return ComputeGraphPtr
///
ComputeGraphPtr ForPass::BuildCondGraph(WhileInfo &while_info) {
  std::string cond_name = while_info.for_body_name + "_Cond";
  CompleteGraphBuilder graph_builder(cond_name);

  // Add parent node
  graph_builder.SetParentNode(while_info.while_node);

  // Add Node
  const std::string mul_name = "Mul";
  graph_builder.AddNode(CreateOpDesc(mul_name, MUL, false));
  const std::string less_name = "Less";
  graph_builder.AddNode(CreateOpDesc(less_name, LESS, false));

  // Set Input
  graph_builder.SetInput(kWhileIInputIndex, { mul_name }, { 0 })
               .SetInput(kWhileAbsDeltaInputIndex, { mul_name }, { 1 })
               .SetInput(kWhileRangeInputIndex, { less_name }, { 1 })
               .SetUselessInput(kWhileStartInputIndex)
               .SetUselessInput(kWhileDeltaInputIndex);
  size_t input_num = while_info.data_inputs.size();
  for (size_t i = kWhileDataInputIndex; i < input_num; i++) {
    graph_builder.SetUselessInput(i);
  }

  // Add Output
  graph_builder.AddOutput(less_name, 0);

  // Add Edges
  graph_builder.AddDataLink(mul_name, 0, less_name, 0);

  // Add Input-Mapping
  std::map<uint32_t, uint32_t> input_mapping;
  for (size_t i = 0; i < input_num; i++) {
    input_mapping[i] = i;
  }
  graph_builder.SetInputMapping(input_mapping);

  graphStatus error_code = GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr cond_graph = graph_builder.Build(error_code, error_msg);
  if (cond_graph == nullptr) {
    GELOGE(FAILED, "Build cond_graph failed: error_code:%u, error_msg:%s.", error_code, error_msg.c_str());
    return nullptr;
  }

  size_t index = while_info.while_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  while_info.while_node->GetOpDesc()->AddSubgraphName(ATTR_NAME_WHILE_COND);
  while_info.while_node->GetOpDesc()->SetSubgraphInstanceName(index, cond_name);
  while_info.while_cond = cond_graph;
  return cond_graph;
}

///
/// @brief Build body_graph for while_node
/// @param [in&out] while_info
/// @return ComputeGraphPtr
///
ComputeGraphPtr ForPass::BuildBodyGraph(WhileInfo &while_info) {
  std::string body_name = while_info.for_body_name + "_Body";
  CompleteGraphBuilder graph_builder(body_name);

  // Add parent node
  graph_builder.SetParentNode(while_info.while_node);

  // Add calculation nodes
  std::string const_name = "Const";
  std::string add_name_0 = "Add_0";
  std::string mul_name = "Mul";
  std::string add_name_1 = "Add_1";
  graph_builder.AddNode(CreateConstDesc(const_name, 1))
               .AddNode(CreateOpDesc(add_name_0, ADD, false))
               .AddNode(CreateOpDesc(mul_name, MUL, false))
               .AddNode(CreateOpDesc(add_name_1, ADD, false));

  // Add Subgraph node
  auto input_num = static_cast<uint32_t>(while_info.data_inputs.size());
  std::string sub_graph_node_name = while_info.for_body_name;
  uint32_t sub_graph_input_num = input_num - kWhileDataInputIndex + kSubgraphInputIndex;
  auto sub_graph_output_num = static_cast<uint32_t>(while_info.data_outputs.size());
  graph_builder.AddNode(CreateSubgraphOpDesc(sub_graph_node_name, sub_graph_input_num, sub_graph_output_num));

  // Set Input
  graph_builder.SetInput(kWhileIInputIndex, { add_name_0, mul_name }, { 0, 0 })
               .SetUselessInput(kWhileAbsDeltaInputIndex)
               .SetUselessInput(kWhileRangeInputIndex)
               .SetInput(kWhileStartInputIndex, { add_name_1 }, { 0 })
               .SetInput(kWhileDeltaInputIndex, { mul_name }, { 1 });
  for (uint32_t i = 0; i < input_num - kWhileDataInputIndex; i++) {
    graph_builder.SetInput(i + kWhileDataInputIndex, { sub_graph_node_name }, { i + kSubgraphInputIndex });
  }

  // Add Outputs
  graph_builder.AddOutput(add_name_0, 0);
  for (uint32_t i = kWhileAbsDeltaInputIndex; i < kWhileDataInputIndex; i++) {
    graph_builder.AddOutput("Data_" + std::to_string(i), 0);
  }
  for (uint32_t i = 0; i < sub_graph_output_num; i++) {
    graph_builder.AddOutput(sub_graph_node_name, i);
  }

  // Add Edges
  graph_builder.AddDataLink(const_name, 0, add_name_0, 1)
               .AddDataLink(mul_name, 0, add_name_1, 1)
               .AddDataLink(add_name_1, 0, sub_graph_node_name, kSubgraphLoopVarInputIndex);

  // Add Input-Mapping
  std::map<uint32_t, uint32_t> input_mapping;
  for (size_t i = 0; i < input_num; i++) {
    input_mapping[i] = i;
  }
  graph_builder.SetInputMapping(input_mapping);

  // Add outputMapping
  std::map<uint32_t, uint32_t> output_mapping;
  for (size_t i = 0; i < sub_graph_output_num + kWhileOutputIndex; i++) {
    output_mapping[i] = i;
  }
  graph_builder.SetOutputMapping(output_mapping);

  graphStatus error_code = GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr body_graph = graph_builder.Build(error_code, error_msg);
  if (body_graph == nullptr) {
    GELOGE(FAILED, "Build body_graph failed: error_code:%u, error_msg:%s.", error_code, error_msg.c_str());
    return nullptr;
  }

  NodePtr sub_graph_node = graph_builder.GetNode(sub_graph_node_name);
  if (sub_graph_node == nullptr) {
    GELOGE(FAILED, "Get sub_graph_node failed: name:%s.", sub_graph_node_name.c_str());
    return nullptr;
  }
  while_info.sub_graph_node = sub_graph_node;

  size_t index = while_info.while_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  while_info.while_node->GetOpDesc()->AddSubgraphName(ATTR_NAME_WHILE_BODY);
  while_info.while_node->GetOpDesc()->SetSubgraphInstanceName(index, body_name);
  while_info.while_body = body_graph;
  return body_graph;
}

///
/// @brief Create op_desc for subgraph node
/// @param [in] name
/// @param [in] input_num
/// @param [in] output_num
/// @return OpDescPtr
///
OpDescPtr ForPass::CreateSubgraphOpDesc(const std::string &name, uint32_t input_num, uint32_t output_num) {
  OpDescBuilder op_desc_builder(name, PARTITIONEDCALL);
  op_desc_builder.AddDynamicInput("args", input_num)
                 .AddDynamicOutput("output", output_num);

  OpDescPtr op_desc = op_desc_builder.Build();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc for subgraph node failed, name:%s.", name.c_str());
    return nullptr;
  }

  size_t index = op_desc->GetSubgraphInstanceNames().size();
  op_desc->AddSubgraphName("f");
  op_desc->SetSubgraphInstanceName(index, name);
  return op_desc;
}

///
/// @brief Update InputMapping for for-body-graph
/// @param [in] while_info
/// @return Status
///
Status ForPass::UpdateForBodyInputMapping(const WhileInfo &while_info) {
  ComputeGraphPtr for_body = while_info.for_body;
  GE_CHECK_NOTNULL(for_body);

  // index_of_cur_graph_node_input -> index_of_new_graph_node_input
  std::map<uint32_t, uint32_t> input_mapping;
  size_t input_num = while_info.data_inputs.size() - kWhileDataInputIndex + FOR_DATA_INPUT;
  for (size_t i = 0; i < input_num; i++) {
    if (i == FOR_START_INPUT) {
      input_mapping[i] = i;
    } else if ((i == FOR_LIMIT_INPUT) || (i == FOR_DELTA_INPUT)) {
      continue;
    } else {
      input_mapping[i] = i - 2;
    }
  }
  for_body->UpdateInputMapping(input_mapping);
  for_body->SetParentNode(while_info.sub_graph_node);
  for_body->SetParentGraph(while_info.while_body);

  return SUCCESS;
}
}  // namespace ge

