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

#include "graph/passes/cond_remove_pass.h"
#include "common/op/ge_op_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"

namespace {
const uint32_t kConditionIndexNum = 1;
const uint32_t kElseBranchIndex = 1;
const uint32_t kTrueIndex = 1;
const uint32_t kFalseIndex = 0;
/// Extra 8 bytes store pointer of string
/// Extra 1 byte store '\0'
const int32_t kStrHeadLen = 9;
const int32_t kInvalidRetVal = -1;
}  // namespace

namespace ge {
Status CondRemovePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  ComputeGraphPtr graph = nullptr;
  OutDataAnchorPtr cond_out_anchor = nullptr;
  InDataAnchorPtr cond_in_anchor = nullptr;
  Status ret = GetCondInfo(node, graph, cond_out_anchor, cond_in_anchor);
  int32_t cond_index = 0;
  GELOGD("Handle cond remove for node %s.", node->GetOpDesc()->GetName().c_str());
  bool if_cond_const = CheckIfCondConstInput(cond_out_anchor, cond_in_anchor, cond_index);
  if (!if_cond_const || (cond_index < 0)) {
    return ge::SUCCESS;
  }
  ComputeGraphPtr chosen_graph = nullptr;
  const std::string &node_type = node->GetType();
  // Keep chosen branch
  if (kIfOpTypes.count(node_type) != 0) {
    ret = GetIfChosenBranch(node, static_cast<uint32_t>(cond_index), chosen_graph);
    if (ret != ge::SUCCESS) {
      return ge::FAILED;
    }
  } else if (kCaseOpTypes.count(node_type) != 0) {
    ret = GetCaseChosenBranch(node, static_cast<uint32_t>(cond_index), chosen_graph);
    if (ret != ge::SUCCESS) {
      return ge::FAILED;
    }
  } else {
    return ge::SUCCESS;
  }
  // Remove unused link from cond->node
  ret = RemoveDeadCondLink(static_cast<int32_t>(IF_COND_INPUT), node);
  if (ret != ge::SUCCESS) {
    return ge::FAILED;
  }
  // Copy If/Case node's relations to the new node
  ret = ReplaceIfCaseNodeWithPartitioncall(node, chosen_graph);
  if (ret != ge::SUCCESS) {
    return ge::FAILED;
  }
  // Isolate and delete the old node
  ret = IsolateAndDeleteNode(node, std::vector<int>());
  return ret;
}

Status CondRemovePass::RemoveDeadCondLink(const int32_t index, const NodePtr &node) {
  const auto &in_anchor = node->GetInDataAnchor(index);
  const auto &peerout_anchor = in_anchor->GetPeerOutAnchor();
  if (GraphUtils::RemoveEdge(peerout_anchor, in_anchor) != SUCCESS) {
    GELOGE(FAILED, "Remove edge from node %s index %d to node %s index %d.",
           peerout_anchor->GetOwnerNode()->GetName().c_str(), peerout_anchor->GetIdx(),
           in_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetIdx());
    return FAILED;
  }
  return SUCCESS;
}

Status CondRemovePass::GetCaseChosenBranch(const NodePtr &node, const uint32_t cond_index,
                                           ComputeGraphPtr &compute_graph) {
  uint32_t subgraph_names_size = static_cast<uint32_t>(node->GetOpDesc()->GetSubgraphInstanceNames().size());
  uint32_t cond_index_new = cond_index;
  if (subgraph_names_size == 0) {
    GELOGE(FAILED, "Node %s has none subgraph.", node->GetName().c_str());
    return ge::FAILED;
  }
  // If cond index is over the maimum subgraph number, choose the last subgraph
  if (cond_index >= subgraph_names_size) {
    cond_index_new = subgraph_names_size - 1;
  }
  const auto &chosen_branch_name = node->GetOpDesc()->GetSubgraphInstanceName(cond_index_new);
  if (chosen_branch_name.empty()) {
    GELOGE(FAILED, "Node %s has no subgraph, index is %u.", node->GetName().c_str(), cond_index_new);
    return ge::FAILED;
  }
  auto chosen_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph())->GetSubgraph(chosen_branch_name);
  compute_graph = chosen_graph;
  // Remove graph from node, in order for remove connection from this node to chosen branch
  node->GetOpDesc()->RemoveSubgraphInstanceName(chosen_branch_name);
  return ge::SUCCESS;
}

Status CondRemovePass::GetIfChosenBranch(const NodePtr &node, const uint32_t cond, ComputeGraphPtr &compute_graph) {
  uint32_t subgraph_names_size = static_cast<uint32_t>(node->GetOpDesc()->GetSubgraphInstanceNames().size());
  uint32_t cond_index_new = 0;
  if (subgraph_names_size == 0) {
    GELOGE(FAILED, "Node %s has none subgraph.", node->GetName().c_str());
    return ge::FAILED;
  }
  // If cond is false, else branch
  if (cond == 0) {
    cond_index_new = kElseBranchIndex;
  }
  const auto &chosen_branch_name = node->GetOpDesc()->GetSubgraphInstanceName(cond_index_new);
  if (chosen_branch_name.empty()) {
    GELOGE(FAILED, "Node %s has no subgraph, index is %u.", node->GetName().c_str(), cond_index_new);
    return ge::FAILED;
  }
  auto chosen_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph())->GetSubgraph(chosen_branch_name);
  if (chosen_graph == nullptr) {
    GELOGE(FAILED, "Can not find branch %s in node %s's parent graph %s.", chosen_branch_name.c_str(),
           node->GetName().c_str(), node->GetOwnerComputeGraph()->GetName().c_str());
    return ge::FAILED;
  }
  compute_graph = chosen_graph;
  // Remove graph from node, in order for remove connection from this node to chosen branch
  node->GetOpDesc()->RemoveSubgraphInstanceName(chosen_branch_name);
  return ge::SUCCESS;
}

int32_t CondRemovePass::GetCondIndex(const ConstGeTensorPtr &tensor) {
  if (tensor == nullptr) {
    return kInvalidRetVal;
  }
  const uint8_t *data_ptr = tensor->GetData().data();
  size_t tensor_size = tensor->GetData().size();
  const auto type = tensor->GetTensorDesc().GetDataType();
  GELOGD("Data type is %d, tensor_size is %zu.", type, tensor_size);
  switch (type) {
    case DT_STRING:
      return static_cast<int32_t>(((tensor_size - kStrHeadLen) > 0) ? kTrueIndex : kFalseIndex);
    case DT_BOOL:
      return static_cast<int32_t>(*reinterpret_cast<const bool *>(data_ptr));
    case DT_FLOAT:
      return static_cast<int32_t>(*reinterpret_cast<const float *>(data_ptr));
    case DT_DOUBLE:
      return static_cast<int32_t>(*reinterpret_cast<const double *>(data_ptr));
    case DT_INT8:
    case DT_UINT8:
      return static_cast<int32_t>(*data_ptr);
    case DT_FLOAT16:
    case DT_INT16:
    case DT_UINT16:
      return static_cast<int32_t>(*reinterpret_cast<const int16_t *>(data_ptr));
    case DT_INT32:
      return static_cast<int32_t>(*reinterpret_cast<const int32_t *>(data_ptr));
    case DT_UINT32:
      return *reinterpret_cast<const int32_t *>(data_ptr);
    case DT_INT64:
    case DT_UINT64:
      return static_cast<int32_t>(*reinterpret_cast<const int64_t *>(data_ptr));
    default:
      return static_cast<int32_t>(*data_ptr);
  }
}

bool CondRemovePass::CheckIfCondConstInput(const OutDataAnchorPtr &cond_out_anchor,
                                           const InDataAnchorPtr &cond_in_anchor, int32_t &cond_index) {
  // if pre or next anchor is null, return
  CHECK_FALSE_EXEC(cond_out_anchor != nullptr, return false);
  CHECK_FALSE_EXEC(cond_in_anchor != nullptr, return false);
  const auto &out_node = cond_out_anchor->GetOwnerNode();
  const auto &cur_node = cond_in_anchor->GetOwnerNode();
  OpDescPtr op_desc = cur_node->GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return false);
  GeTensorDesc cond_tensor = out_node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(cond_out_anchor->GetIdx()));
  GELOGI("Check if condition is const for node %s.", op_desc->GetName().c_str());
  if (kConstOpTypes.count(out_node->GetOpDesc()->GetType()) == 0) {
    return false;
  }
  // Case node only support int32 input
  if ((kCaseOpTypes.count(cur_node->GetType()) != 0) && (cond_tensor.GetDataType() != DT_INT32)) {
    GELOGW("Check input failed, node is %s, condition datatype is %s.", op_desc->GetName().c_str(),
           TypeUtils::DataTypeToSerialString(cond_tensor.GetDataType()).c_str());
    return false;
  }
  // Get weights from peer node
  auto weights = OpDescUtils::GetWeights(out_node);
  if (weights.size() <= static_cast<size_t>(cond_out_anchor->GetIdx())) {
    GELOGI("Get weights of node %s out index %d, weight size %u is not fit for data index %d.",
           out_node->GetName().c_str(), cond_out_anchor->GetIdx(), weights.size(), cond_out_anchor->GetIdx());
    return false;
  }
  ConstGeTensorPtr tensor = weights[cond_out_anchor->GetIdx()];
  GE_CHECK_NOTNULL_EXEC(tensor, return false);
  bool if_zero_dim = false;
  if (!cond_tensor.GetShape().IsScalar()) {
    for (size_t dim = 0; dim < cond_tensor.GetShape().GetDimNum(); dim++) {
      if (cond_tensor.GetShape().GetDim(dim) == 0) {
        if_zero_dim = true;
        break;
      }
    }
    // If dim num is not zero and do not has zero dim, index is 1, else index is 0
    cond_index = static_cast<int32_t>((cond_tensor.GetShape().GetDimNum() != 0) && !if_zero_dim);
  } else {
    // Get condition index
    cond_index = GetCondIndex(tensor);
  }
  GELOGD("Condition index is %d, node name is %s, anchor index is %d, dim num is %zu, zero dim flag %d", cond_index,
         op_desc->GetName().c_str(), cond_out_anchor->GetIdx(), cond_tensor.GetShape().GetDimNum(), if_zero_dim);
  return true;
}

Status CondRemovePass::ReplaceIfCaseNodeWithPartitioncall(const NodePtr &node, const ComputeGraphPtr &save_branch) {
  // Add compute graph to new node
  const auto &input_desc_size = node->GetOpDesc()->GetInputsSize();
  const auto &output_desc_size = node->GetOpDesc()->GetOutputsSize();
  // Create subgraph opdesc & node
  auto partitioncall_opdesc =
    CreateSubgraphOpDesc(save_branch->GetName(), input_desc_size - kConditionIndexNum, output_desc_size);
  auto partitioncall_node = node->GetOwnerComputeGraph()->AddNode(partitioncall_opdesc);
  // Link node's peerout anchors to new node's inanchors
  for (const auto &input_anchor : node->GetAllInAnchors()) {
    for (const auto &peerout_anchor : input_anchor->GetPeerAnchors()) {
      if (GraphUtils::AddEdge(peerout_anchor, partitioncall_node->GetInAnchor(
                                                input_anchor->GetIdx() - kConditionIndexNum)) != ge::GRAPH_SUCCESS) {
        GELOGE(FAILED, "Add edge failed, from node:%s idx:%d to node:%s idx:%d, input num:%d, output num:%d",
               peerout_anchor->GetOwnerNode()->GetName().c_str(), peerout_anchor->GetIdx(),
               partitioncall_node->GetName().c_str(), input_anchor->GetIdx(), input_desc_size, output_desc_size);
        return FAILED;
      }
    }
  }
  // Remove If / Case anchor and peer in anchor
  // Link new node's out anchors to node's peer inanchors
  for (const auto &output_anchor : node->GetAllOutAnchors()) {
    for (const auto &peerin_anchor : output_anchor->GetPeerAnchors()) {
      if (GraphUtils::RemoveEdge(node->GetOutAnchor(output_anchor->GetIdx()), peerin_anchor) != ge::GRAPH_SUCCESS) {
        GELOGE(FAILED, "Remove edge failed, from node:%s idx:%d to node:%s idx:%d, input num:%d, output num:%d",
               node->GetName().c_str(), output_anchor->GetIdx(), peerin_anchor->GetOwnerNode()->GetName().c_str(),
               peerin_anchor->GetIdx(), input_desc_size, output_desc_size);
        return FAILED;
      }
      if (GraphUtils::AddEdge(partitioncall_node->GetOutAnchor(output_anchor->GetIdx()), peerin_anchor) !=
          ge::GRAPH_SUCCESS) {
        GELOGE(FAILED, "Add edge failed, from node:%s idx:%d to node:%s idx:%d, input num:%d, output num:%d",
               partitioncall_node->GetName().c_str(), output_anchor->GetIdx(),
               peerin_anchor->GetOwnerNode()->GetName().c_str(), peerin_anchor->GetIdx(), input_desc_size,
               output_desc_size);
        return FAILED;
      }
    }
  }
  // update save branch information
  std::map<uint32_t, uint32_t> input_mapping;
  uint32_t new_input_num = static_cast<uint32_t>(node->GetOpDesc()->GetAllInputsSize()) - kConditionIndexNum;
  for (uint32_t i = 0; i < new_input_num; i++) {
    // original index + 1 map to index
    input_mapping[i + 1] = i;
  }
  save_branch->UpdateInputMapping(input_mapping);
  save_branch->SetParentNode(partitioncall_node);
  save_branch->SetParentGraph(node->GetOwnerComputeGraph());
  return SUCCESS;
}

///
/// @brief Create op_desc for subgraph node
/// @param [in] name
/// @param [in] input_num
/// @param [in] output_num
/// @return OpDescPtr
///
OpDescPtr CondRemovePass::CreateSubgraphOpDesc(const std::string &name, size_t input_num, size_t output_num) {
  OpDescBuilder op_desc_builder(name, PARTITIONEDCALL);
  op_desc_builder.AddDynamicInput("args", input_num).AddDynamicOutput("output", output_num);

  OpDescPtr op_desc = op_desc_builder.Build();
  GE_CHECK_NOTNULL_EXEC(op_desc, return nullptr);

  size_t index = op_desc->GetSubgraphInstanceNames().size();
  op_desc->AddSubgraphName("f");
  op_desc->SetSubgraphInstanceName(static_cast<uint32_t>(index), name);
  return op_desc;
}

///
/// @brief Get cond info for if/case node
/// @param [in] node: If/Case op
/// @param [out] graph: owner_graph of if node
/// @param [out] cond_out_anchor: peer_cond_anchor
/// @param [out] cond_in_anchor: cond_input of if
/// @return Status
///
Status CondRemovePass::GetCondInfoForIfCase(const NodePtr &node, ComputeGraphPtr &graph,
                                            OutDataAnchorPtr &cond_out_anchor, InDataAnchorPtr &cond_in_anchor) {
  GE_CHECK_NOTNULL(node);
  graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  cond_in_anchor = node->GetInDataAnchor(IF_COND_INPUT);
  GE_CHECK_NOTNULL(cond_in_anchor);
  cond_out_anchor = cond_in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(cond_out_anchor);
  return SUCCESS;
}

Status CondRemovePass::GetCondInfo(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &cond_out_anchor,
                                   InDataAnchorPtr &cond_in_anchor) {
  GE_CHECK_NOTNULL(node);
  std::string type = node->GetType();
  if ((kIfOpTypes.count(type) != 0) || (kCaseOpTypes.count(type) != 0)) {
    if (GetCondInfoForIfCase(node, graph, cond_out_anchor, cond_in_anchor) != SUCCESS) {
      GELOGE(FAILED, "Get cond_info for if node failed.");
      return FAILED;
    }
  } else {
    GELOGD("no need cond_pass for node %s.", node->GetName().c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}
}  // namespace ge
