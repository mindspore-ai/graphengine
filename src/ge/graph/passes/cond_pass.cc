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

#include "graph/passes/cond_pass.h"
#include "common/op/ge_op_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"

namespace {
const std::set<std::string> kIfTypes = {ge::IF, ge::_IF, ge::STATELESSIF};
const std::set<std::string> kWhileTypes = {ge::WHILE, ge::_WHILE, ge::STATELESSWHILE};
const std::string kStringLength = "StringLength";
const size_t kScalarDimNum = 1;
}  // namespace

namespace ge {
Status CondPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  ComputeGraphPtr graph = nullptr;
  OutDataAnchorPtr cond_out_anchor = nullptr;
  InDataAnchorPtr cond_in_anchor = nullptr;
  Status ret = GetCondInfo(node, graph, cond_out_anchor, cond_in_anchor);
  if (ret == NOT_CHANGED) {
    return SUCCESS;
  } else if (ret != SUCCESS) {
    GELOGE(FAILED, "Get cond_info for node %s failed.", node->GetName().c_str());
    return FAILED;
  }

  /// cond
  /// 1. NonScalar: cond->Shape->Shape(int32)->If / NetOutput(while)
  /// 2. String Scalar: cond->StringLength(int32)->If / NetOutput(while)
  /// 3. bool / float / double / uint8 / int16 / int8 / int64 Scalar: cond->Cast(2int32)->If / NetOutput(while)
  /// 4. Int32 Scalar: cond->If / NetOutput(while)
  OpDescPtr op_desc = cond_in_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Handle cond for node %s.", op_desc->GetName().c_str());
  GeTensorDesc cond_tensor = op_desc->GetInputDesc(cond_in_anchor->GetIdx());
  if (!cond_tensor.GetShape().IsScalar()) {
    GE_CHK_STATUS_RET(HandleNonScalarCond(graph, cond_out_anchor, cond_in_anchor), "HandleNonScalarCond for %s failed.",
                      op_desc->GetName().c_str())
  } else {
    switch (cond_tensor.GetDataType()) {
      case DT_STRING:
        GE_CHK_STATUS_RET(HandleStringCond(graph, cond_out_anchor, cond_in_anchor), "HandleStringCond for %s failed.",
                          op_desc->GetName().c_str())
        break;
      case DT_BOOL:
      case DT_FLOAT:
      case DT_DOUBLE:
      case DT_UINT8:
      case DT_INT16:
      case DT_INT8:
      case DT_INT64:
        GE_CHK_STATUS_RET(HandleScalarCond(graph, cond_out_anchor, cond_in_anchor, cond_tensor.GetDataType()),
                          "HandleScalarCond for %s failed.", op_desc->GetName().c_str())
        break;
      case DT_INT32:
        break;
      default:
        GELOGE(FAILED, "UpdateInputDesc for node %s failed.", op_desc->GetName().c_str());
        return FAILED;
    }
  }

  cond_tensor.SetDataType(DT_INT32);
  cond_tensor.SetOriginDataType(DT_INT32);
  cond_tensor.SetShape(GeShape());
  cond_tensor.SetOriginShape(GeShape());
  if (op_desc->UpdateInputDesc(cond_in_anchor->GetIdx(), cond_tensor) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "UpdateInputDesc for node %s failed.", op_desc->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Get cond info for if / while
/// @param [in] node: If / While op
/// @param [out] graph: owner_graph of if node / while_cond subgraph
/// @param [out] cond_out_anchor: peer_cond_anchor
/// @param [out] cond_in_anchor: cond_input
/// @return Status
///
Status CondPass::GetCondInfo(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &cond_out_anchor,
                             InDataAnchorPtr &cond_in_anchor) {
  GE_CHECK_NOTNULL(node);
  std::string type = node->GetType();
  if (kIfTypes.count(type) != 0) {
    if (GetCondInfoForIf(node, graph, cond_out_anchor, cond_in_anchor) != SUCCESS) {
      GELOGE(FAILED, "Get cond_info for if node failed.");
      return FAILED;
    }
  } else if (kWhileTypes.count(type) != 0) {
    if (GetCondInfoForWhile(node, graph, cond_out_anchor, cond_in_anchor) != SUCCESS) {
      GELOGE(FAILED, "Get cond_info for while node failed.");
      return FAILED;
    }
  } else {
    GELOGI("no need cond_pass for node %s.", node->GetName().c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

///
/// @brief Get cond info for if node
/// @param [in] node: If op
/// @param [out] graph: owner_graph of if node
/// @param [out] cond_out_anchor: peer_cond_anchor
/// @param [out] cond_in_anchor: cond_input of if
/// @return Status
///
Status CondPass::GetCondInfoForIf(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &cond_out_anchor,
                                  InDataAnchorPtr &cond_in_anchor) {
  GE_CHECK_NOTNULL(node);
  graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  cond_in_anchor = node->GetInDataAnchor(IF_COND_INPUT);
  GE_CHECK_NOTNULL(cond_in_anchor);
  cond_out_anchor = cond_in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(cond_out_anchor);
  return SUCCESS;
}

///
/// @brief Get cond info for while node
/// @param [in] node: While op
/// @param [out] graph: while_cond subgraph
/// @param [out] cond_out_anchor: peer_cond_anchor
/// @param [out] cond_in_anchor: input of NetOutput in cond_graph
/// @return Status
///
Status CondPass::GetCondInfoForWhile(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &cond_out_anchor,
                                     InDataAnchorPtr &cond_in_anchor) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::map<std::string, uint32_t> subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  auto iter = subgraph_names_to_index.find(ATTR_NAME_WHILE_COND);
  if (iter == subgraph_names_to_index.end()) {
    GELOGE(FAILED, "Get cond_graph index failed, while_node:%s.", node->GetName().c_str());
    return FAILED;
  }
  std::string cond_graph_instance_name = op_desc->GetSubgraphInstanceName(iter->second);
  graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph())->GetSubgraph(cond_graph_instance_name);
  GE_CHECK_NOTNULL(graph);

  NodePtr net_output_node = graph->FindNode(NODE_NAME_NET_OUTPUT);
  GE_CHECK_NOTNULL(net_output_node);
  // cond_graph has and only has one output
  uint32_t output_num = net_output_node->GetAllInDataAnchorsSize();
  if (output_num != 1) {
    GELOGE(FAILED, "output size of cond_graph is invalid, expect 1 but %u exactly, while_node:%s.", output_num,
           node->GetName().c_str());
    return FAILED;
  }

  cond_in_anchor = net_output_node->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(cond_in_anchor);
  cond_out_anchor = cond_in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(cond_out_anchor);

  return SUCCESS;
}

///
/// @brief Process Cond Op with non-scalar cond_input: cond->Shape->Shape->If / NetOutput(while)
/// @param [in] graph
/// @param [in] out_anchor: peer_cond_anchor
/// @param [in] in_anchor: cond_input
/// @return Status
///
Status CondPass::HandleNonScalarCond(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                                     const InDataAnchorPtr &in_anchor) {
  if (InsertNode(graph, out_anchor, in_anchor, SHAPE) != SUCCESS) {
    GELOGE(FAILED, "Insert first Shape node failed.");
    return FAILED;
  }

  if (InsertNode(graph, in_anchor->GetPeerOutAnchor(), in_anchor, SHAPE) != SUCCESS) {
    GELOGE(FAILED, "Insert second Shape node failed.");
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Process Cond Op with scalar-string cond_input: cond->StringLength(int32)->If / NetOutput(while)
/// @param [in] graph
/// @param [in] out_anchor: peer_cond_anchor
/// @param [in] in_anchor: cond_input
/// @return Status
///
Status CondPass::HandleStringCond(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                                  const InDataAnchorPtr &in_anchor) {
  GELOGI("Handle cond with scalar-string cond-input.");
  return InsertNode(graph, out_anchor, in_anchor, kStringLength);
}

///
/// @brief Process Cond Op with scalar cond_input: cond->Cast(2int32)->If / NetOutput(while)
/// @param [in] graph
/// @param [in] out_anchor: peer_cond_anchor
/// @param [in] in_anchor: cond_input
/// @param [in] src_type
/// @return Status
///
Status CondPass::HandleScalarCond(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                                  const InDataAnchorPtr &in_anchor, DataType src_type) {
  GE_CHECK_NOTNULL(in_anchor);
  GE_CHECK_NOTNULL(out_anchor);
  GE_CHECK_NOTNULL(out_anchor->GetOwnerNode()->GetOpDesc());
  GELOGI("Handle cond with scalar cond-input.");

  GeTensorDesc tensor = out_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx());
  std::string cast_name = out_anchor->GetOwnerNode()->GetName() + "_Cast";
  NodePtr cast_node = AddCastNode(graph, cast_name, tensor, src_type, DT_INT32);
  if (cast_node == nullptr) {
    GELOGE(FAILED, "Add Cast node failed, name:%s.", cast_name.c_str());
    return FAILED;
  }

  if (GraphUtils::InsertNodeBefore(out_anchor, {in_anchor}, cast_node) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Insert Cast node %s between %s->%s failed.", cast_node->GetName().c_str(),
           out_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Insert node
/// @param [in] graph
/// @param [in] out_anchor
/// @param [in] in_anchor
/// @param [in] type
/// @return Status
///
Status CondPass::InsertNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                            const InDataAnchorPtr &in_anchor, const std::string &type) {
  GE_CHECK_NOTNULL(out_anchor);
  GE_CHECK_NOTNULL(in_anchor);
  GELOGD("Begin to insert %s node.", type.c_str());

  GE_CHECK_NOTNULL(out_anchor->GetOwnerNode()->GetOpDesc());
  GE_CHECK_NOTNULL(in_anchor->GetOwnerNode()->GetOpDesc());
  GeTensorDesc in_tensor = out_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx());
  GeTensorDesc out_tensor = in_anchor->GetOwnerNode()->GetOpDesc()->GetInputDesc(out_anchor->GetIdx());
  out_tensor.SetDataType(DT_INT32);
  out_tensor.SetOriginDataType(DT_INT32);
  if (type == SHAPE) {
    int64_t size = static_cast<int64_t>(in_tensor.GetShape().GetDimNum());
    if (size == kScalarDimNum) {
      out_tensor.SetShape(GeShape());
      out_tensor.SetOriginShape(GeShape());
    } else {
      std::vector<int64_t> size_v{size};
      out_tensor.SetShape(GeShape(size_v));
      out_tensor.SetOriginShape(GeShape(size_v));
    }
  }

  OpDescBuilder op_desc_builder(out_anchor->GetOwnerNode()->GetName() + "_" + type, type);
  OpDescPtr op_desc = op_desc_builder.AddInput("x", in_tensor).AddOutput("y", out_tensor).Build();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc failed.");
    return FAILED;
  }
  NodePtr new_node = graph->AddNode(op_desc);
  if (new_node == nullptr) {
    GELOGE(FAILED, "Create %s node failed.", type.c_str());
    return FAILED;
  }
  AddRePassNode(new_node);

  if (GraphUtils::InsertNodeBefore(out_anchor, {in_anchor}, new_node) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Insert %s node %s between %s->%s failed.", type.c_str(), new_node->GetName().c_str(),
           out_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Add cast node
/// @param [in] graph
/// @param [in] name
/// @param [in] tensor
/// @param [in] src
/// @param [in] dst
/// @return NodePtr
///
NodePtr CondPass::AddCastNode(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &tensor,
                              DataType src, DataType dst) {
  GELOGI("Begin to create cast op: %s, from %d to %d", name.c_str(), src, dst);

  GeTensorDesc in_tensor = tensor;
  in_tensor.SetDataType(src);
  in_tensor.SetOriginDataType(src);
  GeTensorDesc out_tensor = tensor;
  out_tensor.SetDataType(dst);
  out_tensor.SetOriginDataType(dst);
  OpDescBuilder op_desc_builder(name, CAST);
  OpDescPtr cast_desc = op_desc_builder.AddInput("x", in_tensor).AddOutput("y", out_tensor).Build();
  if (cast_desc == nullptr) {
    GELOGE(FAILED, "Create cast op_desc failed, name: %s.", name.c_str());
    return nullptr;
  }
  if (!(AttrUtils::SetInt(cast_desc, CAST_ATTR_SRCT, src) && AttrUtils::SetInt(cast_desc, CAST_ATTR_DSTT, dst) &&
        AttrUtils::SetInt(cast_desc, CAST_ATTR_DST_TYPE, dst) &&
        AttrUtils::SetBool(cast_desc, CAST_ATTR_TRUNCATE, false))) {
    GELOGE(FAILED, "Set CAST_ATTR failed, node: %s.", name.c_str());
    return nullptr;
  }

  NodePtr cast_node = graph->AddNode(cast_desc);
  if (cast_node == nullptr) {
    GELOGE(FAILED, "Add cast node failed, name: %s.", name.c_str());
    return nullptr;
  }
  AddRePassNode(cast_node);

  return cast_node;
}
}  // namespace ge
