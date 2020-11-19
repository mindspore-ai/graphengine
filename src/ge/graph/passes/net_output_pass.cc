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

#include "graph/passes/net_output_pass.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common/local_context.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
static std::map<std::string, ge::DataType> output_type_str_to_datatype = {
  {"FP32", ge::DT_FLOAT},    {"FP16", ge::DT_FLOAT16},  {"INT8", ge::DT_INT8},    {"INT16", ge::DT_INT16},
  {"UINT16", ge::DT_UINT16}, {"UINT8", ge::DT_UINT8},   {"INT32", ge::DT_INT32},  {"INT64", ge::DT_INT64},
  {"UINT32", ge::DT_UINT32}, {"UINT64", ge::DT_UINT64}, {"DOUBLE", ge::DT_DOUBLE}};

// the size of user defined output datatype or format string after split by ":".
const size_t kUserDefinedElementCount = 2;

Status NetOutputPass::GetRetvalOutputInfo(const ge::NodePtr &node,
                                          std::map<int32_t, RetvalInfo> &retval_node_index_map) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  int64_t output_index = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), RETVAL_ATTR_NAME_INDEX, output_index)) {
    GELOGE(PARAM_INVALID, "Get output index failed.");
    return PARAM_INVALID;
  }
  if (retval_node_index_map.count(output_index) > 0) {
    GELOGE(PARAM_INVALID, "Retval has duplicate index.");
    return PARAM_INVALID;
  }
  int parent_node_index = -1;
  (void)AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_node_index);
  InDataAnchorPtr in_data_anchor = node->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(in_data_anchor);
  GE_CHECK_NOTNULL(in_data_anchor->GetPeerOutAnchor());
  int32_t src_node_index = in_data_anchor->GetPeerOutAnchor()->GetIdx();
  NodePtr src_node_ptr = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
  retval_node_index_map[output_index] = {src_node_ptr, src_node_index, parent_node_index};
  // if user targets include retval node,delete it from set and insert its input node instead
  // better to GetInNodes here
  auto iter = targets_.find(node);
  if (iter != targets_.end()) {
    targets_.erase(iter);
    targets_.insert(src_node_ptr);
    GELOGI("node [%s] is in user def targets, do not output result to user!", node->GetName().c_str());
  }
  is_include_special_node_ = true;
  return SUCCESS;
}

Status NetOutputPass::GetOutputNode(const ge::ComputeGraphPtr &graph, std::vector<RetvalInfo> &output_nodes_info) {
  std::map<int32_t, RetvalInfo> retval_node_index_map;
  for (NodePtr &node : graph->GetDirectNode()) {
    Status ret = SUCCESS;
    if ((node->GetOpDesc() != nullptr) && (node->GetOpDesc()->HasAttr(RETVAL_ATTR_NAME_INDEX))) {
      /// Set the output according to the Retval operator,
      /// identify by whether there is an index parameter
      ret = GetRetvalOutputInfo(node, retval_node_index_map);
    }
    if (ret != SUCCESS) {
      GELOGE(ret, "GetRetvalOutputInfo failed");
      return ret;
    }
  }
  GELOGI("Get retval node size:%zu.", retval_node_index_map.size());
  std::vector<RetvalInfo> out_nodes_tmp;
  /// The Netoutput output is determined by Retval, and the input order
  /// of Netoutput is sorted according to the index value of Retval.
  for (auto &it : retval_node_index_map) {
    out_nodes_tmp.push_back(it.second);
  }

  // when user set targets, mean that no output result
  for (auto &ele : graph->GetGraphOutNodesInfo()) {
    auto iter = targets_.find(ele.first);
    if (iter != targets_.end()) {
      GELOGI("user set out node [%s] is found in user def targets, out node is prio!", ele.first->GetName().c_str());
      targets_.erase(iter);
    }

    auto op_desc = ele.first->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->HasAttr(ATTR_ATC_USER_DEFINE_OUTPUT_NODES)) {
      is_user_define_ouput_nodes = true;
    }
    output_nodes_info.push_back({ele.first, ele.second, -1});
  }
  GELOGI("Output node set by user or leaf node, size:%zu.", output_nodes_info.size());
  for (auto &ele : out_nodes_tmp) {
    // add member, no need to remove duplicated because we need to keep all edges
    output_nodes_info.push_back(ele);
  }
  GELOGI("Get output node, size:%zu.", output_nodes_info.size());

  Status check_ret = CheckOutputNodeInfo(graph, output_nodes_info);
  if (check_ret != SUCCESS) {
    return check_ret;
  }
  return SUCCESS;
}

Status NetOutputPass::CheckOutputNodeInfo(const ComputeGraphPtr &graph, const std::vector<RetvalInfo> &outputs) {
  for (auto &item : outputs) {
    NodePtr node = item.output_node;
    if (node == nullptr) {
      GELOGE(PARAM_INVALID, "Node in outputs is null.");
      return PARAM_INVALID;
    } else {
      if (graph->FindNode(node->GetName()) == nullptr) {
        GELOGE(INTERNAL_ERROR, "Out node (%s) is not in graph.", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
      GE_CHECK_NOTNULL(node->GetOpDesc());
      int32_t out_size = node->GetOpDesc()->GetOutputsSize();
      int32_t index = item.node_output_index;
      if (index < 0 || index >= out_size) {
        GELOGE(PARAM_INVALID,
               "User declared out node (%s) output index:%d must be smaller "
               "than node ouput size:%d and cann't be negative!",
               node->GetName().c_str(), index, out_size);
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

Status NetOutputPass::RemoveUnusedNode(const ge::ComputeGraphPtr &graph) {
  std::vector<ge::NodePtr> node_to_delete;
  // Delete _Retval operator.
  for (auto &node : graph->GetDirectNode()) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, GELOGW("Node OpDesc is nullptr"); continue);
    bool need_be_deleted = node->GetInDataNodes().size() != 0 && node->GetOutDataNodesSize() == 0 &&
                           (node->GetOpDesc()->HasAttr(RETVAL_ATTR_NAME_INDEX));
    if (need_be_deleted) {
      node_to_delete.push_back(node);
    }
  }
  for (NodePtr &node : node_to_delete) {
    auto iter = targets_.find(node);
    if (iter != targets_.end()) {
      GELOGI("[Net output pass] node[%s] is in user set targets.so do not remove!", node->GetName().c_str());
      continue;
    }
    if (graph->RemoveNode(node) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Remove node failed, node name:%s.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status NetOutputPass::UpdateNetOutputDesc(const ge::NodePtr &net_output) {
  OpDescPtr net_output_desc = net_output->GetOpDesc();
  if (net_output_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Opdesc of net output node is nullptr.");
    return INTERNAL_ERROR;
  }
  if (net_output_desc->GetInputsSize() == 0) {
    GELOGE(INTERNAL_ERROR, "Net output node input is empty.");
    return INTERNAL_ERROR;
  }

  std::vector<bool> is_input_const;
  for (const auto &in_anchor : net_output->GetAllInDataAnchors()) {
    GE_CHECK_NOTNULL(in_anchor);
    uint32_t index = static_cast<uint32_t>(in_anchor->GetIdx());
    if (index >= net_output_desc->GetAllInputsDesc().size()) {
      GELOGE(INTERNAL_ERROR, "Index is invalid, index:%u, size:%zu.", index,
             net_output_desc->GetAllInputsDesc().size());
      return INTERNAL_ERROR;
    }
    GE_CHECK_NOTNULL(in_anchor->GetPeerOutAnchor());
    is_input_const.push_back(PassUtils::IsConstant(in_anchor->GetPeerOutAnchor()->GetOwnerNode()));
    OpDescPtr src_op_desc = in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc();
    GE_CHECK_NOTNULL(src_op_desc);
    uint32_t peer_index = static_cast<uint32_t>(in_anchor->GetPeerOutAnchor()->GetIdx());
    ge::GeTensorDesc output_in_desc = src_op_desc->GetOutputDesc(peer_index);
    if (net_output_desc->UpdateInputDesc(index, output_in_desc) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Update input desc failed, index:%u.", index);
      return INTERNAL_ERROR;
    }
    GELOGD("Update desc, format:%s, data type:%s, index:%u.",
           TypeUtils::FormatToSerialString(output_in_desc.GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_in_desc.GetDataType()).c_str(), index);
  }
  net_output_desc->SetIsInputConst(is_input_const);
  return SUCCESS;
}

Status NetOutputPass::AddCtrlEdgeForTargets(const ge::NodePtr &net_out_node) {
  if (net_out_node == nullptr) {
    GELOGE(PARAM_INVALID, "net out node is null.");
    return PARAM_INVALID;
  }
  // Add ctrl edge for targets
  for (auto &node : targets_) {
    if (node == nullptr) {
      continue;
    }
    // no need to check null because have handled it in run SaveAndRemoveTargets function
    graphStatus status = GraphUtils::AddEdge(node->GetOutControlAnchor(), net_out_node->GetInControlAnchor());
    if (status != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Add ctrl edge to netoutput node[%s] for target node [%s] failed!",
             net_out_node->GetName().c_str(), node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GELOGD("Add ctrl edge to netoutput node[%s] for target node [%s] success!", net_out_node->GetName().c_str(),
           node->GetName().c_str());
  }
  return SUCCESS;
}

void NetOutputPass::SaveAndRemoveTargets(const ge::ComputeGraphPtr &graph) {
  // save user targets node
  for (auto &node : graph->GetGraphTargetNodesInfo()) {
    if (node == nullptr) {
      GELOGW("User pointed targets contains null node.ignore it !");
      continue;
    }
    targets_.insert(node);
  }
  GELOGI("User pointed targets size is %zu !", targets_.size());
}

Status NetOutputPass::AddEdgesForNetOutput(const ge::ComputeGraphPtr &graph, const ge::NodePtr &net_out_node,
                                           const std::vector<RetvalInfo> &output_nodes_info) {
  int32_t net_input_index = 0;
  for (auto &item : output_nodes_info) {
    NodePtr src_node = item.output_node;
    GE_CHECK_NOTNULL(src_node);
    graphStatus status = GraphUtils::AddEdge(src_node->GetOutDataAnchor(item.node_output_index),
                                             net_out_node->GetInDataAnchor(net_input_index));
    if (status != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "AddEdge failed, src name:%s, src index:%d, dst index:%d.", src_node->GetName().c_str(),
             item.node_output_index, net_input_index);
      return INTERNAL_ERROR;
    }
    GELOGD("AddEdge to output node, src name:%s, src index:%d, dst index:%d.", src_node->GetName().c_str(),
           item.node_output_index, net_input_index);
    if (item.parent_node_index >= 0) {
      GELOGI("Add parent node index %d for the netoutput input %d on graph %s", item.parent_node_index, net_input_index,
             graph->GetName().c_str());
      auto input_desc = net_out_node->GetOpDesc()->MutableInputDesc(net_input_index);
      if (input_desc == nullptr) {
        GELOGE(INTERNAL_ERROR, "Can not find intput tensor desc from NetOutput, index %d", net_input_index);
        return INTERNAL_ERROR;
      }
      if (!AttrUtils::SetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, item.parent_node_index)) {
        GELOGE(INTERNAL_ERROR, "Failed to add parent index to  NetOutput, index %d", net_input_index);
        return INTERNAL_ERROR;
      }
    }
    net_input_index++;
  }
  if (RemoveUnusedNode(graph) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Remove unused nodes failed.");
    return INTERNAL_ERROR;
  }
  if (AddCtrlEdgeForTargets(net_out_node) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Add ctrl edge for targets failed.");
    return INTERNAL_ERROR;
  }
  // Add true stream, netoutput is 0
  GE_IF_BOOL_EXEC(!ge::AttrUtils::SetInt(net_out_node->GetOpDesc(), ATTR_NAME_TRUE_BRANCH_STREAM, 0),
                  GELOGE(INTERNAL_ERROR, "set ATTR_NAME_TRUE_BRANCH_STREAM failed");
                  return INTERNAL_ERROR);
  return SUCCESS;
}

bool NetOutputPass::CheckNodeIsInOutputNodes(const ge::ComputeGraphPtr &graph, const ge::NodePtr &node) {
  for (auto &ele : graph->GetGraphOutNodesInfo()) {
    auto out_node = ele.first;
    if (node == out_node) {
      return true;
    }
  }
  return false;
}
Status NetOutputPass::UnLinkDataAnchorOfNetoutput(const ge::ComputeGraphPtr &graph, const ge::NodePtr &net_out_node) {
  if (net_out_node == nullptr) {
    GELOGE(PARAM_INVALID, "net out node is null.");
    return PARAM_INVALID;
  }
  Status ret = SUCCESS;

  // unlink all anchor to data anchor of netoutput
  for (auto &in_data_anchor : net_out_node->GetAllInDataAnchors()) {
    if (in_data_anchor == nullptr) {
      continue;
    }
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      GELOGI("PeerOutAnchor is null!");
      continue;
    }
    auto node = peer_out_anchor->GetOwnerNode();
    auto iter = targets_.find(node);
    if (iter != targets_.end()) {
      if (!CheckNodeIsInOutputNodes(graph, node)) {
        ret = in_data_anchor->Unlink(peer_out_anchor);
        if (ret != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "Unlink peer_out_anchor fail!");
          return ret;
        }
      } else {
        targets_.erase(iter);
      }
    }
  }
  return ret;
}

Status NetOutputPass::UnLinkControlAnchorOfNetoutput(const ge::ComputeGraphPtr &graph,
                                                     const ge::NodePtr &net_out_node) {
  if (net_out_node == nullptr) {
    GELOGE(PARAM_INVALID, "net out node is null.");
    return PARAM_INVALID;
  }
  Status ret = SUCCESS;
  auto in_control_anchor = net_out_node->GetInControlAnchor();
  if (in_control_anchor == nullptr) {
    GELOGE(PARAM_INVALID, "in control anchor is null.");
    return PARAM_INVALID;
  }
  // unlink all data anchor to control anchor of netoutput
  for (auto &peer_out_data_anchor : in_control_anchor->GetPeerOutDataAnchors()) {
    if (peer_out_data_anchor == nullptr) {
      GELOGI("PeerOutControlAnchor is null!");
    } else {
      auto node = peer_out_data_anchor->GetOwnerNode();
      auto iter = targets_.find(node);
      if (iter != targets_.end()) {
        if (CheckNodeIsInOutputNodes(graph, node) == false) {
          ret = in_control_anchor->Unlink(peer_out_data_anchor);
          if (ret != SUCCESS) {
            GELOGE(INTERNAL_ERROR, "Unlink peer_out_anchor fail!");
            return ret;
          }
        } else {
          targets_.erase(iter);
        }
      }
    }
  }
  /// check all control anchor to control anchor of netoutput and delete it from targets
  /// to avoid duplicated add control edge;
  for (auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
    if (peer_out_control_anchor == nullptr) {
      GELOGI("PeerOutControlAnchor is null");
    } else {
      auto node = peer_out_control_anchor->GetOwnerNode();
      auto iter = targets_.find(node);
      if (iter != targets_.end()) {
        targets_.erase(iter);
      }
    }
  }
  return ret;
}

Status NetOutputPass::UnLink(const ge::ComputeGraphPtr &graph, const ge::NodePtr &net_out_node) {
  GELOGI("[NetOutputPass] Enter Unlink process.");
  Status ret = UnLinkDataAnchorOfNetoutput(graph, net_out_node);
  if (ret != SUCCESS) {
    GELOGI("[NetOutputPass] UnLinkDataAnchorOfNetoutput process fail.");
    return ret;
  }
  ret = UnLinkControlAnchorOfNetoutput(graph, net_out_node);
  if (ret != SUCCESS) {
    GELOGI("[NetOutputPass] UnLinkControlAnchorOfNetoutput process fail.");
    return ret;
  }
  return ret;
}

Status NetOutputPass::ProcessWithNetoutput(const ge::ComputeGraphPtr &graph, const ge::NodePtr &output_node) {
  if (UpdateNetOutputDesc(output_node) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Update net output desc failed.");
    return INTERNAL_ERROR;
  }

  if (UnLink(graph, output_node) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "UnLink connection between netoutput node and user set target node");
    return INTERNAL_ERROR;
  }
  if (AddCtrlEdgeForTargets(output_node) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Add ctrl edge for targets failed.");
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status NetOutputPass::AddCtrlEdgesBetweenLeafAndNetOutput(const ge::ComputeGraphPtr &graph,
                                                          const ge::NodePtr &net_out_node) {
  GE_CHECK_NOTNULL(net_out_node);
  if (!GetLocalOmgContext().user_out_nodes.empty() || is_user_define_ouput_nodes) {
    GELOGI("No need to add ctrl edge to netoutput because user out nodes have been set.");
    return SUCCESS;
  }
  for (const auto &node : graph->GetDirectNode()) {
    if (node == nullptr || node->GetOpDesc() == nullptr || node->GetOpDesc()->GetType() == NETOUTPUT) {
      continue;
    }
    if ((node->GetInControlNodes().size() != 0 || node->GetInDataNodes().size() != 0) &&
        node->GetOutDataNodesSize() == 0 && node->GetOutControlNodes().size() == 0) {
      GE_CHK_STATUS_RET(GraphUtils::AddEdge(node->GetOutControlAnchor(), net_out_node->GetInControlAnchor()),
                        "add edge failed");
      GELOGD("Add ctrl edge success. src name :%s, dst name :%s", node->GetName().c_str(),
             net_out_node->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status NetOutputPass::CreateNetOutputNode(OpDescPtr &net_output_desc, const ge::ComputeGraphPtr &graph) {
  // Only flush subgraph name
  string node_name =
    (graph->GetParentGraph() != nullptr) ? (graph->GetName() + "_" + NODE_NAME_NET_OUTPUT) : NODE_NAME_NET_OUTPUT;
  net_output_desc = MakeShared<OpDesc>(node_name, NETOUTPUT);
  if (net_output_desc == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared net output op failed.");
    return MEMALLOC_FAILED;
  }
  (void)AttrUtils::SetListStr(net_output_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                              std::move(std::vector<std::string>()));
  return SUCCESS;
}

Status NetOutputPass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "Compute graph is null.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  GELOGI("NetOutputPass Run.");
  NodePtr output_node = graph->FindFirstNodeMatchType(NETOUTPUT);
  // save user targets node
  SaveAndRemoveTargets(graph);
  // If graph already has a netoutput node, doesn't need to create it again.
  if (output_node != nullptr) {
    (void)AttrUtils::SetListStr(output_node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                std::move(std::vector<std::string>()));
    if (ProcessWithNetoutput(graph, output_node) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Process with netoutput node failed.");
      return INTERNAL_ERROR;
    }
  } else {
    if (AddNetOutputNodeToGraph(graph, output_node) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Set user define dtype and format for netoutput failed.");
      return INTERNAL_ERROR;
    }
  }
  // Add userdef attrs to netoutput node
  return SetUserDefDTypeAndFormatFromAtcParams(output_node);
}

Status NetOutputPass::AddNetOutputNodeToGraph(const ge::ComputeGraphPtr &graph, NodePtr &output_node) {
  OpDescPtr net_output_desc = nullptr;
  if (CreateNetOutputNode(net_output_desc, graph) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Get net output nodes failed.");
    return INTERNAL_ERROR;
  }
  std::vector<RetvalInfo> output_nodes_info;
  if (GetOutputNode(graph, output_nodes_info) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Get net output nodes failed.");
    return INTERNAL_ERROR;
  }
  GELOGI("[NETOUTPUT PASS] OutNodesInfo size:%zu, Targets Size:%zu, is_include_special_node_:%d",
         graph->GetGraphOutNodesInfo().size(), graph->GetGraphTargetNodesInfo().size(), is_include_special_node_);
  // If user does not set out nodes and targets and no retval node, return false
  if ((graph->GetGraphOutNodesInfo().empty()) && (graph->GetGraphTargetNodesInfo().empty()) &&
      !is_include_special_node_) {
    GELOGI("[NETOUTPUT PASS] output_nodes and target_nodes and special nodes is empty!It means no need netoutput!");
    return SUCCESS;
  }
  GELOGI("[NETOUTPUT PASS] Output node size:%lu.", output_nodes_info.size());
  if (output_nodes_info.empty()) {
    // because retval node is contained by output_nodes_info, here means targets is non-empty
    output_node = graph->AddNode(net_output_desc);
    if (output_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "Add output node failed.");
      return INTERNAL_ERROR;
    }
    GE_CHK_STATUS_RET(AddCtrlEdgeForTargets(output_node), "add ctrl edge for targets failed");
    // Add true stream, netoutput is 0
    GE_IF_BOOL_EXEC(!ge::AttrUtils::SetInt(output_node->GetOpDesc(), ATTR_NAME_TRUE_BRANCH_STREAM, 0),
                    GELOGE(INTERNAL_ERROR, "set ATTR_NAME_TRUE_BRANCH_STREAM failed");
                    return INTERNAL_ERROR);
    return SUCCESS;
  }

  AddInOutForNetOutputOp(graph, net_output_desc, output_nodes_info);
  output_node = graph->AddNode(net_output_desc);
  if (output_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "Add output node failed.");
    return INTERNAL_ERROR;
  }
  if (AddEdgesForNetOutput(graph, output_node, output_nodes_info) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Add edges for net output node failed.");
    return INTERNAL_ERROR;
  }
  if (AddCtrlEdgesBetweenLeafAndNetOutput(graph, output_node) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Add control edges between leaf and netoutput failed.");
    return INTERNAL_ERROR;
  }
  GELOGI("Add NetOutput node success.");
  return SUCCESS;
}
void NetOutputPass::AddInOutForNetOutputOp(const ComputeGraphPtr &graph, OpDescPtr &net_output_desc,
                                           vector<RetvalInfo> &output_nodes_info) {
  std::vector<bool> is_input_const;
  for (auto iter = output_nodes_info.begin(); iter != output_nodes_info.end();) {
    NodePtr src_node = iter->output_node;
    if (src_node == nullptr) {
      continue;
    }
    int32_t src_index = iter->node_output_index;
    // if src_node is in targets_, no need to Add in and out for netoutput
    auto it = targets_.find(src_node);
    if (it != targets_.end()) {
      iter = output_nodes_info.erase(iter);
      GELOGD("node [%s] is in processed targets, do not add inout for netoutput!", src_node->GetName().c_str());
      continue;
    }
    /// Get the output attribute of src_node,
    /// and set to the input/output of net_out_node.
    if (src_node == nullptr || src_node->GetOpDesc() == nullptr || net_output_desc == nullptr) {
      GELOGE(INTERNAL_ERROR, "src node or net output desc is null.");
      return;
    }
    ge::GeTensorDesc out_desc = src_node->GetOpDesc()->GetOutputDesc(src_index);
    GE_IF_BOOL_EXEC(net_output_desc->AddInputDesc(out_desc) != SUCCESS, GELOGW("add input desc failed"); return );
    is_input_const.push_back(PassUtils::IsConstant(src_node));
    ++iter;
  }
  net_output_desc->SetIsInputConst(is_input_const);
}

bool NeedUpdateOutputByOutputTypeParm(std::string &output_type, OpDescPtr &op_desc, uint32_t &src_index,
                                      ge::DataType &dt) {
  if (output_type_str_to_datatype.find(output_type) != output_type_str_to_datatype.end()) {
    dt = output_type_str_to_datatype[output_type];
    return true;
  }

  vector<string> output_dt_str;
  if (ge::AttrUtils::GetListStr(op_desc, "_user_defined_output_data_type", output_dt_str)) {
    for (const auto &dt_str : output_dt_str) {
      vector<string> dt_str_split = StringUtils::Split(dt_str, ':');
      if (dt_str_split.size() == kUserDefinedElementCount) {
        if (dt_str_split[0] == to_string(src_index)) {
          dt = TypeUtils::SerialStringToDataType(dt_str_split[1]);
          return true;
        }
      } else {
        GELOGW("The size of [%s] is not 2 after split.", dt_str.c_str());
        continue;
      }
    }
  }
  return false;
}

bool NeedUpdateOutputFp16Nc1hwc0(OpDescPtr &op_desc, uint32_t &src_index) {
  vector<string> output_dt_str;
  if (ge::AttrUtils::GetListStr(op_desc, "_user_defined_output_fp16_5hd", output_dt_str)) {
    for (const auto &dt_str : output_dt_str) {
      vector<string> dt_str_split = StringUtils::Split(dt_str, ':');
      if (dt_str_split.size() == kUserDefinedElementCount) {
        if (dt_str_split[0] == to_string(src_index)) {
          return true;
        }
      } else {
        GELOGW("The size of [%s] is not 2 after split.", dt_str.c_str());
        continue;
      }
    }
  }
  return false;
}

Status NetOutputPass::SetUserDefDTypeAndFormatFromAtcParams(const NodePtr &output_node) {
  if (output_node == nullptr) {
    GELOGI("[NETOUTPUT PASS] The graph no need netoutput node!");
    return SUCCESS;
  }
  auto output_type = GetLocalOmgContext().output_type;
  auto op_desc = output_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::vector<std::string> userdef_dtypes;
  std::vector<std::string> userdef_formats;

  ge::DataType output_data_type = ge::DT_FLOAT;
  for (const auto &in_anchor : output_node->GetAllInDataAnchors()) {
    auto index = static_cast<uint32_t>(in_anchor->GetIdx());
    auto peer_out = in_anchor->GetPeerOutAnchor();
    if (peer_out == nullptr) {
      // If user set target, peer_out anchor will be unlinked.
      continue;
    }
    auto src_index = static_cast<uint32_t>(peer_out->GetIdx());
    auto src_node = peer_out->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    OpDescPtr src_op_desc = src_node->GetOpDesc();
    GE_CHECK_NOTNULL(src_op_desc);

    // Update datatype
    if (NeedUpdateOutputByOutputTypeParm(output_type, src_op_desc, src_index, output_data_type)) {
      GELOGD("Add user-define datatype:%s to netoutput node.",
             TypeUtils::DataTypeToSerialString(output_data_type).c_str());
      userdef_dtypes.push_back(
        std::to_string(index).append(":").append(TypeUtils::DataTypeToSerialString(output_data_type)));
      continue;
    }
    // Output_node is not set,check if is_output_adjust_hw_layout is set
    bool set_fp16_nc1hwc0 = NeedUpdateOutputFp16Nc1hwc0(src_op_desc, src_index);
    if (set_fp16_nc1hwc0) {
      // Set DT_FLOAT16 & FORMAT_NC1HWC0
      userdef_dtypes.push_back(std::to_string(index).append(":").append(TypeUtils::DataTypeToSerialString(DT_FLOAT16)));
      userdef_formats.push_back(
        std::to_string(index).append(":").append(TypeUtils::FormatToSerialString(FORMAT_NC1HWC0)));
    }
  }
  if (!userdef_dtypes.empty() && !ge::AttrUtils::SetListStr(op_desc, ATTR_ATC_USER_DEFINE_DATATYPE, userdef_dtypes)) {
    GELOGE(INTERNAL_ERROR, "Set user_define_dtype attr list for netoutput failed.");
    return INTERNAL_ERROR;
  }
  if (!userdef_formats.empty() && !ge::AttrUtils::SetListStr(op_desc, ATTR_ATC_USER_DEFINE_FORMAT, userdef_formats)) {
    GELOGE(INTERNAL_ERROR, "Set user_define_format attr list for netoutput failed.");
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}
}  // namespace ge
