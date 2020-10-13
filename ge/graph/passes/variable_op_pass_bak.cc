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

#include "graph/passes/variable_op_pass_bak.h"
#include <string>
#include <vector>

#include "common/formats/formats.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "graph/ge_context.h"
#include "graph/graph.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
const int kTransOpOutIndex = 0;

Status ByPassTransNode(NodePtr &front_node, NodePtr &back_node) {
  GE_CHECK_NOTNULL(front_node);
  GE_CHECK_NOTNULL(back_node);
  GELOGD("Begin to bypass trans node %s", front_node->GetName().c_str());
  auto ret = GraphUtils::CopyInCtrlEdges(front_node, back_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR,
           "Failed to move control edges from trans "
           "node %s to var-ref %s",
           front_node->GetName().c_str(), back_node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  auto back_node_in_anchor = back_node->GetInDataAnchor(0);
  if (back_node_in_anchor == nullptr) {
    GELOGE(INTERNAL_ERROR,
           "The back node %s does not have an "
           "input anchor",
           back_node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  back_node_in_anchor->UnlinkAll();
  auto trans_in_anchor = front_node->GetInDataAnchor(0);
  if (trans_in_anchor == nullptr) {
    GELOGE(INTERNAL_ERROR,
           "Failed to get the in data anchor from trans"
           " node %s type %s",
           front_node->GetName().c_str(), front_node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  auto prev_trans_node_out_anchor = trans_in_anchor->GetPeerOutAnchor();
  if (prev_trans_node_out_anchor == nullptr) {
    GELOGW(
        "The trans node %s does not have an input, so the ref node %s does"
        " not have any inputs after bypass",
        front_node->GetName().c_str(), front_node->GetName().c_str());
  } else {
    ret = GraphUtils::AddEdge(prev_trans_node_out_anchor, back_node_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR,
             "Failed to add edge between ref node %s "
             "and the prev node of trans node %s",
             back_node->GetName().c_str(), front_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

bool IsTransSupport(const TransNodeInfo &trans_info) {
  if (trans_info.output.GetShape().IsUnknownShape()) {
    return false;
  }
  if (trans_info.node_type == RESHAPE || trans_info.node_type == REFORMAT) {
    return true;
  } else if (trans_info.node_type == TRANSDATA || trans_info.node_type == TRANSPOSED) {
    formats::TransArgs args{nullptr,
                            trans_info.input.GetFormat(),
                            trans_info.output.GetFormat(),
                            trans_info.input.GetShape().GetDims(),
                            trans_info.output.GetShape().GetDims(),
                            trans_info.input.GetDataType()};
    return formats::IsTransFormatSupport(args);
  } else if (trans_info.node_type == CAST) {
    formats::CastArgs datatype_args{nullptr, static_cast<size_t>(trans_info.input.GetShape().GetShapeSize()),
                                    trans_info.input.GetDataType(), trans_info.output.GetDataType()};
    return formats::IsTransDataTypeSupport(datatype_args);
  } else {
    return false;
  }
}

std::string GetInAndOutDecsDiff(NodePtr &trans_node, bool reverse = false) {
  int tran_in_index = TransOpUtil::GetTransOpDataIndex(trans_node->GetType());
  auto op_desc = trans_node->GetOpDesc();
  GeTensorDesc input_desc = op_desc->GetInputDesc(tran_in_index);
  GeTensorDesc output_desc = op_desc->GetOutputDesc(kTransOpOutIndex);
  if (reverse) {
    GeTensorDesc tmp_desc = input_desc;
    input_desc = output_desc;
    output_desc = tmp_desc;
  }
  auto input_format = input_desc.GetFormat();
  auto input_type = input_desc.GetDataType();
  auto input_shape = input_desc.GetShape();
  auto output_format = output_desc.GetFormat();
  auto output_type = output_desc.GetDataType();
  auto output_shape = output_desc.GetShape();
  std::stringstream diff_key;
  diff_key.str("");
  if (input_format != output_format) {
    diff_key << static_cast<int>(input_format) << '-' << static_cast<int>(output_format) << '-';
  } else {
    diff_key << "*-";
  }
  if (input_type != output_type) {
    diff_key << static_cast<int>(input_type) << '-' << static_cast<int>(output_type) << '-';
  } else {
    diff_key << "*-";
  }
  if (!ge::formats::IsShapeEqual(input_shape, output_shape)) {
    for (auto dim : input_shape.GetDims()) {
      diff_key << dim << '-';
    }
    for (auto dim : output_shape.GetDims()) {
      diff_key << dim << '-';
    }
  } else {
    diff_key << "*";
  }
  return diff_key.str();
}
}  // namespace

Status VariableOpPass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    GELOGE(INTERNAL_ERROR, "Failed to run variable op pass, null graph");
    return INTERNAL_ERROR;
  }

  GELOGD("Begin to run variable op pass on graph %s, session %lu, graph id %u", graph->GetName().c_str(),
         GetContext().SessionId(), graph->GetGraphID());

  if (var_accelerate_ctrl_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "Failed to run var op pass, the variable accelerate control is null");
    return INTERNAL_ERROR;
  }

  GELOGD("Begin to generate ref map for variable and refs, graph name:%s.", graph->GetName().c_str());
  if (RenewVarDesc(graph) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to renew var desc on graph");
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  if (GenerateVariableVariableRefMap(graph) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to generate variable map for graph %s", graph->GetName().c_str());
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  GELOGD("Begin to fusion variables and trans nodes");
  for (auto &var_to_refs : var_and_var_ref_map_) {
    auto &node = var_to_refs.first;
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(var_accelerate_ctrl_);
    if (!var_accelerate_ctrl_->IsVarPermitToChangeFormats(node->GetName())) {
      GELOGD("The var %s does not permit to change formats, skip it", node->GetName().c_str());
      continue;
    }

    VarTransRoad fusion_road;
    auto ret = FusionIfNeed(node, fusion_road);
    if (ret != SUCCESS) {
      return ret;
    }

    if (fusion_road.empty()) {
      GELOGD("No need to fusion variable %s because it's fusion road is empty", node->GetName().c_str());
      continue;
    }

    ret = RenewTransRoadDesc(node, fusion_road);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to renew description fusion road for var %s", node->GetName().c_str());
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }

    auto start_iter = fusion_road.begin();
    auto end_iter = fusion_road.rbegin();
    GELOGD(
        "Trans variable data for %s from format %s to %s, shape %s to %s "
        "data-type %s to %s, path len %zu success",
        node->GetName().c_str(), TypeUtils::FormatToSerialString(start_iter->input.GetFormat()).c_str(),
        TypeUtils::FormatToSerialString(end_iter->output.GetFormat()).c_str(),
        formats::ShapeToString(start_iter->input.GetShape().GetDims()).c_str(),
        formats::ShapeToString(end_iter->output.GetShape().GetDims()).c_str(),
        TypeUtils::DataTypeToSerialString(start_iter->input.GetDataType()).c_str(),
        TypeUtils::DataTypeToSerialString(end_iter->output.GetDataType()).c_str(), fusion_road.size());

    ret = VarManager::Instance(graph->GetSessionID())->SetTransRoad(node->GetName(), fusion_road);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to update the format fusion road for var %s", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    ret = VarManager::Instance(graph->GetSessionID())->SetChangedGraphId(node->GetName(), graph->GetGraphID());
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to update the graph id for var %s", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    var_accelerate_ctrl_->SetVarChanged(node->GetName());

    GELOGD("Begin to update format info for var %s.", node->GetName().c_str());
    std::set<ge::NodePtr> node_set({node});
    if (UpdateIOFormatInfo(end_iter->output, node_set) != SUCCESS) {
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }

    // renew var desc if the trans_road is all reshape or reformat
    ret = RenewVarDesc(graph->GetSessionID(), node, fusion_road);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "var manager renew var[%s] descriptor failed!", node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status VariableOpPass::RenewTransRoadDesc(const NodePtr &var, VarTransRoad &fusion_road) {
  auto var_desc = var->GetOpDesc();
  GE_CHECK_NOTNULL(var_desc);
  TransNodeInfo prev_node_info;
  prev_node_info.node_type = var->GetType();
  prev_node_info.output = var_desc->GetOutputDesc(0);
  // two cases
  // fisrt Var->cast->transdata which transdata in fusion road
  // the input of transdata is not equal with output of var
  // case 1 : suppose input dtype of transdata equal with out dtype
  // but not equal with var
  // so we make input dtype and output dytpe of transroad equal with var
  // case 2: suppose input format of transdata not equal with out format
  // and input format not equal with var
  // so we make input format equal with var

  for (auto &cur_trans : fusion_road) {
    if (cur_trans.input.GetFormat() == cur_trans.output.GetFormat()) {
      cur_trans.output.SetFormat(prev_node_info.output.GetFormat());
    }
    if (cur_trans.input.GetDataType() == cur_trans.output.GetDataType()) {
      cur_trans.output.SetDataType(prev_node_info.output.GetDataType());
    }
    if (ge::formats::IsShapeEqual(cur_trans.input.GetShape(), cur_trans.output.GetShape())) {
      cur_trans.output.SetShape(prev_node_info.output.GetShape());
    }
    cur_trans.input = prev_node_info.output;
    prev_node_info.output = cur_trans.output;
  }
  return SUCCESS;
}

Status VariableOpPass::FusionIfNeed(const NodePtr &var, VarTransRoad &fusion_road) {
  bool can_fusion = false;
  while (true) {
    map<string, vector<NodePtr>> trans_type_to_trans_ops ;
    map<string, pair<string, bool>> trans_type_to_changed_desc;
    // record the order of trans op in first path
    vector<string> first_path_trans_order;
    auto ret = CheckIfCouldBeOptimized(var, first_path_trans_order, trans_type_to_changed_desc,
                                       trans_type_to_trans_ops, can_fusion);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Check trans ops after vatiable could be optimized or not failed");
      return ret;
    }

    if (!can_fusion) {
      break;
    }

    vector<pair<NodePtr, NodePtr>> delete_var_ref_trans_nodes;
    ret = GetAndCheckTransOpOfVarRef(var, can_fusion, trans_type_to_changed_desc, delete_var_ref_trans_nodes);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "get and check trans op of varref failed");
      return ret;
    }

    if (!can_fusion) {
      break;
    }

    ret = UpdateTransRoad(fusion_road, first_path_trans_order,
                          trans_type_to_changed_desc, trans_type_to_trans_ops);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Update trans road failed");
      return ret;
    }

    if (fusion_road.empty()) {
      return SUCCESS;
    }

    ret = DealFusion(var, fusion_road, trans_type_to_changed_desc,
                     trans_type_to_trans_ops, delete_var_ref_trans_nodes);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}

Status VariableOpPass::UpdateTransRoad(VarTransRoad &fusion_road, vector<std::string> &first_path_trans_order,
                                       map<std::string,std::pair<std::string, bool>> &trans_type_to_changed_desc,
                                       map<std::string,vector<NodePtr>> &trans_type_to_trans_ops){
  vector<std::string> delete_trans_type;
  for (auto &trans_type : first_path_trans_order) {
    if (trans_type_to_changed_desc.find(trans_type) == trans_type_to_changed_desc.end()) {
      continue;
    }
    bool delete_flag = false;
    for (auto &trans_node : trans_type_to_trans_ops[trans_type]) {
      int tran_in_index = TransOpUtil::GetTransOpDataIndex(trans_node->GetType());
      auto out_op_desc = trans_node->GetOpDesc();
      GE_CHECK_NOTNULL(out_op_desc);
      TransNodeInfo trans_node_info;
      trans_node_info.node_type = trans_node->GetType();
      trans_node_info.input = out_op_desc->GetInputDesc(tran_in_index);
      trans_node_info.output = out_op_desc->GetOutputDesc(kTransOpOutIndex);
      if (!IsTransSupport(trans_node_info)) {
        delete_flag = true;
        GELOGD("The trans node %s does not support, skip the variable accelerating", trans_node_info.node_type.c_str());
        break;
      }
    }
    if (delete_flag) {
      delete_trans_type.push_back(trans_type);
    } else {
      auto &trans_node = *trans_type_to_trans_ops[trans_type].begin();
      auto out_op_desc = trans_node->GetOpDesc();
      int tran_in_index = TransOpUtil::GetTransOpDataIndex(trans_node->GetType());
      TransNodeInfo trans_node_info;
      trans_node_info.node_type = trans_node->GetType();
      trans_node_info.input = out_op_desc->GetInputDesc(tran_in_index);
      trans_node_info.output = out_op_desc->GetOutputDesc(kTransOpOutIndex);
      fusion_road.emplace_back(trans_node_info);
    }
  }
  for (auto &trans_type : delete_trans_type) {
    trans_type_to_changed_desc.erase(trans_type);
  }
  return SUCCESS;
}

Status VariableOpPass::DealFusion(const ge::NodePtr &var_node, VarTransRoad &fusion_road,
                                  map<std::string, std::pair<std::string, bool>> trans_type_to_changed_desc,
                                  map<std::string, vector<NodePtr>> trans_type_to_trans_ops,
                                  vector<pair<NodePtr, NodePtr>> &delete_trans_nodes) {
  GE_CHECK_NOTNULL(var_node);
  GELOGD("Begin to fusion var %s with trans", var_node->GetName().c_str());
  auto graph = var_node->GetOwnerComputeGraph();
  for (auto &trans_type : trans_type_to_changed_desc) {
    for (auto &trans_node : trans_type_to_trans_ops[trans_type.first]) {
      GELOGD("Remove node %s type %s when fusion with variable %s", trans_node->GetName().c_str(),
             trans_node->GetType().c_str(), var_node->GetName().c_str());
      if (RenewTransOpDesc(trans_node, true) != SUCCESS) {
        return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
      }

      if (GraphUtils::IsolateNode(trans_node, {0}) != SUCCESS) {
        return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
      }

      if (GraphUtils::RemoveNodeWithoutRelink(graph, trans_node) != SUCCESS) {
        return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
      }
    }
  }

  // Iterate delete_trans_nodes backward, eg a->b->c, delete_trans_nodes:{{b,c},{a,b}}
  // we should delete {a,b} first , then b->c,then we can delete {b,c}
  // if we delete {b,c} first, then a->c, then we can not get b when we delete {a,b}
  for (auto iter = delete_trans_nodes.rbegin(); iter != delete_trans_nodes.rend(); ++iter) {
    auto front_node = iter->first;
    auto back_node = iter->second;
    if (RenewTransOpDesc(front_node, false) != SUCCESS) {
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }
    if (front_node->GetOutDataNodes().size() > 1) {
      GELOGD("The trans node %s type %s connecting with var-ref %s has more"
             " than one output data nodes, unlink the edge between them",
             front_node->GetName().c_str(), front_node->GetType().c_str(), back_node->GetName().c_str());
      if (ByPassTransNode(front_node, back_node) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to bypass trans node %s to node %s", front_node->GetName().c_str(),
               back_node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    } else {
      GELOGD("The trans node %s type %s connecting with  %s has only"
             " one output data nodes, isolate and remove it.",
             front_node->GetName().c_str(), front_node->GetType().c_str(), back_node->GetName().c_str());
      if (GraphUtils::IsolateNode(front_node, {0}) != SUCCESS) {
        return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
      }
      if (GraphUtils::RemoveNodeWithoutRelink(graph, front_node) != SUCCESS) {
        return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
      }
    }
  }
  return SUCCESS;
}

Status VariableOpPass::RenewTransOpDesc(ge::NodePtr &node, bool is_reverse) {
  int tran_in_index = TransOpUtil::GetTransOpDataIndex(node->GetType());
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GeTensorDesc input_desc = op_desc->GetInputDesc(tran_in_index);
  GeTensorDesc output_desc = op_desc->GetOutputDesc(kTransOpOutIndex);
  GeTensorDesc renew_desc = is_reverse ? output_desc : input_desc;
  bool format_changed = false;
  bool shape_changed = false;
  bool dtype_changed = false;
  if (input_desc.GetFormat() != output_desc.GetFormat()) {
    format_changed = true;
  }
  if (input_desc.GetDataType() != output_desc.GetDataType()) {
    dtype_changed = true;
  }
  if (!ge::formats::IsShapeEqual(input_desc.GetShape(), output_desc.GetShape())) {
    shape_changed = true;
  }
  auto cur_node = node;
  while (TransOpUtil::IsTransOp(cur_node)) {
    tran_in_index = TransOpUtil::GetTransOpDataIndex(cur_node->GetType());
    auto next_node = is_reverse ? NodeUtils::GetInDataNodeByIndex(*cur_node, tran_in_index) :
                     cur_node->GetOutDataNodes().at(kTransOpOutIndex);
    if (!TransOpUtil::IsTransOp(next_node)) {
      break;
    }
    auto prev_desc = next_node->GetOpDesc();
    tran_in_index = TransOpUtil::GetTransOpDataIndex(next_node->GetType());
    auto mutable_output_desc = prev_desc->MutableOutputDesc(kTransOpOutIndex);
    auto mutable_input_desc = prev_desc->MutableInputDesc(tran_in_index);
    GE_CHECK_NOTNULL(prev_desc->MutableOutputDesc(kTransOpOutIndex));
    GE_CHECK_NOTNULL(prev_desc->MutableInputDesc(tran_in_index));
    if (shape_changed) {
      mutable_input_desc->SetShape(renew_desc.GetShape());
      mutable_output_desc->SetShape(renew_desc.GetShape());
    }
    if (dtype_changed) {
      mutable_input_desc->SetDataType(renew_desc.GetDataType());
      mutable_output_desc->SetDataType(renew_desc.GetDataType());
    }
    if (format_changed) {
      mutable_input_desc->SetFormat(renew_desc.GetFormat());
      mutable_output_desc->SetFormat(renew_desc.GetFormat());
    }
    cur_node = next_node;
  }
  return SUCCESS;
}

Status VariableOpPass::CheckIfCouldBeOptimized(const NodePtr &var, vector<string> &first_path_trans_order,
                                               map<string, pair<string, bool>> &trans_type_to_changed_desc,
                                               map<string, vector<NodePtr>> &trans_type_to_trans_ops, bool &flag) {
  bool is_match = true;
  auto ret = GetSameTransOP(var, first_path_trans_order, trans_type_to_changed_desc,
                            trans_type_to_trans_ops, is_match);

  if (ret != SUCCESS) {
    GELOGE(FAILED, "Get same trans op of variable node: %s failed", var->GetName().c_str());
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  if (!is_match) {
    flag = false;
    GELOGI("trans nodes after variable do not meet the condition");
    return SUCCESS;
  }

  flag = true;
  return SUCCESS;
}

Status VariableOpPass::GetSameTransOP(const NodePtr &var, vector<string> &first_path_trans_order,
                                      map<string, pair<string, bool>> &trans_type_to_changed_desc,
                                      map<string, vector<NodePtr>> &trans_type_to_trans_ops, bool &is_match) {
  GELOGD("Begin to get Node: %s trans op info of first path", var->GetName().c_str());
  auto ret = GetFisrtPathTransInfo(var, first_path_trans_order,
                                   trans_type_to_changed_desc, trans_type_to_trans_ops);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Get var: %s first path trans info failed", var->GetName().c_str());
    return FAILED;
  }

  if (first_path_trans_order.empty()) {
    GELOGD("var %s first path has no trans op, not need to pass", var->GetName().c_str());
    is_match = false;
    return SUCCESS;
  }

  GELOGD("Begin to depth first search Node: %s ", var->GetName().c_str());
  VariableDFS(var, trans_type_to_changed_desc, trans_type_to_trans_ops, is_match);

  return SUCCESS;
}

void VariableOpPass::VariableDFS(const NodePtr &node, map<string, pair<string, bool>> &trans_type_to_changed_desc,
                                 map<string, vector<NodePtr>> &trans_type_to_trans_ops, bool &is_match) {
  std::stack<NodePtr> node_stack;
  std::stack<vector<NodePtr>> path_stack;
  for (auto &out_node : node->GetOutDataNodes()) {
    if (!is_match) {
      break;
    }
    if (out_node->GetOutDataNodesSize() == 0 || !ge::TransOpUtil::IsTransOp(out_node)) {
      is_match = false;
      break;
    }
    node_stack.push(out_node);
    path_stack.emplace(vector<NodePtr>{out_node});
    while (!node_stack.empty() && is_match) {
      auto cur_node = node_stack.top();
      auto cur_path = path_stack.top();
      node_stack.pop();
      path_stack.pop();
      if (cur_node->GetOutDataNodesSize() == 0 || !ge::TransOpUtil::IsTransOp(cur_node)) {
        UpdateTransInfo(cur_path, is_match, trans_type_to_changed_desc, trans_type_to_trans_ops);
        continue;
      }
      for (auto &next_node : cur_node->GetOutDataNodes()) {
        node_stack.push(next_node);
        auto next_path = cur_path;
        next_path.push_back(next_node);
        path_stack.emplace(next_path);
      }
    }
  }
}

Status VariableOpPass::UpdateTransInfo(vector<NodePtr> &cur_path, bool& is_match,
                                       map<string, pair<string, bool>> &trans_type_to_changed_desc,
                                       map<string, vector<NodePtr>> &trans_type_to_trans_ops) {
  GELOGD("Begin to update trans info by path");
  std::set<string> trans_op_occured;
  for (auto &trans_node : cur_path) {
    auto trans_node_type = trans_node->GetType();
    if (trans_op_occured.find(trans_node_type) != trans_op_occured.end() ||
        !ge::TransOpUtil::IsTransOp(trans_node_type)) {
      continue;
    }
    trans_op_occured.insert(trans_node_type);
    auto desc_diff = GetInAndOutDecsDiff(trans_node);
    if (trans_type_to_changed_desc.find(trans_node_type) != trans_type_to_changed_desc.end() &&
        desc_diff == trans_type_to_changed_desc[trans_node_type].first) {
      trans_type_to_changed_desc[trans_node_type].second = true;
      auto iter = find(trans_type_to_trans_ops[trans_node_type].begin(),
                       trans_type_to_trans_ops[trans_node_type].end(),
                       trans_node);
      if (iter == trans_type_to_trans_ops[trans_node_type].end()) {
        trans_type_to_trans_ops[trans_node_type].push_back(trans_node);
      }
    }
  }
  std::set<string> delete_trans_types;
  for (auto &trans_item : trans_type_to_changed_desc) {
    if (!trans_item.second.second) {
      delete_trans_types.insert(trans_item.first);
    } else {
      trans_item.second.second = false;
    }
  }
  for (auto& delete_item : delete_trans_types) {
    trans_type_to_changed_desc.erase(delete_item);
  }
  if (trans_type_to_changed_desc.empty()) {
    is_match = false;
  }
  return SUCCESS;
}

Status VariableOpPass::GetFisrtPathTransInfo(const NodePtr &var, vector<string> &first_path_trans_order,
                                             map<string, pair<string, bool>> &trans_type_to_changed_desc,
                                             map<string, vector<NodePtr>> &trans_type_to_trans_ops) {
  auto cur_node = var;
  while (cur_node->GetOutDataNodesSize() != 0) {
    cur_node = cur_node->GetOutDataNodes().at(0);
    GE_CHECK_NOTNULL(cur_node);
    if (!ge::TransOpUtil::IsTransOp(cur_node)) {
      break;
    }
    auto cur_node_type = cur_node->GetType();
    // only get the the first occurrence operator of same type
    if (trans_type_to_changed_desc.find(cur_node_type) == trans_type_to_changed_desc.end()) {
      auto desc_diff = GetInAndOutDecsDiff(cur_node);
      trans_type_to_changed_desc[cur_node->GetType()] = make_pair(desc_diff, false);
      trans_type_to_trans_ops[cur_node->GetType()] = vector<NodePtr>{cur_node};
      first_path_trans_order.push_back(cur_node->GetType());
    }
  }
  GELOGD("get var %s first path trans info success", var->GetName().c_str());
  return SUCCESS;
}

Status VariableOpPass::GetAndCheckTransOpOfVarRef(const ge::NodePtr &var_node, bool &pass_check,
                                                  map<string, pair<string, bool>> &trans_type_to_changed_desc,
                                                  vector<pair<NodePtr, NodePtr>> &delete_var_ref_trans_nodes) {
  auto iterator = var_and_var_ref_map_.find(var_node);
  if (iterator == var_and_var_ref_map_.end()) {
    GELOGD("there is no var_ref of node %s", var_node->GetName().c_str());
    return SUCCESS;
  }
  vector<string> delete_trans_type;
  for (auto &trans_type : trans_type_to_changed_desc) {
    delete_trans_type.push_back(trans_type.first);
  }
  for (auto &ref_node : iterator->second) {
    GE_CHECK_NOTNULL(ref_node);
    auto cur_node = *ref_node->GetInDataNodes().begin();
    auto behind_node = ref_node;
    GE_CHECK_NOTNULL(cur_node);
    vector<string> tmp_delete_trans_type = delete_trans_type;
    while (TransOpUtil::IsTransOp(cur_node)) {
      GE_CHECK_NOTNULL(cur_node);
      auto iter = find(tmp_delete_trans_type.begin(), tmp_delete_trans_type.end(), cur_node->GetType());
      if (iter != tmp_delete_trans_type.end()) {
        CheckTransOpOfVarAndVarRefSymmetry(cur_node, trans_type_to_changed_desc[cur_node->GetType()].first,
                                           pass_check);
        if (!pass_check) {
          GELOGD("trans op : %s of var ref %s is illegal", cur_node->GetName().c_str(), ref_node->GetName().c_str());
          return SUCCESS;
        }
        tmp_delete_trans_type.erase(iter);
        delete_var_ref_trans_nodes.emplace_back(std::make_pair(cur_node, behind_node));
      }
      int tran_in_index = TransOpUtil::GetTransOpDataIndex(cur_node->GetType());
      behind_node = cur_node;
      cur_node = cur_node->GetInDataNodes().at(tran_in_index);
    }
    if (!tmp_delete_trans_type.empty()) {
      pass_check = false;
      return SUCCESS;
    }
  }
  return SUCCESS;
}

Status VariableOpPass::CheckTransOpOfVarAndVarRefSymmetry(NodePtr &var_ref_trans_op, const string &desc_diff,
                                                          bool &is_symmetry){
  auto var_ref_trans_op_desc_diff = GetInAndOutDecsDiff(var_ref_trans_op, true);
  is_symmetry = (var_ref_trans_op_desc_diff == desc_diff);
  return SUCCESS;
}

Status VariableOpPass::UpdateVarAndRefOutputFormatInfo(const GeTensorDesc &final_output, const ge::NodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    GELOGE(FAILED, "node or opdesc is nullptr");
    return FAILED;
  }
  const Format &format = final_output.GetFormat();
  const DataType &data_type = final_output.GetDataType();
  const GeShape &shape = final_output.GetShape();
  GELOGD("last ref is (%s, %s, %lu), var_ref_name is %s.", TypeUtils::DataTypeToSerialString(data_type).c_str(),
         TypeUtils::FormatToSerialString(format).c_str(), shape.GetDims().size(), node->GetName().c_str());

  auto node_desc = node->GetOpDesc()->GetOutputDesc(0);
  CopyVariableFormatDataTypeAndShape(final_output, node_desc);
  if (node->GetOpDesc()->UpdateOutputDesc(0, node_desc) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "update output desc fail.");
    return FAILED;
  }
  GELOGD("node ref is (%s, %s, %lu), var_ref_name is %s.",
         TypeUtils::DataTypeToSerialString(node->GetOpDesc()->GetOutputDesc(0).GetDataType()).c_str(),
         TypeUtils::FormatToSerialString(node->GetOpDesc()->GetOutputDesc(0).GetFormat()).c_str(),
         node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims().size(), node->GetName().c_str());

  auto iterator = var_and_var_ref_map_.find(node);
  if (iterator == var_and_var_ref_map_.end()) {
    auto graph = node->GetOwnerComputeGraph();
    if (GenerateVariableVariableRefMap(graph) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to generate variable map for graph %s", graph->GetName().c_str());
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }
  }
  iterator = var_and_var_ref_map_.find(node);
  if (iterator == var_and_var_ref_map_.end()) {
    GELOGW("The var node %s which belongs to graph %s can not be found on the graph", node->GetName().c_str(),
           node->GetOwnerComputeGraph()->GetName().c_str());
    return SUCCESS;
  }

  for (const auto &var_ref_node : iterator->second) {
    auto var_ref_node_description = var_ref_node->GetOpDesc();
    GE_CHECK_NOTNULL(var_ref_node_description);

    GELOGD("var_ref_node before is (%s, %s, %zu), var_ref_name is %s.",
           TypeUtils::DataTypeToSerialString(data_type).c_str(), TypeUtils::FormatToSerialString(format).c_str(),
           shape.GetDims().size(), var_ref_node->GetName().c_str());
    if (var_ref_node_description->UpdateOutputDesc(0, node_desc) != GRAPH_SUCCESS) {
      GELOGW("UpdateOutputDesc fail.");
    }
    if (var_ref_node_description->UpdateInputDesc(0, node_desc) != GRAPH_SUCCESS) {
      GELOGW("UpdateInputDesc fail.");
    }
    const auto &input_desc = var_ref_node_description->MutableInputDesc(0);
    const auto &output_desc = var_ref_node_description->MutableOutputDesc(0);
    GE_CHECK_NOTNULL(input_desc);
    GE_CHECK_NOTNULL(output_desc);
    GELOGD("var_ref_node ref is (%s, %s, %zu), var_ref_name is %s.",
           TypeUtils::DataTypeToSerialString(input_desc->GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(input_desc->GetFormat()).c_str(), output_desc->GetShape().GetDims().size(),
           var_ref_node->GetName().c_str());
  }

  return SUCCESS;
}

Status VariableOpPass::GenerateVariableVariableRefMap(const ComputeGraphPtr &compute_graph) {
  std::map<std::string, NodePtr> names_to_var;
  std::map<std::string, std::set<NodePtr>> names_to_refs;
  GE_CHECK_NOTNULL(compute_graph);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() != VARIABLE) {
      continue;
    }
    std::string ref_var_name;
    if (!ge::AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_name)) {
      names_to_var[node->GetName()] = node;
    } else {
      names_to_refs[ref_var_name].insert(node);
    }
  }

  for (auto &name_to_var : names_to_var) {
    var_and_var_ref_map_[name_to_var.second] = names_to_refs[name_to_var.first];
  }
  return SUCCESS;
}

void VariableOpPass::CopyVariableFormatDataTypeAndShape(const GeTensorDesc &src_tensor_desc,
                                                        GeTensorDesc &dst_tensor_desc) {
  dst_tensor_desc.SetShape(src_tensor_desc.GetShape());
  dst_tensor_desc.SetFormat(src_tensor_desc.GetFormat());
  dst_tensor_desc.SetDataType(src_tensor_desc.GetDataType());
}

Status VariableOpPass::UpdateIOFormatInfo(const GeTensorDesc &final_output, std::set<NodePtr> &nodes) {
  for (auto &need_set_node : nodes) {
    auto ret = UpdateVarAndRefOutputFormatInfo(final_output, need_set_node);
    if (ret != SUCCESS) {
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }
  }
  return SUCCESS;
}

Status VariableOpPass::RenewVarDesc(ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  // renew var manager desc
  Status ret = SUCCESS;
  for (auto &node : graph->GetDirectNode()) {
    bool is_var_node =
        (node->GetType() == VARIABLE) || (node->GetType() == VARIABLEV2) || (node->GetType() == VARHANDLEOP);
    if (is_var_node) {
      if (!ge::VarManager::Instance(graph->GetSessionID())->IsVarExist(node->GetName())) {
        GELOGD("var manager does not exist var node[%s]", node->GetName().c_str());
        continue;
      }
      GELOGD("var manager exist var node[%s], graph name[%s]", node->GetName().c_str(), graph->GetName().c_str());
      GE_CHECK_NOTNULL(node->GetOpDesc());
      ret = ge::VarManager::Instance(graph->GetSessionID())->RenewCurVarDesc(node->GetName(), node->GetOpDesc());
      if (ret != SUCCESS) {
        GELOGE(FAILED, "var manager renew var[%s] descriptor failed!", node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status VariableOpPass::RenewVarDesc(uint64_t session_id, const NodePtr &node, const VarTransRoad &fusion_road) {
  // renew var desc if the trans_road is all reshape or reformat
  for (auto &road : fusion_road) {
    if (road.node_type != RESHAPE && road.node_type != REFORMAT) {
      return SUCCESS;
    }
  }

  if (!ge::VarManager::Instance(session_id)->IsVarExist(node->GetName())) {
    GELOGD("var manager does not exist var node[%s]", node->GetName().c_str());
    return SUCCESS;
  }
  GELOGD("var manager exist var node[%s]", node->GetName().c_str());
  GE_CHECK_NOTNULL(node->GetOpDesc());
  Status ret = ge::VarManager::Instance(session_id)->RenewCurVarDesc(node->GetName(), node->GetOpDesc());
  if (ret != SUCCESS) {
    GELOGE(FAILED, "var manager renew var[%s] descriptor failed!", node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

}  // namespace ge
