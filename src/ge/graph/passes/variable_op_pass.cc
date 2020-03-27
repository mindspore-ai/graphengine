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

#include "graph/passes/variable_op_pass.h"

#include <string>
#include <vector>

#include "common/formats/formats.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "graph/graph.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
const int kTransOpOutIndex = 0;

std::string GetKey(Format format, DataType type, const std::vector<int64_t> &dims) {
  std::stringstream key;
  key << static_cast<int>(format) << '-';
  key << static_cast<int>(type) << '-';
  for (auto dim : dims) {
    key << dim << '-';
  }
  return key.str();
}

Status ByPassTransNode(NodePtr &trans_node, NodePtr &ref_node) {
  GE_CHECK_NOTNULL(trans_node);
  GE_CHECK_NOTNULL(ref_node);
  GELOGD("Begin to bypass trans node %s", trans_node->GetName().c_str());
  auto ret = GraphUtils::CopyInCtrlEdges(trans_node, ref_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR,
           "Failed to move control edges from trans "
           "node %s to var-ref %s",
           trans_node->GetName().c_str(), ref_node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  auto ref_in_anchor = ref_node->GetInDataAnchor(0);
  if (ref_in_anchor == nullptr) {
    GELOGE(INTERNAL_ERROR,
           "The variable ref node %s does not have an "
           "input anchor",
           ref_node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  ref_in_anchor->UnlinkAll();
  auto trans_in_anchor = trans_node->GetInDataAnchor(0);
  if (trans_in_anchor == nullptr) {
    GELOGE(INTERNAL_ERROR,
           "Failed to get the in data anchor from trans"
           " node %s type %s",
           trans_node->GetName().c_str(), trans_node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  auto prev_trans_node_out_anchor = trans_in_anchor->GetPeerOutAnchor();
  if (prev_trans_node_out_anchor == nullptr) {
    GELOGW(
        "The trans node %s does not have an input, so the ref node %s does"
        " not have any inputs after bypass",
        trans_node->GetName().c_str(), trans_node->GetName().c_str());
  } else {
    ret = GraphUtils::AddEdge(prev_trans_node_out_anchor, ref_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR,
             "Failed to add edge between ref node %s "
             "and the prev node of trans node %s",
             ref_node->GetName().c_str(), trans_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

bool IsTransSupport(const TransNodeInfo &trans_info) {
  if (trans_info.node_type == RESHAPE || trans_info.node_type == REFORMAT) {
    return true;
  } else if (trans_info.node_type == TRANSDATA) {
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
      GELOGD("No need to fusion variable and trans op for var %s", node->GetName().c_str());
      continue;
    }

    auto start_iter = fusion_road.begin();
    auto end_iter = fusion_road.rbegin();
    GELOGI(
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
  }

  return SUCCESS;
}

Status VariableOpPass::DealFusion(const ge::NodePtr &var_node) {
  GE_CHECK_NOTNULL(var_node);
  GELOGD("Begin to fusion var %s with trans", var_node->GetName().c_str());
  auto graph = var_node->GetOwnerComputeGraph();
  for (auto &trans_node : var_node->GetOutDataNodes()) {
    GELOGI("Remove node %s type %s when fusion with variable %s", trans_node->GetName().c_str(),
           trans_node->GetType().c_str(), var_node->GetName().c_str());

    if (GraphUtils::IsolateNode(trans_node, {0}) != SUCCESS) {
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }

    if (GraphUtils::RemoveNodeWithoutRelink(graph, trans_node) != SUCCESS) {
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }
  }

  auto iterator = var_and_var_ref_map_.find(var_node);
  if (iterator == var_and_var_ref_map_.end()) {
    GELOGD("there is no var_ref of node %s", var_node->GetName().c_str());
    return SUCCESS;
  }

  for (auto ref_node : iterator->second) {
    GE_CHECK_NOTNULL(ref_node);
    for (auto &trans_node : ref_node->GetInDataNodes()) {
      GELOGI("Remove node %s type %s when fusion with variable %s", trans_node->GetName().c_str(),
             trans_node->GetType().c_str(), var_node->GetName().c_str());
      if (trans_node->GetOutDataNodes().size() > 1) {
        GELOGD(
            "The trans node %s type %s connecting with var-ref %s has more"
            " than one output data nodes, unlink the edge between them",
            trans_node->GetName().c_str(), trans_node->GetType().c_str(), ref_node->GetName().c_str());
        if (ByPassTransNode(trans_node, ref_node) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "Failed to bypass trans node %s to ref %s", trans_node->GetName().c_str(),
                 ref_node->GetName().c_str());
          return INTERNAL_ERROR;
        }
      } else {
        GELOGD(
            "The trans node %s type %s connecting with var-ref %s has only"
            " one output data nodes, isolate and remove it.",
            trans_node->GetName().c_str(), trans_node->GetType().c_str(), ref_node->GetName().c_str());
        if (GraphUtils::IsolateNode(trans_node, {0}) != SUCCESS) {
          return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
        }
        if (GraphUtils::RemoveNodeWithoutRelink(graph, trans_node) != SUCCESS) {
          return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
        }
      }
    }
  }

  return SUCCESS;
}

Status VariableOpPass::CheckSameAndTransOp(const ge::NodePtr &var_node, bool &is_matched, VarTransRoad &fusion_road) {
  std::set<std::string> data_type_and_formats;
  std::string trans_op_type;
  ge::NodePtr out_node;
  ge::GeTensorDesc output_desc;
  GE_CHECK_NOTNULL(var_node);
  for (auto &out_node_and_anchor : var_node->GetOutDataNodesAndAnchors()) {
    auto in_anchor = out_node_and_anchor.second;
    GE_CHECK_NOTNULL(in_anchor);
    out_node = out_node_and_anchor.first;
    GE_CHECK_NOTNULL(out_node);
    auto trans_op_desc = out_node->GetOpDesc();
    GE_CHECK_NOTNULL(trans_op_desc);
    trans_op_type = trans_op_desc->GetType();

    GELOGD("current node type is %s.", trans_op_type.c_str());
    int data_index = TransOpUtil::GetTransOpDataIndex(trans_op_type);
    if (data_index < 0) {
      GELOGD("Variables only can be fusion with trans_op, the next op is %s type %s", out_node->GetName().c_str(),
             out_node->GetType().c_str());
      return SUCCESS;
    }
    if (data_index != in_anchor->GetIdx()) {
      GELOGD(
          "Variables only can be fusion with trans nodes, the next node %s"
          " type %s index %d does not trans anything(correct index %d)",
          out_node->GetName().c_str(), out_node->GetType().c_str(), in_anchor->GetIdx(), data_index);
      return SUCCESS;
    }

    output_desc = trans_op_desc->GetOutputDesc(kTransOpOutIndex);

    auto trans_op_format = output_desc.GetFormat();
    auto trans_op_data_type = output_desc.GetDataType();
    auto shape = output_desc.GetShape().GetDims();
    auto datatype_and_format = GetKey(trans_op_format, trans_op_data_type, shape);
    data_type_and_formats.insert(datatype_and_format);
  }

  if (data_type_and_formats.empty()) {
    return SUCCESS;
  }

  if (data_type_and_formats.size() > 1) {
    std::stringstream type_and_formats_stream;
    bool first_time = true;
    for (const auto &data_type_and_format : data_type_and_formats) {
      if (first_time) {
        first_time = false;
      } else {
        type_and_formats_stream << "|";
      }
      type_and_formats_stream << data_type_and_format;
    }

    GELOGW(
        "trans_op type size for var Node(%s) is over 1, Currently not"
        " supported, dataTypeAndFormats is %s.",
        var_node->GetName().c_str(), type_and_formats_stream.str().c_str());
    return SUCCESS;
  }

  int tran_in_index = TransOpUtil::GetTransOpDataIndex(out_node->GetType());
  auto out_op_desc = out_node->GetOpDesc();
  GE_CHECK_NOTNULL(out_op_desc);
  TransNodeInfo trans_node_info;
  trans_node_info.node_type = out_node->GetType();
  trans_node_info.input = out_op_desc->GetInputDesc(tran_in_index);
  trans_node_info.output = out_op_desc->GetOutputDesc(kTransOpOutIndex);

  if (!IsTransSupport(trans_node_info)) {
    GELOGD("The trans node %s does not support, skip the variable accelerating", trans_node_info.node_type.c_str());
    return SUCCESS;
  }

  is_matched = true;
  fusion_road.emplace_back(trans_node_info);

  return SUCCESS;
}

Status VariableOpPass::CheckVariableRefLegally(const ge::NodePtr &var_node, bool &is_var_ref_legally) {
  is_var_ref_legally = true;
  GE_CHECK_NOTNULL(var_node);
  auto iterator = var_and_var_ref_map_.find(var_node);
  if (iterator == var_and_var_ref_map_.end()) {
    GELOGD("var name %s are not in var var_ref map", var_node->GetName().c_str());
    return SUCCESS;
  }

  GELOGD("var name %s, ref var count %zu.", var_node->GetName().c_str(), iterator->second.size());

  for (const auto &var_ref_node : iterator->second) {
    if (CheckVarAndVarRefAreAlike(var_node, var_ref_node, is_var_ref_legally) != SUCCESS) {
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }

    GELOGD("is_var_ref_legally is %d", is_var_ref_legally);

    if (!is_var_ref_legally) {
      return SUCCESS;
    }
  }
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
    GELOGD("var_ref_node ref is (%s, %s, %zu), var_ref_name is %s.",
           TypeUtils::DataTypeToSerialString(var_ref_node_description->GetInputDesc(0).GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(var_ref_node_description->GetInputDesc(0).GetFormat()).c_str(),
           var_ref_node_description->GetOutputDesc(0).GetShape().GetDims().size(), var_ref_node->GetName().c_str());
  }

  return SUCCESS;
}

Status VariableOpPass::GenerateVariableVariableRefMap(const ComputeGraphPtr &compute_graph) {
  std::map<std::string, NodePtr> names_to_var;
  std::map<std::string, std::set<NodePtr>> names_to_refs;
  GE_CHECK_NOTNULL(compute_graph);
  for (auto &node : compute_graph->GetAllNodes()) {
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

Status VariableOpPass::CheckVarAndVarRefAreAlike(const NodePtr &var_node, const NodePtr &var_ref_node,
                                                 bool &is_var_and_variable_ref_are_alike) {
  GE_CHECK_NOTNULL(var_node);
  GE_CHECK_NOTNULL(var_ref_node);
  GELOGD("var_node GetOutDataNodes. name is %s.", var_node->GetName().c_str());
  const auto &var_node_trans_nodes = var_node->GetOutDataNodes();
  GELOGD("var_node_trans_nodes size is %zu.", var_node_trans_nodes.size());
  GELOGD("var_ref_node GetOutDataNodes. name is %s.", var_ref_node->GetName().c_str());
  const auto &var_ref_node_trans_nodes = var_ref_node->GetInDataNodes();
  GELOGD("var_ref_node_trans_nodes size is %zu.", var_ref_node_trans_nodes.size());

  if (var_ref_node_trans_nodes.size() > 1) {
    GELOGE(GE_GRAPH_VARIABLE_OP_PASS_FAILED, "var_ref_node_trans_nodes.size() > 1.");
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  const auto &var_node_trans_node = var_node_trans_nodes.at(0);
  const auto &var_ref_node_trans_node = var_ref_node_trans_nodes.at(0);

  if (CheckTransNodeAreInverse(var_node_trans_node, var_ref_node_trans_node, is_var_and_variable_ref_are_alike) !=
      SUCCESS) {
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  return SUCCESS;
}

Status VariableOpPass::CheckTransNodeAreInverse(const NodePtr &node_a, const NodePtr &node_b, bool &is_same) {
  GELOGD("In CheckTransNodeAreInverse.");
  GE_CHECK_NOTNULL(node_a);
  GE_CHECK_NOTNULL(node_b);
  const auto &node_a_op_desc = node_a->GetOpDesc();
  const auto &node_b_op_desc = node_b->GetOpDesc();
  GE_CHECK_NOTNULL(node_a_op_desc);
  GE_CHECK_NOTNULL(node_b_op_desc);
  const auto &node_a_out_op_desc = node_a_op_desc->GetOutputDesc(0);
  const auto &node_a_in_op_desc = node_a_op_desc->GetInputDesc(0);

  const auto &node_b_out_op_desc = node_b_op_desc->GetOutputDesc(0);
  const auto &node_b_in_op_desc = node_b_op_desc->GetInputDesc(0);

  is_same = IsOpDescSame(node_a_out_op_desc, node_b_in_op_desc) && IsOpDescSame(node_b_out_op_desc, node_a_in_op_desc);

  return SUCCESS;
}

bool VariableOpPass::IsOpDescSame(const GeTensorDesc &op_desc_a, const GeTensorDesc &op_desc_b) {
  const auto format_a = op_desc_a.GetFormat();
  const auto type_a = op_desc_a.GetDataType();
  const auto shape_a = op_desc_a.GetShape();

  const auto format_b = op_desc_b.GetFormat();
  const auto type_b = op_desc_b.GetDataType();
  const auto shape_b = op_desc_b.GetShape();

  const auto &dims_a = shape_a.GetDims();
  const auto &dims_b = shape_b.GetDims();
  GELOGD("(format, data type, shape) = (%s, %s, %zu) (%s, %s, %zu)", TypeUtils::FormatToSerialString(format_a).c_str(),
         TypeUtils::DataTypeToSerialString(type_a).c_str(), dims_a.size(),
         TypeUtils::FormatToSerialString(format_b).c_str(), TypeUtils::DataTypeToSerialString(type_b).c_str(),
         dims_b.size());
  return (format_a == format_b) && (type_a == type_b) && (dims_a == dims_b);
}

void VariableOpPass::CopyVariableFormatDataTypeAndShape(const GeTensorDesc &src_tensor_desc,
                                                        GeTensorDesc &dst_tensor_desc) {
  dst_tensor_desc.SetShape(src_tensor_desc.GetShape());
  dst_tensor_desc.SetFormat(src_tensor_desc.GetFormat());
  dst_tensor_desc.SetDataType(src_tensor_desc.GetDataType());
}

Status VariableOpPass::CheckIfCouldBeOptimized(const ge::NodePtr &node, bool &flag, VarTransRoad &fusion_road) {
  if (node == nullptr) {
    return FAILED;
  }
  bool is_matched = false;
  auto ret = CheckSameAndTransOp(node, is_matched, fusion_road);
  if (ret != SUCCESS) {
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }
  if (!is_matched) {
    flag = false;
    return SUCCESS;
  }

  bool is_var_ref_legally = false;
  ret = CheckVariableRefLegally(node, is_var_ref_legally);
  if (ret != SUCCESS) {
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }
  GELOGD("is_var_ref_legally is %d.", is_var_ref_legally);
  if (!is_var_ref_legally) {
    GELOGI("variable ref connection are illegally");
    flag = false;
    fusion_road.clear();
    return SUCCESS;
  }

  flag = true;
  GELOGD("node %s, is_matched = %d is_var_ref_legally = %d, flag = %d", node->GetName().c_str(), is_matched,
         is_var_ref_legally, flag);

  return SUCCESS;
}

Status VariableOpPass::FusionIfNeed(const NodePtr &var, VarTransRoad &fusion_road) {
  bool can_fusion = false;
  while (true) {
    auto ret = CheckIfCouldBeOptimized(var, can_fusion, fusion_road);
    if (ret != SUCCESS) {
      return ret;
    }
    if (!can_fusion) {
      break;
    }

    ret = DealFusion(var);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
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
    bool is_var_node = (node->GetType() == VARIABLE) || (node->GetType() == VARIABLEV2) ||
                       (node->GetType() == VARHANDLEOP);
    if (is_var_node) {
      if (!ge::VarManager::Instance(graph->GetSessionID())->IsVarExist(node->GetName())) {
        GELOGI("var manager does not exist var node[%s]", node->GetName().c_str());
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
}  // namespace ge
