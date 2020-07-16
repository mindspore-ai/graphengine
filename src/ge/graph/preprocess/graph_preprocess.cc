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

#include "graph/preprocess/graph_preprocess.h"
#include <map>
#include <set>
#include <string>
#include <utility>
#include "common/formats/format_transfers/format_transfer_nchw_nc1hwc0.h"
#include "common/formats/format_transfers/format_transfer_nhwc_nc1hwc0.h"
#include "common/formats/format_transfers/format_transfer_transpose.h"
#include "common/helper/model_helper.h"
#include "common/math/math_util.h"
#include "common/op/ge_op_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/common/ge_call_wrapper.h"
#include "graph/common/transop_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/optimize/graph_optimize.h"
#include "graph/passes/addn_pass.h"
#include "graph/passes/aicpu_constant_folding_pass.h"
#include "graph/passes/assert_pass.h"
#include "graph/passes/base_pass.h"
#include "graph/passes/common_subexpression_elimination_pass.h"
#include "graph/passes/cond_pass.h"
#include "graph/passes/cond_remove_pass.h"
#include "graph/passes/constant_folding_pass.h"
#include "graph/passes/constant_fuse_same_pass.h"
#include "graph/passes/control_trigger_pass.h"
#include "graph/passes/dimension_adjust_pass.h"
#include "graph/passes/dimension_compute_pass.h"
#include "graph/passes/dropout_pass.h"
#include "graph/passes/enter_pass.h"
#include "graph/passes/flow_ctrl_pass.h"
#include "graph/passes/for_pass.h"
#include "graph/passes/get_original_format_pass.h"
#include "graph/passes/guarantee_const_pass.h"
#include "graph/passes/hccl_group_pass.h"
#include "graph/passes/hccl_memcpy_pass.h"
#include "graph/passes/identity_pass.h"
#include "graph/passes/infershape_pass.h"
#include "graph/passes/iterator_op_pass.h"
#include "graph/passes/merge_pass.h"
#include "graph/passes/net_output_pass.h"
#include "graph/passes/next_iteration_pass.h"
#include "graph/passes/no_use_reshape_remove_pass.h"
#include "graph/passes/parallel_concat_start_op_pass.h"
#include "graph/passes/placeholder_with_default_pass.h"
#include "graph/passes/prevent_gradient_pass.h"
#include "graph/passes/print_op_pass.h"
#include "graph/passes/prune_pass.h"
#include "graph/passes/replace_transshape_pass.h"
#include "graph/passes/replace_with_empty_const_pass.h"
#include "graph/passes/resource_pair_add_control_pass.h"
#include "graph/passes/resource_pair_remove_control_pass.h"
#include "graph/passes/save_pass.h"
#include "graph/passes/shape_operate_op_remove_pass.h"
#include "graph/passes/snapshot_pass.h"
#include "graph/passes/stop_gradient_pass.h"
#include "graph/passes/subgraph_pass.h"
#include "graph/passes/switch_data_edges_bypass.h"
#include "graph/passes/switch_dead_branch_elimination.h"
#include "graph/passes/switch_fusion_pass.h"
#include "graph/passes/switch_logic_remove_pass.h"
#include "graph/passes/switch_op_pass.h"
#include "graph/passes/switch_split_pass.h"
#include "graph/passes/unused_const_pass.h"
#include "graph/passes/unused_op_remove_pass.h"
#include "graph/passes/var_is_initialized_op_pass.h"
#include "graph/passes/variable_prepare_op_pass.h"
#include "graph/preprocess/insert_op/util_insert_aipp_op.h"
#include "graph/types.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/pass_manager.h"
#include "init/gelib.h"
#include "multi_batch_copy_graph.h"
#include "runtime/dev.h"

#include "graph/passes/dimension_adjust_pass.h"
#include "graph/passes/identify_reference_pass.h"
#include "graph/passes/link_gen_mask_nodes_pass.h"
#include "graph/passes/permute_pass.h"
#include "graph/passes/reshape_remove_pass.h"
#include "graph/passes/same_transdata_breadth_fusion_pass.h"
#include "graph/passes/transop_breadth_fusion_pass.h"
#include "graph/passes/transop_depth_fusion_pass.h"
#include "graph/passes/transop_nearby_allreduce_fusion_pass.h"

#include "graph/passes/cast_remove_pass.h"
#include "graph/passes/data_pass.h"
#include "graph/passes/transop_without_reshape_fusion_pass.h"
#include "graph/passes/transpose_transdata_pass.h"
#include "graph/passes/variable_op_pass.h"
#include "graph/passes/variable_prepare_op_pass.h"
#include "graph/passes/variable_ref_delete_op_pass.h"

namespace ge {
namespace {
static std::map<std::string, ge::DataType> output_type_str_to_datatype = {
  {"FP32", ge::DT_FLOAT},    {"FP16", ge::DT_FLOAT16},  {"INT8", ge::DT_INT8},    {"INT16", ge::DT_INT16},
  {"UINT16", ge::DT_UINT16}, {"UINT8", ge::DT_UINT8},   {"INT32", ge::DT_INT32},  {"INT64", ge::DT_INT64},
  {"UINT32", ge::DT_UINT32}, {"UINT64", ge::DT_UINT64}, {"DOUBLE", ge::DT_DOUBLE}};

const char *const kMbatchSwitchnName = "mbatch-switch-name";

OpDescPtr CreateTensorShape(const GeTensorDesc &data_tensor) {
  GeTensorPtr tensor = MakeShared<GeTensor>();
  if (tensor == nullptr) {
    GELOGE(INTERNAL_ERROR, "Create shared ptr for GeTensor failed");
    return nullptr;
  }
  tensor->MutableTensorDesc().SetDataType(DT_INT32);
  tensor->MutableTensorDesc().SetFormat(FORMAT_ND);
  auto dst_ge_shape = data_tensor.GetShape();
  auto dim_cnt = static_cast<int64_t>(dst_ge_shape.GetDimNum());
  if (dim_cnt == 0) {  // if the dim_cnt is 0, the tensor is a scalar
    tensor->MutableTensorDesc().SetShape(GeShape());
    int64_t dst_shape = 1;
    if (tensor->SetData(reinterpret_cast<const uint8_t *>(&dst_shape), sizeof(int64_t)) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "tensor set data failed");
      return nullptr;
    }
  } else {
    tensor->MutableTensorDesc().SetShape(GeShape(std::vector<int64_t>({dim_cnt})));
    unique_ptr<int64_t[]> dst_shape(new (std::nothrow) int64_t[dim_cnt]());
    if (dst_shape == nullptr) {
      GELOGE(INTERNAL_ERROR, "Create unique ptr failed");
      return nullptr;
    }
    for (int64_t i = 0; i < dim_cnt; ++i) {
      dst_shape[i] = dst_ge_shape.GetDim(static_cast<size_t>(i));
    }

    GE_IF_BOOL_EXEC(
      tensor->SetData(reinterpret_cast<const uint8_t *>(dst_shape.get()), dim_cnt * sizeof(int64_t)) != GRAPH_SUCCESS,
      GELOGE(INTERNAL_ERROR, "tensor set data failed");
      return nullptr;)
  }

  GELOGD("Create shape input dim [%s]", dst_ge_shape.ToString().c_str());
  return OpDescUtils::CreateConstOp(tensor);
}

void AddTransNodeAttr(const std::string &node_type, const GeTensorDesc &input, const GeTensorDesc &output,
                      OpDescPtr &op_desc) {
  // For format transfer node, the IR definition has src/dst format attrs
  if (node_type == TRANSDATA) {
    GE_IF_BOOL_EXEC(
      !AttrUtils::SetStr(op_desc, FORMAT_TRANSFER_SRC_FORMAT, TypeUtils::FormatToSerialString(input.GetFormat())),
      GELOGW("SetStr FORMAT_TRANSFER_SRC_FORMAT failed");)
    GE_IF_BOOL_EXEC(
      !AttrUtils::SetStr(op_desc, FORMAT_TRANSFER_DST_FORMAT, TypeUtils::FormatToSerialString(output.GetFormat())),
      GELOGW("SetStr FORMAT_TRANSFER_DST_FORMAT failed");)
  }

  // For TransposeD node, the IR definition has perm attrs
  if (node_type == TRANSPOSED) {
    Format src_format = input.GetFormat();
    Format dst_format = output.GetFormat();
    std::vector<int64_t> perm_arg;
    GE_CHK_BOOL_EXEC_WARN(formats::GetPermByForamt(src_format, dst_format, perm_arg) == SUCCESS, return,
                          "Get perm by foramt failed.");
    GE_CHK_BOOL_EXEC_WARN(AttrUtils::SetListInt(op_desc, PERMUTE_ATTR_PERM, perm_arg), return,
                          "SetStr PERMUTE_ATTR_PERM failed")
  }
  // For cast node, the IR definition has src/dst attrs
  if (node_type == CAST) {
    GE_IF_BOOL_EXEC(!AttrUtils::SetInt(op_desc, CAST_ATTR_SRCT, static_cast<int64_t>(input.GetDataType())),
                    GELOGW("SetInt CAST_ATTR_SRCT failed");)
    GE_IF_BOOL_EXEC(!AttrUtils::SetInt(op_desc, CAST_ATTR_DSTT, static_cast<int64_t>(output.GetDataType())),
                    GELOGW("SetInt CAST_ATTR_DSTT failed");)
    GE_IF_BOOL_EXEC(!AttrUtils::SetInt(op_desc, CAST_ATTR_DST_TYPE, static_cast<int64_t>(output.GetDataType())),
                    GELOGW("SetInt CAST_ATTR_DST_TYPE failed");)
    GE_IF_BOOL_EXEC(!AttrUtils::SetBool(op_desc, CAST_ATTR_TRUNCATE, false),
                    GELOGW("SetBool CAST_ATTR_TRUNCATE failed");)
  }
}

NodePtr CreateTransNode(const std::string &name, const std::string &node_type, const GeTensorDesc &input,
                        const GeTensorDesc &output, NodePtr &node) {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "node is null.");
    return nullptr;
  }
  auto graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    GELOGE(PARAM_INVALID, "Owner graph is null, node name:%s.", node->GetName().c_str());
    return nullptr;
  }

  auto index = TransOpUtil::GetTransOpDataIndex(node_type);
  if (index < 0) {
    GELOGE(INTERNAL_ERROR, "The trans node type %s does not exists", node_type.c_str());
    return nullptr;
  }
  OpDescPtr op_desc = MakeShared<OpDesc>(name, node_type);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Create shared ptr for OpDesc failed");
    return nullptr;
  }

  // for data dump
  GE_IF_BOOL_EXEC(
    !AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::move(std::vector<std::string>())),
    GELOGW("CreateTransNode: SetListStr failed");)

  // Default single input and single output
  auto ret = op_desc->AddInputDesc(input);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to add input desc when create node %s type %s", name.c_str(), node_type.c_str());
    return nullptr;
  }
  ret = op_desc->AddOutputDesc(output);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to add output desc when create node %s type %s", name.c_str(), node_type.c_str());
    return nullptr;
  }

  AddTransNodeAttr(node_type, input, output, op_desc);

  NodePtr shape_node = nullptr;
  if (node_type == RESHAPE) {
    auto shape_desc = CreateTensorShape(output);
    if (shape_desc == nullptr) {
      GELOGE(INTERNAL_ERROR, "Failed to add shape for reshape %s, can not create the shape input",
             node->GetName().c_str());
      return nullptr;
    }
    ret = op_desc->AddInputDesc(shape_desc->GetOutputDesc(0));
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to add the first input for reshape %s", name.c_str());
      return nullptr;
    }

    shape_node = graph->AddNode(shape_desc);
    if (shape_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "Failed to add shape node for reshape %s, can not add the shape to graph", name.c_str());
      return nullptr;
    }
  }

  auto trans_node = graph->AddNode(op_desc);
  if (trans_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "Failed to add trans node %s to graph", name.c_str());
    return nullptr;
  }

  if (node_type == RESHAPE) {
    if (GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), trans_node->GetInDataAnchor(1)) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to add shape node for reshape %s, can not add the edge", name.c_str());
      return nullptr;
    }
  }

  return trans_node;
}

Status RecoverOneTransNodeForVar(const std::string &name, const TransNodeInfo &trans_node_info, NodePtr node,
                                 NodePtr &trans_node) {
  GE_CHECK_NOTNULL(node);
  trans_node = CreateTransNode(name, trans_node_info.node_type, trans_node_info.output, trans_node_info.input, node);
  if (trans_node == nullptr) {
    return INTERNAL_ERROR;
  }

  auto ret = GraphUtils::ReplaceNodeDataAnchors(trans_node, node, {}, {0});
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to replace out anchors when recover trans node for %s type %s",
           node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }

  ret = GraphUtils::AddEdge(node->GetOutDataAnchor(0), trans_node->GetInDataAnchor(0));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to connect node %s to trans node %s", node->GetName().c_str(),
           trans_node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  ret = GraphUtils::MoveOutCtrlEdges(node, trans_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to move out control edges from %s to %s when recover trans node.",
           node->GetName().c_str(), trans_node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status RecoverOneTransNodeForVarRef(const std::string &name, const TransNodeInfo &trans_node_info, NodePtr node,
                                    NodePtr &trans_node) {
  GE_CHECK_NOTNULL(node);
  trans_node = CreateTransNode(name, trans_node_info.node_type, trans_node_info.input, trans_node_info.output, node);
  if (trans_node == nullptr) {
    return INTERNAL_ERROR;
  }

  auto ret = GraphUtils::ReplaceNodeDataAnchors(trans_node, node, {0}, {});
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to replace int anchors when recover trans node for %s type %s",
           node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }

  ret = GraphUtils::AddEdge(trans_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to connect trans node %s to node %s", trans_node->GetName().c_str(),
           node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  ret = GraphUtils::MoveInCtrlEdges(node, trans_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to move int control edges from %s to %s when recover trans node.",
           node->GetName().c_str(), trans_node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status UpdateVarFormats(const NodePtr &var, const GeTensorDesc &tensor_desc) {
  GE_IF_BOOL_EXEC(var == nullptr, GELOGW("node : var is nullptr"); return INTERNAL_ERROR);
  GE_CHECK_NOTNULL(var->GetOpDesc());
  if (var->GetOpDesc()->GetOutputsSize() > 0) {
    auto output_desc = var->GetOpDesc()->GetOutputDesc(0);
    output_desc.SetFormat(tensor_desc.GetFormat());
    output_desc.SetDataType(tensor_desc.GetDataType());
    output_desc.SetShape(tensor_desc.GetShape());
    output_desc.SetOriginFormat(tensor_desc.GetOriginFormat());
    output_desc.SetOriginDataType(tensor_desc.GetOriginDataType());
    output_desc.SetOriginShape(tensor_desc.GetOriginShape());
    GE_IF_BOOL_EXEC(var->GetOpDesc()->UpdateOutputDesc(0, output_desc) != GRAPH_SUCCESS,
                    GELOGE(INTERNAL_ERROR, "UpdateOutputDesc failed");
                    return INTERNAL_ERROR;);
  }

  if (var->GetOpDesc()->GetInputsSize() > 0) {
    auto desc = var->GetOpDesc()->GetInputDesc(0);
    desc.SetFormat(tensor_desc.GetFormat());
    desc.SetDataType(tensor_desc.GetDataType());
    desc.SetShape(tensor_desc.GetShape());
    desc.SetOriginFormat(tensor_desc.GetOriginFormat());
    desc.SetOriginDataType(tensor_desc.GetOriginDataType());
    desc.SetOriginShape(tensor_desc.GetOriginShape());
    GE_IF_BOOL_EXEC(var->GetOpDesc()->UpdateInputDesc(0, desc) != GRAPH_SUCCESS,
                    GELOGE(INTERNAL_ERROR, "UpdateInputDesc failed");
                    return INTERNAL_ERROR;)
  }
  return SUCCESS;
}

Status RecoverTransRoadForVar(const NodePtr &var, const VarTransRoad &road) {
  GE_CHECK_NOTNULL(var);
  int index = 0;
  NodePtr last_node = var;
  for (auto iter = road.rbegin(); iter != road.rend(); ++iter) {
    auto trans_name = var->GetName() + "_trans_" + std::to_string(index++);
    auto ret = RecoverOneTransNodeForVar(trans_name, *iter, last_node, last_node);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to recover trans node for variable %s, index %d, type %s", var->GetName().c_str(),
             index, iter->node_type.c_str());
      return INTERNAL_ERROR;
    }
    // set stream_label
    OpDescPtr var_desc = var->GetOpDesc();
    GE_CHECK_NOTNULL(var_desc);
    std::string stream_label;
    (void)AttrUtils::GetStr(var_desc, ATTR_NAME_STREAM_LABEL, stream_label);
    if (!stream_label.empty()) {
      GE_CHK_STATUS_RET(SetStreamLabel(last_node, stream_label), "set stream label failed");
    }
    GE_CHK_BOOL_EXEC((ge::AttrUtils::SetBool(last_node->GetOpDesc(), ge::ATTR_INSERTED_BY_GE, true)),
                     return INTERNAL_ERROR, "Set attr ATTR_INSERTED_BY_GE failed.");
    GELOGD("Recover trans node %s type %s success", trans_name.c_str(), iter->node_type.c_str());
  }
  if (road.empty()) {
    return SUCCESS;
  }
  return UpdateVarFormats(var, road.rbegin()->output);
}

Status RecoverTransRoadForVarRef(const std::set<NodePtr> &nodes, const VarTransRoad &road) {
  for (auto &var : nodes) {
    GE_CHECK_NOTNULL(var);
    int index = 0;
    NodePtr last_node = var;
    GELOGI("Recover trans nodes for variable ref %s", var->GetName().c_str());
    for (auto iter = road.rbegin(); iter != road.rend(); ++iter) {
      auto trans_name = var->GetName() + "_trans_" + std::to_string(index++);
      auto ret = RecoverOneTransNodeForVarRef(trans_name, *iter, last_node, last_node);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to recover trans node for variable %s, index %d, type %s",
               var->GetName().c_str(), index, iter->node_type.c_str());
        return INTERNAL_ERROR;
      }
      // set stream_label
      OpDescPtr var_desc = var->GetOpDesc();
      GE_CHECK_NOTNULL(var_desc);
      std::string stream_label;
      (void)AttrUtils::GetStr(var_desc, ATTR_NAME_STREAM_LABEL, stream_label);
      if (!stream_label.empty()) {
        GE_CHK_STATUS_RET(SetStreamLabel(last_node, stream_label), "set stream label failed");
      }

      GE_CHK_BOOL_EXEC((ge::AttrUtils::SetBool(last_node->GetOpDesc(), ge::ATTR_INSERTED_BY_GE, true)),
                       return INTERNAL_ERROR, "Set attr ATTR_INSERTED_BY_GE failed.");
    }
    if (!(road.empty()) && (UpdateVarFormats(var, road.rbegin()->output) != SUCCESS)) {
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

using VarNamesToRefs = std::map<std::string, std::set<NodePtr>>;

VarNamesToRefs CollectVarNamesToRefs(const ComputeGraphPtr &graph) {
  VarNamesToRefs names_to_refs;
  std::string var_name;
  if (graph == nullptr) {
    GELOGE(PARAM_INVALID, "graph is null.");
    return names_to_refs;
  }
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() != VARIABLE) {
      continue;
    }
    if (AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, var_name)) {
      (void)names_to_refs[var_name].insert(node);
    }
  }
  return names_to_refs;
}
Status AddTransNodeBetweenTwoNodes(OutDataAnchorPtr &src_out, InDataAnchorPtr &insert_in,
                                   OutDataAnchorPtr &insert_out) {
  if ((src_out == nullptr) || (insert_in == nullptr) || (insert_out == nullptr)) {
    GELOGE(INTERNAL_ERROR, "anchor is nullptr");
    return FAILED;
  }
  auto vistor = src_out->GetPeerInDataAnchors();
  for (auto it = vistor.begin(); it != vistor.end(); ++it) {
    InDataAnchorPtr dst_in = *it;
    GE_CHK_STATUS_RET(src_out->Unlink(dst_in), "Unlink the anchor failed");
    GE_CHK_STATUS_RET(insert_out->LinkTo(dst_in), "Link the anchor failed");
  }
  GE_CHK_STATUS_RET(src_out->LinkTo(insert_in), "Link the anchor failed");
  return SUCCESS;
}

NodePtr CreateCastOp(const ge::GeShape &shape, const ge::DataType input_data_type, const ge::DataType output_data_type,
                     const ge::Format format, NodePtr &node) {
  static uint32_t transop_count = 0;
  std::string name = std::string("cast_node").append(std::to_string(transop_count++));

  GELOGI("create cast op:%s, input datatype:%s, out datatype:%s.", name.c_str(),
         TypeUtils::DataTypeToSerialString(input_data_type).c_str(),
         TypeUtils::DataTypeToSerialString(output_data_type).c_str());
  GeTensorDesc input(shape, format, input_data_type);
  input.SetOriginFormat(format);
  input.SetOriginShape(shape);
  input.SetOriginDataType(input_data_type);
  ge::TensorUtils::SetRealDimCnt(input, static_cast<uint32_t>(shape.GetDims().size()));

  GeTensorDesc output(shape, format, output_data_type);
  output.SetOriginFormat(format);
  output.SetOriginShape(shape);
  output.SetOriginDataType(output_data_type);
  ge::TensorUtils::SetRealDimCnt(output, static_cast<uint32_t>(shape.GetDims().size()));

  auto cast_node = CreateTransNode(name, CAST, input, output, node);
  GELOGD("Create cast node success.");
  return cast_node;
}

Status ProcessInputFP16(NodePtr &node_ptr, bool &is_dynamic_batch, NodePtr &switchn_node) {
  GE_CHECK_NOTNULL(node_ptr);
  auto op_desc = node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const GeTensorDescPtr &input = op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input);
  ge::DataType src_dtype = input->GetDataType();
  if (src_dtype == DT_FLOAT16) {
    GELOGI("The node name, %s dtype is fp16", node_ptr->GetName().c_str());
    return SUCCESS;
  }
  input->SetDataType(DT_FLOAT16);
  input->SetOriginDataType(DT_FLOAT16);
  int64_t input_shape_size = 0;
  int64_t output_shape_size = 0;
  ge::graphStatus input_graph_status = ge::TensorUtils::GetTensorSizeInBytes(*input, input_shape_size);
  ge::graphStatus output_graph_status = ge::TensorUtils::GetTensorMemorySizeInBytes(*input, output_shape_size);
  if (input_graph_status != ge::GRAPH_SUCCESS && output_graph_status != ge::GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "GetTensorSize failed!");
    return FAILED;
  }
  ge::TensorUtils::SetSize(*input, input_shape_size);
  const GeTensorDescPtr &output = op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(output);
  output->SetDataType(DT_FLOAT16);
  output->SetOriginDataType(DT_FLOAT16);
  ge::TensorUtils::SetSize(*output, output_shape_size);

  if (!is_dynamic_batch) {
    NodePtr cast_node = CreateCastOp(output->GetShape(), DT_FLOAT16, src_dtype, output->GetFormat(), node_ptr);
    GE_CHECK_NOTNULL(cast_node);
    OutDataAnchorPtr src_out = node_ptr->GetOutDataAnchor(0);
    InDataAnchorPtr cast_in = cast_node->GetInDataAnchor(0);
    OutDataAnchorPtr cast_out = cast_node->GetOutDataAnchor(0);
    if (AddTransNodeBetweenTwoNodes(src_out, cast_in, cast_out) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "add node between two nodes failed, src name:%s, cast node name:%s.",
             node_ptr->GetName().c_str(), cast_node->GetName().c_str());
      return FAILED;
    }
  } else {
    auto switchn_op_desc = switchn_node->GetOpDesc();
    GE_CHECK_NOTNULL(switchn_op_desc);
    const GeTensorDescPtr &switchn_input = switchn_op_desc->MutableInputDesc(0);
    GE_CHECK_NOTNULL(switchn_input);
    switchn_input->SetDataType(DT_FLOAT16);
    switchn_input->SetOriginDataType(DT_FLOAT16);
    for (uint32_t i = 0; i < switchn_node->GetAllOutDataAnchorsSize(); ++i) {
      const GeTensorDescPtr &switchn_output = switchn_op_desc->MutableOutputDesc(i);
      GE_CHECK_NOTNULL(switchn_output);
      switchn_output->SetDataType(DT_FLOAT16);
      switchn_output->SetOriginDataType(DT_FLOAT16);
      NodePtr cast_node =
        CreateCastOp(switchn_output->GetShape(), DT_FLOAT16, src_dtype, switchn_output->GetFormat(), node_ptr);
      GE_CHECK_NOTNULL(cast_node);
      OutDataAnchorPtr src_out = switchn_node->GetOutDataAnchor(i);
      InDataAnchorPtr cast_in = cast_node->GetInDataAnchor(0);
      OutDataAnchorPtr cast_out = cast_node->GetOutDataAnchor(0);
      if (AddTransNodeBetweenTwoNodes(src_out, cast_in, cast_out) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "add node between two nodes failed, src name:%s, cast node name:%s.",
               switchn_node->GetName().c_str(), cast_node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

NodePtr CreateTransdataNode(const ge::GeShape &in_shape, const ge::Format input_format, const ge::GeShape &out_shape,
                            const ge::Format output_format, const ge::DataType dt, NodePtr &node) {
  static uint32_t transop_count = 0;
  // Does not involve multithreading.
  std::string name = std::string("transdata_node").append(std::to_string(transop_count++));

  GELOGI("create trandata op:%s, input format:%s, out format:%s.", name.c_str(),
         TypeUtils::FormatToSerialString(input_format).c_str(), TypeUtils::FormatToSerialString(output_format).c_str());

  GeTensorDesc input(in_shape, input_format, dt);
  input.SetOriginFormat(input_format);
  input.SetOriginShape(in_shape);
  input.SetOriginDataType(dt);

  GeTensorDesc output(out_shape, output_format, dt);
  output.SetOriginFormat(output_format);
  output.SetOriginShape(out_shape);
  output.SetOriginDataType(dt);

  return CreateTransNode(name, TRANSDATA, input, output, node);
}

Status TransferShape2NC1HWC0(Format src_format, const std::vector<int64_t> &src_shape, DataType dt, Format dst_format,
                             std::vector<int64_t> &dst_shape) {
  if (src_format == FORMAT_NCHW) {
    formats::FormatTransferNchwNc1hwc0 transfer;
    if (transfer.TransShape(src_format, src_shape, dt, dst_format, dst_shape) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "TransShape failed");
      return FAILED;
    }
  } else if (src_format == FORMAT_NHWC) {
    formats::FormatTransferNhwcNc1hwc0 transfer;
    if (transfer.TransShape(src_format, src_shape, dt, dst_format, dst_shape) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "TransShape failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ModifyInputFormatAndShape(NodePtr &node_ptr) {
  GE_CHECK_NOTNULL(node_ptr);
  auto op_desc = node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const GeTensorDescPtr &input = op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input);
  ge::Format old_format = input->GetFormat();
  std::vector<int64_t> old_shape = input->GetShape().GetDims();
  ge::DataType dt = input->GetDataType();
  std::vector<int64_t> dst_shape_dims;
  if (TransferShape2NC1HWC0(old_format, old_shape, dt, FORMAT_NC1HWC0, dst_shape_dims) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Trans shape failed");
    return FAILED;
  }

  input->SetFormat(FORMAT_NC1HWC0);
  input->SetShape(ge::GeShape(dst_shape_dims));

  auto output = op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(output);
  output->SetFormat(FORMAT_NC1HWC0);
  output->SetShape(ge::GeShape(dst_shape_dims));

  int64_t size = 0;
  graphStatus graph_status = TensorUtils::GetTensorMemorySizeInBytes(*output, size);
  if (graph_status != ge::GRAPH_SUCCESS) {
    GELOGE(graph_status, "GetTensorSizeInBytes failed!");
    return FAILED;
  }
  ge::TensorUtils::SetSize(*output, size);
  ge::TensorUtils::SetSize(*input, size);

  return SUCCESS;
}

Status ModifyFormatAndShapeForSingleTensor(const GeTensorDescPtr &input_output) {
  GE_CHECK_NOTNULL(input_output);
  ge::Format old_format = input_output->GetFormat();
  std::vector<int64_t> old_shape = input_output->GetShape().GetDims();
  ge::DataType dt = input_output->GetDataType();
  std::vector<int64_t> dst_shape_dims;
  if (TransferShape2NC1HWC0(old_format, old_shape, dt, FORMAT_NC1HWC0, dst_shape_dims) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Trans shape failed");
    return FAILED;
  }
  input_output->SetFormat(FORMAT_NC1HWC0);
  input_output->SetShape(ge::GeShape(dst_shape_dims));
  return SUCCESS;
}

Status ProcessInputNC1HWC0(NodePtr &node_ptr, bool &is_dynamic_batch, NodePtr &switchn_node) {
  GE_CHECK_NOTNULL(node_ptr);
  auto op_desc = node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const GeTensorDescPtr &input = op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input);
  ge::Format old_format = input->GetFormat();
  ge::GeShape old_shape = input->GetShape();
  bool support = ((old_format == FORMAT_NC1HWC0) || (old_format == FORMAT_NCHW) || (old_format == FORMAT_NHWC));
  if (!support) {
    GELOGE(INTERNAL_ERROR, "The format [%s] is unsupported", TypeUtils::FormatToSerialString(old_format).c_str());
    return FAILED;
  }
  if (old_format == FORMAT_NC1HWC0) {
    GELOGI("No need to transfer format");
    return SUCCESS;
  }
  if (ModifyInputFormatAndShape(node_ptr) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "modify format and shape failed");
    return FAILED;
  }
  if (!is_dynamic_batch) {
    NodePtr trans_node =
      CreateTransdataNode(input->GetShape(), FORMAT_NC1HWC0, old_shape, old_format, input->GetDataType(), node_ptr);
    GE_CHECK_NOTNULL(trans_node);
    OutDataAnchorPtr src_out = node_ptr->GetOutDataAnchor(0);
    InDataAnchorPtr trans_in = trans_node->GetInDataAnchor(0);
    OutDataAnchorPtr trans_out = trans_node->GetOutDataAnchor(0);
    if (AddTransNodeBetweenTwoNodes(src_out, trans_in, trans_out) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "add node between two nodes failed");
      return FAILED;
    }
  } else {
    auto switchn_op_desc = switchn_node->GetOpDesc();
    GE_CHECK_NOTNULL(switchn_op_desc);
    const GeTensorDescPtr &switchn_input = switchn_op_desc->MutableInputDesc(0);
    if (ModifyFormatAndShapeForSingleTensor(switchn_input) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "modify format and shape failed");
      return FAILED;
    }
    for (uint32_t i = 0; i < switchn_node->GetAllOutDataAnchorsSize(); ++i) {
      const GeTensorDescPtr &switchn_output = switchn_op_desc->MutableOutputDesc(i);
      GE_CHECK_NOTNULL(switchn_output);
      old_format = switchn_output->GetFormat();
      old_shape = switchn_output->GetShape();
      if (ModifyFormatAndShapeForSingleTensor(switchn_output) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "modify format and shape failed");
        return FAILED;
      }
      NodePtr trans_node = CreateTransdataNode(switchn_output->GetShape(), FORMAT_NC1HWC0, old_shape, old_format,
                                               switchn_output->GetDataType(), node_ptr);
      GE_CHECK_NOTNULL(trans_node);
      OutDataAnchorPtr src_out = switchn_node->GetOutDataAnchor(i);
      InDataAnchorPtr cast_in = trans_node->GetInDataAnchor(0);
      OutDataAnchorPtr cast_out = trans_node->GetOutDataAnchor(0);
      if (AddTransNodeBetweenTwoNodes(src_out, cast_in, cast_out) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "add node between two nodes failed, src name:%s, cast node name:%s.",
               switchn_node->GetName().c_str(), trans_node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CheckIfDynamicBatchScene(NodePtr &data_node, bool &is_dynamic_batch, NodePtr &switchn_node) {
  is_dynamic_batch = false;
  std::string related_node_name;
  if (AttrUtils::GetStr(data_node->GetOpDesc(), kMbatchSwitchnName, related_node_name)) {
    if (related_node_name.empty()) {
      GELOGE(INTERNAL_ERROR, "The data node %s has switchn node flag, but the value is empty",
             data_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    for (const NodePtr &next_node : data_node->GetOutNodes()) {
      if (next_node->GetName() == related_node_name) {
        switchn_node = next_node;
        break;
      }
    }
    if (switchn_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "The data node %s has switchn node %s, but can not find it on the graph",
             data_node->GetName().c_str(), related_node_name.c_str());
      return INTERNAL_ERROR;
    }
    is_dynamic_batch = true;
  }
  return SUCCESS;
}

Status ProcessDataNode(NodePtr &node_ptr) {
  bool set_fp16 = false;
  if (!ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), "input_fp16", set_fp16) || !set_fp16) {
    return SUCCESS;
  }
  for (auto const &next_node : node_ptr->GetOutNodes()) {
    if (next_node->GetType() == AIPP) {
      GELOGE(INTERNAL_ERROR,
             "This input node [%s] is linked to aipp, can not be set to fp16,"
             "please check your atc parma insert_op_conf, input_fp16_nodes.",
             node_ptr->GetName().c_str());
      return FAILED;
    }
  }
  GELOGI("input_fp16 is found, the node name is %s.", node_ptr->GetName().c_str());
  bool is_dynamic_batch = false;
  NodePtr switchn_node = nullptr;
  if (CheckIfDynamicBatchScene(node_ptr, is_dynamic_batch, switchn_node)) {
    GELOGE(INTERNAL_ERROR, "CheckIfDynamicBatchScene failed");
    return FAILED;
  }
  if (ProcessInputFP16(node_ptr, is_dynamic_batch, switchn_node) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "ProcessInputFP16 failed");
    return FAILED;
  }
  // check if need to set format
  bool set_format = false;
  if (!ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), "input_set_nc1hwc0", set_format) || !set_format) {
    return SUCCESS;
  }
  GELOGI("The format of node [%s] should be set NC1HWC0.", node_ptr->GetName().c_str());
  if (ProcessInputNC1HWC0(node_ptr, is_dynamic_batch, switchn_node) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "ProcessInputNC1HWC0 failed");
    return FAILED;
  }
  return SUCCESS;
}

bool CheckIfSetOutputType(std::string &output_type, ge::DataType &output_data_type) {
  if (output_type_str_to_datatype.find(output_type) != output_type_str_to_datatype.end()) {
    output_data_type = output_type_str_to_datatype[output_type];
    return true;
  } else {
    GELOGI("output_type [%s] is not set or set unexpected", output_type.c_str());
    return false;
  }
  return false;
}
bool CheckOpType(const NodePtr &node, const std::string type) {
  if (node->GetType() == type) {
    return true;
  }
  return false;
}

Status ProcessFp16Nc1hwc0Dynamic(const OpDescPtr &src_op_desc, NodePtr &node) {
  auto merge_out = src_op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(merge_out);
  if (ModifyFormatAndShapeForSingleTensor(merge_out) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "modify format and shape failed");
    return FAILED;
  }
  for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
    auto merge_in = src_op_desc->MutableInputDesc(i);
    GE_CHECK_NOTNULL(merge_in);
    ge::Format old_format = merge_in->GetFormat();
    ge::GeShape old_shape = merge_in->GetShape();
    if (ModifyFormatAndShapeForSingleTensor(merge_in) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "modify format and shape failed");
      return FAILED;
    }
    ge::GeShape new_shape = merge_in->GetShape();
    NodePtr trans_node = CreateTransdataNode(old_shape, old_format, new_shape, FORMAT_NC1HWC0, DT_FLOAT16, node);
    GE_CHECK_NOTNULL(trans_node);
    const InDataAnchorPtr &dst_in_anchor = node->GetInDataAnchor(i);
    GE_CHECK_NOTNULL(dst_in_anchor);
    const OutDataAnchorPtr &src_out_anchor = dst_in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(src_out_anchor);
    if (GraphUtils::InsertNodeBetweenDataAnchors(src_out_anchor, dst_in_anchor, trans_node) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "InsertNodeBetweenDataAnchors failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ProcessNetoutputNodeFp16Nc1hwc0(GeTensorDesc &src_desc, const InDataAnchorPtr &in_anchor,
                                       GeTensorDescPtr &net_output_input_desc, NodePtr &node) {
  bool is_dynamic = CheckOpType(node, MERGE);
  auto src_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(src_op_desc);
  ge::GeShape src_shape = src_desc.GetShape();
  ge::Format src_format = src_desc.GetFormat();
  ge::DataType src_dtype = src_desc.GetDataType();
  if (src_dtype != DT_FLOAT16) {
    if (!is_dynamic) {
      auto peer_out = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out);
      NodePtr cast_node = CreateCastOp(src_shape, src_dtype, DT_FLOAT16, src_format, node);
      GE_CHECK_NOTNULL(cast_node);
      if (GraphUtils::InsertNodeBetweenDataAnchors(peer_out, in_anchor, cast_node) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "InsertNodeBetweenDataAnchors failed");
        return FAILED;
      }
    } else {
      // Update outputdesc
      const GeTensorDescPtr &merge_output = src_op_desc->MutableOutputDesc(0);
      GE_CHECK_NOTNULL(merge_output);
      merge_output->SetDataType(DT_FLOAT16);
      merge_output->SetOriginDataType(DT_FLOAT16);
      // Update input
      for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
        const GeTensorDescPtr &merge_input = src_op_desc->MutableInputDesc(i);
        GE_CHECK_NOTNULL(merge_input);
        src_shape = merge_input->GetShape();
        src_format = merge_input->GetFormat();
        src_dtype = merge_input->GetDataType();
        merge_input->SetDataType(DT_FLOAT16);
        merge_input->SetOriginDataType(DT_FLOAT16);
        const InDataAnchorPtr &dst_in_anchor = node->GetInDataAnchor(i);
        const OutDataAnchorPtr &src_out_anchor = dst_in_anchor->GetPeerOutAnchor();
        NodePtr cast_node = CreateCastOp(src_shape, src_dtype, DT_FLOAT16, src_format, node);
        if (GraphUtils::InsertNodeBetweenDataAnchors(src_out_anchor, dst_in_anchor, cast_node) != GRAPH_SUCCESS) {
          GELOGE(INTERNAL_ERROR, "InsertNodeBetweenDataAnchors failed");
          return FAILED;
        }
      }
    }
    net_output_input_desc->SetDataType(DT_FLOAT16);
    net_output_input_desc->SetOriginDataType(DT_FLOAT16);
  }
  if (src_format == FORMAT_NC1HWC0) {
    GELOGI("Format is NC1HWC0, no need to transfer");
    return SUCCESS;
  }
  std::vector<int64_t> dst_shape_dims;
  std::vector<int64_t> src_shape_dims = src_shape.GetDims();
  if (TransferShape2NC1HWC0(src_format, src_shape_dims, DT_FLOAT16, FORMAT_NC1HWC0, dst_shape_dims) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Trans shape failed");
    return FAILED;
  }
  ge::GeShape dst_shape(dst_shape_dims);
  net_output_input_desc->SetFormat(FORMAT_NC1HWC0);
  net_output_input_desc->SetOriginFormat(FORMAT_NC1HWC0);
  net_output_input_desc->SetShape(dst_shape);
  net_output_input_desc->SetOriginShape(dst_shape);
  if (!is_dynamic) {
    NodePtr trans_node = CreateTransdataNode(src_shape, src_format, dst_shape, FORMAT_NC1HWC0, DT_FLOAT16, node);
    GE_CHECK_NOTNULL(trans_node);
    auto peer_out_new = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_new);
    if (GraphUtils::InsertNodeBetweenDataAnchors(peer_out_new, in_anchor, trans_node) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "InsertNodeBetweenDataAnchors failed");
      return FAILED;
    }
  } else {
    if (ProcessFp16Nc1hwc0Dynamic(src_op_desc, node) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "ProcessFp16Nc1hwc0Dynamic failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ProcessOutputDynamic(const NodePtr &src_node, NodePtr &node, ge::DataType &output_data_type) {
  OpDescPtr src_op_desc = src_node->GetOpDesc();
  const GeTensorDescPtr &merge_output = src_op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(merge_output);
  merge_output->SetDataType(output_data_type);
  merge_output->SetOriginDataType(output_data_type);
  // Update input
  for (uint32_t i = 0; i < src_node->GetAllInDataAnchorsSize(); ++i) {
    const GeTensorDescPtr &merge_input = src_op_desc->MutableInputDesc(i);
    GE_CHECK_NOTNULL(merge_input);
    ge::GeShape src_shape = merge_input->GetShape();
    ge::Format src_format = merge_input->GetFormat();
    ge::DataType src_dtype = merge_input->GetDataType();
    merge_input->SetDataType(output_data_type);
    merge_input->SetOriginDataType(output_data_type);
    const InDataAnchorPtr &dst_in_anchor = src_node->GetInDataAnchor(i);
    GE_CHECK_NOTNULL(dst_in_anchor);
    const OutDataAnchorPtr &src_out_anchor = dst_in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(src_out_anchor);
    NodePtr cast_node = CreateCastOp(src_shape, src_dtype, output_data_type, src_format, node);
    if (GraphUtils::InsertNodeBetweenDataAnchors(src_out_anchor, dst_in_anchor, cast_node) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "InsertNodeBetweenDataAnchors failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ProcessNetoutputNode(NodePtr &node, std::string &output_type) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  ge::DataType output_data_type = ge::DT_FLOAT;
  bool is_set_output_type = CheckIfSetOutputType(output_type, output_data_type);

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    auto index = static_cast<uint32_t>(in_anchor->GetIdx());
    auto peer_out = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out);
    auto src_index = static_cast<uint32_t>(peer_out->GetIdx());
    auto src_node = peer_out->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    bool is_dynamic = CheckOpType(src_node, MERGE);

    OpDescPtr src_op_desc = src_node->GetOpDesc();
    GE_CHECK_NOTNULL(src_op_desc);
    auto net_output_input_desc = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(net_output_input_desc);

    ge::GeShape src_shape = src_op_desc->GetOutputDesc(src_index).GetShape();
    ge::Format src_format = src_op_desc->GetOutputDesc(src_index).GetFormat();
    ge::DataType src_dtype = src_op_desc->GetOutputDesc(src_index).GetDataType();
    // Update datatype
    if (is_set_output_type) {
      GELOGI("Enter into process output_type schedule");
      if (src_dtype == output_data_type) {
        GELOGI("Data type is same ,no need to transfer.");
        continue;
      }
      if (!is_dynamic) {
        NodePtr cast_node = CreateCastOp(src_shape, src_dtype, output_data_type, src_format, node);
        if (GraphUtils::InsertNodeBetweenDataAnchors(peer_out, in_anchor, cast_node) != GRAPH_SUCCESS) {
          GELOGE(INTERNAL_ERROR, "InsertNodeBetweenDataAnchors failed");
          return FAILED;
        }
      } else {
        // Update outputdesc
        if (ProcessOutputDynamic(src_node, node, output_data_type) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "ProcessOutputDynamic failed");
          return FAILED;
        }
      }
      net_output_input_desc->SetDataType(output_data_type);
      net_output_input_desc->SetOriginDataType(output_data_type);
      continue;
    }
    // output_node is not set,check if is_output_adjust_hw_layout is set
    bool set_fp16_nc1hwc0 = false;
    if (!is_dynamic) {
      (void)AttrUtils::GetBool(src_op_desc, "output_set_fp16_nc1hwc0", set_fp16_nc1hwc0);
    } else {
      // need check dynamic scene, graph structure: node->merge->netoutput
      const InDataAnchorPtr &merge_input_anchor = src_node->GetInDataAnchor(0);
      GE_CHECK_NOTNULL(merge_input_anchor);
      const OutDataAnchorPtr &src_out_anchor = merge_input_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(src_out_anchor);
      auto src_merge_node = src_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(src_merge_node);
      auto src_merge_node_opdesc = src_merge_node->GetOpDesc();
      (void)AttrUtils::GetBool(src_merge_node_opdesc, "output_set_fp16_nc1hwc0", set_fp16_nc1hwc0);
    }
    if (set_fp16_nc1hwc0) {
      GELOGI("Node [%s] should be set FP16 and NC1HWC0", src_op_desc->GetName().c_str());
      if ((src_format != FORMAT_NCHW) && (src_format != FORMAT_NHWC) && (src_format != FORMAT_NC1HWC0)) {
        GELOGE(INTERNAL_ERROR, "Format is not one of NCHW, NHWC, NC1HWC0.");
        return FAILED;
      }
      GeTensorDesc src_desc(src_shape, src_format, src_dtype);
      if (ProcessNetoutputNodeFp16Nc1hwc0(src_desc, in_anchor, net_output_input_desc, src_node) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Process netoutput fp16 nc1hwc0.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CheckIfNeedSetNdFormat(const NodePtr &node_ptr) {
  auto op = node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(op);
  auto inputDescsPtr = op->GetAllInputsDescPtr();
  auto outputDescsPtr = op->GetAllOutputsDescPtr();
  ge::Format format = ge::FORMAT_ND;
  // if user set shape larger than 4, inferformat may set NCHW or NHWC, GE should set ND before FE
  // process, otherwise fe will insert transdata.
  for (auto &inputDescPtr : inputDescsPtr) {
    GE_CHECK_NOTNULL(inputDescPtr);
    if ((inputDescPtr->GetShape().GetDims().size() > ge::DIM_DEFAULT_SIZE) &&
        ((inputDescPtr->GetFormat() == ge::FORMAT_NCHW) || (inputDescPtr->GetFormat() == ge::FORMAT_NHWC))) {
      GELOGI("The node inputdesc [%s] format need to be set ND", op->GetName().c_str());
      inputDescPtr->SetFormat(format);
      inputDescPtr->SetOriginFormat(format);
    }
  }
  for (auto &outputDescPtr : outputDescsPtr) {
    GE_CHECK_NOTNULL(outputDescPtr);
    if ((outputDescPtr->GetShape().GetDims().size() > ge::DIM_DEFAULT_SIZE) &&
        ((outputDescPtr->GetFormat() == ge::FORMAT_NCHW) || (outputDescPtr->GetFormat() == ge::FORMAT_NHWC))) {
      GELOGI("The node outputdesc [%s] format need to be set ND", op->GetName().c_str());
      outputDescPtr->SetFormat(format);
      outputDescPtr->SetOriginFormat(format);
    }
  }
  return SUCCESS;
}

// A new function ending in 'DynShape' has been added for the dynamic shape processing.
// In the dynamic shape process, transnode insertion by FE is advanced to the stage of whole
// graph optimization, GE only sets the final data_type/format/shape information for variable,
// data and netoutput, and no longer inserts the transnode.
Status ProcessInputFP16DynShape(NodePtr &node_ptr, bool &is_dynamic_batch, NodePtr &switchn_node) {
  GE_CHECK_NOTNULL(node_ptr);
  auto op_desc = node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const GeTensorDescPtr &input = op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input);
  ge::DataType src_dtype = input->GetDataType();
  if (src_dtype == DT_FLOAT16) {
    GELOGI("The node name, %s dtype is fp16", node_ptr->GetName().c_str());
    return SUCCESS;
  }
  input->SetDataType(DT_FLOAT16);
  input->SetOriginDataType(DT_FLOAT16);
  int64_t input_shape_size = 0;
  int64_t output_shape_size = 0;
  ge::graphStatus input_graph_status = ge::TensorUtils::GetTensorSizeInBytes(*input, input_shape_size);
  ge::graphStatus output_graph_status = ge::TensorUtils::GetTensorMemorySizeInBytes(*input, output_shape_size);
  if (input_graph_status != ge::GRAPH_SUCCESS && output_graph_status != ge::GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "GetTensorSize failed!");
    return FAILED;
  }
  ge::TensorUtils::SetSize(*input, input_shape_size);
  const GeTensorDescPtr &output = op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(output);
  output->SetDataType(DT_FLOAT16);
  output->SetOriginDataType(DT_FLOAT16);
  ge::TensorUtils::SetSize(*output, output_shape_size);
  if (is_dynamic_batch) {
    GELOGI("The node [%s] dtype set fp16", switchn_node->GetName().c_str());
    auto switchn_op_desc = switchn_node->GetOpDesc();
    GE_CHECK_NOTNULL(switchn_op_desc);
    auto switchn_input = switchn_op_desc->MutableInputDesc(0);
    GE_CHECK_NOTNULL(switchn_input);
    switchn_input->SetDataType(DT_FLOAT16);
    switchn_input->SetOriginDataType(DT_FLOAT16);
    for (uint32_t i = 0; i < switchn_node->GetAllOutDataAnchorsSize(); ++i) {
      const GeTensorDescPtr &switchn_output = switchn_op_desc->MutableOutputDesc(i);
      GE_CHECK_NOTNULL(switchn_output);
      switchn_output->SetDataType(DT_FLOAT16);
      switchn_output->SetOriginDataType(DT_FLOAT16);
    }
  }
  return SUCCESS;
}

Status ProcessInputNC1HWC0DynShape(NodePtr &node_ptr, bool &is_dynamic_batch, NodePtr &switchn_node) {
  GE_CHECK_NOTNULL(node_ptr);
  auto op_desc = node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const GeTensorDescPtr &input = op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input);
  ge::Format old_format = input->GetFormat();
  ge::GeShape old_shape = input->GetShape();
  bool support = ((old_format == FORMAT_NC1HWC0) || (old_format == FORMAT_NCHW) || (old_format == FORMAT_NHWC));
  if (!support) {
    GELOGE(INTERNAL_ERROR, "The format [%s] is unsupported", TypeUtils::FormatToSerialString(old_format).c_str());
    return FAILED;
  }
  if (old_format == FORMAT_NC1HWC0) {
    GELOGI("No need to transfer format");
    return SUCCESS;
  }
  if (ModifyInputFormatAndShape(node_ptr) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "modify format and shape failed");
    return FAILED;
  }
  if (is_dynamic_batch) {
    auto switchn_op_desc = switchn_node->GetOpDesc();
    GE_CHECK_NOTNULL(switchn_op_desc);
    const GeTensorDescPtr &switchn_input = switchn_op_desc->MutableInputDesc(0);
    if (ModifyFormatAndShapeForSingleTensor(switchn_input) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "modify format and shape failed");
      return FAILED;
    }
    for (uint32_t i = 0; i < switchn_node->GetAllOutDataAnchorsSize(); ++i) {
      auto switchn_output = switchn_op_desc->MutableOutputDesc(i);
      GE_CHECK_NOTNULL(switchn_output);
      old_format = switchn_output->GetFormat();
      old_shape = switchn_output->GetShape();
      if (ModifyFormatAndShapeForSingleTensor(switchn_output) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "modify format and shape failed");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status ProcessDataNodeDynShape(NodePtr &node_ptr) {
  auto op_desc = node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  bool set_fp16 = false;
  if (!ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), "input_fp16", set_fp16) || !set_fp16) {
    return SUCCESS;
  }
  for (auto const &next_node : node_ptr->GetOutNodes()) {
    if (next_node->GetType() == AIPP) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10049", {"opname"}, {node_ptr->GetName()});
      GELOGE(INTERNAL_ERROR,
             "This input op [%s] is linked to aipp, can not be set to fp16, "
             "please check your atc parameter --insert_op_conf, --input_fp16_nodes.",
             node_ptr->GetName().c_str());
      return FAILED;
    }
  }
  GELOGI("input_fp16 is found, the node name is %s.", node_ptr->GetName().c_str());
  bool is_dynamic_batch = false;
  NodePtr switchn_node = nullptr;
  if (CheckIfDynamicBatchScene(node_ptr, is_dynamic_batch, switchn_node)) {
    GELOGE(INTERNAL_ERROR, "CheckIfDynamicBatchScene failed");
    return FAILED;
  }
  if (ProcessInputFP16DynShape(node_ptr, is_dynamic_batch, switchn_node) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "ProcessInputFP16 failed");
    return FAILED;
  }
  // check if need to set format
  bool set_format = false;
  (void)ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), "input_set_nc1hwc0", set_format);
  if (set_format) {
    GELOGI("The format of node [%s] should be set NC1HWC0.", node_ptr->GetName().c_str());
    if (ProcessInputNC1HWC0DynShape(node_ptr, is_dynamic_batch, switchn_node) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "ProcessInputNC1HWC0 failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ProcessNetoutputNodeFp16Nc1hwc0DynShape(GeTensorDesc &src_desc, GeTensorDescPtr &net_output_input_desc,
                                               NodePtr &node) {
  bool is_dynamic = CheckOpType(node, MERGE);
  auto src_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(src_op_desc);
  ge::GeShape src_shape = src_desc.GetShape();
  ge::Format src_format = src_desc.GetFormat();

  net_output_input_desc->SetDataType(DT_FLOAT16);
  net_output_input_desc->SetOriginDataType(DT_FLOAT16);
  if (is_dynamic) {
    auto merge_output = src_op_desc->MutableOutputDesc(0);
    GE_CHECK_NOTNULL(merge_output);
    merge_output->SetDataType(DT_FLOAT16);
    merge_output->SetOriginDataType(DT_FLOAT16);
    for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
      auto merge_input = src_op_desc->MutableInputDesc(i);
      GE_CHECK_NOTNULL(merge_input);
      merge_input->SetDataType(DT_FLOAT16);
      merge_input->SetOriginDataType(DT_FLOAT16);
    }
  }

  if (src_format == FORMAT_NC1HWC0) {
    GELOGI("Format is NC1HWC0, no need to transfer");
    return SUCCESS;
  }
  std::vector<int64_t> dst_shape_dims;
  std::vector<int64_t> src_shape_dims = src_shape.GetDims();
  if (TransferShape2NC1HWC0(src_format, src_shape_dims, DT_FLOAT16, FORMAT_NC1HWC0, dst_shape_dims) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Trans shape failed");
    return FAILED;
  }
  ge::GeShape dst_shape(dst_shape_dims);
  net_output_input_desc->SetFormat(FORMAT_NC1HWC0);
  net_output_input_desc->SetShape(dst_shape);
  if (is_dynamic) {
    auto merge_out = src_op_desc->MutableOutputDesc(0);
    GE_CHECK_NOTNULL(merge_out);
    if (ModifyFormatAndShapeForSingleTensor(merge_out) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "modify format and shape failed");
      return FAILED;
    }
    for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
      auto merge_in = src_op_desc->MutableInputDesc(i);
      GE_CHECK_NOTNULL(merge_in);
      if (ModifyFormatAndShapeForSingleTensor(merge_in) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "modify format and shape failed");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

bool NeedUpdateOutputByOutputTypeParm(std::string &output_type, NodePtr &src_node, uint32_t src_index,
                                      ge::DataType &dt) {
  if (CheckIfSetOutputType(output_type, dt)) {
    GELOGI("All output node should be set datatype.");
    return true;
  }
  bool is_dynamic = CheckOpType(src_node, MERGE);
  auto op_desc = src_node->GetOpDesc();
  if (is_dynamic) {
    const InDataAnchorPtr &merge_input_anchor = src_node->GetInDataAnchor(0);
    GE_RT_FALSE_CHECK_NOTNULL(merge_input_anchor);
    const OutDataAnchorPtr &src_out_anchor = merge_input_anchor->GetPeerOutAnchor();
    GE_RT_FALSE_CHECK_NOTNULL(src_out_anchor);
    src_index = static_cast<uint32_t>(src_out_anchor->GetIdx());
    auto src_merge_node = src_out_anchor->GetOwnerNode();
    GE_RT_FALSE_CHECK_NOTNULL(src_merge_node);
    op_desc = src_merge_node->GetOpDesc();
    GE_RT_FALSE_CHECK_NOTNULL(op_desc);
  }
  vector<ge::DataType> output_data_type_vec;
  vector<uint32_t> index_vec;
  if ((ge::AttrUtils::GetListDataType(op_desc, "_output_dt_list", output_data_type_vec)) &&
      (ge::AttrUtils::GetListInt(op_desc, "_output_dt_index", index_vec))) {
    if (output_data_type_vec.size() != index_vec.size()) {
      GELOGW("output_dt_list size is not match output_dt_index size");
      return false;
    }
    for (uint32_t i = 0; i < index_vec.size(); ++i) {
      if (index_vec[i] == src_index) {
        dt = output_data_type_vec[i];
        GELOGI("Find node %s output %u datatype should set %s .", op_desc->GetName().c_str(), i,
               TypeUtils::DataTypeToSerialString(dt).c_str());
        return true;
      }
    }
  }
  return false;
}

Status ProcessNetoutputNodeDynShape(NodePtr &node, std::string &output_type) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  ge::DataType output_data_type = ge::DT_FLOAT;

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    auto index = static_cast<uint32_t>(in_anchor->GetIdx());
    auto peer_out = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out);
    auto src_index = static_cast<uint32_t>(peer_out->GetIdx());
    auto src_node = peer_out->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    bool is_dynamic = CheckOpType(src_node, MERGE);

    OpDescPtr src_op_desc = src_node->GetOpDesc();
    GE_CHECK_NOTNULL(src_op_desc);
    auto net_output_input_desc = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(net_output_input_desc);

    ge::GeShape src_shape = src_op_desc->GetOutputDesc(src_index).GetShape();
    ge::Format src_format = src_op_desc->GetOutputDesc(src_index).GetFormat();
    ge::DataType src_dtype = src_op_desc->GetOutputDesc(src_index).GetDataType();
    // Update datatype
    if (NeedUpdateOutputByOutputTypeParm(output_type, src_node, src_index, output_data_type)) {
      GELOGI("Enter into process output_type schedule");
      net_output_input_desc->SetDataType(output_data_type);
      net_output_input_desc->SetOriginDataType(output_data_type);
      if (is_dynamic) {
        auto merge_output = src_op_desc->MutableOutputDesc(0);
        GE_CHECK_NOTNULL(merge_output);
        merge_output->SetDataType(output_data_type);
        merge_output->SetOriginDataType(output_data_type);
        for (uint32_t i = 0; i < src_node->GetAllInDataAnchorsSize(); ++i) {
          auto merge_input = src_op_desc->MutableInputDesc(i);
          GE_CHECK_NOTNULL(merge_input);
          merge_input->SetDataType(output_data_type);
          merge_input->SetOriginDataType(output_data_type);
        }
      }
      continue;
    }
    // output_node is not set,check if is_output_adjust_hw_layout is set
    bool set_fp16_nc1hwc0 = false;
    if (!is_dynamic) {
      (void)AttrUtils::GetBool(src_op_desc, "output_set_fp16_nc1hwc0", set_fp16_nc1hwc0);
    } else {
      // need check dynamic scene, graph structure: node->merge->netoutput
      const InDataAnchorPtr &merge_input_anchor = src_node->GetInDataAnchor(0);
      GE_CHECK_NOTNULL(merge_input_anchor);
      const OutDataAnchorPtr &src_out_anchor = merge_input_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(src_out_anchor);
      auto src_merge_node = src_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(src_merge_node);
      auto src_merge_node_opdesc = src_merge_node->GetOpDesc();
      (void)AttrUtils::GetBool(src_merge_node_opdesc, "output_set_fp16_nc1hwc0", set_fp16_nc1hwc0);
    }

    if (set_fp16_nc1hwc0) {
      GELOGI("Node [%s] should be set FP16 and NC1HWC0", src_op_desc->GetName().c_str());
      if ((src_format != FORMAT_NCHW) && (src_format != FORMAT_NHWC) && (src_format != FORMAT_NC1HWC0)) {
        GELOGE(INTERNAL_ERROR, "Format is not one of NCHW, NHWC, NC1HWC0.");
        return FAILED;
      }
      GeTensorDesc src_desc(src_shape, src_format, src_dtype);
      if (ProcessNetoutputNodeFp16Nc1hwc0DynShape(src_desc, net_output_input_desc, src_node) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Process netoutput fp16 nc1hwc0.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

}  // namespace

GraphPrepare::GraphPrepare() : compute_graph_(nullptr) {}

GraphPrepare::~GraphPrepare() {}

/**
 * @param graph
 * @return
 */
Status GraphPrepare::UpdateVariableFormats(ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  auto var_names_to_refs = CollectVarNamesToRefs(graph);
  for (auto &node : graph->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() != VARIABLE) {
      continue;
    }
    auto trans_road = VarManager::Instance(graph->GetSessionID())->GetTransRoad(node->GetName());
    if (trans_road == nullptr) {
      GELOGD("The variable %s does not have any trans road", node->GetName().c_str());
      continue;
    }

    GELOGI("Recover the trans road for var %s reversely", node->GetName().c_str());

    auto ret = RecoverTransRoadForVar(node, *trans_road);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to recovery trans road for var %s", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    auto iter = var_names_to_refs.find(node->GetName());
    if (iter != var_names_to_refs.end()) {
      ret = RecoverTransRoadForVarRef(iter->second, *trans_road);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to recovery trans road for var ref %s", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status GraphPrepare::UpdateVariableFormatsDynShape(ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  auto var_names_to_refs = CollectVarNamesToRefs(graph);
  for (auto &node : graph->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() != VARIABLE) {
      continue;
    }
    auto trans_road = VarManager::Instance(graph->GetSessionID())->GetTransRoad(node->GetName());
    if (trans_road == nullptr) {
      GELOGD("The variable %s does not have any trans road", node->GetName().c_str());
      continue;
    }

    GELOGI("Recover the trans road for var %s reversely", node->GetName().c_str());

    if (!(trans_road->empty())) {
      auto ret = UpdateVarFormats(node, trans_road->rbegin()->output);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to update var formats for var %s", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }

    auto iter = var_names_to_refs.find(node->GetName());
    if (iter != var_names_to_refs.end()) {
      for (auto &var : iter->second) {
        if (!(trans_road->empty()) && (UpdateVarFormats(var, trans_road->rbegin()->input) != SUCCESS)) {
          GELOGE(INTERNAL_ERROR, "Failed to update var formats for ref var %s", var->GetName().c_str());
          return INTERNAL_ERROR;
        }
      }
    }
  }

  return SUCCESS;
}

void GraphPrepare::SetOptions(const ge::GraphManagerOptions &options) { options_ = options; }

Status GraphPrepare::Init(const ge::Graph &graph, uint64_t session_id) {
  compute_graph_ = GraphUtils::GetComputeGraph(graph);
  if (compute_graph_ != nullptr) {
    compute_graph_->SetSessionID(session_id);
  }
  Status ret = CheckGraph();
  if (ret != SUCCESS) {
    GELOGE(ret, "RunGraph graph check fail, ret:%u", ret);
    return ret;
  }
  (void)compute_graph_->TopologicalSorting();
  ret = CheckRefOp();
  if (ret != SUCCESS) {
    GELOGE(ret, "RunGraph check ref op fail, ret:%u", ret);
    return ret;
  }

  return SUCCESS;
}

Status GraphPrepare::CheckGraph() {
  if (compute_graph_ == nullptr) {
    GELOGE(GE_GRAPH_INIT_FAILED, "Graph prepare init compute graph is NULLPTR");
    return GE_GRAPH_INIT_FAILED;
  }
  auto nodes = compute_graph_->GetAllNodes();
  if (nodes.empty()) {
    GELOGE(GE_GRAPH_INIT_FAILED, "Invalid graph, no nodes in this graph.");
    return GE_GRAPH_INIT_FAILED;
  }
  for (const NodePtr &node : compute_graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetOpDesc() == nullptr) {
      GELOGE(GE_GRAPH_INIT_FAILED, "Check Graph node opdesc is NULL");
      return GE_GRAPH_INIT_FAILED;
    }
  }
  return SUCCESS;
}

Status GraphPrepare::CheckRefInputNode(const NodePtr &node, const std::string &input_name,
                                       const std::unordered_set<NodePtr> &ref_nodes) {
  // Acceptable input types should be ref node, variable or Switch operator, which is issued by ME for dynamic
  // lossscale and would be optimized in SwitchOpPass. Since ME dont differentiate between RefSwitch and Switch,
  // and only issue Switch.
  static std::unordered_set<std::string> acceptable_types = {ge::VARIABLE,         ge::VARIABLEV2, ge::VARHANDLEOP,
                                                             ge::REFSWITCH,        ge::REFMERGE,   ge::REFENTER,
                                                             ge::REFNEXTITERATION, ge::REFEXIT,    ge::SWITCH};
  GE_CHECK_NOTNULL(node);
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto input_index = op_desc->GetInputIndexByName(input_name);
  const auto &in_anchor = node->GetInDataAnchor(input_index);
  GE_CHECK_NOTNULL(in_anchor);
  const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);
  const auto &input_node = peer_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(input_node);
  const auto &input_op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(input_op_desc);

  bool is_ref = (ref_nodes.find(input_node) != ref_nodes.end());
  if (is_ref) {
    return SUCCESS;
  }
  auto input_type = input_op_desc->GetType();
  if (input_type == ge::FRAMEWORKOP) {
    if (!ge::AttrUtils::GetStr(input_op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, input_type)) {
      GELOGE(PARAM_INVALID, "Get original type failed.");
      return PARAM_INVALID;
    }
  }
  bool is_acceptable = (acceptable_types.find(input_type) != acceptable_types.end());

  if (!is_acceptable) {
    GELOGE(PARAM_INVALID, "The ref input of ref node %s[%s] must be ref node or variable, but %s[%s]isn't.",
           node->GetName().c_str(), node->GetType().c_str(), input_op_desc->GetName().c_str(),
           input_op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status GraphPrepare::CheckRefOp() {
  GE_CHECK_NOTNULL(compute_graph_);
  std::unordered_set<NodePtr> ref_nodes;
  for (const NodePtr &node : compute_graph_->GetDirectNode()) {
    if (node == nullptr) {
      GELOGE(PARAM_INVALID, "param [node] must not be null.");
      return PARAM_INVALID;
    }
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGE(PARAM_INVALID, "OpDesc of param [node] must not be null.");
      return PARAM_INVALID;
    }

    auto input_names = op_desc->GetAllInputNames();
    auto outputs = op_desc->GetAllOutputName();
    std::unordered_set<std::string> all_output_name;

    for (auto &output : outputs) {
      all_output_name.insert(output.first);
    }
    for (const auto &input_name : input_names) {
      if (all_output_name.find(input_name) != all_output_name.end()) {
        if (CheckRefInputNode(node, input_name, ref_nodes) != SUCCESS) {
          GELOGE(PARAM_INVALID, "CheckRefInputNode failed.");
          return PARAM_INVALID;
        }
        (void)ref_nodes.insert(node);
      }
    }
  }
  return SUCCESS;
};

Status GraphPrepare::SetRtContext(rtContext_t rt_context, rtCtxMode_t mode) {
  GELOGI("set rt_context %d, device id:%u.", static_cast<int>(mode), ge::GetContext().DeviceId());
  GE_CHK_RT_RET(rtCtxCreate(&rt_context, mode, ge::GetContext().DeviceId()));
  GE_CHK_RT_RET(rtCtxSetCurrent(rt_context));
  RtContextUtil::GetInstance().AddrtContext(rt_context);
  return SUCCESS;
}

Status GraphPrepare::AdjustDataOpOutput(const NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "Input node is NULL");
    return GE_GRAPH_GRAPH_NODE_NULL;
  }
  OpDescPtr op_desc_ptr = node->GetOpDesc();
  if (op_desc_ptr == nullptr) {
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "Input node opdesc is NULL");
    return GE_GRAPH_GRAPH_NODE_NULL;
  }
  GeTensorDesc output = op_desc_ptr->GetOutputDesc(0);
  int64_t tensor_size = 0;
  graphStatus graph_status = TensorUtils::GetTensorMemorySizeInBytes(output, tensor_size);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(graph_status, "GetTensorMemorySizeInBytes failed!");
    return FAILED;
  }
  TensorUtils::SetSize(output, tensor_size);
  graphStatus graph_ret = op_desc_ptr->UpdateOutputDesc(0, output);
  if (graph_ret != GRAPH_SUCCESS) {
    GELOGE(graph_ret, "UpdateOutputDesc fail, graph_ret:%u", graph_ret);
    return graph_ret;
  }
  return SUCCESS;
}

Status GraphPrepare::UpdateInput(const std::vector<GeTensor> &user_input) {
  compute_graph_->SaveDataFormat(ge::TypeUtils::DomiFormatToFormat(domi::GetContext().format));
  for (NodePtr &input_node : compute_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    OpDescPtr op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    if (op->GetType() == DATA) {
      GeAttrValue::INT index = 0;
      if ((!(AttrUtils::GetInt(op, ATTR_NAME_INDEX, index))) || (domi::GetContext().is_dynamic_input)) {
        GELOGW("Get index from data attr failed");
        continue;
      }

      if ((index < 0) || (static_cast<size_t>(index) >= user_input.size())) {
        GELOGE(PARAM_INVALID, "user_input size = %zu, graph data op index = %ld.", user_input.size(), index);
        return FAILED;
      }

      GeTensorDesc desc(user_input[index].GetTensorDesc());
      auto format = desc.GetFormat();
      auto origin_format = desc.GetOriginFormat();
      bool is_internal = TypeUtils::IsInternalFormat(format) || TypeUtils::IsInternalFormat(origin_format);
      bool need_check_internal_format = (!options_.is_single_op) && is_internal;
      if (need_check_internal_format) {
        GELOGE(PARAM_INVALID, "Input format %s or origin_format %s is not support.",
               TypeUtils::FormatToSerialString(format).c_str(), TypeUtils::FormatToSerialString(origin_format).c_str());
        return FAILED;
      }

      auto data_type = desc.GetDataType();
      uint32_t length = 1;
      bool type_ret = TypeUtils::GetDataTypeLength(data_type, length);
      if (!type_ret) {
        GELOGE(PARAM_INVALID, "Input datatype %s is not support.",
               TypeUtils::DataTypeToSerialString(data_type).c_str());
        return FAILED;
      }
      int64_t desc_shape = desc.GetShape().GetShapeSize();
      FMK_INT64_UINT32_MULCHECK(desc_shape, length);
      int64_t shape_size = desc_shape * length;
      GE_IF_BOOL_EXEC(shape_size == 0 && desc.GetShape().GetDimNum() == 0, shape_size = static_cast<int64_t>(length));
      int64_t size = 0;
      GE_IF_BOOL_EXEC(ge::TensorUtils::GetSize(desc, size) != GRAPH_SUCCESS,
                      GELOGE(INTERNAL_ERROR, "TensorUtils GetSize failed");
                      return FAILED);
      if ((size != 0) && (shape_size != size)) {
        GELOGE(PARAM_INVALID, "input data size =%ld, shape_size =%ld.", size, shape_size);
        return FAILED;
      }
      ge::TensorUtils::SetSize(desc, shape_size);
      graphStatus graph_ret = op->UpdateInputDesc(0, desc);
      if (graph_ret != GRAPH_SUCCESS) {
        GELOGE(graph_ret, "UpdateInputDesc fail, graph_ret:%u", graph_ret);
        return graph_ret;
      }
      // Size will be recalculated in the build stage
      ge::TensorUtils::SetSize(desc, 0);
      graph_ret = op->UpdateOutputDesc(0, desc);
      if (graph_ret != GRAPH_SUCCESS) {
        GELOGE(graph_ret, "UpdateOutputDesc fail, graph_ret:%u", graph_ret);
        return graph_ret;
      }

      if (!options_.train_graph_flag) {
        Status ret = AdjustDataOpOutput(input_node);
        if (ret != SUCCESS) {
          GELOGE(ret, "AdjustDataOpOutput fail, ret:%u", ret);
          return ret;
        }
      }
    }
  }

  return SUCCESS;
}

Status GraphPrepare::TryDoAipp() {
  // infer and with aipp configure file, then call aipp insert
  if ((!options_.train_graph_flag) && (!options_.insert_op_file.empty())) {
    GE_DUMP(compute_graph_, "Before_insert_aipp");
    Status ret = ge::InsertNewOpUtil::Instance().Init();
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "TryDoAipp: InsertNewOpUtil instance failed.");
      return INTERNAL_ERROR;
    }
    ret = ge::InsertNewOpUtil::Instance().Parse(options_.insert_op_file.c_str());
    if (ret != SUCCESS) {
      GELOGE(GE_GRAPH_OPTIMIZE_INSERT_OP_PARSE_FAILED, "TryDoAipp: parse config file %s failed",
             options_.insert_op_file.c_str());
      return GE_GRAPH_OPTIMIZE_INSERT_OP_PARSE_FAILED;
    }
    ret = ge::InsertNewOpUtil::Instance().InsertAippOps(compute_graph_, options_.insert_op_file);
    if (ret != SUCCESS) {
      GELOGE(GE_GRAPH_OPTIMIZE_INSERT_DYN_OP_FAILED, "TryDoAipp: insert aipp op ret failed, ret:%u", ret);
      return GE_GRAPH_OPTIMIZE_INSERT_DYN_OP_FAILED;
    }
  }
  return SUCCESS;
}

Status GraphPrepare::FormatAndShapeProcess() {
  Status ret = ResourcePairProcess("add");
  if (ret != SUCCESS) {
    GELOGE(ret, "ResourcePairProcess failed");
    return ret;
  }

  GE_TIMESTAMP_START(InferOriginFormat1);
  ret = compute_graph_->InferOriginFormat();
  GE_TIMESTAMP_END(InferOriginFormat1, "GraphPrepare::InferOriginFormat1");
  GE_DUMP(compute_graph_, "after_first_inferformat");
  if (ret != SUCCESS) {
    GELOGE(ret, "Prepare Graph first inferformat failed");
    return ret;
  }

  GE_TIMESTAMP_START(InferShapeForPreprocess);
  ret = InferShapeForPreprocess();
  GE_TIMESTAMP_END(InferShapeForPreprocess, "GraphPrepare::InferShapeForPreprocess");
  GE_DUMP(compute_graph_, "after_infershape");
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_INFERSHAPE_FAILED, "Prepare Graph infershape failed");
    return GE_GRAPH_INFERSHAPE_FAILED;
  }

  GE_TIMESTAMP_START(InferOriginFormat2);
  ret = compute_graph_->InferOriginFormat();
  GE_TIMESTAMP_END(InferOriginFormat2, "GraphPrepare::InferOriginFormat2");
  if (ret != SUCCESS) {
    GELOGE(ret, "Prepare Graph inferformat failed");
    return ret;
  }

  ret = ResourcePairProcess("remove");
  if (ret != SUCCESS) {
    return ret;
  }
  return ret;
}

Status GraphPrepare::ResourcePairProcess(const std::string &action) {
  PassManager control_pass;
  // Graph pass tmp logic for resource infershape
  if (options_.train_graph_flag) {
    try {
      if (action == "add") {
        (void)control_pass.AddPass("ResourcePairProcess::ResourcePairAddControlPass", new ResourcePairAddControlPass);
      } else {
        (void)control_pass.AddPass("ResourcePairProcess::ResourcePairRemoveControlPass",
                                   new ResourcePairRemoveControlPass);
      }
    } catch (std::bad_alloc &e) {
      GELOGE(INTERNAL_ERROR, "Add pass failed, bad memory allocation occur, action:%s.", action.c_str());
      return INTERNAL_ERROR;
    }
  }
  Status ret = control_pass.Run(compute_graph_);
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "Run ResourcePairControlPass failed, action:%s, ret:%u.", action.c_str(), ret);
    return ret;
  }
  return SUCCESS;
}

Status GraphPrepare::OptimizeAfterInfershapeByAtcParams() {
  if (options_.train_graph_flag) {
    GELOGI("This is train mode, no need to do this schedule.");
    return SUCCESS;
  }
  GE_RETURN_IF_ERROR(InsertNewOpUtil::Instance().UpdateDataNodeByAipp(compute_graph_));
  for (auto &node_ptr : compute_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr);
    if (CheckIfNeedSetNdFormat(node_ptr) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Set node [%s] format ND failed", node_ptr->GetName().c_str());
      return FAILED;
    }
    if (node_ptr->GetType() == DATA) {
      if (ProcessDataNode(node_ptr) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Process data node failed");
        return FAILED;
      }
    }

    if (node_ptr->GetType() == ge::NETOUTPUT) {
      if (ProcessNetoutputNode(node_ptr, options_.output_datatype) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Process netoutput node failed");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

void GraphPrepare::ProcessCCEFormat() {
  static const char *const parser_priority = std::getenv("PARSER_PRIORITY");
  static const bool keep_cce = parser_priority != nullptr && string(parser_priority) == "cce";
  if (keep_cce) {
    GELOGI("keep cce priority");
    for (const ge::NodePtr &n : compute_graph_->GetDirectNode()) {
      auto node_op_desc = n->GetOpDesc();
      GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
      if (node_op_desc->GetType() == MULTIPLY || node_op_desc->GetType() == ASSIGN) {
        auto input_size = static_cast<uint32_t>(node_op_desc->GetInputsSize());
        for (uint32_t i = 0; i < input_size; ++i) {
          ge::GeTensorDesc org_tensor_input = node_op_desc->GetInputDesc(i);
          GELOGD("keep cce name:%s, type:%s", node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str());
          if (org_tensor_input.GetFormat() == FORMAT_ND) {
            org_tensor_input.SetFormat(FORMAT_NCHW);
            org_tensor_input.SetOriginFormat(FORMAT_NCHW);
            // [No need to check value]
            (void)node_op_desc->UpdateInputDesc(i, org_tensor_input);
          }
        }
        auto output_size = static_cast<uint32_t>(node_op_desc->GetOutputsSize());
        for (uint32_t i = 0; i < output_size; ++i) {
          ge::GeTensorDesc org_tensor_output = node_op_desc->GetOutputDesc(i);
          GELOGD("keep cce name:%s, type:%s", node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str());
          if (org_tensor_output.GetFormat() == FORMAT_ND) {
            org_tensor_output.SetFormat(FORMAT_NCHW);
            org_tensor_output.SetOriginFormat(FORMAT_NCHW);
            // [No need to check value]
            (void)node_op_desc->UpdateOutputDesc(i, org_tensor_output);
          }
        }
      }
    }
  }
}

Status GraphPrepare::OptimizeBeforeInfershape() {
  PassManager graph_passes_before_infershape;
  // Graph pass
  try {
    if (options_.train_graph_flag) {
      (void)graph_passes_before_infershape.AddPass("OptimizeBeforeInfershape::SavePass", new SavePass);
    }
    (void)graph_passes_before_infershape.AddPass("OptimizeBeforeInfershape::NetOutputPass", new NetOutputPass);
  } catch (std::bad_alloc &e) {
    GELOGE(INTERNAL_ERROR, "Add pass failed, bad memory allocation occurs.");
    return INTERNAL_ERROR;
  }
  GE_TIMESTAMP_START(graph_passes_before_infershape);
  Status ret = graph_passes_before_infershape.Run(compute_graph_);
  GE_TIMESTAMP_END(graph_passes_before_infershape, "GraphPrepare::BeforeInfershape");
  bool status = (ret != SUCCESS && ret != NOT_CHANGED);
  if (status) {
    GELOGE(ret, "Run graph_passes_before_infershape failed, ret:%u.", ret);
    return ret;
  }

  graphStatus ret_topo = compute_graph_->TopologicalSorting();
  if (ret_topo != GRAPH_SUCCESS) {
    GELOGE(ret_topo, "Graph topological sort failed, ret:%u.", ret_topo);
    return ret_topo;
  }
  return SUCCESS;
}

Status GraphPrepare::SaveOriginalGraphToOmModel() {
  if (options_.save_original_model == "true") {
    ModelHelper model_helper;
    Status ret = model_helper.SaveOriginalGraphToOmModel(ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph_),
                                                         options_.original_model_file);
    if (ret != SUCCESS) {
      // If save original model fail, process continue
      GELOGW("SaveOriginalGraphToOmModel fail");
    }
  }
  return SUCCESS;
}

Status GraphPrepare::Preprocess(const std::vector<GeTensor> &user_input) {
  // rtContext_t...
  Status ret = SetRtContext(rtContext_t(), RT_CTX_GEN_MODE);
  if (ret != SUCCESS) {
    GELOGE(ret, "Set rt context failed.");
    return ret;
  }

  ret = CheckAndUpdateInput(user_input);
  if (ret != SUCCESS) {
    GELOGE(ret, "Check user input failed.");
    return ret;
  }
  GE_DUMP(compute_graph_, "after_update_input");

  GEPass ge_passes(compute_graph_);
  NamesToPass names_to_passes;
  ForPass for_pass;
  names_to_passes.emplace_back("ForPass", &for_pass);
  GE_TIMESTAMP_START(names_to_passes);
  ret = ge_passes.Run(names_to_passes);
  GE_TIMESTAMP_END(names_to_passes, "GraphPrepare::ForPass");
  if (ret != SUCCESS) {
    GELOGE(ret, "Run ForPass optimize for preprocess failed, ret:%u.", ret);
    return ret;
  }
  GE_DUMP(compute_graph_, "after_for_pass");

  GE_TIMESTAMP_START(netoutput_process);
  ret = ProcessNetOutput();
  GE_TIMESTAMP_END(netoutput_process, "GraphPrepare::NetOutputProcess")
  if (ret != SUCCESS) {
    return ret;
  }
  GE_TIMESTAMP_START(multibatch_process);
  ret = multibatch::ProcessMultiBatch(compute_graph_);
  GE_TIMESTAMP_END(multibatch_process, "GraphPrepare::MultiBatchProcess")
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to do multi-batch processing");
    return ret;
  }
  GE_DUMP(compute_graph_, "after_multibatch_process");

  ret = TryDoAipp();
  if (ret != SUCCESS) {
    return ret;
  }

  GE_TIMESTAMP_START(FormatAndShapeProcess);
  ret = FormatAndShapeProcess();
  GE_TIMESTAMP_END(FormatAndShapeProcess, "GraphPrepare::FormatAndShapeProcess");
  if (ret != SUCCESS) {
    GELOGE(ret, "FormatAndShape process failed");
    return ret;
  }
  GE_DUMP(compute_graph_, "after_inferformat_before_preprocess");

  ProcessCCEFormat();

  ret = OptimizeAfterInfershapeByAtcParams();
  if (ret != SUCCESS) {
    GELOGE(ret, "Optimize for input if set inputfp16 failed.");
    return ret;
  }

  SaveOriginalGraphToOmModel();

  GE_TIMESTAMP_START(OptimizeForPreprocess);
  ret = OptimizeForPreprocess();
  GE_TIMESTAMP_END(OptimizeForPreprocess, "GraphPrepare::OptimizeForPreprocess");
  if (ret != SUCCESS) {
    GELOGE(ret, "Optimize for preprocess failed.");
    return ret;
  }
  GELOGI("Optimize for preprocess success.");

  GE_TIMESTAMP_START(UpdateVariableFormats);
  ret = UpdateVariableFormats(compute_graph_);
  GE_TIMESTAMP_END(UpdateVariableFormats, "GraphPrepare::UpdateVariableFormats");
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to update variables formats");
    return ret;
  }
  GELOGI("Update variable formats success.");

  GE_DUMP(compute_graph_, "Optimize_after_preprocess");
  return SUCCESS;
}

#define PP_RUN_AND_DUMP(name, func, ...)                                               \
  do {                                                                                 \
    GE_RUN(Prepare, func, __VA_ARGS__);                                                \
    GE_DUMP(compute_graph, "PrepareAfter" name);                                       \
    GELOGI("Prepare %s on graph %s success.", name, compute_graph->GetName().c_str()); \
  } while (0)

#define PP_RUN(name, func, ...)                                                        \
  do {                                                                                 \
    GE_RUN(Prepare, func, __VA_ARGS__);                                                \
    GELOGI("Prepare %s on graph %s success.", name, compute_graph->GetName().c_str()); \
  } while (0)

Status GraphPrepare::PrepareDynShape(ConstGraphPtr graph, const std::vector<GeTensor> &user_input,
                                     ge::ComputeGraphPtr &compute_graph, uint64_t session_id) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(compute_graph);
  if (options_.train_graph_flag) {
    domi::GetContext().train_flag = true;
  }
  domi::GetContext().type = static_cast<domi::FrameworkType>(options_.framework_type);
  const Graph &const_graph = *graph;

  PP_RUN("Init", Init, const_graph, session_id);
  PP_RUN("SetRtContext", SetRtContext, rtContext_t(), RT_CTX_GEN_MODE);
  PP_RUN_AND_DUMP("CheckAndUpdateInput", CheckAndUpdateInput, user_input);
  PP_RUN_AND_DUMP("GraphEquivalentTransformation", GraphEquivalentTransformation);
  PP_RUN_AND_DUMP("ProcessOutput", ProcessNetOutput);
  PP_RUN_AND_DUMP("ProcessMultiBatch", multibatch::ProcessMultiBatch, compute_graph_);
  PP_RUN_AND_DUMP("InsertAipp", TryDoAipp);
  PP_RUN_AND_DUMP("ProcessBeforeInfershape", ProcessBeforeInfershape);
  PP_RUN_AND_DUMP("InferFormatAndShape", FormatAndShapeProcess);
  PP_RUN_AND_DUMP("GetDynamicOutputShape", multibatch::GetDynamicOutputShape, compute_graph_);
  PP_RUN_AND_DUMP("ProcessAippStage2", InsertNewOpUtil::Instance().UpdateDataNodeByAipp, compute_graph_);
  // todo: return when save mode
  PP_RUN("SaveOriginalGraphToOmModel", SaveOriginalGraphToOmModel);
  PP_RUN_AND_DUMP("PrepareOptimize", PrepareOptimize);

  return SUCCESS;
}

Status GraphPrepare::RecordAIPPInfo(ge::ComputeGraphPtr &compute_graph) {
  PP_RUN("RecordAIPPInfo", InsertNewOpUtil::Instance().RecordAIPPInfoToData, compute_graph_);
  return SUCCESS;
}

Status GraphPrepare::PrepareRunningFormatRefiner() {
  auto compute_graph = compute_graph_;
  PassManager pass_manager;
  GE_CHK_STATUS_RET(pass_manager.AddPass("PrepareRunningFormatRefiner::VariablePrepareOpPass",
                                         new (std::nothrow) VariablePrepareOpPass))
  GE_CHK_STATUS_RET(pass_manager.AddPass("PrepareRunningFormatRefiner::SubgraphPass", new (std::nothrow) SubgraphPass))
  GE_TIMESTAMP_START(pass_manager);
  auto ret = pass_manager.Run(compute_graph);
  GE_TIMESTAMP_END(pass_manager, "GraphPrepare::PrepareRunningFormatRefiner");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "Run passes for running format refiner failed, ret:%u.", ret);
    return ret;
  }
  PP_RUN_AND_DUMP("UpdateInputOutputByUserOptions", UpdateInputOutputByOptions);
  PP_RUN_AND_DUMP("UpdateVariableFormats", UpdateVariableFormats, compute_graph_);
  return SUCCESS;
}

Status GraphPrepare::SwitchOpOptimize(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    GELOGE(GE_GRAPH_NULL_INPUT, "Input Graph is NULL");
    return GE_GRAPH_NULL_INPUT;
  }
  GEPass ge_passes(compute_graph);
  NamesToPass hccl_group;
  HcclGroupPass hccl_group_pass;
  GELOGD("Add hccl group pass success");
  hccl_group.emplace_back("HcclGroupPass", &hccl_group_pass);
  auto ret = ge_passes.Run(hccl_group);
  if (ret != SUCCESS) {
    GELOGE(ret, "Run HcclGroupPass pass for preprocess failed, ret:%u.", ret);
    return ret;
  }
  ret = compute_graph->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph topological sort failed, ret:%u.", ret);
    return ret;
  }
  return SUCCESS;
}
#undef PP_RUN_AND_DUMP
#undef PP_RUN

Status GraphPrepare::GenerateInfershapeGraph(ConstGraphPtr graph) {
  if (graph == nullptr) {
    GELOGE(GE_GRAPH_NULL_INPUT, "Input Graph is NULL");
    return GE_GRAPH_NULL_INPUT;
  }
  const Graph &const_graph = *graph;
  Status ret = Init(const_graph, 0);
  if (ret != SUCCESS) {
    GELOGE(ret, "Init graph_prepare fail, ret:%u", ret);
    return ret;
  }
  GELOGI("Start infershape for dump json.");
  GEPass ge_passes(compute_graph_);
  NamesToPass names_to_passes;
  InferShapePass infer_shape_pass;
  names_to_passes.emplace_back("InferShapePass", &infer_shape_pass);
  ret = ge_passes.Run(names_to_passes);
  if (ret != SUCCESS) {
    GELOGE(ret, "Run ge_passes infershape for preprocess failed, ret:%u.", ret);
    return ret;
  }
  return SUCCESS;
}

Status GraphPrepare::Prepare(ConstGraphPtr graph, const std::vector<GeTensor> &user_input,
                             ge::ComputeGraphPtr &compute_graph, VarAccelerateCtrl &var_acc_ctrl, uint64_t session_id) {
  // train graph flag
  if (options_.train_graph_flag) {
    domi::GetContext().train_flag = true;
  }
  domi::GetContext().type = static_cast<domi::FrameworkType>(options_.framework_type);

  if (graph == nullptr) {
    GELOGE(GE_GRAPH_NULL_INPUT, "Input Graph is NULL");
    return GE_GRAPH_NULL_INPUT;
  }
  const Graph &const_graph = *graph;
  Status ret = Init(const_graph, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "Init graph_prepare fail, ret:%u", ret);
    return ret;
  }

  GraphOptimize graph_optimize;
  if (!domi::GetContext().train_flag) {
    GE_DUMP(compute_graph_, "BeforeOriginalGraphForQuantize");
    GE_TIMESTAMP_START(OptimizeOriginalGraphForQuantize);
    ret = graph_optimize.OptimizeOriginalGraphForQuantize(compute_graph_);
    GE_TIMESTAMP_END(OptimizeOriginalGraphForQuantize, "GraphPrepare::OptimizeOriginalGraphForQuantize");
    if (ret != SUCCESS) {
      GELOGE(ret, "originalGraph optimize for Quantize Failed");
      return ret;
    }
  }
  GE_DUMP(compute_graph_, "BeforePreprocess");

  GE_TIMESTAMP_START(Preprocess);
  ret = Preprocess(user_input);
  GE_TIMESTAMP_END(Preprocess, "GraphPrepare::Preprocess");
  if (ret != SUCCESS) {
    GELOGE(ret, "Run graph_prepare fail, ret:%u", ret);
    return ret;
  }
  // OriginalGraph optimize
  ret = graph_optimize.SetOptions(options_);
  GE_CHK_STATUS_RET(ret, "Graph optimize initial fail");
  if (options_.local_fmk_op_flag) {
    graph_optimize.TranFrameOp(compute_graph_);
  }
  GE_DUMP(compute_graph_, "Prepare");

  GE_TIMESTAMP_START(OptimizeOriginalGraph);
  const char *buffer_optimize_on = std::getenv("BUFFER_OPTIMIZE_ON");
  if (buffer_optimize_on != nullptr) {
    ret = graph_optimize.NewOptimizeOriginalGraph(compute_graph_);
  } else {
    ret = graph_optimize.OptimizeOriginalGraph(compute_graph_);
  }
  GE_TIMESTAMP_END(OptimizeOriginalGraph, "GraphPrepare::OptimizeOriginalGraph");
  GE_DUMP(compute_graph_, "PreProcessOptimizeOriginalGraphAfter");
  if (ret != SUCCESS) {
    GELOGE(ret, "originalGraph optimize Failed");
    return ret;
  }

  GE_RETURN_IF_ERROR(RecordAIPPInfo(compute_graph_));

  GE_TIMESTAMP_START(OptimizeBeforeSubGraph);

  if (buffer_optimize_on != nullptr) {
    ret = NewOptimizeGraphBeforeSubGraph(var_acc_ctrl);
  } else {
    ret = OptimizeGraphBeforeSubGraph();
  }
  GE_TIMESTAMP_END(OptimizeBeforeSubGraph, "GraphPrepare::OptimizeBeforeSubGraph");
  if (ret != SUCCESS) {
    GELOGE(ret, "originalGraph optimize Failed");
    return ret;
  }
  compute_graph = compute_graph_;
  return SUCCESS;
}

Status GraphPrepare::CheckConstOp() {
  for (auto &node_ptr : compute_graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node_ptr);
    if (node_ptr->GetType() == CONSTANT) {
      Status ret = VerifyConstOp(node_ptr);
      GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "Const Op Check failed");
    } else if (node_ptr->GetType() == FRAMEWORKOP) {
      auto op_desc = node_ptr->GetOpDesc();
      if (op_desc == nullptr) {
        GELOGE(PARAM_INVALID, "Get op desc failed");
        return PARAM_INVALID;
      }
      std::string original_type;
      GE_IF_BOOL_EXEC(ge::AttrUtils::GetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type),
                      GELOGI("Get FrameWorkOp original type [%s]", original_type.c_str()));
      GELOGI("original type is %s", original_type.c_str());
      if (original_type == CONSTANT) {
        Status ret = VerifyConstOp(node_ptr);
        GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "Const Op Check failed");
      }
    }
  }
  return SUCCESS;
}
Status GraphPrepare::VerifyConstOp(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  ConstGeTensorPtr ge_tensor_ptr;
  if (!(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor_ptr))) {
    GELOGE(PARAM_INVALID, "Get value from const attr failed");
    return PARAM_INVALID;
  }
  GE_CHECK_NOTNULL(ge_tensor_ptr);
  auto data_size = ge_tensor_ptr->GetData().GetSize();
  auto ge_tensor_desc = ge_tensor_ptr->GetTensorDesc();
  int64_t shape_size = ge_tensor_desc.GetShape().GetShapeSize();
  auto data_type = ge_tensor_desc.GetDataType();
  uint32_t length = 1;
  bool type_ret = TypeUtils::GetDataTypeLength(data_type, length);
  if (!type_ret) {
    GELOGE(PARAM_INVALID, "Input datatype %s is not support.", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return FAILED;
  }
  FMK_INT64_UINT32_MULCHECK(shape_size, length);
  GELOGI("Const real value Size:%zu, op_desc Shape Size:%ld, data_type:%s.", data_size, shape_size * length,
         TypeUtils::DataTypeToSerialString(data_type).c_str());
  if (shape_size == 0) {
    if (ge_tensor_desc.GetShape().GetDims().size() == 0) {
      // shape = [], means it's a sclar tensor.
      GE_CHK_BOOL_EXEC(data_size / length == 1, return PARAM_INVALID, "Const is invalid scalar tensor.");
    } else {
      // shape = [x, y, 0,...], means it's a vector tensor that value is [].
      GE_CHK_BOOL_EXEC(data_size == 0, return PARAM_INVALID, "Const is invalid vector scalar.");
    }
  } else {
    GE_CHK_BOOL_EXEC(data_size == static_cast<size_t>(shape_size * length) && data_size != 0, return PARAM_INVALID,
                     "Const input data size is not equal with tensor desc shape");
  }
  return SUCCESS;
}

Status GraphPrepare::CheckUserInput(const std::vector<GeTensor> &user_input) {
  if (domi::GetContext().is_dynamic_input) {
    return SUCCESS;
  }
  unsigned int node_num = 0;
  unsigned int data_num = 0;
  for (NodePtr &input_node : compute_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    OpDescPtr op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    node_num++;
    if (op->GetType() == DATA || op->GetType() == AIPPDATA) {
      data_num++;
      GeAttrValue::INT index = 0;
      if (!(AttrUtils::GetInt(op, ATTR_NAME_INDEX, index))) {
        GELOGE(GE_GRAPH_INIT_FAILED, "Get index from attr failed");
        return GE_GRAPH_INIT_FAILED;
      }
      if ((index < 0) || (static_cast<size_t>(index) >= user_input.size())) {
        GELOGE(GE_GRAPH_INIT_FAILED, "user_input size:%zu, data op index:%ld.", user_input.size(), index);
        return GE_GRAPH_INIT_FAILED;
      }
      GeTensorDesc desc(user_input[index].GetTensorDesc());

      for (size_t i = 0; i < desc.GetShape().GetDimNum(); ++i) {
        if (desc.GetShape().GetDim(i) < 0) {
          GELOGE(GE_GRAPH_INIT_FAILED, "data dim %zu is not supported, need >= 0, real:%ld.", i,
                 desc.GetShape().GetDim(i));
          return GE_GRAPH_INIT_FAILED;
        }
      }
    }
  }
  if (node_num <= data_num) {
    GELOGW("Prepare check user input, data_num = %u, node_num = %u", data_num, node_num);
  }
  return SUCCESS;
}

Status GraphPrepare::InferShapeForPreprocess() {
  GELOGI("Start infershape for preprocess.");
  GEPass ge_passes(compute_graph_);
  NamesToPass names_to_passes;
  AssertPass assert_pass;
  if (!options_.train_graph_flag) {
    names_to_passes.emplace_back("AssertPass", &assert_pass);
  }
  InferShapePass infer_shape_pass;
  names_to_passes.emplace_back("InferShapePass", &infer_shape_pass);
  ReplaceWithEmptyConstPass replace_with_empty_const_pass;
  names_to_passes.emplace_back("ReplaceWithEmptyConstPass", &replace_with_empty_const_pass);
  DimensionComputePass dimension_compute_pass;
  names_to_passes.emplace_back("DimensionComputePass", &dimension_compute_pass);
  ConstantFoldingPass constant_folding_pass;
  names_to_passes.emplace_back("ConstantFoldingPass", &constant_folding_pass);

  int32_t dev_count = 0;
  AicpuConstantFoldingPass aicpu_constant_folding_pass;
  const char *aicpu_constant_folding_on = std::getenv("AICPU_CONSTANT_FOLDING_ON");
  rtError_t rt_err = RT_ERROR_NONE;
  if (aicpu_constant_folding_on != nullptr) {
    rt_err = rtGetDeviceCount(&dev_count);
    if (rt_err == RT_ERROR_NONE) {
      Status result = SetRtContext(rtContext_t(), RT_CTX_NORMAL_MODE);
      if (result != SUCCESS) {
        GELOGE(result, "Set rt context failed.");
        return result;
      }
      names_to_passes.emplace_back("AicpuConstantFoldingPass", &aicpu_constant_folding_pass);
    }
  }
  Status ret = ge_passes.Run(names_to_passes);
  if (aicpu_constant_folding_on != nullptr) {
    if (rt_err == RT_ERROR_NONE) {
      Status result = SetRtContext(rtContext_t(), RT_CTX_GEN_MODE);
      if (result != SUCCESS) {
        GELOGE(result, "Set rt context failed.");
        return result;
      }
    }
  }
  if (ret != SUCCESS) {
    GELOGE(ret, "Run ge_passes infershape for preprocess failed, ret:%u.", ret);
    return ret;
  }
  return SUCCESS;
}
Status GraphPrepare::PrepareOptimize() {
  GELOGI("Start optimize for preprocess.");
  PassManager original_graph_passes;
  // Graph pass
  try {
    (void)original_graph_passes.AddPass("PrepareOptimize::ShapeOperateOpRemovePass", new ShapeOperateOpRemovePass);
    (void)original_graph_passes.AddPass("PrepareOptimize::ReplaceTransShapePass", new ReplaceTransShapePass);
  } catch (std::bad_alloc &e) {
    GELOGE(INTERNAL_ERROR, "Add pass failed, bad memory allocation occurs.");
    return INTERNAL_ERROR;
  }

  GE_TIMESTAMP_START(original_graph_passes);
  Status ret = original_graph_passes.Run(compute_graph_);
  GE_TIMESTAMP_END(original_graph_passes, "GraphPrepare::OriginalGraphPasses");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "Run graph passes optimize for preprocess failed, ret:%u.", ret);
    return ret;
  }
  // New pass
  GEPass ge_passes(compute_graph_);
  NamesToPass names_to_passes;
  EnterPass enter_pass;
  PrintOpPass print_pass;
  names_to_passes.emplace_back("EnterPass", &enter_pass);
  CondPass cond_pass;
  names_to_passes.emplace_back("CondPass", &cond_pass);
  if (options_.enable_print_op_pass) {
    names_to_passes.emplace_back("PrintOpPass", &print_pass);
  }
  NoUseReshapeRemovePass no_use_reshape_remove_pass;
  names_to_passes.emplace_back("NoUseReshapeRemovePass", &no_use_reshape_remove_pass);

  DropOutPass dropout_pass;
  AssertPass assert_pass;
  UnusedConstPass unused_const_pass;
  StopGradientPass stop_gradient_pass;
  PreventGradientPass prevent_gradient_pass;
  PlaceholderWithDefaultPass placeholder_with_default_pass;
  GuaranteeConstPass guarantee_const_pass;
  VarIsInitializedOpPass var_is_initialized_pass;
  ParallelConcatStartOpPass parallel_concat_start_op_pass;
  IdentityPass identity_pass(false);
  SnapshotPass snapshot_pass;
  if (!options_.train_graph_flag) {
    names_to_passes.emplace_back("DropOutPass", &dropout_pass);
    names_to_passes.emplace_back("AssertPass", &assert_pass);
  }
  names_to_passes.emplace_back("UnusedConstPass", &unused_const_pass);
  names_to_passes.emplace_back("StopGradientPass", &stop_gradient_pass);
  names_to_passes.emplace_back("PreventGradientPass", &prevent_gradient_pass);
  names_to_passes.emplace_back("PlaceholderWithDefaultPass", &placeholder_with_default_pass);
  names_to_passes.emplace_back("SnapshotPass", &snapshot_pass);
  names_to_passes.emplace_back("GuaranteeConstPass", &guarantee_const_pass);
  names_to_passes.emplace_back("VarIsInitializedOpPass", &var_is_initialized_pass);
  names_to_passes.emplace_back("ParallelConcatStartOpPass", &parallel_concat_start_op_pass);
  names_to_passes.emplace_back("IdentityPass", &identity_pass);
  GE_TIMESTAMP_START(names_to_passes);
  ret = ge_passes.Run(names_to_passes);
  GE_TIMESTAMP_END(names_to_passes, "GraphPrepare::NamesToPasses");
  if (ret != SUCCESS) {
    GELOGE(ret, "Run ge_passes optimize for preprocess failed, ret:%u.", ret);
    return ret;
  }

  PassManager graph_pass;
  try {
    (void)graph_pass.AddPass("PrepareOptimize::PrunePass", new PrunePass);
    // todo 临时把hccl的memcpy插入放到图准备，为了防止其多插memcpy
    (void)graph_pass.AddPass("PrepareOptimize::HcclMemcpyPass", new (std::nothrow) HcclMemcpyPass);
  } catch (std::bad_alloc &e) {
    GELOGE(INTERNAL_ERROR, "Add pass failed, bad memory allocation occurs.");
    return INTERNAL_ERROR;
  }

  GE_TIMESTAMP_START(graph_passes);
  ret = graph_pass.Run(compute_graph_);
  GE_TIMESTAMP_END(graph_passes, "GraphPrepare::GraphPasses");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "Run graph passes optimize for preprocess failed, ret:%u.", ret);
    return ret;
  }
  // The constant for train is CONSTANTOP, and is CONSTANT for inference. They will be unified in future.
  TypeConversionOfConstant();

  ret = compute_graph_->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph topological sort failed, ret:%u.", ret);
    return ret;
  }

  GELOGI("End optimize for preprocess.");

  return SUCCESS;
}

void GraphPrepare::TypeConversionOfConstant() {
  if (options_.train_graph_flag) {
    GELOGD("trans CONSTANT to CONSTANTOP in train.");
    for (ge::NodePtr &n : compute_graph_->GetAllNodes()) {
      // This can ensure that n is not a null pointer
      if (n->GetOpDesc()->GetType() == CONSTANT) {
        n->GetOpDesc()->SetType(CONSTANTOP);
      }
    }
  } else {
    GELOGD("trans CONSTANTOP to CONSTANT in inferrence.");
    for (ge::NodePtr &n : compute_graph_->GetAllNodes()) {
      // This can ensure that n is not a null pointer
      if (n->GetOpDesc()->GetType() == CONSTANTOP) {
        n->GetOpDesc()->SetType(CONSTANT);
      }
    }
  }
}

Status GraphPrepare::OptimizeForPreprocess() {
  GELOGI("Start optimize for preprocess.");
  PassManager original_graph_passes;
  // Graph pass
  try {
    (void)original_graph_passes.AddPass("OptimizeForPreprocess::ConstantFuseSamePass", new ConstantFuseSamePass);
    (void)original_graph_passes.AddPass("OptimizeForPreprocess::VariablePrepareOpPass", new VariablePrepareOpPass);
    (void)original_graph_passes.AddPass("OptimizeForPreprocess::IteratorOpPass", new IteratorOpPass);
    (void)original_graph_passes.AddPass("OptimizeForPreprocess::ShapeOperateOpRemovePass",
                                        new ShapeOperateOpRemovePass);
    (void)original_graph_passes.AddPass("OptimizeForPreprocess::ReplaceTransShapePass", new ReplaceTransShapePass);
  } catch (std::bad_alloc &e) {
    GELOGE(INTERNAL_ERROR, "Add pass failed, bad memory allocation occurs.");
    return INTERNAL_ERROR;
  }

  GE_TIMESTAMP_START(original_graph_passes);
  Status ret = original_graph_passes.Run(compute_graph_);
  GE_TIMESTAMP_END(original_graph_passes, "GraphPrepare::OriginalGraphPasses");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "Run graph passes optimize for preprocess failed, ret:%u.", ret);
    return ret;
  }
  // New pass
  GEPass ge_passes(compute_graph_);
  NamesToPass names_to_passes;
  EnterPass enter_pass;
  names_to_passes.emplace_back("EnterPass", &enter_pass);
  CondPass cond_pass;
  names_to_passes.emplace_back("CondPass", &cond_pass);
  AddNPass addn_pass;
  names_to_passes.emplace_back("AddNPass", &addn_pass);
  PrintOpPass print_pass;
  if (options_.enable_print_op_pass) {
    names_to_passes.emplace_back("PrintOpPass", &print_pass);
  }
  NoUseReshapeRemovePass no_use_reshape_remove_pass;
  names_to_passes.emplace_back("NoUseReshapeRemovePass", &no_use_reshape_remove_pass);

  // for infer
  DropOutPass dropout_pass;
  AssertPass assert_pass;
  if (!options_.train_graph_flag) {
    names_to_passes.emplace_back("DropOutPass", &dropout_pass);
    names_to_passes.emplace_back("AssertPass", &assert_pass);
  }
  UnusedConstPass unused_const_pass;
  names_to_passes.emplace_back("UnusedConstPass", &unused_const_pass);
  StopGradientPass stop_gradient_pass;
  names_to_passes.emplace_back("StopGradientPass", &stop_gradient_pass);
  PreventGradientPass prevent_gradient_pass;
  names_to_passes.emplace_back("PreventGradientPass", &prevent_gradient_pass);
  PlaceholderWithDefaultPass placeholder_with_default_pass;
  names_to_passes.emplace_back("PlaceholderWithDefaultPass", &placeholder_with_default_pass);
  SnapshotPass snapshot_pass;
  names_to_passes.emplace_back("SnapshotPass", &snapshot_pass);
  GuaranteeConstPass guarantee_const_pass;
  names_to_passes.emplace_back("GuaranteeConstPass", &guarantee_const_pass);
  VarIsInitializedOpPass var_is_initialized_pass;
  names_to_passes.emplace_back("VarIsInitializedOpPass", &var_is_initialized_pass);
  ParallelConcatStartOpPass parallel_concat_start_op_pass;
  names_to_passes.emplace_back("ParallelConcatStartOpPass", &parallel_concat_start_op_pass);
  IdentityPass identity_pass(false);
  names_to_passes.emplace_back("IdentityPass", &identity_pass);
  SwitchDeadBranchElimination switch_dead_branch_elimination;
  names_to_passes.emplace_back("SwitchDeadBranchElimination", &switch_dead_branch_elimination);
  SwitchLogicRemovePass switch_logic_remove_pass;
  names_to_passes.emplace_back("SwitchLogicRemovePass", &switch_logic_remove_pass);
  MergePass merge_pass;
  names_to_passes.emplace_back("MergePass", &merge_pass);
  GE_TIMESTAMP_START(names_to_passes);
  ret = ge_passes.Run(names_to_passes);
  GE_TIMESTAMP_END(names_to_passes, "GraphPrepare::NamesToPasses");
  if (ret != SUCCESS) {
    GELOGE(ret, "Run ge_passes optimize for preprocess failed, ret:%u.", ret);
    return ret;
  }

  PassManager graph_pass;
  try {
    (void)graph_pass.AddPass("OptimizeForPreprocess::PrunePass", new PrunePass);
    (void)graph_pass.AddPass("OptimizeForPreprocess::NextIterationPass", new NextIterationPass);
    (void)graph_pass.AddPass("OptimizeForPreprocess::ControlTriggerPass", new ControlTriggerPass);
    (void)graph_pass.AddPass("OptimizeForPreprocess::SwitchOpPass", new SwitchOpPass);
    (void)graph_pass.AddPass("OptimizeForPreprocess::HcclMemcpyPass", new HcclMemcpyPass);
    GE_IF_BOOL_EXEC(options_.train_graph_flag,
                    (void)graph_pass.AddPass("OptimizeForPreprocess::FlowCtrlPass", new FlowCtrlPass););
  } catch (std::bad_alloc &e) {
    GELOGE(INTERNAL_ERROR, "Add pass failed, bad memory allocation occurs.");
    return INTERNAL_ERROR;
  }

  ret = graph_pass.Run(compute_graph_);
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "Run graph passes optimize for preprocess failed, ret:%u.", ret);
    return ret;
  }

  ret = compute_graph_->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph topological sort failed, ret:%u.", ret);
    return ret;
  }

  GELOGI("End optimize for preprocess.");

  return SUCCESS;
}

Status GraphPrepare::GraphEquivalentTransformation() {
  NamesToPass names_to_pass;
  ForPass for_pass;
  names_to_pass.emplace_back("ForToWhilePass", &for_pass);
  return GEPass(compute_graph_).Run(names_to_pass);
}

Status GraphPrepare::ProcessBeforeInfershape() {
  NamesToPass names_to_passes;
  CondRemovePass condition_remove_pass;
  names_to_passes.emplace_back("CondRemovePass", &condition_remove_pass);
  GE_TIMESTAMP_START(ProcessCondRemove);
  auto ret = GEPass(compute_graph_).Run(names_to_passes);
  GE_TIMESTAMP_END(ProcessCondRemove, "GraphManager::ProcessCondRemove");
  if (ret != SUCCESS) {
    GELOGE(ret, "Run ge_passes optimize for OptimizeAfterMergeSubGraph failed, ret:%d.", ret);
    return ret;
  }
  return SUCCESS;
}

Status GraphPrepare::ProcessNetOutput() {
  PassManager graph_passes_before_infershape;
  try {
    if (options_.train_graph_flag) {
      graph_passes_before_infershape.AddPass("ProcessNetOutput::SavePass", new (std::nothrow) SavePass);
    }
    graph_passes_before_infershape.AddPass("ProcessNetOutput::NetOutputPass", new (std::nothrow) NetOutputPass);
    graph_passes_before_infershape.AddPass("ProcessNetOutput::DataPass",
                                           new (std::nothrow) DataPass);  // Add NetOutput first.
  } catch (std::bad_alloc) {
    GELOGE(INTERNAL_ERROR, "Add pass failed, bad memory allocation occurs.");
    return INTERNAL_ERROR;
  }

  auto ret = graph_passes_before_infershape.Run(compute_graph_);
  if ((ret != SUCCESS) && (ret != NOT_CHANGED)) {
    GELOGE(ret, "Run graph_passes_before_infershape failed, ret:%d.", ret);
    return ret;
  }
  return SUCCESS;
}

Status GraphPrepare::NewOptimizeGraphBeforeSubGraph(VarAccelerateCtrl &var_acc_ctrl) {
  GELOGD("NewOptimizeGraphBeforeSubGraph in");
  PassManager passes;
  (void)passes.AddPass("NewOptimizeGraphBeforeSubGraph::CommonSubexpressionEliminationPass",
                       new (std::nothrow) CommonSubexpressionEliminationPass);
  auto ret = passes.Run(compute_graph_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to optimize for graph");
    return ret;
  }

  GEPass ge_passes_for_shape(compute_graph_);
  NamesToPass names_to_passes_for_shape;
  IdentifyReferencePass identify_reference_pass;
  names_to_passes_for_shape.emplace_back("IdentifyReferencePass", &identify_reference_pass);
  CastRemovePass cast_remove_pass;
  names_to_passes_for_shape.emplace_back("CastRemovePass", &cast_remove_pass);
  TransposeTransDataPass transpose_transdata_pass;
  names_to_passes_for_shape.emplace_back("TransposeTransDataPass", &transpose_transdata_pass);
  GE_TIMESTAMP_START(ge_passes_for_shape);
  ret = ge_passes_for_shape.Run(names_to_passes_for_shape);
  GE_TIMESTAMP_END(ge_passes_for_shape, "GraphManager::GePassesForShape");
  if (ret != SUCCESS) {
    GELOGE(ret, "Run ge_passes_for_shape optimize for OptimizeGraphBeforeSubGraph failed, ret:%d.", ret);
    return ret;
  }

  string options = "default";
  if (GetContext().GetOption("ge.exec.variable_acc", options) != SUCCESS) {
    GELOGI("get ge.exec.variable_acc failed. set default value.");
  }
  PassManager pass_manager;
  GE_CHK_STATUS_RET(pass_manager.AddPass("NewOptimizeGraphBeforeSubGraph::PermutePass", new (std::nothrow) PermutePass))
  GE_CHK_STATUS_RET(pass_manager.AddPass("NewOptimizeGraphBeforeSubGraph::VariablePrepareOpPass",
                                         new (std::nothrow) VariablePrepareOpPass))
  GE_IF_BOOL_EXEC(options == "default" || options == "1", GELOGI("turn on variable accelerator");
                  GE_CHK_STATUS_RET(pass_manager.AddPass("NewOptimizeGraphBeforeSubGraph::VariableOpPass",
                                                         new (std::nothrow) VariableOpPass(&var_acc_ctrl))))
  GE_CHK_STATUS_RET(pass_manager.AddPass("NewOptimizeGraphBeforeSubGraph::TransOpWithoutReshapeFusionPass",
                                         new (std::nothrow) TransOpWithoutReshapeFusionPass))
  GE_CHK_STATUS_RET(pass_manager.AddPass("NewOptimizeGraphBeforeSubGraph::TransOpDepthFusionPass",
                                         new (std::nothrow) TransOpDepthFusionPass))
  GE_CHK_STATUS_RET(pass_manager.AddPass("NewOptimizeGraphBeforeSubGraph::TransOpBreadthFusionPass",
                                         new (std::nothrow) TransOpBreadthFusionPass))
  GE_CHK_STATUS_RET(pass_manager.AddPass("NewOptimizeGraphBeforeSubGraph::VariableRefDeleteOpPass",
                                         new (std::nothrow) VariableRefDeleteOpPass))
  GE_CHK_STATUS_RET(pass_manager.AddPass("NewOptimizeGraphBeforeSubGraph::SameTransdataBreadthFusionPass",
                                         new (std::nothrow) SameTransdataBreadthFusionPass))
  GE_CHK_STATUS_RET(pass_manager.AddPass("NewOptimizeGraphBeforeSubGraph::LinkGenMaskNodesPass",
                                         new (std::nothrow) LinkGenMaskNodesPass(options_.stream_max_parallel_num)))

  GE_TIMESTAMP_START(pass_manager);
  ret = pass_manager.Run(compute_graph_);
  GE_TIMESTAMP_END(pass_manager, "GraphManager::BeforeSubGraph");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "Run passes after merge sub graph failed, ret:%d.", ret);
    return ret;
  }

  // add variable attr for hccl broadcast,need to be removed after variable pass online
  for (const ge::NodePtr &node : compute_graph_->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() != VARIABLE) {
      continue;
    }
    if (IsBroadCastOpData(node)) {
      AdjustBroadCastOpData(node);
    }
    if (IsAssignOpData(node)) {
      AdjustAssignOpData(node);
    }
  }

  NamesToPass names_to_passes;
  TransOpNearbyAllreduceFusionPass trans_op_nearby_allreduce_fusion_pass;
  names_to_passes.emplace_back("TransOpNearbyAllreduceFusionPass", &trans_op_nearby_allreduce_fusion_pass);
  ReshapeRemovePass reshape_remove_pass;
  names_to_passes.emplace_back("ReshapeRemovePass", &reshape_remove_pass);
  ConstantFoldingPass constant_folding_pass;
  names_to_passes.emplace_back("ConstantFoldingPass", &constant_folding_pass);
  DimensionAdjustPass dimension_adjust_pass;
  names_to_passes.emplace_back("DimensionAdjustPass", &dimension_adjust_pass);
  GEPass ge_passes(compute_graph_);
  ret = ge_passes.Run(names_to_passes);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to optimize for graph");
    return ret;
  }
  return SUCCESS;
}

Status GraphPrepare::OptimizeGraphBeforeSubGraph() {
  PassManager passes;
  (void)passes.AddPass("OptimizeGraphBeforeSubGraph::VariablePrepareOpPass", new (std::nothrow) VariablePrepareOpPass);
  (void)passes.AddPass("OptimizeGraphBeforeSubGraph::CommonSubexpressionEliminationPass",
                       new (std::nothrow) CommonSubexpressionEliminationPass);
  auto ret = passes.Run(compute_graph_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to optimize for graph");
    return ret;
  }
  ConstantFoldingPass constant_folding_pass;
  DimensionComputePass dimension_compute_pass;
  NamesToPass names_to_passes;
  names_to_passes.emplace_back("DimensionComputePass", &dimension_compute_pass);
  names_to_passes.emplace_back("ConstantFoldingPass", &constant_folding_pass);
  GEPass ge_passes(compute_graph_);
  ret = ge_passes.Run(names_to_passes);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to optimize for graph");
    return ret;
  }
  return SUCCESS;
}
Status GraphPrepare::CheckAndUpdateInput(const std::vector<GeTensor> &user_input) {
  compute_graph_->SetInputSize(user_input.size());
  if (user_input.empty()) {
    return SUCCESS;
  }

  auto ret = CheckUserInput(user_input);
  if (ret != SUCCESS) {
    GELOGE(ret, "Check user input failed.");
    return ret;
  }

  ret = UpdateInput(user_input);
  if (ret != SUCCESS) {
    GELOGE(ret, "UpdateInput fail, ret:%u", ret);
    return ret;
  }
  if (user_input.size() != 0) {
    ret = CheckConstOp();
    if (ret != SUCCESS) {
      GELOGE(ret, "CheckConstOp fail, ret:%u", ret);
      return ret;
    }
  } else {
    ret = compute_graph_->TopologicalSorting();
    if (ret != SUCCESS) {
      GELOGE(ret, "graph prepare error: compute_graph_->Topological Sorting");
      return FAILED;
    }
  }
  return SUCCESS;
}
Status GraphPrepare::UpdateInputOutputByOptions() {
  if (options_.train_graph_flag) {
    GELOGI("This is train mode, no need to do this schedule.");
    return SUCCESS;
  }
  for (auto &node_ptr : compute_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr);
    if (CheckIfNeedSetNdFormat(node_ptr) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Set node [%s] format ND failed", node_ptr->GetName().c_str());
      return FAILED;
    }

    if (node_ptr->GetType() == DATA) {
      if (ProcessDataNodeDynShape(node_ptr) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Process data node failed");
        return FAILED;
      }
    }

    if (node_ptr->GetType() == ge::NETOUTPUT) {
      if (ProcessNetoutputNodeDynShape(node_ptr, options_.output_datatype) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Process netoutput node failed");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

bool GraphPrepare::IsBroadCastOpData(const ge::NodePtr &var_node) {
  for (auto &out_anchor : var_node->GetAllOutDataAnchors()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_anchor);
    for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
      ge::NodePtr dst_node = in_anchor->GetOwnerNode();
      GE_RT_FALSE_CHECK_NOTNULL(dst_node);
      if (dst_node->GetType() == HCOMBROADCAST || dst_node->GetType() == HVDCALLBACKBROADCAST) {
        return true;
      }
    }
  }
  return false;
}

bool GraphPrepare::ConfirmUseOpAndIndexByAnchor(const ge::InDataAnchorPtr &in_anchor,
                                                const map<string, std::set<int>> &confirm_ops, ge::NodePtr &use_node) {
  GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
  ge::NodePtr dst_node = in_anchor->GetOwnerNode();
  GE_RT_FALSE_CHECK_NOTNULL(dst_node);
  ge::OpDescPtr dst_op_desc = dst_node->GetOpDesc();
  GE_RT_FALSE_CHECK_NOTNULL(dst_op_desc);
  const string &dst_type = dst_op_desc->GetType();
  int input_index = in_anchor->GetIdx();

  GELOGD("ConfirmUseOpAndIndex, var name %s, dst_type = %s, input index %d", dst_node->GetName().c_str(),
         dst_type.c_str(), input_index);

  if (confirm_ops.count(dst_type) > 0) {
    if (confirm_ops.at(dst_type).count(input_index) > 0) {
      use_node = dst_node;
      return true;
    }
  }
  return false;
}

bool GraphPrepare::ConfirmUseOpAndIndexByNode(const ge::NodePtr &var_node,
                                              const map<string, std::set<int>> &confirm_ops, ge::NodePtr &use_node) {
  GE_RT_FALSE_CHECK_NOTNULL(var_node);
  for (auto &out_anchor : var_node->GetAllOutDataAnchors()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_anchor);
    for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
      if (ConfirmUseOpAndIndexByAnchor(in_anchor, confirm_ops, use_node)) {
        return true;
      }
    }
  }
  return false;
}
void GraphPrepare::AdjustBroadCastOpData(const ge::NodePtr &var_node) {
  if (!ge::AttrUtils::SetStr(var_node->GetOpDesc(), VAR_ATTR_VAR_IS_BROADCAST, "var_is_restore")) {
    GELOGW("set var_is_restore failed");
  }
}

bool GraphPrepare::IsAssignOpData(const ge::NodePtr &var_node) {
  GELOGD("IsAssignOpData var_node %s", var_node->GetName().c_str());
  std::map<std::string, std::set<int>> assign_ops = {{ASSIGN, {0}}};

  ge::NodePtr assign_node = nullptr;
  if (ConfirmUseOpAndIndexByNode(var_node, assign_ops, assign_node)) {
    return true;
  }

  return false;
}

void GraphPrepare::AdjustAssignOpData(const ge::NodePtr &var_node) {
  if (!ge::AttrUtils::SetStr(var_node->GetOpDesc(), VAR_ATTR_VAR_IS_RESTORE, "var_is_restore")) {
    GELOGW("SetStr var_is_restore failed");
  }
}
}  // namespace ge
