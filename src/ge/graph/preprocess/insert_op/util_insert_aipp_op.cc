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

#include "graph/preprocess/insert_op/util_insert_aipp_op.h"
#include <fstream>
#include <utility>
#include "common/dynamic_aipp.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/ge/ge_util.h"
#include "common/op/ge_op_utils.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/preprocess/insert_op/ge_aipp_op.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

using domi::AippOpParams;

namespace ge {
namespace {
const char *const kMbatchSwitchnName = "mbatch-switch-name";
}  // namespace
static void ConvertShape2Nhwc(Format &format, vector<int64_t> &shape_vec) {
  if ((format == FORMAT_NHWC) || (shape_vec.size() != static_cast<size_t>(NORMAL_TENSOR_SIZE))) {
    return;
  }
  if (format != FORMAT_NCHW) {
    GELOGW("The format is not NCHW, current format is %s.", TypeUtils::FormatToSerialString(format).c_str());
    return;
  }
  vector<int64_t> shape_vec_tmp;
  shape_vec.swap(shape_vec_tmp);
  shape_vec.push_back(shape_vec_tmp[NCHW_DIM_N]);
  shape_vec.push_back(shape_vec_tmp[NCHW_DIM_H]);
  shape_vec.push_back(shape_vec_tmp[NCHW_DIM_W]);
  shape_vec.push_back(shape_vec_tmp[NCHW_DIM_C]);
  return;
}

Status InsertNewOpUtil::Init() {
  insert_op_conf_.reset((new (std::nothrow) domi::InsertNewOps()));
  GE_CHECK_NOTNULL(insert_op_conf_);
  return SUCCESS;
}

Status InsertNewOpUtil::Parse(const char *conf_path) {
  if (conf_path == nullptr || *conf_path == '\0') {
    return SUCCESS;
  }

  GE_CHK_BOOL_RET_STATUS(ReadProtoFromText(conf_path, insert_op_conf_.get()), FAILED, "Read AIPP conf file error: %s",
                         conf_path);

  GE_CHK_STATUS_RET(CheckPositionNotRepeat(), "Check insert position of op failed");

  for (int i = 0; i < insert_op_conf_->aipp_op_size(); i++) {
    domi::AippOpParams *aipp_op_params = insert_op_conf_->mutable_aipp_op(i);
    std::unique_ptr<AippOp> aipp_op(new (std::nothrow) AippOp());
    GE_CHECK_NOTNULL(aipp_op);
    GE_CHK_STATUS_RET(aipp_op->Init(aipp_op_params), "Aipp op init failed.");
    insert_ops_.push_back(std::move(aipp_op));
  }

  for (auto &dynamic_op : insert_ops_) {
    GE_CHECK_NOTNULL(dynamic_op);
    GE_CHK_STATUS_RET(dynamic_op->ValidateParams(), "Validate insert_op config file failed");
    GE_CHK_STATUS_RET(dynamic_op->SetDefaultParams(), "Set default value of insert_op failed");
  }

  return SUCCESS;
}

Status InsertNewOpUtil::InsertAippOps(ComputeGraphPtr &graph, std::string &aippConfigPath) {
  GE_CHECK_NOTNULL(graph);
  for (uint32_t index = 0; index < insert_ops_.size(); ++index) {
    GE_CHK_STATUS_RET(insert_ops_[index]->InsertAippToGraph(graph, aippConfigPath, index), "insert op to graph failed");
  }

  GE_CHK_STATUS_RET(CheckGraph(graph), "after inserting all ops, check graph failed");

  GE_CHK_STATUS_RET(graph->TopologicalSorting(), "after insert dynamic op, sort graph failed");

  ClearNewOps();

  return SUCCESS;
}

void InsertNewOpUtil::ClearNewOps() {
  if (insert_op_conf_ != nullptr) {
    insert_op_conf_->Clear();
    insert_ops_.clear();
  }
}

Status InsertNewOpUtil::CheckPositionNotRepeat() {
  for (int i = 0; i < insert_op_conf_->aipp_op_size(); i++) {
    const domi::AippOpParams *item = insert_op_conf_->mutable_aipp_op(i);

    for (int j = i + 1; j < insert_op_conf_->aipp_op_size(); j++) {
      const domi::AippOpParams *another_item = insert_op_conf_->mutable_aipp_op(j);

      GE_IF_BOOL_EXEC(item->related_input_rank() != another_item->related_input_rank(), continue;);

      GE_IF_BOOL_EXEC(
        item->input_edge_idx_size() == 0 || another_item->input_edge_idx_size() == 0 ||
          item->input_edge_idx(0) == another_item->input_edge_idx(0),
        GELOGE(PARAM_INVALID,
               "Can not insert aipp op to the same postion! please check related_input_rank and input_edge_idx.");
        return PARAM_INVALID;);
    }
  }

  return SUCCESS;
}

Status InsertNewOpUtil::CheckGraph(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  domi::AippOpParams::AippMode aippMode = domi::AippOpParams::undefined;

  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      continue;
    }
    size_t next_nodes_cnt = 0;
    std::vector<NodePtr> aippNodes;
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      for (const auto &inAnchor : anchor->GetPeerInDataAnchors()) {
        const std::string &nodeType = inAnchor->GetOwnerNode()->GetType();
        next_nodes_cnt++;
        if (nodeType == AIPP) {
          aippNodes.push_back(inAnchor->GetOwnerNode());
          continue;
        }
      }
    }
    GE_CHK_BOOL_RET_STATUS((aippNodes.size() == 0) || (aippNodes.size() == next_nodes_cnt), PARAM_INVALID,
                           "Can not config part of outputs of Data node to support AIPP, config all "
                           "of the outputs of Data to support AIPP, or config none of them");

    std::unique_ptr<domi::AippOpParams> aippParams(new (std::nothrow) domi::AippOpParams());
    GE_CHECK_NOTNULL(aippParams);

    GE_IF_BOOL_EXEC(aippNodes.size() > 0, GE_CHK_STATUS(GetAippParams(aippParams, aippNodes[0]));
                    aippMode = (aippMode == domi::AippOpParams::undefined) ? aippParams->aipp_mode() : aippMode;
                    GE_CHK_BOOL_RET_STATUS(aippMode == aippParams->aipp_mode(), PARAM_INVALID,
                                           "The aipp_mode of all aipp_op must be the same"););
    GE_IF_BOOL_EXEC(
      aippNodes.size() > 1, for (decltype(aippNodes)::size_type i = 1; i < aippNodes.size(); i++) {
        std::unique_ptr<domi::AippOpParams> currAippParam(new (std::nothrow) domi::AippOpParams());
        GE_CHECK_NOTNULL(currAippParam);
        GE_CHK_STATUS(GetAippParams(currAippParam, aippNodes[i]));

        GE_CHK_BOOL_RET_STATUS(aippMode == currAippParam->aipp_mode(), PARAM_INVALID,
                               "The aipp_mode of all aipp_op must be the same");
        if (aippMode == domi::AippOpParams::static_) {
          GE_CHK_BOOL_RET_STATUS(aippParams->input_format() == currAippParam->input_format(), PARAM_INVALID,
                                 "The input_format of all aipp_ops after one Data should be the same");
          GE_CHK_BOOL_RET_STATUS(aippParams->src_image_size_w() == currAippParam->src_image_size_w(), PARAM_INVALID,
                                 "The src_image_size_w of all aipp_ops after one Data should be the same");
          GE_CHK_BOOL_RET_STATUS(aippParams->src_image_size_h() == currAippParam->src_image_size_h(), PARAM_INVALID,
                                 "The src_image_size_h of all aipp_ops after one Data should be the same");
        } else {
          GE_CHK_BOOL_RET_STATUS(aippParams->max_src_image_size() == currAippParam->max_src_image_size(), PARAM_INVALID,
                                 "The max_src_image_size of all aipp_ops after one Data should be the same");
        }
      });
  }

  return SUCCESS;
}

Status InsertNewOpUtil::GetAippParams(const std::unique_ptr<domi::AippOpParams> &aippParams, const NodePtr &aipp_node) {
  GE_CHECK_NOTNULL(aipp_node);
  ge::GeAttrValue::NAMED_ATTRS aipp_attr;
  const OpDescPtr tmpOpPtr = aipp_node->GetOpDesc();
  GE_CHECK_NOTNULL(tmpOpPtr);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetNamedAttrs(tmpOpPtr, ATTR_NAME_AIPP, aipp_attr), FAILED,
                         "Aipp node should contain param aipp!");
  GE_CHK_STATUS_RET(OpUtils::ConvertAippParams(aipp_attr, aippParams.get()), "get aipp params failed");

  return SUCCESS;
}
Status InsertNewOpUtil::UpdateDataNodeByAipp(const ComputeGraphPtr &graph) {
  std::map<std::string, NodePtr> switchn_names_to_data;
  std::set<NodePtr> updated_switchn;

  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == DATA) {
      std::string switchn_name;
      if (AttrUtils::GetStr(node->GetOpDesc(), kMbatchSwitchnName, switchn_name)) {
        switchn_names_to_data[switchn_name] = node;
      }
    }
    if (node->GetType() == AIPP) {
      GE_RETURN_IF_ERROR(UpdatePrevNodeByAipp(node, updated_switchn));
    }
  }

  for (auto &switchn : updated_switchn) {
    auto data_iter = switchn_names_to_data.find(switchn->GetName());
    if (data_iter == switchn_names_to_data.end()) {
      GELOGE(INTERNAL_ERROR, "Failed to find relative data node by switchn %s", switchn->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GE_RETURN_IF_ERROR(UpdateDataBySwitchN(switchn, data_iter->second));
  }

  return SUCCESS;
}
Status InsertNewOpUtil::UpdatePrevNodeByAipp(NodePtr &node, std::set<NodePtr> &switchns) {
  GELOGI("Start to update prev node size by aipp %s.", node->GetName().c_str());
  auto aipp_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(aipp_op_desc);
  auto aipp_input = aipp_op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(aipp_input);

  int64_t size = 0;
  graphStatus graph_ret = ge::TensorUtils::GetSize(*aipp_input, size);
  if (graph_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "UpdateOutputDesc fail, graph_ret:%d", graph_ret);
    return FAILED;
  }
  GELOGI("Get input size [%ld] from aipp [%s].", size, aipp_op_desc->GetName().c_str());
  if (size == 0) {
    GELOGE(FAILED, "Can not get size from aipp [%s]", aipp_op_desc->GetName().c_str());
    return FAILED;
  }
  (void)AttrUtils::SetInt(aipp_input, ATTR_NAME_INPUT_ORIGIN_SIZE, size);

  auto in_data_anchor = node->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(in_data_anchor);
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);
  const auto &src_node = peer_out_anchor->GetOwnerNode();
  const auto &src_op = src_node->GetOpDesc();
  GE_CHECK_NOTNULL(src_op);

  // if the type of src_node is SwitchN, the input of it may be updated to a size not the max one
  // the correct size will be updated in function `UpdateDataBySwitchN`
  DataType aipp_dt = aipp_input->GetDataType();
  aipp_input->SetOriginDataType(aipp_dt);
  DataType aipp_origni_dt = aipp_input->GetOriginDataType();
  GeShape aipp_shape = aipp_input->GetShape();
  Format aipp_format = aipp_input->GetFormat();
  GELOGI("Aipp [%s] input datatype is %s, origin datatype is %s, input shape is %s", aipp_op_desc->GetName().c_str(),
         TypeUtils::DataTypeToSerialString(aipp_dt).c_str(), TypeUtils::DataTypeToSerialString(aipp_origni_dt).c_str(),
         ge::formats::ShapeToString(aipp_shape.GetDims()).c_str());

  const GeTensorDescPtr &input = src_op->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input);
  input->SetDataType(aipp_dt);
  input->SetOriginDataType(aipp_origni_dt);
  input->SetShape(aipp_shape);
  input->SetOriginShape(aipp_shape);
  input->SetFormat(aipp_format);
  input->SetOriginFormat(aipp_format);
  ge::TensorUtils::SetSize(*input, size);

  const GeTensorDescPtr &output = src_op->MutableOutputDesc(peer_out_anchor->GetIdx());
  GE_CHECK_NOTNULL(output);
  output->SetDataType(aipp_dt);
  output->SetOriginDataType(aipp_origni_dt);
  output->SetShape(aipp_shape);
  output->SetOriginShape(aipp_shape);
  output->SetFormat(aipp_format);
  output->SetOriginFormat(aipp_format);
  ge::TensorUtils::SetSize(*output, size);
  if (src_node->GetType() == SWITCHN) {
    switchns.insert(src_node);
  }
  GELOGI("Set node %s output %d size %ld by aipp.", src_node->GetName().c_str(), peer_out_anchor->GetIdx(), size);

  return SUCCESS;
}
Status InsertNewOpUtil::UpdateDataBySwitchN(const NodePtr &switchn, const NodePtr &data) {
  size_t max_index = switchn->GetOpDesc()->GetOutputsSize();
  int64_t max_size = 0;
  for (size_t i = 0; i < switchn->GetOpDesc()->GetOutputsSize(); ++i) {
    int64_t size = 0;
    auto output_desc = switchn->GetOpDesc()->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    if (TensorUtils::GetSize(*output_desc, size) == GRAPH_SUCCESS) {
      if (max_size < size) {
        max_size = size;
        max_index = i;
      }
    }
  }
  if (max_index >= switchn->GetOpDesc()->GetOutputsSize()) {
    GELOGE(INTERNAL_ERROR, "No max size found from switchn node %s", switchn->GetName().c_str());
    return INTERNAL_ERROR;
  }
  auto output_desc = switchn->GetOpDesc()->MutableOutputDesc(max_index);
  auto input_desc = switchn->GetOpDesc()->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  input_desc->SetDataType(output_desc->GetDataType());
  input_desc->SetOriginDataType(output_desc->GetOriginDataType());
  input_desc->SetShape(output_desc->GetShape());
  input_desc->SetOriginShape(output_desc->GetOriginShape());
  input_desc->SetFormat(output_desc->GetFormat());
  input_desc->SetOriginFormat(output_desc->GetOriginFormat());
  TensorUtils::SetSize(*input_desc, max_size);

  auto data_opdesc = data->GetOpDesc();
  GE_CHECK_NOTNULL(data_opdesc);
  Format old_format = data_opdesc->MutableOutputDesc(0)->GetFormat();

  auto ret = data_opdesc->UpdateOutputDesc(0, *input_desc);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to update data %s output using switchn %s", data->GetName().c_str(),
           switchn->GetName().c_str());
    return INTERNAL_ERROR;
  }
  ret = data_opdesc->UpdateInputDesc(0, *input_desc);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to update data %s input using switchn %s", data->GetName().c_str(),
           switchn->GetName().c_str());
    return INTERNAL_ERROR;
  }
  // Update attr _mbatch_origin_input_dims for data when it is linked to aipp
  UpdateMultiBatchInputDims(data_opdesc, old_format);
  return SUCCESS;
}

void InsertNewOpUtil::UpdateMultiBatchInputDims(const OpDescPtr &data_opdesc, Format &old_format) {
  if (!data_opdesc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS)) {
    GELOGW("Failed to acquire _mbatch_origin_input_dims attr from node [%s]", data_opdesc->GetName().c_str());
    return;
  }
  auto new_data_dims = data_opdesc->GetOutputDesc(0).GetShape().GetDims();
  vector<int64_t> origin_input_dims;
  (void)AttrUtils::GetListInt(data_opdesc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);
  // Convert origin_input_dims to NHWC because data format is set to NHWC when it is linked to aipp.
  ConvertShape2Nhwc(old_format, origin_input_dims);
  if (new_data_dims.size() != origin_input_dims.size()) {
    return;
  }
  for (size_t i = 0; i < origin_input_dims.size(); ++i) {
    // Need to update shape when aipp has crop function because H,W is different, ignore -1.
    if (origin_input_dims[i] > 0) {
      origin_input_dims[i] = new_data_dims[i];
    }
  }
  (void)AttrUtils::SetListInt(data_opdesc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);
  return;
}

Status InsertNewOpUtil::GetDataRelatedNode(NodePtr &node, std::map<NodePtr, std::set<NodePtr>> &data_next_node_map) {
  GELOGI("Start to get data and next node %s.", node->GetName().c_str());
  OpDescPtr data_op = node->GetOpDesc();
  GE_CHECK_NOTNULL(data_op);
  if (!data_op->HasAttr(ATTR_NAME_AIPP)) {
    GELOGI("there is not AIPP info for Data: %s.", data_op->GetName().c_str());
    return SUCCESS;
  }

  std::unique_ptr<domi::AippOpParams> aipp_params(new (std::nothrow) domi::AippOpParams());
  ge::GeAttrValue::NAMED_ATTRS aipp_attr;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetNamedAttrs(data_op, ATTR_NAME_AIPP, aipp_attr), GE_AIPP_NOT_EXIST,
                         "Data node do not contain param aipp!");
  GE_CHK_STATUS_RET(OpUtils::ConvertAippParams(aipp_attr, aipp_params.get()), "get aipp params failed");

  if (aipp_params->aipp_mode() != domi::AippOpParams::static_) {
    return SUCCESS;
  }

  for (auto out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    auto peer_in_anchors = out_data_anchor->GetPeerInDataAnchors();
    for (auto peer_in_data_anchor : peer_in_anchors) {
      GE_CHECK_NOTNULL(peer_in_data_anchor);
      const auto &dst_node = peer_in_data_anchor->GetOwnerNode();
      const auto &dst_op = dst_node->GetOpDesc();
      GE_CHECK_NOTNULL(dst_op);

      if (dst_op->GetType() == AIPP || dst_op->GetType() == SWITCHN) {
        auto data_iter = data_next_node_map.find(node);
        if (data_iter == data_next_node_map.end()) {
          std::set<NodePtr> next_node_set;
          next_node_set.insert(dst_node);
          data_next_node_map[node] = next_node_set;
        } else {
          if (data_next_node_map[node].find(dst_node) == data_next_node_map[node].end()) {
            data_next_node_map[node].insert(dst_node);
          }
        }
      }
    }
  }

  return SUCCESS;
}

Status InsertNewOpUtil::GetAllAipps(const NodePtr &node, std::vector<NodePtr> &aipps) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr op = node->GetOpDesc();
  GE_CHECK_NOTNULL(op);
  GELOGI("Get all aipp node from this node %s.", op->GetName().c_str());
  if (op->GetType() == AIPP) {
    aipps.emplace_back(node);
  } else if (op->GetType() == SWITCHN) {
    for (auto out_data_anchor : node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(out_data_anchor);
      auto peer_in_anchors = out_data_anchor->GetPeerInDataAnchors();
      if (peer_in_anchors.size() > 0) {
        auto peer_in_anchor = peer_in_anchors.at(0);
        GE_CHECK_NOTNULL(peer_in_anchor);
        auto dst_aipp_node = peer_in_anchor->GetOwnerNode();
        if (dst_aipp_node->GetType() == AIPP) {
          aipps.emplace_back(dst_aipp_node);
        }
      }
    }
  }
  return SUCCESS;
}

Status InsertNewOpUtil::RecordAIPPInfoToData(const ComputeGraphPtr &graph) {
  GELOGI("Start to record aipp info to Data.");
  std::map<NodePtr, std::set<NodePtr>> data_next_node_map;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == DATA) {
      GE_RETURN_IF_ERROR(GetDataRelatedNode(node, data_next_node_map));
    }
  }

  for (auto it : data_next_node_map) {
    std::vector<std::string> input_dims;
    std::vector<std::string> output_dims;
    auto data_node = it.first;
    std::set<NodePtr> aipps_or_switchs = it.second;
    if (aipps_or_switchs.size() != 1) {
      GELOGW("The number of successors swith or aipp of data is more than 1");
      continue;
    }

    std::vector<NodePtr> aipps;
    GE_RETURN_IF_ERROR(GetAllAipps(*aipps_or_switchs.begin(), aipps));
    GELOGI("RecordAIPPInfoToData: Data: name[%s], type[%s], batch size[%u]", data_node->GetName().c_str(),
           data_node->GetType().c_str(), aipps.size());

    for (auto aipp_it : aipps) {
      string input;
      string output;
      GetInputOutputInfo(data_node, aipp_it, input, output);
      input_dims.emplace_back(input);
      output_dims.emplace_back(output);

      // When static aipp is set, need to get the model input dims which processed by aipp
      GE_RETURN_IF_ERROR(SetModelInputDims(data_node, aipp_it));
    }

    if (!AttrUtils::SetListStr(data_node->GetOpDesc(), ATTR_NAME_AIPP_INPUTS, input_dims)) {
      GELOGE(FAILED, "SetListStr of %s failed.", ATTR_NAME_AIPP_INPUTS.c_str());
      return FAILED;
    }

    if (!AttrUtils::SetListStr(data_node->GetOpDesc(), ATTR_NAME_AIPP_OUTPUTS, output_dims)) {
      GELOGE(FAILED, "SetListStr of %s failed.", ATTR_NAME_AIPP_OUTPUTS.c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status InsertNewOpUtil::GetInputOutputInfo(NodePtr &data_node, NodePtr &aipp_node, std::string &input,
                                           std::string &output) {
  GE_CHECK_NOTNULL(data_node);
  GE_CHECK_NOTNULL(aipp_node);
  OpDescPtr data_op = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(data_op);
  OpDescPtr aipp_op = aipp_node->GetOpDesc();
  GE_CHECK_NOTNULL(aipp_op);

  // aipp node's original output shape equals to original model data's shape
  ConstGeTensorDescPtr output_desc = aipp_op->GetOutputDescPtr(0);
  Format orig_format = output_desc->GetOriginFormat();
  DataType orig_data_type = output_desc->GetOriginDataType();
  std::string tensor_name = data_op->GetName();
  size_t dim_num = output_desc->GetOriginShape().GetDimNum();
  int64_t tensor_size = 0;
  (void)TensorUtils::CalcTensorMemSize(output_desc->GetOriginShape(), orig_format, orig_data_type, tensor_size);
  int64_t input_size = tensor_size;
  input = TypeUtils::FormatToSerialString(orig_format) + ":" + TypeUtils::DataTypeToSerialString(orig_data_type) + ":" +
          tensor_name + ":" + std::to_string(input_size) + ":" + std::to_string(dim_num) + ":" +
          formats::JoinToString(output_desc->GetOriginShape().GetDims());

  Format format = output_desc->GetFormat();
  DataType data_type = output_desc->GetDataType();
  std::string output_name = aipp_op->GetOutputNameByIndex(0);
  size_t output_dim_num = output_desc->GetShape().GetDimNum();
  (void)TensorUtils::CalcTensorMemSize(output_desc->GetShape(), output_desc->GetFormat(), output_desc->GetDataType(),
                                       tensor_size);
  int64_t output_size = tensor_size;
  output = TypeUtils::FormatToSerialString(format) + ":" + TypeUtils::DataTypeToSerialString(data_type) + ":" +
           output_name + ":" + std::to_string(output_size) + ":" + std::to_string(output_dim_num) + ":" +
           formats::JoinToString(output_desc->GetShape().GetDims());

  GELOGI("GetInputOutputInfo: get data[%s] node related aipp[%s] node info, input[%s], output[%s].",
         data_node->GetName().c_str(), aipp_node->GetName().c_str(), input.c_str(), output.c_str());
  return SUCCESS;
}

Status InsertNewOpUtil::SetModelInputDims(NodePtr &data_node, NodePtr &aipp_node) {
  GE_CHECK_NOTNULL(data_node);
  GE_CHECK_NOTNULL(aipp_node);
  OpDescPtr data_opdesc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(data_opdesc);
  OpDescPtr aipp_opdesc = aipp_node->GetOpDesc();
  GE_CHECK_NOTNULL(aipp_opdesc);

  // In dynamic bacth/hw scenario, the new model input dims only need be set once
  if (data_node->GetOpDesc()->HasAttr(ATTR_NAME_INPUT_DIMS)) {
    GELOGD("Data %s already has attribute %s", data_node->GetOpDesc()->GetName().c_str(), ATTR_NAME_INPUT_DIMS.c_str());
    return SUCCESS;
  }
  vector<int64_t> model_input_dims;
  vector<int64_t> origin_input_dims;
  if (AttrUtils::GetListInt(aipp_opdesc, ATTR_NAME_INPUT_DIMS, model_input_dims) && !model_input_dims.empty()) {
    // When dynamic bacth/hw is set, N or HW need to be set to -1
    if (AttrUtils::GetListInt(data_opdesc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims) &&
        !origin_input_dims.empty()) {
      GELOGI("In dynamic bacth/hw scenario, N or HW need to be set to -1. model_input_dims: %s, origin_input_dims: %s",
             formats::JoinToString(model_input_dims).c_str(), formats::JoinToString(origin_input_dims).c_str());
      for (size_t i = 0; i < origin_input_dims.size(); ++i) {
        // N or HW need to be set to -1
        if (origin_input_dims[i] < 0) {
          model_input_dims[i] = origin_input_dims[i];
        }
      }
    }
    GELOGD("After set H/W to -1, the model input dims: %s.", formats::JoinToString(model_input_dims).c_str());
    if (!AttrUtils::SetListInt(data_opdesc, ATTR_NAME_INPUT_DIMS, model_input_dims)) {
      GELOGE(FAILED, "SetListInt of %s failed.", ATTR_NAME_INPUT_DIMS.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace ge
