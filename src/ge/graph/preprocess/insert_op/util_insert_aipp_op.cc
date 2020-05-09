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
#include "inc/common/dynamic_aipp.h"
#include "common/formats/utils/formats_trans_utils.h"

using domi::AippOpParams;

namespace ge {
namespace {
const char *const kMbatchSwitchnName = "mbatch-switch-name";
}  // namespace
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
  ge::GeAttrValue::NamedAttrs aipp_attr;
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
  GELOGI("Get size [%ld] from aipp [%s].", size, aipp_op_desc->GetName().c_str());
  if (size == 0) {
    GELOGE(FAILED, "Can not get size from aipp [%s]", aipp_op_desc->GetName().c_str());
    return FAILED;
  }

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
  GELOGI("Aipp [%s] input datatype is %s, origin datatype is %s, input shape is %s", aipp_op_desc->GetName().c_str(),
         TypeUtils::DataTypeToSerialString(aipp_dt).c_str(), TypeUtils::DataTypeToSerialString(aipp_origni_dt).c_str(),
         ge::formats::ShapeToString(aipp_shape.GetDims()).c_str());

  const GeTensorDescPtr &input = src_op->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input);
  input->SetDataType(aipp_dt);
  input->SetOriginDataType(aipp_origni_dt);
  input->SetShape(aipp_shape);
  input->SetOriginShape(aipp_shape);
  ge::TensorUtils::SetSize(*input, size);

  const GeTensorDescPtr &output = src_op->MutableOutputDesc(peer_out_anchor->GetIdx());
  GE_CHECK_NOTNULL(output);
  output->SetDataType(aipp_dt);
  output->SetOriginDataType(aipp_origni_dt);
  output->SetShape(aipp_shape);
  output->SetOriginShape(aipp_shape);
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
  TensorUtils::SetSize(*input_desc, max_size);

  auto data_opdesc = data->GetOpDesc();
  GE_CHECK_NOTNULL(data_opdesc);
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
  return SUCCESS;
}
}  // namespace ge
