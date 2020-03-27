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
#include "common/ge/ge_util.h"
#include "common/op/ge_op_utils.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
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
Status InsertNewOpUtil::Init() {
  insert_op_conf_.reset((new (std::nothrow) domi::InsertNewOps()));
  GE_CHECK_NOTNULL(insert_op_conf_);
  return SUCCESS;
}

namespace {
constexpr uint64_t kMinTransferShape = 3;
constexpr int64_t kMaxBatchCountNum = 32768;

Status ExpandDimsAndNormalizedToNCHW(ge::Format src_format, const std::vector<int64_t> &src_dims,
                                     std::vector<int64_t> &nchw_dims) {
  GELOGD("Enter ExpandDimsAndNormalizedToNCHW process!");
  // The input of 3-dimension and 4-dimension is considered as picture dimension,
  // which needs to be converted according to specific format
  if (src_dims.size() != DIM_DEFAULT_SIZE && src_dims.size() != kMinTransferShape) {
    GELOGE(PARAM_INVALID, "expand and normalize format failed, src size [%lu] is not in range [3,4]", src_dims.size());
    return PARAM_INVALID;
  }

  switch (src_format) {
    case ge::FORMAT_NCHW:
      if (src_dims.size() == DIM_DEFAULT_SIZE) {
        nchw_dims = src_dims;
      } else {
        nchw_dims.push_back(1);
        nchw_dims.push_back(src_dims[0]);
        nchw_dims.push_back(src_dims[1]);
        nchw_dims.push_back(src_dims[2]);
      }
      break;
    case ge::FORMAT_NHWC:
      if (src_dims.size() == DIM_DEFAULT_SIZE) {
        nchw_dims.push_back(src_dims[NHWC_DIM_N]);
        nchw_dims.push_back(src_dims[NHWC_DIM_C]);
        nchw_dims.push_back(src_dims[NHWC_DIM_H]);
        nchw_dims.push_back(src_dims[NHWC_DIM_W]);
      } else {
        nchw_dims.push_back(1);
        nchw_dims.push_back(src_dims[HWC_DIM_C]);
        nchw_dims.push_back(src_dims[HWC_DIM_H]);
        nchw_dims.push_back(src_dims[HWC_DIM_W]);
      }
      break;
    default:
      GELOGE(PARAM_INVALID, "Not support src format: %d", src_format);
      return PARAM_INVALID;
  }

  return ge::SUCCESS;
}
Status GetDataOpDims(const ge::NodePtr data_node, ge::Format format, std::vector<int64_t> &nchw_dims) {
  GELOGD("Enter GetDataOpDims process!");

  auto data_input_desc_ptr = data_node->GetOpDesc()->GetInputDescPtr(0);  // GetOpDesc() has check null before logic
  if (data_input_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "data_node's input desc object is null");
    return PARAM_INVALID;
  }
  auto shape = data_input_desc_ptr->GetShape().GetDims();
  if ((shape.size() < kMinTransferShape) && (shape.size() > DIM_DEFAULT_SIZE)) {
    GELOGE(PARAM_INVALID, "when dynamic aipp, shape must be in range [3, 4], but is %lu", shape.size());
    return PARAM_INVALID;
  }

  return ExpandDimsAndNormalizedToNCHW(format, shape, nchw_dims);
}
}  // namespace
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

Status InsertNewOpUtil::AddAippInputData(const ge::NodePtr &aipp_node, const ge::ComputeGraphPtr &graph) {
  GELOGD("Enter add aipp data node process!");
  static int index = 0;

  // get previous node, it should be DATA
  auto data_node = aipp_node->GetInDataNodes().at(0);
  if (data_node->GetOpDesc() == nullptr) {
    GELOGE(PARAM_INVALID, "data node has no opdesc!");
    return PARAM_INVALID;
  }
  if (data_node->GetOpDesc()->GetType() != DATA) {
    GELOGE(PARAM_INVALID, "aipp node should follow one data node, but previous node's type is %s",
           data_node->GetOpDesc()->GetType().c_str());
    return PARAM_INVALID;
  }
  auto ori_data_format = static_cast<ge::Format>(static_cast<int>(domi::GetContext().format));
  if (ori_data_format != FORMAT_NCHW && ori_data_format != FORMAT_NHWC) {
    GELOGE(PARAM_INVALID, "when dynamic aipp,input_format must be NCHW or NHWC, but [%s] format is %s",
           data_node->GetName().c_str(), ge::TypeUtils::FormatToSerialString(ori_data_format).c_str());
    return PARAM_INVALID;
  }

  std::vector<int64_t> nchw_dims;
  auto ret = GetDataOpDims(data_node, ori_data_format, nchw_dims);
  if (ret != ge::SUCCESS) {
    GELOGE(PARAM_INVALID, "get data_node dims and transfer to nchw_dims failed!");
    return PARAM_INVALID;
  }

  auto batch_count = nchw_dims[NCHW_DIM_N];
  // new add aipp_data ops for dynamic aipp param input
  OpDescPtr op_desc_ptr_data =
      ge::MakeShared<ge::OpDesc>(std::string("aipp_data_").append(std::to_string(index++)), AIPPDATA);

  // calc max size
  if (batch_count <= 0 || batch_count > kMaxBatchCountNum) {
    GELOGE(PARAM_INVALID, "batch_cout must be in range(0, %ld]", kMaxBatchCountNum);
    return PARAM_INVALID;
  }
  uint64_t max_dynamic_aipp_size = sizeof(kAippDynamicPara) + (batch_count - 1) * sizeof(kAippDynamicBatchPara);

  GELOGI("Add aipp input data, batch count: %ld, max_dynamic_aipp_size: %ld", batch_count, max_dynamic_aipp_size);
  vector<int64_t> input_shape_dim(1, 1);
  input_shape_dim[0] = static_cast<int64_t>(max_dynamic_aipp_size);
  GeShape input_shape(input_shape_dim);
  // construct input tensor
  GeTensorDesc input_tensor(input_shape, FORMAT_ND, DT_UINT8);
  TensorUtils::SetReuseInput(input_tensor, false);
  TensorUtils::SetSize(input_tensor, static_cast<uint32_t>(max_dynamic_aipp_size));

  auto stat1 = op_desc_ptr_data->AddInputDesc(input_tensor);

  GeShape output_shape(input_shape_dim);
  // construct output tensor
  GeTensorDesc output_tensor(output_shape, FORMAT_ND, DT_UINT8);
  TensorUtils::SetReuseInput(output_tensor, false);
  TensorUtils::SetSize(output_tensor, static_cast<uint32_t>(max_dynamic_aipp_size));
  auto stat2 = op_desc_ptr_data->AddOutputDesc(output_tensor);

  NodePtr aipp_data_node_ptr = graph->AddNode(op_desc_ptr_data);
  if (aipp_data_node_ptr == nullptr) {
    GELOGE(INTERNAL_ERROR, "graph add node failed.");
    return INTERNAL_ERROR;
  }
  // add node desc for aipp node
  GE_CHECK_NOTNULL(aipp_node->GetOpDesc());
  auto stat3 = aipp_node->GetOpDesc()->UpdateInputDesc(1, output_tensor);
  if (stat1 != SUCCESS || stat2 != SUCCESS || stat3 != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "node process desc failed!");
    return INTERNAL_ERROR;
  }
  // aipp_node should have two input data but now tbe only one input
  if (GraphUtils::AddEdge(aipp_data_node_ptr->GetOutDataAnchor(0), aipp_node->GetInDataAnchor(1)) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Add Anchor anchor between aipp data node and aipp failed!");
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status InsertNewOpUtil::InsertNewOps(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &insert_op : insert_ops_) {
    GE_CHK_STATUS_RET(insert_op->InsertOpToGraph(graph), "insert op to graph failed");
  }

  GE_CHK_STATUS_RET(CheckGraph(graph), "after inserting all ops, check graph failed");

  GE_CHK_STATUS_RET(graph->TopologicalSorting(), "after insert dynamic op, sort graph failed");

  ClearNewOps();

  return SUCCESS;
}

Status InsertNewOpUtil::InsertAippOps(ComputeGraphPtr &graph, std::string &aippConfigPath) {
  GE_CHECK_NOTNULL(graph);
  for (auto &insert_op : insert_ops_) {
    AippOpParams::AippMode aipp_mode = insert_op->GetAippMode();
    ge::NodePtr aipp_node = nullptr;
    GE_CHK_STATUS_RET(insert_op->InsertAippToGraph(graph, aippConfigPath, aipp_node), "insert op to graph failed");
    if (aipp_node == nullptr) {
      GELOGE(FAILED, "aipp node is null!");
      return FAILED;
    }
    if (aipp_mode == AippOpParams::dynamic) {
      Status stat = AddAippInputData(aipp_node, graph);
      if (stat != SUCCESS) {
        GELOGE(FAILED, "Add aipp input data failed");
        return FAILED;
      }
    }
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
                 "Can not insert aipp op to the same position! please check related_input_rank and input_edge_idx.");
          return PARAM_INVALID;);
    }
  }

  return SUCCESS;
}

Status InsertNewOpUtil::CheckGraph(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  domi::AippOpParams::AippMode aippMode = domi::AippOpParams::undefined;

  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() != DATA) {
      continue;
    }

    std::vector<NodePtr> aippNodes;
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      for (const auto &inAnchor : anchor->GetPeerInDataAnchors()) {
        const std::string &nodeType = inAnchor->GetOwnerNode()->GetType();

        GE_IF_BOOL_EXEC(nodeType == SSDPRIORBOX || nodeType == SHAPE, continue;);

        GE_CHK_BOOL_RET_STATUS(aippNodes.size() == 0 || nodeType == AIPP, PARAM_INVALID,
                               "Can not config part of outputs of Data node to support AIPP, config all of the "
                               "outputs of Data to support AIPP, or config none of them");

        if (nodeType == AIPP) {
          aippNodes.push_back(inAnchor->GetOwnerNode());
          continue;
        }
      }
    }

    std::unique_ptr<domi::AippOpParams> aippParams(new (std::nothrow) domi::AippOpParams());
    GE_CHECK_NOTNULL(aippParams);

    GE_IF_BOOL_EXEC(aippNodes.size() > 0, GE_CHK_STATUS(GetAippParams(aippParams, aippNodes[0]));
                    aippMode = (aippMode == domi::AippOpParams::undefined) ? aippParams->aipp_mode() : aippMode;
                    GE_CHK_BOOL_RET_STATUS(aippMode == aippParams->aipp_mode(), PARAM_INVALID,
                                           "The aipp_mode of all aipp_op must be the same"););

    GE_IF_BOOL_EXEC(aippNodes.size() > 1, for (decltype(aippNodes)::size_type i = 1; i < aippNodes.size(); i++) {
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

Status InsertNewOpUtil::GetAippParams(const std::unique_ptr<domi::AippOpParams> &aipp_params,
                                      const NodePtr &aipp_node) {
  GE_CHECK_NOTNULL(aipp_node);
  ge::GeAttrValue::NamedAttrs aipp_attr;
  const OpDescPtr tmpOpPtr = aipp_node->GetOpDesc();
  GE_CHECK_NOTNULL(tmpOpPtr);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetNamedAttrs(tmpOpPtr, ATTR_NAME_AIPP, aipp_attr), FAILED,
                         "Aipp node should contain param aipp!");
  GE_CHK_STATUS_RET(OpUtils::ConvertAippParams(aipp_attr, aipp_params.get()), "get aipp params failed");

  return SUCCESS;
}

Status InsertNewOpUtil::AddMultiShapeInputData(const ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (node->GetOpDesc()->GetType() != MULTISHAPE) {
      continue;
    }

    GE_CHK_BOOL_RET_STATUS(node->GetInDataNodes().size() == 1, FAILED,
                           "multi_shape node should follow one data node, but size of input edges is %zu",
                           node->GetInDataNodes().size());

    NodePtr dataNode = node->GetInDataNodes().at(0);
    GE_CHK_BOOL_RET_STATUS(dataNode->GetOpDesc()->GetType() == DATA, FAILED,
                           "multi_shape node should follow one data node, but previous node's type is %s",
                           dataNode->GetOpDesc()->GetType().c_str());

    OpDescPtr opDescPtrData = MakeShared<ge::OpDesc>(std::string("multi_shape_data"), DATA);
    if (opDescPtrData == nullptr) {
      return PARAM_INVALID;
    }

    const uint32_t shapeSize = 4;
    const uint32_t REALDIM_CNT = 4;

    vector<int64_t> inputShapeDim(4, 1);  // 4 dimensions: NCHW
    inputShapeDim[0] = shapeSize;

    GeShape inputShape(inputShapeDim);
    GeTensorDesc input_tensor(inputShape, FORMAT_NCHW, DT_UINT32);
    TensorUtils::SetReuseInput(input_tensor, false);
    TensorUtils::SetSize(input_tensor, shapeSize * sizeof(uint32_t));
    GE_CHK_STATUS_RET(opDescPtrData->AddInputDesc(input_tensor));

    GeShape outputShape(inputShapeDim);
    GeTensorDesc output_tensor(outputShape, FORMAT_NCHW, DT_UINT32);
    TensorUtils::SetReuseInput(output_tensor, false);
    TensorUtils::SetSize(output_tensor, shapeSize * sizeof(uint32_t));
    TensorUtils::SetRealDimCnt(output_tensor, REALDIM_CNT);

    GE_CHK_STATUS_RET(opDescPtrData->AddOutputDesc(output_tensor), "AddOutputDesc failed!");

    NodePtr shapeDataNodePtr = graph->AddNode(opDescPtrData);
    GE_CHECK_NOTNULL(shapeDataNodePtr);
    GE_CHK_STATUS_RET(GraphUtils::AddEdge(shapeDataNodePtr->GetOutDataAnchor(0), node->GetInDataAnchor(1)),
                      "Add Anchor anchor between shape data and multi_shape failed!");
  }

  return SUCCESS;
}
}  // namespace ge
