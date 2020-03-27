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

#include "graph/preprocess/insert_op/base_insert_op.h"

#include <utility>
#include <vector>

#include "common/ge/ge_util.h"
#include "common/math/math_util.h"
#include "common/op/attr_value_util.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "common/util.h"
#include "external/graph/operator.h"
#include "external/graph/operator_factory.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
static const char *const kAippConfigPath = "aipp_config_route";
static const uint32_t kImageRatioYuv420SpU8Mul = 3;
static const uint32_t kImageRatioYuv420SpU8Div = 2;
static const uint32_t kImageRatioXrgb8888U8 = 4;
static const uint32_t kImageRatioRgb888U8 = 3;

Status InsertOpBase::InsertAippToGraph(ComputeGraphPtr &graph, std::string &aipp_config_path,
                                       ge::NodePtr &inserted_aipp_node) {
  GE_CHECK_NOTNULL(graph);
  NodePtr target_input = nullptr;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  GE_CHK_STATUS_RET(this->GetTargetPosition(graph, target_input, target_edges), "Get data nodes position failed");
  OpDescPtr op_desc = ge::MakeShared<OpDesc>("", "");
  if (op_desc == nullptr) {
    return FAILED;
  }
  GE_CHK_STATUS_RET(this->GenerateOpDesc(op_desc), "Generate aipp node opdesc failed");
  ge::GeAttrValue::NamedAttrs aipp_attr;
  GE_IF_BOOL_EXEC(!AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr),
                  GELOGW("InsertAippToGraph: GetNamedAttrs failed");
                  return FAILED)

  auto opdesc_src_data = target_input->GetOpDesc()->GetOutputDesc(0);
  if (opdesc_src_data.GetDataType() != DT_FLOAT) {
    GELOGW("The datatype of data node %s is not FP32", target_input->GetName().c_str());
    opdesc_src_data.SetDataType(DT_FLOAT);
  }

  static uint32_t op_idx = 0;
  // Does not involve multithreading.
  std::string current_name = std::string("aipp_node").append(std::to_string(op_idx++));
  auto aipp_op = ge::OperatorFactory::CreateOperator(current_name, "Aipp");
  GE_CHK_BOOL_RET_STATUS(!aipp_op.IsEmpty(), PARAM_INVALID, "Aipp is not registered");
  auto aipp_opdesc_ptr = ge::OpDescUtils::GetOpDescFromOperator(aipp_op);
  GE_CHECK_NOTNULL(aipp_opdesc_ptr);
  GE_IF_BOOL_EXEC(!AttrUtils::SetNamedAttrs(aipp_opdesc_ptr, ATTR_NAME_AIPP, aipp_attr),
                  GELOGE(FAILED, "SetNameAttrs failed");
                  return FAILED;)

  unique_ptr<domi::AippOpParams> aipp_params(new (std::nothrow) domi::AippOpParams());
  GE_CHECK_NOTNULL(aipp_params);
  GE_CHK_STATUS_RET(ge::OpUtils::ConvertAippParams(aipp_attr, aipp_params.get()), "Get aipp params failed")
  GE_CHK_STATUS_RET(aipp_opdesc_ptr->UpdateInputDesc(0, opdesc_src_data))

  if (aipp_params->aipp_mode() == domi::AippOpParams::dynamic) {
    Status ret = aipp_opdesc_ptr->UpdateInputDesc(1, opdesc_src_data);
    if (ret != SUCCESS) {
      return FAILED;
    }
  }
  GE_IF_BOOL_EXEC(!AttrUtils::SetStr(aipp_opdesc_ptr, kAippConfigPath, aipp_config_path),
                  GELOGW("SetStr kAippConfigPath failed");)
  GELOGI("Aipp config path is %s", aipp_config_path.c_str());

  // for data dump
  GE_IF_BOOL_EXEC(!AttrUtils::SetListStr(aipp_opdesc_ptr, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                         std::move(std::vector<std::string>())),
                  GELOGW("InsertAippToGraph: SetListStr failed");)

  NodePtr insert_op = graph->AddNode(aipp_opdesc_ptr);
  GE_CHECK_NOTNULL(insert_op);
  OutDataAnchorPtr target_input_out = target_input->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(target_input_out);
  InDataAnchorPtr insert_op_in = insert_op->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(insert_op_in);
  OutDataAnchorPtr insert_op_out = insert_op->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(insert_op_out);

  inserted_aipp_node = insert_op;
  if (target_edges.size() == 1) {
    OutDataAnchorPtr src_out = target_edges[0].first;
    InDataAnchorPtr dst_in = target_edges[0].second;
    GE_CHK_STATUS_RET(GraphUtils::InsertNodeBetweenDataAnchors(src_out, dst_in, insert_op))
    return SUCCESS;
  }
  for (auto &edge : target_edges) {
    OutDataAnchorPtr src_out = edge.first;
    GE_CHECK_NOTNULL(src_out);
    InDataAnchorPtr dst_in = edge.second;
    GE_CHK_STATUS_RET(src_out->Unlink(dst_in), "Unlink the anchor failed");
    GE_CHK_STATUS_RET(insert_op_out->LinkTo(dst_in), "Link the anchor failed");
  }
  GE_CHK_STATUS_RET(target_input_out->LinkTo(insert_op_in), "Link the anchor failed");
  return SUCCESS;
}

uint32_t InsertOpBase::AdjustDataSize(const GeTensorDesc &input_desc, unique_ptr<domi::AippOpParams> &aipp_params) {
  GE_CHECK_NOTNULL(aipp_params);
  if (aipp_params->aipp_mode() == domi::AippOpParams::static_) {
    uint32_t size = input_desc.GetShape().GetDim(NCHW_DIM_N);
    const uint32_t h = (input_desc.GetFormat() == ge::FORMAT_NHWC) ? NHWC_DIM_H : NCHW_DIM_H;
    const uint32_t w = (input_desc.GetFormat() == ge::FORMAT_NHWC) ? NHWC_DIM_W : NCHW_DIM_W;
    const uint32_t shape_h =
        aipp_params->src_image_size_h() ? aipp_params->src_image_size_h() : input_desc.GetShape().GetDim(h);
    FMK_UINT32_MULCHECK(size, shape_h);
    size *= shape_h;
    const uint32_t shape_w =
        aipp_params->src_image_size_w() ? aipp_params->src_image_size_w() : input_desc.GetShape().GetDim(w);
    FMK_UINT32_MULCHECK(size, shape_w);
    size *= shape_w;
    if (aipp_params->input_format() == domi::AippOpParams::YUV420SP_U8) {
      FMK_UINT32_MULCHECK((size / kImageRatioYuv420SpU8Div), kImageRatioYuv420SpU8Mul);
      size = size / kImageRatioYuv420SpU8Div * kImageRatioYuv420SpU8Mul;  // avoid use float
    } else if (aipp_params->input_format() == domi::AippOpParams::XRGB8888_U8) {
      FMK_UINT32_MULCHECK(size, kImageRatioXrgb8888U8);
      size *= kImageRatioXrgb8888U8;
    } else if (aipp_params->input_format() == domi::AippOpParams::RGB888_U8) {
      FMK_UINT32_MULCHECK(size, kImageRatioRgb888U8);
      size *= kImageRatioRgb888U8;
    }
    return size;
  } else {
    return aipp_params->max_src_image_size();
  }
}

Status InsertOpBase::InsertOpToGraph(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  NodePtr target_input = nullptr;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;
  GE_CHK_STATUS_RET(this->GetTargetPosition(graph, target_input, target_edges), "Get nodes position failed");

  // insertOp
  OpDescPtr op_desc = MakeShared<OpDesc>("", "");
  if (op_desc == nullptr) {
    return FAILED;
  }
  GE_CHK_STATUS_RET(this->GenerateOpDesc(op_desc), "Generate aipp node failed");
  NodePtr insert_op = graph->AddNode(op_desc);
  GE_CHECK_NOTNULL(insert_op);
  OutDataAnchorPtr target_input_out = target_input->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(target_input_out);
  InDataAnchorPtr insert_op_in = insert_op->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(insert_op_in);
  OutDataAnchorPtr insert_op_out = insert_op->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(insert_op_out);

  if (target_edges.size() == 1) {
    OutDataAnchorPtr src_out = target_edges[0].first;
    InDataAnchorPtr dst_in = target_edges[0].second;
    GE_CHK_STATUS_RET(GraphUtils::InsertNodeBetweenDataAnchors(src_out, dst_in, insert_op),
                      "InsertNodeBetweenDataAnchors failed");

    return SUCCESS;
  }

  for (auto &edge : target_edges) {
    OutDataAnchorPtr src_out = edge.first;
    GE_CHECK_NOTNULL(src_out);
    InDataAnchorPtr dst_in = edge.second;

    GE_CHK_STATUS_RET(src_out->Unlink(dst_in), "Unlink the anchor failed");

    GE_CHK_STATUS_RET(insert_op_out->LinkTo(dst_in), "Link the anchor failed");
  }

  GE_CHK_STATUS_RET(target_input_out->LinkTo(insert_op_in), "Link the anchor failed");

  return SUCCESS;
}

Status InsertOpBase::GetInputNode(ComputeGraphPtr graph, NodePtr &target_input, uint32_t rank) {
  GE_CHECK_NOTNULL(graph);
  std::vector<NodePtr> input_nodes;

  for (ge::NodePtr &node : graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);

    ge::OpDescPtr op = node->GetOpDesc();
    GE_CHECK_NOTNULL(op);

    if (op->GetType() == DATA_TYPE) {
      GE_CHK_BOOL_RET_STATUS(node->GetOutDataNodes().size() > 0, FAILED, "Data node %s has no output",
                             node->GetName().c_str());
      input_nodes.push_back(node);
    }
  }

  GE_CHK_BOOL_RET_STATUS(rank < input_nodes.size(), PARAM_INVALID,
                         "Get intput of index %d failed, There is %zu input nodes", rank, input_nodes.size());

  target_input = input_nodes[rank];

  return SUCCESS;
}
}  // namespace ge
