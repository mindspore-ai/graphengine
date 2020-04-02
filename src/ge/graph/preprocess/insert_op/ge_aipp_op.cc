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

#include "graph/preprocess/insert_op/ge_aipp_op.h"
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/util.h"
#include "graph/optimize/common/params.h"

#define SAVE_AIPP_ATTR(KEY, SAVE_TYPE)                                                       \
  do {                                                                                       \
    (void)aipp_attrs.SetAttr(#KEY, GeAttrValue::CreateFrom<SAVE_TYPE>(aipp_params_->KEY())); \
  } while (0)

#define SAVE_AIPP_ATTR_LIST(KEY, SAVE_TYPE)                                                     \
  do {                                                                                          \
    if (aipp_params_->KEY##_size() > 0) {                                                       \
      (void)aipp_attrs.SetAttr(#KEY, GeAttrValue::CreateFrom<SAVE_TYPE>(aipp_params_->KEY(0))); \
    }                                                                                           \
  } while (0)

namespace {
const int32_t kDefaultMatrixR0C0Yuv2Rgb = 298;
const int32_t kDefaultMatrixR0C1Yuv2Rgb = 0;
const int32_t kDefaultMatrixR0C2Yuv2Rgb = 409;
const int32_t kDefaultMatrixR1C0Yuv2Rgb = 298;
const int32_t kDefaultMatrixR1C1Yuv2Rgb = -100;
const int32_t kDefaultMatrixR1C2Yuv2Rgb = -208;
const int32_t kDefaultMatrixR2C0Yuv2Rgb = 298;
const int32_t kDefaultMatrixR2C1Yuv2Rgb = 516;
const int32_t kDefaultMatrixR2C2Yuv2Rgb = 0;
const int32_t kDefaultMatrixR0C0Rgb2Yuv = 66;
const int32_t kDefaultMatrixR0C1Rgb2Yuv = 129;
const int32_t kDefaultMatrixR0C2Rgb2Yuv = 25;
const int32_t kDefaultMatrixR1C0Rgb2Yuv = -38;
const int32_t kDefaultMatrixR1C1Rgb2Yuv = -74;
const int32_t kDefaultMatrixR1C2Rgb2Yuv = 112;
const int32_t kDefaultMatrixR2C0Rgb2Yuv = 112;
const int32_t kDefaultMatrixR2C1Rgb2Yuv = -94;
const int32_t kDefaultMatrixR2C2Rgb2Yuv = -18;
const int32_t kDefaultOutputBias0 = 16;
const int32_t kDefaultOutputBias1 = 128;
const int32_t kDefaultOutputBias2 = 128;
const int32_t kDefaultInputBias0 = 16;
const int32_t kDefaultInputBias1 = 128;
const int32_t kDefaultInputBias2 = 128;
const float kDefaultVarReciChn = 1.0;
}  // namespace

namespace ge {
namespace {
const std::set<std::string> kInsertAippExceptOp = {SHAPE, SSDPRIORBOX};
}

Status AippOp::Init(domi::AippOpParams *aipp_params) {
  aipp_params_ = new (std::nothrow) domi::AippOpParams();
  if (aipp_params_ == nullptr) {
    return FAILED;
  }
  aipp_params_->CopyFrom(*aipp_params);
  return SUCCESS;
}

AippOp::~AippOp() {
  if (aipp_params_ != nullptr) {
    delete aipp_params_;
    aipp_params_ = nullptr;
  }
}

domi::AippOpParams::AippMode AippOp::GetAippMode() {
  if (aipp_params_ == nullptr) {
    return domi::AippOpParams::undefined;
  }
  return aipp_params_->aipp_mode();
}

Status AippOp::GetTargetPosition(ComputeGraphPtr graph, NodePtr &target_input,
                                 std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &target_edges) {
  GE_CHECK_NOTNULL(graph);
  target_input = nullptr;
  target_edges.clear();

  GE_CHECK_NOTNULL(aipp_params_);
  const uint32_t related_input_rank = aipp_params_->related_input_rank();
  GE_CHK_STATUS_RET(GetInputNode(graph, target_input, related_input_rank), "get target input node failed");

  const bool is_edge_configed = aipp_params_->input_edge_idx_size() > 0;

  GE_CHK_BOOL_RET_STATUS(!is_edge_configed || aipp_params_->input_edge_idx(0) < target_input->GetOutDataNodes().size(),
                         PARAM_INVALID, "input_edge_idx %u should smaller than out edge size of target input %zu ",
                         aipp_params_->input_edge_idx(0), target_input->GetOutDataNodes().size());

  uint32_t i = 0;
  for (OutDataAnchorPtr &src_out : target_input->GetAllOutDataAnchors()) {
    GE_RETURN_WITH_LOG_IF_FALSE(src_out != nullptr, "OutDataAnchor is null.");
    auto vistor = src_out->GetPeerInDataAnchors();
    for (auto it = vistor.begin(); it != vistor.end(); ++it, ++i) {
      InDataAnchorPtr dst_in = *it;
      GE_RETURN_WITH_LOG_IF_FALSE(dst_in != nullptr, "InDataAnchor is null.");

      if ((is_edge_configed && i == aipp_params_->input_edge_idx(0)) || !is_edge_configed) {
        NodePtr dst_node = dst_in->GetOwnerNode();
        OpDescPtr dst_op = dst_node->GetOpDesc();
        if (kInsertAippExceptOp.find(dst_op->GetType()) == kInsertAippExceptOp.end()) {
          target_edges.push_back(make_pair(src_out, dst_in));
          continue;
        }

        GE_CHK_BOOL_RET_STATUS(!is_edge_configed, PARAM_INVALID, "index %d of input node is %s node, can not do aipp",
                               aipp_params_->input_edge_idx(0), dst_op->GetType().c_str());
      }
    }
  }

  GE_CHK_BOOL_RET_STATUS(target_edges.size() > 0, FAILED, "get target edges failed");

  return SUCCESS;
}

Status AippOp::SetDefaultParams() {
  GE_CHECK_NOTNULL(aipp_params_);
  const domi::AippOpParams::AippMode aipp_mode = aipp_params_->aipp_mode();
  if (aipp_mode == domi::AippOpParams::static_) {
    if (aipp_params_->csc_switch()) {
      SetCscDefaultValue();
    }

    SetDtcDefaultValue();

    GELOGI("parse aipp params:input_format:%s, csc_switch:%d.",
           domi::AippOpParams::InputFormat_Name(aipp_params_->input_format()).c_str(), aipp_params_->csc_switch());

    GELOGI("parse aipp params:mean_chn_0:%d, mean_chn_1:%d, mean_chn_2:%d.", aipp_params_->mean_chn_0(),
           aipp_params_->mean_chn_1(), aipp_params_->mean_chn_2());

    GELOGI("parse aipp params:min_chn_0:%f, min_chn_1:%f, min_chn_2:%f.", aipp_params_->min_chn_0(),
           aipp_params_->min_chn_1(), aipp_params_->min_chn_2());

    GE_IF_BOOL_EXEC(!aipp_params_->crop(), aipp_params_->set_load_start_pos_h(0); aipp_params_->set_load_start_pos_w(0);
                    aipp_params_->set_crop_size_h(0); aipp_params_->set_crop_size_w(0););

    GE_IF_BOOL_EXEC(!aipp_params_->resize(), aipp_params_->set_resize_output_h(0);
                    aipp_params_->set_resize_output_w(0););

    GE_IF_BOOL_EXEC(!aipp_params_->padding(), aipp_params_->set_left_padding_size(0);
                    aipp_params_->set_right_padding_size(0); aipp_params_->set_top_padding_size(0);
                    aipp_params_->set_bottom_padding_size(0););
  }

  return SUCCESS;
}

Status AippOp::ValidateParams() {
  GE_CHECK_NOTNULL(aipp_params_);
  GE_CHK_BOOL_RET_STATUS(aipp_params_->aipp_mode() != domi::AippOpParams::undefined, PARAM_INVALID,
                         "when insert AIPP op, aipp_mode must be configured as static or dynamic ");

  GE_CHK_BOOL_RET_STATUS(aipp_params_->var_reci_chn_0_size() <= 1, PARAM_INVALID,
                         "The parameter var_reci_chn_0 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->var_reci_chn_1_size() <= 1, PARAM_INVALID,
                         "The parameter var_reci_chn_1 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->var_reci_chn_2_size() <= 1, PARAM_INVALID,
                         "The parameter var_reci_chn_2 can not be configed repeatedly");

  GE_CHK_BOOL_RET_STATUS(aipp_params_->matrix_r0c0_size() <= 1, PARAM_INVALID,
                         "The parameter matrix_r0c0 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->matrix_r0c1_size() <= 1, PARAM_INVALID,
                         "The parameter matrix_r0c1 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->matrix_r0c2_size() <= 1, PARAM_INVALID,
                         "The parameter matrix_r0c2 can not be configed repeatedly");

  GE_CHK_BOOL_RET_STATUS(aipp_params_->matrix_r1c0_size() <= 1, PARAM_INVALID,
                         "The parameter matrix_r1c0 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->matrix_r1c1_size() <= 1, PARAM_INVALID,
                         "The parameter matrix_r1c1 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->matrix_r1c2_size() <= 1, PARAM_INVALID,
                         "The parameter matrix_r1c2 can not be configed repeatedly");

  GE_CHK_BOOL_RET_STATUS(aipp_params_->matrix_r2c0_size() <= 1, PARAM_INVALID,
                         "The parameter matrix_r2c0 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->matrix_r2c1_size() <= 1, PARAM_INVALID,
                         "The parameter matrix_r2c1 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->matrix_r2c2_size() <= 1, PARAM_INVALID,
                         "The parameter matrix_r2c2 can not be configed repeatedly");

  GE_CHK_BOOL_RET_STATUS(aipp_params_->output_bias_0_size() <= 1, PARAM_INVALID,
                         "The parameter output_bias_0 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->output_bias_1_size() <= 1, PARAM_INVALID,
                         "The parameter output_bias_1 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->output_bias_2_size() <= 1, PARAM_INVALID,
                         "The parameter output_bias_2 can not be configed repeatedly");

  GE_CHK_BOOL_RET_STATUS(aipp_params_->input_bias_0_size() <= 1, PARAM_INVALID,
                         "The parameter input_bias_0 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->input_bias_1_size() <= 1, PARAM_INVALID,
                         "The parameter input_bias_1 can not be configed repeatedly");
  GE_CHK_BOOL_RET_STATUS(aipp_params_->input_bias_2_size() <= 1, PARAM_INVALID,
                         "The parameter input_bias_2 can not be configed repeatedly");

  GE_CHK_BOOL_RET_STATUS(aipp_params_->input_edge_idx_size() <= 1, PARAM_INVALID,
                         "The parameter input_edge_idx can not be configed repeatedly");

  const domi::AippOpParams::AippMode aipp_mode = aipp_params_->aipp_mode();
  if (aipp_mode == domi::AippOpParams::dynamic) {
    GE_CHK_BOOL_RET_STATUS(aipp_params_->max_src_image_size() > 0, PARAM_INVALID,
                           "for dynamic AIPP params, max_src_image_size must greater than 0");
  } else {
    GE_CHK_BOOL_RET_STATUS(aipp_params_->input_format() != domi::AippOpParams::UNDEFINED, PARAM_INVALID,
                           "Input format of AIPP conf is undefined");

    GE_CHK_BOOL_RET_STATUS(aipp_params_->src_image_size_w() >= 0, PARAM_INVALID,
                           "src_image_size_w must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->src_image_size_h() >= 0, PARAM_INVALID,
                           "src_image_size_h must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->load_start_pos_w() >= 0, PARAM_INVALID,
                           "load_start_pos_w must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->load_start_pos_h() >= 0, PARAM_INVALID,
                           "load_start_pos_h must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->crop_size_w() >= 0, PARAM_INVALID,
                           "crop_size_w must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->resize_output_w() >= 0, PARAM_INVALID,
                           "resize_output_w must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->resize_output_h() >= 0, PARAM_INVALID,
                           "resize_output_h must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->left_padding_size() >= 0, PARAM_INVALID,
                           "left_padding_size must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->right_padding_size() >= 0, PARAM_INVALID,
                           "right_padding_size must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->top_padding_size() >= 0, PARAM_INVALID,
                           "top_padding_size must not be configed smaller than 0");
    GE_CHK_BOOL_RET_STATUS(aipp_params_->bottom_padding_size() >= 0, PARAM_INVALID,
                           "bottom_padding_size must not be configed smaller than 0");
  }

  return SUCCESS;
}

void AippOp::SetCscDefaultValue() {
  GE_CHECK_NOTNULL_JUST_RETURN(aipp_params_);
  if (aipp_params_->input_format() == domi::AippOpParams::YUV420SP_U8) {
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c0_size() > 0, aipp_params_->add_matrix_r0c0(kDefaultMatrixR2C0Yuv2Rgb));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c1_size() > 0, aipp_params_->add_matrix_r0c1(kDefaultMatrixR2C1Yuv2Rgb));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c2_size() > 0, aipp_params_->add_matrix_r0c2(kDefaultMatrixR2C2Yuv2Rgb));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c0_size() > 0, aipp_params_->add_matrix_r1c0(kDefaultMatrixR1C0Yuv2Rgb));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c1_size() > 0, aipp_params_->add_matrix_r1c1(kDefaultMatrixR1C1Yuv2Rgb));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c2_size() > 0, aipp_params_->add_matrix_r1c2(kDefaultMatrixR1C2Yuv2Rgb));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c0_size() > 0, aipp_params_->add_matrix_r2c0(kDefaultMatrixR0C0Yuv2Rgb));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c1_size() > 0, aipp_params_->add_matrix_r2c1(kDefaultMatrixR0C1Yuv2Rgb));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c2_size() > 0, aipp_params_->add_matrix_r2c2(kDefaultMatrixR0C2Yuv2Rgb));
  } else {
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c0_size() > 0, aipp_params_->add_matrix_r0c0(kDefaultMatrixR0C0Rgb2Yuv));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c1_size() > 0, aipp_params_->add_matrix_r0c1(kDefaultMatrixR0C1Rgb2Yuv));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c2_size() > 0, aipp_params_->add_matrix_r0c2(kDefaultMatrixR0C2Rgb2Yuv));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c0_size() > 0, aipp_params_->add_matrix_r1c0(kDefaultMatrixR1C0Rgb2Yuv));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c1_size() > 0, aipp_params_->add_matrix_r1c1(kDefaultMatrixR1C1Rgb2Yuv));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c2_size() > 0, aipp_params_->add_matrix_r1c2(kDefaultMatrixR1C2Rgb2Yuv));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c0_size() > 0, aipp_params_->add_matrix_r2c0(kDefaultMatrixR2C0Rgb2Yuv));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c1_size() > 0, aipp_params_->add_matrix_r2c1(kDefaultMatrixR2C1Rgb2Yuv));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c2_size() > 0, aipp_params_->add_matrix_r2c2(kDefaultMatrixR2C2Rgb2Yuv));
  }
  CHECK_FALSE_EXEC(aipp_params_->input_bias_0_size() > 0, aipp_params_->add_input_bias_0(kDefaultInputBias0));
  CHECK_FALSE_EXEC(aipp_params_->input_bias_1_size() > 0, aipp_params_->add_input_bias_1(kDefaultInputBias1));
  CHECK_FALSE_EXEC(aipp_params_->input_bias_2_size() > 0, aipp_params_->add_input_bias_2(kDefaultInputBias2));
  CHECK_FALSE_EXEC(aipp_params_->output_bias_0_size() > 0, aipp_params_->add_output_bias_0(kDefaultOutputBias0));
  CHECK_FALSE_EXEC(aipp_params_->output_bias_1_size() > 0, aipp_params_->add_output_bias_1(kDefaultOutputBias1));
  CHECK_FALSE_EXEC(aipp_params_->output_bias_2_size() > 0, aipp_params_->add_output_bias_2(kDefaultOutputBias2));
}

void AippOp::SetDtcDefaultValue() {
  GE_CHECK_NOTNULL_JUST_RETURN(aipp_params_);
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_0_size() > 0, aipp_params_->add_var_reci_chn_0(kDefaultVarReciChn));
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_1_size() > 0, aipp_params_->add_var_reci_chn_1(kDefaultVarReciChn));
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_2_size() > 0, aipp_params_->add_var_reci_chn_2(kDefaultVarReciChn));
}

Status AippOp::GenerateOpDesc(OpDescPtr op_desc) {
  GE_CHECK_NOTNULL(op_desc);

  static int op_idx = 0;
  op_desc->SetName(std::string("aipp_node").append(std::to_string(op_idx++)));
  op_desc->SetType(AIPP);

  // Add two InputDesc, add the second after the first one is added successfully.
  if ((op_desc->AddInputDesc(GeTensorDesc()) != GRAPH_SUCCESS) ||
      (op_desc->AddInputDesc(GeTensorDesc()) != GRAPH_SUCCESS)) {
    GELOGE(FAILED, "failed to add input desc");
    return FAILED;
  }

  if (op_desc->AddOutputDesc(GeTensorDesc()) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "add output desc failed.");
    return FAILED;
  }
  GeAttrValue::NamedAttrs aipp_attrs;
  ConvertParamToAttr(aipp_attrs);

  GE_IF_BOOL_EXEC(!AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attrs),
                  GELOGE(FAILED, "failed to set ATTR_NAME_AIPP");
                  return FAILED);

  return SUCCESS;
}

void AippOp::ConvertParamToAttr(GeAttrValue::NamedAttrs &aipp_attrs) {
  GE_CHECK_NOTNULL_JUST_RETURN(aipp_params_);
  SAVE_AIPP_ATTR(aipp_mode, GeAttrValue::INT);

  if (aipp_params_->aipp_mode() == domi::AippOpParams::static_) {
    SAVE_AIPP_ATTR(input_format, GeAttrValue::INT);
    SAVE_AIPP_ATTR(csc_switch, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(crop, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(resize, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(load_start_pos_w, GeAttrValue::INT);
    SAVE_AIPP_ATTR(load_start_pos_h, GeAttrValue::INT);
    SAVE_AIPP_ATTR(crop_size_w, GeAttrValue::INT);
    SAVE_AIPP_ATTR(crop_size_h, GeAttrValue::INT);
    SAVE_AIPP_ATTR(resize, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(resize_output_w, GeAttrValue::INT);
    SAVE_AIPP_ATTR(resize_output_h, GeAttrValue::INT);
    SAVE_AIPP_ATTR(padding, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(left_padding_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(right_padding_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(top_padding_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(bottom_padding_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(src_image_size_w, GeAttrValue::INT);
    SAVE_AIPP_ATTR(src_image_size_h, GeAttrValue::INT);
    SAVE_AIPP_ATTR(cpadding_value, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR(rbuv_swap_switch, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(ax_swap_switch, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(single_line_mode, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(mean_chn_0, GeAttrValue::INT);
    SAVE_AIPP_ATTR(mean_chn_1, GeAttrValue::INT);
    SAVE_AIPP_ATTR(mean_chn_2, GeAttrValue::INT);
    SAVE_AIPP_ATTR(min_chn_0, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR(min_chn_1, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR(min_chn_2, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR_LIST(var_reci_chn_0, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR_LIST(var_reci_chn_1, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR_LIST(var_reci_chn_2, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR_LIST(matrix_r0c0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r0c1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r0c2, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r1c0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r1c1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r1c2, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r2c0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r2c1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r2c2, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(output_bias_0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(output_bias_1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(output_bias_2, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(input_bias_0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(input_bias_1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(input_bias_2, GeAttrValue::INT);
  } else {
    SAVE_AIPP_ATTR(max_src_image_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(support_rotation, GeAttrValue::BOOL);
  }
}
}  // namespace ge
