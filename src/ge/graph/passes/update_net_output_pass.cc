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

#include "graph/passes/update_net_output_pass.h"
#include <map>
#include <string>
#include <vector>
#include "omg/omg_inner_types.h"
#include "common/util.h"
#include "common/formats/formats.h"
#include "common/formats/format_transfers/format_transfer_nhwc_nc1hwc0.h"
#include "common/formats/format_transfers/format_transfer_nchw_nc1hwc0.h"

namespace ge {
static std::map<std::string, ge::DataType> kOutputTypeStrToDataType = {
    {"FP32", ge::DT_FLOAT},    {"FP16", ge::DT_FLOAT16},  {"INT8", ge::DT_INT8},     {"INT16", ge::DT_INT16},
    {"UINT16", ge::DT_UINT16}, {"UINT8", ge::DT_UINT8},   {"INT32", ge::DT_INT32},   {"INT64", ge::DT_INT64},
    {"UINT32", ge::DT_UINT32}, {"UINT64", ge::DT_UINT64}, {"DOUBLE", ge::DT_DOUBLE},
};

static void SetNetoutputDataType(OpDescPtr &op_desc,
                                 uint32_t index,
                                 ge::DataType output_data_type) {
  // op_desc is judged not nullptr
  auto net_output_in_desc = op_desc->MutableInputDesc(index);
  if (net_output_in_desc != nullptr) {
    net_output_in_desc->SetDataType(output_data_type);
    net_output_in_desc->SetOriginDataType(output_data_type);
    GELOGI("Update input desc, datatype:%s,",
           TypeUtils::DataTypeToSerialString(op_desc->GetInputDesc(0).GetDataType()).c_str());
  }
  auto net_output_out_desc = op_desc->MutableOutputDesc(index);
  if (net_output_out_desc != nullptr) {
    net_output_out_desc->SetDataType(output_data_type);
    net_output_out_desc->SetOriginDataType(output_data_type);
    GELOGI("Update out desc, datatype:%s",
           TypeUtils::DataTypeToSerialString(op_desc->GetOutputDesc(0).GetDataType()).c_str());
  }
}

static Status SetNetoutputFormat(OpDescPtr op_desc, uint32_t index, ge::Format format) {
  // op_desc is judged not nullptr
  auto net_output_in_desc = op_desc->MutableInputDesc(index);
  GE_CHECK_NOTNULL(net_output_in_desc);
  ge::Format old_format = net_output_in_desc->GetFormat();
  bool support = ((old_format == FORMAT_NC1HWC0) ||
      (old_format == FORMAT_NCHW) ||
      (old_format == FORMAT_NHWC));
  if (!support) {
    GELOGE(INTERNAL_ERROR, "The node %s format [%s] is unsupported", op_desc->GetName().c_str(),
           TypeUtils::FormatToSerialString(old_format).c_str());
    return FAILED;
  }
  if (old_format == FORMAT_NC1HWC0) {
    GELOGI("No need to transfer format");
    return SUCCESS;
  }
  std::vector<int64_t> old_shape = net_output_in_desc->GetShape().GetDims();
  ge::DataType dt = net_output_in_desc->GetDataType();
  std::vector<int64_t> dst_shape_dims;
  if (old_format == FORMAT_NCHW) {
    formats::FormatTransferNchwNc1hwc0 transfer;
    if (transfer.TransShape(old_format, old_shape, dt, format, dst_shape_dims) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "TransShape failed");
      return FAILED;
    }
  }
  if (old_format == FORMAT_NHWC) {
    formats::FormatTransferNhwcNc1hwc0 transfer;
    if (transfer.TransShape(old_format, old_shape, dt, format, dst_shape_dims) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "TransShape failed");
      return FAILED;
    }
  }
  net_output_in_desc->SetShape(ge::GeShape(dst_shape_dims));
  net_output_in_desc->SetOriginShape(ge::GeShape(dst_shape_dims));
  net_output_in_desc->SetFormat(format);
  net_output_in_desc->SetOriginFormat(format);
  GELOGI("Update input desc, format:%s,",
         TypeUtils::FormatToSerialString(op_desc->GetInputDesc(0).GetFormat()).c_str());

  auto net_output_out_desc = op_desc->MutableOutputDesc(index);
  if (net_output_out_desc == nullptr) {
    GELOGW("The opdesc is nullptr");
    return FAILED;
  }
  net_output_out_desc->SetShape(ge::GeShape(dst_shape_dims));
  net_output_out_desc->SetOriginShape(ge::GeShape(dst_shape_dims));
  net_output_out_desc->SetFormat(format);
  net_output_out_desc->SetOriginFormat(format);
  GELOGI("Update out desc, format:%s",
         TypeUtils::FormatToSerialString(op_desc->GetOutputDesc(0).GetFormat()).c_str());
  return SUCCESS;
}

Status ReUpdateNetOutputPass::Run(ge::NodePtr &node) {
  GELOGD("ReUpdateNetOutputPass running");
  if (node == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return FAILED;
  }
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "op_desc is null.");
    return FAILED;
  }

  std::string op_type = op_desc->GetType();
  if (op_type != NETOUTPUT) {
    return SUCCESS;
  }
  GELOGD("NetOutput start ReUpdateNetOutputPass");
  bool is_set_output_type = false;
  ge::DataType output_data_type = ge::DT_FLOAT;
  std::string output_type = domi::GetContext().output_type;
  if (kOutputTypeStrToDataType.find(output_type) != kOutputTypeStrToDataType.end()) {
    output_data_type = kOutputTypeStrToDataType[output_type];
    is_set_output_type = true;
  } else {
    GELOGW("output_type [%s] set can not find", output_type.c_str());
    is_set_output_type = false;
  }

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    auto index = static_cast<uint32_t>(in_anchor->GetIdx());
    // Update datatype
    if (is_set_output_type) {
      SetNetoutputDataType(op_desc, index, output_data_type);
      continue;
    }
    // output_node is not set,check if is_output_adjust_hw_layout is set
    auto peer_out = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out);
    auto own_node = peer_out->GetOwnerNode();
    GE_CHECK_NOTNULL(own_node);
    OpDescPtr src_op_desc = own_node->GetOpDesc();
    GE_CHECK_NOTNULL(src_op_desc);
    bool set_fp16_nc1hwc0 = false;
    if (AttrUtils::GetBool(src_op_desc, "output_set_fp16_nc1hwc0", set_fp16_nc1hwc0)) {
      GELOGI("This output [%s] should be set FP16 and NC1HWC0", src_op_desc->GetName().c_str());
      if (set_fp16_nc1hwc0) {
        SetNetoutputDataType(op_desc, index, ge::DT_FLOAT16);
        if (SetNetoutputFormat(op_desc, index, FORMAT_NC1HWC0) != SUCCESS) {
          GELOGE(PARAM_INVALID, "SetNetoutputFormat failed");
          return FAILED;
        }
        // set the outputdesc originformat NC1HWC0, as partition insert placehold node format is based on originformat
        auto src_index = static_cast<uint32_t>(in_anchor->GetPeerOutAnchor()->GetIdx());
        auto src_output_desc = src_op_desc->MutableOutputDesc(src_index);
        if (src_output_desc == nullptr) {
          GELOGE(PARAM_INVALID, "src_output_desc is m=nullptr");
          return FAILED;
        }
        src_output_desc->SetOriginFormat(FORMAT_NC1HWC0);
      }
    }
  }
  GELOGD("node[%s] ReUpdateNetOutputPass done", op_type.c_str());
  return SUCCESS;
}
}  // namespace ge
