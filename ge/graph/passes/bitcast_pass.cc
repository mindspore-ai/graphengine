/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "graph/passes/bitcast_pass.h"

#include <vector>
#include <string>
#include "common/ge/ge_util.h"
#include "graph/utils/type_utils.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
namespace {
const char *const kAttrNameType = "type";
}  // namespace

Status BitcastPass::Run(NodePtr &node) {
  GELOGD("Bitcast running");
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "Param [node] must not be null.");
    return PARAM_INVALID;
  }

  if (node->GetType() != BITCAST) {
    return SUCCESS;
  }

  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return PARAM_INVALID;
  }
  ge::DataType dst_data_type;
  if (CheckDstDataCheckOutputShapeType(op_desc, dst_data_type) != SUCCESS) {
    return PARAM_INVALID;
  }

  if ((op_desc, dst_data_type) != SUCCESS) {
    return PARAM_INVALID;
  }

  return IsolateAndDeleteNode(node, {0});
}

Status BitcastPass::CheckDstDataType(const OpDescPtr op_desc, ge::DataType &dst_data_type) {

  if (!ge::AttrUtils::GetDataType(op_desc, kAttrNameType, dst_data_type)) {
    GELOGE(PARAM_INVALID, "Node failed to get attribute type.");
    return PARAM_INVALID;
  }
  if (dst_data_type >= ge::DT_UNDEFINED) {
    GELOGE(PARAM_INVALID, "dst_data_type[%s] is not valid.", 
           TypeUtils::DataTypeToSerialString(dst_data_type).c_str());
    return PARAM_INVALID;
  }

  if (op_desc->GetOutputDescPtr(0) == nullptr) {
    GELOGE(PARAM_INVALID, "Bitcast node outputDesc is null.");
    return PARAM_INVALID;
  }
  if (op_desc->GetOutputDescPtr(0)->GetDataType() != dst_data_type) {
    GELOGE(PARAM_INVALID, "dst_data_type[%s] is not equal to output_data_type[%s].",
           TypeUtils::DataTypeToSerialString(dst_data_type).c_str(),
           TypeUtils::DataTypeToSerialString(op_desc->GetOutputDescPtr(0)->GetDataType()).c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status BitcastPass::CheckOutputShape(const OpDescPtr op_desc, const ge::DataType dst_data_type) {
  const GeTensorDescPtr &input_tensor_desc = op_desc->MutableInputDesc(0);
  const GeTensorDescPtr &output_tensor_desc = op_desc->MutableOutputDesc(0);
  if (input_tensor_desc == nullptr) {
    GELOGE(PARAM_INVALID, "input_tensor_desc must not be null.");
    return PARAM_INVALID;
  }

  // get origin data_type and shape
  ge::DataType ori_data_type = input_tensor_desc->GetDataType();
  if (ori_data_type >= ge::DT_UNDEFINED) {
    GELOGE(PARAM_INVALID, "ori_data_type[%s] is not valid.", 
           TypeUtils::DataTypeToSerialString(ori_data_type).c_str());
    return PARAM_INVALID;
  }

  if (ori_data_type == dst_data_type) {
    GELOGW("Origin data type is equal to dest data type.");
    return SUCCESS;
  }

  BitcastPass::kVecInt64 dim_vec(input_tensor_desc->GetShape().GetDims());
  if (CalcAndUpdateShape(dim_vec, ori_data_type, dst_data_type) != SUCCESS) {
    GELOGE(PARAM_INVALID, "CalcAndUpdateShape failed.");
    return PARAM_INVALID;
  }

  if (dim_vec != output_tensor_desc->GetShape().GetDims()) {
    GELOGE(PARAM_INVALID, "out_put_shape is different from expectations.");
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status BitcastPass::CalcAndUpdateShape(BitcastPass::kVecInt64 &dim_vec, ge::DataType ori_data_type,
                                       ge::DataType dst_data_type) {
  if (dim_vec.size() == 0) {
    GELOGE(PARAM_INVALID, "Pre node shape size is zero.");
    return PARAM_INVALID;
  }
  int64_t ori_data_size = GetSizeByDataType(ori_data_type);
  int64_t dst_data_size = GetSizeByDataType(dst_data_type);

  if (ori_data_size == dst_data_size) {
    return SUCCESS;
  } else if (ori_data_size > dst_data_size) {
    if (ori_data_size % dst_data_size != 0) {
      GELOGE(PARAM_INVALID, "ori_data_size is not divisible by dst_data_size.");
      return PARAM_INVALID;
    }
    dim_vec.push_back(ori_data_size / dst_data_size);
    return SUCCESS;
  } else {
    if (dst_data_size % ori_data_size != 0) {
      GELOGE(PARAM_INVALID, "dst_data_size is not divisible by ori_data_size.");
      return PARAM_INVALID;
    }

    if (dim_vec[dim_vec.size() - 1] != (dst_data_size / ori_data_size)) {
      GELOGE(PARAM_INVALID, "The last dim is not equal to dst_data_size / ori_data_size.");
      return PARAM_INVALID;
    }
    dim_vec.pop_back();
  }
  return SUCCESS;
}

} // namespace ge
