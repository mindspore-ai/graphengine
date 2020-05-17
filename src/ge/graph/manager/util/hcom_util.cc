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

#include "graph/manager/util/hcom_util.h"

#include "common/debug/log.h"
#include "common/math/math_util.h"
#include "common/op/attr_value_util.h"
#include "common/op/ge_op_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

using domi::HCOM_ATTR_DATA_TYPE;
using domi::HCOM_ATTR_RANK_SIZE;
using domi::HCOM_ATTR_REDUCE_TYPE;
using domi::HCOM_ATTR_ROOT_RANK;
using domi::HCOM_ATTR_SHAPE;
using domi::HCOMRECEIVE;
using domi::HCOMREDUCESCATTER;

namespace ge {
Status HcomOmeUtil::GetHcomDataType(const ge::ConstOpDescPtr &op_desc, hcclDataType_t &data_type) {
  GE_CHECK_NOTNULL(op_desc);
  ge::DataType src_data_type = ge::DT_FLOAT;
  if (op_desc->GetType() == HCOMRECEIVE) {
    bool ret = ge::AttrUtils::GetDataType(op_desc, HCOM_ATTR_DATA_TYPE, src_data_type);
    if (ret == false) {
      GELOGE(PARAM_INVALID, "op:HcomReceive, op desc no attr: dtype.");
      return PARAM_INVALID;
    }
  } else {
    auto input_desc_ptr = op_desc->GetInputDescPtr(0);
    GE_CHECK_NOTNULL(input_desc_ptr);
    src_data_type = input_desc_ptr->GetDataType();
  }

  auto iter = kConstOpHcomDataType.find(static_cast<int64_t>(src_data_type));
  if (iter == kConstOpHcomDataType.end()) {
    GELOGE(PARAM_INVALID, "HcomOmeUtil:: HcomDataType cann't support! Current Davinci Data Type : %s",
           ge::TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    return PARAM_INVALID;
  }

  data_type = iter->second;
  return SUCCESS;
}

Status HcomOmeUtil::GetHcomTypeSize(hcclDataType_t data_type, int32_t &size) {
  auto iter = kConstOpHcomDataTypeSize.find(data_type);
  GE_CHK_BOOL_EXEC(iter != kConstOpHcomDataTypeSize.end(), return PARAM_INVALID,
                   "HcomOmeUtil::HcomDataTypeSize , No DataTypeSize!");

  size = iter->second;
  return SUCCESS;
}

Status HcomOmeUtil::GetHcomCount(const ge::ConstOpDescPtr &op_desc, hcclDataType_t data_type, bool is_allgather,
                                 int &count) {
  GE_CHECK_NOTNULL(op_desc);
  int64_t total_size = 0;
  int64_t align_size = 512;
  int32_t size = 0;
  GE_CHK_STATUS_RET(HcomOmeUtil::GetHcomTypeSize(data_type, size), "GetHcomCount: GetHcomTypeSize fail!");
  if (op_desc->GetType() == HCOMRECEIVE) {
    vector<int64_t> shape_dims;
    bool ret = ge::AttrUtils::GetListInt(op_desc, HCOM_ATTR_SHAPE, shape_dims);
    if (ret == false) {
      GELOGE(PARAM_INVALID, "op:HcomReceive, op desc no attr: shape.");
      return PARAM_INVALID;
    }
    ge::GeShape shape = ge::GeShape(shape_dims);
    int64_t input_size = shape.GetShapeSize() * size;
    total_size = (input_size + align_size - 1) / align_size * align_size;
  } else {
    for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
      int64_t input_size = 0;
      int64_t block_size = 0;
      GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(i));
      GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetInputDescPtr(i), input_size),
                        "get size from TensorDesc failed, op : %s, input index : %zu", op_desc->GetName().c_str(), i);

      GE_IF_BOOL_EXEC(
        op_desc->GetType() == HCOMREDUCESCATTER, int32_t rank_size = 0;
        GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetInt(op_desc, HCOM_ATTR_RANK_SIZE, rank_size), PARAM_INVALID,
                               "get HCOM_ATTR_RANK_SIZE failed");
        GE_CHK_BOOL_RET_STATUS(rank_size != 0, PARAM_INVALID, "rank size is zero");
        int64_t shape_size = op_desc->GetInputDescPtr(i)->GetShape().GetShapeSize(); GE_CHK_STATUS_RET(
          ge::CheckInt64Uint32MulOverflow(shape_size, size), "Product of shape size and size beyond INT64_MAX");
        block_size = (shape_size * size) / rank_size;
        GE_CHK_STATUS_RET(ge::CheckInt64AddOverflow(total_size, block_size), "Total size is beyond the INT64_MAX");
        total_size = total_size + block_size; continue;);

      int64_t shape_size = op_desc->GetInputDescPtr(i)->GetShape().GetShapeSize();
      GE_CHK_STATUS_RET(ge::CheckInt64Int32MulOverflow(shape_size, size),
                        "Product of shape size and size beyond INT64_MAX");
      GE_IF_BOOL_EXEC(is_allgather, block_size = shape_size * size;);
      GE_IF_BOOL_EXEC(!is_allgather, block_size = (input_size + align_size - 1) / align_size * align_size;);
      GE_CHK_STATUS_RET(ge::CheckInt64AddOverflow(total_size, block_size), "Total size is beyond the INT64_MAX");
      total_size = total_size + block_size;
    }
  }

  GE_CHK_BOOL_RET_STATUS(size != 0, PARAM_INVALID, "Size is zero");
  count = static_cast<int>(total_size / size);

  GE_CHK_BOOL_EXEC(total_size % size == 0, return PARAM_INVALID, "total_size:%ld is not divisiable by size:%d.",
                   total_size, size);

  return SUCCESS;
}

Status HcomOmeUtil::GetHcomOperationType(const ge::ConstOpDescPtr &op_desc, hcclRedOp_t &op_type) {
  GE_CHECK_NOTNULL(op_desc);

  std::string hcom_op_type;
  GE_CHK_BOOL_EXEC(ge::AttrUtils::GetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, hcom_op_type), return PARAM_INVALID,
                   "HcomOmeUtil::Get HCOM_ATTR_REDUCE_TYPE fail, not support!");

  if (hcom_op_type == "min") {
    op_type = HCCL_REP_OP_MIN;
  } else if (hcom_op_type == "max") {
    op_type = HCCL_REP_OP_MAX;
  } else if (hcom_op_type == "prod") {
    op_type = HCCL_REP_OP_PROD;
  } else if (hcom_op_type == "sum") {
    op_type = HCCL_REP_OP_SUM;
  } else {
    GELOGE(PARAM_INVALID, "HcomOmeUtil::Get HCOM_ATTR_REDUCE_TYPE fail, [%s] not support!", hcom_op_type.c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcomRootId(const ge::ConstOpDescPtr &op_desc, int64_t &root_id) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::GetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id), return PARAM_INVALID,
                   "HcomOmeUtil::Get HCOM_ATTR_ROOT_INDEX fail, not support!");

  return SUCCESS;
}
}  // namespace ge
