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

namespace ge {
Status HcomOmeUtil::GetHcclDataType(const ge::ConstOpDescPtr &op_desc,
                                    std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (CheckKernelHcclInfo(op_desc, kernel_hccl_infos) != SUCCESS) {
    GELOGE(PARAM_INVALID, "HcomOmeUtil:: the number of GETaskKernelHcclInfo is invalid.");
    return PARAM_INVALID;
  }
  GELOGI("GetHcclDataType start, node[%s], opType[%s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (op_desc->GetType() == HVDWAIT) {
    return SUCCESS;
  }
  ge::DataType src_data_type = ge::DT_FLOAT;
  for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
    if (op_desc->GetType() == HCOMRECEIVE) {
      bool ret = ge::AttrUtils::GetDataType(op_desc, HCOM_ATTR_DATA_TYPE, src_data_type);
      if (ret == false) {
        GELOGE(PARAM_INVALID, "op:HcomReceive, op desc no attr: dtype.");
        return PARAM_INVALID;
      }
    } else {
      auto input_desc_ptr = op_desc->GetInputDescPtr(i);
      GE_CHECK_NOTNULL(input_desc_ptr);
      src_data_type = input_desc_ptr->GetDataType();
    }

    auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
    if (iter == kConstOpHcclDataType.end()) {
      GELOGE(PARAM_INVALID,
             "HcomOmeUtil::  Node: %s Optype: %s HcomDataType cann't support! Current Davinci Data Type : %s",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             ge::TypeUtils::DataTypeToSerialString(src_data_type).c_str());
      return PARAM_INVALID;
    }

    kernel_hccl_infos[i].dataType = iter->second;
  }
  return SUCCESS;
}

Status HcomOmeUtil::GetHcclTypeSize(HcclDataType data_type, int32_t &size) {
  auto iter = kConstOpHcclDataTypeSize.find(data_type);
  GE_CHK_BOOL_EXEC(iter != kConstOpHcclDataTypeSize.end(), return PARAM_INVALID,
                   "HcomOmeUtil::HcomDataTypeSize , No DataTypeSize!");

  size = iter->second;
  return SUCCESS;
}

Status HcomOmeUtil::GetHcomCount(const ge::ConstOpDescPtr &op_desc, HcclDataType data_type, bool is_allgather,
                                 int &count) {
  GE_CHECK_NOTNULL(op_desc);
  if (!IsHCOMOp(op_desc->GetType())) {
    GELOGE(PARAM_INVALID, "HcomOmeUtil:: operator is not Hcom operator.");
    return PARAM_INVALID;
  }
  int64_t total_size = 0;
  int64_t align_size = 512;
  int32_t size = 0;
  GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclTypeSize(data_type, size), "GetHcomCount: GetHcclTypeSize fail!");
  if (op_desc->GetType() == HCOMRECEIVE) {
    for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
      int64_t output_size = 0;
      GE_CHECK_NOTNULL(op_desc->GetOutputDescPtr(i));
      GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetOutputDescPtr(i), output_size),
                        "Get size from TensorDesc failed, op: %s, output index: %zu.", op_desc->GetName().c_str(), i);
      output_size = (output_size + align_size - 1) / align_size * align_size;
      total_size += output_size;
    }
  } else {
    for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
      int64_t input_size = 0;
      int64_t block_size = 0;
      GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(i));
      GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetInputDescPtr(i), input_size),
                        "get size from TensorDesc failed, op : %s, input index : %zu", op_desc->GetName().c_str(), i);
      // dynamic shape hccl op get size from output tensor desc
      if (op_desc->HasAttr(ATTR_NAME_IS_UNKNOWN_SHAPE)) {
        GE_CHECK_NOTNULL(op_desc->GetOutputDescPtr(i));
        GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetOutputDescPtr(i), input_size),
                          "get size from TensorDesc failed, op : %s, input index : %zu", op_desc->GetName().c_str(), i);
      }

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
      GELOGD("hcom util node %s inputsize %ld, shapesize %ld, datasize %d.",
             op_desc->GetName().c_str(), input_size, shape_size, size);
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

Status HcomOmeUtil::GetHorovodCount(const ge::ConstOpDescPtr &op_desc,
                                    std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (!IsHorovodOp(op_desc->GetType())) {
    GELOGE(PARAM_INVALID, "HcomOmeUtil:: operator is not Horovod operator.");
    return PARAM_INVALID;
  }
  int64_t align_size = 512;
  int32_t size = 0;
  for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclTypeSize(static_cast<HcclDataType>(kernel_hccl_infos[i].dataType), size),
                      "GetHorovodCount: GetHcclTypeSize fail!");
    int64_t input_size = 0;
    int64_t block_size = 0;
    GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(i));
    GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetInputDescPtr(i), input_size),
                      "get size from TensorDesc failed, op : %s, input index : %zu", op_desc->GetName().c_str(), i);

    int64_t shape_size = op_desc->GetInputDescPtr(i)->GetShape().GetShapeSize();
    GE_CHK_STATUS_RET(ge::CheckInt64Int32MulOverflow(shape_size, size),
                      "Product of shape size and size beyond INT64_MAX");
    if (kernel_hccl_infos[0].hccl_type == HVDCALLBACKALLGATHER) {
      block_size = shape_size * size;
    } else {
      block_size = (input_size + align_size - 1) / align_size * align_size;
    }

    GE_CHK_BOOL_RET_STATUS(size != 0, PARAM_INVALID, "Size is zero");
    GE_CHK_BOOL_EXEC(block_size % size == 0, return PARAM_INVALID, "block_size:%ld is not divisiable by size:%d.",
                     block_size, size);
    kernel_hccl_infos[i].count = static_cast<int>(block_size / size);
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcclCount(const ge::ConstOpDescPtr &op_desc,
                                 std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  Status ret;
  ret = CheckKernelHcclInfo(op_desc, kernel_hccl_infos);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "HcomOmeUtil:: the number of GETaskKernelHcclInfo is invalid.");
    return PARAM_INVALID;
  }
  GELOGI("GetHcclCount start, node[%s], opType[%s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (IsHCOMOp(op_desc->GetType())) {
    int32_t count = 0;
    ret = GetHcomCount(op_desc, static_cast<HcclDataType>(kernel_hccl_infos[0].dataType),
                       kernel_hccl_infos[0].hccl_type == HCOMALLGATHER, count);
    if (ret != SUCCESS) {
      GELOGE(ret, "HcomOmeUtil:: Node: %s Optype: %s get the Hcom operator hccl count fail.",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
    kernel_hccl_infos[0].count = count;
  }

  if (IsHorovodOp(op_desc->GetType())) {
    ret = GetHorovodCount(op_desc, kernel_hccl_infos);
    if (ret != SUCCESS) {
      GELOGE(PARAM_INVALID, "HcomOmeUtil:: Node: %s Optype: %s get the Horovod hccl operator count fail.",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcclOperationType(const ge::ConstOpDescPtr &op_desc, HcclReduceOp &op_type) {
  GE_CHECK_NOTNULL(op_desc);

  if (IsHCOMOp(op_desc->GetType())) {
    std::string hcom_op_type;
    GE_CHK_BOOL_EXEC(ge::AttrUtils::GetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, hcom_op_type), return PARAM_INVALID,
                     "HcomOmeUtil:: Node: %s Optype: %s Get HCOM_ATTR_REDUCE_TYPE fail, not support!",
                     op_desc->GetName().c_str(), op_desc->GetType().c_str());

    if (hcom_op_type == "min") {
      op_type = HCCL_REDUCE_MIN;
    } else if (hcom_op_type == "max") {
      op_type = HCCL_REDUCE_MAX;
    } else if (hcom_op_type == "prod") {
      op_type = HCCL_REDUCE_PROD;
    } else if (hcom_op_type == "sum") {
      op_type = HCCL_REDUCE_SUM;
    } else {
      GELOGE(PARAM_INVALID, "HcomOmeUtil::Get HCOM_ATTR_REDUCE_TYPE fail, [%s] not support!", hcom_op_type.c_str());
      return PARAM_INVALID;
    }
  }

  if (IsHorovodOp(op_desc->GetType())) {
    int64_t horovod_op_type;
    GE_CHK_BOOL_EXEC(ge::AttrUtils::GetInt(op_desc, ATTR_HOROVOD_ATTR_REDUCE_TYPE, horovod_op_type),
                     return PARAM_INVALID,
                     "HcomOmeUtil:: Node: %s Optype: %s Get ATTR_HOROVOD_ATTR_REDUCE_TYPE fail, not support!",
                     op_desc->GetName().c_str(), op_desc->GetType().c_str());

    auto iter = kHorovodRedOpToHcclRedOp.find(static_cast<HorovodReduceOp>(horovod_op_type));
    if (iter == kHorovodRedOpToHcclRedOp.end()) {
      GELOGE(PARAM_INVALID, "HcomOmeUtil::  Node: %s Optype: %s HcomOpType cann't support! Current HcomOpType : %ld",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), horovod_op_type);
      return PARAM_INVALID;
    }
    op_type = iter->second;
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcclRootId(const ge::ConstOpDescPtr &op_desc, int64_t &root_id) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::GetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id), return PARAM_INVALID,
                   "HcomOmeUtil::Node %s Optype: %s Get HCOM_ATTR_ROOT_INDEX fail, not support!",
                   op_desc->GetName().c_str(), op_desc->GetType().c_str());

  return SUCCESS;
}

Status HcomOmeUtil::GetAllRootId(const ge::ConstOpDescPtr &op_desc,
                                 std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetType() == HCOMBROADCAST ||
      op_desc->GetType() == HVDCALLBACKBROADCAST || op_desc->GetType() == HCOMREDUCE) {
    GELOGI("GetAllRootId Node[%s] opType[%s] get hccl rootId.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    int64_t root_id = 0;
    Status dmrt = GetHcclRootId(op_desc, root_id);
    if (dmrt != SUCCESS) {
      GELOGE(FAILED, "davinci_model: GetHcomRootId fail! domi error: %u", dmrt);
      return FAILED;
    }

    for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
      kernel_hccl_infos[i].rootId = root_id;
    }
  }
  return SUCCESS;
}

bool HcomOmeUtil::IsHCOMOp(const string &op_type) {
  return (op_type == HCOMALLREDUCE) || (op_type == HCOMALLGATHER) || (op_type == HCOMBROADCAST) ||
         (op_type == HCOMSEND) || (op_type == HCOMRECEIVE) || (op_type == HCOMREDUCESCATTER) || (op_type == HCOMREDUCE);
}

bool HcomOmeUtil::IsHorovodOp(const string &op_type) {
  return (op_type == HVDCALLBACKALLREDUCE) || (op_type == HVDCALLBACKALLGATHER) || (op_type == HVDCALLBACKBROADCAST) ||
         (op_type == HVDWAIT);
}

Status HcomOmeUtil::CheckKernelHcclInfo(const ge::ConstOpDescPtr &op_desc,
                                        std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (IsHCOMOp(op_desc->GetType()) && kernel_hccl_infos.size() != 1) {
    GELOGE(PARAM_INVALID, "HcomOmeUtil:: in Hcom scenario, the number of GETaskKernelHcclInfo is invalid.");
    return PARAM_INVALID;
  }

  if (IsHorovodOp(op_desc->GetType())) {
    if (op_desc->GetType() == HVDWAIT) {
      return SUCCESS;
    }
    if (kernel_hccl_infos.empty() || op_desc->GetInputsSize() != kernel_hccl_infos.size()) {
      GELOGE(PARAM_INVALID, "HcomOmeUtil:: in Horovod scenario, the number of GETaskKernelHcclInfo is invalid.");
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

void HcomOmeUtil::GetHcclType(const domi::TaskDef &task_def, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  auto hccl_def = task_def.kernel_hccl();
  std::string hccl_type = hccl_def.hccl_type();
  for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
    kernel_hccl_infos[i].hccl_type = hccl_type;
  }
}

Status HcomOmeUtil::GetHorovodInputs(const ge::ConstOpDescPtr &op_desc,
                                     std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (!IsHorovodOp(op_desc->GetType())) {
    return SUCCESS;
  }

  if (CheckKernelHcclInfo(op_desc, kernel_hccl_infos) != SUCCESS) {
    GELOGE(PARAM_INVALID, "HcomOmeUtil:: Node: %s Optype: %s the number of GETaskKernelHcclInfo is invalid.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  if (op_desc->GetType() == HVDWAIT) {
    return SUCCESS;
  }

  for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
    ConstGeTensorDescPtr input_desc = op_desc->GetInputDescPtr(i);
    GETaskKernelHcclInfo &kernel_hccl_info = kernel_hccl_infos.at(i);
    kernel_hccl_info.input_name = op_desc->GetInputNameByIndex(i);
    kernel_hccl_info.dims = input_desc->GetShape().GetDims();
  }
  return SUCCESS;
}
}  // namespace ge
