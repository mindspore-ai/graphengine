/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include "common/datatype_transfer.h"

#include <cstdint>
#include <map>
#include <utility>

#include "formats/utils/formats_trans_utils.h"
#include "common/fp16_t.h"
#include "common/plugin/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/utils/type_utils.h"
#include "securec.h"

namespace ge {
namespace formats {
namespace {
enum class DataTypeTransMode {
  kTransferWithDatatypeFloatToFloat16,
  kTransferWithDatatypeFloatToInt32,
  kTransferWithDatatypeFloat16ToFloat,
  kTransferWithDatatypeFloat16ToInt32,
  kTransferWithDatatypeInt32ToFloat,
  kTransferWithDatatypeInt32ToFloat16,
  kTransferWithDatatypeInt32ToUint8,
  kTransferWithDatatypeInt32ToInt8,
  kTransferWithDatatypeUint8ToFloat,
  kTransferWithDatatypeUint8ToInt32,
  kTransferWithDatatypeInt8ToFloat,
  kTransferWithDatatypeInt8ToInt32,
  kTransferWithDatatypeInt64ToInt32,
  kTransferWithDatatypeInt32ToInt64,
  kTransferWithDatatypeInt32ToDouble,
  kTransferWithDatatypeDoubleToInt32,
};

std::map<std::pair<DataType, DataType>, DataTypeTransMode> trans_mode_map {
  {std::pair<DataType, DataType>(DT_FLOAT, DT_FLOAT16), DataTypeTransMode::kTransferWithDatatypeFloatToFloat16},
  {std::pair<DataType, DataType>(DT_FLOAT, DT_INT32), DataTypeTransMode::kTransferWithDatatypeFloatToInt32},
  {std::pair<DataType, DataType>(DT_FLOAT16, DT_FLOAT), DataTypeTransMode::kTransferWithDatatypeFloat16ToFloat},
  {std::pair<DataType, DataType>(DT_FLOAT16, DT_INT32), DataTypeTransMode::kTransferWithDatatypeFloat16ToInt32},
  {std::pair<DataType, DataType>(DT_INT32, DT_FLOAT), DataTypeTransMode::kTransferWithDatatypeInt32ToFloat},
  {std::pair<DataType, DataType>(DT_INT32, DT_FLOAT16), DataTypeTransMode::kTransferWithDatatypeInt32ToFloat16},
  {std::pair<DataType, DataType>(DT_INT32, DT_UINT8), DataTypeTransMode::kTransferWithDatatypeInt32ToUint8},
  {std::pair<DataType, DataType>(DT_INT32, DT_INT8), DataTypeTransMode::kTransferWithDatatypeInt32ToInt8},
  {std::pair<DataType, DataType>(DT_UINT8, DT_FLOAT), DataTypeTransMode::kTransferWithDatatypeUint8ToFloat},
  {std::pair<DataType, DataType>(DT_UINT8, DT_INT32), DataTypeTransMode::kTransferWithDatatypeUint8ToInt32},
  {std::pair<DataType, DataType>(DT_INT8, DT_FLOAT), DataTypeTransMode::kTransferWithDatatypeInt8ToFloat},
  {std::pair<DataType, DataType>(DT_INT8, DT_INT32), DataTypeTransMode::kTransferWithDatatypeInt8ToInt32},
  {std::pair<DataType, DataType>(DT_INT64, DT_INT32), DataTypeTransMode::kTransferWithDatatypeInt64ToInt32},
  {std::pair<DataType, DataType>(DT_INT32, DT_INT64), DataTypeTransMode::kTransferWithDatatypeInt32ToInt64},
  {std::pair<DataType, DataType>(DT_INT32, DT_DOUBLE), DataTypeTransMode::kTransferWithDatatypeInt32ToDouble},
  {std::pair<DataType, DataType>(DT_DOUBLE, DT_INT32), DataTypeTransMode::kTransferWithDatatypeDoubleToInt32},
};

template <typename SrcT, typename DstT>
Status TransDataSrc2Dst(const CastArgs &args, uint8_t * const dst, const size_t data_size) {
  SrcT src_data;
  for (size_t idx = 0U; idx != data_size; idx++) {
    src_data = reinterpret_cast<const SrcT *>(args.data)[idx];
    reinterpret_cast<DstT *>(dst)[idx] = static_cast<DstT>(src_data);
  }
  return SUCCESS;
}

template <typename SrcT>
Status TransDataSrc2Fp16(const CastArgs &args, uint8_t *const dst, const size_t data_size) {
  fp16_t src_data;
  for (size_t idx = 0U; idx != data_size; idx++) {
    src_data = reinterpret_cast<const SrcT *>(args.data)[idx];
    reinterpret_cast<uint16_t *>(dst)[idx] = src_data.val;
  }
  return SUCCESS;
}

Status CastKernel(const CastArgs &args, uint8_t *const dst, const size_t data_size,
                  const DataTypeTransMode trans_mode) {
  static std::map<DataTypeTransMode, std::function<Status(const CastArgs &, uint8_t *, const size_t)>>
  transfer_handle = {
      {DataTypeTransMode::kTransferWithDatatypeFloatToFloat16, &TransDataSrc2Fp16<float>},
      {DataTypeTransMode::kTransferWithDatatypeFloatToInt32, &TransDataSrc2Dst<float, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeFloat16ToFloat, &TransDataSrc2Dst<fp16_t, float>},
      {DataTypeTransMode::kTransferWithDatatypeFloat16ToInt32, &TransDataSrc2Dst<fp16_t, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToFloat, &TransDataSrc2Dst<int32_t, float>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToFloat16, &TransDataSrc2Fp16<int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToUint8, &TransDataSrc2Dst<int32_t, uint8_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToInt8, &TransDataSrc2Dst<int32_t, int8_t>},
      {DataTypeTransMode::kTransferWithDatatypeUint8ToFloat, &TransDataSrc2Dst<uint8_t, float>},
      {DataTypeTransMode::kTransferWithDatatypeUint8ToInt32, &TransDataSrc2Dst<uint8_t, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt8ToFloat, &TransDataSrc2Dst<int8_t, float>},
      {DataTypeTransMode::kTransferWithDatatypeInt8ToInt32, &TransDataSrc2Dst<int8_t, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt64ToInt32, &TransDataSrc2Dst<int64_t, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToInt64, &TransDataSrc2Dst<int32_t, int64_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToDouble, &TransDataSrc2Dst<int32_t, double>},
      {DataTypeTransMode::kTransferWithDatatypeDoubleToInt32, &TransDataSrc2Dst<double, int32_t>},
  };
  const std::map<DataTypeTransMode, std::function<Status(const CastArgs &, uint8_t *, const size_t)>>::const_iterator
  it = transfer_handle.find(trans_mode);
  if (it == transfer_handle.end()) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  } else {
    return (it->second)(args, dst, data_size);
  }
}
}  // namespace

Status DataTypeTransfer::TransDataType(const CastArgs &args, TransResult &result) {
  GELOGD("Begin trans data from %s to %s, data size %zu", TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
         TypeUtils::DataTypeToSerialString(args.dst_data_type).c_str(), args.src_data_size);
  const std::pair<DataType, DataType> trans_info(args.src_data_type, args.dst_data_type);
  const std::map<std::pair<DataType, DataType>, DataTypeTransMode>::const_iterator
  iter = trans_mode_map.find(trans_info);
  if (iter == trans_mode_map.end()) {
    const std::string error = "Failed to trans data from datatype " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.src_data_type)) + " to " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.dst_data_type)) + " , it is not supported.";
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_DATATYPE_INVALID, error.c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  const auto trans_mode = iter->second;

  const int32_t size = GetSizeByDataType(args.dst_data_type);
  if (size <= 0) {
    const std::string error = "Failed to calc size from data type" +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.dst_data_type)) + ", it is not supported.";
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_DATATYPE_INVALID, error.c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (args.src_data_size > (SIZE_MAX / static_cast<size_t>(size))) {
    const std::string error = "args.src_data_size" + FmtToStr(args.src_data_size) +
        " or data type size" + FmtToStr(size) + " is too big";
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_PARAM_INVALID, error.c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  const size_t total_size = args.src_data_size * static_cast<size_t>(size);
  result.length = total_size;
  if (total_size == 0U) {
    GELOGI("In TransDataType, total_size is zero, has no data.");
    return SUCCESS;
  }

  const std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[total_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Allocate][DSTMemory]Failed, memory for dst buf %zu, data size %zu",
           total_size, args.src_data_size);
    REPORT_CALL_ERROR("E19999", "Failed to allocate memory for dst buf %zu, data size %zu",
                      total_size, args.src_data_size);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  if (CastKernel(args, dst.get(), args.src_data_size, trans_mode) != SUCCESS) {
    const std::string error = "Failed to cast data from datatype " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.src_data_type)) + " to " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.dst_data_type)) + ", data size is " +
        FmtToStr(std::to_string(args.src_data_size));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_INTERNAL_ERROR, error.c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  result.data = dst;
  return SUCCESS;
}

bool DataTypeTransfer::DataTypeTransferExists(const CastArgs &args) {
  const std::pair<DataType, DataType> trans_info(args.src_data_type, args.dst_data_type);
  const std::map<std::pair<DataType, DataType>, DataTypeTransMode>::const_iterator
  iter = trans_mode_map.find(trans_info);
  return iter != trans_mode_map.cend();
}

bool IsTransDataTypeSupport(const CastArgs &args) {
  return DataTypeTransfer::DataTypeTransferExists(args);
}

Status TransTensorDataType(const CastArgs &args, TransResult &result) {
  if (!DataTypeTransfer::DataTypeTransferExists(args)) {
    const std::string error = "Failed to trans data from datatype " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.src_data_type)) + " to " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.dst_data_type));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_DATATYPE_INVALID, error.c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  if ((args.data == nullptr) && (args.src_data_size != 0UL)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param]Failed, input data is null "
           "or data size not equal to 0, src_data_size %ld", args.src_data_size);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  return DataTypeTransfer::TransDataType(args, result);
}
}  // namespace formats
}  // namespace ge
