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

#include "hybrid/node_executor/aicpu/aicpu_ext_info.h"
#include "framework/common/util.h"
#include "framework/common/fmk_error_codes.h"
#include "framework/common/debug/log.h"

namespace ge {
namespace hybrid {
namespace {
// if dim count is not reach kMaxShapeDims(8), use INT64_MIN to mark dim end.
constexpr int64_t kDimEndFlag = INT64_MIN;
}

Status AicpuExtInfoHandler::Parse(const std::string &ext_info) {
  GELOGI("Node[%s] parse ext info start.", node_name_.c_str());
  if (ext_info.empty()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Node[%s] parse ext info failed as ext info is empty.",
           node_name_.c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  ext_info_len_ = ext_info.size();
  ext_info_.reset(new(std::nothrow)uint8_t[ext_info_len_]);
  GE_CHECK_NOTNULL(ext_info_);

  if (memcpy_s(ext_info_.get(), ext_info_len_, ext_info.c_str(), ext_info.size()) != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[%s] Failed to coy ext info", node_name_.c_str());
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }

  input_shape_and_type_.clear();
  output_shape_and_type_.clear();

  auto ext_info_data = ext_info_.get();
  size_t offset = 0;
  while (offset + sizeof(AicpuExtInfo) <= ext_info_len_) {
    auto aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(ext_info_data + offset);
    GELOGD("Ext infoType=%d, infoLen=%u.", aicpu_ext_info->infoType, aicpu_ext_info->infoLen);
    switch (aicpu_ext_info->infoType) {
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
        GE_CHK_STATUS_RET(ParseExtShapeType(aicpu_ext_info), "Parse ext shape type failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
        GE_CHK_STATUS_RET(ParseExtInputShape(aicpu_ext_info), "Parse ext input shape failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
        GE_CHK_STATUS_RET(ParseExtOutputShape(aicpu_ext_info), "Parse ext output shape failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO:
        GE_CHK_STATUS_RET(ParseExtSessionInfo(aicpu_ext_info), "Parse ext session info failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP:
        GE_CHK_STATUS_RET(ParseExtBitMap(aicpu_ext_info), "Parse ext bit map failed.");
        break;
      default:
        GELOGD("Node[%s] ignore infoType=%d, infoLen=%u.",
               node_name_.c_str(), aicpu_ext_info->infoType, aicpu_ext_info->infoLen);
        break;
    }
    offset += sizeof(AicpuExtInfo);
    offset += aicpu_ext_info->infoLen;
  }

  GE_CHK_BOOL_RET_STATUS(offset == ext_info_len_, ACL_ERROR_GE_PARAM_INVALID,
                         "Node[%s] ext_info format error, parse not reach end, offset=%zu, ext_info_len=%zu.",
                         node_name_.c_str(), offset, ext_info_len_);
  GELOGI("Node[%s] parse ext info end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtShapeType(AicpuExtInfo *aicpu_ext_info) {
  GE_CHK_BOOL_RET_STATUS(aicpu_ext_info->infoLen == sizeof(int32_t), ACL_ERROR_GE_PARAM_INVALID,
                         "Node[%s] parse ext shape type failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(int32_t), aicpu_ext_info->infoLen);

  auto type = reinterpret_cast<const int32_t *>(aicpu_ext_info->infoMsg);

  GE_CHK_BOOL_RET_STATUS(*type == unknown_type_, ACL_ERROR_GE_PARAM_INVALID,
                         "Node[%s] parse ext shape type failed as need %d but %d.",
                         node_name_.c_str(), unknown_type_, *type);
  GELOGI("Node[%s] parse ext shape type success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtInputShape(AicpuExtInfo *aicpu_ext_info) {
  auto need_len = input_num_ * sizeof(AicpuShapeAndType);
  GE_CHK_BOOL_RET_STATUS(aicpu_ext_info->infoLen == need_len, ACL_ERROR_GE_PARAM_INVALID,
                         "Node[%s] parse ext input shape failed as infoLen must be "
                         "input_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                         node_name_.c_str(), input_num_, sizeof(AicpuShapeAndType), aicpu_ext_info->infoLen);

  auto input = reinterpret_cast<AicpuShapeAndType *>(aicpu_ext_info->infoMsg);

  for (uint32_t index = 0; index < input_num_; ++index) {
    input_shape_and_type_.emplace_back(&input[index]);
  }
  GELOGI("Node[%s] parse ext input shape success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtOutputShape(AicpuExtInfo *aicpu_ext_info) {
  if (unknown_type_ == DEPEND_COMPUTE) {
    GELOGD("Node[%s] is depend compute type no need ext output shape, ignore it, infoLen=%u.",
           node_name_.c_str(), aicpu_ext_info->infoLen);
    return SUCCESS;
  }
  auto need_len = output_num_ * sizeof(AicpuShapeAndType);
  GE_CHK_BOOL_RET_STATUS(aicpu_ext_info->infoLen == need_len, ACL_ERROR_GE_PARAM_INVALID,
                         "Node[%s] parse ext output shape failed as infoLen must be "
                         "output_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                         node_name_.c_str(), output_num_, sizeof(AicpuShapeAndType), aicpu_ext_info->infoLen);

  auto output = reinterpret_cast<AicpuShapeAndType *>(aicpu_ext_info->infoMsg);
  for (uint32_t index = 0; index < output_num_; ++index) {
    output_shape_and_type_.emplace_back(&output[index]);
  }
  GELOGI("Node[%s] parse ext output shape success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtSessionInfo(AicpuExtInfo *aicpu_ext_info) {
  GE_CHK_BOOL_RET_STATUS(aicpu_ext_info->infoLen == sizeof(AicpuSessionInfo), ACL_ERROR_GE_PARAM_INVALID,
                         "Node[%s] parse ext session info failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(SessionInfo), aicpu_ext_info->infoLen);

  session_info_ = reinterpret_cast<AicpuSessionInfo *>(aicpu_ext_info->infoMsg);
  GELOGI("Node[%s] parse session info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtBitMap(AicpuExtInfo *aicpu_ext_info) {
  GE_CHK_BOOL_RET_STATUS(aicpu_ext_info->infoLen == sizeof(uint64_t), PARAM_INVALID,
                         "Node[%s] parse bit_map info failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(uint64_t), aicpu_ext_info->infoLen);

  bit_map_ = reinterpret_cast<uint64_t *>(aicpu_ext_info->infoMsg);
  GELOGI("Node[%s] bit_map info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateExecuteMode(bool flag) {
  if (bit_map_ == nullptr) {
    GELOGD("There is no bit_map in ext_info, no need update.");
    return SUCCESS;
  }
  if (flag) {
    *(bit_map_) |= 1;
  } else {
    *(bit_map_) &= ~1;
  }
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateSessionInfo(uint64_t session_id, uint64_t kernel_id, bool sess_flag) {
  if (session_info_ == nullptr) {
    GELOGD("There is no session info in ext_info, no need update.");
    return SUCCESS;
  }

  session_info_->sessionId = session_id;
  session_info_->kernelId = kernel_id;
  session_info_->sessFlag = sess_flag;
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateSessionInfoSessionId(uint64_t session_id) {
  if (session_info_ == nullptr) {
    GELOGD("There is no session info in ext_info, no need update.");
    return SUCCESS;
  }

  session_info_->sessionId = session_id;
  session_info_->sessFlag = true;
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateInputShapeAndType(uint32_t input_index, const GeTensorDesc &input_desc) {
  GE_CHECK_LE(input_index, input_num_);
  const auto &shape = input_desc.GetShape();

  GE_CHK_STATUS_RET(UpdateShapeAndType(shape, input_desc.GetDataType(), input_shape_and_type_[input_index]),
                    "Node[%s] input[%u] update input shape and type failed.",
                    node_name_.c_str(), input_index);
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateOutputShapeAndType(uint32_t output_index, const GeTensorDesc &output_desc) {
  GE_CHK_BOOL_RET_STATUS((unknown_type_ != DEPEND_COMPUTE), ACL_ERROR_GE_INTERNAL_ERROR,
                         "Node[%s] is depend compute is no need update output shape and type by ext.",
                         node_name_.c_str());
  GE_CHECK_LE(output_index, output_num_);
  auto shape = output_desc.GetShape();

  // shape range need use range update shape
  if (unknown_type_ == DEPEND_SHAPE_RANGE) {
    std::vector<std::pair<int64_t, int64_t>> range;
    auto range_ret = output_desc.GetShapeRange(range);
    GE_CHK_BOOL_RET_STATUS(range_ret == GRAPH_SUCCESS, ACL_ERROR_GE_INTERNAL_ERROR,
                           "Node[%s] is shape range type but get GetShapeRange failed, ret=%u.",
                           node_name_.c_str(), range_ret);
    for (size_t k = 0; k < range.size(); ++k) {
      if (shape.GetDim(k) < 0 && k < range.size()) {
        GELOGD("Node[%s] output[%u] update dim[%zu] from %ld to range max %ld.",
               node_name_.c_str(), output_index, k, shape.GetDim(k), range[k].second);
        shape.SetDim(k, range[k].second);
      }
    }
  }

  return UpdateShapeAndType(shape, output_desc.GetDataType(), output_shape_and_type_[output_index]);
}

Status AicpuExtInfoHandler::GetOutputShapeAndType(uint32_t output_index, GeShape &shape, DataType &data_type) {
  GE_CHK_BOOL_RET_STATUS((unknown_type_ != DEPEND_COMPUTE), INTERNAL_ERROR,
                         "Node[%s] is depend compute type can not get output shape and type by ext.",
                         node_name_.c_str());
  GetShapeAndType(output_shape_and_type_[output_index], shape, data_type);
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateShapeAndType(const GeShape &shape, DataType data_type,
                                               AicpuShapeAndType *shape_and_type) {
  auto dim_num = shape.GetDimNum();
  if (dim_num > aicpu::FWKAdapter::kMaxShapeDims) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Update shape and type failed, as dim_num %zu is over max shape dims %u.",
           dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t index = 0;
  for (; index < dim_num; ++index) {
    shape_and_type->dims[index] = shape.GetDim(index);
  }
  if (index < aicpu::FWKAdapter::kMaxShapeDims) {
    shape_and_type->dims[index] = kDimEndFlag;
  }

  // now only support update shape, type is not support
  return SUCCESS;
}

void AicpuExtInfoHandler::GetShapeAndType(const AicpuShapeAndType *shape_and_type,
                                          GeShape &shape,
                                          DataType &data_type) {
  std::vector<int64_t> dims;
  for (uint32_t index = 0; index < aicpu::FWKAdapter::kMaxShapeDims; ++index) {
    auto tmpDim = shape_and_type->dims[index];
    if (tmpDim == kDimEndFlag) {
      break;
    }
    dims.emplace_back(tmpDim);
  }
  data_type = static_cast<DataType>(shape_and_type->type);
  shape = GeShape(dims);
}
}  // namespace hybrid
}  // namespace ge
