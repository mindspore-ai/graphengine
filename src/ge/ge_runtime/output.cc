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

#include "ge_runtime/output.h"

#include "./op_info_utils.h"
#include "cce/dnn_base.h"
#include "cce/dnn_base_def.hpp"
#include "common/ge_inner_error_codes.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"

using cce::ccTensorDescriptor_t;
using cce::ccDestroyTensorDescriptor;

namespace ge {
namespace model_runner {
Output::Output(const OpInfoPtr &op_info, const std::shared_ptr<DavinciModel> &model)
    : model_(model), op_info_(op_info), input_num_(0) {}

Output::~Output() {}

bool Output::Init() {
  if (op_info_ == nullptr || model_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "The op_desc_ or model_ is nullptr.");
    return false;
  }

  input_num_ = op_info_->input_tensors.size();
  v_input_size_.clear();
  v_input_data_addr_.clear();

  auto input_vector = op_info_->input_addrs;
  if (input_num_ != input_vector.size()) {
    GELOGE(INTERNAL_ERROR, "The input desc size: %zu !=  input addr size: %zu.", input_num_, input_vector.size());
    return false;
  }

  for (size_t i = 0; i < input_num_; i++) {
    uint32_t tensorSize = 0;
    const auto &input_info = op_info_->input_tensors.at(i);
    tensorSize = input_info.size;
    v_input_size_.push_back(tensorSize);
    v_input_data_addr_.push_back(reinterpret_cast<uint8_t *>(input_vector.at(i)));
  }

  GELOGI("Init output:%zu, %zu, %zu", input_num_, v_input_size_.size(), v_input_data_addr_.size());

  return true;
}

///
/// @ingroup domi_ome
/// @brief Copy Op Output to user space.
/// @brief when model running, Add one DataOp as input node, Add one Output Op as output node.
/// @return Status
///
bool Output::CopyRslt(OutputData *rslt, uint32_t data_begin, uint32_t &data_index, bool support_mem_share) {
  if (rslt == nullptr) {
    GELOGE(FAILED, "OutputData is null.");
    return false;
  }
  uint32_t data_count = 0;
  if (v_input_size_.empty() || v_input_data_addr_.empty()) {
    GELOGE(INTERNAL_ERROR, "v_output_size_ or v_output_data_addr_ is empty!");
    return false;
  }

  for (size_t i = 0; i < input_num_; i++) {
    DataBuffer data_buf = rslt->blobs[data_begin + data_count];
    bool ret = SetDataBuf(data_buf, data_count, i, support_mem_share);
    if (!ret) {
      GELOGE(FAILED, "Copy data to host error. index: %lu", i);
      return ret;
    }
    data_index = data_begin + data_count;
  }

  return true;
}

bool Output::SetDataBuf(DataBuffer &data_buf, uint32_t &data_count, size_t i, bool support_mem_share) {
  if (op_info_ == nullptr) {
    GELOGE(FAILED, "op_info_ is null");
    return false;
  }
  if (data_buf.length == 0) {
    ++data_count;
    GELOGD("data_buf.length = 0,do not need copy, output op : %s, output tensor index : %zu!",
           op_info_->name.c_str(), i);
    return true;
  }

  ccTensorDescriptor_t cc_tensor_desc = nullptr;
  GE_MAKE_GUARD_TENSOR(cc_tensor_desc);

  if (i >= op_info_->input_tensors.size()) {
    GELOGE(FAILED, "tensor_info is null");
    return false;
  }

  auto tensor_info = op_info_->input_tensors.at(i);

  if (data_buf.isDataSupportMemShare && support_mem_share) {
    GELOGI("No need to copy input data, user's output data buffer can be shared.");
  } else {
    // copy result to Databuf
    uint32_t size = v_input_size_[i];
    GELOGI("Tensor data size before: %u", size);
    if (!OpInfoUtils::InitTensorDescriptor(tensor_info.format, tensor_info.datatype, tensor_info.dims,
                                           cc_tensor_desc)) {
      GELOGE(FAILED, "OpUtils::InitTensorDescriptor tensorDesc failed.");
      return false;
    }
    if (ccGetTensorSizeInBytes(cc_tensor_desc, &size) != CC_STATUS_SUCCESS) {
      return false;
    }
    rtError_t rt_ret = rtMemcpy(data_buf.data, size, v_input_data_addr_[i], size, RT_MEMCPY_DEVICE_TO_HOST);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(rt_ret, "rtmemcpy error");
      return false;
    }
    GELOGI("Tensor data size: %u data_buflength: %u", size, data_buf.length);
    OpInfoUtils::DestroyTensorDescriptor(cc_tensor_desc);
  }

  ++data_count;
  GELOGD("Successfully copy the output tensor memory to buffer, output op : %s, output tensor index : %lu!",
         op_info_->name.c_str(), i);

  return false;
}

}  // namespace model_runner
}  // namespace ge
