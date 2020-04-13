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

#include "graph/load/output/output.h"

#include <memory.h>

#include "common/properties_manager.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
Output::Output(const OpDescPtr &op_desc, DavinciModel *model)
    : base_(nullptr),
      var_base_(nullptr),
      logic_base_(0),
      logic_var_base_(0),
      model_(model),
      op_desc_(op_desc),
      input_num_(0) {}

Output::~Output() {
  var_base_ = nullptr;
  base_ = nullptr;
  model_ = nullptr;
}

///
/// @ingroup domi
/// @brief Initialize input/output params
/// @return Status
///
Status Output::Init() {
  if (op_desc_ == nullptr || model_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "The op_desc_ or model_ is nullptr.");
    return INTERNAL_ERROR;
  }

  base_ = model_->MemBase();
  var_base_ = model_->VarMemBase();
  logic_base_ = model_->GetRtBaseAddr();
  logic_var_base_ = model_->GetRtVarAddr();

  input_num_ = op_desc_->GetInputsSize();
  v_input_size_.clear();
  v_input_data_addr_.clear();

  auto input_vector = op_desc_->GetInputOffset();
  if (input_num_ != input_vector.size()) {
    GELOGE(INTERNAL_ERROR, "input desc size: %zu !=  input offset size: %zu.", input_num_, input_vector.size());
    return INTERNAL_ERROR;
  }

  for (size_t i = 0; i < input_num_; i++) {
    uint32_t tensor_size = 0;
    auto input_desc = op_desc_->GetInputDescPtr(i);
    GE_CHECK_NOTNULL(input_desc);
    Status ret = TensorUtils::GetSize(*input_desc, tensor_size);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(ret, "Get size from TensorDesc failed, op : %s, input index : %zu", op_desc_->GetName().c_str(), i);
      return ret;
    }
    v_input_size_.push_back(tensor_size);

    if (VarManager::Instance(model_->SessionId())->IsVarAddr(input_vector[i])) {
      v_input_data_addr_.push_back(static_cast<uint8_t *>(var_base_ + input_vector[i] - logic_var_base_));
    } else {
      v_input_data_addr_.push_back(static_cast<uint8_t *>(base_ + input_vector[i]));
    }
  }

  return SUCCESS;
}

///
/// @ingroup domi
/// @brief Copy Op Output to user space.
/// @brief when model running, Add one DataOp as input node, Add one Output Op as output node.
/// @return Status
///
Status Output::CopyResult(OutputData &rslt, uint32_t data_begin, uint32_t &data_index, bool support_mem_share) {
  uint32_t data_count = 0;
  for (size_t i = 0; i < input_num_; i++) {
    DataBuffer data_buf = rslt.blobs[data_begin + data_count];
    Status ret = SetDataBuf(data_buf, data_count, i, support_mem_share);
    if (ret != SUCCESS) {
      GELOGE(ret, "Copy data to host error. index: %zu", i);
      return ret;
    }
    data_index = data_begin + data_count;
  }

  return SUCCESS;
}

Status Output::SetDataBuf(DataBuffer &data_buf, uint32_t &data_count, size_t i, bool support_mem_share) {
  if (data_buf.length == 0) {
    ++data_count;
    GELOGD("Length of data_buffer is zero, No need to copy. output op : %s, output tensor index : %zu!",
           op_desc_->GetName().c_str(), i);
    return SUCCESS;
  }

  auto tensor_desc = op_desc_->GetInputDescPtr(static_cast<uint32_t>(i));
  if (tensor_desc == nullptr) {
    GELOGE(FAILED, "tensor_desc is null");
    return FAILED;
  }

  if (data_buf.isDataSupportMemShare && support_mem_share) {
    GELOGD("No need to copy input data, user's output data buffer can be shared.");
  } else {
    // Copy result to Databuf
    uint32_t size = v_input_size_[i];

    graphStatus graph_status = TensorUtils::GetTensorSizeInBytes(*tensor_desc, size);
    if (graph_status != ge::GRAPH_SUCCESS) {
      GELOGE(graph_status, "GetTensorSizeInBytes failed!");
      return FAILED;
    }

    rtError_t rt_ret = rtMemcpy(data_buf.data, size, v_input_data_addr_[i], size, RT_MEMCPY_DEVICE_TO_HOST);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(rt_ret, "rtmemcpy error");
      return FAILED;
    }
    GELOGD("Tensor data size: %u data_buflength: %u", size, data_buf.length);
  }

  ++data_count;
  GELOGD("Successfully copy the output tensor memory to buffer, output op : %s, output tensor index : %zu!",
         op_desc_->GetName().c_str(), i);

  return SUCCESS;
}

void Output::GetOutputData(vector<void *> &v_data_addr, vector<uint32_t> &v_data_size) {
  for (size_t i = 0; i < input_num_; ++i) {
    v_data_addr.push_back(v_input_data_addr_[i]);
    v_data_size.push_back(v_input_size_[i]);
  }
}
}  // namespace ge
