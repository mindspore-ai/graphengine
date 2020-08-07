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

#include "single_op/task/op_task.h"

#include <chrono>
#include <thread>
#include <google/protobuf/extension_set.h>

#include "runtime/rt.h"
#include "register/op_tiling.h"
#include "framework/common/debug/log.h"

namespace ge {
namespace {
constexpr int kLaunchRetryTimes = 1000;
constexpr int kSleepTime = 10;
}  // namespace

void TbeOpTask::SetStubFunc(const std::string &name, const void *stub_func) {
  this->stub_name_ = name;
  this->stub_func_ = stub_func;
}

void TbeOpTask::SetKernelArgs(std::unique_ptr<uint8_t[]> &&args, size_t arg_size, uint32_t block_dim) {
  args_ = std::move(args);
  arg_size_ = arg_size;
  block_dim_ = block_dim;
}

void TbeOpTask::SetSmDesc(void *sm_desc) { sm_desc_ = sm_desc; }

const vector<int64_t> &OpTask::GetWorkspaceSizes() const { return workspace_sizes_; }

void OpTask::SetWorkspaceSizes(const vector<int64_t> &workspace_sizes) { workspace_sizes_ = workspace_sizes; }

TbeOpTask::~TbeOpTask() {
  if (sm_desc_ != nullptr) {
    (void)rtMemFreeManaged(sm_desc_);
  }

  if (tiling_buffer_ != nullptr) {
    (void)rtFree(tiling_buffer_);
  }
}

const void *TbeOpTask::GetArgs() const { return args_.get(); }

size_t TbeOpTask::GetArgSize() const { return arg_size_; }

const std::string &TbeOpTask::GetStubName() const { return stub_name_; }

Status TbeOpTask::LaunchKernel(rtStream_t stream) {
  GELOGD("To invoke rtKernelLaunch. task = %s, block_dim = %u", this->stub_name_.c_str(), block_dim_);
  auto *sm_desc = reinterpret_cast<rtSmDesc_t *>(sm_desc_);
  auto ret = rtKernelLaunch(stub_func_, block_dim_, args_.get(), static_cast<uint32_t>(arg_size_), sm_desc, stream);
  int retry_times = 0;
  while (ret != RT_ERROR_NONE && retry_times < kLaunchRetryTimes) {
    retry_times++;
    GELOGW("Retry after %d ms, retry_times: %d", kSleepTime, retry_times);
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepTime));
    ret = rtKernelLaunch(stub_func_, block_dim_, args_.get(), arg_size_, sm_desc, stream);
  }

  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Invoke rtKernelLaunch failed. ret = %d, task = %s", ret, this->stub_name_.c_str());
    return RT_FAILED;
  }

  GELOGI("[TASK_INFO] %s", this->stub_name_.c_str());
  return SUCCESS;
}

Status TbeOpTask::UpdateRunInfo(const vector<GeTensorDesc> &input_desc, const vector<GeTensorDesc> &output_desc) {
  GE_CHK_STATUS_RET_NOLOG(UpdateNodeByShape(input_desc, output_desc));
  // invoke OpParaCalculate
  GELOGD("Start to invoke OpParaCalculate.");
  optiling::OpRunInfo run_info;
  auto ret = optiling::OpParaCalculate(*node_, run_info);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to invoke OpParaCalculate. ret = %u", ret);
    return FAILED;
  }
  SetWorkspaceSizes(run_info.workspaces);
  block_dim_ = run_info.block_dim;
  tiling_data_ = run_info.tiling_data.str();
  GELOGD("Done invoking OpParaCalculate successfully. block_dim = %u, tiling size = %zu", block_dim_,
         tiling_data_.size());
  return SUCCESS;
}

Status TbeOpTask::UpdateTensorDesc(const GeTensorDesc &src_tensor, GeTensorDesc &dst_tensor) {
  int64_t storage_format_val = static_cast<Format>(FORMAT_RESERVED);
  (void)AttrUtils::GetInt(src_tensor, ge::ATTR_NAME_STORAGE_FORMAT, storage_format_val);
  auto storage_format = static_cast<Format>(storage_format_val);
  if (storage_format == FORMAT_RESERVED) {
    GELOGD("Storage format not set. update shape to [%s], and original shape to [%s]",
           src_tensor.GetShape().ToString().c_str(), src_tensor.GetOriginShape().ToString().c_str());
    dst_tensor.SetShape(src_tensor.GetShape());
    dst_tensor.SetOriginShape(src_tensor.GetOriginShape());
  } else {
    std::vector<int64_t> storage_shape;
    if (!AttrUtils::GetListInt(src_tensor, ge::ATTR_NAME_STORAGE_SHAPE, storage_shape)) {
      GELOGE(PARAM_INVALID, "Failed to get storage_shape while storage_format was set");
      return PARAM_INVALID;
    }

    GELOGD("Storage format set. update shape to [%s], and original shape to [%s]",
           GeShape(storage_shape).ToString().c_str(), src_tensor.GetShape().ToString().c_str());
    dst_tensor.SetShape(GeShape(std::move(storage_shape)));
    dst_tensor.SetOriginShape(src_tensor.GetShape());
  }

  return SUCCESS;
}

Status TbeOpTask::UpdateNodeByShape(const vector<GeTensorDesc> &input_desc, const vector<GeTensorDesc> &output_desc) {
  auto op_desc = node_->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // Set runtime shape to node
  for (size_t i = 0; i < input_desc.size(); ++i) {
    auto tensor_desc = op_desc->MutableInputDesc(i);
    auto &runtime_tensor_desc = input_desc[i];
    GE_CHECK_NOTNULL(tensor_desc);
    GE_CHK_STATUS_RET(UpdateTensorDesc(runtime_tensor_desc, *tensor_desc));
  }

  for (size_t i = 0; i < output_desc.size(); ++i) {
    auto tensor_desc = op_desc->MutableOutputDesc(i);
    auto &runtime_tensor_desc = output_desc[i];
    GE_CHECK_NOTNULL(tensor_desc);
    GE_CHK_STATUS_RET(UpdateTensorDesc(runtime_tensor_desc, *tensor_desc));
  }

  return SUCCESS;
}

void TbeOpTask::EnableDynamicSupport(const NodePtr &node, void *tiling_buffer, size_t max_tiling_size) {
  node_ = node;
  tiling_buffer_ = tiling_buffer;
  max_tiling_size_ = max_tiling_size;
}

Status TbeOpTask::LaunchKernel(const vector<void *> &inputs, const vector<void *> &outputs,
                               const vector<void *> &workspaces, rtStream_t stream) {
  GELOGD("[%s] Start to launch kernel", node_->GetName().c_str());
  std::vector<void *> args;
  args.insert(args.end(), inputs.begin(), inputs.end());
  args.insert(args.end(), outputs.begin(), outputs.end());
  args.insert(args.end(), workspaces.begin(), workspaces.end());

  if (tiling_buffer_ != nullptr) {
    GELOGD("[%s] Start to copy tiling info. size = %zu", node_->GetName().c_str(), tiling_data_.size());
    GE_CHK_RT_RET(rtMemcpyAsync(tiling_buffer_, max_tiling_size_, tiling_data_.data(), tiling_data_.size(),
                                RT_MEMCPY_HOST_TO_DEVICE_EX, stream));

    args.emplace_back(tiling_buffer_);
  }

  if (memcpy_s(args_.get(), arg_size_, args.data(), args.size() * sizeof(void *)) != EOK) {
    GELOGE(INTERNAL_ERROR, "[%s] Failed to update kernel args.", node_->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("[%s] Start to invoke rtKernelLaunch", node_->GetName().c_str());
  GE_CHK_RT_RET(rtKernelLaunch(stub_func_, block_dim_, args_.get(), arg_size_, nullptr, stream));
  GELOGD("[%s] Done invoking rtKernelLaunch successfully", node_->GetName().c_str());
  return SUCCESS;
}

AiCpuTask::~AiCpuTask() {
  if (args_ != nullptr) {
    (void)rtFree(args_);
  }

  if (io_addr_ != nullptr) {
    (void)rtFree(io_addr_);
  }
}

const void *AiCpuTask::GetIOAddr() const { return io_addr_; }

Status AiCpuTask::LaunchKernel(rtStream_t stream) {
  auto ret = rtMemcpyAsync(workspace_addr_, task_info_.size(), task_info_.data(), task_info_.size(),
                           RT_MEMCPY_HOST_TO_DEVICE_EX, stream);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMemcpyAsync workspace data failed. ret = %d, task = %s", ret, this->op_type_.c_str());
    return RT_FAILED;
  }

  GELOGD("To invoke rtKernelLaunchEx. task = %s", this->op_type_.c_str());
  ret = rtKernelLaunchEx(args_, arg_size_, 0, stream);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Invoke rtKernelLaunch failed. ret = %d, task = %s", ret, this->op_type_.c_str());
    return RT_FAILED;
  }
  GELOGI("[TASK_INFO] %s", this->task_info_.c_str());
  return SUCCESS;
}

void AiCpuCCTask::SetKernelArgs(void *args, size_t arg_size) {
  args_ = args;
  arg_size_ = arg_size;
  // the blockdim value is defult "1" for rtCpuKernelLaunch
  block_dim_ = 1;
}

void AiCpuCCTask::SetSoName(const std::string &so_name) { so_name_ = so_name; }

void AiCpuCCTask::SetkernelName(const std::string &kernel_Name) { kernel_name_ = kernel_Name; }

void AiCpuCCTask::SetIoAddr(void *io_addr) { io_addr_ = io_addr; }

const void *AiCpuCCTask::GetIOAddr() const { return io_addr_; }

const void *AiCpuCCTask::GetArgs() const { return args_; }

size_t AiCpuCCTask::GetArgSize() const { return arg_size_; }

AiCpuCCTask::~AiCpuCCTask() {
  if (args_ != nullptr) {
    free(args_);
    args_ = nullptr;
  }
}

Status AiCpuCCTask::LaunchKernel(rtStream_t stream) {
  GELOGI("To invoke rtCpuKernelLaunch. block_dim = %u, so_name is %s, kernel_name is %s", block_dim_, so_name_.data(),
         kernel_name_.data());
  // sm_desc is nullptr, because l2 buffer does not support
  auto *sm_desc = reinterpret_cast<rtSmDesc_t *>(sm_desc_);
  auto ret =
    rtCpuKernelLaunch(static_cast<const void *>(so_name_.data()), static_cast<const void *>(kernel_name_.data()),
                      block_dim_, args_, static_cast<uint32_t>(arg_size_), sm_desc, stream);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Invoke rtCpuKernelLaunch failed. ret = %d", ret);
    return RT_FAILED;
  }
  GELOGD("Invoke rtCpuKernelLaunch succeeded");
  return SUCCESS;
}
}  // namespace ge
