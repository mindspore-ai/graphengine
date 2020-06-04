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

#include "runtime/rt.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
void TbeOpTask::SetStubFunc(const std::string &name, const void *stub_func) {
  this->stub_name_ = name;
  this->stub_func_ = stub_func;
}

void TbeOpTask::SetKernelArgs(void *args, size_t arg_size, uint32_t block_dim) {
  args_ = args;
  arg_size_ = arg_size;
  block_dim_ = block_dim;
}

void TbeOpTask::SetSmDesc(void *sm_desc) { sm_desc_ = sm_desc; }

TbeOpTask::~TbeOpTask() {
  if (args_ != nullptr) {
    (void)rtFreeHost(args_);
  }

  if (sm_desc_ != nullptr) {
    (void)rtMemFreeManaged(sm_desc_);
  }
}

const void *TbeOpTask::GetArgs() const { return args_; }

size_t TbeOpTask::GetArgSize() const { return arg_size_; }

const std::string &TbeOpTask::GetStubName() const { return stub_name_; }

Status TbeOpTask::LaunchKernel(rtStream_t stream) {
  GELOGD("To invoke rtKernelLaunch. task = %s, block_dim = %u", this->stub_name_.c_str(), block_dim_);
  auto *sm_desc = reinterpret_cast<rtSmDesc_t *>(sm_desc_);
  auto ret = rtKernelLaunch(stub_func_, block_dim_, args_, static_cast<uint32_t>(arg_size_), sm_desc, stream);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Invoke rtKernelLaunch failed. ret = %d, task = %s", ret, this->stub_name_.c_str());
    return RT_FAILED;
  }

  GELOGD("Invoke rtKernelLaunch succeeded. task = %s", this->stub_name_.c_str());
  return SUCCESS;
}

AiCpuTask::~AiCpuTask() {
  if (args_ != nullptr) {
    rtFree(args_);
  }

  if (io_addr_ != nullptr) {
    (void)rtFree(io_addr_);
  }
}

void *AiCpuTask::GetIOAddr() { return io_addr_; }

Status AiCpuTask::LaunchKernel(rtStream_t stream) {
  auto ret = rtMemcpyAsync(workspace_addr_, task_info_.size(), task_info_.data(), task_info_.size(),
                           RT_MEMCPY_HOST_TO_DEVICE, stream);
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

  GELOGD("Invoke rtKernelLaunch succeeded. task = %s", this->op_type_.c_str());
  return SUCCESS;
}
}  // namespace ge
