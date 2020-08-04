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

#include "aicore_op_task.h"
#include "framework/common/debug/log.h"

namespace ge {
namespace hybrid {

Status AiCoreOpTask::LaunchKernel(rtStream_t stream) {
  GELOGI("AiCoreOpTask LaunchKernel Start (task = %s, block_dim = %u).", stub_name_.c_str(), block_dim_);

  GE_CHK_RT_RET(rtKernelLaunch(stub_func_, block_dim_, args_.get(), args_size_, nullptr, stream));
  GELOGI("AiCoreOpTask LaunchKernel End (task = %s, block_dim = %u).", stub_name_.c_str(), block_dim_);
  return SUCCESS;
}

}  // namespace hybrid
}  // namespace ge