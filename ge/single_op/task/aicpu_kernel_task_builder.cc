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

#include "single_op/task/aicpu_kernel_task_builder.h"
#include "framework/common/taskdown_common.h"
#include "graph/load/new_model_manager/model_manager.h"

namespace ge {
AiCpuCCTaskBuilder::AiCpuCCTaskBuilder(const OpDescPtr &op_desc, const domi::KernelDef &kernel_def)
    : op_desc_(op_desc), kernel_def_(kernel_def) {}

Status AiCpuCCTaskBuilder::SetKernelArgs(AiCpuCCTask &task) {
  size_t aicpu_arg_size = kernel_def_.args_size();
  if (aicpu_arg_size <= 0) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "aicpu_arg_size is invalid, value = %zu", aicpu_arg_size);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  std::unique_ptr<uint8_t[]> aicpu_args;
  aicpu_args.reset(new(std::nothrow) uint8_t[aicpu_arg_size]());
  if (aicpu_args == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "malloc failed, size = %zu", aicpu_arg_size);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  auto err = memcpy_s(aicpu_args.get(), aicpu_arg_size, kernel_def_.args().data(), aicpu_arg_size);
  if (err != EOK) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "memcpy_s args failed, size = %zu, err = %d", aicpu_arg_size, err);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  task.SetIoAddr(aicpu_args.get() + sizeof(aicpu::AicpuParamHead));
  task.SetKernelArgs(std::move(aicpu_args), aicpu_arg_size);
  return SUCCESS;
}

Status AiCpuCCTaskBuilder::BuildTask(AiCpuCCTask &task, uint64_t kernel_id) {
  auto ret = SetKernelArgs(task);
  if (ret != SUCCESS) {
    return ret;
  }
  const std::string &so_name = kernel_def_.so_name();
  const std::string &kernel_name = kernel_def_.kernel_name();
  task.SetSoName(so_name);
  task.SetkernelName(kernel_name);
  task.op_desc_ = op_desc_;

  const auto &context = kernel_def_.context();
  auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
  if (kernel_type == ccKernelType::CUST_AI_CPU) {
    task.is_custom_ = true;
    task.dump_flag_ |= RT_KERNEL_CUSTOM_AICPU;
    bool loaded = false;
    GE_CHK_STATUS_RET(ModelManager::GetInstance()->LoadCustAicpuSo(op_desc_, so_name, loaded),
        "launch cust aicpu so failed");
    if (!loaded) {
      GE_CHK_STATUS_RET(ModelManager::GetInstance()->LaunchCustAicpuSo(), "launch cust aicpu so failed.");
    }
  }

  task.num_inputs_ = op_desc_->GetInputsSize();
  task.num_outputs_ = op_desc_->GetOutputsSize();

  // get kernel_ext_info
  auto &kernel_ext_info = kernel_def_.kernel_ext_info();
  auto kernel_ext_info_size = kernel_def_.kernel_ext_info_size();
  GE_CHK_BOOL_RET_STATUS(kernel_ext_info.size() == kernel_ext_info_size, FAILED,
                         "task def kernel_ext_info.size=%zu, but kernel_ext_info_size=%u.",
                         kernel_ext_info.size(), kernel_ext_info_size);

  ret = task.SetExtInfoAndType(kernel_ext_info, kernel_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "Init ext info failed.");
    return ret;
  }

  auto aicpu_param_head = reinterpret_cast<aicpu::AicpuParamHead *>(task.args_.get());
  if (task.ext_info_addr_dev_ != nullptr) {
    aicpu_param_head->extInfoLength = kernel_ext_info.size();
    aicpu_param_head->extInfoAddr = reinterpret_cast<uintptr_t>(task.ext_info_addr_dev_);
  }

  return SUCCESS;
}
}  // namespace ge