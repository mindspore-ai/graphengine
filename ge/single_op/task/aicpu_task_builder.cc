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

#include "single_op/task/aicpu_task_builder.h"
#include <vector>
#include "single_op/task/build_task_utils.h"
#include "runtime/mem.h"
#include "framework/common/debug/ge_log.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/load/new_model_manager/model_manager.h"

namespace ge {
AiCpuTaskBuilder::AiCpuTaskBuilder(const OpDescPtr &op_desc, const domi::KernelExDef &kernel_def)
    : op_desc_(op_desc), kernel_def_(kernel_def) {}

Status AiCpuTaskBuilder::SetInputOutputAddr(void **io_addr, const std::vector<void *> &addresses) {
  size_t arg_size = kernel_def_.args_size();
  auto rt_ret = rtMalloc(io_addr, arg_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMalloc failed, size = %zu, ret = %d", arg_size, rt_ret);
    return RT_FAILED;
  }

  const void *src_addr = reinterpret_cast<const void *>(addresses.data());
  uint64_t src_len = sizeof(void *) * addresses.size();
  rt_ret = rtMemcpy(*io_addr, arg_size, src_addr, src_len, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    (void)rtFree(*io_addr);
    GELOGE(RT_FAILED, "rtMemcpy addresses failed, ret = %d", rt_ret);
    return RT_FAILED;
  }

  return SUCCESS;
}

Status AiCpuTaskBuilder::SetFmkOpKernel(void *io_addr, void *ws_addr, STR_FWK_OP_KERNEL &fwk_op_kernel) {
  auto sec_ret =
    memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL), kernel_def_.args().data(), kernel_def_.args().size());
  if (sec_ret != EOK) {
    GELOGE(FAILED, "memcpy failed, ret: %d", sec_ret);
    return FAILED;
  }

  auto io_addr_val = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(io_addr));
  fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = io_addr_val;
  auto ws_addr_val = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ws_addr));
  fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr = ws_addr_val;
  return SUCCESS;
}

Status AiCpuTaskBuilder::SetKernelArgs(void **args, STR_FWK_OP_KERNEL &fwk_op_kernel) {
  void *fwk_op_args = nullptr;
  auto rt_ret = rtMalloc(&fwk_op_args, sizeof(STR_FWK_OP_KERNEL), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "malloc arg memory failed, ret = %d", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtMemcpy(fwk_op_args, sizeof(STR_FWK_OP_KERNEL), &fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL),
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    (void)rtFree(fwk_op_args);
    GELOGE(RT_FAILED, "copy args failed, ret = %d", rt_ret);
    return RT_FAILED;
  }
  *args = fwk_op_args;
  return SUCCESS;
}

Status AiCpuTaskBuilder::InitWorkspaceAndIO(void **io_addr, void **kernel_workspace, const SingleOpModelParam &param,
                                            bool dynamic_flag) {
  if (kernel_def_.args_size() > sizeof(STR_FWK_OP_KERNEL)) {
    GELOGE(PARAM_INVALID, "sizeof STR_FWK_OP_KERNEL is: %lu, but args_size is: %d", sizeof(STR_FWK_OP_KERNEL),
           kernel_def_.args_size());
    return PARAM_INVALID;
  }
  auto addresses = BuildTaskUtils::GetAddresses(op_desc_, param);
  auto ws_addr_vec = addresses.at(BuildTaskUtils::kAddressIndexWorkspace);

  if (dynamic_flag) {
    GE_CHK_RT_RET(rtMalloc(kernel_workspace, kernel_def_.task_info_size(), RT_MEMORY_HBM));
  } else {
    if (ws_addr_vec.empty()) {
      GELOGE(PARAM_INVALID, "workspace Data Address is empty.");
      return PARAM_INVALID;
    }
    *kernel_workspace = ws_addr_vec[0];
  }
  GE_CHK_RT_RET(rtMemcpy(*kernel_workspace, kernel_def_.task_info_size(), kernel_def_.task_info().data(),
                         kernel_def_.task_info_size(), RT_MEMCPY_HOST_TO_DEVICE));

  auto ret = SetInputOutputAddr(io_addr, BuildTaskUtils::JoinAddresses(addresses));
  if (ret != SUCCESS) {
    return ret;
  }
  return SUCCESS;
}

Status AiCpuTaskBuilder::BuildTask(ge::AiCpuTask &task, const SingleOpModelParam &param, bool dynamic_flag,
                                   uint64_t session_id) {
  void *io_addr = nullptr;
  void *kernel_workspace = nullptr;
  GE_CHK_STATUS_RET_NOLOG(InitWorkspaceAndIO(&io_addr, &kernel_workspace, param, dynamic_flag));

  STR_FWK_OP_KERNEL fwk_op_kernel = {0};
  auto ret = SetFmkOpKernel(io_addr, kernel_workspace, fwk_op_kernel);
  if (ret != SUCCESS) {
    (void)rtFree(io_addr);
    return ret;
  }

  task.op_desc_ = op_desc_;
  task.num_inputs_ = op_desc_->GetInputsSize();
  task.num_outputs_ = op_desc_->GetOutputsSize();

  // get kernel_ext_info
  auto &kernel_ext_info = kernel_def_.kernel_ext_info();
  auto kernel_ext_info_size = kernel_def_.kernel_ext_info_size();
  GE_CHK_BOOL_RET_STATUS(kernel_ext_info.size() == kernel_ext_info_size, FAILED,
                         "task def kernel_ext_info.size=%zu, but kernel_ext_info_size=%u.", kernel_ext_info.size(),
                         kernel_ext_info_size);
  GE_CHK_STATUS_RET(task.SetExtInfoAndType(kernel_ext_info), "Init ext info failed.");

  if (task.ext_info_addr_dev_ != nullptr) {
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = reinterpret_cast<uintptr_t>(task.ext_info_addr_dev_);
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = kernel_ext_info_size;
  }
  GE_CHK_STATUS_RET(task.InitForSummaryAndCopy(), "AiCpuTask init for summary and copy task failed.");

  // Create session
  fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID = session_id;
  GELOGI("Begin to CreateAicpuSession, session id: %lu", session_id);
  GE_CHECK_NOTNULL(ModelManager::GetInstance());
  GE_IF_BOOL_EXEC(ModelManager::GetInstance()->CreateAicpuSession(session_id) != SUCCESS,
                  GELOGE(FAILED, "CreateAicpuSession error. session id: %lu", session_id);
                  return FAILED;)
  ret = SetKernelArgs(&task.args_, fwk_op_kernel);
  if (ret != SUCCESS) {
    (void)rtFree(io_addr);
    return ret;
  }

  task.arg_size_ = sizeof(STR_FWK_OP_KERNEL);
  task.op_type_ = op_desc_->GetName();
  task.io_addr_ = io_addr;
  task.task_info_ = kernel_def_.task_info();
  task.workspace_addr_ = kernel_workspace;
  task.dynamic_flag_ = dynamic_flag;

  auto debug_info = BuildTaskUtils::GetTaskInfo(op_desc_);
  GELOGI("[TASK_INFO] %s %s", task.task_info_.c_str(), debug_info.c_str());
  return SUCCESS;
}
}  // namespace ge
