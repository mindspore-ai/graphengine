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

#include "graph/load/new_model_manager/task_info/kernel_ex_task_info.h"

#include <vector>

#include "cce/aicpu_engine_struct.h"
#include "common/ge/ge_util.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "graph/attr_value.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/model_manager.h"

namespace ge {
Status KernelExTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("KernelExTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }

  davinci_model_ = davinci_model;
  Status ret = SetStream(task_def.stream_id(), davinci_model_->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  auto kernel_ex_def = task_def.kernel_ex();
  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();

  // 1. Copy context from kernelExDef.private to workspace
  uint32_t op_index = kernel_ex_def.op_index();
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Init aicpu task info error, index is out of range!");
    return INTERNAL_ERROR;
  }

  if (CopyTaskInfo(kernel_ex_def, rts_param, op_desc) != SUCCESS) {
    GELOGE(FAILED, "copy task info to workspace failed.");
    return FAILED;
  }

  const vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);
  if (workspace_data_addrs.empty()) {
    GELOGE(FAILED, "workspace_data_addrs is empty.");
    return FAILED;
  }

  // 2. Reconstruct kernelExDef.args to STR_FWK_OP_KERNEL
  STR_FWK_OP_KERNEL fwk_op_kernel = {0};
  if (sizeof(STR_FWK_OP_KERNEL) < kernel_ex_def.args_size()) {
    GELOGE(FAILED, "sizeof STR_FWK_OP_KERNEL is: %zu, but args_size is: %u", sizeof(STR_FWK_OP_KERNEL),
           kernel_ex_def.args_size());
    return FAILED;
  }
  errno_t sec_ret =
    memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL), kernel_ex_def.args().data(), kernel_ex_def.args_size());
  if (sec_ret != EOK) {
    GELOGE(FAILED, "memcpy failed, ret: %d", sec_ret);
    return FAILED;
  }

  // 2.1 get loop cond variable for tensor array write
  uint64_t step_id_addr = 0;
  OpDescPtr step_id_node = davinci_model_->GetVariableOp(NODE_NAME_GLOBAL_STEP);
  if (step_id_node != nullptr) {
    vector<void *> v_step_id_addr = ModelUtils::GetOutputDataAddrs(rts_param, step_id_node);
    if (!v_step_id_addr.empty()) {
      step_id_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(v_step_id_addr[0]));
    }
  }

  // 3. Set workspaceaddr, inputOutputDataAddr
  uint64_t workspace_base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(workspace_data_addrs[0]));
  const vector<void *> input_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  const vector<void *> output_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  vector<void *> io_addrs;
  io_addrs.insert(io_addrs.end(), input_addrs.begin(), input_addrs.end());
  io_addrs.insert(io_addrs.end(), output_addrs.begin(), output_addrs.end());

  auto addrs_size = sizeof(uint64_t) * (io_addrs.size());
  if (addrs_size > 0) {
    rtError_t rt_ret = rtMalloc(&input_output_addr_, addrs_size, RT_MEMORY_HBM);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "rtMalloc error, ret: 0x%X", rt_ret); return RT_FAILED;)

    rt_ret = rtMemcpy(input_output_addr_, addrs_size, io_addrs.data(), addrs_size, RT_MEMCPY_HOST_TO_DEVICE);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtMemcpy to input_output_addr_ error: 0x%X", rt_ret);
                    return FAILED;)

    if (PropertiesManager::Instance().IsLayerNeedDump(davinci_model_->Name(), op_desc->GetName())) {
      dump_flag_ = RT_KERNEL_DUMPFLAG;
      dump_args_ =
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(input_output_addr_) + sizeof(void *) * input_addrs.size());
    }
  }

  uint64_t input_output_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_output_addr_));
  fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr = workspace_base_addr;
  fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = input_output_addr;
  fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = step_id_addr;
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoNum = 0;
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = 0;

  // 4. Create session
  auto session_id = fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID;
  GE_CHECK_NOTNULL(ModelManager::GetInstance());
  GE_IF_BOOL_EXEC(ModelManager::GetInstance()->CreateAicpuSession(session_id) != SUCCESS,
                  GELOGE(FAILED, "CreateAicpuSession error. session id: %lu", session_id);
                  return FAILED;)
  // 4.1 Collect aicpu kernel
  uint64_t kernel_id = fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID;
  GE_IF_BOOL_EXEC(ModelManager::GetInstance()->CreateAicpuKernel(session_id, davinci_model->Id(), kernel_id) != SUCCESS,
                  GELOGE(FAILED, "CreateAicpuKernel error.");
                  return FAILED;)
  // 5. Return result
  rtError_t rt_ret = rtMalloc(&kernel_buf_, sizeof(STR_FWK_OP_KERNEL), RT_MEMORY_HBM);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtMalloc error: 0x%X", rt_ret); return FAILED;)

  rt_ret = rtMemcpy(kernel_buf_, sizeof(STR_FWK_OP_KERNEL), static_cast<void *>(&fwk_op_kernel),
                    sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtMemcpy error, ret: Ox%X", rt_ret); return FAILED;)

  vector<void *> virtual_io_addrs;  // use virtual address for zero copy key.
  const vector<void *> virtual_in_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc, false);
  const vector<void *> virtual_out_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc, false);
  virtual_io_addrs.insert(virtual_io_addrs.end(), virtual_in_addrs.begin(), virtual_in_addrs.end());
  virtual_io_addrs.insert(virtual_io_addrs.end(), virtual_out_addrs.begin(), virtual_out_addrs.end());
  davinci_model_->SetZeroCopyAddr(op_desc, virtual_io_addrs, io_addrs.data(), input_output_addr_, addrs_size, 0);

  kernel_buf_size_ = sizeof(STR_FWK_OP_KERNEL);

  GELOGI("KernelExTaskInfo Init Success. session id: %lu", session_id);
  return SUCCESS;
}

Status KernelExTaskInfo::CopyTaskInfo(const domi::KernelExDef &kernel_def, const RuntimeParam &rts_param,
                                      const OpDescPtr &op_desc) {
  // Userspace copy need virtual address.
  const vector<int64_t> workspace_data_sizes = ModelUtils::GetWorkspaceSize(op_desc);
  const vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc, false);
  if (workspace_data_addrs.empty() || workspace_data_sizes.empty()) {
    GELOGE(FAILED, "Node:%s invalid workspace, addrs is %zu, size is %zu.", op_desc->GetName().c_str(),
           workspace_data_addrs.size(), workspace_data_sizes.size());
    return FAILED;
  }

  if (workspace_data_addrs[0] == nullptr) {
    GELOGE(FAILED, "Node:%s workspace addrs is null.", op_desc->GetName().c_str());
    return FAILED;
  }

  if (workspace_data_sizes[0] < static_cast<int64_t>(kernel_def.task_info_size())) {
    GELOGE(FAILED, "Node:%s workspace size is %zu, task info size is %zu.", op_desc->GetName().c_str(),
           workspace_data_sizes[0], kernel_def.task_info_size());
    return FAILED;
  }

  rtError_t rt_ret = rtMemcpy(workspace_data_addrs[0], kernel_def.task_info_size(), kernel_def.task_info().data(),
                              kernel_def.task_info_size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "rtMemcpy error: 0x%X", rt_ret);
    return FAILED;
  }

  return SUCCESS;
}

Status KernelExTaskInfo::Distribute() {
  GELOGI("KernelExTaskInfo Distribute Start.");
  rtError_t rt_ret = rtKernelLaunchEx(kernel_buf_, kernel_buf_size_, dump_flag_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  if (davinci_model_ == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model_ is null.");
    return PARAM_INVALID;
  }

  uint32_t task_id = 0;
  uint32_t stream_id = 0;  //  for profiling
  rt_ret = rtModelGetTaskId(davinci_model_->GetRtModelHandle(), &task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  task_id_ = task_id;
  stream_id_ = stream_id;

  GELOGI("KernelExTaskInfo Distribute Success. task id: %u, stream id: %u", task_id_, stream_id_);
  return SUCCESS;
}

Status KernelExTaskInfo::Release() {
  Status ret = SUCCESS;
  if (kernel_buf_ != nullptr) {
    rtError_t rt_ret = rtFree(kernel_buf_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree error, ret: 0x%X", rt_ret);
      ret = FAILED;
    } else {
      kernel_buf_ = nullptr;
    }
  }
  if (input_output_addr_ != nullptr) {
    rtError_t rt_ret = rtFree(input_output_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree error, ret: 0x%X", rt_ret);
      ret = FAILED;
    } else {
      input_output_addr_ = nullptr;
    }
  }
  return ret;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_KERNEL_EX, KernelExTaskInfo);
}  // namespace ge
