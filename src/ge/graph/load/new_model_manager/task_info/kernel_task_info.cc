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

#include "graph/load/new_model_manager/task_info/kernel_task_info.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "aicpu/common/aicpu_task_struct.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/l2_cache_optimize.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "runtime/kernel.h"

namespace ge {
static constexpr uint8_t kL2LoadToDdr = 1;
static constexpr uint8_t kL2NotLoadToDdr = 0;

Status KernelTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGD("KernelTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }
  davinci_model_ = davinci_model;

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  domi::KernelDef kernel_def = task_def.kernel();
  block_dim_ = kernel_def.block_dim();
  args_size_ = kernel_def.args_size();
  // get opcontext stored in model
  const domi::KernelContext &context = kernel_def.context();
  // get kernel_type
  kernel_type_ = static_cast<cce::ccKernelType>(context.kernel_type());
  // get bin_file_key
  OpDescPtr op_desc = davinci_model->GetOpByIndex(context.op_index());
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Get op_desc failed, index is out of range!");
    return INTERNAL_ERROR;
  }
  string session_graph_model_id;
  davinci_model->GetUniqueId(op_desc, session_graph_model_id);
  const char *bin_file_key = DavinciModel::GetRegisterStub(op_desc->GetName(), session_graph_model_id);
  // new aicpu kernel(rtCpuKernelLaunch) no need to check function
  if (kernel_type_ == cce::ccKernelType::CCE_AI_CORE) {
    rtError_t rt_ret;
    rt_ret = rtGetFunctionByName(const_cast<char *>(kernel_def.stub_func().c_str()), &stub_func_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    GELOGE(RT_FAILED,
                           "execute rtGetFunctionByName failed. stub_func: %s",
                           kernel_def.stub_func().c_str());
    return RT_FAILED;);
  } else if (kernel_type_ != cce::ccKernelType::AI_CPU) {
    rtError_t rt_ret;
    rt_ret = rtGetFunctionByName(bin_file_key, &stub_func_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    GELOGE(RT_FAILED, "execute rtGetFunctionByName failed. bin_file_key: %s", bin_file_key);
    return RT_FAILED;);
  }

  if (context.origin_op_index_size() > CC_FUSION_OP_MAX) {
    GELOGE(PARAM_INVALID, "context.origin_op_index_size() is more than CC_FUSION_OP_MAX(%d)", CC_FUSION_OP_MAX);
    return PARAM_INVALID;
  }

  for (int32_t i = 0; i < context.origin_op_index_size(); ++i) {
    ctx_.opIndex2[i] = context.origin_op_index(i);
  }
  ctx_.opCount = context.origin_op_index_size();
  if (kernel_type_ == cce::ccKernelType::TE) {
    ctx_.opIndex = context.op_index();
    uint16_t *args_offset_tmp = reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data()));
    if (context.args_offset().size() / sizeof(uint16_t) < 1) {
      GELOGE(FAILED, "context.args_offset().size() / sizeof(uint16_t) less than 1");
      return FAILED;
    }

    ret = InitTVMTask(davinci_model, args_offset_tmp[0], kernel_def);
  } else if (kernel_type_ == cce::ccKernelType::CUSTOMIZED) {
    ret = InitAICPUCustomTask(davinci_model->GetOpList(), context.op_index(), kernel_def);
  } else if (kernel_type_ == cce::ccKernelType::AI_CPU) {
    ret = InitAicpuTask(davinci_model->GetOpList(), context.op_index(), kernel_def);
  } else {
    if (kernel_def.args().empty() || args_size_ == 0) {
      GELOGE(FAILED, "args is null.");
      return FAILED;
    }
    ret = InitCceTask(davinci_model, kernel_def);
  }
  GELOGD("KernelTaskInfo Init end.");

  return ret;
}

Status KernelTaskInfo::Distribute() {
  GELOGD("KernelTaskInfo Distribute Start.");
  rtError_t rt_ret;

  if (kernel_type_ == cce::ccKernelType::AI_CPU) {
    // blockDim is reserved parameter, set to 1
    rt_ret =
        rtCpuKernelLaunchWithFlag(reinterpret_cast<const void *>(so_name_.c_str()),
                                  reinterpret_cast<const void *>(kernel_name_.c_str()),
                                  1, args_, args_size_, nullptr, stream_, dump_flag_);
  } else {
    rt_ret = rtKernelLaunchWithFlag(stub_func_, block_dim_, args_, args_size_, static_cast<rtSmDesc_t *>(sm_desc_),
                                    stream_, dump_flag_);
  }

  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  uint32_t taskid = 0;
  GE_CHECK_NOTNULL(davinci_model_);
  rt_ret = rtModelGetTaskId(davinci_model_->GetRtModelHandle(), &taskid);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  task_id_ = taskid;

  return SUCCESS;
}

Status KernelTaskInfo::Release() {
  FreeRtMem(&args_);
  FreeRtMem(&flowtable_);
  FreeRtMem(&custom_info_.input_descs);
  FreeRtMem(&custom_info_.input_addrs);
  FreeRtMem(&custom_info_.output_descs);
  FreeRtMem(&custom_info_.output_addrs);
  FreeRtMem(&custom_info_.attr_handle);

  if (ctx_.argsOffset != nullptr) {
    delete[] ctx_.argsOffset;
    ctx_.argsOffset = nullptr;
  }

  rtError_t ret = (sm_desc_ != nullptr) ? rtMemFreeManaged(sm_desc_) : RT_ERROR_NONE;
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", static_cast<int>(ret));
    return FAILED;
  }
  sm_desc_ = nullptr;

  return SUCCESS;
}

Status KernelTaskInfo::InitTVMTask(DavinciModel *davinci_model, uint16_t offset, const domi::KernelDef &kernel_def) {
  GELOGD("Do InitTVMTask");
  GE_CHECK_NOTNULL(davinci_model);
  // get tvm op desc
  OpDescPtr op_desc = davinci_model->GetOpByIndex(ctx_.opIndex);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "InitTVMTaskInfo error, index is out of range!");
    return INTERNAL_ERROR;
  }

  // Update Stub
  // When training, when the the second call to DavinciModel::init() comes here, stub_func_ is already valid,
  // and does not need to be modified.
  // When inferencing, stub_func_ is different from dynamic-registration to runtime, and needs to be modified.
  string session_graph_model_id;
  const char *bin_file_key;
  davinci_model->GetUniqueId(op_desc, session_graph_model_id);
  bin_file_key = DavinciModel::GetRegisterStub(op_desc->GetName(), session_graph_model_id);
  rtError_t rt_ret;
  rt_ret = rtQueryFunctionRegistered(const_cast<char *>(bin_file_key));
  if (rt_ret != RT_ERROR_NONE) {
    stub_func_ = const_cast<char *>(bin_file_key);
  }

  const vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(davinci_model->GetRuntimeParam(), op_desc);
  const vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(davinci_model->GetRuntimeParam(), op_desc);
  const vector<void *> workspace_data_addrs =
      ModelUtils::GetWorkspaceDataAddrs(davinci_model->GetRuntimeParam(), op_desc);
  vector<void *> tensor_device_addrs;

  tensor_device_addrs.insert(tensor_device_addrs.end(), input_data_addrs.begin(), input_data_addrs.end());
  tensor_device_addrs.insert(tensor_device_addrs.end(), output_data_addrs.begin(), output_data_addrs.end());
  tensor_device_addrs.insert(tensor_device_addrs.end(), workspace_data_addrs.begin(), workspace_data_addrs.end());

  // malloc args memory
  rt_ret = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  // copy orign args
  rt_ret = rtMemcpy(args_, args_size_, static_cast<void *>(const_cast<char *>(kernel_def.args().data())), args_size_,
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  if (args_size_ <= static_cast<uint32_t>(offset) ||
      args_size_ - static_cast<uint32_t>(offset) < static_cast<uint32_t>(sizeof(void *) * tensor_device_addrs.size())) {
    GELOGE(FAILED, "offset >= kernelInfo.argsSize or copy content beyond applied memory.");
    return FAILED;
  }

  // copy args
  rt_ret = rtMemcpy(static_cast<char *>(args_) + offset, sizeof(void *) * tensor_device_addrs.size(),
                    tensor_device_addrs.data(), sizeof(void *) * tensor_device_addrs.size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  if (PropertiesManager::Instance().IsLayerNeedDump(davinci_model->Name(), op_desc->GetName())) {
    dump_flag_ = RT_KERNEL_DUMPFLAG;
    dump_args_ = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(args_) + offset +
        sizeof(void *) * input_data_addrs.size());
  }

  davinci_model_->SetZeroCopyAddr(tensor_device_addrs, static_cast<char *>(args_) + offset);
  // update origin l2 data
  string sm_desc = kernel_def.sm_desc();
  char *sm_contrl = nullptr;
  rtL2Ctrl_t *l2_ctrl_info = nullptr;
  if (!sm_desc.empty()) {
    sm_contrl = const_cast<char *>(sm_desc.data());
    l2_ctrl_info = reinterpret_cast<rtL2Ctrl_t *>(sm_contrl);

    uint64_t gen_base_addr = davinci_model->GetRtBaseAddr();

    // There is no weight for te op now. Update L2_mirror_addr by data memory base.
    uint64_t data_base_addr = (uint64_t)(uintptr_t)davinci_model->MemBase() - (uint64_t)gen_base_addr;
    const uint32_t l2_ctrl_info_data_count = 8;
    for (uint32_t data_index = 0; data_index < l2_ctrl_info_data_count; ++data_index) {
      if (l2_ctrl_info->data[data_index].L2_mirror_addr != 0) {
        l2_ctrl_info->data[data_index].L2_mirror_addr += data_base_addr;
      }
    }

    rt_ret = rtMemAllocManaged(&sm_desc_, sm_desc.size(), RT_MEMORY_SPM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }

    rt_ret = rtMemcpy(sm_desc_, sm_desc.size(), sm_desc.data(), sm_desc.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }
  GELOGD("Do InitTVMTask end");
  return SUCCESS;
}

Status KernelTaskInfo::InitAICPUCustomTask(const std::map<uint32_t, std::shared_ptr<OpDesc>> &op_list,
                                           uint32_t op_index, const domi::KernelDef &kernel_def) {
  GELOGI("Do InitAICPUCustomTask");

  auto iter = op_list.find(op_index);
  if (iter == op_list.end()) {
    GELOGE(INTERNAL_ERROR, "index is out of range, index: %u", op_index);
    return INTERNAL_ERROR;
  }

  auto op_desc = iter->second;

  const domi::KernelContext &context = kernel_def.context();
  const uint32_t kCustomAicpuArgsLen = 5;
  ctx_.argsOffset = new (std::nothrow) uint16_t[kCustomAicpuArgsLen]();
  if (ctx_.argsOffset == nullptr) {
    GELOGE(PARAM_INVALID, "ctx_.argsOffset is null!");
    return PARAM_INVALID;
  }

  if (context.args_offset().size() / sizeof(uint16_t) < kCustomAicpuArgsLen) {
    GELOGE(PARAM_INVALID, "context.args_offset().size() / sizeof(uint16_t) is less than kCustomAicpuArgsLen");
    return PARAM_INVALID;
  }

  for (uint32_t i = 0; i < kCustomAicpuArgsLen; ++i) {
    ctx_.argsOffset[i] = (reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[i];
  }

  const std::vector<void *> input_data_addrs =
      ModelUtils::GetInputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
  const std::vector<void *> output_data_addrs =
      ModelUtils::GetOutputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);

  Status ret = StoreInputOutputTensor(input_data_addrs, output_data_addrs, ModelUtils::GetInputDescs(op_desc),
                                      ModelUtils::GetOutputDescs(op_desc));
  if (ret != SUCCESS) {
    GELOGE(ret, "StoreInputOutputTensor Failed");
    return ret;
  }

  // attrHandle
  Buffer buffer;
  if (!AttrUtils::GetBytes(op_desc, ATTR_NAME_OPATTR, buffer)) {
    GELOGE(FAILED, "can't find opattr bytes!.");
    return FAILED;
  }

  uint32_t op_attr_size = buffer.GetSize();
  if (op_attr_size == 0) {
    GELOGE(PARAM_INVALID, "param op_attr_size is out of range");
    return PARAM_INVALID;
  }

  rtError_t rt_ret = rtMalloc(&custom_info_.attr_handle, op_attr_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtMemcpy(custom_info_.attr_handle, op_attr_size, buffer.GetData(), op_attr_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  // args
  char *args = const_cast<char *>(kernel_def.args().data());

  for (uint32_t i = 0; i < kCustomAicpuArgsLen; ++i) {
    if (kernel_def.args().size() < ((size_t)ctx_.argsOffset[i] + sizeof(uint64_t))) {
      GELOGE(FAILED, "ctx.argsOffset[%u]: %u + sizeof(uint64_t): %zu >= kernelDef.args().size():%zu", i,
             (uint32_t)ctx_.argsOffset[i], sizeof(uint64_t), kernel_def.args().size());
      return FAILED;
    }
  }
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[0])) =
      reinterpret_cast<uint64_t>(custom_info_.input_descs);  // arg 0
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[1])) =
      reinterpret_cast<uint64_t>(custom_info_.input_addrs);  // arg 1
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[2])) =
      reinterpret_cast<uint64_t>(custom_info_.output_descs);  // arg 2
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[3])) =
      reinterpret_cast<uint64_t>(custom_info_.output_addrs);  // arg 3
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[4])) =
      reinterpret_cast<uint64_t>(custom_info_.attr_handle);  // arg 4

  rt_ret = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtMemcpy(args_, kernel_def.args_size(), kernel_def.args().data(), kernel_def.args_size(),
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  davinci_model_->SetZeroCopyAddr(input_data_addrs, custom_info_.input_addrs);
  davinci_model_->SetZeroCopyAddr(output_data_addrs, custom_info_.output_addrs);
  return SUCCESS;
}

Status KernelTaskInfo::InitCceTask(DavinciModel *davinci_model, const domi::KernelDef &kernel_def) {
  GELOGI("Do InitCCETask");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }
  Status ret = SetContext(kernel_def);
  if (ret != SUCCESS) {
    GELOGE(ret, "SetContext Fail.");
    return ret;
  }

  string flowtable = kernel_def.flowtable();
  const domi::KernelContext &context = kernel_def.context();

  if (context.is_flowtable()) {
    if (flowtable.empty()) {
      GELOGE(FAILED, "flowtable is null.");
      return FAILED;
    }
    flowtable_size_ = flowtable.size();
  }

  // get smDesc stored in model
  string sm_desc = kernel_def.sm_desc();
  uint64_t sm_contrl_size = sm_desc.empty() ? 0 : sizeof(rtSmDesc_t);

  // Passing the memory info when the offline-model-generated to the CCE, which uses this info for address refresh
  ctx_.genDataBaseAddr = davinci_model->GetRtBaseAddr();
  ctx_.genDataBaseSize = davinci_model->TotalMemSize();
  ctx_.genWeightBaseAddr = davinci_model->GetRtWeightAddr();
  ctx_.genWeightBaseSize = davinci_model->TotalWeightsMemSize();
  ctx_.genVariableBaseAddr = davinci_model->GetRtVarAddr();
  ctx_.genVariableBaseSize = davinci_model->TotalVarMemSize();
  ctx_.l2ctrlSize = sm_contrl_size;

  if (UpdateCceArgs(sm_desc, flowtable, davinci_model, kernel_def) != SUCCESS) {
    GELOGE(ret, "update cce args fail");
    return ret;
  }

  // flowtable
  ret = SetFlowtable(flowtable, kernel_def);
  if (ret != SUCCESS) {
    GELOGE(ret, "SetFlowtable Fail");
    return ret;
  }

  // args
  rtError_t rt_ret = rtMalloc(&args_, kernel_def.args_size(), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtMemcpy(args_, kernel_def.args_size(), kernel_def.args().data(), kernel_def.args_size(),
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  // L2
  if (!sm_desc.empty()) {
    rt_ret = rtMemAllocManaged(&sm_desc_, sm_desc.size(), RT_MEMORY_SPM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }

    rt_ret = rtMemcpy(sm_desc_, sm_desc.size(), sm_desc.data(), sm_desc.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }
  return SUCCESS;
}

Status KernelTaskInfo::InitAicpuTask(const std::map<uint32_t, OpDescPtr> &op_list, uint32_t op_index,
                                     const domi::KernelDef &kernel_def) {
  GELOGI("Do InitAicpuTask");
  so_name_ = kernel_def.so_name();
  kernel_name_ = kernel_def.kernel_name();

  auto iter = op_list.find(op_index);
  if (iter == op_list.end()) {
    GELOGE(INTERNAL_ERROR, "index is out of range, index: %u", op_index);
    return INTERNAL_ERROR;
  }

  // copy args to new host memory
  std::unique_ptr<uint8_t[]> args_addr(new (std::nothrow) uint8_t[args_size_]);
  errno_t sec_ret = memcpy_s(static_cast<void *>(args_addr.get()), args_size_,
                             static_cast<const void *>(kernel_def.args().data()), args_size_);
  if (sec_ret != EOK) {
    GELOGE(FAILED, "memcpy failed, ret: %d", sec_ret);
    return FAILED;
  }

  OpDescPtr op_desc = iter->second;
  vector<void *> input_addrs = ModelUtils::GetInputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
  vector<void *> output_addrs = ModelUtils::GetOutputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
  vector<void *> io_addrs;
  io_addrs.insert(io_addrs.end(), input_addrs.begin(), input_addrs.end());
  io_addrs.insert(io_addrs.end(), output_addrs.begin(), output_addrs.end());
  if (!io_addrs.empty()) {
    // refresh io addrs
    uintptr_t io_addr =
        reinterpret_cast<uintptr_t>(args_addr.get()) + static_cast<uintptr_t>(sizeof(aicpu::AicpuParamHead));
    auto addrs_size = sizeof(uint64_t) * (io_addrs.size());
    sec_ret = memcpy_s(reinterpret_cast<void *>(io_addr), addrs_size, static_cast<void *>(io_addrs.data()), addrs_size);
    if (sec_ret != EOK) {
      GELOGE(FAILED, "memcpy failed, ret: %d", sec_ret);
      return FAILED;
    }
  }

  // malloc device memory for args
  rtError_t rt_ret = rtMalloc(static_cast<void **>(&args_), args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api(rtMalloc) failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  // copy args to device
  rt_ret = rtMemcpy(args_, args_size_, static_cast<void *>(args_addr.get()), args_size_, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api(rtMemcpy) failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  if (PropertiesManager::Instance().IsLayerNeedDump(davinci_model_->Name(), op_desc->GetName())) {
    dump_flag_ = RT_KERNEL_DUMPFLAG;
    dump_args_ = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(args_) + sizeof(aicpu::AicpuParamHead) +
        sizeof(void *) * input_addrs.size());
  }

  davinci_model_->SetZeroCopyAddr(io_addrs, static_cast<char *>(args_) + sizeof(aicpu::AicpuParamHead));
  return SUCCESS;
}

Status KernelTaskInfo::StoreInputOutputTensor(const std::vector<void *> &input_data_addrs,
                                              const std::vector<void *> &output_data_addrs,
                                              const std::vector<::tagCcAICPUTensor> &input_descs,
                                              const std::vector<::tagCcAICPUTensor> &output_descs) {
  auto input_size = input_descs.size();
  auto output_size = output_descs.size();

  // inputDescs
  rtError_t rt_ret = rtMalloc(&custom_info_.input_descs, sizeof(opTensor_t) * input_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  for (std::size_t i = 0; i < input_size; ++i) {
    rt_ret = rtMemcpy(static_cast<opTensor_t *>(custom_info_.input_descs) + i, sizeof(opTensor_t),
                      const_cast<tagOpTensor *>(&input_descs[i]), sizeof(opTensor_t), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  // inputAddrs
  rt_ret = rtMalloc(&custom_info_.input_addrs, sizeof(opTensor_t) * input_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  if (!input_data_addrs.empty()) {
    rt_ret = rtMemcpy(custom_info_.input_addrs, sizeof(void *) * input_size, &input_data_addrs[0],
                      sizeof(void *) * input_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  // outputDescs
  rt_ret = rtMalloc(&custom_info_.output_descs, sizeof(opTensor_t) * output_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  for (std::size_t i = 0; i < output_size; ++i) {
    rt_ret = rtMemcpy(static_cast<opTensor_t *>(custom_info_.output_descs) + i, sizeof(opTensor_t),
                      const_cast<tagOpTensor *>(&input_descs[i]), sizeof(opTensor_t), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  // outputAddrs
  rt_ret = rtMalloc(&custom_info_.output_addrs, sizeof(opTensor_t) * output_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  if (!output_data_addrs.empty()) {
    rt_ret = rtMemcpy(custom_info_.output_addrs, sizeof(void *) * output_size, &output_data_addrs[0],
                      sizeof(void *) * output_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  return SUCCESS;
}

Status KernelTaskInfo::SetContext(const domi::KernelDef &kernel_def) {
  const domi::KernelContext &context = kernel_def.context();
  ctx_.kernelType = static_cast<cce::ccKernelType>(context.kernel_type());
  ctx_.opId = context.op_id();
  ctx_.kernelFuncId = context.kernel_func_id();
  ctx_.isFlowtable = context.is_flowtable();
  ctx_.argsCount = context.args_count();
  if (ctx_.argsCount == 0) {
    GELOGE(INTERNAL_ERROR, "check argsCount fail:%u.", ctx_.argsCount);
    return INTERNAL_ERROR;
  }

  if (context.args_offset().size() / sizeof(uint16_t) < ctx_.argsCount) {
    GELOGE(PARAM_INVALID, "param [context.args_offset().size() / sizeof(uint16_t)] is less than [ctx_.argsCount]");
    return PARAM_INVALID;
  }

  // ctx_.argsOffset stores the offset of the internal information of agrs_, equal to the ctx_.argsCount
  ctx_.argsOffset = new (std::nothrow) uint16_t[ctx_.argsCount]();
  if (ctx_.argsOffset == nullptr) {
    GELOGE(PARAM_INVALID, "(param [ctx_.argsOffset] must not be null.");
    return PARAM_INVALID;
  }

  for (uint32_t i = 0; i < ctx_.argsCount; ++i) {
    ctx_.argsOffset[i] = (reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[i];
  }

  return SUCCESS;
}

void KernelTaskInfo::FreeRtMem(void **ptr) {
  if (ptr == nullptr || *ptr == nullptr) {
    return;
  }
  rtError_t ret = rtFree(*ptr);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", ret);
  }

  *ptr = nullptr;
}

Status KernelTaskInfo::UpdateCceArgs(std::string &sm_desc, std::string &flowtable, DavinciModel *davinci_model,
                                     const domi::KernelDef &kernel_def) {
  GE_CHECK_NOTNULL(davinci_model);
  const domi::KernelContext &context = kernel_def.context();
  char *sm_contrl = nullptr;

  if (!sm_desc.empty()) {
    sm_contrl = const_cast<char *>(sm_desc.data());
  }

  uint64_t data_base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(davinci_model->MemBase())) -
                            davinci_model->GetRtBaseAddr();
  uint64_t weight_base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(davinci_model->WeightsMemBase())) -
                              davinci_model->GetRtWeightAddr();
  uint64_t var_base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(davinci_model->VarMemBase())) -
                           davinci_model->GetRtVarAddr();
  cce::ccStatus_t cc_ret;
  if (context.is_flowtable()) {
    cc_ret = ccUpdateKernelArgs(ctx_, data_base_addr, weight_base_addr, var_base_addr,
                                const_cast<char *>(flowtable.data()), kernel_def.flowtable().size(), sm_contrl);
  } else {
    cc_ret = ccUpdateKernelArgs(ctx_, data_base_addr, weight_base_addr, var_base_addr,
                                const_cast<char *>(kernel_def.args().data()), args_size_, sm_contrl);
  }

  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    return CCE_FAILED;
  }

  return SUCCESS;
}

Status KernelTaskInfo::SetFlowtable(std::string &flowtable, const domi::KernelDef &kernel_def) {
  const domi::KernelContext &context = kernel_def.context();
  if (context.is_flowtable()) {
    rtError_t rt_ret = rtMalloc(&flowtable_, flowtable.size(), RT_MEMORY_HBM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }

    rt_ret = rtMemcpy(flowtable_, flowtable.size(), flowtable.data(), flowtable.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }

    // modify flowtable addr in args
    char *args = const_cast<char *>(kernel_def.args().data());

    if (kernel_def.args().size() <
        ((reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0] + sizeof(uint64_t))) {
      GELOGE(FAILED, "(context.args_offset().data()))[0]:%u + sizeof(uint64_t):%zu > kernelDef.args().size():%zu",
             (uint32_t)((reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0]),
             sizeof(uint64_t), kernel_def.args().size());
      return FAILED;
    }

    *(reinterpret_cast<uint64_t *>(
        args + (reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0])) =
        reinterpret_cast<uint64_t>(flowtable_);
  }
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_KERNEL, KernelTaskInfo);
}  // namespace ge
