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

#include "single_op/task/tbe_task_builder.h"

#include <mutex>
#include <utility>
#include <vector>

#include "common/helper/model_helper.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/load/new_model_manager/task_info/task_info.h"
#include "graph/manager/graph_var_manager.h"
#include "runtime/rt.h"
#include "single_op/task/build_task_utils.h"

namespace ge {
namespace {
std::mutex g_reg_mutex;

inline void GetKernelName(const OpDescPtr &op_desc, std::string &kernel_name) {
  (void)AttrUtils::GetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);
}

inline TBEKernelPtr GetTbeKernel(const OpDescPtr &op_desc) {
  return op_desc->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
}
}  // namespace

KernelHolder::KernelHolder(const char *stub_func, std::shared_ptr<ge::OpKernelBin> kernel_bin)
    : stub_func_(stub_func), bin_handle_(nullptr), kernel_bin_(std::move(kernel_bin)) {}

KernelHolder::~KernelHolder() {
  if (bin_handle_ != nullptr) {
    GE_CHK_RT(rtDevBinaryUnRegister(bin_handle_));
  }
}

KernelBinRegistry::~KernelBinRegistry() {
  for (auto &iter : registered_bins_) {
    delete iter.second;
    iter.second = nullptr;
  }
}

const char *KernelBinRegistry::GetUnique(const string &stub_func) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = unique_stubs_.find(stub_func);
  if (it != unique_stubs_.end()) {
    return it->c_str();
  } else {
    it = unique_stubs_.insert(unique_stubs_.end(), stub_func);
    return it->c_str();
  }
}

const char *KernelBinRegistry::GetStubFunc(const std::string &stub_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = registered_bins_.find(stub_name);
  if (iter != registered_bins_.end()) {
    return iter->second->stub_func_;
  }

  return nullptr;
}

bool KernelBinRegistry::AddKernel(const std::string &stub_name, const KernelHolder *holder) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto ret = registered_bins_.emplace(stub_name, holder);
  return ret.second;
}

TbeTaskBuilder::TbeTaskBuilder(const std::string &model_name, const OpDescPtr &op_desc,
                               const domi::KernelDef &kernel_def)
    : op_desc_(op_desc), kernel_def_(kernel_def), stub_name_(model_name + "/" + op_desc->GetName() + "_tvmbin") {}

Status TbeTaskBuilder::DoRegisterBinary(const OpKernelBin &kernel_bin, void **bin_handle) const {
  rtDevBinary_t binary;
  binary.version = 0;
  binary.data = kernel_bin.GetBinData();
  binary.length = kernel_bin.GetBinDataSize();
  binary.magic = RT_DEV_BINARY_MAGIC_ELF;
  auto ret = rtDevBinaryRegister(&binary, bin_handle);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtDevBinaryRegister failed, bin key = %s, rt ret = %d", stub_name_.c_str(),
           static_cast<int>(ret));
    return RT_FAILED;
  }

  return SUCCESS;
}

Status TbeTaskBuilder::DoRegisterMeta(void *bin_handle) {
  std::string meta_data;
  (void)AttrUtils::GetStr(op_desc_, TVM_ATTR_NAME_METADATA, meta_data);
  GELOGI("TBE: meta data: %s", meta_data.empty() ? "null" : meta_data.c_str());
  if (!meta_data.empty()) {
    auto rt_ret = rtMetadataRegister(bin_handle, meta_data.c_str());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtMetadataRegister failed. bin key = %s, meta_data = %s, rt ret = %d", stub_name_.c_str(),
             meta_data.c_str(), static_cast<int>(rt_ret));
      return RT_FAILED;
    }
  }

  return SUCCESS;
}

Status TbeTaskBuilder::DoRegisterFunction(void *bin_handle, const char *stub_name, const char *kernel_name) {
  auto rt_ret = rtFunctionRegister(bin_handle, stub_name, stub_name, kernel_name, FUNC_MODE_NORMAL);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtFunctionRegister failed. bin key = %s, kernel name = %s, rt ret = %d", stub_name, kernel_name,
           static_cast<int>(rt_ret));
    return RT_FAILED;
  }

  return SUCCESS;
}

Status TbeTaskBuilder::DoRegisterKernel(const ge::OpKernelBin &tbe_kernel, const char *bin_file_key,
                                        void **bin_handle) {
  std::string kernel_name;
  GetKernelName(op_desc_, kernel_name);

  void *handle = nullptr;
  auto ret = DoRegisterBinary(tbe_kernel, &handle);
  if (ret != SUCCESS) {
    return ret;
  }

  ret = DoRegisterMeta(handle);
  if (ret != SUCCESS) {
    GE_CHK_RT(rtDevBinaryUnRegister(handle));
    return ret;
  }

  ret = DoRegisterFunction(handle, bin_file_key, kernel_name.c_str());
  if (ret != SUCCESS) {
    GE_CHK_RT(rtDevBinaryUnRegister(handle));
    return ret;
  }

  GELOGI("Register function succeeded: kernel_name = %s", kernel_name.c_str());
  *bin_handle = handle;
  return SUCCESS;
}

Status TbeTaskBuilder::RegisterKernel(TbeOpTask &task) {
  KernelBinRegistry &registry = KernelBinRegistry::GetInstance();
  // check if already registered
  const char *stub_func = registry.GetStubFunc(stub_name_);
  if (stub_func != nullptr) {
    task.SetStubFunc(stub_name_, stub_func);
    return SUCCESS;
  }

  // to avoid repeat register
  std::lock_guard<std::mutex> lock(g_reg_mutex);
  // check again
  stub_func = registry.GetStubFunc(stub_name_);
  if (stub_func == nullptr) {
    stub_func = registry.GetUnique(stub_name_);
    GELOGI("RegisterKernel begin, stub_func = %s", stub_func);

    auto tbe_kernel = GetTbeKernel(op_desc_);
    if (tbe_kernel == nullptr) {
      GELOGE(PARAM_INVALID, "OP EXT ATTR NAME TBE_KERNEL not found. op = %s", op_desc_->GetName().c_str());
      return PARAM_INVALID;
    }

    auto *holder = new (std::nothrow) KernelHolder(stub_func, tbe_kernel);
    if (holder == nullptr) {
      GELOGE(MEMALLOC_FAILED, "create KernelHodler failed.");
      return MEMALLOC_FAILED;
    }

    void *bin_handle = nullptr;
    auto ret = DoRegisterKernel(*tbe_kernel, stub_func, &bin_handle);
    if (ret == SUCCESS) {
      holder->SetBinHandle(bin_handle);
      if (!registry.AddKernel(stub_name_, holder)) {
        // should not happen. only one thread can reach here
        delete holder;
        holder = nullptr;
        GELOGE(INTERNAL_ERROR, "Add kernel failed. stub name = %s", stub_name_.c_str());
        return INTERNAL_ERROR;
      }
    } else {
        delete holder;
        holder = nullptr;
    }
  }

  task.SetStubFunc(stub_name_, stub_func);
  return SUCCESS;
}

Status TbeTaskBuilder::GetSmDesc(void **sm_desc, const SingleOpModelParam &param) const {
  const std::string &sm_desc_str = kernel_def_.sm_desc();
  if (sm_desc_str.empty()) {
    *sm_desc = nullptr;
  } else {
    GELOGD("To process sm desc, size = %zu", sm_desc_str.size());
    char *sm_control = const_cast<char *>(sm_desc_str.data());
    auto *l2_ctrl_info = reinterpret_cast<rtL2Ctrl_t *>(sm_control);
    uint64_t gen_base_addr = param.base_addr;
    // There is no weight for te op now. Update L2_mirror_addr by data memory base.
    uint64_t data_base_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.mem_base)) - gen_base_addr;
    for (auto &data_index : l2_ctrl_info->data) {
      if (data_index.L2_mirror_addr != 0) {
        data_index.L2_mirror_addr += data_base_addr;
      }
    }

    auto rtRet = rtMemAllocManaged(sm_desc, sm_desc_str.size(), RT_MEMORY_SPM);
    if (rtRet != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtMemAllocManaged failed, ret: %d", static_cast<int>(rtRet));
      return RT_FAILED;
    }

    rtRet = rtMemcpy(*sm_desc, sm_desc_str.size(), sm_desc_str.data(), sm_desc_str.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rtRet != RT_ERROR_NONE) {
      (void)rtMemFreeManaged(*sm_desc);
      GELOGE(RT_FAILED, "rtMemcpy, ret: %d", static_cast<int>(rtRet));
      return RT_FAILED;
    }
  }

  return SUCCESS;
}

Status TbeTaskBuilder::SetKernelArgs(TbeOpTask &task, const SingleOpModelParam &param) {
  uint8_t *args = nullptr;
  size_t arg_size = kernel_def_.args_size();
  auto rtRet = rtMallocHost(reinterpret_cast<void **>(&args), arg_size);
  if (rtRet != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMallocHost failed, size = %zu, ret = %d", arg_size, static_cast<int>(rtRet));
    return RT_FAILED;
  }

  task.SetKernelArgs(args, arg_size, kernel_def_.block_dim());

  rtRet = rtMemcpy(args, arg_size, kernel_def_.args().data(), arg_size, RT_MEMCPY_HOST_TO_HOST);
  if (rtRet != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMemcpy args failed, size = %zu, ret = %d", arg_size, static_cast<int>(rtRet));
    return RT_FAILED;
  }

  const domi::KernelContext &context = kernel_def_.context();
  const auto *args_offset_tmp = reinterpret_cast<const uint16_t *>(context.args_offset().data());
  uint16_t offset = *args_offset_tmp;

  // copy args
  std::vector<void *> tensor_device_addr_vec = BuildTaskUtils::GetKernelArgs(op_desc_, param);
  void *src_addr = reinterpret_cast<void *>(tensor_device_addr_vec.data());
  uint64_t src_len = sizeof(void *) * tensor_device_addr_vec.size();
  rtRet = rtMemcpy(args + offset, arg_size - offset, src_addr, src_len, RT_MEMCPY_HOST_TO_HOST);
  if (rtRet != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMemcpy addresses failed, ret = %d", static_cast<int>(rtRet));
    return RT_FAILED;
  }

  return SUCCESS;
}

Status TbeTaskBuilder::BuildTask(TbeOpTask &task, const SingleOpModelParam &param) {
  GELOGD("Build tbe task begin");
  auto ret = SetKernelArgs(task, param);
  if (ret != SUCCESS) {
    return ret;
  }

  ret = RegisterKernel(task);
  if (ret != SUCCESS) {
    return ret;
  }

  void *stub_func = nullptr;
  auto rtRet = rtGetFunctionByName(stub_name_.c_str(), &stub_func);
  if (rtRet != SUCCESS) {
    GELOGE(RT_FAILED, "rtGetFunctionByName failed.");
    return RT_FAILED;
  }

  task.SetStubFunc(stub_name_, stub_func);
  return SUCCESS;
}
}  // namespace ge
