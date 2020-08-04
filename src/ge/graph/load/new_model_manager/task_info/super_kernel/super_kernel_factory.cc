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

#include "super_kernel_factory.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace skt {
SuperKernelFactory &SuperKernelFactory::GetInstance() {
  static SuperKernelFactory factory;
  return factory;
}

Status SuperKernelFactory::Init() {
  if (!is_init_) {
    std::string skt_bin = "libcce_aicore.so";
    handle_ = dlopen(skt_bin.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle_ == nullptr) {
      GELOGE(FAILED, "SKT: open skt lib failed, please check LD_LIBRARY_PATH.");
    }
    rtError_t rt_ret;
    rt_ret = rtGetFunctionByName(this->sk_stub_name_.c_str(), &this->func_stub_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret,
                                                    "rtGetFunctionByName "
                                                    "failed. stub_func: %s, please export LD_LIBRARY_PATH for "
                                                    "libcce_aicore.so",
                                                    this->sk_stub_name_.c_str());
                    return FAILED;)
    rt_ret = rtGetAddrByFun(this->func_stub_, &this->func_ptr_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtGetAddrByFun failed. error: 0x%X", rt_ret);
                    return FAILED;)
    if (this->use_physical_address_ != nullptr) {
      void *skt_func = nullptr;
      rt_ret = rtKernelConfigTransArg(this->func_ptr_, sizeof(uint64_t), 0, &skt_func);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtKernelConfigTransArg failed. error: 0x%X", rt_ret);
                      return FAILED;)
      GELOGD(
        "SKT: fuseKernels super_kernel_template subFunc %p, device func "
        "address %p, device physic PC %p",
        this->func_stub_, this->func_ptr_, skt_func);
    } else {
      GELOGD(
        "SKT: fuseKernels super_kernel_template subFunc %p, device func "
        "address %p",
        this->func_stub_, this->func_ptr_);
    }
  }
  is_init_ = true;

  return SUCCESS;
}

Status SuperKernelFactory::Uninitialize() {
  is_init_ = false;
  func_stub_ = nullptr;
  func_ptr_ = nullptr;
  return SUCCESS;
}

Status SuperKernelFactory::FuseKernels(const std::vector<void *> &stub_func_list,
                                       const std::vector<void *> &args_addr_list, uint32_t block_dim, SuperKernel *&h) {
  // Iterate through the ops to be fused
  // Each subkernel to be fused contains 2 fields: fn address offset, args
  // address.
  // Generate the nav table contents. The format is as follows:
  // [[fn_ptr_address, args_addr1], [fn_ptr_address2, args_addr2],
  // ...]
  if (this->func_stub_ == nullptr) {
    GELOGW("SKT: func_stub_ is empty. Please make sure init() is run first");
    return FAILED;
  }

  size_t super_kernel_size = stub_func_list.size();
  if (super_kernel_size != args_addr_list.size()) {
    GELOGW("SKT: The size of stub_func_list doesn't match args_addr_list");
    return FAILED;
  }

  if (super_kernel_size < 2) {
    GELOGW(
      "SKT: the number of kernels being fused must be greater than or "
      "equal to 2");
    return FAILED;
  }
  GELOGI("SKT: superkernel start fuse, superkernel size %d.", stub_func_list.size());
  uint64_t nav_table[2 * stub_func_list.size()];
  uint64_t nav_table_size = 2 * stub_func_list.size() * sizeof(int64_t);

  rtError_t rt_ret;
  void *hbm_nav_table_addr = nullptr;
  if (this->use_physical_address_ != nullptr) {
    for (unsigned i = 0; i < stub_func_list.size(); i++) {
      void *sub_device_func = nullptr;
      rt_ret = rtGetAddrByFun(stub_func_list[i], &sub_device_func);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtGetAddrByFun failed. error: 0x%X", rt_ret);
                      return FAILED;)
      void *sub_device_func_pys = nullptr;
      void *args_addr_pys = nullptr;
      rt_ret = rtKernelConfigTransArg(sub_device_func, sizeof(uint64_t), 0, &sub_device_func_pys);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtKernelConfigTransArg failed. error: 0x%X", rt_ret);
                      return FAILED;)
      rt_ret = rtKernelConfigTransArg(args_addr_list[i], sizeof(uint64_t), 0, &args_addr_pys);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtKernelConfigTransArg failed. error: 0x%X", rt_ret);
                      return FAILED;)
      GELOGD(
        "SKT: fuseKernels subFunc %p, device func address %p, device "
        "physic func address %p",
        stub_func_list[i], sub_device_func, sub_device_func_pys);
      // store two uint64_t address
      // address divided by 4 because of 32bits encoding, call offset will *4 when calculating
      nav_table[i * 2] = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(sub_device_func_pys)) / 4;
      GELOGD("SKT: CALL offset %lu", nav_table[i * 2]);
      nav_table[i * 2 + 1] = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(args_addr_pys));

      GELOGD("SKT: fuseKernels args base address %lu", nav_table[i * 2 + 1]);
    }

    void *hbm_nav_table_addr_pys = nullptr;
    rt_ret = rtMalloc((void **)&hbm_nav_table_addr, nav_table_size, RT_MEMORY_HBM);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtMalloc failed. error: 0x%X", rt_ret); return FAILED;)
    rt_ret =
      rtMemcpy((void *)hbm_nav_table_addr, nav_table_size, (void *)nav_table, nav_table_size, RT_MEMCPY_HOST_TO_DEVICE);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtMemcpy failed. error: 0x%X", rt_ret);
                    GE_CHK_RT(rtFree(hbm_nav_table_addr)); return FAILED;)
    rt_ret = rtKernelConfigTransArg(hbm_nav_table_addr, sizeof(uint64_t), 0, &hbm_nav_table_addr_pys);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtKernelConfigTransArg failed. error: 0x%X", rt_ret);
                    GE_CHK_RT(rtFree(hbm_nav_table_addr)); return FAILED;)

    GELOGD("SKT: hbm_nav_table_addr %p, hbm_nav_table_addr_pys %p", hbm_nav_table_addr, hbm_nav_table_addr_pys);
    // Create the necessary metadata for the super kernel
    h = new SuperKernel(this->func_stub_, hbm_nav_table_addr_pys, nav_table_size, block_dim);
  } else {
    for (unsigned i = 0; i < stub_func_list.size(); i++) {
      void *sub_device_func = nullptr;
      rt_ret = rtGetAddrByFun(stub_func_list[i], &sub_device_func);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtGetAddrByFun failed. error: 0x%X", rt_ret);
                      return FAILED;)
      GELOGD("SKT: fuseKernels subFunc %p, device func address %p", stub_func_list[i], sub_device_func);
      // store two uint64_t address
      // address divided by 4 because of 32bits encoding, call offset will *4 when calculating
      nav_table[i * 2] = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(sub_device_func)) / 4;
      GELOGD("SKT: CALL offet %lu", nav_table[i * 2]);
      nav_table[i * 2 + 1] = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(args_addr_list[i]));
      GELOGD("SKT: fuseKernels args base address %lu", nav_table[i * 2 + 1]);
    }
    rt_ret = rtMalloc((void **)&hbm_nav_table_addr, nav_table_size, RT_MEMORY_HBM);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtMalloc failed. error: 0x%X", rt_ret); return FAILED;)
    rt_ret =
      rtMemcpy((void *)hbm_nav_table_addr, nav_table_size, (void *)nav_table, nav_table_size, RT_MEMCPY_HOST_TO_DEVICE);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtMemcpy failed. error: 0x%X", rt_ret);
                    GE_CHK_RT(rtFree(hbm_nav_table_addr)); return FAILED;)
    // Create the necessary metadata for the super kernel
    h = new SuperKernel(this->func_stub_, hbm_nav_table_addr, nav_table_size, block_dim);
  }
  return SUCCESS;
}
}  // namespace skt
}  // namespace ge
