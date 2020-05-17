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

#ifndef SUPER_KERNEL_FACTORY_H
#define SUPER_KERNEL_FACTORY_H

#include <vector>
#include "super_kernel.h"
#include "framework/common/debug/log.h"

namespace ge {
namespace skt {
class SuperKernelFactory {
 private:
  void *func_stub_ = nullptr;
  void *func_ptr_ = nullptr;
  std::string sk_stub_name_ = "_Z21super_kernel_templatePmm";
  const char *use_physical_address_ = getenv("GE_USE_PHYSICAL_ADDRESS");
  bool is_init_ = false;
  SuperKernelFactory(){};

 public:
  SuperKernelFactory(SuperKernelFactory const &) = delete;
  void operator=(SuperKernelFactory const &) = delete;
  static SuperKernelFactory &GetInstance();
  SuperKernelFactory(const std::string &sk_stub_name_, const std::string &bin_file);
  Status Init();
  Status Uninitialize();
  Status FuseKernels(const std::vector<void *> &stub_func_list, const std::vector<void *> &args_addr_list,
                     uint32_t block_dim, SuperKernel *&h);
};
}  // namespace skt
}  // namespace ge
#endif
