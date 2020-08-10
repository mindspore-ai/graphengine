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

#ifndef GE_HYBRID_NODE_EXECUTOR_HOST_AICPU_KERNEL_FACTORY_H_
#define GE_HYBRID_NODE_EXECUTOR_HOST_AICPU_KERNEL_FACTORY_H_

#include <functional>
#include <map>
#include <string>
#include "common/ge/ge_util.h"
#include "hybrid/node_executor/hostaicpu/kernel/kernel.h"

namespace ge {
namespace hybrid {
namespace host_aicpu {
using KERNEL_CREATOR_FUNC = std::function<std::shared_ptr<Kernel>(const NodePtr &)>;

/**
 * manage all the host_aicpu_kernel, support create kernel.
 */
class KernelFactory {
 public:
  static KernelFactory &Instance();

  /**
   *  @brief create Kernel.
   *  @param [in] node
   *  @return not nullptr success
   *  @return nullptr fail
   */
  std::shared_ptr<Kernel> CreateKernel(const NodePtr &node);

  /**
   *  @brief Register Kernel create function.
   *  @param [in] type: Kernel type
   *  @param [in] func: Kernel create func
   */
  void RegisterCreator(const std::string &type, const KERNEL_CREATOR_FUNC &func);

  KernelFactory(const KernelFactory &) = delete;
  KernelFactory &operator=(const KernelFactory &) = delete;
  KernelFactory(KernelFactory &&) = delete;
  KernelFactory &operator=(KernelFactory &&) = delete;

 private:
  KernelFactory() = default;
  ~KernelFactory() = default;

  // the kernel creator function map
  std::map<std::string, KERNEL_CREATOR_FUNC> kernel_creator_map_;
};

class KernelRegistrar {
 public:
  KernelRegistrar(const std::string &type, const KERNEL_CREATOR_FUNC &func) {
    KernelFactory::Instance().RegisterCreator(type, func);
  }
  ~KernelRegistrar() = default;

  KernelRegistrar(const KernelRegistrar &) = delete;
  KernelRegistrar &operator=(const KernelRegistrar &) = delete;
  KernelRegistrar(KernelRegistrar &&) = delete;
  KernelRegistrar &operator=(KernelRegistrar &&) = delete;
};

#define REGISTER_KERNEL_CREATOR(type, clazz)                                                              \
  std::shared_ptr<Kernel> Creator_##type##Kernel(const NodePtr &node) { return MakeShared<clazz>(node); } \
  KernelRegistrar g_##type##Kernel_creator(#type, Creator_##type##Kernel)
}  // namespace host_aicpu
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_NODE_EXECUTOR_HOST_AICPU_KERNEL_FACTORY_H_
