/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_HOST_CPU_OPS_KERNEL_BUILDER_H_
#define GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_HOST_CPU_OPS_KERNEL_BUILDER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include "common/opskernel/ops_kernel_builder.h"

namespace ge {
namespace st {
class GE_FUNC_VISIBILITY StubOpsKernelBuilder : public OpsKernelBuilder {
 public:
  Status Initialize(const map<std::string, std::string> &options) override;

  Status Finalize() override;

  Status CalcOpRunningParam(Node &node) override;

  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override;
};
}  // namespace st
}  // namespace ge

#endif  // GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_HOST_CPU_OPS_KERNEL_BUILDER_H_
