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

#ifndef GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_HOST_CPU_OPS_KERNEL_INFO_H_
#define GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_HOST_CPU_OPS_KERNEL_INFO_H_

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

#include <map>
#include <string>
#include <vector>

#include "common/opskernel/ops_kernel_info_store.h"

namespace ge {
namespace st {
/*const vector<std::string> kStubOpKernelLibNameVec = {
  "AiCoreLib",
  "AicpuLib",
  "HcclLib",
  "RTSLib"
};*/
class GE_FUNC_VISIBILITY StubOpsKernelInfoStore : public OpsKernelInfoStore {
 public:
  StubOpsKernelInfoStore(std::string store_name) : store_name_(store_name) {}
  ~StubOpsKernelInfoStore() override = default;
  Status Initialize(const std::map<std::string, std::string> &options) override;
  Status Finalize() override;
  bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override;
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override;
  std::string GetOpsKernelStoreName() const {
    return store_name_;
  }

  StubOpsKernelInfoStore(const StubOpsKernelInfoStore &ops_kernel_store) = delete;
  StubOpsKernelInfoStore(const StubOpsKernelInfoStore &&ops_kernel_store) = delete;
  StubOpsKernelInfoStore &operator=(const StubOpsKernelInfoStore &ops_kernel_store) = delete;
  StubOpsKernelInfoStore &operator=(StubOpsKernelInfoStore &&ops_kernel_store) = delete;

 private:
  // store op name and OpInfo key-value pair
  std::map<std::string, ge::OpInfo> op_info_map_;
  std::string store_name_;
};
}  // namespace st
}  // namespace ge

#endif  // GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_HOST_CPU_OPS_KERNEL_INFO_H_
