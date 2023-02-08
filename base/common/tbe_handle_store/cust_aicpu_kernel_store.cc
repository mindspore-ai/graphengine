/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include "common/tbe_handle_store/cust_aicpu_kernel_store.h"

namespace ge {
CustAICPUKernelStore::CustAICPUKernelStore() : KernelStore() {}

void CustAICPUKernelStore::AddCustAICPUKernel(const CustAICPUKernelPtr &kernel) {
  AddKernel(kernel);
}

void CustAICPUKernelStore::LoadCustAICPUKernelBinToOpDesc(const std::shared_ptr<OpDesc> &op_desc) const {
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
  const auto &kernel_bin = FindKernel(op_desc->GetName());
  if (kernel_bin != nullptr) {
    GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, kernel_bin),
                    GELOGW("LoadKernelCustAICPUBinToOpDesc: SetExtAttr for kernel_bin failed"));
    GELOGI("Load cust aicpu kernel:%s, %zu", kernel_bin->GetName().c_str(), kernel_bin->GetBinDataSize());
  }
}
}  // namespace ge
