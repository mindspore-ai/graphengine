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

#ifndef GE_COMMON_CUST_AICPU_KERNEL_STORE_H_
#define GE_COMMON_CUST_AICPU_KERNEL_STORE_H_

#include "common/tbe_handle_store/kernel_store.h"

namespace ge {
class CustAICPUKernelStore : public KernelStore {
 public:
  using KernelStore::KernelStore;
  ~CustAICPUKernelStore() override = default;

  void AddCustAICPUKernel(const CustAICPUKernelPtr &kernel);

  void LoadCustAICPUKernelBinToOpDesc(const std::shared_ptr<OpDesc> &op_desc) const;
};
}  // namespace ge

#endif  // GE_COMMON_CUST_AICPU_KERNEL_STORE_H_
