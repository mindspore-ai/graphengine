/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

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
