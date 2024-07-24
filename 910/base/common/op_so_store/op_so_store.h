/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GE_COMMON_OP_SO_STORE_H
#define GE_COMMON_OP_SO_STORE_H

#include <cstdint>
#include <string>
#include <vector>
#include <securec.h>
#include "common/plugin/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/op_desc.h"
#include "graph/op_so_bin.h"

namespace ge {
struct SoStoreItemHead {
  uint32_t magic;
  uint32_t so_name_len;
  uint32_t vendor_name_len;
  uint32_t bin_len;
};

struct SoStoreHead {
  uint32_t so_num;
};

class OpSoStore {
public:
  OpSoStore() = default;
  virtual ~OpSoStore() = default;
  virtual bool Build();

  virtual bool Load(const uint8_t *const data, const size_t len);

  virtual const uint8_t *Data() const;
  virtual size_t DataSize() const;
  virtual void AddKernel(const OpSoBinPtr &so_bin_ptr);
  bool CalculateAndAllocMem();
  std::vector<OpSoBinPtr> GetSoBin() const;
  uint32_t GetKernelNum() const;
private:
  std::vector<OpSoBinPtr> kernels_;
  std::shared_ptr<uint8_t> buffer_;
  uint32_t buffer_size_ = 0;
  uint32_t so_num_ = 0;
};
}  // namespace ge

#endif  // GE_COMMON_OP_SO_STORE_H
