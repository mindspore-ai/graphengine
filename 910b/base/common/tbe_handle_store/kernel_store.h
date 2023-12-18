
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

#ifndef GE_COMMON_KERNEL_STORE_H_
#define GE_COMMON_KERNEL_STORE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <securec.h>

#include "common/plugin/ge_util.h"
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "runtime/base.h"
#include "runtime/context.h"
#include "runtime/rt_preload_task.h"

#define ALIGN_MEM(MEM_SIZE, ALIGN_SIZE)                                                    \
  if (((MEM_SIZE) > 0UL) && ((ALIGN_SIZE) != 0UL) && ((MEM_SIZE) % (ALIGN_SIZE) != 0UL)) { \
    GELOGI("assign before, size[%zu]", (MEM_SIZE));                                        \
    (MEM_SIZE) = ((MEM_SIZE) + (ALIGN_SIZE)-1) / (ALIGN_SIZE) * (ALIGN_SIZE);              \
    GELOGI("assign after, size[%zu]", (MEM_SIZE));                                         \
  }

namespace ge {
using KernelBin = ge::OpKernelBin;
using KernelBinPtr = std::shared_ptr<ge::OpKernelBin>;
using CustAICPUKernelPtr = std::shared_ptr<ge::OpKernelBin>;
using TBEKernelPtr = std::shared_ptr<ge::OpKernelBin>;

struct KernelStoreItemHead {
  uint32_t magic;
  uint32_t name_len;
  uint32_t bin_len;
};

class KernelStore {
 public:
  KernelStore() = default;
  virtual ~KernelStore() = default;
  virtual bool Build();
  virtual bool PreBuild();

  virtual bool Load(const uint8_t *const data, const size_t len);

  virtual const uint8_t *Data() const;
  virtual size_t DataSize() const;
  virtual const uint8_t *PreData() const;
  virtual size_t PreDataSize() const;
  virtual void AddKernel(const KernelBinPtr &kernel);
  virtual KernelBinPtr FindKernel(const std::string &name) const;
  virtual bool IsEmpty() const;
  std::unordered_map<std::string, uint32_t> GetKernelOffset() const {
    return pre_buffer_offset_;
  }

 private:
  std::unordered_map<std::string, KernelBinPtr> kernels_;
  std::vector<uint8_t> buffer_;
  std::vector<uint8_t> pre_buffer_;
  std::unordered_map<std::string, uint32_t> pre_buffer_offset_;
};
}  // namespace ge

#endif  // GE_COMMON_KERNEL_STORE_H_
