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

#ifndef GE_COMMON_TBE_KERNEL_STORE_H_
#define GE_COMMON_TBE_KERNEL_STORE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/common/fmk_types.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"

namespace ge {
using TBEKernel = ge::OpKernelBin;
using TBEKernelPtr = std::shared_ptr<ge::OpKernelBin>;

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY TBEKernelStore {
 public:
  TBEKernelStore();
  ~TBEKernelStore() = default;
  void AddTBEKernel(const TBEKernelPtr &kernel);
  bool Build();

  bool Load(const uint8_t *data, const size_t &len);
  TBEKernelPtr FindTBEKernel(const std::string &name) const;

  void LoadTBEKernelBinToOpDesc(const std::shared_ptr<ge::OpDesc> &op_desc) const;

  const uint8_t *Data() const;
  size_t DataSize() const;

 private:
  std::unordered_map<std::string, TBEKernelPtr> kernels_;
  std::vector<uint8_t> buffer_;
};
}  // namespace ge

#endif  // GE_COMMON_TBE_KERNEL_STORE_H_
