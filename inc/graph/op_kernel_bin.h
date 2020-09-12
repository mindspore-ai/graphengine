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

#ifndef INC_GRAPH_OP_KERNEL_BIN_H_
#define INC_GRAPH_OP_KERNEL_BIN_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ge {
class OpKernelBin {
 public:
  OpKernelBin(std::string name, std::vector<char> &&data) : name_(std::move(name)), data_(std::move(data)) {}

  ~OpKernelBin() = default;

  const std::string &GetName() const { return name_; }
  const uint8_t *GetBinData() const { return (const uint8_t *)data_.data(); }
  size_t GetBinDataSize() const { return data_.size(); }
  OpKernelBin(const OpKernelBin &) = delete;
  const OpKernelBin &operator=(const OpKernelBin &) = delete;

 private:
  std::string name_;
  std::vector<char> data_;
};

using OpKernelBinPtr = std::shared_ptr<OpKernelBin>;
const char *const OP_EXTATTR_NAME_TBE_KERNEL = "tbeKernel";
const char *const OP_EXTATTR_CUSTAICPU_KERNEL = "cust_aicpu_kernel";
}  // namespace ge

#endif  // INC_GRAPH_OP_KERNEL_BIN_H_
