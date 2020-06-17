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

#ifndef INC_FRAMEWORK_GE_RUNTIME_OP_INFO_H_
#define INC_FRAMEWORK_GE_RUNTIME_OP_INFO_H_

#include <memory>
#include <string>
#include <vector>

namespace ge {
namespace model_runner {
struct TensorInfo {
  int64_t GetShapeSize() const {
    int64_t res = 1;
    if (dims.empty()) {
      return 0;
    }
    for (auto dim : dims) {
      res *= dim;
    }
    return res;
  }

  int64_t GetDim(uint32_t index) {
    if (index >= dims.size()) {
      return 0;
    }
    return dims[index];
  }

  std::vector<int64_t> dims;
  uint32_t datatype;
  uint32_t format;
  uint32_t real_dim_cnt;
  uint32_t size;
  bool is_output;
};

struct OpInfo {
  uint32_t index;
  std::string name;
  std::string type;
  bool var_is_broadcast;
  std::vector<uintptr_t> input_addrs;
  std::vector<uintptr_t> output_addrs;
  std::vector<TensorInfo> input_tensors;
  std::vector<TensorInfo> output_tensors;
  std::vector<TensorInfo> weight_tensors;
  std::vector<std::string> src_name;
  std::vector<int64_t> src_index;
  std::string weight_data;
};

using TensorInfoPtr = std::shared_ptr<TensorInfo>;
using OpInfoPtr = std::shared_ptr<OpInfo>;
}  // namespace model_runner
}  // namespace ge
#endif  // INC_FRAMEWORK_GE_RUNTIME_OP_INFO_H_
