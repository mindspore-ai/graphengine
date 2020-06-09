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

#ifndef GE_SINGLE_OP_SINGLE_OP_H_
#define GE_SINGLE_OP_SINGLE_OP_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/ge_inner_error_codes.h"
#include "framework/executor/ge_executor.h"
#include "runtime/stream.h"
#include "task/op_task.h"

namespace ge {
class SingleOp {
 public:
  SingleOp() = default;
  ~SingleOp();

  Status ExecuteAsync(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  void SetStream(rtStream_t stream);

 private:
  Status ValidateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  Status UpdateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  Status GetArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);

  friend class SingleOpModel;
  rtStream_t stream_ = nullptr;
  std::vector<void *> input_addr_list_;
  std::vector<size_t> input_sizes_;
  std::vector<void *> output_addr_list_;
  std::vector<size_t> output_sizes_;
  std::vector<uintptr_t> args_;

  std::vector<OpTask *> tasks_;
  std::vector<std::vector<uintptr_t *>> arg_table_;
  bool use_physical_addr_ = false;
};
}  // namespace ge
#endif  // GE_SINGLE_OP_SINGLE_OP_H_
