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

#ifndef GE_SINGLE_OP_TASK_OP_TASK_H_
#define GE_SINGLE_OP_TASK_OP_TASK_H_

#include <memory>
#include <string>

#include "runtime/stream.h"
#include "common/ge_inner_error_codes.h"
#include "graph/op_kernel_bin.h"

namespace ge {
enum OpTaskType {
  OP_TASK_TBE = 0,
  OP_TASK_AICPU,
  OP_TASK_INVALID,
};

class OpTask {
 public:
  OpTask() = default;
  virtual ~OpTask() = default;
  virtual Status LaunchKernel(rtStream_t stream) = 0;
  virtual OpTaskType GetOpTaskType() = 0;
};

class TbeOpTask : public OpTask {
 public:
  ~TbeOpTask() override;
  Status LaunchKernel(rtStream_t stream) override;
  OpTaskType GetOpTaskType() override { return OP_TASK_TBE; }

  void SetSmDesc(void *sm_desc);
  void SetStubFunc(const std::string &name, const void *stub_func);
  void SetKernelArgs(void *args, size_t arg_size, uint32_t block_dim);
  const void *GetArgs() const;
  size_t GetArgSize() const;
  const std::string &GetStubName() const;

 private:
  const void *stub_func_ = nullptr;
  void *args_ = nullptr;
  size_t arg_size_ = 0;
  uint32_t block_dim_ = 1;
  void *sm_desc_ = nullptr;
  std::string stub_name_;
};

class AiCpuTask : public OpTask {
 public:
  AiCpuTask() = default;
  ~AiCpuTask() override;

  Status LaunchKernel(rtStream_t stream) override;
  OpTaskType GetOpTaskType() override { return OP_TASK_AICPU; }
  const void *GetIOAddr() const;

 private:
  friend class AiCpuTaskBuilder;
  void *workspace_addr_ = nullptr;
  std::string task_info_;
  void *args_ = nullptr;
  size_t arg_size_ = 0;
  std::string op_type_;
  void *io_addr_ = nullptr;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_TASK_OP_TASK_H_
