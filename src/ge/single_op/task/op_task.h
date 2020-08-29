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
#include <external/graph/tensor.h>

#include "common/dump/dump_op.h"
#include "common/dump/dump_properties.h"
#include "common/ge_inner_error_codes.h"
#include "graph/op_kernel_bin.h"
#include "runtime/stream.h"
#include "graph/node.h"

namespace ge {
enum OpTaskType {
  OP_TASK_TBE = 0,
  OP_TASK_AICPU,
  OP_TASK_AICPUCC,
  OP_TASK_INVALID,
};

class OpTask {
 public:
  OpTask() = default;
  virtual ~OpTask() = default;
  virtual Status LaunchKernel(rtStream_t stream) = 0;
  virtual Status UpdateRunInfo(const vector<GeTensorDesc> &input_desc, const vector<GeTensorDesc> &output_desc) {
    return UNSUPPORTED;
  }
  virtual Status LaunchKernel(const std::vector<void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<void *> &workspaces, rtStream_t stream) {
    return UNSUPPORTED;
  }
  virtual OpTaskType GetOpTaskType() = 0;
  virtual const void *GetIOAddr() const = 0;
  const vector<int64_t> &GetWorkspaceSizes() const;
  void SetWorkspaceSizes(const vector<int64_t> &workspace_sizes);

 private:
  std::vector<int64_t> workspace_sizes_;

 protected:
  Status OpenDump(void *arg, const OpDescPtr &op_desc, rtStream_t stream);
  DumpProperties dump_properties_;
  DumpOp dump_op_;
};

class TbeOpTask : public OpTask {
 public:
  ~TbeOpTask() override;
  Status LaunchKernel(rtStream_t stream) override;
  OpTaskType GetOpTaskType() override { return OP_TASK_TBE; }
  const void *GetIOAddr() const override { return nullptr; }
  void SetSmDesc(void *sm_desc);
  void SetStubFunc(const std::string &name, const void *stub_func);
  void SetKernelArgs(std::unique_ptr<uint8_t[]> &&args, size_t arg_size, uint32_t block_dim, const OpDescPtr &op_desc);

  Status UpdateRunInfo(const vector<GeTensorDesc> &input_desc, const vector<GeTensorDesc> &output_desc) override;

  Status LaunchKernel(const vector<void *> &inputs, const vector<void *> &outputs, const vector<void *> &workspaces,
                      rtStream_t stream) override;

  const void *GetArgs() const;
  size_t GetArgSize() const;
  const std::string &GetStubName() const;
  void EnableDynamicSupport(const NodePtr &node, void *tiling_buffer, size_t max_tiling_size);

 private:
  static Status UpdateTensorDesc(const GeTensorDesc &src_tensor, GeTensorDesc &dst_tensor);
  Status UpdateNodeByShape(const vector<GeTensorDesc> &input_desc, const vector<GeTensorDesc> &output_desc);

  const void *stub_func_ = nullptr;
  std::unique_ptr<uint8_t[]> args_;
  size_t arg_size_ = 0;
  uint32_t block_dim_ = 1;
  void *sm_desc_ = nullptr;
  std::string stub_name_;

  void *tiling_buffer_ = nullptr;
  uint32_t max_tiling_size_ = 0;
  std::string tiling_data_;
  NodePtr node_;
  OpDescPtr op_desc_;
};

class AiCpuTask : public OpTask {
 public:
  AiCpuTask() = default;
  ~AiCpuTask() override;

  Status LaunchKernel(rtStream_t stream) override;
  OpTaskType GetOpTaskType() override { return OP_TASK_AICPU; }
  const void *GetIOAddr() const override;

 private:
  friend class AiCpuTaskBuilder;
  void *workspace_addr_ = nullptr;
  std::string task_info_;
  void *args_ = nullptr;
  size_t arg_size_ = 0;
  std::string op_type_;
  void *io_addr_ = nullptr;
  OpDescPtr op_desc_;
};

class AiCpuCCTask : public OpTask {
 public:
  AiCpuCCTask() = default;
  ~AiCpuCCTask() override;
  AiCpuCCTask(const AiCpuCCTask &) = delete;
  AiCpuCCTask &operator=(const AiCpuCCTask &) = delete;

  Status LaunchKernel(rtStream_t stream) override;
  OpTaskType GetOpTaskType() override { return OP_TASK_AICPUCC; }
  const void *GetIOAddr() const override;
  const void *GetArgs() const;
  void SetKernelArgs(std::unique_ptr<uint8_t[]> args, size_t arg_size);
  void SetSoName(const std::string &so_name);
  void SetkernelName(const std::string &kernel_Name);
  void SetIoAddr(void *io_addr);
  size_t GetArgSize() const;

 private:
  friend class AiCpuCCTaskBuilder;
  std::string so_name_;
  std::string kernel_name_;
  std::unique_ptr<uint8_t[]> args_;
  size_t arg_size_ = 0;
  uint32_t block_dim_ = 1;
  void *sm_desc_ = nullptr;
  void *io_addr_ = nullptr;
  OpDescPtr op_desc_;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_TASK_OP_TASK_H_
