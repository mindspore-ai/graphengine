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
#include "cce/aicpu_engine_struct.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info.h"
#include "init/gelib.h"

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
  const OpDescPtr &GetOpdesc() const { return op_desc_; }
  Status OpenDump(rtStream_t stream);
  void SetIoAddrsForDump(const vector<uint64_t> &io_addrs_for_dump) { io_addrs_for_dump_ = io_addrs_for_dump; }
  virtual Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc, const std::vector<DataBuffer> &input_buffers,
                              std::vector<GeTensorDesc> &output_desc, std::vector<DataBuffer> &output_buffers,
                              rtStream_t stream) {
    return UNSUPPORTED;
  }

 private:
  std::vector<int64_t> workspace_sizes_;

 protected:
  DumpProperties dump_properties_;
  DumpOp dump_op_;
  OpDescPtr op_desc_;
  std::vector<uint64_t> io_addrs_for_dump_;
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
};

class AiCpuBaseTask : public OpTask {
 public:
  AiCpuBaseTask() = default;
  ~AiCpuBaseTask() override;
  const UnknowShapeOpType GetUnknownType() const { return unknown_type_; }

 protected:
  Status SetExtInfoAndType(const std::string &kernel_ext_info);

  Status UpdateExtInfo(const std::vector<GeTensorDesc> &input_desc, std::vector<GeTensorDesc> &output_desc,
                       rtStream_t stream);
  Status UpdateOutputShape(vector<GeTensorDesc> &output_desc);
  Status UpdateShapeToOutputDesc(const GeShape &shape_new, GeTensorDesc &output_desc);

 protected:
  size_t num_inputs_ = 0;
  size_t num_outputs_ = 0;
  UnknowShapeOpType unknown_type_ = DEPEND_IN_SHAPE;
  std::unique_ptr<ge::hybrid::AicpuExtInfoHandler> aicpu_ext_handle_;
  void *ext_info_addr_dev_ = nullptr;
};

class AiCpuTask : public AiCpuBaseTask {
 public:
  AiCpuTask() = default;
  ~AiCpuTask() override;

  Status LaunchKernel(rtStream_t stream) override;
  OpTaskType GetOpTaskType() override { return OP_TASK_AICPU; }
  const void *GetIOAddr() const override;

  Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc, const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc, std::vector<DataBuffer> &output_buffers,
                      rtStream_t stream) override;
  Status SetMemCopyTask(const domi::KernelExDef &kernel_def);

 private:
  Status SetIO(const vector<void *> &inputs, vector<void *> &outputs);

  // for copy task.
  Status InitForSummaryAndCopy();
  Status UpdateShapeAndDataByResultSummary(vector<GeTensorDesc> &output_desc, vector<DataBuffer> &outputs,
                                           rtStream_t stream);
  Status ReadResultSummaryAndPrepareMemory();

  Status CopyDataToHbm(vector<DataBuffer> &outputs, rtStream_t stream);
  Status PrepareCopyInputs(vector<DataBuffer> &outputs);

  Status UpdateShapeByHbmBuffer(vector<GeTensorDesc> &output_desc);

  friend class AiCpuTaskBuilder;
  void *workspace_addr_ = nullptr;
  std::string task_info_;
  // device addr
  void *args_ = nullptr;
  size_t arg_size_ = 0;
  std::string op_type_;
  // device addr
  void *io_addr_ = nullptr;

  bool dynamic_flag_ = false;
  // for copy task
  void *copy_task_args_buf_;
  void *copy_workspace_buf_;

  std::vector<void *> output_summary_;
  std::vector<aicpu::FWKAdapter::ResultSummary> output_summary_host_;

  void *copy_ioaddr_dev_;

  void *copy_input_release_flag_dev_;
  void *copy_input_data_size_dev_;
  void *copy_input_src_dev_;
  void *copy_input_dst_dev_;

  vector<void *> out_shape_hbm_;
};

class AiCpuCCTask : public AiCpuBaseTask {
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

  Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc, const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc, std::vector<DataBuffer> &output_buffers,
                      rtStream_t stream) override;

 private:
  friend class AiCpuCCTaskBuilder;
  std::string so_name_;
  std::string kernel_name_;
  std::unique_ptr<uint8_t[]> args_;
  size_t arg_size_ = 0;
  uint32_t block_dim_ = 1;
  void *sm_desc_ = nullptr;
  void *io_addr_ = nullptr;
  bool is_custom_ = false;
  uint32_t dump_flag_ = RT_KERNEL_DEFAULT;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_TASK_OP_TASK_H_
