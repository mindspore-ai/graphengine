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

#ifndef GE_HYBRID_KERNEL_AICPU_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_AICPU_NODE_EXECUTOR_H_

#include "external/graph/types.h"
#include "cce/aicpu_engine_struct.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
class AicpuTfNodeTask : public NodeTask {
 public:
  AicpuTfNodeTask(const NodePtr &node, const domi::TaskDef &task_def) : node_(node), task_def_(task_def) {}

  Status Init(const HybridModel &model);

  ~AicpuTfNodeTask() override = default;

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;

 private:
  Status InitExtInfo();
  Status InitForDependComputeTask();

  Status SetShapeToBuf(const GeShape &shape, int64_t buf[], uint32_t buf_size);
  void GetShapeFromBuf(const int64_t buf[], uint32_t buf_size, std::vector<int64_t> &dims);
  Status UpdateOutputShapeFromExtInfo();

  Status UpdateShapeAndDataByResultSummary(TaskContext &context);

  Status UpdateShapeToOutputDesc(const GeShape &shape_new, size_t output_index, GeTensorDescPtr &output_desc);

  ///
  /// read result summary and prepare copy task memory.
  /// @param context task context
  /// @param out_shape_hbm if scalar, TensorBuffer->data is null, size=0
  /// @return SUCCESS:success other:failed
  ///
  Status ReadResultSummaryAndPrepareMemory(TaskContext &context,
                                           std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);
  Status CopyDataToHbm(TaskContext &context, const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);

  Status UpdateShapeByHbmBuffer(TaskContext &context, const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);

  // common method
  static Status AllocTensorBuffer(size_t size, std::unique_ptr<TensorBuffer> &tensor_buffer);
  static Status EnsureSessionCreated(uint64_t session_id);
  static Status GenMemCopyTask(uint64_t count, STR_FWK_OP_KERNEL &task, string &task_info);

 private:
  const NodePtr node_;
  // just reference.
  const domi::TaskDef &task_def_;

  UnknowShapeOpType unknown_type_ = DEPEND_IN_SHAPE;

  size_t input_num_ = 0;

  size_t output_num_ = 0;

  // kernel buf, device mem
  std::unique_ptr<TensorBuffer> kernel_buf_;

  std::unique_ptr<TensorBuffer> kernel_workspace_;

  // input and output addr, device mem
  std::unique_ptr<TensorBuffer> input_output_addr_;

  // ext info addr, device mem
  std::unique_ptr<TensorBuffer> ext_info_addr_dev_;
  std::unique_ptr<uint8_t[]> ext_info_addr_host_;
  uint32_t ext_info_num_ = 0;

  // just used for depend DEPEND_COMPUTE op
  std::unique_ptr<TensorBuffer> copy_task_args_buf_;
  std::vector<std::unique_ptr<TensorBuffer>> output_summary_;
  std::vector<aicpu::FWKAdapter::ResultSummary> output_summary_host_;

  std::unique_ptr<TensorBuffer> copy_ioaddr_dev_;

  std::unique_ptr<TensorBuffer> copy_input_release_flag_dev_;
  std::unique_ptr<TensorBuffer> copy_input_data_size_dev_;
  std::unique_ptr<TensorBuffer> copy_input_src_dev_;
  std::unique_ptr<TensorBuffer> copy_input_dst_dev_;
};

class AiCpuNodeExecutor : public NodeExecutor {
 public:
  Status LoadTask(const HybridModel &model, const NodePtr &node, std::shared_ptr<NodeTask> &task) const override;

  Status PrepareTask(NodeTask &task, TaskContext &context) const override;
};

}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_KERNEL_AICPU_NODE_EXECUTOR_H_
