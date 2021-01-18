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

#ifndef GE_HYBRID_KERNEL_AICORE_OP_TASK_H_
#define GE_HYBRID_KERNEL_AICORE_OP_TASK_H_

#include <memory>
#include <vector>
#include "common/ge_inner_error_codes.h"
#include "runtime/stream.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/node_executor/task_context.h"
#include "proto/task.pb.h"
#include "register/op_tiling.h"

namespace ge {
namespace hybrid {
class AiCoreOpTask {
 public:
  AiCoreOpTask() = default;
  virtual ~AiCoreOpTask() = default;

  virtual Status Init(const OpDesc &op_desc, const domi::TaskDef &task_def);

  bool IsDynamicShapeSupported();

  // do preparation with shape(without actual io memory)
  Status PrepareWithShape(TaskContext &context);

  virtual Status UpdateArgs(TaskContext &task_context);

  Status LaunchKernel(rtStream_t stream);

  const std::string& GetName() const;

  bool GetClearAtomic() const {return clear_atomic_;}

  uint32_t GetBlockDim() const {return block_dim_;}

 protected:
  Status UpdateTilingInfo(TaskContext &context);
  virtual std::string GetKeyForOpParamSize() const;
  virtual Status CalcTilingInfo(const NodePtr &node, optiling::OpRunInfo &tiling_info);

  std::unique_ptr<TensorBuffer> tiling_buffer_ = nullptr;
  std::string tiling_data_;
  uintptr_t *arg_base_ = nullptr;
  uint32_t max_arg_count_ = 0;

 private:
  static Status ValidateTaskDef(const domi::TaskDef &task_def);
  Status InitWithTaskDef(const OpDesc &node, const domi::TaskDef &task_def);
  Status InitTilingInfo(const OpDesc &op_desc);
  Status RegisterTbeHandle(const OpDesc &op_desc);

  std::string stub_name_;
  void *stub_func_ = nullptr;
  std::unique_ptr<uint8_t[]> args_ = nullptr;
  uint32_t args_size_ = 0;
  uint32_t block_dim_ = 1;
  bool clear_atomic_ = true;
};

class AtomicAddrCleanOpTask : public AiCoreOpTask {
 public:
  Status Init(const OpDesc &op_desc, const domi::TaskDef &task_def) override;
  Status UpdateArgs(TaskContext &task_context) override;

 protected:
  std::string GetKeyForOpParamSize() const override;
  Status CalcTilingInfo(const NodePtr &node, optiling::OpRunInfo &tiling_info) override;

 private:
  Status InitAtomicAddrCleanIndices(const OpDesc &op_desc);
  std::vector<int> atomic_output_indices_;
  std::vector<int> atomic_workspace_indices_;
};
}  // namespace hybrid
}  // namespace ge
#endif //GE_HYBRID_KERNEL_AICORE_OP_TASK_H_
