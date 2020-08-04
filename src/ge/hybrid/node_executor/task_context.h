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

#ifndef GE_HYBRID_KERNEL_TASK_CONTEXT_H_
#define GE_HYBRID_KERNEL_TASK_CONTEXT_H_

#include <map>
#include <mutex>
#include <vector>
#include "external/ge/ge_api_error_codes.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/executor/rt_callback_manager.h"
#include "hybrid/model/node_item.h"

namespace ge {
namespace hybrid {
class GraphExecutionContext;

class TaskContext {
 public:
  static std::unique_ptr<TaskContext> Create(const NodeItem &node_item, GraphExecutionContext *graph_context);

  ~TaskContext();

  int NumInputs() const;
  int NumOutputs() const;
  size_t NumWorkspaces() const;
  const NodeItem &GetNodeItem() const;
  const char *GetNodeName() const;
  TensorValue *MutableInput(int index);
  void ReleaseInput(int index);
  const TensorValue *GetInput(int index) const;
  const TensorValue *GetOutput(int index) const;
  TensorValue *MutableOutput(int index);
  rtStream_t GetStream();
  int64_t GetSessionId();

  Status SetOutput(int index, const TensorValue &tensor);
  Status AllocateOutput(int index, const GeTensorDesc &tensor_desc, TensorValue **tensor);
  Status AllocateOutputs();
  Status AllocateWorkspaces();
  Status AllocateWorkspace(size_t size, void **buffer, void *ori_addr = nullptr);

  const GraphExecutionContext *GetExecutionContext() { return execution_context_; }

  Status AllocateTemp(size_t size, TensorValue &tensor);
  void *MutableWorkspace(int index);
  const void *GetVarBaseAddr();

  Status RegisterCallback(const std::function<void()> &callback_fun) const;

  Status PropagateOutputs();

  Status GetStatus() const;

  void SetStatus(Status status);

 private:
  explicit TaskContext(GraphExecutionContext *execution_context);
  TensorValue *inputs_start_ = nullptr;
  TensorValue *outputs_start_ = nullptr;
  static string TensorDesc2String(const GeTensorDesc &desc);
  Status AllocateTensor(const GeTensorDesc &tensor_desc, TensorValue &tensor);

  GraphExecutionContext *execution_context_;
  const NodeItem *node_item_ = nullptr;
  Status status_ = SUCCESS;
  std::vector<void *> workspaces_;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_KERNEL_TASK_CONTEXT_H_
