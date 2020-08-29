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

#ifndef GE_HYBRID_KERNEL_HOST_CPU_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_HOST_CPU_NODE_EXECUTOR_H_

#include "inc/kernel.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
class HostNodeTaskBase : public NodeTask {
 public:
  explicit HostNodeTaskBase(const NodePtr &node) : node_(node) {}
  ~HostNodeTaskBase() = default;
  virtual Status UpdateArgs(TaskContext &context);
  virtual Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback);

 protected:
  NodePtr node_;

 private:
  virtual Status Execute(TaskContext &context, const std::vector<GeTensorPtr> &inputs,
                         std::vector<GeTensorPtr> &outputs) = 0;
  virtual Status ProcessInputs(TaskContext &context, std::vector<GeTensorPtr> &inputs);
  virtual Status ProcessOutputs(TaskContext &context, std::vector<GeTensorPtr> &outputs);
};

class CpuKernelNodeTask : public HostNodeTaskBase {
 public:
  explicit CpuKernelNodeTask(const NodePtr &node) : HostNodeTaskBase(node) {}
  ~CpuKernelNodeTask() = default;

 private:
  Status Execute(TaskContext &context, const std::vector<GeTensorPtr> &inputs,
                 std::vector<GeTensorPtr> &outputs) override;
};

class HostKernelNodeTask : public HostNodeTaskBase {
 public:
  explicit HostKernelNodeTask(const NodePtr &node) : HostNodeTaskBase(node) {}
  ~HostKernelNodeTask() = default;

 private:
  Status Execute(TaskContext &context, const std::vector<GeTensorPtr> &inputs,
                 std::vector<GeTensorPtr> &outputs) override;
};

class HostCpuNodeTask : public HostNodeTaskBase {
 public:
  explicit HostCpuNodeTask(const NodePtr &node) : HostNodeTaskBase(node) {}
  ~HostCpuNodeTask() = default;

 private:
  Status Execute(TaskContext &context, const std::vector<GeTensorPtr> &inputs,
                 std::vector<GeTensorPtr> &outputs) override;
  Status ProcessInputs(TaskContext &context, std::vector<GeTensorPtr> &inputs) override;
  Status ProcessOutputs(TaskContext &context, std::vector<GeTensorPtr> &outputs) override;
};

class HostCpuNodeExecutor : public NodeExecutor {
 public:
  Status PrepareTask(NodeTask &task, TaskContext &context) const override;

  Status LoadTask(const HybridModel &model, const NodePtr &node, std::shared_ptr<NodeTask> &task) const override;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_KERNEL_HOST_CPU_NODE_EXECUTOR_H_
