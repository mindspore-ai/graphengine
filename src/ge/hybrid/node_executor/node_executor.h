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

#ifndef GE_HYBRID_KERNEL_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_NODE_EXECUTOR_H_

#include "external/ge/ge_api_error_codes.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/node.h"
#include "proto/task.pb.h"
#include "task_context.h"

namespace ge {
namespace hybrid {
class HybridModel;

class NodeTask {
 public:
  NodeTask() = default;
  virtual ~NodeTask() = default;
  virtual Status UpdateArgs(TaskContext &context) = 0;
  virtual Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) = 0;
  virtual Status Init(TaskContext &context) { return SUCCESS; }
};

class NodeExecutor {
 public:
  NodeExecutor() = default;
  virtual ~NodeExecutor() = default;

  virtual Status Initialize() { return SUCCESS; }

  virtual Status Finalize() { return SUCCESS; }

  virtual Status LoadTask(const HybridModel &model, const NodePtr &node, std::shared_ptr<NodeTask> &task) const;

  virtual Status CompileTask(const HybridModel &model, const NodePtr &node, std::shared_ptr<NodeTask> &task) const;

  virtual Status PrepareTask(NodeTask &task, TaskContext &context) const;
  virtual Status ExecuteTask(NodeTask &task, TaskContext &context, const std::function<void()> &callback) const;
};

class NodeExecutorManager {
 public:
  enum class ExecutorType { AICORE, GE_LOCAL, AICPU_TF, AICPU_CUSTOM, COMPILED_SUBGRAPH, HCCL, RESERVED };

  static NodeExecutorManager &GetInstance() {
    static NodeExecutorManager instance;
    return instance;
  }

  Status CalcOpRunningParam(Node &node) const;

  void RegisterExecutorBuilder(ExecutorType executor_type, const std::function<NodeExecutor *()> &builder);

  Status EnsureInitialized();

  Status GetExecutor(Node &node, const NodeExecutor **executor) const;

  ExecutorType ResolveExecutorType(Node &node) const;

  std::map<ExecutorType, std::unique_ptr<NodeExecutor>> executors_;
  std::map<ExecutorType, std::function<NodeExecutor *()>> builders_;
  std::map<std::string, std::shared_ptr<OpsKernelInfoStore>> kernel_stores_;
  std::map<std::string, NodeExecutorManager::ExecutorType> engine_mapping_;
  std::mutex mu_;
  bool initialized_ = false;
};

class NodeExecutorRegistrar {
 public:
  NodeExecutorRegistrar(NodeExecutorManager::ExecutorType executor_type, NodeExecutor *(*builder)());
  ~NodeExecutorRegistrar() = default;
};
}  // namespace hybrid
}  // namespace ge

#define REGISTER_NODE_EXECUTOR_BUILDER(engine_type, executor) \
  REGISTER_NODE_EXECUTOR_BUILDER_UNIQ_HELPER(__COUNTER__, engine_type, executor)

#define REGISTER_NODE_EXECUTOR_BUILDER_UNIQ_HELPER(ctr, engine_type, executor) \
  REGISTER_NODE_EXECUTOR_BUILDER_UNIQ(ctr, engine_type, executor)

#define REGISTER_NODE_EXECUTOR_BUILDER_UNIQ(ctr, engine_type, executor)               \
  static ::ge::hybrid::NodeExecutorRegistrar register_##ctr __attribute__((unused)) = \
    ::ge::hybrid::NodeExecutorRegistrar(                                              \
      engine_type, []() -> ::ge::hybrid::NodeExecutor * { return new (std::nothrow) executor(); })

#endif  // GE_HYBRID_KERNEL_NODE_EXECUTOR_H_
