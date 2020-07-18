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

#ifndef GE_HYBRID_KERNEL_AICORE_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_AICORE_NODE_EXECUTOR_H_

#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "hybrid/node_executor/aicore/aicore_task_compiler.h"
#include "hybrid/node_executor/node_executor.h"
#include <map>
#include <mutex>

namespace ge {
namespace hybrid {
class AiCoreNodeTaskRegistry {
 public:
  ~AiCoreNodeTaskRegistry() = default;

  static AiCoreNodeTaskRegistry &GetInstance() {
    static AiCoreNodeTaskRegistry instance;
    return instance;
  }

  std::shared_ptr<NodeTask> GetTask(const std::string &node_key);
  bool AddTask(const std::string &node_key, const std::shared_ptr<NodeTask> task);

 private:
  AiCoreNodeTaskRegistry() = default;
  std::map<std::string, std::shared_ptr<NodeTask>> reg_node_tasks_;
  std::mutex mutex_;
};

class AiCoreNodeTask : public NodeTask {
 public:
  explicit AiCoreNodeTask(std::vector<std::unique_ptr<AiCoreOpTask>> &&tasks);
  ~AiCoreNodeTask() override = default;
  bool IsSupportDynamicShape() override;
  Status UpdateTilingData(TaskContext &context) override;

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;

 private:
  std::vector<std::unique_ptr<AiCoreOpTask>> tasks_;
};

class AiCoreNodeExecutor : public NodeExecutor {
 public:
  Status Initialize() override;
  Status LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const override;
  Status CompileTask(const HybridModel &model, const NodePtr &node, std::shared_ptr<NodeTask> &task) const override;

 private:
  static Status GenNodeKey(const NodePtr &node, std::string &node_key);
  std::unique_ptr<AiCoreTaskCompiler> compiler_;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_KERNEL_AICORE_NODE_EXECUTOR_H_
