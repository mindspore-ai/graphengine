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

#ifndef GE_HYBRID_EXECUTOR_EXECUTOR_EXECUTION_ENGINE_H_
#define GE_HYBRID_EXECUTOR_EXECUTOR_EXECUTION_ENGINE_H_

#include "common/thread_pool.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/rt_callback_manager.h"
#include "hybrid/node_executor/task_context.h"

namespace ge {
namespace hybrid {
class ExecutionEngine {
 public:
  explicit ExecutionEngine(GraphExecutionContext *context, CallbackManager *callback_manager);
  ~ExecutionEngine() = default;

  Status Start();

 private:
  Status PropagateOutputs(const NodeItem &node_item, TaskContext &task_context);

  Status ExecutionProcess();

  Status ExecuteAsync(NodeState &node_state, TaskContext &task_context, const std::function<void()> &callback);

  GraphExecutionContext *context_;
  CallbackManager *callback_manager_;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_EXECUTOR_EXECUTOR_EXECUTION_ENGINE_H_
