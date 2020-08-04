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

#ifndef GE_HYBRID_EXECUTOR_COMPILE_TASK_COMPILE_ENGINE_H_
#define GE_HYBRID_EXECUTOR_COMPILE_TASK_COMPILE_ENGINE_H_

#include <memory>
#include <thread>
#include "common/thread_pool.h"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
class TaskCompileEngine {
 public:
  explicit TaskCompileEngine(GraphExecutionContext *context);

  ~TaskCompileEngine();

  Status Init();

  Status Start(ThreadPool &pool);

 private:
  struct ResultQueueEntry {
    NodeStatePtr node_state;
    std::unique_ptr<std::future<Status>> future;
  };

  Status CompileProcess();

  Status CompileDone(Status status);

 private:
  Status DoCompile(const NodeItem &node_item, NodeState &node_state);
  Status CompileAsync(const NodeItem &node_item, ResultQueueEntry &entry);
  Status DistributeCompiledTasks();
  void Reset();

  rtContext_t rt_context_ = nullptr;
  GraphExecutionContext *context_;
  BlockingQueue<unique_ptr<ResultQueueEntry>> complete_queue_;
  ThreadPool pool_;
  std::future<Status> worker_future_;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_EXECUTOR_COMPILE_TASK_COMPILE_ENGINE_H_
