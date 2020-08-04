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

#include "hybrid/executor/worker/task_compile_engine.h"
#include "init/gelib.h"
#include "framework/common/debug/log.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
namespace {
uint32_t kDefaultWorkerCnt = 4;
uint32_t kDefaultDeviceId = 0;
}  // namespace
TaskCompileEngine::TaskCompileEngine(GraphExecutionContext *context) : context_(context), pool_(kDefaultWorkerCnt) {}

TaskCompileEngine::~TaskCompileEngine() {
  if (rt_context_ != nullptr) {
    GELOGD("To destroy compile context: %p.", rt_context_);
    GE_CHK_RT(rtCtxDestroy(rt_context_));
  }
}

Status TaskCompileEngine::Init() {
  GELOGD("Start to init CompileEngine");
  rtContext_t current_ctx = nullptr;
  GE_CHK_RT(rtCtxGetCurrent(&current_ctx));
  GE_CHK_RT_RET(rtCtxCreate(&rt_context_, RT_CTX_GEN_MODE, kDefaultDeviceId));
  GELOGD("Context created for compiling. ctx = %p", rt_context_);
  GE_CHK_RT_RET(rtCtxSetCurrent(current_ctx));
  return SUCCESS;
}

void TaskCompileEngine::Reset() {
  complete_queue_.Push(nullptr);  // ensure iteration can stop
  unique_ptr<ResultQueueEntry> entry;
  while (true) {
    complete_queue_.Pop(entry);
    if (entry == nullptr) {
      break;
    }

    if (entry->future != nullptr) {
      entry->future->wait();
    }
  }

  complete_queue_.Clear();
}

Status TaskCompileEngine::Start(ThreadPool &pool) {
  pool.commit([&]() { (void)this->CompileProcess(); });

  worker_future_ = pool_.commit([&]() -> Status { return this->DistributeCompiledTasks(); });

  if (!worker_future_.valid()) {
    GELOGE(INTERNAL_ERROR, "Failed to start worker thread");
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status TaskCompileEngine::CompileProcess() {
  auto &compile_queue = context_->compile_queue;
  while (true) {
    NodeStatePtr node_state;
    // Stop() will not be invoked, Pop won't failed
    (void)compile_queue.Pop(node_state);

    // EOF
    if (node_state == nullptr) {
      GELOGD("Got EOF");
      complete_queue_.Push(unique_ptr<ResultQueueEntry>());
      break;
    }

    auto entry = unique_ptr<ResultQueueEntry>(new (std::nothrow) ResultQueueEntry());
    GE_CHECK_NOTNULL(entry);
    entry->node_state = node_state;

    auto node_item = *node_state->node_item;
    if (node_item.kernel_task != nullptr) {
      GELOGD("use precompiled task. node name = %s", node_item.NodeName().c_str());
      node_state->kernel_task = node_item.kernel_task;
      complete_queue_.Push(std::move(entry));
      continue;
    }

    auto ret = CompileAsync(*node_state->node_item, *entry);
    if (ret == SUCCESS) {
      complete_queue_.Push(std::move(entry));
      continue;
    }

    // On Error
    worker_future_.wait();
    Reset();
    return CompileDone(ret);
  }

  Status ret = worker_future_.get();
  Reset();
  return CompileDone(ret);
}

Status TaskCompileEngine::CompileDone(Status status) {
  if (status != SUCCESS) {
    GELOGE(status, "Error occurred while compiling node.");
    context_->OnError(status);
  } else {
    context_->execution_queue.Push(nullptr);
  }
  GELOGI("CompileEngine worker END. ret = %u", status);
  return status;
}

Status TaskCompileEngine::DoCompile(const NodeItem &node_item, NodeState &node_state) {
  RECORD_COMPILE_EVENT(context_, node_state.GetName().c_str(), "Start");
  GE_CHK_RT_RET(rtCtxSetCurrent(rt_context_));
  auto ret = node_item.node_executor->CompileTask(*context_->model, node_item.node, node_state.kernel_task);
  RECORD_COMPILE_EVENT(context_, node_state.GetName().c_str(), "End");
  GE_CHK_STATUS_RET(ret, "Failed to create task for node: %s", node_item.NodeName().c_str());
  GELOGI("Compiling node %s successfully", node_state.GetName().c_str());
  return SUCCESS;
}

Status TaskCompileEngine::CompileAsync(const NodeItem &node_item, ResultQueueEntry &entry) {
  auto node_state = entry.node_state;
  auto f = pool_.commit([this, node_item, node_state]() -> Status { return DoCompile(node_item, *node_state); });

  if (!f.valid()) {
    GELOGE(INTERNAL_ERROR, "Failed to commit compile task");
    return INTERNAL_ERROR;
  }

  entry.future = unique_ptr<std::future<Status>>(new (std::nothrow) std::future<Status>(std::move(f)));
  GE_CHECK_NOTNULL(entry.future);
  return SUCCESS;
}

Status TaskCompileEngine::DistributeCompiledTasks() {
  GELOGD("DistributeCompiledTasks start.");
  auto &execute_queue = context_->execution_queue;
  unique_ptr<ResultQueueEntry> entry;
  bool ret = SUCCESS;
  while (true) {
    if (!complete_queue_.Pop(entry)) {
      GELOGE(INTERNAL_ERROR, "Failed to pop item from queue");
      ret = INTERNAL_ERROR;
      break;
    }

    // EOF
    if (entry == nullptr) {
      break;
    }

    // if has compile future
    if (entry->future != nullptr) {
      ret = entry->future->get();
      if (ret != SUCCESS) {
        break;
      }
    }

    execute_queue.Push(entry->node_state);
  }

  GELOGD("DistributeCompiledTasks out. ret = %u.", ret);
  return ret;
}
}  // namespace hybrid
}  // namespace ge
