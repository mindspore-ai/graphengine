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

#ifndef GE_HYBRID_EXECUTOR_HYBRID_EXECUTION_CONTEXT_H_
#define GE_HYBRID_EXECUTOR_HYBRID_EXECUTION_CONTEXT_H_

#include <atomic>
#include <unordered_map>
#include "common/blocking_queue.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/executor/hybrid_profiler.h"
#include "hybrid/executor/node_done_manager.h"
#include "hybrid/executor/node_state.h"
#include "hybrid/executor/rt_callback_manager.h"
#include "hybrid/model/hybrid_model.h"

namespace ge {
namespace hybrid {
struct GraphExecutionContext {
  void SetErrorCode(Status error_code);
  Status GetStatus() const;

  uint64_t session_id = 0;
  const HybridModel *model = nullptr;
  rtStream_t stream = nullptr;
  rtContext_t rt_context = nullptr;
  rtContext_t rt_gen_context = nullptr;
  std::unique_ptr<CallbackManager> callback_manager;
  NpuMemoryAllocator *allocator = nullptr;
  mutable std::unique_ptr<HybridProfiler> profiler;
  DumpProperties dump_properties;
  bool trace_enabled = false;
  bool dump_enabled = false;
  long profiling_level = 0;
  long iteration = 0;
  Status status = SUCCESS;
  mutable std::mutex mu;
};

#define RECORD_PROFILING_EVENT(context, evt_type, fmt, category, node_name, ...) \
do { \
  if ((context != nullptr) && (context)->profiler != nullptr) { \
    if (node_name != nullptr) { \
      context->profiler->RecordEvent(evt_type, "tid:%lu [%s] [%s] " fmt, GetTid(), node_name, category, ##__VA_ARGS__);\
    } else { \
      context->profiler->RecordEvent(evt_type, "tid:%lu [%s] " fmt, GetTid(), category, ##__VA_ARGS__); \
    }\
  } \
} while (0)

#define RECORD_MODEL_EXECUTION_EVENT(context, fmt, ...) \
  RECORD_PROFILING_EVENT((context), HybridProfiler::GENERAL, fmt, "ModelExecutor", nullptr, ##__VA_ARGS__)

#define RECORD_SHAPE_INFERENCE_EVENT(context, name, fmt, ...) \
  RECORD_PROFILING_EVENT((context), HybridProfiler::SHAPE_INFERENCE, fmt, "ShapeInference", name,  ##__VA_ARGS__)

#define RECORD_COMPILE_EVENT(context, name, fmt, ...) \
  RECORD_PROFILING_EVENT((context), HybridProfiler::COMPILE, fmt, "Compilation", name,  ##__VA_ARGS__)

#define RECORD_EXECUTION_EVENT(context, name, fmt, ...) \
  RECORD_PROFILING_EVENT((context), HybridProfiler::EXECUTION, fmt, "Execution", name,  ##__VA_ARGS__)

#define RECORD_CALLBACK_EVENT(context, name, fmt, ...) \
  RECORD_PROFILING_EVENT((context), HybridProfiler::CALLBACK, fmt, "Callback", name,  ##__VA_ARGS__)
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_HYBRID_EXECUTION_CONTEXT_H_
