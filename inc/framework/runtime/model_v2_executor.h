/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
#define AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
#include <memory>
#include "graph/compute_graph.h"
#include "graph/ge_error_codes.h"
#include "model_desc.h"
#include "runtime/stream.h"
#include "exe_graph/runtime/tensor.h"
#include "common/ge_visibility.h"
#include "exe_graph_resource_guard.h"
#include "exe_graph_executor.h"
#include "subscriber/executor_subscribers_scheduler.h"
#include "common/ge_types.h"
#include "gert_api.h"
#include "mem_allocator.h"

namespace gert {
enum class ExecutorState { kInit, kLoaded };
enum SubExeGraphType { kInitExeGraph, kMainExeGraph, kDeInitExeGraph, kSubExeGraphTypeEnd };
inline const char *GetSubExeGraphTypeStr(SubExeGraphType type) {
  constexpr const char *kSubExeGraphTypeStrs[kSubExeGraphTypeEnd] = {"Init", "Main", "DeInit"};
  return kSubExeGraphTypeStrs[type];
}

enum class ExecuteArgIndex { kExternalAllocator = -2, kStream = -1, kEnd };

struct ModelExecuteArg {
  rtStream_t stream;
  ExternalAllocators *external_allocator;
  ModelExecuteArg() : stream(nullptr), external_allocator(nullptr) {}
  ModelExecuteArg(rtStream_t stream_, ExternalAllocators *external_allocator_ = nullptr)
      : stream(stream_), external_allocator(external_allocator_) {}
};
static_assert(std::is_standard_layout<ModelExecuteArg>::value, "The class ModelExecuteArg must be a POD");
class VISIBILITY_EXPORT ModelV2Executor {
 public:
  static std::unique_ptr<ModelV2Executor> Create(const ge::ComputeGraphPtr &root_graph, const ge::ModelData &model_data,
                                                 const std::shared_ptr<ge::GeRootModel> &root_model);
  static std::unique_ptr<ModelV2Executor> Create(const ge::ComputeGraphPtr &root_graph);

  ge::graphStatus Load();
  ge::graphStatus Execute(const ModelExecuteArg &arg, Tensor **inputs, size_t input_num, Tensor **outputs,
                          size_t output_num);
  ge::graphStatus ExecuteSync(Tensor **inputs, size_t input_num, Tensor **outputs, size_t output_num);
  ge::graphStatus UnLoad();

  const ModelDesc &GetModelDesc() const;
  void SetModelDesc(ModelDesc *model_desc);
  ExeGraphExecutor *GetExeGraphExecutor(SubExeGraphType type) {
    if (type >= kSubExeGraphTypeEnd) {
      return nullptr;
    }
    return &graphs_[static_cast<size_t>(type)];
  }
  ExecutorSubscribersScheduler &GetSubscribers();
  const ExecutorSubscribersScheduler &GetSubscribers() const;

  ModelV2Executor(const ModelV2Executor &) = delete;
  ModelV2Executor(ModelV2Executor &&) = delete;
  ModelV2Executor &operator=(const ModelV2Executor &) = delete;
  ModelV2Executor &operator=(ModelV2Executor &&) = delete;

 private:
  friend class ModelV2ExecutorBuilder;
  friend class ModelV2ExecutorTestHelper;
  ModelV2Executor();

 private:
  std::array<ExeGraphExecutor, kSubExeGraphTypeEnd> graphs_;
  ResourceGuard resource_guard_;
  ModelDesc *model_desc_ = nullptr;
  rtStream_t default_stream_ = nullptr;
  ExecutorSubscribersScheduler subscribers_;
  ExecutorState state_ = ExecutorState::kInit;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
