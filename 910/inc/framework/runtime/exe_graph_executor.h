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

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXE_GRAPH_EXECUTOR_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXE_GRAPH_EXECUTOR_H_
#include "graph/ge_error_codes.h"

#include "common/ge_visibility.h"
#include "exe_graph_resource_guard.h"
#include "subscriber/executor_subscriber_c.h"
namespace gert {
enum SubExeGraphType { kInitExeGraph, kMainExeGraph, kDeInitExeGraph, kSubExeGraphTypeEnd };
using ResourceGuardPtr = std::unique_ptr<ResourceGuard>;
class VISIBILITY_EXPORT ExeGraphExecutor {
 public:
  using ExecuteFunc = UINT32 (*)(void *);
  using ExecuteWithCallbackFunc = UINT32 (*)(int32_t, void *, ExecutorSubscriber *);
  ge::graphStatus Load() const {
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus UnLoad() const {
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 设置图执行的输入/输出，需要注意的是，使用者需要自己保证inputs/outputs刷新完全！！！
   */
  ge::graphStatus SpecifyInputs(void *const *inputs, size_t start, size_t num) const;
  ge::graphStatus SpecifyOutputs(void *const *outputs, size_t num) const;
  ge::graphStatus Execute() const;
  ge::graphStatus Execute(SubExeGraphType sub_graph_type, ExecutorSubscriber *callback) const;

  const void *GetExecutionData() const {
    return execution_data_;
  }
  void SetExecutionData(void *execution_data, ResourceGuardPtr resource_guard);

  void SetExecuteFunc(ExecuteFunc execute_func, ExecuteWithCallbackFunc callback_func);

 private:
  friend class ModelV2ExecutorTestHelper;

  void *execution_data_{nullptr};
  ExecuteFunc execute_func_{nullptr};
  ExecuteWithCallbackFunc execute_with_callback_func_{nullptr};
  ResourceGuardPtr resource_guard_;
};
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXE_GRAPH_EXECUTOR_H_
