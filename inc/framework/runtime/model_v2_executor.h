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

namespace gert {
enum SubExeGraphType { kInitExeGraph, kMainExeGraph, kDeInitExeGraph, kSubExeGraphTypeEnd };
static constexpr char *kSubExeGraphTypeStrs[kSubExeGraphTypeEnd] = {(char *)"Init", (char *)"Main", (char *)"DeInit"};
inline const char *GetSubExeGraphTypeStr(SubExeGraphType type) {
  return kSubExeGraphTypeStrs[type];
}

class ResourceGuard {
 public:
  void *ResetExecutionData(std::unique_ptr<uint8_t[]> execution_data);
  void ResetAnyValue(std::unique_ptr<uint8_t[]> any_values, size_t count);
  void PushNode(void *node);
  void PushWatcher(void *watcher);
  void *ResetNodesArray(std::unique_ptr<uint8_t[]> nodes_array);
  void *ResetStartNodesArray(std::unique_ptr<uint8_t[]> start_nodes_array);
  void *ResetNodesIndgreeArray(std::unique_ptr<uint8_t[]> nodes_indgree_array);
  void *ResetNodesWaitIndgreeArray(std::unique_ptr<uint8_t[]> nodes_indgree_array);
  void *ResetInputsArray(std::unique_ptr<uint8_t[]> inputs_array);
  void *ResetOutputsArray(std::unique_ptr<uint8_t[]> outputs_array);
  void *ResetWatchersArray(std::unique_ptr<uint8_t[]> watchers_array);
  void *ResetReadyQueue(void *ready_queue);
  void *ResetBuffer(std::unique_ptr<uint8_t[]> buffer);
  void *ResetComputeNodeInfo(std::unique_ptr<uint8_t[]> compute_node_info);
  void *ResetKernelExtendInfo(std::unique_ptr<uint8_t[]> kernel_extend_info);
  void *ResetModelDesc(std::unique_ptr<uint8_t[]> model_desc);

  ~ResourceGuard();

 private:
  std::unique_ptr<uint8_t[]> execution_data_holder_;
  size_t any_values_num_;
  std::unique_ptr<uint8_t[]> any_values_guard_;

  std::vector<std::unique_ptr<void, decltype(&free)>> nodes_guarder_;
  std::vector<std::unique_ptr<void, decltype(&free)>> watchers_guarder_;
  std::unique_ptr<uint8_t[]> continuous_buffer_guarder_;
  std::unique_ptr<uint8_t[]> buffer_guarder_;
  std::unique_ptr<uint8_t[]> compute_node_info_guarder_;
  std::unique_ptr<uint8_t[]> kernel_extend_info_guarder_;
  std::unique_ptr<uint8_t[]> model_desc_guarder_;

  std::unique_ptr<uint8_t[]> nodes_array_guarder_;
  std::unique_ptr<uint8_t[]> start_nodes_array_guarder_;
  std::unique_ptr<uint8_t[]> nodes_indgree_array_guarder_;
  std::unique_ptr<uint8_t[]> nodes_wait_indgree_array_guarder_;
  std::unique_ptr<uint8_t[]> inputs_array_guarder_;
  std::unique_ptr<uint8_t[]> outputs_array_guarder_;
  std::unique_ptr<uint8_t[]> watchers_array_guarder_;
  std::unique_ptr<void, decltype(&free)> ready_queue_guarder_{nullptr, nullptr};
};

struct ModelExecuteArg {
  rtStream_t stream;
};
static_assert(std::is_standard_layout<ModelExecuteArg>::value, "The class ModelExecuteArg must be a POD");

class ExeGraphExecutor {
 public:
  // todo unload时释放anyvalue资源
  ge::graphStatus Load() {
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus UnLoad() {
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 设置图执行的输入/输出，需要注意的是，使用者需要自己保证inputs/outputs刷新完全！！！
   */
  ge::graphStatus SpecifyInputs(void **inputs, size_t start, size_t num);
  ge::graphStatus SpecifyOutputs(void **outputs, size_t num);
  ge::graphStatus Execute();

  const void *GetExecutionData() const {
    return execution_data_;
  }

  ResourceGuard &GetResourceGuard();
  void *SetExecutionData(std::unique_ptr<uint8_t[]> execution_data);

 private:
  friend class ModelV2ExecutorTestHelper;

  void *execution_data_;
  ResourceGuard resource_guard_;
};
class ModelV2Executor {
 public:
  static std::unique_ptr<ModelV2Executor> Create(const ge::ComputeGraphPtr &root_graph);

  ge::graphStatus Load();
  ge::graphStatus Execute(const ModelExecuteArg &arg, Tensor **inputs, size_t input_num, Tensor **outputs,
                          size_t output_num);
  ge::graphStatus ExecuteSync(Tensor **inputs, size_t input_num, Tensor **outputs, size_t output_num);
  ge::graphStatus UnLoad();

  const ModelDesc &GetModelDesc() const;
  void SetModelDesc(ModelDesc *model_desc);
  ModelV2Executor(const ModelV2Executor &) = delete;
  ModelV2Executor(ModelV2Executor &&) = delete;
  ModelV2Executor &operator=(const ModelV2Executor &) = delete;
  ModelV2Executor &operator=(ModelV2Executor &&) = delete;

 private:
  friend class ModelV2ExecutorBuilder;
  friend class ModelV2ExecutorTestHelper;
  ModelV2Executor() = default;

 private:
  std::array<ExeGraphExecutor, kSubExeGraphTypeEnd> graphs_;
  ResourceGuard resource_guard_;
  ModelDesc *model_desc_ = nullptr;
  rtStream_t default_stream_ = nullptr;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
