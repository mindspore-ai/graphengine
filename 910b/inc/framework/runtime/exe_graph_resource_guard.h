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

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXE_GRAPH_RESOURCE_GUARD_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXE_GRAPH_RESOURCE_GUARD_H_
#include <memory>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include "common/ge_visibility.h"

namespace gert {
class VISIBILITY_EXPORT ResourceGuard {
 public:
  void *ResetExecutionData(std::unique_ptr<uint8_t[]> execution_data);
  void *GetExecutionData();
  virtual ~ResourceGuard() = default;

 private:
  std::unique_ptr<uint8_t[]> execution_data_holder_;
};

/*
* 这里将原来的resource guard平移为子类，
* 若后续有诉求需要拆分model上使用的resource guarder和sub exe graph上使用的resource guard
* 可以再行拆分
*/
class VISIBILITY_EXPORT TopologicalResourceGuard : public ResourceGuard {
 public:
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

  ~TopologicalResourceGuard() override;

 private:
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
}
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXE_GRAPH_RESOURCE_GUARD_H_
