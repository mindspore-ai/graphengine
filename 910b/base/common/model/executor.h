/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef GE_COMMON_EXECUTOR_H
#define GE_COMMON_EXECUTOR_H

#include "external/ge/ge_api_types.h"
#include "graph/ge_local_context.h"
#include "graph/manager/graph_manager_utils.h"

namespace ge {
struct RunArgs {
  GraphNodePtr graph_node;
  GraphId graph_id;
  uint64_t session_id;
  struct error_message::Context error_context;
  std::vector<ge::Tensor> input_tensor;
  FlowModelPtr flow_model;
  GEThreadLocalContext context;
  RunAsyncCallback callback;
};

class Executor {
 public:
  Executor() = default;
  virtual ~Executor() = default;
  Executor &operator=(const Executor &)& = delete;
  Executor(const Executor &) = delete;

  /**
   * @ingroup ge
   * @brief Load mode from graph.
   * @param [in] FlowModel: root model of graph compiled.
   * @param [in] GraphNode: node of graph.
   * @return Status result of function
   */
  virtual Status LoadGraph(const FlowModelPtr &flow_model, const GraphNodePtr &graph_node,
                           const rtStream_t stream = nullptr) = 0;

  /**
   * @ingroup ge
   * @brief Unload mode.
   * @param [in] FlowModel: root model of graph compiled.
   * @param [in] graph_id: graph identifier.
   * @return Status result of function
   */
  virtual Status UnloadGraph(const FlowModelPtr &flow_model, const uint32_t graph_id) = 0;

  /**
   * @ingroup ge
   * @brief Push model execution params to queue.
   * @param [in] RunArgs of for model execution.
   * @return Status result of function
   */
  virtual Status PushRunArgs(const RunArgs &args) = 0;

  /**
   * @ingroup ge
   * @brief Run graph for synchronize model.
   * @param [in] graph_node: node of graph.
   * @param [in] graph_id: graph identifier.
   * @param [in] inputs: input data for the graph running.
   * @param [out] outputs: output data of the graph running
   * @return Status result of function
   */
  virtual Status RunGraph(const GraphNodePtr &graph_node, const GraphId graph_id,
                          const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) = 0;

  /**
   * @ingroup ge
   * @brief Run graph for NN synchronize model.
   * @param [in] graph_node: node of graph.
   * @param [in] graph_id: graph identifier.
   * @param [in] stream: Stream for model running.
   * @param [in] inputs: input data for the graph running.
   * @param [out] outputs: output data of the graph running
   * @return Status result of function
   */
  virtual Status RunGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id, const rtStream_t stream,
                                    const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) = 0;

  /**
   * @ingroup ge
   * @brief Update graph feature memory base after load model
   * @param [in] graph_node: node of graph.
   * @param [in] mem_base: graph feature memory base(without input and output).
   * @param [in] size: memory size.
   * @return Status result of function
   */
  virtual Status UpdateFeatureMemoryBase(const GraphNodePtr &graph_node, const uintptr_t mem_base, const size_t size) {
    (void)graph_node;
    (void)mem_base;
    (void)size;
    return SUCCESS;
  }
};
}  // namespace ge
#endif // GE_COMMON_EXECUTOR_H
