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

#ifndef GE_GRAPH_BUILD_TASK_GENERATOR_H_
#define GE_GRAPH_BUILD_TASK_GENERATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/model.h"
#include "proto/task.pb.h"
#include "runtime/rt.h"

namespace ge {
struct ProfilingPoint {
  uint32_t fp_index = 0;
  uint32_t bp_index = 0;
  uint32_t end_index = 0;
};

class TaskGenerator {
 public:
  TaskGenerator() = default;

  TaskGenerator(const TaskGenerator &) = delete;

  TaskGenerator &operator=(const TaskGenerator &) = delete;

  virtual ~TaskGenerator();

  TaskGenerator(uint8_t *var_mem_base, uint64_t var_mem_size);

  ///
  /// get task info.
  /// @param model model
  /// @param graph compute graph
  /// @param buffer weights buffer
  /// @param session_id session id
  /// @return SUCCESS: success
  ///         other:failed
  ///
  Status GetTaskInfo(Model &model, ComputeGraphPtr &graph, uint64_t session_id, RunContext &run_context);

 private:
  Status UpdateAnchorStatus(const NodePtr &node);

  Status UpdateOpIsVarAttr(const OpDescPtr &op_desc, uint64_t session_id);

  ///
  /// call engine to generate task.
  /// @param run_context run context
  /// @param graph compute graph
  /// @param task_def_list task def list generate by engine
  /// @param op_name_map relation of task index and op
  /// @return SUCCESS:seccess
  /// Other: failed
  ///
  Status GenerateTask(RunContext &run_context, ComputeGraphPtr &graph, std::vector<domi::TaskDef> &task_def_list,
                      std::map<uint32_t, string> &op_name_map);

  ///
  /// AddModelTaskToModel
  /// @param model_task_def model task
  /// @param model_def model
  /// @return SUCCESS:seccess
  ///         Other: failed
  ///
  Status AddModelTaskToModel(const domi::ModelTaskDef &model_task_def, uint64_t session_id, Model &model_def,
                             RunContext &run_context);

  // Mark first and last node according to the same stream and engine
  Status MarkFirstAndLastNode(ComputeGraphPtr &graph);

  // profiling interface
  Status FindProfilingTaskIndex(const ComputeGraphPtr &graph, ProfilingPoint &ppoint,
                                std::vector<uint32_t> &ar_ppoint) const;
  Status InsertProfilingTaskBefore(const OpDescPtr &op_desc, const ProfilingPoint &ppoint,
                                   std::vector<uint32_t> &ar_ppoint, uint32_t node_index,
                                   std::vector<domi::TaskDef> &task_def_list);
  Status InsertProfilingTaskAfter(const OpDescPtr &op_desc, const ProfilingPoint &ppoint,
                                  std::vector<uint32_t> &ar_ppoint, uint32_t node_index,
                                  std::vector<domi::TaskDef> &task_def_list);

  uint8_t *var_mem_base_ = nullptr;
  uint64_t var_mem_size_ = 0;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_TASK_GENERATOR_H_
