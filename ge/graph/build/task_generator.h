/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
class GELib;
class OpsKernelManager;

struct ProfilingPoint {
  uint32_t fp_index = 0;
  uint32_t bp_index = 0;
  std::set<uint32_t> end_index;
};
// Describes infos needed by generate task for fusion node
struct FusionTaskInfo {
  RunContext &run_context;
  ComputeGraphPtr &graph;
  NodePtr &node;
  OpDescPtr &fusion_op_desc;
  uint32_t &node_index;
  std::shared_ptr<GELib> &ge_lib;
  const OpsKernelManager &ops_kernel_manager;
  std::vector<domi::TaskDef> &task_def_list;
  std::map<uint32_t, string> &op_name_map;
  ProfilingPoint &profiling_point;
  vector<uint32_t> all_reduce_nodes;
  uint64_t all_reduce_node_idx;
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

  Status FindProfilingNodeIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                                std::vector<uint32_t> &all_reduce_nodes);
 private:
  Status UpdateAnchorStatus(const NodePtr &node);

  Status UpdateOpIsVarAttr(const OpDescPtr &op_desc, uint64_t session_id);

  ///
  /// call engine to generate known shape task.
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

  Status MarkNodeAndSetIndex(ComputeGraphPtr &graph);

  // Mark first and last op according to the same stream and engine
  Status MarkFirstAndLastOps(const vector<OpDescPtr> &ops, bool is_single_stream) const;

  // profiling interface
  Status AutoFindFpOpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point) const;
  Status AutoFindBpOpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                           vector<uint32_t> &all_reduce_nodes) const;
  uint32_t FindLastBpFromBpNode(const ComputeGraphPtr &graph, const NodePtr &bp_node) const;

  Status FindFpOfEnv(const ComputeGraphPtr &graph, const std::string &fp_point_str,
                     ProfilingPoint &profiling_point) const;
  Status FindBpOfEnv(const ComputeGraphPtr &graph, const std::string &bp_point_str, ProfilingPoint &profiling_point,
                     vector<uint32_t> &all_reduce_nodes) const;

  Status GetFpBpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point, vector<uint32_t> &all_reduce_nodes,
                      std::string& fp_point_str, std::string& bp_point_str) const;

  Status FindProfilingTaskIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                                std::vector<uint32_t> &all_reduce_nodes) const;
  Status InsertProfilingTaskBefore(const OpDescPtr &op_desc, const ProfilingPoint &profiling_point,
                                   std::vector<uint32_t> &all_reduce_nodes, uint32_t node_index,
                                   std::vector<domi::TaskDef> &task_def_list, uint64_t &all_reduce_node_idx);
  Status InsertProfilingTaskAfter(const OpDescPtr &op_desc, const ProfilingPoint &profiling_point,
                                  std::vector<uint32_t> &all_reduce_nodes, uint32_t node_index,
                                  std::vector<domi::TaskDef> &task_def_list, uint64_t all_reduce_node_idx);

  static bool IsProfPoint(const OpDescPtr &op, const std::string &name);

  /// call engine to generate task for fusion node.
  /// @param FusionTaskInfo
  /// @param fusion_nodes: nodes in graph with groud_id attr which means fusion node
  /// @param fusion_nodes_seen: fusion node has been called generate task
  /// @return SUCCESS:seccess
  ///         Other: failed
  ///
  Status GenerateTaskForFusionNode(FusionTaskInfo &fusion_task_info,
                                   std::map<int64_t, std::vector<NodePtr>> &fusion_nodes,
                                   std::unordered_set<Node *> &fusion_nodes_seen);

  Status SaveFusionNodes(map<int64_t, std::vector<NodePtr>> &fusion_nodes, ComputeGraphPtr &graph);

  Status SetUnknownShapeStream(RunContext &run_context, rtStream_t &stream);

  Status DestroyUnknownShapeStream(RunContext &run_context, rtStream_t &stream);

  Status SetKnownShapeStream(RunContext &run_context, int64_t stream_id);

  uint8_t *var_mem_base_ = nullptr;
  uint64_t var_mem_size_ = 0;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_TASK_GENERATOR_H_
