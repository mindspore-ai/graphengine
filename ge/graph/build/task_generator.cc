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

#include "graph/build/task_generator.h"
#include <string>
#include <utility>
#include "common/profiling/profiling_manager.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/model_serialize.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/common/ge_call_wrapper.h"
#include "init/gelib.h"
#include "graph/ge_local_context.h"
#include "external/ge/ge_api_types.h"
#include "opskernel_manager/ops_kernel_builder_manager.h"

using domi::LogTimeStampDef;
using domi::ModelTaskDef;
using domi::TaskDef;
using std::map;
using std::set;
using std::string;
using std::vector;

namespace {
const char *const kIsFirstNode = "is_first_node";
const char *const kIsLastNode = "is_last_node";
const char *const kIsInputVar = "INPUT_IS_VAR";
const char *const kIsOutputVar = "OUTPUT_IS_VAR";
const char *const kProfilingMode = "PROFILING_MODE";
const char *const kIteratorV2 = "IteratorV2";
const uint32_t kProfilingArStep = 2;
const uint64_t kProfilingFpStartLogid = 1;
const uint64_t kProfilingBpEndLogid = 2;
const uint64_t kProfilingArStartLogid = 3;
const uint64_t kProfilingArEndLogid = 4;
const uint64_t kProfilingIterEndLogid = 65535;
const int64_t kHashFactor = 100000;
const int64_t kInvalidGroupId = -1;
const std::set<std::string> kFpNodeTypes = {ge::DATA, ge::GETNEXT, kIteratorV2};
}  // namespace
namespace ge {
TaskGenerator::TaskGenerator(uint8_t *var_mem_base, uint64_t var_mem_size) {
  var_mem_base_ = var_mem_base;
  var_mem_size_ = var_mem_size;
}
TaskGenerator::~TaskGenerator() {}

Status TaskGenerator::GetTaskInfo(Model &model, ComputeGraphPtr &graph, uint64_t session_id, RunContext &run_context) {
  GELOGD("Begin to Get TaskInfo. session_id=%lu", session_id);
  // Check params
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param graph is null, session_id:%lu", session_id);
    GELOGE(PARAM_INVALID, "[Check][Param] GetTaskInfo param graph is null. session_id=%lu", session_id);
    return PARAM_INVALID;
  }

  std::vector<TaskDef> task_def_list;
  std::map<uint32_t, string> op_name_map;
  GE_DUMP(graph, "GenerateTaskBefore");
  Status ret = GenerateTask(run_context, graph, task_def_list, op_name_map);
  GE_DUMP(graph, "GenerateTaskAfter");

  if (ret != SUCCESS) {
    GELOGE(ret, "[Generate][Task] failed. session_id=%lu", session_id);
    return ret;
  }

  // op_name_map used when graph load
  graph->SetGraphOpName(op_name_map);

  // Set op_name for infer profiling
  vector<string> op_name;
  for (auto &iter : op_name_map) {
    op_name.push_back(iter.second);
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(model, ATTR_MODEL_TASK_INDEX_OP_NAME, op_name),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for model:%s",
                                      ATTR_MODEL_TASK_INDEX_OP_NAME.c_str(), model.GetName().c_str());
                   GELOGE(FAILED, "[Set][Attr] %s fail for model:%s",
                          ATTR_MODEL_TASK_INDEX_OP_NAME.c_str(), model.GetName().c_str());
                   return FAILED);

  GELOGI("GenerateTask Success, task list:%zu, op map:%zu, logic mem base:%p, logic weight base:%p, logic var base:%p",
         task_def_list.size(), op_name_map.size(), run_context.dataMemBase, run_context.weightMemBase, var_mem_base_);

  // Init and serialize model_task_def
  ModelTaskDef model_task_def;
  model_task_def.set_memory_size(run_context.dataMemSize);
  model_task_def.set_weight_size(run_context.weightMemSize);
  for (const TaskDef &task_def_temp : task_def_list) {
    TaskDef *task_def = model_task_def.add_task();
    if (task_def == nullptr) {
      REPORT_INNER_ERROR("E19999", "Add task_def in ModelTaskDef fail, session_id:%lu, graph:%s, model:%s",
                         session_id, graph->GetName().c_str(), model.GetName().c_str());
      GELOGE(FAILED, "[Check][Param] task_def is nullptr, session_id:%lu, graph:%s, model:%s",
             session_id, graph->GetName().c_str(), model.GetName().c_str());
      return FAILED;
    }
    *task_def = task_def_temp;
  }

  ret = AddModelTaskToModel(model_task_def, session_id, model, run_context);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Add][ModelTask] To Model failed. session_id=%lu", session_id);
    return ret;
  }

  GELOGD("Get TaskInfo success. session_id=%lu", session_id);
  return SUCCESS;
}

Status TaskGenerator::AddModelTaskToModel(const ModelTaskDef &model_task_def, uint64_t session_id, ge::Model &model,
                                          RunContext &run_context) {
  GE_CHK_BOOL_EXEC(
      AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_BASE_ADDR, reinterpret_cast<uintptr_t>(run_context.dataMemBase)),
      REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for model:%s",
                         MODEL_ATTR_TASK_GEN_BASE_ADDR.c_str(), model.GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] %s fail for model:%s",
             MODEL_ATTR_TASK_GEN_BASE_ADDR.c_str(), model.GetName().c_str());
      return FAILED);
  GE_CHK_BOOL_EXEC(
      AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, reinterpret_cast<uintptr_t>(run_context.weightMemBase)),
      REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for model:%s",
                         MODEL_ATTR_TASK_GEN_WEIGHT_ADDR.c_str(), model.GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] %s fail for model:%s",
             MODEL_ATTR_TASK_GEN_WEIGHT_ADDR.c_str(), model.GetName().c_str());
      return FAILED);
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(model, ATTR_MODEL_TASK_GEN_VAR_ADDR, reinterpret_cast<uintptr_t>(var_mem_base_)),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for model:%s",
                                      ATTR_MODEL_TASK_GEN_VAR_ADDR.c_str(), model.GetName().c_str());
                   GELOGE(FAILED, "[Set][Attr] %s fail for model:%s",
                          ATTR_MODEL_TASK_GEN_VAR_ADDR.c_str(), model.GetName().c_str());
                   return FAILED);
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(model, ATTR_MODEL_VAR_SIZE, var_mem_size_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for model:%s",
                                      ATTR_MODEL_VAR_SIZE.c_str(), model.GetName().c_str());
                   GELOGE(FAILED, "[Set][Attr] %s fail for model:%s",
                          ATTR_MODEL_VAR_SIZE.c_str(), model.GetName().c_str());
                   return FAILED);
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(model, MODEL_ATTR_SESSION_ID, session_id),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for mode:%s",
                                      MODEL_ATTR_SESSION_ID.c_str(), model.GetName().c_str());
                   GELOGE(FAILED, "[Set][Attr] %s fail for mode:%s",
                          MODEL_ATTR_SESSION_ID.c_str(), model.GetName().c_str());
                   return FAILED);

  size_t task_size = model_task_def.ByteSizeLong();
  ge::Buffer serial_buff(task_size);
  if (!model_task_def.SerializePartialToArray(serial_buff.GetData(), static_cast<int>(task_size))) {
    REPORT_INNER_ERROR("E19999", "model_task_def's serialize failed,  model name = %s, task_size=%zu",
                       model.GetName().c_str(), task_size);
    GELOGE(FAILED, "[Call][SerializePartialToArray] failed,  model name = %s, task_size=%zu.",
           model.GetName().c_str(), task_size);
    return FAILED;
  }
  if (!AttrUtils::SetZeroCopyBytes(model, MODEL_ATTR_TASKS, std::move(serial_buff))) {
    REPORT_INNER_ERROR("E19999", "Set model task to model failed,  model name = %s, task_size=%zu",
                       model.GetName().c_str(), task_size);
    GELOGE(FAILED, "[Call][SetZeroCopyBytes] Set model task to model failed,  model name = %s, task_size=%zu.",
           model.GetName().c_str(), task_size);
    return FAILED;
  }

  return SUCCESS;
}

Status TaskGenerator::UpdateOpIsVarAttr(const OpDescPtr &op_desc, uint64_t session_id) {
  GELOGD("Update is var attr, node[name:%s(%s), id:%ld, stream_id:%ld].", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), op_desc->GetId(), op_desc->GetStreamId());
  // input
  vector<int64_t> input_offsets = op_desc->GetInputOffset();
  if (!(input_offsets.empty())) {
    vector<bool> input_var;
    size_t valid_input_index = 0;
    for (uint32_t i = 0; i < op_desc->GetAllInputsSize(); i++) {
      vector<int64_t> output_list;
      auto input_tensor_desc = op_desc->MutableInputDesc(i);
      if (input_tensor_desc == nullptr) {
        continue;
      }
      if (valid_input_index >= input_offsets.size()) {
        break;
      }
      int64_t inner_offset = 0;
      (void)ge::AttrUtils::GetInt(input_tensor_desc, ATTR_NAME_INNER_OFFSET, inner_offset);
      GELOGD("Node[%s] input[%u] has inner_offset[%ld]", op_desc->GetName().c_str(), i, inner_offset);
      input_var.push_back(VarManager::Instance(session_id)->IsVarAddr(input_offsets[valid_input_index] - inner_offset));
      valid_input_index++;
    }
    GE_CHK_BOOL_EXEC(AttrUtils::SetListBool(op_desc, kIsInputVar, input_var),
                     REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", kIsInputVar,
                                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
                     GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", kIsInputVar, op_desc->GetName().c_str(),
                            op_desc->GetType().c_str());
                     return FAILED);
  }
  // output
  vector<int64_t> output_offsets = op_desc->GetOutputOffset();
  if (!(output_offsets.empty())) {
    vector<bool> output_var;
    size_t valid_output_index = 0;
    for (uint32_t i = 0; i < op_desc->GetAllOutputsDescSize(); i++) {
      vector<int64_t> output_list;
      auto output_tensor_desc = op_desc->MutableOutputDesc(i);
      if (output_tensor_desc == nullptr) {
        continue;
      }
      if (valid_output_index >= output_offsets.size()) {
        break;
      }
      int64_t inner_offset = 0;
      (void)ge::AttrUtils::GetInt(output_tensor_desc, ATTR_NAME_INNER_OFFSET, inner_offset);
      GELOGD("Node[%s] output[%u] has inner_offset[%ld]", op_desc->GetName().c_str(), i, inner_offset);
      output_var.push_back(
        VarManager::Instance(session_id)->IsVarAddr(output_offsets[valid_output_index] - inner_offset));
      valid_output_index++;
    }
    GE_CHK_BOOL_EXEC(AttrUtils::SetListBool(op_desc, kIsOutputVar, output_var),
                     REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", kIsOutputVar,
                                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
                     GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", kIsOutputVar, op_desc->GetName().c_str(),
                            op_desc->GetType().c_str());
                     return FAILED);
  }
  return SUCCESS;
}

Status TaskGenerator::SaveFusionNodes(map<int64_t, std::vector<NodePtr>> &fusion_nodes, ComputeGraphPtr &graph) {
  std::map<NodePtr, int64_t> nodes_with_group_attr;
  for (auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    int64_t group_id = kInvalidGroupId;
    string name = node->GetName();
    string type = node->GetType();
    // For fusion ddb pass, task def must be continuous.
    // Part1: store
    // If op_desc have this tag, store it in the map firstly,
    // call the elements in the map GenerateTask at last
    // l1 and l2 is for now
    if (ge::AttrUtils::GetInt(op_desc, ATTR_NAME_L1_FUSION_GROUP_ID, group_id) ||
        ge::AttrUtils::GetInt(op_desc, ATTR_NAME_L2_FUSION_GROUP_ID, group_id)) {
      auto stream_id = op_desc->GetStreamId();
      auto group_key = group_id + stream_id * kHashFactor;
      (void)ge::AttrUtils::SetInt(op_desc, ATTR_NAME_FUSION_GROUP_KEY, group_key);
      GELOGD("Fusion: store node[name:%s(%s), group id:%ld, group key:%ld, stream_id:%ld] task.", name.c_str(),
             type.c_str(), group_id, group_key, op_desc->GetStreamId());
      fusion_nodes[group_key].push_back(node);
      nodes_with_group_attr.insert({node, group_id});
    }

    // if node's all in nodes both with same group attr
    // and it have no attr or group attr different
    // which means bad case, return error
    bool call_check = true;
    std::set<int64_t> input_group_ids;
    for (const auto &input_node : node->GetInNodes()) {
      auto iter = nodes_with_group_attr.find(input_node);
      if (iter == nodes_with_group_attr.end()) {
        call_check = false;
        break;
      } else {
        input_group_ids.insert(iter->second);
      }
    }
    call_check = (call_check && (input_group_ids.size() == 1));
    if (call_check) {
      auto input_group_id = *input_group_ids.begin();
      if (group_id != input_group_id) {
        GELOGW("Fusion: node[name:%s(%s) with group id:%ld and diff from it's input nodes's group id:%ld ",
               name.c_str(), type.c_str(), group_id, input_group_id);
      }
    }
  }
  GELOGD("Fusion: get fusion group numbers [%zu].", fusion_nodes.size());
  return SUCCESS;
}

bool TaskGenerator::IsSubGraphOfDynamicGraph(const ComputeGraphPtr &graph) const {
  auto parent_graph_ptr = graph->GetParentGraph();
  if (parent_graph_ptr == nullptr) {
    return false;
  }
  auto root_graph_ptr = GraphUtils::FindRootGraph(parent_graph_ptr);
  if (root_graph_ptr == nullptr) {
    return false;
  }

  return root_graph_ptr->GetGraphUnknownFlag();
}

Status TaskGenerator::GenerateTask(RunContext &run_context, ComputeGraphPtr &graph,
                                   vector<domi::TaskDef> &task_def_list, map<uint32_t, string> &op_name_map) {
  GELOGD("Beign to generate task, graph name is %s.", graph->GetName().c_str());
  std::shared_ptr<GELib> ge_lib = GELib::GetInstance();
  if ((ge_lib == nullptr) || !ge_lib->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Check GELib instance not init before");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GenerateTask failed, because GELib instance not init before.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  GE_CHK_STATUS_RET(MarkNodeAndSetIndex(graph),
                    "[Call][MarkNodeAndSetIndex] failed, graph:%s.", graph->GetName().c_str());
  ProfilingPoint profiling_point;
  vector<uint32_t> all_reduce_nodes;
  GE_CHK_STATUS_RET(FindProfilingTaskIndex(graph, profiling_point, all_reduce_nodes));

  const OpsKernelManager &ops_kernel_manager = ge_lib->OpsKernelManagerObj();

  GE_TIMESTAMP_CALLNUM_START(GenerateTask);
  // map store fusion nodes
  map<int64_t, std::vector<NodePtr>> fusion_nodes;
  string buffer_optimize = "off_optimize";
  (void)ge::GetContext().GetOption(BUFFER_OPTIMIZE, buffer_optimize);
  if (buffer_optimize != "off_optimize") {
    GE_CHK_STATUS_RET(SaveFusionNodes(fusion_nodes, graph));
  }
  std::unordered_set<Node *> fusion_nodes_seen;
  int64_t group_key;
  uint32_t node_index = 0;
  rtStream_t stream = nullptr;
  bool is_unknown_shape = graph->GetGraphUnknownFlag() || GetContext().GetHostExecFlag();
  if (is_unknown_shape) {
    GE_CHK_STATUS_RET(SetUnknownShapeStream(run_context, stream),
                      "[Set][UnknownShapeStream] failed, graph:%s.", graph->GetName().c_str());
  }
  std::function<void()> callback = [&]() {
    if (is_unknown_shape) {
      if (DestroyUnknownShapeStream(run_context, stream) != SUCCESS) {
        GELOGE(FAILED, "[Destroy][UnknownShapeStream] failed.");
      }
    }
  };
  GE_MAKE_GUARD(release, callback);

  auto ffts_filter = [](const Node &node, const char *, const ComputeGraphPtr &) {
    return !node.GetOpDesc()->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH);
  };
  for (auto &node : graph->GetNodes(graph->GetGraphUnknownFlag(), nullptr, ffts_filter)) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    node_index++;
    string name = node->GetName();
    string type = node->GetType();
    bool attr_notask = false;
    bool get_attr_notask_flag = ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOTASK, attr_notask);
    GE_IF_BOOL_EXEC(get_attr_notask_flag && attr_notask,
                    GELOGI("Node[name:%s, type:%s] does not need to generate task.", name.c_str(), type.c_str());
                    continue);

    GE_CHK_STATUS_RET(UpdateOpIsVarAttr(op_desc, graph->GetSessionID()));
    string op_kernel_lib_name = op_desc->GetOpKernelLibName();
    // For fusion ddb pass, task def must be continuous.
    // Part2: Call
    auto fusion_task_info =
        FusionTaskInfo{run_context,        graph,         node,        op_desc,         node_index,      ge_lib,
                       ops_kernel_manager, task_def_list, op_name_map, profiling_point, all_reduce_nodes};
    GE_CHK_STATUS_RET(GenerateTaskForFusionNode(fusion_task_info, fusion_nodes, fusion_nodes_seen),
                      "[Call][GenerateTaskForFusionNode] node:%s(%s) failed", name.c_str(), type.c_str());
    // continue directly
    if (ge::AttrUtils::GetInt(op_desc, ATTR_NAME_FUSION_GROUP_KEY, group_key)) {
      GELOGI("Fusion node[name:%s, type:%s] do not need generate task again.", name.c_str(), type.c_str());
      continue;
    }
    GE_CHK_BOOL_EXEC_INFO(!op_kernel_lib_name.empty(), continue,
                          "Node[name:%s, type:%s] does not need to generate task.", name.c_str(), type.c_str());
    auto kernel_info_store = ops_kernel_manager.GetOpsKernelInfoStore(op_kernel_lib_name);
    if (kernel_info_store == nullptr) {
      REPORT_INNER_ERROR("E19999", "Get ops kernel info store failed for op:%s(%s), op_kernel_name:%s",
                         node->GetName().c_str(), node->GetType().c_str(), op_kernel_lib_name.c_str());
      GELOGE(INTERNAL_ERROR, "[Call][GetOpsKernelInfoStore] No ops kernel store or ops kernel builder found. "
             "node:%s(%s), op_kernel_lib_name=%s.", name.c_str(), type.c_str(), op_kernel_lib_name.c_str());
      return INTERNAL_ERROR;
    }
    GE_CHK_STATUS_RET(UpdateAnchorStatus(node), "[Call][UpdateAnchorStatus] node:%s(%s) failed", name.c_str(),
                      type.c_str());
    if (node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH)) {
      GE_CHK_STATUS_RET(UpdateAnchorStatusForFfts(node), "[Call][UpdateAnchorStatusForFfts] node:%s(%s) failed",
                        name.c_str(), type.c_str());
    }
    // Profiling task
    size_t task_list_size_before = task_def_list.size();
    GE_CHK_STATUS_RET(InsertProfilingTaskBefore(op_desc, profiling_point, all_reduce_nodes, node_index, task_def_list));
    int64_t op_id = op_desc->GetId();
    // Compatible with dynamic shape scenes, the default is 0
    int64_t stream_id = 0;
    if (!is_unknown_shape) {
      stream_id = op_desc->GetStreamId();
      GE_CHK_STATUS_RET(SetKnownShapeStream(run_context, stream_id),
                        "[Set][KnownShapeStream] node[name:%s(%s), id:%ld] stream id is invalid.",
                        name.c_str(), type.c_str(), op_id);
    }
    GELOGD("Call %s to generate node[name:%s(%s), id:%ld, stream_id:%ld] task.", op_kernel_lib_name.c_str(),
           name.c_str(), type.c_str(), op_id, stream_id);
    GE_TIMESTAMP_RESTART(GenerateTask);
    auto ret = OpsKernelBuilderManager::Instance().GenerateTask(*node, run_context, task_def_list);
    GE_TIMESTAMP_ADD(GenerateTask);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Call OpsKernelBuilderManager GenerateTask fail for op:%s(%s)",
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(ret, "[Generate][Task] fail for op:%s(%s)", node->GetName().c_str(), node->GetType().c_str());
      return ret;
    }
    // Profiling task
    GE_CHK_STATUS_RET(InsertProfilingTaskAfter(op_desc, profiling_point, all_reduce_nodes, node_index, task_def_list));
    size_t task_list_size_after = task_def_list.size();
    // If tasks is reduced
    if (task_list_size_after < task_list_size_before) {
      REPORT_INNER_ERROR("E19999", "Call %s to generate node[name:%s(%s), id:%ld, stream_id:%ld] task "
                         "but task num from %zu to %zu, check invalid", op_kernel_lib_name.c_str(), name.c_str(),
                         type.c_str(), op_id, stream_id, task_list_size_before, task_list_size_after);
      GELOGE(FAILED, "[Check][Param] Call %s to generate node[name:%s(%s), id:%ld, stream_id:%ld] task. "
             "but task num from %zu to %zu.", op_kernel_lib_name.c_str(), name.c_str(), type.c_str(),
             op_id, stream_id, task_list_size_before, task_list_size_after);
      return FAILED;
    }

    // Reset stream id to ge stream id, as graph load must use ge stream to reassign stream
    void *ops_kernel_info_store_ptr = kernel_info_store.get();
    for (size_t idx = task_list_size_before; idx < task_list_size_after; ++idx) {
      task_def_list[idx].set_stream_id(static_cast<uint32_t>(stream_id));
      op_name_map[idx] = name;
      // Set opsKernelInfoStorePtr and op_index, the two fields be use in DistributeTask and InitTaskInfo
      TaskDef *task_def_ptr = &task_def_list[idx];
      GE_CHECK_NOTNULL(task_def_ptr);
      task_def_ptr->set_ops_kernel_store_ptr(reinterpret_cast<uintptr_t>(ops_kernel_info_store_ptr));
    }
    GELOGD("Call %s to generate node[name:%s(%s), id:%ld, stream_id:%ld] task finished, generate %zu task(s).",
           op_kernel_lib_name.c_str(), name.c_str(), type.c_str(), op_id, stream_id,
           task_list_size_after - task_list_size_before);
  }
  GE_TIMESTAMP_CALLNUM_EVENT_END(GenerateTask, "GraphBuild::GenerateTask");
  return SUCCESS;
}

Status TaskGenerator::GenerateTaskForFusionNode(FusionTaskInfo &fusion_task_info,
                                                std::map<int64_t, std::vector<NodePtr>> &fusion_nodes,
                                                std::unordered_set<Node *> &fusion_nodes_seen) {
  Status ret = SUCCESS;
  int64_t group_key;
  auto &run_context = fusion_task_info.run_context;
  auto &graph = fusion_task_info.graph;
  auto &node = fusion_task_info.node;
  auto &fusion_op_desc = fusion_task_info.fusion_op_desc;
  auto &node_index = fusion_task_info.node_index;
  const auto &ops_kernel_manager = fusion_task_info.ops_kernel_manager;
  auto &task_def_list = fusion_task_info.task_def_list;
  auto &op_name_map = fusion_task_info.op_name_map;
  auto &profiling_point = fusion_task_info.profiling_point;
  auto &all_reduce_nodes = fusion_task_info.all_reduce_nodes;
  // If op_desc have this attr, call nodes with same group key in a stream together
  if (ge::AttrUtils::GetInt(fusion_op_desc, ATTR_NAME_FUSION_GROUP_KEY, group_key) &&
      (fusion_nodes_seen.count(node.get()) == 0)) {
    GELOGI("Fusion: start fusion group index[%ld], nodes size[%zu].", group_key, fusion_nodes[group_key].size());
    for (auto &fusion_node : fusion_nodes[group_key]) {
      OpDescPtr op_desc = fusion_node->GetOpDesc();

      UpdateOpIsVarAttr(op_desc, graph->GetSessionID());
      std::string fusion_node_name = fusion_node->GetName();
      std::string fusion_node_type = fusion_node->GetType();
      std::string op_kernel_lib_name = op_desc->GetOpKernelLibName();
      if (op_kernel_lib_name.empty()) {
        GELOGI("Fusion: fusion_node[name:%s(%s)] task no need to generate task.", fusion_node_name.c_str(),
               fusion_node_type.c_str());
        continue;
      }
      bool attr_notask = false;
      GE_IF_BOOL_EXEC(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOTASK, attr_notask) && attr_notask,
                      GELOGI("Fusion: fusion_node[name:%s, type:%s] does not need to generate task.",
                             fusion_node_name.c_str(), fusion_node_type.c_str());
                      continue);

      size_t task_list_size_before = task_def_list.size();
      OpsKernelInfoStorePtr kernel_info_store = ops_kernel_manager.GetOpsKernelInfoStore(op_kernel_lib_name);
      if (kernel_info_store == nullptr) {
        REPORT_INNER_ERROR("E19999", "Get ops kernel info store failed for op:%s(%s), op_kernel_name:%s",
                           op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                           op_kernel_lib_name.c_str());
        GELOGE(INTERNAL_ERROR, "[Call][GetOpsKernelInfoStore] Fusion: No ops kernel store or ops kernel builder found."
               " fusion_node:%s(%s), op_kernel_lib_name=%s.",
               fusion_node_name.c_str(), fusion_node_type.c_str(), op_kernel_lib_name.c_str());
        return INTERNAL_ERROR;
      }

      ret = UpdateAnchorStatus(fusion_node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Update][AnchorStatus] fusion_node:%s(%s) failed", fusion_node_name.c_str(),
               fusion_node_type.c_str());
        return ret;
      }

      int64_t op_id = op_desc->GetId();
      int64_t stream_id = op_desc->GetStreamId();
      if (stream_id < 0 || stream_id >= (int64_t)run_context.graphStreamList.size()) {
        REPORT_INNER_ERROR("E19999", "Fusion: fusion_node[name:%s(%s), id:%ld] stream id is invalid, "
                           "stream list size=%zu", fusion_node_name.c_str(), fusion_node_type.c_str(),
                           op_id, run_context.graphStreamList.size());
        GELOGE(INTERNAL_ERROR, "[Check][Param] Fusion: fusion_node[name:%s(%s), id:%ld] stream id is invalid, "
               "stream list size=%zu", fusion_node_name.c_str(), fusion_node_type.c_str(), op_id,
               run_context.graphStreamList.size());
        return INTERNAL_ERROR;
      }
      // profiling task
      (void)InsertProfilingTaskBefore(op_desc, profiling_point, all_reduce_nodes, node_index, task_def_list);
      run_context.stream = run_context.graphStreamList[stream_id];
      GELOGI("Fusion: Call %s to generate fusion_node:[fusion_node_name:%s(%s), id:%ld, stream_id:%ld] task.",
             op_kernel_lib_name.c_str(), fusion_node_name.c_str(), fusion_node_type.c_str(), op_id, stream_id);
      ret = OpsKernelBuilderManager::Instance().GenerateTask(*fusion_node, run_context, task_def_list);
      if (ret != SUCCESS) {
        REPORT_CALL_ERROR("E19999", " Call %s to generate fusion_node:[fusion_node_name:%s(%s), "
                          "id:%ld, stream_id:%ld] task failed", op_kernel_lib_name.c_str(),
                          fusion_node_name.c_str(), fusion_node_type.c_str(), op_id, stream_id);
        GELOGE(ret, "[Generate][Task] Fusion: Call %s to generate fusion_node:[fusion_node_name:%s(%s), "
               "id:%ld, stream_id:%ld] task failed.",
               op_kernel_lib_name.c_str(), fusion_node_name.c_str(), fusion_node_type.c_str(), op_id, stream_id);
        return ret;
      }
      // profiling task
      (void)InsertProfilingTaskAfter(op_desc, profiling_point, all_reduce_nodes, node_index, task_def_list);
      size_t task_list_size_after = task_def_list.size();
      // if tasks is reduced
      if (task_list_size_after < task_list_size_before) {
        REPORT_INNER_ERROR("E19999", "InsertProfilingTask for fusion_node:[fusion_node_name:%s(%s), kernel_name:%s"
                           "id:%ld, stream_id:%ld] task, but task num from %zu to %zu, check invalid",
                           fusion_node_name.c_str(), fusion_node_type.c_str(), op_kernel_lib_name.c_str(),
                           op_id, stream_id, task_list_size_before, task_list_size_after);
        GELOGE(FAILED, "[Check][Param] Fusion: Call %s to generate fusion_node:[fusion_node_name:%s(%s), "
               "id:%ld, stream_id:%ld] task. but task num from %zu to %zu.",
               op_kernel_lib_name.c_str(), fusion_node_name.c_str(), fusion_node_type.c_str(), op_id, stream_id,
               task_list_size_before, task_list_size_after);
        return FAILED;
      }

      // reset stream id to ge stream id, as graph load must use ge stream to reassign stream
      void *ops_kernel_info_store_ptr = kernel_info_store.get();
      for (size_t idx = task_list_size_before; idx < task_list_size_after; ++idx) {
        task_def_list[idx].set_stream_id(static_cast<uint32_t>(stream_id));
        op_name_map[idx] = fusion_node_name;
        // set opsKernelInfoStorePtr and op_index, the two fields be use in DistributeTask and InitTaskInfo
        TaskDef *task_def_ptr = &task_def_list[idx];
        task_def_ptr->set_ops_kernel_store_ptr(reinterpret_cast<uintptr_t>(ops_kernel_info_store_ptr));
      }

      GELOGI("Fusion: Call %s to generate fusion_node:[fusion_node_name:%s(%s), id:%ld, stream_id:%ld]"
             " task finished, generate %zu task(s).",
             op_kernel_lib_name.c_str(), fusion_node_name.c_str(), fusion_node_type.c_str(), op_id, stream_id,
             task_list_size_after - task_list_size_before);

      // record nodes which have call generate task successfully
      fusion_nodes_seen.insert(fusion_node.get());
      node_index++;
    }
  }
  // without tag or has been seen, skip directly
  return ret;
}

Status TaskGenerator::UpdateAnchorStatusForFfts(const NodePtr &node) {
  GELOGD("Start UpdateAnchorStatusForFfts for %s.", node->GetName().c_str());
  if (!node->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
    for (size_t i = 0; i < node->GetOpDesc()->GetSubgraphInstanceNames().size(); ++i) {
      auto sub_graph = NodeUtils::GetSubgraph(*node, i);
      GE_CHECK_NOTNULL(sub_graph);
      GELOGD("Start update anchor status for %s.", sub_graph->GetName().c_str());
      for (auto &ffts_node : sub_graph->GetDirectNode()) {
        GE_CHK_STATUS_RET(UpdateAnchorStatus(ffts_node), "[Call][UpdateAnchorStatus] node:%s(%s) failed",
                          ffts_node->GetName().c_str(), ffts_node->GetType().c_str());
      }
    }
  }
  return SUCCESS;
}

Status TaskGenerator::UpdateAnchorStatus(const NodePtr &node) {
  GELOGD("Start UpdateAnchorStatus for %s.", node->GetName().c_str());
  if (NodeUtils::SetAllAnchorStatus(node) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "SetAllAnchorStatus fail for op:%s(%s)",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][AllAnchorStatus] failed, op:%s(%s)",
           node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  for (auto &anchor : node->GetAllInDataAnchors()) {
    auto peer_anchor = anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_SUSPEND) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Set in peer anchor status fail for op:%s(%s), anchor_index:%d",
                          node->GetName().c_str(), node->GetType().c_str(), anchor->GetIdx());
        GELOGE(INTERNAL_ERROR, "[Set][Status] failed, op:%s(%s), anchor_index:%d",
               node->GetName().c_str(), node->GetType().c_str(), anchor->GetIdx());
        return INTERNAL_ERROR;
      }
      continue;
    }

    std::string const_type;
    bool is_const = NodeUtils::GetConstOpType(peer_anchor->GetOwnerNode(), const_type);
    if (is_const && (const_type == CONSTANT)) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_CONST) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Set in anchor CONST status fail for op:%s(%s), anchor_index:%d",
                          node->GetName().c_str(), node->GetType().c_str(), anchor->GetIdx());
        GELOGE(INTERNAL_ERROR, "[Set][Status] failed. op:%s(%s), anchor_index:%d.",
               node->GetName().c_str(), node->GetType().c_str(), anchor->GetIdx());
        return INTERNAL_ERROR;
      }
    } else {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_DATA) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Set in anchor DATA status fail for op:%s(%s), anchor_index:%d",
                          node->GetName().c_str(), node->GetType().c_str(), anchor->GetIdx());
        GELOGE(INTERNAL_ERROR, "[Set][Status] failed, op:%s(%s), anchor_index:%d.",
               node->GetName().c_str(), node->GetType().c_str(), anchor->GetIdx());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status TaskGenerator::MarkNodeAndSetIndex(ComputeGraphPtr &graph) {
  auto ge_lib = GELib::GetInstance();
  if ((ge_lib == nullptr) || !ge_lib->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Check GELib instance not init before");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GE is not initialized or is finalized.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  const auto all_nodes = graph->GetNodes(graph->GetGraphUnknownFlag());
  if (all_nodes.empty()) {
    REPORT_INNER_ERROR("E19999", "Check param all_nodes empty in graph:%s",
                       graph->GetName().c_str());
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] Graph's node is empty, graph:%s", graph->GetName().c_str());
    return GE_GRAPH_GRAPH_NODE_NULL;
  }

  map<int64_t, vector<OpDescPtr>> all_stream_ops;
  for (auto &node : all_nodes) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    // Reset op kernel lib name
    if (op_desc->GetOpKernelLibName().empty()) {
      (void)ge_lib->DNNEngineManagerObj().GetDNNEngineName(node);
    }

    (void)op_desc->DelAttr(kIsFirstNode);
    (void)op_desc->DelAttr(kIsLastNode);

    all_stream_ops[op_desc->GetStreamId()].emplace_back(op_desc);
  }

  bool is_single_stream = all_stream_ops.size() == 1;
  for (const auto &stream_ops : all_stream_ops) {
    Status status = MarkFirstAndLastOps(stream_ops.second, is_single_stream);
    if (status != SUCCESS) {
      GELOGE(status, "[Mark][FirstAndLastOps] failed, graph:%s.", graph->GetName().c_str());
      return status;
    }
  }

  return SUCCESS;
}

Status TaskGenerator::MarkFirstAndLastOps(const vector<OpDescPtr> &ops, bool is_single_stream) const {
  vector<vector<OpDescPtr>> continuous_op_lists(1);
  const set<string> separator_types(
      {LABELSET, LABELGOTO, LABELGOTOEX, LABELSWITCH, LABELSWITCHBYINDEX, STREAMSWITCH, STREAMSWITCHN});
  for (auto &op_desc : ops) {
    bool attr_notask = false;
    if (ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOTASK, attr_notask) && attr_notask) {
      continue;
    }
    string op_type = op_desc->GetType();
    if ((!is_single_stream && !op_desc->GetSubgraphInstanceNames().empty()) || separator_types.count(op_type) != 0) {
      continuous_op_lists.emplace_back(vector<OpDescPtr>());
    } else {
      continuous_op_lists.back().emplace_back(op_desc);
    }
  }
  GELOGD("Number of continuous node lists is %zu.", continuous_op_lists.size());

  for (const auto &continuous_ops : continuous_op_lists) {
    map<string, std::pair<OpDescPtr, OpDescPtr>> first_and_last_ops;
    for (auto &op_desc : continuous_ops) {
      string op_kernel_lib_name = op_desc->GetOpKernelLibName();
      if (op_kernel_lib_name.empty()) {
        REPORT_INNER_ERROR("E19999", "Get ops kernel info store failed for op:%s(%s), op_kernel_name:%s",
                           op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_kernel_lib_name.c_str());
        GELOGE(INTERNAL_ERROR, "[Check][Param] node:%s(%s) get op kernel lib failed.", op_desc->GetName().c_str(),
               op_desc->GetType().c_str());
        return INTERNAL_ERROR;
      }

      auto it = first_and_last_ops.find(op_kernel_lib_name);
      if (it == first_and_last_ops.end()) {
        first_and_last_ops.emplace(op_kernel_lib_name, std::make_pair(op_desc, op_desc));
      } else {
        it->second.second = op_desc;
      }
    }

    for (auto &it : first_and_last_ops) {
      auto &op_pair = it.second;
      GE_CHK_BOOL_EXEC(ge::AttrUtils::SetBool(op_pair.first, kIsFirstNode, true),
                       REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", kIsFirstNode,
                                          op_pair.first->GetName().c_str(), op_pair.first->GetType().c_str());
                       GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", kIsFirstNode,
                              op_pair.first->GetName().c_str(), op_pair.first->GetType().c_str());
                       return FAILED);
      GE_CHK_BOOL_EXEC(ge::AttrUtils::SetBool(op_pair.second, kIsLastNode, true),
                       REPORT_INNER_ERROR("E19999", "Set Attr:%s fail for op:%s(%s)", kIsLastNode,
                                          op_pair.second->GetName().c_str(), op_pair.second->GetType().c_str());
                       GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", kIsLastNode,
                              op_pair.second->GetName().c_str(), op_pair.second->GetType().c_str());
                       return FAILED);
    }
  }

  return SUCCESS;
}

Status TaskGenerator::AutoFindFpOpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point) const {
  GELOGI("Start AutoFindFpOpIndex");
  OpDescPtr fp_op_desc = nullptr;
  uint32_t current_idx = 0;
  uint32_t first_fp = 0;
  for (auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    string op_kernel_lib_name = op_desc->GetOpKernelLibName();
    if (op_kernel_lib_name.empty()) {
      continue;
    }
    auto type = op_desc->GetType();
    std::string original_type;
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type);
    if (kFpNodeTypes.find(type) != kFpNodeTypes.end() || kFpNodeTypes.find(original_type) != kFpNodeTypes.end()) {
      auto out_anchor = node->GetOutDataAnchor(0);
      for (auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        auto in_node_desc = peer_in_anchor->GetOwnerNode()->GetOpDesc();
        GE_CHECK_NOTNULL(in_node_desc);
        if (fp_op_desc == nullptr || ((in_node_desc->GetId()) < (fp_op_desc->GetId()))) {
          fp_op_desc = in_node_desc;
        }
      }
      break;
    }
  }

  if (fp_op_desc == nullptr) {
    GELOGW("not find fp_op_desc.");
    return SUCCESS;
  }
  GELOGI("Find fp_op_desc is %s, id is %ld", fp_op_desc->GetName().c_str(), fp_op_desc->GetId());
  for (auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    current_idx++;
    if (op_desc->GetName() == fp_op_desc->GetName()) {
      first_fp = current_idx;
      GELOGI("First fp name is %s, idx is %u", op_desc->GetName().c_str(), first_fp);
      break;
    }
  }
  profiling_point.fp_index = first_fp;
  return SUCCESS;
}

Status TaskGenerator::AutoFindBpOpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                                        vector<uint32_t> &all_reduce_nodes) const {
  GELOGI("Start AutoFindBpOpIndex");
  NodePtr bp_node = nullptr;
  uint32_t current_idx = 0;
  for (auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    current_idx++;
    string op_kernel_lib_name = op_desc->GetOpKernelLibName();
    if (op_kernel_lib_name.empty()) {
      continue;
    }

    if (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE) {
      bp_node = node;
      all_reduce_nodes.emplace_back(current_idx);
      GELOGI("Allreduce name %s, idx %u", op_desc->GetName().c_str(), current_idx);
    }
    if (op_desc->GetName() == NODE_NAME_NET_OUTPUT) {
      if (bp_node == nullptr) {
        bp_node = node;
      }
    }
    if (graph->GetNeedIteration()) {
      if (op_desc->GetName() == NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %u, from Node_Output_IteratorCtrl_StreamSwitch_StreamActive",
               op_desc->GetName().c_str(), current_idx);
      }
      if (op_desc->GetName() == NODE_NAME_FLOWCTRL_LOOP_ASSIGN) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %u, from FlowCtrl_LoopCond_ASSIGN",
               op_desc->GetName().c_str(), current_idx);
      }
    } else {
      if (op_desc->GetName() == NODE_NAME_NET_OUTPUT) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %u, from NETOUTPUT", op_desc->GetName().c_str(), current_idx);
      }
    }
  }

  if (bp_node == nullptr) {
    GELOGW("not find bp_node.");
    return SUCCESS;
  }

  return FindLastBpFromBpNode(graph, bp_node, profiling_point.bp_index);
}

Status TaskGenerator::FindLastBpFromBpNode(const ComputeGraphPtr &graph, const NodePtr &target_node,
                                           uint32_t &bp_index) const {
  bp_index = 0;
  auto target_desc = target_node->GetOpDesc();
  GE_CHECK_NOTNULL(target_desc);
  OpDescPtr bp_op_desc = nullptr;
  for (auto &in_node : target_node->GetInAllNodes()) {
    GE_CHECK_NOTNULL(in_node);
    auto in_node_desc = in_node->GetOpDesc();
    GE_CHECK_NOTNULL(in_node_desc);
    if ((bp_op_desc == nullptr || (in_node_desc->GetId() > bp_op_desc->GetId())) &&
        (in_node_desc->GetStreamId() == target_desc->GetStreamId())){
      bp_op_desc = in_node_desc;
    }
  }

  if (bp_op_desc == nullptr) {
    GELOGI("Did not find bp node.");
    return SUCCESS;
  }
  uint32_t current_idx = 0;
  for (auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    current_idx++;
    if (op_desc->GetName() == bp_op_desc->GetName()) {
      bp_index = current_idx;
      GELOGI("Find bp name %s, idx %u", op_desc->GetName().c_str(), bp_index);
      break;
    }
  }
  GELOGI("Last bp node[%s], type[%s], index[%u], stream id[%ld]", bp_op_desc->GetName().c_str(),
         bp_op_desc->GetType().c_str(), bp_index, bp_op_desc->GetStreamId());
  return SUCCESS;
}

Status TaskGenerator::FindFpOfEnv(const ComputeGraphPtr &graph, const std::string &fp_point_str,
                                  ProfilingPoint &profiling_point) const {
  GELOGI("Start FindFpOfEnv");
  uint32_t current_idx = 0;
  uint32_t first_fp = 0;
  for (auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(node->GetOpDesc());
    current_idx++;
    string op_kernel_lib_name = op_desc->GetOpKernelLibName();
    if (op_kernel_lib_name.empty()) {
      continue;
    }

    if (first_fp == 0 && IsProfPoint(op_desc, fp_point_str)) {
      first_fp = current_idx;
      GELOGI("First fp name from env is %s, idx %u", op_desc->GetName().c_str(), first_fp);
    }
  }

  profiling_point.fp_index = first_fp;
  return SUCCESS;
}

Status TaskGenerator::FindBpOfEnv(const ComputeGraphPtr &graph, const std::string &bp_point_str,
                                  ProfilingPoint &profiling_point, vector<uint32_t> &all_reduce_nodes) const {
  GELOGI("Start FindBpOfEnv");
  uint32_t current_idx = 0;
  uint32_t last_bp = 0;
  for (auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(node->GetOpDesc());
    current_idx++;
    string op_kernel_lib_name = op_desc->GetOpKernelLibName();
    if (op_kernel_lib_name.empty()) {
      continue;
    }

    if (graph->GetNeedIteration()) {
      if (op_desc->GetName() == NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %u, from Node_Output_IteratorCtrl_StreamSwitch_StreamActive",
               op_desc->GetName().c_str(), current_idx);
      }
      if (op_desc->GetName() == NODE_NAME_FLOWCTRL_LOOP_ASSIGN) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %u, from FlowCtrl_LoopCond_ASSIGN",
               op_desc->GetName().c_str(), current_idx);
      }
    } else {
      if (op_desc->GetName() == NODE_NAME_NET_OUTPUT) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %u, from NETOUTPUT", op_desc->GetName().c_str(), current_idx);
      }
    }

    if (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE) {
      all_reduce_nodes.emplace_back(current_idx);
      GELOGI("Allreduce name %s, idx %u", op_desc->GetName().c_str(), current_idx);
    }
    if (IsProfPoint(op_desc, bp_point_str)) {
      last_bp = current_idx;
      GELOGI("Last bp name from env is %s, idx %u", op_desc->GetName().c_str(), last_bp);
    }
  }

  profiling_point.bp_index = last_bp;
  return SUCCESS;
}

Status TaskGenerator::GetFpBpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                                   vector<uint32_t> &all_reduce_nodes, std::string &fp_point_str,
                                   std::string &bp_point_str) const {

  ProfilingManager::Instance().GetFpBpPoint(fp_point_str, bp_point_str);

  Status ret = SUCCESS;
  if (fp_point_str.empty()) {
    ret = AutoFindFpOpIndex(graph, profiling_point);
    if (ret != SUCCESS) {
      GELOGW("First forward profiling op_index not set and FindFpOpIndex failed.");
      return FAILED;
    }
  }

  if (bp_point_str.empty()) {
    ret = AutoFindBpOpIndex(graph, profiling_point, all_reduce_nodes);
    if (ret != SUCCESS) {
      GELOGW("Last backward profiling op_index not set and FindBpOpIndex failed.");
      return FAILED;
    }
  }

  return SUCCESS;
}

Status TaskGenerator::FindProfilingNodeIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                                             std::vector<uint32_t> &all_reduce_nodes) {
  return FindProfilingTaskIndex(graph, profiling_point, all_reduce_nodes);
}

Status TaskGenerator::FindProfilingTaskIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                                             vector<uint32_t> &all_reduce_nodes) const {
  GE_CHECK_NOTNULL(graph);
  const char *profiling_mode = std::getenv(kProfilingMode);
  bool is_profiling = (profiling_mode != nullptr) || ProfilingManager::Instance().ProfilingOn() ||
                      ProfilingManager::Instance().ProfilingTrainingTraceOn();
  if (!is_profiling) {
    GELOGD("Profiling is not open.");
    return SUCCESS;
  }

  // subgraph  of dynamic graph no need to find index, has been found in parent graph
  if (IsSubGraphOfDynamicGraph(graph)) {
    GELOGI("Graph[%s] is subgraph of dynamic graph, no need to find index.", graph->GetName().c_str());
    return SUCCESS;
  }

  GELOGI("Start get FP/BP index.");
  std::string fp_point_str;
  std::string bp_point_str;
  Status ret = GetFpBpIndex(graph, profiling_point, all_reduce_nodes, fp_point_str, bp_point_str);
  if (ret != SUCCESS) {
    GELOGW("Get FP_POINT BP_POINT failed.");
    return SUCCESS;
  }

  GELOGI("fp_point_str:%s, bp_point_str:%s.", fp_point_str.c_str(), bp_point_str.c_str());

  if (!fp_point_str.empty()) {
    ret = FindFpOfEnv(graph, fp_point_str, profiling_point);
    if (ret != SUCCESS) {
      GELOGW("First backward profiling op name set but FindFpOfEnv failed.");
      return SUCCESS;
    }
  }

  if (!bp_point_str.empty()) {
    ret = FindBpOfEnv(graph, bp_point_str, profiling_point, all_reduce_nodes);
    if (ret != SUCCESS) {
      GELOGW("Last backward profiling op name set but FindBpOfEnv failed.");
      return SUCCESS;
    }
  }

  bool train_graph = graph->GetNeedIteration();
  if (profiling_point.fp_index == 0 && train_graph) {
    GELOGW("First forward op name can't be found in graph for training trace.");
  }
  if (profiling_point.bp_index == 0 && train_graph) {
    GELOGW("Last backward op name can't be found in graph for training trace.");
  }
  return SUCCESS;
}

Status TaskGenerator::InsertProfilingArTaskBefore(const OpDescPtr &op_desc, std::vector<uint32_t> &all_reduce_nodes,
                                                  uint32_t node_index, std::vector<domi::TaskDef> &task_def_list,
                                                  bool is_insert_bp_profiling_task) {
  bool is_insert_all_reduce_task = false;
  int64_t ar_log_id = 0xFFFF;
  if (is_insert_bp_profiling_task) {
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, ar_log_id);
    is_insert_all_reduce_task = true;
  }
  if (!is_insert_all_reduce_task) {
    for (size_t i = 0; i < all_reduce_nodes.size(); i++) {
      if (all_reduce_nodes[i] == node_index) {
        GE_IF_BOOL_EXEC(TypeUtils::CheckUint64MulOverflow(i, kProfilingArStep),
                        REPORT_INNER_ERROR("E19999", "Multiply result is out of range when calc profiling ar log id "
                                           "for node:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
                        GELOGE(FAILED, "[Check][Param] Multiply result is out of range. node:%s(%s)",
                               op_desc->GetName().c_str(), op_desc->GetType().c_str());
                        return FAILED);
        ar_log_id = i * kProfilingArStep + kProfilingArStartLogid;
        is_insert_all_reduce_task = true;
        break;
      }
    }
  }

  if (is_insert_all_reduce_task) {
    GELOGI("The start allreduce operator is %s, idx %u, log_id %ld", op_desc->GetName().c_str(), node_index, ar_log_id);
    TaskDef ar_task_def;
    ar_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
    ar_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *ar_log_def = ar_task_def.mutable_log_timestamp();
    if (ar_log_def != nullptr) {
      ar_log_def->set_logid(ar_log_id);
      ar_log_def->set_notify(false);
    }
    task_def_list.push_back(ar_task_def);
  }

  return SUCCESS;
}

Status TaskGenerator::InsertProfilingTaskBefore(const OpDescPtr &op_desc, const ProfilingPoint &profiling_point,
                                                vector<uint32_t> &all_reduce_nodes, uint32_t node_index,
                                                vector<domi::TaskDef> &task_def_list) {
  const char *profiling_mode = std::getenv(kProfilingMode);
  bool is_profiling = (profiling_mode != nullptr) || ProfilingManager::Instance().ProfilingOn() ||
                      ProfilingManager::Instance().ProfilingTrainingTraceOn();
  bool is_insert_fp_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_FP_PROFILILNG_TASK, is_insert_fp_profiling_task);
  bool is_insert_bp_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, is_insert_bp_profiling_task);
  bool no_insert_profiling_task = ((profiling_point.fp_index == 0) || (profiling_point.bp_index == 0) ||
                                   (profiling_point.end_index.empty())) &&
                                  (!(is_insert_fp_profiling_task || is_insert_bp_profiling_task));
  if (!is_profiling || no_insert_profiling_task) {
    return SUCCESS;
  }
  GELOGD("Insert fp profiling task: %d, insert bp profiling task: %d, fp index: %u, bp index: %u, end index size: %zu",
         is_insert_fp_profiling_task, is_insert_bp_profiling_task, profiling_point.fp_index, profiling_point.bp_index,
         profiling_point.end_index.size());

  if ((profiling_point.fp_index == node_index) || is_insert_fp_profiling_task) {
    uint64_t jobid_log_id = ge::GetContext().TraceId();
    GELOGI("The first FP operator is %s, idx %u, job_id %lu", op_desc->GetName().c_str(), node_index, jobid_log_id);

    TaskDef job_task_def;
    job_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
    job_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *job_log_def = job_task_def.mutable_log_timestamp();
    if (job_log_def != nullptr) {
      job_log_def->set_logid(jobid_log_id);
      job_log_def->set_notify(false);
    }
    task_def_list.emplace_back(job_task_def);
    TaskDef fp_task_def;
    fp_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
    fp_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *fp_log_def = fp_task_def.mutable_log_timestamp();
    if (fp_log_def != nullptr) {
      fp_log_def->set_logid(kProfilingFpStartLogid);
      fp_log_def->set_notify(false);
    }
    task_def_list.emplace_back(fp_task_def);
  }

  bool is_all_reduce = (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE);
  if (is_all_reduce) {
    (void)InsertProfilingArTaskBefore(op_desc, all_reduce_nodes, node_index,
                                      task_def_list, is_insert_bp_profiling_task);
  }

  return SUCCESS;
}

Status TaskGenerator::InsertProfilingArTaskAfter(const OpDescPtr &op_desc, std::vector<uint32_t> &all_reduce_nodes,
                                                 uint32_t node_index, std::vector<domi::TaskDef> &task_def_list,
                                                 bool is_insert_bp_profiling_task) {
  bool is_insert_all_reduce_task = false;
  int64_t ar_log_id = 0xFFFF;
  if (is_insert_bp_profiling_task) {
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, ar_log_id);
    ar_log_id += 1;
    is_insert_all_reduce_task = true;
  }
  if (!is_insert_all_reduce_task) {
    for (size_t i = 0; i < all_reduce_nodes.size(); i++) {
      if (all_reduce_nodes[i] == node_index) {
        GE_IF_BOOL_EXEC(TypeUtils::CheckUint64MulOverflow(i, kProfilingArStep),
                        REPORT_INNER_ERROR("E19999", "Multiply result is out of range when calc profiling ar log id "
                                           "for node:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
                        GELOGE(FAILED, "[Check][Param] Multiply result is out of range. node:%s(%s)",
                               op_desc->GetName().c_str(), op_desc->GetType().c_str());
                        return FAILED);
        ar_log_id = i * kProfilingArStep + kProfilingArEndLogid;
        is_insert_all_reduce_task = true;
        break;
      }
    }
  }

  if (is_insert_all_reduce_task) {
    GELOGI("The start allreduce operator is %s, idx %u, log_id %ld", op_desc->GetName().c_str(), node_index, ar_log_id);
    TaskDef ar_task_def;
    ar_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
    ar_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *ar_log_def = ar_task_def.mutable_log_timestamp();
    if (ar_log_def != nullptr) {
      ar_log_def->set_logid(ar_log_id);
      ar_log_def->set_notify(false);
    }
    task_def_list.push_back(ar_task_def);
  }

  return SUCCESS;
}

Status TaskGenerator::InsertProfilingTaskAfter(const OpDescPtr &op_desc, const ProfilingPoint &profiling_point,
                                               vector<uint32_t> &all_reduce_nodes, uint32_t node_index,
                                               vector<domi::TaskDef> &task_def_list) {
  GE_CHECK_NOTNULL(op_desc);
  const char *profiling_mode = std::getenv(kProfilingMode);
  bool is_profiling = (profiling_mode != nullptr) || ProfilingManager::Instance().ProfilingOn() ||
                      ProfilingManager::Instance().ProfilingTrainingTraceOn();
  bool is_insert_bp_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, is_insert_bp_profiling_task);
  bool is_insert_end_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_END_PROFILILNG_TASK, is_insert_end_profiling_task);
  bool no_insert_profiling_task = ((profiling_point.fp_index == 0) || (profiling_point.bp_index == 0) ||
                                   (profiling_point.end_index.empty())) &&
                                  (!(is_insert_bp_profiling_task || is_insert_end_profiling_task));
  if (!is_profiling || no_insert_profiling_task) {
    return SUCCESS;
  }
  GELOGD("Insert bp profiling task: %d, insert end profiling task: %d, fp index: %u, bp index: %u, end index size: %zu",
         is_insert_bp_profiling_task, is_insert_end_profiling_task, profiling_point.fp_index, profiling_point.bp_index,
         profiling_point.end_index.size());

  bool is_all_reduce = (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE);
  if ((profiling_point.bp_index == node_index) || (!is_all_reduce && is_insert_bp_profiling_task)) {
    GELOGI("The last BP operator is %s, idx %u", op_desc->GetName().c_str(), node_index);
    TaskDef bp_task_def;
    bp_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
    bp_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *bp_log_def = bp_task_def.mutable_log_timestamp();
    GE_CHECK_NOTNULL(bp_log_def);
    bp_log_def->set_logid(kProfilingBpEndLogid);
    bp_log_def->set_notify(false);
    task_def_list.emplace_back(bp_task_def);
  }

  if (profiling_point.end_index.find(node_index) != profiling_point.end_index.end() ||
      is_insert_end_profiling_task) {
    GELOGI("The iteration end operator is %s, idx %u", op_desc->GetName().c_str(), node_index);
    TaskDef end_task_def;
    end_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
    end_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *end_log_def = end_task_def.mutable_log_timestamp();
    GE_CHECK_NOTNULL(end_log_def);
    end_log_def->set_logid(kProfilingIterEndLogid);
    end_log_def->set_notify(true);
    task_def_list.emplace_back(end_task_def);
  }

  if (is_all_reduce) {
    (void)InsertProfilingArTaskAfter(op_desc, all_reduce_nodes, node_index,
                                     task_def_list, is_insert_bp_profiling_task);
  }

  return SUCCESS;
}

bool TaskGenerator::IsProfPoint(const OpDescPtr &op, const std::string &name) {
  if (op == nullptr) {
    return false;
  }

  if (op->GetName() == name) {
    return true;
  }

  std::vector<std::string> original_op_names;
  bool ret = AttrUtils::GetListStr(op, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);
  if (!ret) {
    return false;
  }

  for (auto &origin_name : original_op_names) {
    if (origin_name == name) {
      return true;
    }
  }

  return false;
}

Status TaskGenerator::SetUnknownShapeStream(RunContext &run_context, rtStream_t &stream) {
  GE_CHK_RT_RET(rtStreamCreate(&stream, 0));
  run_context.stream = stream;
  rtError_t rt_ret = rtModelBindStream(run_context.model, stream, 0);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtModelBindStream failed, ret:0x%X", rt_ret);
    GELOGE(FAILED, "[Call][RtModelBindStream] failed, ret: 0x%X", rt_ret);
    GE_CHK_RT_RET(rtStreamDestroy(stream));
    return FAILED;
  }
  return SUCCESS;
}

Status TaskGenerator::DestroyUnknownShapeStream(RunContext &run_context, rtStream_t &stream) {
  GE_CHK_RT(rtModelUnbindStream(run_context.model, stream));
  GE_CHK_RT_RET(rtStreamDestroy(stream));
  return SUCCESS;
}

Status TaskGenerator::SetKnownShapeStream(RunContext &run_context, int64_t stream_id) {
  if (stream_id < 0 || stream_id >= static_cast<int64_t>(run_context.graphStreamList.size())) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] Stream id[%ld] is invalid, stream list size=%zu", stream_id,
           run_context.graphStreamList.size());
    return INTERNAL_ERROR;
  }
  run_context.stream = run_context.graphStreamList[stream_id];
  return SUCCESS;
}
}  // namespace ge
