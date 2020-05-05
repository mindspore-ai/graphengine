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

#include "graph/build/task_generator.h"
#include <string>
#include <utility>
#include "common/util.h"
#include "common/types.h"
#include "framework/common/debug/ge_log.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/model_serialize.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"

using domi::LogTimeStampDef;
using domi::ModelTaskDef;
using domi::TaskDef;
using std::map;
using std::string;
using std::vector;

namespace {
const char *const kIsFirstNode = "is_first_node";
const char *const kIsLastNode = "is_last_node";
const char *const kIsInputVar = "INPUT_IS_VAR";
const char *const kIsOutputVar = "OUTPUT_IS_VAR";
const char *const kProfilingMode = "PROFILING_MODE";
const char *const kProfilingFpPoint = "FP_POINT";
const char *const kProfilingBpPoint = "BP_POINT";
const uint32_t kProfilingArStep = 2;
const uint64_t kProfilingFpStartLogid = 1;
const uint64_t kProfilingBpEndLogid = 2;
const uint64_t kProfilingArStartLogid = 3;
const uint64_t kProfilingArEndLogid = 4;
const uint64_t kProfilingIterEndLogid = 255;
const int64_t kMaxNodeNumInNormalStream = 350;
const int64_t kInvalidGroupId = -1;
}  // namespace
namespace ge {
TaskGenerator::TaskGenerator(uint8_t *var_mem_base, uint64_t var_mem_size) {
  var_mem_base_ = var_mem_base;
  var_mem_size_ = var_mem_size;
}
TaskGenerator::~TaskGenerator() {}

Status TaskGenerator::GetTaskInfo(Model &model, ComputeGraphPtr &graph, uint64_t session_id, RunContext &run_context) {
  GELOGI("Begin to Get TaskInfo. session_id=%lu", session_id);
  // Check params
  if (graph == nullptr) {
    GELOGE(PARAM_INVALID, "GetTaskInfo param graph is null. session_id=%lu", session_id);
    return PARAM_INVALID;
  }

  std::vector<TaskDef> task_def_list;
  std::map<uint32_t, string> op_name_map;

  GraphUtils::DumpGEGraph(graph, "GenerateTaskBefore");
  GraphUtils::DumpGEGraphToOnnx(*graph, "GenerateTaskBefore");
  Status ret = GenerateTask(run_context, graph, task_def_list, op_name_map);
  GraphUtils::DumpGEGraph(graph, "GenerateTaskAfter");
  GraphUtils::DumpGEGraphToOnnx(*graph, "GenerateTaskAfter");
  if (ret != SUCCESS) {
    GELOGE(ret, "GenerateTask failed. session_id=%lu", session_id);
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
                   GELOGE(FAILED, "SetListStr failed.");
                   return FAILED);

  GELOGI("Call GenerateTask Success, task_def_list.size:%zu, op_name_map.size:%zu", task_def_list.size(),
         op_name_map.size());

  // Init and serialize model_task_def
  ModelTaskDef model_task_def;
  model_task_def.set_memory_size(run_context.dataMemSize);
  model_task_def.set_weight_size(run_context.weightMemSize);
  for (const TaskDef &task_def_temp : task_def_list) {
    TaskDef *task_def = model_task_def.add_task();
    if (task_def == nullptr) {
      GELOGE(FAILED, "task_def is nullptr.");
      return FAILED;
    }
    *task_def = task_def_temp;
  }

  ret = AddModelTaskToModel(model_task_def, session_id, model, run_context);
  if (ret != SUCCESS) {
    GELOGE(ret, "AddModelTaskToModel failed. session_id=%lu", session_id);
    return ret;
  }

  GELOGI("Get TaskInfo success. session_id=%lu", session_id);
  return SUCCESS;
}

Status TaskGenerator::AddModelTaskToModel(const ModelTaskDef &model_task_def, uint64_t session_id, ge::Model &model,
                                          RunContext &run_context) {
  GE_CHK_BOOL_EXEC(
    AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_BASE_ADDR, reinterpret_cast<uintptr_t>(run_context.dataMemBase)),
    GELOGE(FAILED, "SetInt MODEL_ATTR_TASK_GEN_BASE_ADDR failed.");
    return FAILED);
  GE_CHK_BOOL_EXEC(
    AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, reinterpret_cast<uintptr_t>(run_context.weightMemBase)),
    GELOGE(FAILED, "SetInt MODEL_ATTR_TASK_GEN_WEIGHT_ADDR failed.");
    return FAILED);
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(model, ATTR_MODEL_TASK_GEN_VAR_ADDR, reinterpret_cast<uintptr_t>(var_mem_base_)),
                   GELOGE(FAILED, "SetInt ATTR_MODEL_TASK_GEN_VAR_ADDR failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(model, ATTR_MODEL_VAR_SIZE, var_mem_size_),
                   GELOGE(FAILED, "SetInt ATTR_MODEL_VAR_SIZE failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(model, MODEL_ATTR_SESSION_ID, session_id),
                   GELOGE(FAILED, "SetInt MODEL_ATTR_SESSION_ID failed.");
                   return FAILED);

  size_t task_size = model_task_def.ByteSizeLong();
  ge::Buffer serial_buff(task_size);
  if (!model_task_def.SerializePartialToArray(serial_buff.GetData(), static_cast<int>(task_size))) {
    GELOGE(FAILED, "model_task_def's serialize failed,  model name = %s, task_size=%zu.", model.GetName().c_str(),
           task_size);
    return FAILED;
  }
  if (!AttrUtils::SetZeroCopyBytes(model, MODEL_ATTR_TASKS, std::move(serial_buff))) {
    GELOGE(FAILED, "Set model task to model failed,  model name = %s, task_size=%zu.", model.GetName().c_str(),
           task_size);
    return FAILED;
  }

  return SUCCESS;
}

Status TaskGenerator::UpdateOpIsVarAttr(const OpDescPtr &op_desc, uint64_t session_id) {
  vector<int64_t> input_offsets = op_desc->GetInputOffset();
  GELOGD("Update is var attr, node[name:%s(%s), id:%ld, stream_id:%ld].", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), op_desc->GetId(), op_desc->GetStreamId());
  if (!(input_offsets.empty())) {
    vector<bool> input_var;
    for (int64_t input : input_offsets) {
      input_var.push_back(VarManager::Instance(session_id)->IsVarAddr(input));
    }
    GE_CHK_BOOL_EXEC(AttrUtils::SetListBool(op_desc, kIsInputVar, input_var), GELOGE(FAILED, "SetListBool failed.");
                     return FAILED);
  }

  vector<int64_t> output_offsets = op_desc->GetOutputOffset();
  if (!(output_offsets.empty())) {
    vector<bool> output_var;
    for (int64_t output : output_offsets) {
      output_var.push_back(VarManager::Instance(session_id)->IsVarAddr(output));
    }
    GE_CHK_BOOL_EXEC(AttrUtils::SetListBool(op_desc, kIsOutputVar, output_var), GELOGE(FAILED, "SetListBool failed.");
                     return FAILED);
  }
  return SUCCESS;
}

Status TaskGenerator::SaveL1fusionNodes(map<int64_t, std::vector<NodePtr>> &l1_fusion_nodes, ComputeGraphPtr &graph) {
  std::map<NodePtr, int64_t> nodes_with_group_attr;
  for (auto &node : graph->GetAllNodes()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    int64_t group_id = kInvalidGroupId;
    string name = node->GetName();
    string type = node->GetType();
    // For l1 fusion ddb pass, task def must be continuous.
    // Part1: store
    // If op_desc have this tag, store it in the map firstly,
    // call the elements in the map GenerateTask at last
    if (ge::AttrUtils::GetInt(op_desc, ATTR_NAME_L1_FUSION_GROUP_ID, group_id)) {
      auto stream_id = op_desc->GetStreamId();
      auto group_key = group_id + stream_id * kMaxNodeNumInNormalStream;
      (void)ge::AttrUtils::SetInt(op_desc, ATTR_NAME_L1_FUSION_GROUP_KEY, group_key);
      GELOGI("L1Fusion: store node[name:%s(%s), group id:%ld, group key:%ld, stream_id:%ld] task.", name.c_str(),
             type.c_str(), group_id, group_key, op_desc->GetStreamId());
      l1_fusion_nodes[group_key].push_back(node);
      nodes_with_group_attr.insert({node, group_id});
    }

    // if node's all in nodes both with same group attr
    // and it have no attr or group attr different
    // which means bad case, return error
    bool call_check = true;
    std::unordered_set<int64_t> input_group_ids;
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
        GELOGE(INTERNAL_ERROR,
               "L1Fusion: node[name:%s(%s) with group id:%ld and diff from it's input nodes's group id:%ld ",
               name.c_str(), type.c_str(), group_id, input_group_id);
        return INTERNAL_ERROR;
      }
    }
  }
  GELOGI("L1Fusion: get fusion group numbers [%zu].", l1_fusion_nodes.size());
  return SUCCESS;
}

Status TaskGenerator::GenerateTask(RunContext &run_context, ComputeGraphPtr &graph,
                                   vector<domi::TaskDef> &task_def_list, map<uint32_t, string> &op_name_map) {
  std::shared_ptr<GELib> ge_lib = GELib::GetInstance();
  if ((ge_lib == nullptr) || !ge_lib->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "GenerateTask failed.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  GE_CHK_STATUS_RET(MarkNodeAndSetIndex(graph), "MarkFirstAndLastNode failed.");
  ProfilingPoint ppoint;
  vector<uint32_t> ar_ppoint;
  GE_CHK_STATUS_RET(FindProfilingTaskIndex(graph, ppoint, ar_ppoint));

  const OpsKernelManager &ops_kernel_manager = ge_lib->OpsKernelManagerObj();

  GE_TIMESTAMP_CALLNUM_START(GenerateTask);
  // map store l1 fusion nodes
  map<int64_t, std::vector<NodePtr>> l1_fusion_nodes;
  string is_l1_fusion_enable = "false";
  graphStatus ret = ge::GetContext().GetOption("ge.l1Fusion", is_l1_fusion_enable);
  if ((ret == GRAPH_SUCCESS) && (is_l1_fusion_enable == "true")) {
    GE_CHK_STATUS_RET(SaveL1fusionNodes(l1_fusion_nodes, graph));
  }
  std::unordered_set<Node *> l1_fusion_nodes_seen;
  int64_t group_id;
  uint32_t node_index = 0;
  for (auto &node : graph->GetAllNodes()) {
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
    // For l1 fusion ddb pass, task def must be continuous.
    // Part2: Call
    auto l1_fusion_task_info =
      L1FusionTaskInfo{run_context,        graph,         node,        op_desc, node_index, ge_lib,
                       ops_kernel_manager, task_def_list, op_name_map, ppoint,  ar_ppoint};
    GE_CHK_STATUS_RET(GenerateTaskForL1FusionNode(l1_fusion_task_info, l1_fusion_nodes, l1_fusion_nodes_seen),
                      "Call GenerateTaskForL1FusionNode node:%s(%s) failed", name.c_str(), type.c_str());
    // continue directly
    if (ge::AttrUtils::GetInt(op_desc, ATTR_NAME_L1_FUSION_GROUP_ID, group_id)) {
      GELOGI("L1Fusion not %s to generate node[name:%s(%s) task again.", op_kernel_lib_name.c_str(), name.c_str(),
             type.c_str());
      continue;
    }
    if (op_kernel_lib_name.empty()) {
      GELOGI("Node[name:%s, type:%s] does not need to generate task.", name.c_str(), type.c_str());
      continue;
    }

    OpsKernelInfoStorePtr kernel_info_store = ops_kernel_manager.GetOpsKernelInfoStore(op_kernel_lib_name);
    if (kernel_info_store == nullptr) {
      GELOGE(INTERNAL_ERROR, "No ops kernel store found. node:%s(%s), op_kernel_lib_name=%s.", name.c_str(),
             type.c_str(), op_kernel_lib_name.c_str());
      return INTERNAL_ERROR;
    }
    GE_CHK_STATUS_RET(UpdateAnchorStatus(node), "Call UpdateAnchorStatus node:%s(%s) failed", name.c_str(),
                      type.c_str());
    int64_t op_id = op_desc->GetId();
    int64_t stream_id = op_desc->GetStreamId();
    if (stream_id < 0 || stream_id >= static_cast<int64_t>(run_context.graphStreamList.size())) {
      GELOGE(INTERNAL_ERROR, "node[name:%s(%s), id:%ld] stream id is invalid, stream list size=%zu", name.c_str(),
             type.c_str(), op_id, run_context.graphStreamList.size());
      return INTERNAL_ERROR;
    }

    // Profiling task
    size_t task_list_size_before = task_def_list.size();
    GE_CHK_STATUS_RET(InsertProfilingTaskBefore(op_desc, ppoint, ar_ppoint, node_index, task_def_list));
    run_context.stream = run_context.graphStreamList[stream_id];
    GELOGD("Call %s to generate node[name:%s(%s), id:%ld, stream_id:%ld] task.", op_kernel_lib_name.c_str(),
           name.c_str(), type.c_str(), op_id, stream_id);
    GE_TIMESTAMP_RESTART(GenerateTask);
    auto ret = kernel_info_store->GenerateTask(*node, run_context, task_def_list);
    GE_TIMESTAMP_ADD(GenerateTask);
    if (ret != SUCCESS) {
      GELOGE(ret, "Call %s to generate node[name:%s(%s), id:%ld, stream_id:%ld] task failed.",
             op_kernel_lib_name.c_str(), name.c_str(), type.c_str(), op_id, stream_id);
      return ret;
    }
    // Profiling task
    GE_CHK_STATUS_RET(InsertProfilingTaskAfter(op_desc, ppoint, ar_ppoint, node_index, task_def_list));

    size_t task_list_size_after = task_def_list.size();
    // If tasks is reduced
    if (task_list_size_after < task_list_size_before) {
      GELOGE(FAILED, "Call %s to generate node[name:%s(%s), id:%ld, stream_id:%ld] task. but task num from %zu to %zu.",
             op_kernel_lib_name.c_str(), name.c_str(), type.c_str(), op_id, stream_id, task_list_size_before,
             task_list_size_after);
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

    GELOGD("Call %s to generate node[name:%s(%s), id:%ld, stream_id:%ld] task finished, generate %lu task(s).",
           op_kernel_lib_name.c_str(), name.c_str(), type.c_str(), op_id, stream_id,
           task_list_size_after - task_list_size_before);
  }
  GE_TIMESTAMP_CALLNUM_END(GenerateTask, "GraphBuild::GenerateTask");
  return SUCCESS;
}

Status TaskGenerator::GenerateTaskForL1FusionNode(L1FusionTaskInfo &fusion_task_info,
                                                  std::map<int64_t, std::vector<NodePtr>> &l1_fusion_nodes,
                                                  std::unordered_set<Node *> &l1_fusion_nodes_seen) {
  Status ret = SUCCESS;
  int64_t group_id;
  auto &run_context = fusion_task_info.run_context;
  auto &graph = fusion_task_info.graph;
  auto &node = fusion_task_info.node;
  auto &fusion_op_desc = fusion_task_info.fusion_op_desc;
  auto &node_index = fusion_task_info.node_index;
  const auto &ops_kernel_manager = fusion_task_info.ops_kernel_manager;
  auto &task_def_list = fusion_task_info.task_def_list;
  auto &op_name_map = fusion_task_info.op_name_map;
  auto &ppoint = fusion_task_info.ppoint;
  auto &ar_ppoint = fusion_task_info.ar_ppoint;
  auto stream_id = fusion_op_desc->GetStreamId();
  // If op_desc have this attr, call nodes with same group id in a stream together
  if (ge::AttrUtils::GetInt(fusion_op_desc, ATTR_NAME_L1_FUSION_GROUP_ID, group_id) &&
      (l1_fusion_nodes_seen.count(node.get()) == 0)) {
    auto group_key = group_id + stream_id * kMaxNodeNumInNormalStream;
    GELOGI("L1Fusion: start fusion group index[%ld], nodes size[%ld].", group_key, l1_fusion_nodes[group_key].size());
    for (auto &fusion_node : l1_fusion_nodes[group_key]) {
      OpDescPtr op_desc = fusion_node->GetOpDesc();

      UpdateOpIsVarAttr(op_desc, graph->GetSessionID());
      std::string fusion_node_name = fusion_node->GetName();
      std::string fusion_node_type = fusion_node->GetType();
      std::string op_kernel_lib_name = op_desc->GetOpKernelLibName();
      if (op_kernel_lib_name.empty()) {
        GELOGI("L1Fusion: fusion_node[name:%s(%s)] task no need to generate task.", fusion_node_name.c_str(),
               fusion_node_type.c_str());
        continue;
      }

      size_t task_list_size_before = task_def_list.size();
      OpsKernelInfoStorePtr kernel_info_store = ops_kernel_manager.GetOpsKernelInfoStore(op_kernel_lib_name);
      if (kernel_info_store == nullptr) {
        GELOGE(INTERNAL_ERROR, "L1Fusion: No ops kernel store found. fusion_node:%s(%s), op_kernel_lib_name=%s.",
               fusion_node_name.c_str(), fusion_node_type.c_str(), op_kernel_lib_name.c_str());
        return INTERNAL_ERROR;
      }

      ret = UpdateAnchorStatus(fusion_node);
      if (ret != SUCCESS) {
        GELOGE(ret, "L1Fusion: Call UpdateAnchorStatus fusion_node:%s(%s) failed", fusion_node_name.c_str(),
               fusion_node_type.c_str());
        return ret;
      }

      int64_t op_id = op_desc->GetId();
      int64_t stream_id = op_desc->GetStreamId();
      if (stream_id < 0 || stream_id >= (int64_t)run_context.graphStreamList.size()) {
        GELOGE(INTERNAL_ERROR, "L1Fusion: fusion_node[name:%s(%s), id:%ld] stream id is invalid, stream list size=%zu",
               fusion_node_name.c_str(), fusion_node_type.c_str(), op_id, run_context.graphStreamList.size());
        return INTERNAL_ERROR;
      }
      // profiling task
      (void)InsertProfilingTaskBefore(op_desc, ppoint, ar_ppoint, node_index, task_def_list);
      run_context.stream = run_context.graphStreamList[stream_id];
      GELOGI("L1Fusion: Call %s to generate fusion_node:[fusion_node_name:%s(%s), id:%ld, stream_id:%ld] task.",
             op_kernel_lib_name.c_str(), fusion_node_name.c_str(), fusion_node_type.c_str(), op_id, stream_id);
      ret = kernel_info_store->GenerateTask(*fusion_node, run_context, task_def_list);
      if (ret != SUCCESS) {
        GELOGE(ret,
               "L1Fusion: Call %s to generate fusion_node:[fusion_node_name:%s(%s), "
               "id:%ld, stream_id:%ld] task failed.",
               op_kernel_lib_name.c_str(), fusion_node_name.c_str(), fusion_node_type.c_str(), op_id, stream_id);
        return ret;
      }
      // profiling task
      (void)InsertProfilingTaskAfter(op_desc, ppoint, ar_ppoint, node_index, task_def_list);
      size_t task_list_size_after = task_def_list.size();
      // if tasks is reduced
      if (task_list_size_after < task_list_size_before) {
        GELOGE(FAILED,
               "L1Fusion: Call %s to generate fusion_node:[fusion_node_name:%s(%s), "
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

      GELOGI(
        "L1Fusion: Call %s to generate fusion_node:[fusion_node_name:%s(%s), id:%ld, stream_id:%ld]"
        " task finished, generate %u task(s).",
        op_kernel_lib_name.c_str(), fusion_node_name.c_str(), fusion_node_type.c_str(), op_id, stream_id,
        task_list_size_after - task_list_size_before);

      // record nodes which have call generate task successfully
      l1_fusion_nodes_seen.insert(fusion_node.get());
      node_index++;
    }
  }
  // without tag or has been seen, skip directly
  return ret;
}

Status TaskGenerator::UpdateAnchorStatus(const NodePtr &node) {
  if (NodeUtils::SetAllAnchorStatus(node) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "NodeUtils::SetAllAnchorStatus failed.");
    return INTERNAL_ERROR;
  }
  for (auto &anchor : node->GetAllInDataAnchors()) {
    auto peer_anchor = anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_SUSPEND) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "AnchorUtils::SetStatus failed.");
        return INTERNAL_ERROR;
      }
    } else if (peer_anchor->GetOwnerNode()->GetType() == CONSTANT) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_CONST) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "AnchorUtils::SetStatus failed.");
        return INTERNAL_ERROR;
      }
    } else {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_DATA) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "AnchorUtils::SetStatus failed.");
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status TaskGenerator::MarkNodeAndSetIndex(ComputeGraphPtr &graph) {
  std::shared_ptr<GELib> ge_lib = GELib::GetInstance();
  if ((ge_lib == nullptr) || !ge_lib->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "GE is not initialized or is finalized");
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  int64_t node_index = 0;
  map<string, map<int64_t, std::pair<NodePtr, NodePtr>>> engine_stream_stat;
  for (auto &node : graph->GetAllNodes()) {
    const OpDescPtr &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    string op_kernel_lib_name = op_desc->GetOpKernelLibName();
    int64_t stream_id = op_desc->GetStreamId();
    op_desc->SetId(node_index++);

    if (op_kernel_lib_name.empty()) {
      // Reset op kernel lib
      (void)ge_lib->DNNEngineManagerObj().GetDNNEngineName(op_desc);
      op_kernel_lib_name = op_desc->GetOpKernelLibName();
      if (op_kernel_lib_name.empty()) {
        GELOGE(INTERNAL_ERROR, "node:%s(%s) get op kernel lib failed.", node->GetName().c_str(),
               node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }

    auto it = engine_stream_stat.find(op_kernel_lib_name);
    if (it == engine_stream_stat.end()) {
      map<int64_t, std::pair<NodePtr, NodePtr>> stream_map;
      std::pair<NodePtr, NodePtr> node_pair(node, node);
      (void)stream_map.emplace(stream_id, node_pair);
      (void)engine_stream_stat.emplace(op_kernel_lib_name, stream_map);
    } else {
      auto stream_it = it->second.find(stream_id);
      if (stream_it == it->second.end()) {
        std::pair<NodePtr, NodePtr> node_pair(node, node);
        (void)it->second.emplace(stream_id, node_pair);
      } else {
        stream_it->second.second = node;
      }
    }
  }

  for (auto &it : engine_stream_stat) {
    for (auto &stream_it : it.second) {
      NodePtr &first_node = stream_it.second.first;
      GE_CHK_BOOL_EXEC(ge::AttrUtils::SetBool(first_node->GetOpDesc(), kIsFirstNode, true),
                       GELOGE(FAILED, "SetBool failed.");
                       return FAILED);
      NodePtr &last_node = stream_it.second.second;
      GE_CHK_BOOL_EXEC(ge::AttrUtils::SetBool(last_node->GetOpDesc(), kIsLastNode, true),
                       GELOGE(FAILED, "SetBool failed.");
                       return FAILED);
    }
  }
  return SUCCESS;
}

Status TaskGenerator::FindProfilingTaskIndex(const ComputeGraphPtr &graph, ProfilingPoint &ppoint,
                                             vector<uint32_t> &ar_ppoint) const {
  GE_CHECK_NOTNULL(graph);
  const char *is_profiling = std::getenv(kProfilingMode);
  if (is_profiling == nullptr) {
    return SUCCESS;
  }
  const char *fp_point = std::getenv(kProfilingFpPoint);
  if (fp_point == nullptr) {
    GELOGW("first forward profiling op name not set.");
    return SUCCESS;
  }
  string fp_point_str = string(fp_point);
  const char *bp_point = std::getenv(kProfilingBpPoint);
  if (bp_point == nullptr) {
    GELOGW("last backward profiling op name not set.");
    return SUCCESS;
  }
  string bp_point_str = string(bp_point);
  uint32_t current_idx = 0;
  uint32_t iter_end = 0;
  uint32_t last_bp = 0;
  uint32_t first_fp = 0;
  for (auto &node : graph->GetAllNodes()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(node->GetOpDesc());
    current_idx++;
    string op_kernel_lib_name = op_desc->GetOpKernelLibName();
    if (op_kernel_lib_name.empty()) {
      continue;
    }

    if (op_desc->GetType() == NETOUTPUT) {
      iter_end = current_idx;
      GELOGI("Iter end name %s, idx %u", op_desc->GetName().c_str(), iter_end);
    }

    if (op_desc->GetType() == HCOMALLREDUCE) {
      ar_ppoint.emplace_back(current_idx);
      GELOGI("Allreduce name %s, idx %u", op_desc->GetName().c_str(), current_idx);
    }

    if (first_fp == 0 && IsProfPoint(op_desc, fp_point_str)) {
      first_fp = current_idx;
      GELOGI("First fp name %s, idx %u", op_desc->GetName().c_str(), first_fp);
    }

    if (IsProfPoint(op_desc, bp_point_str)) {
      last_bp = current_idx;
      GELOGI("Last bp name %s, idx %u", op_desc->GetName().c_str(), last_bp);
    }
  }
  ppoint.fp_index = first_fp;
  ppoint.bp_index = last_bp;
  ppoint.end_index = iter_end;
  bool train_graph = graph->GetNeedIteration();
  if (ppoint.fp_index == 0 && train_graph) {
    GELOGE(FAILED, "First forward op name can't be found in graph for training trace.");
  }
  if (ppoint.bp_index == 0 && train_graph) {
    GELOGE(FAILED, "Last backward op name can't be found in graph for training trace.");
  }
  return SUCCESS;
}

Status TaskGenerator::InsertProfilingTaskBefore(const OpDescPtr &op_desc, const ProfilingPoint &ppoint,
                                                vector<uint32_t> &ar_ppoint, uint32_t node_index,
                                                vector<domi::TaskDef> &task_def_list) {
  const char *is_profiling = std::getenv(kProfilingMode);
  if ((is_profiling == nullptr) || (ppoint.fp_index == 0) || (ppoint.bp_index == 0) || (ppoint.end_index == 0)) {
    return SUCCESS;
  }
  if (ppoint.fp_index == node_index) {
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

  for (size_t i = 0; i < ar_ppoint.size(); i++) {
    if (ar_ppoint[i] != node_index) {
      continue;
    }
    GELOGI("The start allreduce operator is %s, idx %u", op_desc->GetName().c_str(), node_index);
    TaskDef ar_task_def;
    ar_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
    ar_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *ar_log_def = ar_task_def.mutable_log_timestamp();
    if (ar_log_def != nullptr) {
      GE_IF_BOOL_EXEC(TypeUtils::CheckUint64MulOverflow(i, kProfilingArStep),
                      GELOGE(FAILED, "Multiply result is out of range.");
                      return FAILED);
      auto log_id = i * kProfilingArStep + kProfilingArStartLogid;
      ar_log_def->set_logid(log_id);
      ar_log_def->set_notify(false);
    }
    task_def_list.push_back(ar_task_def);
  }
  return SUCCESS;
}

Status TaskGenerator::InsertProfilingTaskAfter(const OpDescPtr &op_desc, const ProfilingPoint &ppoint,
                                               vector<uint32_t> &ar_ppoint, uint32_t node_index,
                                               vector<domi::TaskDef> &task_def_list) {
  GE_CHECK_NOTNULL(op_desc);
  const char *is_profiling = std::getenv(kProfilingMode);
  if ((is_profiling == nullptr) || (ppoint.fp_index == 0) || (ppoint.bp_index == 0) || (ppoint.end_index == 0)) {
    return SUCCESS;
  }
  if (ppoint.bp_index == node_index) {
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
  if (ppoint.end_index == node_index) {
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

  for (size_t i = 0; i < ar_ppoint.size(); i++) {
    if (ar_ppoint[i] != node_index) {
      continue;
    }
    GELOGI("The end allreduce operator is %s, idx %u", op_desc->GetName().c_str(), node_index);
    TaskDef ar_task_def;
    ar_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
    ar_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *ar_log_def = ar_task_def.mutable_log_timestamp();
    GE_CHECK_NOTNULL(ar_log_def);
    GE_IF_BOOL_EXEC(TypeUtils::CheckUint64MulOverflow(i, kProfilingArStep),
                    GELOGE(FAILED, "Multiply result is out of range.");
                    return FAILED);
    auto log_id = i * kProfilingArStep + kProfilingArEndLogid;
    ar_log_def->set_logid(log_id);
    ar_log_def->set_notify(false);
    task_def_list.emplace_back(ar_task_def);
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

}  // namespace ge
