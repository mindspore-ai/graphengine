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

#include "graph/manager/graph_manager.h"

#include <pthread.h>
#include <algorithm>
#include <future>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include "common/ge/ge_util.h"
#include "common/math/math_util.h"
#include "common/thread_pool.h"
#include "common/util.h"
#include "external/graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/common/transop_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/ge_local_context.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/passes/atomic_addr_clean_pass.h"
#include "graph/passes/compile_nodes_pass.h"
#include "graph/passes/constant_folding_pass.h"
#include "graph/passes/control_op_attr_pass.h"
#include "graph/passes/dimension_adjust_pass.h"
#include "graph/passes/identify_reference_pass.h"
#include "graph/passes/link_gen_mask_nodes_pass.h"
#include "graph/passes/multi_batch_pass.h"
#include "graph/passes/permute_pass.h"
#include "graph/passes/reshape_remove_pass.h"
#include "graph/passes/same_transdata_breadth_fusion_pass.h"
#include "graph/passes/transop_breadth_fusion_pass.h"
#include "graph/passes/transop_depth_fusion_pass.h"
#include "graph/passes/transop_nearby_allreduce_fusion_pass.h"
#include "graph/passes/transop_without_reshape_fusion_pass.h"
#include "graph/passes/cast_remove_pass.h"
#include "graph/passes/transpose_transdata_pass.h"
#include "graph/passes/variable_op_pass.h"
#include "graph/passes/variable_prepare_op_pass.h"
#include "graph/passes/variable_ref_delete_op_pass.h"
#include "graph/passes/replace_with_empty_const_pass.h"
#include "graph/utils/tensor_adapter.h"
#include "inc/pass_manager.h"
#include "init/gelib.h"

namespace {
const char *const kSummary = "Summary";
const char *const kSave = "Save";
const char *const kNetOutput = "NetOutput";
const char *const kVariable = "Variable";
const char *const kSend = "Send";
const char *const kRecv = "Recv";
}  // namespace

namespace ge {
GraphManager::GraphManager() : thread_run_flag_(false), graph_run_listener_(nullptr), init_flag_(false) {}

Status GraphManager::Initialize(const std::map<string, string> &options) {
  if (init_flag_) {
    GELOGW("[Initialize] GraphManager already initialized.");
    return SUCCESS;
  }

  // malloc
  graph_run_listener_ = MakeShared<GraphModelListener>(sync_run_mutex_, condition_);
  if (graph_run_listener_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed");
    return MEMALLOC_FAILED;
  }
  // graph context
  graph_context_ = MakeShared<GraphContext>();
  if (graph_context_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed.");
    return MEMALLOC_FAILED;
  }

  // parse option parameters
  Status ret = ParseOptions(options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Initialize] parse options failed.");
    return ret;
  }

  graph_builder_.SetOptions(options_);
  ret = graph_optimize_.SetOptions(options_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Initialize] Graph optimize initialize failed.");
    return ret;
  }
  graph_preparer_.SetOptions(options_);

  ret = graph_context_->Initialize(options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Initialize] GraphContext initialize failed.");
    return ret;
  }

  graph_map_.clear();
  cache_helper_map_.clear();
  init_flag_ = true;

  thread_run_flag_ = true;
  prerun_thread_ = std::thread(GraphManager::PreRunThread, this);
  run_thread_ = std::thread(GraphManager::RunThread, this);

  return SUCCESS;
}

Status GraphManager::Finalize() {
  if (!init_flag_) {
    GELOGW("GraphManager has not been initialized.");
    return SUCCESS;
  }

  if (graph_executor_.FreeExecuteMemory() != SUCCESS) {
    GELOGW("Graph executor FreeExecuteMemory failed, resources may not be released correctly.");
  }

  StopQueue(this);

  if (prerun_thread_.joinable()) {
    prerun_thread_.join();
  }
  if (run_thread_.joinable()) {
    run_thread_.join();
  }

  // check graph whether running or not
  Status unload_model_ret = SUCCESS;
  Status ret;
  rtError_t rt_ret;
  for (auto iter = graph_map_.begin(); iter != graph_map_.end(); ++iter) {
    GraphNodePtr graph_node = iter->second;
    if (graph_node->GetRunFlag()) {
      GELOGW("[GraphManager] finalize failed, graphId=%u.", iter->first);
      unload_model_ret = GE_GRAPH_GRAPH_IS_RUNNING;
      continue;
    }

    // unload model
    auto ge_model = graph_node->GetGeModel();
    if (ge_model != nullptr && ge_model->GetModelId() != INVALID_MODEL_ID && graph_node->GetLoadFlag()) {
      rt_ret = rtSetDevice(GetContext().DeviceId());
      if (rt_ret != RT_ERROR_NONE) {
        GELOGW("[GraphManager] rtSetDevice failed, modelId=%u, graphId=%u.", ge_model->GetModelId(), iter->first);
        unload_model_ret = FAILED;
        continue;
      }
      ret = GraphLoader::UnloadModel(ge_model->GetModelId());
      if (ret != SUCCESS) {
        GELOGW("[GraphManager] unload model failed, modelId=%u, graphId=%u.", ge_model->GetModelId(), iter->first);
        unload_model_ret = ret;
      }
      rt_ret = rtDeviceReset(GetContext().DeviceId());
      if (rt_ret != RT_ERROR_NONE) {
        GELOGW("[GraphManager] rtDeviceReset failed, modelId=%u, graphId=%u.", ge_model->GetModelId(), iter->first);
        unload_model_ret = FAILED;
        continue;
      }
    }
  }
  graph_map_.clear();
  cache_helper_map_.clear();

  // graph context
  if (graph_context_ != nullptr) {
    Status ret_final = graph_context_->Finalize();
    if (ret_final != SUCCESS) {
      GELOGE(ret_final, "[GraphManager] graph context Finalize failed!");
      unload_model_ret = ret_final;
    }
  }

  init_flag_ = false;
  return unload_model_ret;
}

Status GraphManager::AddGraph(const GraphId &graph_id, const Graph &graph,
                              const std::map<std::string, std::string> &options) {
  if (graph_map_.find(graph_id) != graph_map_.end()) {
    GELOGE(GE_GRAPH_GRAPH_ALREADY_EXIST, "[GraphManager] graph exists, graph_id = %u.", graph_id);
    return GE_GRAPH_GRAPH_ALREADY_EXIST;
  }

  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph != nullptr) {
    compute_graph->SetGraphID(graph_id);
  } else {
    GELOGE(FAILED, "compute graph is null");
    return FAILED;
  }
  std::string session_graph_id;
  if (!AttrUtils::GetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) || session_graph_id.empty()) {
    session_graph_id = "-1_" + to_string(graph_id);
    if (!AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
      GELOGW("Set attribute of compute graph failed.");
    }
    GELOGW("Get graph session_graph_id attr failed, set session id to default value: [0]");
  }

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  if (graph_node == nullptr) {
    GELOGE(FAILED, "GraphNode make shared failed");
    return FAILED;
  }
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  if (graph_ptr == nullptr) {
    GELOGE(FAILED, "GraphPtr make shared failed");
    return FAILED;
  }

  graph_node->SetGraph(graph_ptr);
  graph_node->SetOptions(options);

  graph_map_.insert(std::make_pair(graph_id, graph_node));

  GELOGI("[GraphManager] add graph success, graph_id = %u.", graph_id);

  var_acc_ctrl_.AddGraph(graph_id, compute_graph);
  return SUCCESS;
}

Status GraphManager::MergeSubGraph(ComputeGraphPtr &compute_graph, const ge::ComputeGraphPtr &original_compute_graph) {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr != nullptr && instance_ptr->InitFlag()) {
    Status ret = graph_partitioner_.MergeAfterSubGraphOptimization(compute_graph, original_compute_graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "merge end and placeholder after subGraph optimization failed.");
      return FAILED;
    }

    Status ret_topo = compute_graph->TopologicalSorting();
    if (ret_topo != SUCCESS) {
      GELOGE(ret_topo, "[GraphManager]: TopologicalSorting the merged graph failed.");
      return ret_topo;
    }
  } else {
    auto subgraph_list = graph_partitioner_.GetSubGraphMap();
    if (subgraph_list.find(original_compute_graph) != subgraph_list.end() &&
        !subgraph_list[original_compute_graph].empty() && subgraph_list[original_compute_graph][0] != nullptr) {
      compute_graph = subgraph_list[original_compute_graph][0]->GetSubGraph();
    }
  }

  return SUCCESS;
}

Status GraphManager::SetSubgraph(uint64_t session_id, ComputeGraphPtr compute_graph) {
  // use default 16 multi thread
  const uint32_t thread_num = 16;
  ThreadPool executor(thread_num);
  auto sub_graph_map = graph_partitioner_.GetSubGraphMap();
  std::vector<std::future<Status>> vector_future;
  const auto &root_subgraph_list = sub_graph_map[compute_graph];
  for (const auto &subgraph : root_subgraph_list) {
    std::future<Status> f = executor.commit(GraphManager::ProcessSubGraphWithMultiThreads, this, subgraph, session_id,
                                            GetThreadLocalContext());
    if (!f.valid()) {
      GELOGE(FAILED, "Future is invalid");
      return FAILED;
    }
    vector_future.emplace_back(std::move(f));
  }

  for (auto &function_graph : compute_graph->GetAllSubgraphs()) {
    auto subgraph_list = sub_graph_map[function_graph];
    for (const auto &subgraph : subgraph_list) {
      std::future<Status> f = executor.commit(GraphManager::ProcessSubGraphWithMultiThreads, this, subgraph, session_id,
                                              GetThreadLocalContext());
      if (!f.valid()) {
        GELOGE(FAILED, "Future is invalid");
        return FAILED;
      }
      vector_future.emplace_back(std::move(f));
    }
  }
  GELOGI("All sub graph num is %zu", vector_future.size());
  for (size_t i = 0; i < vector_future.size(); ++i) {
    Status ret_status = vector_future[i].get();
    if (ret_status != SUCCESS) {
      GELOGE(ret_status, "subgraph %zu optimize failed", i);
      return ret_status;
    }
  }
  return SUCCESS;
}

Status GraphManager::PreRun(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs,
                            vector<GeModelPtr> &ge_models, GeModelPtr &ge_model, uint64_t session_id) {
  GELOGI("Ready For PreRun Start session_id = %lu.", session_id);
  GE_TIMESTAMP_START(PreRun);
  GE_CHECK_NOTNULL(graph_node);
  // it will not execute graph preprocess, optimize, parition, build if the graph has built successful.
  GE_CHECK_NOTNULL(graph_node->GetGraph());
  auto compute_graph = GraphUtils::GetComputeGraph(*graph_node->GetGraph());
  GE_IF_BOOL_EXEC(compute_graph == nullptr, GELOGE(FAILED, "compute graph is NULL."); return FAILED);
  GraphUtils::DumpGEGraph(compute_graph, "BeforeSummaryHandle");
  GraphUtils::DumpGEGraphToOnnx(*compute_graph, "BeforeSummaryHandle");
  GEEVENT("PreRun start, graph node size is %zu", compute_graph->GetDirectNodesSize());
  // optimize the summary op in graph: store the summary name and replace the summary ops with net_output op.
  GE_TIMESTAMP_START(HandleSummaryOp);
  auto ret = graph_optimize_.HandleSummaryOp(compute_graph);
  GE_TIMESTAMP_END(HandleSummaryOp, "GraphManager::HandleSummaryOp");
  GE_CHK_BOOL_EXEC(ret == SUCCESS, return ret, "[RunTrainGraph] HandleSummaryOp failed.");
  GE_TIMESTAMP_START(GraphPrepare);
  ret = graph_preparer_.Prepare(graph_node->GetGraph(), inputs, compute_graph, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "ATC RunGraph input compute graph is NULL");
    return ret;
  }
  GE_TIMESTAMP_END(GraphPrepare, "GraphPrepare::Prepare");
  compute_graph->SetSessionID(session_id);
  GraphUtils::DumpGEGraph(compute_graph, "OptimizeOriginalGraphAfter");
  GraphUtils::DumpGEGraphToOnnx(*compute_graph, "OptimizeOriginalGraphAfter");

  GE_TIMESTAMP_START(InferShape);
  // Origin graph infershape
  GE_CHK_STATUS_EXEC(compute_graph->InferShapeInNeed(),
                     GELOGE(GE_GRAPH_INFERSHAPE_FAILED, " OriginGraph infershape failed");
                     return GE_GRAPH_INFERSHAPE_FAILED;)
  GE_TIMESTAMP_END(InferShape, "ComputeGraph::InferShapeInNeed");
  // graph partition
  // all sub graph list of root graph and sub graph
  GE_TIMESTAMP_START(GraphPartition);
  ret = graph_partitioner_.Partition(compute_graph, GraphPartitioner::kPartitioning);
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph partition Failed");
    return ret;
  }
  GE_TIMESTAMP_END(GraphPartition, "GraphPartitioner::Partition1");
  GE_TIMESTAMP_START(SetSubgraph);
  ret = SetSubgraph(session_id, compute_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph set subgraph Failed");
    return ret;
  }
  GE_TIMESTAMP_END(SetSubgraph, "SetSubGraph");

  ComputeGraphPtr merged_compute_graph = nullptr;
  std::vector<ComputeGraphPtr> merged_sub_graph_list;

  GE_TIMESTAMP_START(MergeSubgraph);
  ret = MergeSubGraph(merged_compute_graph, compute_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Merge SubGraph Failed");
    return ret;
  }
  merged_compute_graph->SetSessionID(session_id);
  merged_compute_graph->SetGraphID(graph_node->GetGraphId());
  GraphUtils::DumpGEGraph(merged_compute_graph, "mergedComputeGraph");
  GraphUtils::DumpGEGraphToOnnx(*merged_compute_graph, "mergedComputeGraph");
  for (auto &sub_graph : merged_compute_graph->GetAllSubgraphs()) {
    string subgraph_name = "mergedComputeGraph" + sub_graph->GetName();
    sub_graph->SetSessionID(session_id);
    sub_graph->SetGraphID(graph_node->GetGraphId());
    GraphUtils::DumpGEGraph(merged_compute_graph, subgraph_name);
    GraphUtils::DumpGEGraphToOnnx(*merged_compute_graph, subgraph_name);
  }
  GE_TIMESTAMP_END(MergeSubgraph, "GraphManager::MergeSubGraph");

  std::shared_ptr<GELib> instance_ge = ge::GELib::GetInstance();
  if (instance_ge != nullptr && instance_ge->InitFlag()) {
    // optimize after merge subgraph
    GE_TIMESTAMP_START(OptimizeAfterMergeSubgraph);
    ret = OptimizeAfterMergeSubGraph(merged_compute_graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "Optimize after merge subgraph failed.");
      return ret;
    }
    GE_TIMESTAMP_END(OptimizeAfterMergeSubgraph, "GraphManager::OptimizeAfterMergeSubGraph");
  }
  GraphUtils::DumpGEGraph(merged_compute_graph, "OptimizeMergeSubGraphAfter");
  GraphUtils::DumpGEGraphToOnnx(*merged_compute_graph, "OptimizeMergeSubGraphAfter");

  // build
  if (merged_compute_graph != nullptr) {
    std::string graph_name = merged_compute_graph->GetName();
    graph_name.append("_");
    graph_name.append(std::to_string(graph_node->GetGraphId()));
    merged_compute_graph->SetName(graph_name);
  }
  std::vector<SubGraphInfoPtr> sub_graph_list;
  ret = graph_builder_.Build(merged_compute_graph, sub_graph_list, ge_model, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "SubGraph build Failed.");
    return ret;
  }

  bool is_always_dump = false;
  PropertiesManager &properties_manager = PropertiesManager::Instance();
  if (!properties_manager.GetDumpOutputPath().empty()) {
    is_always_dump = true;
  }

  GraphUtils::DumpGEGraph(merged_compute_graph, "Build", is_always_dump);
  GraphUtils::DumpGEGraphToOnnx(*merged_compute_graph, "Build");

  // set modelptr to subgraph
  for (const auto &sub_graph_info : sub_graph_list) {
    sub_graph_info->SetGeModelPtr(ge_model);
  }

  ge_models.push_back(ge_model);

  GE_IF_BOOL_EXEC(sub_graph_list.empty(), GELOGE(FAILED, "Input graph must have at least one calculation op Node");
                  return FAILED;);
  sub_graph_list[0]->SetSubGraph(merged_compute_graph);
  // set subgraphlist to graphnode
  graph_node->SetSubGraph(sub_graph_list);
  // when set incre build, save om model and var manager
  auto save_ret = SaveCacheAfterBuild(graph_node->GetGraphId(), merged_compute_graph, ge_model);
  if (save_ret != SUCCESS) {
    GELOGW("Fail to save cache.");
  }
  // release rts generate context
  RtContextUtil::GetInstance().DestroyrtContexts();
  GE_TIMESTAMP_END(PreRun, "GraphManager::PreRun");
  GEEVENT("[GEPERFTRACE] GE PreRun End");
  return ret;
}

Status GraphManager::StartForRunGraph(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs,
                                      vector<GeModelPtr> &ge_models, uint64_t session_id) {
  // it will not execute graph prreprocess, optimize, parition, build if the graph has built successful.
  Status ret = SUCCESS;
  if (IsGraphNeedBuild(graph_node)) {
    if (graph_node->GetBuildFlag()) {
      GELOGE(PARAM_INVALID,
             "The graph %u need to re-build, you should remove it from GE "
             "first, then AddGraph again and rebuild it.",
             graph_node->GetGraphId());
      return PARAM_INVALID;
    }
    GeModelPtr ge_model = nullptr;
    // check need incre build.
    ret = IncreBuild(graph_node, ge_model);
    if (ret != SUCCESS) {
      ret = PreRun(graph_node, inputs, ge_models, ge_model, session_id);
      if (ret != SUCCESS) {
        GELOGE(ret, "PreRun Failed.");
        return ret;
      }
    }
    ret = LoadGraph(ge_model, graph_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "LoadGraph Failed.");
      return ret;
    }
    graph_node->SetBuildFlag(true);
    var_acc_ctrl_.SetGraphBuildEnd(graph_node->GetGraphId());
  } else if (!graph_node->GetLoadFlag()) {
    GeModelPtr ge_model = graph_node->GetGeModel();
    ret = LoadGraph(ge_model, graph_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "LoadGraph Failed.");
      return ret;
    }
  }
  return ret;
}
Status GraphManager::LoadGraph(const GeModelPtr &ge_model, const GraphNodePtr &graph_node) {
  GELOGI("[LoadGraph] run_graph_flag[%d], graph_id[%u]", options_.run_graph_flag, graph_node->GetGraphId());
  if (options_.run_graph_flag && ge_model != nullptr) {
    // synchronization run graph with model
    std::shared_ptr<GraphModelListener> model_listener = GetModelListener();
    ModelIdInfo model_id_info;
    if (getenv(kEnvGeuseStaticMemory) != nullptr) {
      GELOGI("[LoadGraph] GE_USE_STATIC_MEMORY is seted.");
    } else {
      GE_CHK_STATUS_RET(CheckAndReleaseMemory(ge_model, graph_node))
    }
    GE_TIMESTAMP_START(LoadGraph);
    Status ret = graph_loader_.LoadGraph(ge_model, model_listener, model_id_info);
    GE_TIMESTAMP_END(LoadGraph, "GraphManager::LoadGraph");
    if (ret != SUCCESS) {
      GELOGE(ret, "[StartForRunGraph] LoadGraph Failed");
      graph_node->SetRunFlag(false);
      return ret;
    }
    graph_node->SetLoadFlag(true);
    ge_model->SetModelId(model_id_info.model_id);
    graph_node->SetGeModel(ge_model);
  }
  return SUCCESS;
}

Status GraphManager::LoadFromCache(const GraphNodePtr &graph_node, const ModelCacheHelperPtr &cache_helper,
                                   GeModelPtr &ge_model) {
  auto graph_id = graph_node->GetGraphId();
  auto ret = cache_helper->LoadOmModelFromCache(ge_model);
  if (ret != SUCCESS) {
    GELOGW("Fail to load om model from cache.");
    if (cache_helper->ClearCache(graph_id) != SUCCESS) {
      GELOGW("Fail to clear cache of graph %u.", graph_id);
    }
    return FAILED;
  }
  ret = cache_helper->RecoverVarManagerFromCache();
  if (ret != SUCCESS) {
    GELOGW("Fail to recover VarManager from cache.");
    if (cache_helper->ClearCache(graph_id) != SUCCESS) {
      GELOGW("Fail to clear cache of graph %u.", graph_id);
    }
    return FAILED;
  }
  ComputeGraphPtr compute_graph_in_model = GraphUtils::GetComputeGraph(ge_model->GetGraph());
  if (compute_graph_in_model == nullptr) {
    GELOGW("Error occurred when get compute graph from om, abandon.");
    return FAILED;
  } else {
    graph_node->SetComputeGraph(compute_graph_in_model);
    graph_node->SetGeModel(ge_model);
    GELOGI("Load model and graph form cache om file.");
  }
  return SUCCESS;
}

Status GraphManager::SaveCacheBeforeBuild(uint32_t graph_id, const ModelCacheHelperPtr &cache_helper) {
  auto ret = cache_helper->SaveCacheInfoToCache();
  if (ret != SUCCESS) {
    GELOGW("Fail to save cache info of graph[%d] to cache.", graph_id);
    return FAILED;
  }
  ret = cache_helper->SaveVarManagerToCache(true);
  if (ret != SUCCESS) {
    GELOGW("Fail to save var manager to cache.");
    cache_helper->ClearCache(graph_id);
    return FAILED;
  }
  GELOGI("Cache files have been saved.");
  return SUCCESS;
}

Status GraphManager::SaveCacheAfterBuild(uint32_t graph_id, ge::ComputeGraphPtr graph, GeModelPtr &ge_model) {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || !instance_ptr->InitFlag()) {
    GELOGW("GELib not initialized.");
    return FAILED;
  }

  if (instance_ptr->IsIncreBuild()) {
    auto iter = cache_helper_map_.find(graph_id);
    if (iter == cache_helper_map_.end()) {
      GELOGW("Can not find ModelCacheHelper of graph[%u]", graph_id);
      return FAILED;
    } else {
      ModelCacheHelperPtr cache_helper = iter->second;
      auto ret = cache_helper->RefreshComputeGraph(graph);
      if (ret != SUCCESS) {
        cache_helper->ClearCache(graph_id);
        GELOGW("Fail to refresh cache helper's compute graph");
        return FAILED;
      }
      ret = cache_helper->SaveVarManagerToCache(false);
      if (ret != SUCCESS) {
        cache_helper->ClearCache(graph_id);
        GELOGW("Fail to save VarManager to cache");
        return FAILED;
      }
      ret = cache_helper->SaveOmModelToCache(ge_model);
      if (ret != SUCCESS) {
        cache_helper->ClearCache(graph_id);
        GELOGW("Fail to save om model to cache");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status GraphManager::InnerRunGraph(GraphNodePtr &graph_node, const GraphId &graph_id,
                                   const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) {
  Status ret = graph_executor_.SetCondition(&sync_run_mutex_, &condition_, graph_run_listener_);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_RUNGRAPH_FAILED, "[RunGraph] set condition failed, graph_id = %u.", graph_id);
    graph_node->SetRunFlag(false);
    return GE_GRAPH_RUNGRAPH_FAILED;
  }

  if (GetTrainFlag()) {
    GE_CHK_STATUS_RET(graph_executor_.SetGraphContext(GetGraphContext()))
    graph_executor_.SetTrainFlag(options_.train_graph_flag);
  }
  ret = graph_executor_.ExecuteGraph(graph_id, graph_node->GetGeModel(), inputs, outputs);

  graph_node->SetRunFlag(false);
  if (ret != SUCCESS) {
    GELOGE(ret, "[RunGraph] execute graph failed, graph_id = %u.", graph_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphManager::RunGraph(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                              std::vector<GeTensor> &outputs, uint64_t session_id) {
  std::lock_guard<std::mutex> lock(run_mutex_);
  GELOGI("[RunGraph] start to run graph, graph_id = %u, is_train_graph: %d", graph_id, GetTrainFlag());

  if (inputs.empty()) {
    GELOGI("[RunGraph] initilize sub graph has no inputs.");
  }

  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[RunGraph] graph not exist, graph_id = %u.", graph_id);
    return ret;
  }

  if (graph_node == nullptr) {
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[RunGraph] graph node is NULL, graph_id = %u.", graph_id);
    return GE_GRAPH_GRAPH_NODE_NULL;
  }

  if (graph_node->GetRunFlag()) {
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[RunGraph] graph already running, graph id = %u", graph_id);
    return GE_GRAPH_ALREADY_RUNNING;
  }
  // set graph's run flag
  graph_node->SetRunFlag(true);
  ComputeGraphPtr compute_graph_tmp = GraphUtils::GetComputeGraph(*(graph_node->GetGraph()));

  GE_IF_BOOL_EXEC(
    GetTrainFlag(),
    GE_IF_BOOL_EXEC(compute_graph_tmp == nullptr,
                    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[RunGraph] compute_graph_tmp is NULL, graph id = %u.", graph_id);
                    return GE_GRAPH_GRAPH_NODE_NULL;))

  // when set incre build, add cache helper map
  AddModelCacheHelperToMap(graph_id, session_id, compute_graph_tmp);

  std::vector<GeModelPtr> ge_models;

  if (options_.local_fmk_op_flag) {
    graph_optimize_.TranFrameOp(compute_graph_tmp);
  }

  ret = StartForRunGraph(graph_node, inputs, ge_models, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[RunGraph] StartForRunGraph failed!");
    graph_node->SetRunFlag(false);
    return ret;
  }

  const std::vector<SubGraphInfoPtr> &all_sub_graph = graph_node->GetAllSubGraph();

  // excute graph
  ret = InnerRunGraph(graph_node, graph_id, inputs, outputs);
  if (ret != SUCCESS) {
    return ret;
  }

  if (GetTrainFlag()) {
    if (compute_graph_tmp->IsSummaryGraph()) {
      ret = SummaryHandle(graph_id, outputs);
      if (ret != SUCCESS) {
        GELOGE(ret, "[RunGraph] SummaryHandle failed!");
      }
    }

    if (!all_sub_graph.empty()) {
      auto checkPointGraph = all_sub_graph[0]->GetSubGraph();
      if (IsCheckpointGraph(checkPointGraph)) {
        ret = CheckpointHandle(graph_id, checkPointGraph, outputs);
        if (ret != SUCCESS) {
          GELOGE(ret, "[RunGraph] CheckpointHandle failed!");
        }
      }
    }
  }

  GELOGI("[RunGraph] run graph success, graph_id = %u.", graph_id);
  return SUCCESS;
}

Status GraphManager::BuildGraph(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                                std::vector<GeModelPtr> &models) {
  GELOGI("[BuildGraph] start to build graph, graph_id=%u.", graph_id);
  if (inputs.empty()) {
    GELOGW("[BuildGraph] BuildGraph warning: empty GeTensor inputs");
  }

  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[BuildGraph] graph not exist, graph_id = %u.", graph_id);
    return ret;
  }

  if (graph_node == nullptr) {
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[BuildGraph] graph node is NULL, graphId = %u.", graph_id);
    return GE_GRAPH_GRAPH_NODE_NULL;
  }

  if (graph_node->GetRunFlag()) {
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[BuildGraph] graph already running, graph id = %u", graph_node->GetGraphId());
    return GE_GRAPH_ALREADY_RUNNING;
  }
  // set graph's run flag
  graph_node->SetRunFlag(true);

  struct timeval tv;
  if (gettimeofday(&tv, nullptr) != 0) {
    GELOGE(INTERNAL_ERROR, "get the time of day failed.");
    return INTERNAL_ERROR;
  }
  uint64_t session_id = static_cast<uint64_t>(tv.tv_sec * 1000000 + tv.tv_usec);  // 1000000us
  ret = StartForRunGraph(graph_node, inputs, models, session_id);
  graph_node->SetRunFlag(false);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_PRERUN_FAILED, "[BuildGraph] StartForRunGraph failed!");
    return GE_GRAPH_PRERUN_FAILED;
  }

  GELOGI("[BuildGraph] build graph success, graph_id=%u.", graph_id);
  return ret;
}

///
/// @ingroup ge_graph
/// @brief Save extra attribute to Model
/// @param [in] model: Model attribues will save to.
/// @param [in] type: type of OpDesc.
/// @param [in] attrs: attributes of OpDesc.
/// @param [in] inputs: inputs tensor.
/// @param [in] outputs: outputs tensor.
/// @return: Status
///
Status GraphManager::SaveParams(ge::GeModel &model, const std::string &type, const std::map<string, GeAttrValue> &attrs,
                                const std::vector<GeTensor> &inputs, const std::vector<GeTensor> &outputs) {
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(&model, "ATTR_MODEL_OP_TYPE", type), return FAILED, "Set Op[%s] type fail",
                   type.c_str());

  for (const auto &it : attrs) {
    GE_CHK_BOOL_EXEC(model.SetAttr("ATTR_MODEL_" + it.first, it.second) == GRAPH_SUCCESS, return FAILED,
                     "Set OpDesc attribute[%s] fail", it.first.c_str());
  }

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListTensor(&model, "ATTR_MODEL_TENSOR_INPUTS", inputs), return FAILED,
                   "Set Inputs tensor list fail");
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListTensor(&model, "ATTR_MODEL_TENSOR_OUTPUTS", outputs), return FAILED,
                   "Set Outputs tensor list fail");

  return SUCCESS;
}

void GraphManager::RemoveModelCacheHelper(const GraphId &graph_id) {
  auto iter = cache_helper_map_.find(graph_id);
  if (iter != cache_helper_map_.end()) {
    cache_helper_map_.erase(iter);
  } else {
    GELOGW("[GraphManager] cache helper does not exist, graph_id = %u", graph_id);
  }
}

Status GraphManager::RemoveGraph(const GraphId &graph_id) {
  auto it = graph_map_.find(graph_id);
  if (it == graph_map_.end()) {
    GELOGE(GE_GRAPH_GRAPH_NOT_EXIST, "[GraphManager] Id %u does not exists.", graph_id);
    return GE_GRAPH_GRAPH_NOT_EXIST;
  }

  GraphNodePtr graph_node = it->second;
  if ((graph_node == nullptr) || (graph_node->GetRunFlag())) {
    GELOGE(GE_GRAPH_GRAPH_IS_RUNNING, "[GraphManager] Id %u is running, can't be deleted.", graph_id);
    return GE_GRAPH_GRAPH_IS_RUNNING;
  }
  Status ret = SUCCESS;
  Status middle_ret;
  rtError_t rt_ret;
  const std::vector<SubGraphInfoPtr> &all_sub_graph = graph_node->GetAllSubGraph();
  for (size_t i = 0; i < all_sub_graph.size(); ++i) {
    // must free buffer firstly
    middle_ret = all_sub_graph[i]->FreeInOutBuffer();
    if (middle_ret != SUCCESS) {
      GELOGE(middle_ret, "[GraphManager] RemoveGraph free mem failed, graph_id=%u.", graph_id);
      ret = middle_ret;
    }
    if (all_sub_graph[i]->GeModelIsValid() && all_sub_graph[i]->GetModelIdInfo().model_id != INVALID_MODEL_ID) {
      // unload model
      GELOGI("UnloadModel via new ome.");
      rt_ret = rtSetDevice(GetContext().DeviceId());
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "[GraphManager:] rtSetDevice failed, modelId=%u, graphId=%u.",
               all_sub_graph[i]->GetModelIdInfo().model_id, graph_id);
        ret = FAILED;
        continue;
      }
      middle_ret = GraphLoader::UnloadModel(all_sub_graph[i]->GetModelIdInfo().model_id);
      if (middle_ret != SUCCESS) {
        GELOGE(middle_ret, "[GraphManager:] unload model failed, modelId=%u, graph_id=%u.",
               all_sub_graph[i]->GetModelIdInfo().model_id, graph_id);
        ret = middle_ret;
      }
      rt_ret = rtDeviceReset(GetContext().DeviceId());
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "[GraphManager:] unload model failed, modelId=%u, graphId=%u.",
               all_sub_graph[i]->GetModelIdInfo().model_id, graph_id);
        ret = FAILED;
      }
    }
  }
  var_acc_ctrl_.RemoveGraph(graph_id);
  graph_map_.erase(it);

  RemoveModelCacheHelper(graph_id);

  auto ge_model = graph_node->GetGeModel();
  if (ge_model != nullptr) {
    GELOGI("Unload model %u.", ge_model->GetModelId());
    rt_ret = rtSetDevice(GetContext().DeviceId());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[GraphManager:] rtSetDevice failed, modelId=%u, graphId=%u.", ge_model->GetModelId(),
             graph_id);
      return FAILED;
    }
    middle_ret = GraphLoader::UnloadModel(ge_model->GetModelId());
    if (middle_ret != SUCCESS) {
      GELOGE(middle_ret, "[GraphManager:] unload model failed, modelId=%u, graph_id=%u.", ge_model->GetModelId(),
             graph_id);
      ret = middle_ret;
    }
    rt_ret = rtDeviceReset(GetContext().DeviceId());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[GraphManager:] rtDeviceReset failed, modelId=%u, graphId=%u.", ge_model->GetModelId(),
             graph_id);
      ret = FAILED;
    }
  }
  GE_CHK_STATUS_RET(ret, "[GraphManager:] Remove graph failed, graph_id=%u.", graph_id);
  GELOGI("[GraphManager] remove graph success, graph_id=%u.", graph_id);
  return SUCCESS;
}

Status GraphManager::ParseOptions(const std::map<std::string, std::string> &options) {
  Status ret;

  ParseOption(options, "ge.INPUT_NODES_SET_FP16", options_.input_nodes_set_fp16);
  // parse streams max parallel num
  ret = ParseOption(options, STREAM_MAX_PARALLEL_NUM, options_.stream_max_parallel_num);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID,
           "parse Key:%s value failed, it must be same format as "
           "DNN_V100:2,DNN_HCCL:3",
           STREAM_MAX_PARALLEL_NUM.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // get stream num
  ret = ParseOption(options, STREAM_NUM, options_.stream_num);
  if ((ret != SUCCESS) || (options_.stream_num == 0)) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:ge.stream_num, its value %d is invalid, must be not equal zero.",
           options_.stream_num);
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // get perf level, its value please see enum PerfLevel
  ret = ParseOption(options, PERF_LEVEL, options_.perf_level);
  if ((ret != SUCCESS) || IsPerfLevelInvalid(options_.perf_level)) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:ge.perfLevel, its value %d is invalid, must be enum PerfLevel type.",
           options_.perf_level);
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // get encrypt mode
  ret = ParseOption(options, ENCRYPT_MODE, options_.encrypt_mode);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:ge.encryptMode value invalid.");
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // get ek file
  ParseOption(options, EK_FILE, options_.ek_file);

  // get cert file
  ParseOption(options, CERT_FILE, options_.cert_file);

  // get hw key file
  ParseOption(options, HW_KEY_FILE, options_.hw_key_file);

  // get private file
  ParseOption(options, PRIVATE_KEY_FILE, options_.private_key_file);

  // get framework type, its value please see enum FrameworkType
  ret = ParseOption(options, FRAMEWORK_TYPE, options_.framework_type);
  if (ret != SUCCESS) {
    // print error log in ParseOption
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // get calibration info file
  ParseOption(options, CALIBRATION_CONF_FILE, options_.calibration_conf_file);

  // get insert op info file
  ParseOption(options, INSERT_OP_FILE, options_.insert_op_file);

  // get output node name
  ParseOption(options, OUTPUT_NODE_NAME, options_.output_node_name);

  // get function bin path
  ParseOption(options, "ge.func_bin_path", options_.func_bin_path);

  // get core type
  ParseOption(options, CORE_TYPE, options_.core_type);

  // get weight compress flag
  ret = ParseOption(options, COMPRESS_FLAG, options_.compress_flag);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:ge.compressFlag value is invalid, must be 0 or 1.");
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // ge.graphType.
  options_.run_graph_flag = true;
  ret = ParseOption(options, RUN_FLAG, options_.run_graph_flag);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:ge.runFlag value is invalid, must be 0 or 1.");
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // ge.graphType
  ret = ParseTrainGraphFlag(options_.run_graph_flag, options_.train_graph_flag);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:ge.runFlag value is invalid");
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // parse FmkOp
  options_.local_fmk_op_flag = false;
  ret = ParseOption(options, LOCAL_FMKOP_FLAG, options_.local_fmk_op_flag);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:ge.localFmkopFlag value is invalid, must be 0 or 1.");
    return GE_GRAPH_OPTIONS_INVALID;
  }
  options_.enable_print_op_pass = true;
  ret = ParseOption(options, ENABLE_PRINT_OP_PASS, options_.enable_print_op_pass);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:ge.enablePrintOpPass value is invalid, must be 0 or 1.");
                  return GE_GRAPH_OPTIONS_INVALID);
  // parse hcom parallel
  options_.hcom_parallel = false;
  ret = ParseOption(options, HCOM_PARALLEL, options_.hcom_parallel);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:ge.hcomParallel value is invalid, must be 0 or 1.");
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // net output node dataType
  ParseOption(options, OUTPUT_DATATYPE, options_.output_datatype);
  if (!options_.output_datatype.empty()) {
    domi::GetContext().output_type = options_.output_datatype;
  }

  // Set save_original_model flag (ge.save_original_model)
  ParseOption(options, SAVE_ORIGINAL_MODEL, options_.save_original_model);
  GELOGI("Set save original model flag %s", options_.save_original_model.c_str());
  // Original model file name
  ParseOption(options, ORIGINAL_MODEL_FILE, options_.original_model_file);

  return SUCCESS;
}

Status GraphManager::ParseTrainGraphFlag(bool &options, bool &option) {
  std::shared_ptr<GELib> ge_instance_ptr = ge::GELib::GetInstance();
  if (ge_instance_ptr == nullptr) {
    GELOGW("[Initialize] set train_graph_flag_ to 0 when GE is not initialized or finalized.");
    option = false;
  } else if (!ge_instance_ptr->isTrainMode()) {
    option = false;
  } else {  //  ge_instance_ptr->isTrainMode() is true
    if (!options) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID,
             "Key:ge.runFlag, its value %d is invalid, it must be 1 when GElib::is_train_mode_ flag is 1", options);
      return GE_GRAPH_OPTIONS_INVALID;
    }
    option = true;
  }
  return SUCCESS;
}

bool GraphManager::IsPerfLevelInvalid(int32_t perf_level) {
  return ((perf_level != static_cast<int32_t>(GEN_TASK_WITHOUT_L2FUSION)) &&
          (perf_level != static_cast<int32_t>(GEN_TASK_WITHOUT_FUSION)) && (perf_level != -1));
}

void GraphManager::ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                               std::string &option) {
  auto iter = options.find(key);
  if (iter != options.end()) {
    option = iter->second;
  }
}

Status GraphManager::ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                                 bool &option) {
  auto iter = options.find(key);
  if (iter != options.end()) {
    string flag = iter->second;
    if (flag == "0") {
      option = false;
    } else if (flag == "1") {
      option = true;
    } else {
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:%s, its value %s is invalid, it must be 0 or 1.", key.c_str(),
             flag.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }
  }
  return SUCCESS;
}

Status GraphManager::ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                                 int &option) {
  const int kDecimal = 10;
  char *ptr = nullptr;
  auto iter = options.find(key);
  if (iter != options.end()) {
    option = static_cast<int32_t>(std::strtol(iter->second.c_str(), &ptr, kDecimal));
    if (ptr != nullptr && *ptr != '\0') {
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "Key:%s, its value %s is invalid, must be int32_t type.", key.c_str(),
             iter->second.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }
  }
  return SUCCESS;
}

void GraphManager::Trim(std::string &str) {
  if (!str.empty()) {
    auto it = str.find_first_not_of(" ");
    if (it != std::string::npos) {
      str.erase(0, it);
    }
    it = str.find_last_not_of(" ");
    if (it != std::string::npos) {
      str.erase(it + 1);
    }
  }
}

Status GraphManager::ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                                 std::map<std::string, int> &option) {
  auto iter = options.find(key);
  if (iter == options.end()) {
    return SUCCESS;
  }
  GELOGI("Start to parse %s", key.c_str());
  option.clear();
  std::string op_num = iter->second;

  // split string by ','
  std::vector<std::string> split;
  std::istringstream f(op_num);
  std::string str_tmp;
  while (getline(f, str_tmp, ',')) {
    split.push_back(str_tmp);
  }

  for (const std::string &engine_parallel : split) {
    // split engine and num by :
    size_t pos = engine_parallel.find(':');
    if (pos == string::npos) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID,
             "engine and num must be connected by :, "
             "while your input is %s",
             engine_parallel.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }
    std::string engine_name = engine_parallel.substr(0, pos);
    std::string parallel_num = engine_parallel.substr(pos + 1);
    Trim(engine_name);
    Trim(parallel_num);

    Status ret = CheckEngineName(engine_name, key, option);
    if (ret != SUCCESS) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "check engine name : %s failed, ", engine_name.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }

    int num = 0;
    ret = ParseParallelNum(parallel_num, key, num);
    if (ret != SUCCESS) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "parse parallel num failed");
      return GE_GRAPH_OPTIONS_INVALID;
    }

    option.insert(std::make_pair(engine_name, num));
  }
  GELOGI("Parse %s successfully", key.c_str());
  return SUCCESS;
}

Status GraphManager::CheckEngineName(const std::string &engine_name, const std::string &key,
                                     const std::map<std::string, int> &option) {
  if (engine_name.empty()) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "engine name of %s is empty", key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }
  // judge whether exist in engine list
  if (!GELib::GetInstance()->DNNEngineManagerObj().IsEngineRegistered(engine_name)) {
    GELOGW("engine : %s is not registered in %s", engine_name.c_str(), key.c_str());
  }

  auto it_stream_repeat = option.find(engine_name);
  if (it_stream_repeat != option.end()) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "engine : %s of %s is repeated", engine_name.c_str(), key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }
  return SUCCESS;
}

Status GraphManager::ParseParallelNum(const std::string &parallel_num, const std::string &key, int &num) {
  if (parallel_num.empty()) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "parallel num of %s is empty", key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }
  for (char c : parallel_num) {
    if (!isdigit(c)) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "%s input is invalid ", key.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }
  }

  try {
    num = std::stoi(parallel_num);
  } catch (std::invalid_argument &) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "parallel num : %s of %s is invalid argument", parallel_num.c_str(), key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  } catch (std::out_of_range &) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "parallel num : %s of %s is out of range", parallel_num.c_str(), key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  } catch (...) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "parallel num : %s of %s is invalid argument", parallel_num.c_str(), key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }

  if (num < 1) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "parallel num : %s of %s must bigger than 0", parallel_num.c_str(), key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }
  return SUCCESS;
}

Status GraphManager::GetGraphNode(const GraphId &graph_id, GraphNodePtr &out) {
  auto iter = graph_map_.find(graph_id);
  if (iter == graph_map_.end()) {
    out = nullptr;
    GELOGE(GE_GRAPH_GRAPH_NOT_EXIST, "[GraphManager] graph not exist, graph_id= %u.", graph_id);
    return GE_GRAPH_GRAPH_NOT_EXIST;
  }
  out = iter->second;
  return SUCCESS;
}

Status GraphManager::GetVariable(const std::string &name, Tensor &val) {
  GeTensorPtr ge_tensor_ptr = TensorAdapter::AsGeTensorPtr(val);
  GE_CHECK_NOTNULL(ge_tensor_ptr);
  return GetGraphContext()->GetVariableTensor(name, *(ge_tensor_ptr.get()));
}

Status GraphManager::SummaryHandle(const GraphId &graph_id, std::vector<GeTensor> &outputs) {
  std::vector<GeTensor> without_summary_outputs;
  std::set<int> summary_output_index;
  GELOGI("[GraphManager] SummaryHandle, outputsSize=%zu.", outputs.size());
  const std::map<uint32_t, std::map<string, size_t>> &whole_summary_output_indexes =
    graph_optimize_.GetSummaryOutputIndexes();
  if (whole_summary_output_indexes.find(graph_id) == whole_summary_output_indexes.end()) {
    GELOGE(FAILED, "No Summary graph found in map.");
    return FAILED;
  }
  const std::map<string, size_t> &summary_output_indexes = whole_summary_output_indexes.at(graph_id);
  GELOGI("[GraphManager] SummaryHandle, summaryOutputIndexesSize=%zu.", summary_output_indexes.size());
  std::map<string, Tensor> summary_results;
  for (auto iter = summary_output_indexes.begin(); iter != summary_output_indexes.end(); ++iter) {
    GELOGI("[GraphManager] SummaryHandle, summaryName=%s, outputIndex=%zu.", iter->first.c_str(), iter->second);
    summary_results.emplace(iter->first, TensorAdapter::AsTensor(outputs.at(iter->second)));
    summary_output_index.emplace(iter->second);
  }

  // remove summary data from outputs
  if (!summary_output_index.empty()) {
    for (size_t j = 0; j < outputs.size(); ++j) {
      if (summary_output_index.count(j) == 0) {
        without_summary_outputs.emplace_back(outputs.at(j));
      }
    }
    outputs.swap(without_summary_outputs);
    GELOGI("[GraphManager] SummaryHandle, after swap outputsSize=%zu.", outputs.size());
  }

  if (!summary_results.empty()) {
    return PushSummaryData2ME(graph_id, summary_results);
  }

  return SUCCESS;
}

Status GraphManager::CheckpointHandle(const GraphId &graph_id, const ComputeGraphPtr &compute_graph,
                                      const std::vector<GeTensor> &outputs) {
  GELOGI("[GraphManager] CheckpointHandle, outputsSize=%zu.", outputs.size());
  std::vector<InputOutputDescInfo> outputs_desc = graph_executor_.GetOutputsDesc();
  GELOGI("[GraphManager] CheckpointHandle, outputsDescSize=%zu.", outputs_desc.size());

  std::map<string, Tensor> save_results;
  NodePtr netoutput = nullptr;
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == kNetOutput) {
      netoutput = node;
      break;
    }
  }
  if (netoutput == nullptr) {
    GELOGE(FAILED, "Netoutput is null.");
    return FAILED;
  }
  for (const auto &in : netoutput->GetAllInDataAnchors()) {
    std::string desc_name;
    auto out_anchor = in->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      GELOGE(FAILED, "out_anchor is null.");
      return FAILED;
    }
    ge::NodePtr peer_node = out_anchor->GetOwnerNode();
    // find the variable node in graph
    while (peer_node != nullptr && peer_node->GetType() != kVariable) {
      if (peer_node->GetAllInDataAnchors().size() != 1) {
        GELOGE(FAILED, "More than one prior nodes of peer_node %s in checkpoint Graph.", peer_node->GetName().c_str());
        return FAILED;
      }
      auto peer_node_in = peer_node->GetAllInDataAnchors().at(0);
      auto peer_node_out_anchor = peer_node_in->GetPeerOutAnchor();
      if (peer_node_out_anchor != nullptr) {
        peer_node = peer_node_out_anchor->GetOwnerNode();
        if (peer_node->GetType() == kVariable) {
          break;
        }
      }
    }
    if (peer_node == nullptr) {
      GELOGE(FAILED, "No variable op found in one branch, checkpoint graph illegal.");
      return FAILED;
    }
    desc_name = peer_node->GetName();
    GELOGI("[GraphManager] CheckpointHandle, descName=%s.", desc_name.c_str());
    if (in->GetIdx() >= static_cast<int>(outputs.size())) {
      GELOGE(FAILED, "variable index out of range.");
      return FAILED;
    }
    save_results.emplace(desc_name, TensorAdapter::AsTensor(outputs.at(in->GetIdx())));
  }

  if (!save_results.empty()) {
    return PushSaveData2ME(graph_id, save_results);
  }

  return SUCCESS;
}

Status GraphManager::RegisterCallBackFunc(
  const std::string &key, const std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)> &callback) {
  GELOGI("[GraphManager] RegisterCallBackFunc, key=%s.", key.c_str());
  me_callback_map_[key] = callback;
  return SUCCESS;
}

Status GraphManager::PushSummaryData2ME(const GraphId &graph_id,
                                        const std::map<std::string, ge::Tensor> &summary_data) {
  GELOGI("[GraphManager] PushSummaryData2ME, dataSize=%zu.", summary_data.size());
  auto itr = me_callback_map_.find(kSummary);
  if (itr == me_callback_map_.end()) {
    GELOGE(FAILED, "[GraphManager] PushSummaryData2ME failed, not found summary callback.");
    return FAILED;
  }
  return itr->second(graph_id, summary_data);
}

Status GraphManager::PushSaveData2ME(const GraphId &graph_id, const std::map<std::string, ge::Tensor> &save_data) {
  GELOGI("[GraphManager] PushSaveData2ME, dataSize=%zu.", save_data.size());
  auto itr = me_callback_map_.find(kSave);
  if (itr == me_callback_map_.end()) {
    GELOGE(FAILED, "[GraphManager] PushSaveData2ME failed, not found checkpoint callback.");
    return FAILED;
  }
  return itr->second(graph_id, save_data);
}

bool GraphManager::CheckNetOutputForCheckpointGraph(NodePtr &node) {
  size_t in_data_anchor_size = node->GetAllInDataAnchors().size();
  for (size_t i = 0; i < in_data_anchor_size; ++i) {
    auto in = node->GetInDataAnchor(i);
    if (in == nullptr) {
      return false;
    }
    auto peerin = in->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peerin == nullptr, return false);
    if (peerin->GetOwnerNode()->GetType() != kVariable && (!TransOpUtil::IsTransOp(peerin->GetOwnerNode()))) {
      return false;
    }
  }
  return true;
}

bool GraphManager::CheckVariableForCheckpointGraph(NodePtr &node) {
  auto out = node->GetOutDataAnchor(0);
  if (out == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "out is nullptr.");
    return false;
  }
  auto peer_out = out->GetPeerInDataAnchors();
  for (size_t i = 0; i < peer_out.size(); ++i) {
    if (peer_out.at(i)->GetOwnerNode()->GetType() != kNetOutput &&
        (!TransOpUtil::IsTransOp(peer_out.at(i)->GetOwnerNode()))) {
      return false;
    }
  }
  return true;
}

bool GraphManager::CheckTransOpForCheckpointGraph(NodePtr &node) {
  for (const auto &out_node : node->GetOutAllNodes()) {
    if ((!TransOpUtil::IsTransOp(out_node)) && (out_node->GetType() != kNetOutput) && (out_node->GetType() != kSend)) {
      return false;
    }
  }

  for (const auto &in_node : node->GetInAllNodes()) {
    if ((!TransOpUtil::IsTransOp(in_node)) && (in_node->GetType() != kVariable) && (in_node->GetType() != kRecv)) {
      return false;
    }
  }
  return true;
}

static inline bool CheckConstanOpForCheckpointGraph(NodePtr &node) { return node->GetOutDataNodes().empty(); }

bool GraphManager::IsCheckpointGraph(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[IsCheckpointGraph] computeGraph is nullptr.");
    return false;
  }
  for (auto &node : compute_graph->GetAllNodes()) {
    OpDescPtr op = node->GetOpDesc();
    GE_RT_FALSE_CHECK_NOTNULL(op);
    if (op->GetType() == kNetOutput) {
      if (!CheckNetOutputForCheckpointGraph(node)) {
        return false;
      }
    } else if (op->GetType() == kVariable) {
      if (!CheckVariableForCheckpointGraph(node)) {
        return false;
      }
    } else if ((TransOpUtil::IsTransOp(node))) {
      if (!CheckTransOpForCheckpointGraph(node)) {
        return false;
      }
    } else if (op->GetType() == CONSTANTOP) {
      if (!CheckConstanOpForCheckpointGraph(node)) {
        return false;
      }
    } else if (op->GetType() != kSend && op->GetType() != kRecv) {
      GELOGI("this node is not allow in checkpoint sub graph, node_type: %s, node_name: %s.", op->GetType().c_str(),
             op->GetName().c_str());
      return false;
    }
  }
  GELOGI("current graph %s is checkpoint sub graph.", compute_graph->GetName().c_str());
  return true;
}

bool GraphManager::IsBroadCastOpData(const ge::NodePtr &var_node) {
  for (auto &out_anchor : var_node->GetAllOutDataAnchors()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_anchor);
    for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
      ge::NodePtr dst_node = in_anchor->GetOwnerNode();
      GE_RT_FALSE_CHECK_NOTNULL(dst_node);
      if (dst_node->GetType() == HCOMBROADCAST) {
        return true;
      }
    }
  }
  return false;
}

void GraphManager::AdjustBroadCastOpData(const ge::NodePtr &var_node) {
  if (!ge::AttrUtils::SetStr(var_node->GetOpDesc(), VAR_ATTR_VAR_IS_BROADCAST, "var_is_restore")) {
    GELOGW("set var_is_restore failed");
  }
}

bool GraphManager::IsAssignOpData(const ge::NodePtr &var_node) {
  GELOGD("IsAssignOpData var_node %s", var_node->GetName().c_str());
  std::map<std::string, std::set<int>> assign_ops = {{ASSIGN, {0}}};

  ge::NodePtr assign_node = nullptr;
  if (ConfirmUseOpAndIndexByNode(var_node, assign_ops, assign_node)) {
    return true;
  }

  return false;
}

void GraphManager::AdjustAssignOpData(const ge::NodePtr &var_node) {
  if (!ge::AttrUtils::SetStr(var_node->GetOpDesc(), VAR_ATTR_VAR_IS_RESTORE, "var_is_restore")) {
    GELOGW("SetStr var_is_restore failed");
  }
}

bool GraphManager::ConfirmUseOpAndIndexByAnchor(const ge::InDataAnchorPtr &in_anchor,
                                                const map<string, std::set<int>> &confirm_ops, ge::NodePtr &use_node) {
  GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
  ge::NodePtr dst_node = in_anchor->GetOwnerNode();
  GE_RT_FALSE_CHECK_NOTNULL(dst_node);
  ge::OpDescPtr dst_op_desc = dst_node->GetOpDesc();
  GE_RT_FALSE_CHECK_NOTNULL(dst_op_desc);
  const string &dst_type = dst_op_desc->GetType();
  int input_index = in_anchor->GetIdx();

  GELOGD("ConfirmUseOpAndIndex, var name %s, dst_type = %s, input index %d", dst_node->GetName().c_str(),
         dst_type.c_str(), input_index);

  if (confirm_ops.count(dst_type) > 0) {
    if (confirm_ops.at(dst_type).count(input_index) > 0) {
      use_node = dst_node;
      return true;
    }
  }
  return false;
}

bool GraphManager::ConfirmUseOpAndIndexByNode(const ge::NodePtr &var_node,
                                              const map<string, std::set<int>> &confirm_ops, ge::NodePtr &use_node) {
  GE_RT_FALSE_CHECK_NOTNULL(var_node);
  for (auto &out_anchor : var_node->GetAllOutDataAnchors()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_anchor);
    for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
      if (ConfirmUseOpAndIndexByAnchor(in_anchor, confirm_ops, use_node)) {
        return true;
      }
    }
  }
  return false;
}

Status GraphManager::RemoveIsolatedConst(ge::ComputeGraphPtr &compute_graph) {
  for (ge::NodePtr &n : compute_graph->GetAllNodes()) {
    if (n->GetOpDesc() == nullptr) {
      continue;
    }
    if (n->GetOpDesc()->GetType() == CONSTANT || n->GetOpDesc()->GetType() == CONSTANTOP) {
      // reset const type depend on train_flag
      options_.train_graph_flag ? n->GetOpDesc()->SetType(CONSTANTOP) : n->GetOpDesc()->SetType(CONSTANT);
      if (n->GetOutAllNodes().empty() && n->GetInAllNodes().empty()) {
        // it is an isolated constant, just remove it
        if (GraphUtils::RemoveJustNode(compute_graph, n) != GRAPH_SUCCESS) {
          GELOGE(FAILED, "remove constant %s failed.", n->GetName().c_str());
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

Status GraphManager::OptimizeAfterMergeSubGraph(ge::ComputeGraphPtr &compute_graph) {
  GELOGI("Start optimize after merge sub graph.");

  GEPass ge_passes_for_shape(compute_graph);
  NamesToPass names_to_passes_for_shape;
  IdentifyReferencePass identify_reference_pass;
  names_to_passes_for_shape.emplace_back("IdentifyReferencePass", &identify_reference_pass);
  CastRemovePass cast_remove_pass;
  names_to_passes_for_shape.emplace_back("CastRemovePass", &cast_remove_pass);
  TransposeTransDataPass transpose_transdata_pass;
  names_to_passes_for_shape.emplace_back("TransposeTransDataPass", &transpose_transdata_pass);
  GE_TIMESTAMP_START(ge_passes_for_shape);
  Status ret = ge_passes_for_shape.Run(names_to_passes_for_shape);
  GE_TIMESTAMP_END(ge_passes_for_shape, "GraphManager::GePassesForShape");
  if (ret != SUCCESS) {
    GELOGE(ret, "Run ge_passes_for_shape optimize for OptimizeAfterMergeSubGraph failed, ret:%d.", ret);
    return ret;
  }

  string options = "default";
  if (GetContext().GetOption("ge.exec.variable_acc", options) != SUCCESS) {
    GELOGI("get ge.exec.variable_acc failed. set default value.");
  }
  PassManager after_merge_passes;
  GE_CHK_STATUS_RET(after_merge_passes.AddPass(new (std::nothrow) PermutePass))
  GE_CHK_STATUS_RET(after_merge_passes.AddPass(new (std::nothrow) VariablePrepareOpPass))
  GE_IF_BOOL_EXEC(options == "default" || options == "1", GELOGI("turn on variable accelerator");
                  GE_CHK_STATUS_RET(after_merge_passes.AddPass(new (std::nothrow) VariableOpPass(&var_acc_ctrl_))))
  GE_CHK_STATUS_RET(after_merge_passes.AddPass(new (std::nothrow) TransOpDepthFusionPass))
  GE_CHK_STATUS_RET(after_merge_passes.AddPass(new (std::nothrow) TransOpBreadthFusionPass))
  GE_CHK_STATUS_RET(after_merge_passes.AddPass(new (std::nothrow) VariableRefDeleteOpPass))
  GE_CHK_STATUS_RET(after_merge_passes.AddPass(new (std::nothrow) SameTransdataBreadthFusionPass))
  GE_CHK_STATUS_RET(after_merge_passes.AddPass(new (std::nothrow) TransOpWithoutReshapeFusionPass))
  GE_CHK_STATUS_RET(after_merge_passes.AddPass(new (std::nothrow) AtomicAddrCleanPass))
  GE_CHK_STATUS_RET(
    after_merge_passes.AddPass(new (std::nothrow) LinkGenMaskNodesPass(options_.stream_max_parallel_num)))

  GE_TIMESTAMP_START(after_merge_passes);
  ret = after_merge_passes.Run(compute_graph);
  GE_TIMESTAMP_END(after_merge_passes, "GraphManager::AfterMergePasses");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "Run passes after merge sub graph failed, ret:%d.", ret);
    return ret;
  }

  // add variable attr for hccl broadcast,need to be removed after variable pass online
  for (const ge::NodePtr &node : compute_graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() != VARIABLE) {
      continue;
    }

    if (IsBroadCastOpData(node)) {
      AdjustBroadCastOpData(node);
    }
    if (IsAssignOpData(node)) {
      AdjustAssignOpData(node);
    }
  }

  GEPass ge_passes(compute_graph);
  NamesToPass names_to_passes;
  TransOpNearbyAllreduceFusionPass trans_op_nearby_allreduce_fusion_pass;
  names_to_passes.emplace_back("ReshapeRemovePass", &trans_op_nearby_allreduce_fusion_pass);
  ReshapeRemovePass reshape_remove_pass;
  names_to_passes.emplace_back("ReshapeRemovePass", &reshape_remove_pass);
  ConstantFoldingPass constant_folding_pass;
  names_to_passes.emplace_back("ConstantFoldingPass", &constant_folding_pass);
  DimensionAdjustPass dimension_adjust_pass;
  names_to_passes.emplace_back("DimensionAdjustPass", &dimension_adjust_pass);
  GE_TIMESTAMP_START(names_to_passes);
  ret = ge_passes.Run(names_to_passes);
  GE_TIMESTAMP_END(names_to_passes, "GraphManager::MergedGraphNameToPasses");
  if (ret != SUCCESS) {
    GELOGE(ret, "Run ge_passes optimize for OptimizeAfterMergeSubGraph failed, ret:%d.", ret);
    return ret;
  }

  ret = RemoveIsolatedConst(compute_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Remove isolated Constant failed, ret:%d.", ret);
    return ret;
  }

  PassManager pass_for_control_attr_optimize;
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass(new (std::nothrow) MultiBatchPass))
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass(new (std::nothrow) ControlOpAttrPass))
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass(new (std::nothrow) CompileNodesPass))

  GE_TIMESTAMP_START(pass_for_control_attr_optimize);
  ret = pass_for_control_attr_optimize.Run(compute_graph);
  GE_TIMESTAMP_END(pass_for_control_attr_optimize, "GraphManager::ControlAttrOptimize");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "Run ControlOpAttrPass failed");
    return ret;
  }

  ret = compute_graph->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(ret, "Graph topological sort failed, ret:%d.", ret);
    return ret;
  }

  GELOGI("End optimize after merge sub graph.");
  return SUCCESS;
}

Status GraphManager::LoadGraphAsync(const GeModelPtr &ge_model, const GraphNodePtr &graph_node) {
  GELOGI("[LoadGraphAsync] run_graph_flag[%d], graph_id[%u]", options_.run_graph_flag, graph_node->GetGraphId());
  if (options_.run_graph_flag && ge_model != nullptr) {
    // synchronization run graph with model
    ModelIdInfo model_id_info;
    if (getenv(kEnvGeuseStaticMemory) != nullptr) {
      GELOGI("[LoadGraphAsync] GE_USE_STATIC_MEMORY is seted.");
    } else {
      GE_CHK_STATUS_RET(CheckAndReleaseMemory(ge_model, graph_node))
    }
    GE_TIMESTAMP_START(LoadGraph);
    GE_CHECK_NOTNULL(graph_node->graph_run_async_listener_);
    Status ret = graph_loader_.LoadGraphAsync(ge_model, graph_node->graph_run_async_listener_, model_id_info);
    GE_TIMESTAMP_END(LoadGraph, "GraphManager::LoadGraphAsync");
    if (ret != SUCCESS) {
      GELOGE(ret, "[LoadGraphAsync] LoadGraphAsync Failed");
      graph_node->SetRunFlag(false);
      return ret;
    }
    ge_model->SetModelId(model_id_info.model_id);
    graph_node->SetGeModel(ge_model);
  }
  return SUCCESS;
}

Status GraphManager::CheckAndReleaseMemory(const GeModelPtr &ge_model, const GraphNodePtr &graph_node) {
  GELOGI("CheckAndReleaseMemory graph_id[%u]", graph_node->GetGraphId());
  int64_t value = 0;
  bool ret = ge::AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, value);
  int64_t memory_size = ret ? value : 0;
  ret = ge::AttrUtils::GetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, value);
  int64_t weight_size = ret ? value : 0;
  ret = ge::AttrUtils::GetInt(ge_model, MODEL_ATTR_SESSION_ID, value);
  uint64_t session_id = ret ? value : 0;

  int64_t free_memory = 0;
  Status result = GraphLoader::GetMemoryInfo(free_memory);
  if (result != SUCCESS) {
    return result;
  }

  GELOGI(
    "CheckAndReleaseMemory Graph[%u] need memory_size[%ld], weight_size[%ld],"
    " Device[%u] free_memory_size[%ld]",
    graph_node->GetGraphId(), memory_size, weight_size, GetContext().DeviceId(), free_memory);
  if (ge::CheckInt64AddOverflow(memory_size, weight_size) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "The sum of Memory size and weight size exceeds INT64_MAX");
    return INTERNAL_ERROR;
  }
  if (free_memory >= (memory_size + weight_size)) {
    return SUCCESS;
  }
  rtError_t rt_ret;
  for (auto &it : graph_map_) {
    auto graph_id = it.second->GetGraphId();
    auto model = it.second->GetGeModel();
    if (model == nullptr) {
      continue;
    }
    auto model_id = model->GetModelId();
    // not loaded,no need unload
    if (!it.second->GetLoadFlag()) {
      GELOGI("CheckAndReleaseMemory graph[%u] has not been loaded.", graph_id);
      continue;
    }
    uint64_t max_memory_size = 0;
    result = GraphLoader::GetMaxUsedMemory(model_id, max_memory_size);
    if (result != SUCCESS) {
      continue;
    }
    GELOGI("CheckAndReleaseMemory try to UnloadGraph[%u], model[%u] which MaxUsedMemory[%lu].", graph_id, model_id,
           max_memory_size);
    rt_ret = rtSetDevice(GetContext().DeviceId());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[GraphManager:] rtSetDevice failed, modelId=%u, graphId=%u.", model_id, graph_id);
      continue;
    }
    result = GraphLoader::UnloadModel(model_id);
    if (result != SUCCESS) {
      GELOGW("[GraphManager:] unload model failed, modelId=%u, graphId=%u.", model_id, graph_id);
    }
    result = GraphLoader::DestroyAicpuKernel(session_id, model_id);
    if (result != SUCCESS) {
      GELOGW("[GraphManager:] destroy aicpu kernel failed when dynamic memory, modelId=%u, graphId=%u.", model_id,
             graph_id);
    }
    rt_ret = rtDeviceReset(GetContext().DeviceId());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[GraphManager:] rtDeviceReset failed, modelId=%u, graphId=%u.", model_id, graph_id);
      continue;
    }
    it.second->SetLoadFlag(false);
    GELOGI("CheckAndReleaseMemory UnloadGraph[%u], model[%u] success and set LoadFlag to false.", graph_id, model_id);
  }
  return SUCCESS;
}

Status GraphManager::ProcessSubGraphWithMultiThreads(GraphManager *graph_manager,
                                                     const SubGraphInfoPtr &sub_graph_info_ptr, uint64_t session_id,
                                                     const GEThreadLocalContext &ge_context) {
  Status ret = SUCCESS;
  GetThreadLocalContext() = ge_context;
  if (sub_graph_info_ptr != nullptr && graph_manager != nullptr) {
    ComputeGraphPtr compute_graph_tmp = sub_graph_info_ptr->GetSubGraph();
    const std::string &engine_name = sub_graph_info_ptr->GetEngineName();
    GELOGI("ProcessSubGraphWithMultiThreads start, graph name is %s, engine_name is %s, thread id is %lu",
           compute_graph_tmp != nullptr ? compute_graph_tmp->GetName().c_str() : "", engine_name.c_str(),
           pthread_self());
    GraphUtils::DumpGEGraph(compute_graph_tmp, "OptimizeSubGraphBefore");
    GraphUtils::DumpGEGraphToOnnx(*compute_graph_tmp, "OptimizeSubGraphBefore");
    GE_CHECK_NOTNULL(compute_graph_tmp);
    compute_graph_tmp->SetSessionID(session_id);
    ret = graph_manager->graph_optimize_.OptimizeSubGraph(compute_graph_tmp, engine_name);
    if (ret != SUCCESS) {
      GELOGE(ret, "SubGraph optimize Failed %s", engine_name.c_str());
      return ret;
    } else {
      GELOGI("SubGraph optimize success %s", engine_name.c_str());
    }
    GraphUtils::DumpGEGraph(compute_graph_tmp, "OptimizeSubGraphAfter");
    GraphUtils::DumpGEGraphToOnnx(*compute_graph_tmp, "OptimizeSubGraphAfter");
    sub_graph_info_ptr->SetSubGraph(compute_graph_tmp);
    GELOGI("ProcessSubGraphWithMultiThreads end, graph name is %s, engine_name is %s, thread id is %lu",
           compute_graph_tmp != nullptr ? compute_graph_tmp->GetName().c_str() : "", engine_name.c_str(),
           pthread_self());
  } else {
    GELOGE(ret, "graph_manager or sub_graph_info_ptr is nullptr");
    return FAILED;
  }
  return SUCCESS;
}

// run graph async on session
Status GraphManager::RunGraphAsync(const GraphId &graph_id, const std::vector<ge::TensorInfo> &inputs,
                                   std::vector<ge::TensorInfo> &outputs, uint64_t session_id,
                                   std::function<void(Status)> callback) {
  GELOGI("[GraphManager] Start to run graph async, graph_id=%u, inputsSize=%zu, outputsSize=%zu.", graph_id,
         inputs.size(), outputs.size());

  bool ret =
    prerun_args_q_.Push(PreRunArgs({graph_id, inputs, outputs, session_id, GetThreadLocalContext(), callback}));
  if (!ret) {
    GELOGE(FAILED, "[GraphManager] Run graph async failed, graph_id=%u.", graph_id);
    return FAILED;
  }

  GELOGI("[GraphManager] Run graph async success, graph_id=%u.", graph_id);
  return SUCCESS;
}

void GraphManager::AddModelCacheHelperToMap(const GraphId &graph_id, uint64_t session_id,
                                            ComputeGraphPtr &compute_graph) {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr != nullptr && instance_ptr->IsIncreBuild()) {
    auto iter = cache_helper_map_.find(graph_id);
    if (iter == cache_helper_map_.end()) {
      ModelCacheHelperPtr cache_helper = MakeShared<ge::ModelCacheHelper>(session_id, graph_id, compute_graph);
      if (cache_helper != nullptr) {
        cache_helper_map_.emplace(std::make_pair(graph_id, cache_helper));
      } else {
        GELOGW("Cache helper make shared failed, graph_id = %u.", graph_id);
      }
    }
  }
}

Status GraphManager::IncreBuild(const GraphNodePtr &graph_node, GeModelPtr &ge_model) {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->IsIncreBuild()) {
    return FAILED;
  }
  const uint32_t graph_id = graph_node->GetGraphId();
  auto iter = cache_helper_map_.find(graph_id);
  if (iter == cache_helper_map_.end()) {
    GELOGW("Can not find ModelCacheHelper of graph[%u]", graph_id);
    return FAILED;
  }
  ModelCacheHelperPtr cache_helper = iter->second;
  if (cache_helper->IsModelCacheHit()) {
    GEEVENT("Model cache hit.");
    Status ret = LoadFromCache(graph_node, cache_helper, ge_model);
    if (ret == SUCCESS) {
      return SUCCESS;
    } else {
      GELOGW("Error occurred when load from cache, abandon.");
    }
  } else {
    GEEVENT("Model cache miss.");
  }
  if (SaveCacheBeforeBuild(graph_node->GetGraphId(), cache_helper) != SUCCESS) {
    GELOGW("Error occurred when save cache.");
  }
  return FAILED;
}

void GraphManager::PreRunThread(GraphManager *graph_manager) {
  if (prctl(PR_SET_NAME, ("GE_PreRun")) != 0) {
    GELOGW("Set thread name failed.");
  }
  PreRunArgs args;
  while (graph_manager->thread_run_flag_) {
    bool pop_status = graph_manager->prerun_args_q_.Pop(args);
    if (!pop_status) {
      continue;
    }
    GetThreadLocalContext() = args.context;
    GELOGI("A new loop start.");
    std::vector<ge::GeTensor> ge_inputs;
    for (auto const &input : args.input_tensor) {
      std::vector<int64_t> input_dims;
      std::transform(input.shapeInfo.dims.begin(), input.shapeInfo.dims.end(), std::back_inserter(input_dims),
                     [](uint32_t x) -> int64_t { return static_cast<int64_t>(x); });
      GeShape input_shape(input_dims);
      GeTensorDesc input_tensor_desc;
      input_tensor_desc.SetShape(input_shape);
      input_tensor_desc.SetDataType(static_cast<ge::DataType>(input.dataType));
      ge_inputs.emplace_back(input_tensor_desc);
    }
    // find graph
    GraphNodePtr graph_node = nullptr;
    Status ret = graph_manager->GetGraphNode(args.graph_id, graph_node);
    if (ret != SUCCESS) {
      ReturnError(graph_manager, args.callback, GE_GRAPH_ALREADY_RUNNING,
                  "[RunGraph] graph not exist, graph_id=" + std::to_string(args.graph_id));
      return;
    }

    graph_node->Lock();

    if (graph_node->GetRunFlag()) {
      ReturnError(graph_manager, args.callback, GE_GRAPH_GRAPH_NODE_NULL,
                  "[RunGraph] graph already running, graph id=" + std::to_string(args.graph_id));
      graph_node->Unlock();
      return;
    }
    // set graph's run flag
    graph_node->SetRunFlag(true);

    ComputeGraphPtr compute_graph_tmp = GraphUtils::GetComputeGraph(*(graph_node->GetGraph()));

    if (graph_manager->GetTrainFlag()) {
      if (compute_graph_tmp == nullptr) {
        ReturnError(graph_manager, args.callback, GE_GRAPH_GRAPH_NODE_NULL,
                    "[RunGraph] compute_graph_tmp is NULL, graph id = %u.");
        graph_node->Unlock();
        return;
      }
    }
    // when set incre build, save cache helper.
    graph_manager->AddModelCacheHelperToMap(args.graph_id, args.session_id, compute_graph_tmp);

    std::vector<GeModelPtr> ge_models;

    if (graph_manager->options_.local_fmk_op_flag) {
      graph_manager->graph_optimize_.TranFrameOp(compute_graph_tmp);
    }

    // it will not execute graph preprocess, optimize, parition, build if the graph has built successful.

    GELOGI("Start for run graph async.");

    GeModelPtr ge_model = nullptr;
    if (graph_manager->IsGraphNeedBuild(graph_node)) {
      if (graph_node->GetBuildFlag()) {
        ReturnError(graph_manager, args.callback, PARAM_INVALID,
                    "The graph " + std::to_string(graph_node->GetGraphId()) +
                      " need to re-build, you should remove it"
                      " from GE first, then AddGraph again and rebuild it.");
        graph_node->Unlock();
        return;
      }

      // check need incre build.
      if (graph_manager->IncreBuild(graph_node, ge_model) != SUCCESS) {
        ret = graph_manager->PreRun(graph_node, ge_inputs, ge_models, ge_model, args.session_id);
        if (ret != SUCCESS) {
          graph_node->SetRunFlag(false);
          ReturnError(graph_manager, args.callback, ret, "PreRun Failed, thread exit..");
          graph_node->Unlock();
          return;
        }
      }
      graph_node->SetBuildFlag(true);
      graph_manager->var_acc_ctrl_.SetGraphBuildEnd(graph_node->GetGraphId());
    } else {
      ge_model = graph_node->GetGeModel();
    }

    graph_manager->run_args_q_.Push(RunArgs({graph_node, args.graph_id, args.input_tensor, args.output_tensor, ge_model,
                                             GetThreadLocalContext(), args.callback}));
    GELOGI("Loop end.");
  }
}

void GraphManager::RunThread(GraphManager *graph_manager) {
  if (prctl(PR_SET_NAME, ("GE_Run")) != 0) {
    GELOGW("Set thread name failed.");
  }
  RunArgs args;
  while (graph_manager->thread_run_flag_) {
    bool pop_status = graph_manager->run_args_q_.Pop(args);
    if (!pop_status) {
      continue;
    }
    GELOGI("A new loop start.");
    GetThreadLocalContext() = args.context;
    if (args.graph_node->graph_run_async_listener_ != nullptr) {
      args.graph_node->graph_run_async_listener_->SetCallback(args.callback);
    }

    Status ret;
    if (!args.graph_node->GetLoadFlag()) {
      ret = graph_manager->LoadGraphAsync(args.ge_model, args.graph_node);
      if (ret != SUCCESS) {
        StopQueue(graph_manager);
        ReturnError(graph_manager, args.callback, ret, "LoadGraphAsync failed, thread exit.");
        args.graph_node->Unlock();
        return;
      }
      args.graph_node->SetLoadFlag(true);
      GELOGI("LoadGraph[%u], model[%u] success and set LoadFlag to true.", args.graph_node->GetGraphId(),
             args.ge_model->GetModelId());
    }

    if (graph_manager->GetTrainFlag()) {
      ret = graph_manager->graph_executor_.SetGraphContext(graph_manager->GetGraphContext());
      if (ret != SUCCESS) {
        GELOGW("[GraphManager] SetGraphContext failed, graph_id=%u.", args.graph_id);
      }
      graph_manager->graph_executor_.SetTrainFlag(graph_manager->options_.train_graph_flag);
    }

    ret = graph_manager->graph_executor_.ExecuteGraphAsync(args.graph_id, args.graph_node->GetGeModel(),
                                                           args.input_tensor, args.output_tensor);
    args.graph_node->SetRunFlag(false);
    args.graph_node->Unlock();
    if (ret != SUCCESS) {
      GELOGE(ret, "[GraphManager] Run graph async failed, graph_id=%u.", args.graph_id);
      StopQueue(graph_manager);
      return;
    }
    GELOGI("[GraphManager] Run graph async success, graph_id=%u.", args.graph_id);
  }
}

void GraphManager::StopQueue(GraphManager *graph_manager) {
  if (graph_manager == nullptr) {
    return;
  }

  graph_manager->thread_run_flag_.store(false);
  graph_manager->prerun_args_q_.Stop();
  graph_manager->run_args_q_.Stop();
}

void GraphManager::ReturnError(GraphManager *graph_manager, std::function<void(Status)> callback, Status ret,
                               const string &log) {
  if (graph_manager == nullptr) {
    return;
  }

  GELOGE(ret, "%s.", log.c_str());
  StopQueue(graph_manager);
  callback(ret);
}

bool GraphManager::IsGraphNeedRebuild(uint32_t graph_id) {
  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[RunGraph] graph not exist, graph_id=%u.", graph_id);
    return true;
  }

  if (graph_node == nullptr) {
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[RunGraph] graph node is NULL, graphId=%u.", graph_id);
    return true;
  }

  return IsGraphNeedBuild(graph_node);
}

bool GraphManager::IsGraphNeedBuild(const GraphNodePtr &graph_node) {
  return !graph_node->GetBuildFlag() || var_acc_ctrl_.IsGraphNeedRebuild(graph_node->GetGraphId());
}
const map<std::string, std::string> *GraphManager::GetGraphOptions(uint32_t graph_id) {
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[RunGraph] graph not exist, graph_id=%u.", graph_id);
    return nullptr;
  }

  if (!graph_node) {
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[RunGraph] graph node is NULL, graph_id=%u.", graph_id);
    return nullptr;
  }
  return &(graph_node->GetOptions());
}
}  // namespace ge
