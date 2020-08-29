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

#ifndef GE_GRAPH_MANAGER_GRAPH_MANAGER_H_
#define GE_GRAPH_MANAGER_GRAPH_MANAGER_H_

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/blocking_queue.h"
#include "common/ge_inner_error_codes.h"
#include "common/helper/model_cache_helper.h"
#include "external/graph/types.h"
#include "ge/ge_api_types.h"
#include "graph/build/graph_builder.h"
#include "graph/execute/graph_execute.h"
#include "graph/ge_local_context.h"
#include "graph/load/graph_loader.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/manager/util/variable_accelerate_ctrl.h"
#include "graph/optimize/graph_optimize.h"
#include "graph/partition/graph_partition.h"
#include "graph/preprocess/graph_preprocess.h"
#include "model/ge_model.h"

namespace ge {
class GraphManager {
 public:
  GraphManager();

  ~GraphManager() = default;

  ///
  /// @ingroup ge_graph
  /// @brief graph manager init
  /// @param [in] options user config params
  /// @return Status result of function
  ///
  Status Initialize(const std::map<string, string> &options);

  ///
  /// @ingroup ge_graph
  /// @brief graph manager finalize
  /// @return Status result of function
  ///
  Status Finalize();

  ///
  /// @ingroup ge_graph
  /// @brief add specific graph
  /// @param [in] graph_id graph id
  /// @param [out] Graph output graph
  /// @return Status result of function
  ///
  Status AddGraph(const GraphId &graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  ///
  /// @ingroup ge_graph
  /// @brief remove specific graph
  /// @param [in] graph_id graph id
  /// @return Status result of function
  ///
  Status RemoveGraph(const GraphId &graph_id);

  ///
  /// @ingroup ge_graph
  /// @brief run specific graph
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @param [out] outputs output data
  /// @return Status result of function
  ///
  Status RunGraph(const GraphId &graph_id, const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs,
                  uint64_t session_id = INVALID_SESSION_ID);

  ///
  /// @ingroup ge_graph
  /// @brief build specific graph
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @param [out] models build result
  /// @return Status result of function
  ///
  ge::Status BuildGraph(const GraphId &graph_id, const std::vector<GeTensor> &inputs, GeRootModelPtr &models,
                        uint64_t session_id = 0, bool async = false);

  Status BuildGraphForUnregisteredOp(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                                     GeRootModelPtr &ge_root_model, uint64_t session_id);

  ///
  /// @ingroup ge_graph
  /// @brief Save extra attribute to Model
  /// @param [in] model: Model attribues will save to.
  /// @param [in] type: type of OpDesc.
  /// @param [in] attrs: attributes of OpDesc
  /// @param [in] inputs: input tensor
  /// @param [in] outputs: output tensor
  /// @return: Status
  ///
  Status SaveParams(ge::GeModel &model, const std::string &type, const std::map<string, GeAttrValue> &attrs,
                    const std::vector<GeTensor> &inputs, const std::vector<GeTensor> &outputs);

  ///
  /// @ingroup ge_graph
  /// @brief get variable value from the session with specific session id
  /// @param [in] sessionId session id
  /// @param [in] name op name
  /// @param [out] val out value tensor
  /// @return Status result of function
  ///
  Status GetVariable(const std::string &name, Tensor &val);

  ///
  /// @ingroup ge_graph
  /// @brief run graph async on session with specific session id
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @param [out] callback: callback while run graph async finish
  /// @return Status result of function
  ///
  Status RunGraphAsync(const GraphId &graph_id, const std::vector<ge::InputTensorInfo> &inputs, uint64_t session_id,
                       RunAsyncCallback callback);

  ///
  /// @ingroup ge_graph
  /// @brief me register the callback function to get the result of summary or checkpoin
  /// @param [in] key: summary or checkpoint
  /// @param [in] callbak: The real callback object of me
  /// @return Status result of function
  ///
  Status RegisterCallBackFunc(
    const std::string &key, const std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)> &callback);

  const bool GetTrainFlag() const { return options_.train_graph_flag; }

  bool IsGraphNeedRebuild(uint32_t graph_id);

  Status GenerateInfershapeGraph(GraphId &graph_id);

  const std::map<std::string, std::string> *GetGraphOptions(uint32_t graph_id);

  void SetOptionsRunGraphFlag(bool run_graph_flag);

  Status GenCheckPointGraph(const std::map<std::string, GeTensorDesc> &all_variables, Graph &graph);

  Status SaveVariables(const Graph &graph, const std::vector<std::string> &var_names,
                       const std::vector<Tensor> &outputs, std::vector<Tensor> &var_values);

  Status SaveCheckPointResult(const Graph &graph, const std::vector<Tensor> &outputs, map<string, Tensor> &var_results);

 private:
  struct PreRunArgs {
    GraphId graph_id;
    std::vector<ge::InputTensorInfo> input_tensor;
    uint64_t session_id;
    GEThreadLocalContext context;
    RunAsyncCallback callback;
  };

  struct RunArgs {
    GraphNodePtr graph_node;
    GraphId graph_id;
    std::vector<ge::InputTensorInfo> input_tensor;
    GeRootModelPtr ge_root_model;
    GEThreadLocalContext context;
    RunAsyncCallback callback;
  };

  Status GetGraphNode(const GraphId &graph_id, GraphNodePtr &out);

  std::shared_ptr<GraphModelListener> GetModelListener() const { return graph_run_listener_; }

  static Status ProcessSubGraphWithMultiThreads(GraphManager *graph_manager, const SubGraphInfoPtr &sub_graph_info_ptr,
                                                uint64_t session_id, const GEThreadLocalContext &ge_context);
  Status PreRun(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs, GeRootModelPtr &ge_root_model,
                uint64_t session_id = INVALID_SESSION_ID);

  Status OptimizeSubgraph(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph, uint64_t session_id);

  Status Build(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph, GeRootModelPtr &ge_root_model,
               uint64_t session_id);

  Status StartForRunGraph(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs,
                          GeRootModelPtr &ge_root_model, uint64_t session_id = INVALID_SESSION_ID);

  Status InnerRunGraph(GraphNodePtr &graph_node, const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                       std::vector<GeTensor> &outputs);

  Status ParseOptions(const std::map<std::string, std::string> &options);

  static void ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                          std::string &option);

  static Status ParseOption(const std::map<std::string, std::string> &options, const std::string &key, bool &option);

  static Status ParseOption(const std::map<std::string, std::string> &options, const std::string &key, int &option);

  static Status ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                            std::map<std::string, int> &option);

  static void Trim(std::string &str);

  static Status CheckEngineName(const std::string &engine_name, const std::string &key,
                                const std::map<std::string, int> &option);

  static Status ParseParallelNum(const std::string &parallel_num, const std::string &key, int &num);

  static Status ParseTrainGraphFlag(bool &options, bool &option);

  static bool IsPerfLevelInvalid(int32_t perf_level);

  Status SummaryHandle(const GraphId &graph_id, std::vector<GeTensor> &outputs);

  Status CheckpointHandle(const GraphId &graph_id, const ComputeGraphPtr &compute_graph,
                          const std::vector<GeTensor> &outputs);

  // call the callback function of ME to push summary result data to ME
  Status PushSummaryData2ME(const GraphId &graph_id, const std::map<std::string, ge::Tensor> &summary_data);

  // call the callback function of ME to push save result data to ME
  Status PushSaveData2ME(const GraphId &graph_id, const std::map<std::string, ge::Tensor> &save_data);

  bool IsCheckpointGraph(ComputeGraphPtr &compute_graph);

  bool CheckNetOutputForCheckpointGraph(NodePtr &node);

  bool CheckVariableForCheckpointGraph(NodePtr &node);

  bool CheckTransOpForCheckpointGraph(NodePtr &node);

  Status MergeSubGraph(ComputeGraphPtr &compute_graph, const ge::ComputeGraphPtr &original_compute_graph);

  Status SetSubgraph(uint64_t session_id, ComputeGraphPtr compute_graph);

  void SetAttrForHcomBroadCastOp(ge::ComputeGraphPtr &compute_graph);

  bool IsBroadCastOpData(const ge::NodePtr &var_node);

  void AdjustBroadCastOpData(const ge::NodePtr &var_node);

  bool IsAssignOpData(const ge::NodePtr &var_node);

  void AdjustAssignOpData(const ge::NodePtr &var_node);

  bool ConfirmUseOpAndIndexByAnchor(const ge::InDataAnchorPtr &in_anchor, const map<string, std::set<int>> &confirm_ops,
                                    ge::NodePtr &use_node);

  bool ConfirmUseOpAndIndexByNode(const ge::NodePtr &var_node, const map<string, std::set<int>> &confirm_ops,
                                  ge::NodePtr &use_node);

  // graph context
  std::shared_ptr<GraphContext> GetGraphContext() const { return graph_context_; }

  Status RemoveIsolatedConst(ge::ComputeGraphPtr &compute_graph);
  Status RemoveIsolatedConstInThisGraph(ge::ComputeGraphPtr &compute_graph);

  Status OptimizeStage1(ComputeGraphPtr &compute_graph);
  Status OptimizeStage2(ComputeGraphPtr &compute_graph);

  Status SubexpressionMigration(ComputeGraphPtr &compute_graph);

  Status LoadGraphAsync(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node);

  Status CheckAndReleaseMemory(const GeModelPtr &ge_model, const GraphNodePtr &graph_node);

  bool CheckModelLoad(const GeRootModelPtr &ge_model, bool load_flag);

  Status LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node);

  bool IsGraphNeedBuild(const GraphNodePtr &graph_node);

  Status LoadFromCache(const GraphNodePtr &graph_node, const ModelCacheHelperPtr &cache_helper, GeModelPtr &ge_model);
  Status SaveCacheBeforeBuild(uint32_t graph_id, const ModelCacheHelperPtr &cache_helper);
  Status SaveCacheAfterBuild(uint32_t graph_id, ComputeGraphPtr graph, GeModelPtr &ge_model);
  void AddModelCacheHelperToMap(const GraphId &graph_id, uint64_t session_id, ComputeGraphPtr &compute_graph);
  Status IncreBuild(const GraphNodePtr &graph_node, GeModelPtr &ge_model);
  void RemoveModelCacheHelper(const GraphId &graph_id);

  static void ConstructGeInput(std::vector<ge::GeTensor> &ge_inputs, PreRunArgs &args);
  static void PreRunThread(GraphManager *graph_manager);
  static void RunThread(GraphManager *graph_manager);
  static void StopQueue(GraphManager *graph_manager);
  static void ReturnError(GraphManager *graph_manager, RunAsyncCallback callback, Status ret, const string &log);
  static void ReturnError(GraphManager *graph_manager, GraphNodePtr &graph_node, RunAsyncCallback callback, Status ret,
                          const string &log);

  void ChangeConstTypeWhenTraining(const ComputeGraphPtr &compute_graph);

  std::atomic_bool thread_run_flag_;
  BlockingQueue<PreRunArgs> prerun_args_q_{};
  BlockingQueue<RunArgs> run_args_q_{};
  std::thread prerun_thread_;
  std::thread run_thread_;

  std::map<GraphId, GraphNodePtr> graph_map_;

  std::map<GraphId, ModelCacheHelperPtr> cache_helper_map_;

  // for run graph synchronous return
  std::mutex sync_run_mutex_;
  std::condition_variable condition_;
  // run graph synchronization call back listener
  std::shared_ptr<GraphModelListener> graph_run_listener_;

  // summary and checkpoint callback function list for ME, key is summary or checkpoint
  std::map<std::string, std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)>> me_callback_map_;

  bool init_flag_;

  GraphManagerOptions options_;

  GraphPrepare graph_preparer_;
  GraphOptimize graph_optimize_;
  GraphPartitioner graph_partitioner_;
  GraphBuilder graph_builder_;
  GraphLoader graph_loader_;
  GraphExecutor graph_executor_;
  GraphContextPtr graph_context_ = nullptr;

  VarAccelerateCtrl var_acc_ctrl_;

  std::mutex run_mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MANAGER_H_
