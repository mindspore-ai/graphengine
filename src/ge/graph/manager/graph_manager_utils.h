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

#ifndef GE_GRAPH_MANAGER_GRAPH_MANAGER_UTILS_H_
#define GE_GRAPH_MANAGER_GRAPH_MANAGER_UTILS_H_

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/blocking_queue.h"
#include "common/ge_types.h"
#include "common/types.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "model/ge_model.h"
#include "register/register_fmk_types.h"

namespace ge {
// state for graph task in life cycle
enum GraphNodeState {
  GRAPH_NODE_INIT = 0,
  GRAPH_NODE_READY,
};

using GraphId = uint32_t;
using ConstModelPtr = std::shared_ptr<const ge::Model>;
using GeModelPtr = std::shared_ptr<ge::GeModel>;

using ConstGraphPtr = std::shared_ptr<const ge::Graph>;
using GraphPtr = std::shared_ptr<ge::Graph>;

const uint64_t INVALID_SESSION_ID = 0xffffffffffffffffULL;

struct ModelIdInfo {
  uint32_t model_id{INVALID_MODEL_ID};
};

class SubGraphInfo {
 public:
  SubGraphInfo();

  ~SubGraphInfo();

  void SetSubGraph(const ComputeGraphPtr &sub_graph_ptr) { subgraph_ptr_ = sub_graph_ptr; }
  ComputeGraphPtr GetSubGraph() const { return subgraph_ptr_; }

  void SetEngineName(const std::string &engine_name) { engine_name_ = engine_name; }
  const std::string &GetEngineName() const { return engine_name_; }

  void SetInputFlag(const std::vector<bool> &input_flag) { input_flag_ = input_flag; }
  const std::vector<bool> &GetInputFlag() const { return input_flag_; }

  void SetOutputFlag(const std::vector<bool> &output_flag) { output_flag_ = output_flag; }
  const std::vector<bool> &GetOutputFlag() const { return output_flag_; }

  void SetModelIdInfo(const ModelIdInfo &model_id_info) { model_id_info_ = model_id_info; }
  ModelIdInfo GetModelIdInfo() const { return model_id_info_; }

  void SetGeModelPtr(const GeModelPtr &ge_model_ptr) { ge_model_ptr_ = ge_model_ptr; }
  bool GeModelIsValid() const { return ge_model_ptr_ != nullptr; }

  Status FreeInOutBuffer();

  void SetOutputContext(const std::string &output) { output_names_ = output; }
  std::string GetOutputContext() const { return output_names_; }

  void SetStreamLabel(const std::string &stream_label) { stream_label_ = stream_label; }
  const std::string &GetStreamLabel() const { return stream_label_; }

  void SetEnd2PldMap(std::unordered_map<ge::NodePtr, ge::NodePtr> &end_map) { end_to_pld_ = end_map; }
  const std::unordered_map<ge::NodePtr, ge::NodePtr> &GetEnd2PldMap() const { return end_to_pld_; }

  void SetPld2EndMap(std::unordered_map<ge::NodePtr, ge::NodePtr> &pld_map) { pld_to_end_ = pld_map; }
  const std::unordered_map<ge::NodePtr, ge::NodePtr> &GetPld2EndMap() const { return pld_to_end_; }

 private:
  ComputeGraphPtr subgraph_ptr_;
  std::string engine_name_;
  std::vector<bool> input_flag_;
  std::vector<bool> output_flag_;
  ModelIdInfo model_id_info_;
  GeModelPtr ge_model_ptr_;
  bool malloc_flag_;
  std::vector<void *> buffer_addr_;
  std::string output_names_;
  std::vector<uint32_t> buffer_size_;
  std::string stream_label_;
  std::unordered_map<ge::NodePtr, ge::NodePtr> end_to_pld_;
  std::unordered_map<ge::NodePtr, ge::NodePtr> pld_to_end_;
};

using SubGraphInfoPtr = std::shared_ptr<ge::SubGraphInfo>;

// for run graph async listener
class RunAsyncListener : public ge::ModelListener {
 public:
  RunAsyncListener() : sem_(1) {}

  ~RunAsyncListener() = default;

  void SetCallback(const std::function<void(Status)> &callback);

  // callback
  Status OnComputeDone(uint32_t model_id, uint32_t task_id, uint32_t result) override;

 private:
  std::function<void(Status)> callback_;
  BlockingQueue<uint8_t> sem_;
};

// single graph node info
class GraphNode {
 public:
  explicit GraphNode(GraphId graph_id);
  ~GraphNode();

  GraphId GetGraphId() const { return graph_id_; }

  ConstGraphPtr GetGraph() const { return graph_; }
  void SetGraph(const GraphPtr &graph) { graph_ = graph; }

  ComputeGraphPtr GetComputeGraph() const { return compute_graph_; }
  void SetComputeGraph(const ComputeGraphPtr &compute_graph) { compute_graph_ = compute_graph; }

  bool GetRunFlag() const { return run_flag_; }
  void SetRunFlag(bool flag) { run_flag_ = flag; }

  void SetSubGraph(std::vector<SubGraphInfoPtr> &subgraph_ptr_list) { subgraph_ptr_list_ = subgraph_ptr_list; }
  const std::vector<SubGraphInfoPtr> &GetAllSubGraph() const { return subgraph_ptr_list_; }

  bool GetBuildFlag() const { return build_flag_; }
  void SetBuildFlag(bool buildFlag) { build_flag_ = buildFlag; }
  bool GetLoadFlag() const { return load_flag_; }
  void SetLoadFlag(bool load_flag) { load_flag_ = load_flag; }
  void SetGeModel(const GeModelPtr &ge_model) { ge_model_ = ge_model; }
  GeModelPtr GetGeModel() const { return ge_model_; }
  const std::map<std::string, std::string> &GetOptions() const { return options_; }
  void SetOptions(const std::map<std::string, std::string> &options) { options_ = options; }
  void Lock();
  void Unlock();

  // run graph asynchronous listener
  std::shared_ptr<RunAsyncListener> graph_run_async_listener_;

 private:
  GraphId graph_id_;
  std::map<std::string, std::string> options_;
  bool run_flag_;
  std::vector<SubGraphInfoPtr> subgraph_ptr_list_;

  GraphPtr graph_;
  ComputeGraphPtr compute_graph_;
  bool build_flag_;
  bool load_flag_;
  GeModelPtr ge_model_;
  BlockingQueue<uint8_t> sem_;
};

using GraphNodePtr = std::shared_ptr<GraphNode>;
using ConstGraphNodePtr = shared_ptr<const GraphNode>;

class GraphModelListener : public ge::ModelListener {
 public:
  GraphModelListener();

  ~GraphModelListener() = default;

  // callback
  Status OnComputeDone(uint32_t model_id, uint32_t task_id, uint32_t result) override;

  Status SetCondition(std::mutex *mutex, std::condition_variable *cond);

  Status ResetResult();

  // need lock by caller
  uint32_t GetResultCode() const;

  bool IsFinished() const { return is_finished_; }

 private:
  uint32_t result_code_;
  bool is_finished_;

  // not owner
  std::mutex *mutex_;
  // not owner
  std::condition_variable *condition_;
};

Status ParseOutNodes(const string &out_nodes);

struct GraphManagerOptions {
  int32_t stream_num;
  int32_t perf_level;
  int32_t encrypt_mode;
  int32_t framework_type;
  std::string ek_file;
  std::string cert_file;
  std::string hw_key_file;
  std::string private_key_file;
  std::string calibration_conf_file;
  std::string insert_op_file;
  std::string output_node_name;
  std::string func_bin_path;
  std::string input_nodes_set_fp16;
  std::string core_type;
  bool compress_flag;
  bool run_graph_flag;
  bool train_graph_flag;
  bool local_fmk_op_flag;
  bool hcom_parallel;
  bool enable_print_op_pass;
  std::map<std::string, int> stream_max_parallel_num;
  std::string output_datatype;
  std::string original_model_file;
  bool save_original_model;
  GraphManagerOptions()
      : stream_num(1),
        perf_level(domi::GEN_TASK_WITHOUT_FUSION),
        encrypt_mode(-1),
        framework_type(domi::FMK_TYPE_T),
        ek_file(""),
        cert_file(""),
        hw_key_file(""),
        private_key_file(""),
        calibration_conf_file(""),
        insert_op_file(""),
        output_node_name(""),
        func_bin_path(""),
        core_type(""),
        compress_flag(false),
        run_graph_flag(false),
        train_graph_flag(false),
        local_fmk_op_flag(false),
        hcom_parallel(false),
        enable_print_op_pass(true),
        save_original_model(false) {}
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MANAGER_UTILS_H_
