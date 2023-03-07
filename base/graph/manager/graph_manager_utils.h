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
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "common/context/local_context.h"
#include "common/context/ome_context.h"
#include "external/graph/graph.h"
#include "graph/model.h"
#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "pne/model/flow_model.h"
#include "external/register/register_fmk_types.h"
#include "external/ge/ge_api_types.h"
#include "framework/common/util.h"

namespace ge {

using GraphId = uint32_t;
using ConstGraphPtr = std::shared_ptr<const ge::Graph>;

constexpr uint64_t INVALID_SESSION_ID = 0xffffffffffffffffUL;
constexpr uint32_t kMaxLoadNum = 8U;

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

  void SetOutputFlag(const std::vector<bool> &output_flag) { output_flag_ = output_flag; }

  void SetOutputContext(const std::string &output) { output_names_ = output; }

  void SetStreamLabel(const std::string &stream_label) { stream_label_ = stream_label; }
  const std::string &GetStreamLabel() const { return stream_label_; }

  void SetEnd2PldMap(const std::unordered_map<ge::NodePtr, ge::NodePtr> &end_map) { end_to_pld_ = end_map; }
  const std::unordered_map<ge::NodePtr, ge::NodePtr> &GetEnd2PldMap() const { return end_to_pld_; }

  void SetPld2EndMap(const std::unordered_map<ge::NodePtr, ge::NodePtr> &pld_map) { pld_to_end_ = pld_map; }
  const std::unordered_map<ge::NodePtr, ge::NodePtr> &GetPld2EndMap() const { return pld_to_end_; }

 private:
  ComputeGraphPtr subgraph_ptr_;
  std::string engine_name_;
  std::vector<bool> input_flag_;
  std::vector<bool> output_flag_;
  ModelIdInfo model_id_info_;
  GeModelPtr ge_model_ptr_;
  std::string output_names_;
  std::string stream_label_;
  std::unordered_map<ge::NodePtr, ge::NodePtr> end_to_pld_;
  std::unordered_map<ge::NodePtr, ge::NodePtr> pld_to_end_;
};

using SubGraphInfoPtr = std::shared_ptr<ge::SubGraphInfo>;
using Graph2SubGraphInfoList = std::unordered_map<ComputeGraphPtr, std::vector<SubGraphInfoPtr>>;
using Graph2InputNodesSubGraphInfo = std::unordered_map<ComputeGraphPtr, SubGraphInfoPtr>;

// for run graph async listener
class RunAsyncListener : public ModelListener {
 public:
  RunAsyncListener() : ModelListener(), sem_(1U) {}

  ~RunAsyncListener() override = default;

  void SetCallback(const RunAsyncCallback &callback) override;

  // callback
  Status OnComputeDone(const uint32_t model_id, const uint32_t data_index, const uint32_t result_code,
                       std::vector<Tensor> &outputs) override;

 private:
  RunAsyncCallback callback_;
  BlockingQueue<uint8_t> sem_;
};

// single graph node info
class GraphNode {
 public:
  explicit GraphNode(const GraphId graph_id);
  ~GraphNode();

  GraphId GetGraphId() const { return graph_id_; }

  ConstGraphPtr GetGraph() const { return graph_; }
  void SetGraph(const GraphPtr &graph) { graph_ = graph; }

  ComputeGraphPtr GetComputeGraph() const { return compute_graph_; }
  void SetComputeGraph(const ComputeGraphPtr &compute_graph) { compute_graph_ = compute_graph; }

  bool GetRunFlag() const { return run_flag_; }
  void SetRunFlag(const bool flag) { run_flag_ = flag; }

  void SetOmeContext(const OmeContext &context) { context_ = context; }
  const OmeContext &GetOmeContext() const { return context_; }

  bool IsAsync() const { return async_; }
  void SetAsync(const bool flag) { async_ = flag; }

  bool GetBuildFlag() const { return build_flag_; }
  void SetBuildFlag(const bool buildFlag) { build_flag_ = buildFlag; }
  bool GetLoadFlag() const { return load_flag_; }
  // allow repeatively load graph owns same graph id
  void UpdateLoadFlag() { load_flag_ = ((load_count_ == 0U) || (load_record_ >= kMaxLoadNum)); }
  void SetLoadFlag(const bool load_flag) { load_flag_ = load_flag; }
  void SetIsSpecificStream(const bool specific_stream) { is_specific_stream_ = specific_stream; }
  bool IsSpecificStream() const { return is_specific_stream_; }
  void SetFlowModel(const FlowModelPtr &flow_model) { flow_model_ = flow_model; }
  FlowModelPtr GetFlowModel() const { return flow_model_; }
  const std::map<std::string, std::string>& GetOptions() const { return options_; }
  void SetOptions(const std::map<std::string, std::string> &options) { options_ = options; }
  void Lock();
  void Unlock();

  void SetSemSize(const uint32_t size) { sem_.SetMaxSize(size); }

  void SetLoadCount(const uint32_t count) { load_count_ = count; }
  uint32_t GetLoadRecord() const { return load_record_; }
  void SetLoadRecord(const uint32_t record) { load_record_ = record; }
  void IncreaseLoadCount();
  void SetLoaded();

 private:
  GraphId graph_id_;
  std::map<std::string, std::string> options_;
  bool run_flag_{false};
  std::vector<SubGraphInfoPtr> subgraph_ptr_list_;

  OmeContext context_;

  GraphPtr graph_;
  ComputeGraphPtr compute_graph_;
  bool build_flag_{false};
  // load_flag_ is true if more than 1 model were loaded
  bool load_flag_{false};
  bool async_{false};
  bool is_specific_stream_{false};
  GeModelPtr ge_model_;
  FlowModelPtr flow_model_;
  BlockingQueue<uint8_t> sem_;
  // consist with graph_count of same graph_id in graph_manager
  uint32_t load_count_{0U};
  // total times of loading a graph with same graph_id.
  uint32_t load_record_{0U};
  std::mutex load_count_mu_;
};

using GraphNodePtr = std::shared_ptr<GraphNode>;

class GraphModelListener : public ge::ModelListener {
 public:
  GraphModelListener();

  ~GraphModelListener() override = default;

  // callback
  Status OnComputeDone(const uint32_t model_id, const uint32_t data_index, const uint32_t result_code,
                       std::vector<ge::Tensor> &outputs) override;

  uint32_t GetResultCode() override;

  Status ResetResult() override;

 private:
  uint32_t result_code_;
  bool is_finished_;

  std::mutex mutex_;
  std::condition_variable condition_;
};

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
  std::string input_format;
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
  bool is_single_op;
  std::string dynamic_image_size;
  std::map<std::string, int32_t> stream_max_parallel_num;
  std::string output_datatype;
  std::string original_model_file;
  std::string save_original_model;
  std::string build_mode;
  std::string build_step;
  std::string tuning_path;
  std::string input_shape;
  std::string dynamic_dims;
  int32_t dynamic_node_type = -1;
  std::set<std::string> exclude_engines;
  std::string build_inner_model = "true";
  GraphManagerOptions()
      : stream_num(1),
        perf_level(domi::GEN_TASK_WITHOUT_FUSION),
        encrypt_mode(-1),
        framework_type(domi::TENSORFLOW),
        ek_file(""),
        cert_file(""),
        hw_key_file(""),
        private_key_file(""),
        calibration_conf_file(""),
        insert_op_file(""),
        input_format(""),
        output_node_name(""),
        func_bin_path(""),
        core_type(""),
        compress_flag(false),
        run_graph_flag(false),
        train_graph_flag(false),
        local_fmk_op_flag(false),
        hcom_parallel(false),
        enable_print_op_pass(true),
        is_single_op(false),
        save_original_model("false"),
        build_mode(""),
        build_step(""),
        tuning_path("") {}
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MANAGER_UTILS_H_
