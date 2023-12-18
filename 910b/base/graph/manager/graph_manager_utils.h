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
#include "framework/pne/flow_model.h"
#include "external/register/register_fmk_types.h"
#include "external/ge/ge_api_types.h"
#include "external/ge/ge_graph_compile_summary.h"
#include "external/graph/types.h"
#include "framework/common/util.h"
#include "ge/ge_allocator.h"
#include "graph/ge_tensor.h"

namespace ge {

using GraphId = uint32_t;
using ConstGraphPtr = std::shared_ptr<const ge::Graph>;

constexpr uint64_t INVALID_SESSION_ID = 0xFFFFFFFFFFFFFFFFULL;
constexpr uint32_t INVALID_GRAPH_ID = 0xFFFFFFFFU;
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

  void SetInputFlag(const vector_bit_t &input_flag) { input_flag_ = input_flag; }

  void SetOutputFlag(const vector_bit_t &output_flag) { output_flag_ = output_flag; }

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
  vector_bit_t input_flag_;
  vector_bit_t output_flag_;
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

using InputMemoryBaseInfo = std::pair<const void *, size_t>; // <mem_base, mem_size>
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

  bool GetCompiledFlag() const { return compiled_flag_; }
  void SetCompiledFlag(const bool flag) { compiled_flag_ = flag; }
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
  void SetFeatureBaseRefreshable(const bool refreshable) { is_feature_base_refreshable_ = refreshable; }
  bool IsFeatureBaseRefreshable() const { return is_feature_base_refreshable_; }
  void SetConstMemoryBase(const void * const memory, const size_t size) {
    const_mem_ = std::make_pair(memory, size);
  }
  const InputMemoryBaseInfo &GetConstMemoryBase() const { return const_mem_; }
  void SetFeatureMemoryBase(const void * const memory, const size_t size) {
    feature_mem_ = std::make_pair(memory, size);
  }
  const InputMemoryBaseInfo &GetFeatureMemoryBase() const { return feature_mem_; }
  CompiledGraphSummaryPtr GetCompiledGraphSummary() const { return compiled_summary_; }
  void SaveCompiledGraphSummary(const CompiledGraphSummaryPtr &summary) { compiled_summary_ = summary; }
  void SetAppRefreshConstMemoryFlag() {
    app_refresh_const_memory_flag_ = true;
  }
  bool IsAppRefreshConstMemory() const {
    return app_refresh_const_memory_flag_;
  }
  void SetAppRefreshFeatureMemoryFlag() {
    app_refresh_feature_memory_flag_ = true;
  }
  bool IsAppRefreshFeatureMemory() const {
    return app_refresh_feature_memory_flag_;
  }
  void SetConstMemBlock(MemBlock *const const_mem_block) {
    const_mem_block_ = const_mem_block;
  }
  void SetFeatureMemBlock(MemBlock *const feature_mem_block) {
    feature_mem_block_ = feature_mem_block;
  }
  MemBlock *GetConstMemBlock() {
    return const_mem_block_;
  }
  MemBlock *GetFeatureMemBlock() {
    return feature_mem_block_;
  }
  void SetNetOutputNode(const NodePtr &net_output_node) { net_output_node_ = net_output_node; }
  NodePtr GetNetOutputNode() const { return net_output_node_; }

  void SetTensorSize(size_t tensor_size) {
    tensor_sizes_.emplace_back(tensor_size);
  }
  const std::vector<size_t> &GetTensorSize() const {
    return tensor_sizes_;
  }
  void SetGeTensorDescPtr(GeTensorDescPtr &ge_tensor_desc) {
    ge_tensor_descs_.emplace_back(ge_tensor_desc);
  }
  const std::vector<GeTensorDescPtr> &GetGeTensorDescPtr() const {
    return ge_tensor_descs_;
  }
  bool IsSavedNetOutputTensorInfoFlag() const {
    return is_saved_net_output_tensor_info_flag_;
  }
  void SetSavedNetOutputTensorInfoFlag(const bool is_saved_net_output_tensor_info_flag) {
    is_saved_net_output_tensor_info_flag_ = is_saved_net_output_tensor_info_flag;
  }

 private:
  GraphId graph_id_;
  std::map<std::string, std::string> options_;
  bool run_flag_{false};
  std::vector<SubGraphInfoPtr> subgraph_ptr_list_;

  OmeContext context_;

  GraphPtr graph_;
  ComputeGraphPtr compute_graph_;
  // set true only when Session::CompileGraph is called
  bool compiled_flag_{false};
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
  bool is_feature_base_refreshable_ = false;

  InputMemoryBaseInfo const_mem_;
  InputMemoryBaseInfo feature_mem_;
  CompiledGraphSummaryPtr compiled_summary_{nullptr};

  bool app_refresh_const_memory_flag_{false};
  bool app_refresh_feature_memory_flag_{false};
  MemBlock *const_mem_block_{nullptr};
  MemBlock *feature_mem_block_{nullptr};
  NodePtr net_output_node_;
  std::vector<size_t> tensor_sizes_;
  bool is_saved_net_output_tensor_info_flag_{false};
  std::vector<GeTensorDescPtr> ge_tensor_descs_;
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
  uint32_t result_code_ = 0U;
  bool is_finished_ = false;

  std::mutex mutex_;
  std::condition_variable condition_;
};

struct GraphManagerOptions {
  int32_t stream_num{1};
  int32_t perf_level{domi::GEN_TASK_WITHOUT_FUSION};
  int32_t encrypt_mode{-1};
  int32_t framework_type{domi::TENSORFLOW};
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
  bool compress_flag{false};
  bool run_graph_flag{false};
  bool train_graph_flag{false};
  bool local_fmk_op_flag{false};
  bool hcom_parallel{false};
  bool enable_print_op_pass{false};
  bool is_single_op{false};
  std::string dynamic_image_size;
  std::map<std::string, int32_t> stream_max_parallel_num;
  std::string output_datatype;
  std::string original_model_file;
  std::string save_original_model{"false"};
  std::string build_mode;
  std::string build_step;
  std::string tuning_path;
  std::string input_shape;
  std::string dynamic_dims;
  int32_t dynamic_node_type = -1;
  std::set<std::string> exclude_engines;
  std::string build_inner_model{"true"};
  std::string event{"event"};
  GraphManagerOptions() {}
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MANAGER_UTILS_H_
