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
#ifndef GE_MODEL_GE_ROOT_MODEL_H_
#define GE_MODEL_GE_ROOT_MODEL_H_

#include <map>
#include "graph/compute_graph.h"
#include "common/model/ge_model.h"
#include "common/model/model_relation.h"
#include "framework/pne/pne_model.h"
#include "common/op_so_store/op_so_store.h"

namespace ge {
 class GeRootModel : public std::enable_shared_from_this<GeRootModel>, public PneModel {
 public:
  GeRootModel() = default;
  explicit GeRootModel(const ComputeGraphPtr &root_graph) : PneModel(root_graph) {};
  ~GeRootModel() override = default;

  void SetSubgraphInstanceNameToModel(const std::string &instance_name, const GeModelPtr &ge_model);
  void RemoveInstanceSubgraphModel(const std::string &instance_name);
  const std::map<std::string, GeModelPtr> &GetSubgraphInstanceNameToModel() const {
    return subgraph_instance_name_to_model_;
  };

  void SetModelId(uint32_t model_id) override {
    const std::lock_guard<std::mutex> lock(model_ids_mutex_);
    PneModel::SetModelId(model_id);
    // cached for removement
    model_ids_.emplace_back(model_id);
  }

  void SetIsSpecificStream(const bool is_specific_stream) { is_specific_stream_ = is_specific_stream; }

  bool IsSpecificStream() const { return is_specific_stream_; }

  std::vector<uint32_t> GetAllModelId() const { return model_ids_; }

  void ClearAllModelId() { model_ids_.clear(); }

  Status CheckIsUnknownShape(bool &is_dynamic_shape) const;

  Status SerializeModel(ModelBufferData &model_buff) override;

  Status UnSerializeModel(const ModelBufferData &model_buff) override;

  std::string GetLogicDeviceId() const override;

  Status SetLogicDeviceId(const std::string &logic_device_id) override;

  void SetWeightSize(const int64_t weight_size) { total_weight_size_ = weight_size; }
  int64_t GetWeightSize() const { return total_weight_size_; }

  void SetFlattenGraph(const ComputeGraphPtr &flatten_graph) {
    const std::lock_guard<std::mutex> lock(model_ids_mutex_);
    flatten_graph_ = flatten_graph;
  }
  ComputeGraphPtr GetFlattenGraph() const { return flatten_graph_; }

  void SetNodesToTaskDef(const std::unordered_map<ge::NodePtr, std::vector<domi::TaskDef>> &nodes_2_task_def) {
    const std::lock_guard<std::mutex> lock(model_ids_mutex_);
    nodes_to_task_defs_ = nodes_2_task_def;
  }
  const std::unordered_map<ge::NodePtr, std::vector<domi::TaskDef>> &GetNodesToTaskDef() const {
    return nodes_to_task_defs_;
  }

  void SetGraphToStaticModels(const std::unordered_map<std::string, ge::GeModelPtr> &graph_2_static_models) {
    const std::lock_guard<std::mutex> lock(model_ids_mutex_);
    graph_to_static_models_ = graph_2_static_models;
  }
  const std::unordered_map<std::string, ge::GeModelPtr> &GetGraphToStaticModels() const {
    return graph_to_static_models_;
  }

  const uint8_t *GetOpSoStoreData() const;

  size_t GetOpStoreDataSize() const;

  bool LoadSoBinData(const uint8_t *const data, const size_t len);

  std::vector<OpSoBinPtr> GetAllSoBin() const;

  bool CheckAndSetNeedSoInOM();

  bool GetSoInOmFlag() const;

  void SetSoInOmInfo(const SoInOmInfo &so_info);

  SoInOmInfo GetSoInOmInfo() const;

  void SetFileConstantWeightDir(const std::string &file_constant_weight_dir) {
    file_constant_weight_dir_ = file_constant_weight_dir;
  }

  const std::string GetFileConstantWeightDir() const {
    return file_constant_weight_dir_;
  }

 private:
  std::map<std::string, GeModelPtr> subgraph_instance_name_to_model_;
  // In multithread online secenario, same graph can owns different davinci_model for for concurrency
  std::vector<uint32_t> model_ids_;
  std::mutex model_ids_mutex_;
  bool is_specific_stream_ = false;

  // loaded model info
  int64_t total_weight_size_ = 0;
  // Compile results of dynamic compiled graph
  std::unordered_map<ge::NodePtr, std::vector<domi::TaskDef>> nodes_to_task_defs_;
  // Compile results of static compiled graph
  std::unordered_map<std::string, ge::GeModelPtr> graph_to_static_models_;
  // flattend graph after load model
  ComputeGraphPtr flatten_graph_ = nullptr;
  OpSoStore op_so_store_;
  bool so_in_om_ = false;
  SoInOmInfo so_info_ = {};
  std::string file_constant_weight_dir_;
};
using GeRootModelPtr = std::shared_ptr<ge::GeRootModel>;
}  // namespace ge
#endif  // GE_MODEL_GE_ROOT_MODEL_H_
