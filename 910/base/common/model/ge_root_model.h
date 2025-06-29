/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GE_MODEL_GE_ROOT_MODEL_H_
#define GE_MODEL_GE_ROOT_MODEL_H_

#include <map>
#include <sstream>
#include "ge/ge_graph_compile_summary.h"
#include "ge/ge_allocator.h"
#include "graph/compute_graph.h"
#include "common/model/ge_model.h"
#include "common/model/model_relation.h"
#include "framework/pne/pne_model.h"
#include "common/op_so_store/op_so_store.h"
#include "common/memory/mem_type_utils.h"
#include "common/memory/feature_memory_impl.h"
#include "common/host_resource_center/host_resource_center.h"

namespace ge {
struct FixedFeatureMemory {
  std::string ToString() const {
    std::stringstream ss;
    ss << "rts memory type: " << MemTypeUtils::ToString(type) << ", addr: " << std::hex << addr << ", size: "
       << std::dec << size << ", user_alloc: " << user_alloc << ", ge_alloc: " << ge_alloc << ", block: "
       << std::hex << block;
    return ss.str();
  }
  rtMemType_t type;
  void *addr;
  size_t size;
  bool user_alloc;
  bool ge_alloc;
  MemBlock *block; // 外置allocator调用malloc返回MemBlock指针，释放内存时使用
};

 class GeRootModel : public std::enable_shared_from_this<GeRootModel>, public PneModel {
 public:
  GeRootModel() = default;
  ~GeRootModel() override = default;

  Status Initialize(const ComputeGraphPtr &root_graph);
  // host_resource_manager基于root_graph实现资源共享，而反序列化的时候，每个model单独反序列化出了graph对象，
  // 这会导致共享资源丢失。在反序列化后调用该接口，从root_graph上获取subgraph设置到submodel上
  Status ModifyOwnerGraphForSubModels();
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

  std::string GetRedundantLogicDeviceId() const override;

  Status SetRedundantLogicDeviceId(const std::string &logic_device_id) override;

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

  uint32_t GetCurModelId() const { return cur_model_id_; }

  void SetCurModelId(uint32_t model_id) { cur_model_id_ = model_id; }

  const std::map<rtMemType_t, FixedFeatureMemory> &GetFixedFeatureMemory() const {
    return fixed_feature_mems_;
  }

  std::map<rtMemType_t, FixedFeatureMemory> &MutableFixedFeatureMemory() {
    return fixed_feature_mems_;
  }

  Status GetSummaryFeatureMemory(std::vector<FeatureMemoryPtr> &all_feature_memory,
                                 size_t &hbm_fixed_feature_mem);

  bool IsNeedMallocFixedFeatureMem() const;
  bool IsNeedMallocFixedFeatureMemByType(const rtMemType_t rt_mem_type) const;
  HostResourceCenterPtr GetHostResourceCenterPtr() const;

 private:
  Status SetLogicDeviceId(const std::string &logic_device_id, bool is_redundant);

  std::string GetLogicDeviceId(bool is_redundant) const;

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
  uint32_t cur_model_id_ = 0U;

  bool all_feature_memory_init_flag_ = false;
  std::vector<FeatureMemoryPtr> all_feature_memory_;
  std::map<rtMemType_t, FixedFeatureMemory> fixed_feature_mems_;
  HostResourceCenterPtr host_resource_center_ = ge::MakeShared<HostResourceCenter>();;
};
using GeRootModelPtr = std::shared_ptr<ge::GeRootModel>;
}  // namespace ge
#endif  // GE_MODEL_GE_ROOT_MODEL_H_
