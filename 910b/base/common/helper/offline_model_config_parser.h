/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef OFFLINE_MODEL_CONFIG_PARSER_H_
#define OFFLINE_MODEL_CONFIG_PARSER_H_

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include "mmpa/mmpa_api.h"
#include "nlohmann/json.hpp"
#include "ge/ge_api_error_codes.h"
#include "common/model/model_relation.h"
#include "common/model/model_deploy_resource.h"
namespace ge {
void to_json(nlohmann::json &json_obj, const HcomCommGroup &comm_group);
struct DeployConfig {
  std::string submodel_name;
  std::string deploy_device;
};
struct SubModelNameId {
  std::string submodel_name;
  uint32_t submode_id;
};
struct CfgCommGroup {
  std::string group_name;
  std::string rank_list;
};
struct CfgRankTable {
  uint32_t rank_id;
  uint32_t submodel_instance_id;
};

struct ConfigModelRelation {
  std::string edge_type; // it is only support flow type which maybe support event in the future
  std::string edge_src;  // -1:0 input0 of root model; 1:0 output0 of submodel1
  std::string edge_dst;  // -1:0 output0 of root model; 1:0 input0 of submodel1
};

class OfflineModelRelation {
public:
  struct EdgeInfo {
    int32_t related_model_instance_id; // -1 means root model
    uint32_t related_indices; // indices of related model input or output (while type is flow)
  };

  struct SubmodelConfigInfo {
    uint32_t submodel_instance_id = 0xFFFFFFFFU; // set by user
    std::string submodel_name; // part of submodel path
    std::string submodel_path;
    std::string model_type; // CPU NPU
    uint32_t rank_id = 0xFFFFFFFFU;;
    std::vector<int32_t> deploy_logic_device_ids;         // nodeid:itemid:deviceid
    std::map<uint32_t, EdgeInfo> flow_input_mapping;      // key: input indices of current model
    std::map<uint32_t, std::vector<EdgeInfo>> flow_output_mapping;     // key: output indices of current model
  };
  explicit OfflineModelRelation(std::map<std::string, std::string> submodel_name_to_file);
  OfflineModelRelation() = default;
  ~OfflineModelRelation() = default;

  Status InitSubmodelInfos(const std::map<std::string, std::string> &name_to_logic_dev_list,
                           const std::map<std::string, uint32_t> &name_to_instance_ids);
  Status FillSubModelFlowRelation(const std::vector<std::string> &model_id_with_src_id,
                                  const std::vector<std::string> &model_id_with_dst_id);
  void SetWithRelation(const bool is_with_relation) { with_model_relation_ = is_with_relation; }

  // find name and indices according to model file
  void GetModelNameByFileName(const std::string &model_file, std::string &model_name) const;
  // find indices according to model name
  void GetModelInputIndices(const std::string &model_name,
                            std::map<uint32_t, uint32_t> &related_indices) const;

  void GetAllModelFiles(std::vector<std::string> &model_files) const;
  Status GetRelatedSubmodelnameAndIndices(const std::string &submodel_name,
      std::map<std::string, std::map<uint32_t, uint32_t>> &input_name_to_indices,
      std::map<std::string, std::map<uint32_t, uint32_t>> &output_name_to_indices) const;
  bool IsTensorParallel() const { return !with_model_relation_; }
  Status GetLogicDeviceId(const std::string &model_name, std::string &logic_device_id);
  bool IsCpuModel(const std::string &model_name) const;
  void PrintSubmodelConfig() const;
  Status SetGroupNameToRankIds(const std::vector<CfgCommGroup> &cfg_comm_groups);
  Status SetRankIdToModelIds(const std::vector<CfgRankTable> &cfg_rank_table);
  Status CheckHcomInfoValid();
  Status GetRankIdBySubmodelName(const std::string &submodel_name, uint32_t &rank_id) const;
  Status GetLogicDeviceIdToRankIds(std::map<std::string, std::vector<uint32_t>> &device_ids_to_rank_ids) const;
  void GetGroupNameToRankIds(std::map<std::string, std::vector<uint32_t>> &group_name_to_rank_ids) const;
  bool IsWithHcomInfo() const { return with_hcom_info_; }
  Status GetCfgGroupNameTable(std::string &hcom_group_to_ranks) const;
private:
  Status TransferToLogicDeviceId(const std::vector<int32_t> &logic_dev_ids, std::string &logic_device_id) const;
  // data split is without model relation
  bool with_model_relation_ = false;
  bool with_hcom_info_ = false;
 // submodel name is part of model path
  std::map<std::string, SubmodelConfigInfo> submodel_name_to_info_;
 // key submodel name, same as file name xxxx.onnx, value file path
  std::map<std::string, std::string> submodel_name_to_file_list_;
  std::map<uint32_t, std::string> submode_id_to_name_;
  // key model file path related input;value(key:current submodel indices value: root indices)
  std::map<std::string, std::map<uint32_t, uint32_t>> input_files_to_related_indices_;
  // key model file path related output;value(key:current submodel indices value: root indices)
  std::map<std::string, std::map<uint32_t, uint32_t>> output_files_to_related_indices_;
  std::map<std::uint32_t, std::set<string>> rank_id_to_group_names_;
  std::map<uint32_t, uint32_t> model_instance_id_to_rank_id_;
};

class OfflineModelConfigParser {
 public:
  static OfflineModelConfigParser &Instance();
  void Reset();
  // using for atc
  Status ParserOfflineModelConfig(const std::string &offline_model_relation_path,
                                  const std::string &offline_model_path);
  // using for irbuild
  Status ParserOfflineModelConfig(const std::string &offline_model_relation_path,
                                  const std::vector<std::string> &src_graph_names);

  // find name according to model file
  void GetModelNameByFileName(const std::string &model_file, std::string &model_name) const;
  // find indices according to model name
  void GetModelInputIndices(const std::string &model_name,
                            std::map<uint32_t, uint32_t> &related_indices) const;

  void GetAllModelFiles(std::vector<std::string> &model_files) const;

  Status GetRelatedSubmodelnameAndIndices(const std::string &submodel_name,
      std::map<std::string, std::map<uint32_t, uint32_t>> &input_name_to_indices,
      std::map<std::string, std::map<uint32_t, uint32_t>> &output_name_to_indices) const;

  bool IsTensorParallel() const;
  bool IsActive() const;
  Status GetLogicDeviceId(const std::string &model_name, std::string &logic_device_id) const;
  bool IsCpuModel(const std::string &model_name) const;
  Status GetRankIdBySubmodelName(const std::string &submodel_name, uint32_t &rank_id) const;
  Status GetLogicDeviceIdToRankIds(std::map<std::string, std::vector<uint32_t>> &device_ids_to_rank_ids) const;
  Status GetGroupNameToRankIds(std::map<std::string, std::vector<uint32_t>> &group_name_to_rank_ids) const;
  bool IsWithHcomInfo() const;
  Status GetCfgGroupNameTable(std::string &hcom_group_to_ranks) const;
 private:
  OfflineModelConfigParser() = default;
  ~OfflineModelConfigParser() = default;

  Status ParserOfflineModelConfig(const std::string &offline_model_relation_path,
                                  OfflineModelRelation &model_relation) const;
  Status ParserSubmodelDeployConfig(const std::vector<DeployConfig> &deploy_configs,
                                    std::map<std::string, std::string> &name_to_logic_dev) const;
  Status ParserSubmodelInstanceId(const std::vector<SubModelNameId> &name_to_ids,
                                  std::map<std::string, uint32_t> &name_to_instance_id) const;
  Status ParserSubmodelRelation(const std::vector<ConfigModelRelation> &relations,
                                OfflineModelRelation &model_relation) const;
  Status ResolveSubmodelPath(const std::string &offline_model_path,
                             std::map<std::string, std::string> &name_to_file) const;
  std::shared_ptr<OfflineModelRelation> GetParsedRelation() const;

  mutable std::mutex mutex_;
  std::map<std::string, std::shared_ptr<OfflineModelRelation>> relation_path_to_relation_;
};
}  // namespace ge
#endif  // OFFLINE_MODEL_CONFIG_PARSER_H_