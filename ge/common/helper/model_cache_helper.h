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

#ifndef GE_COMMON_HELPER_MODEL_CACHE_HELPER_H_
#define GE_COMMON_HELPER_MODEL_CACHE_HELPER_H_

#include <nlohmann/json.hpp>
#include <set>
#include <string>

#include "ge/ge_api_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_var_manager.h"
#include "model/ge_model.h"

namespace ge {
using Json = nlohmann::json;

struct CacheInfo {
  size_t node_num;
  size_t edge_num;
  size_t graph_hash;
  map<std::string, size_t> nodes_hash;
  CacheInfo() : node_num(0), edge_num(0), graph_hash(0) {}
};

class ModelCacheHelper {
 public:
  ModelCacheHelper(uint64_t session_id, uint32_t graph_id, ComputeGraphPtr &compute_graph);
  ~ModelCacheHelper();

  Status SaveCacheInfoToCache () const;
  Status SaveVarManagerToCache(bool before_build) const;
  Status SaveOmModelToCache(const GeModelPtr &ge_model) const;
  bool IsModelCacheHit() const;
  Status RecoverVarManagerFromCache() const;
  Status LoadOmModelFromCache(GeModelPtr &ge_model) const;
  Status RefreshComputeGraph(const ComputeGraphPtr &compute_graph);
  Status ClearCache(uint32_t graph_id) const;

 private:
  Status GetComputeGraphHash(size_t &hash) const;
  Status GetNodesHash(map<std::string, size_t> &hash_map) const;
  Status GetCacheInfo(CacheInfo &cache_info) const;

  Status RecoverMemResource(const Json &json) const;
  Status RecoverAllocatedGraphId(const Json &json) const;
  Status RecoverChangedGraphId(const Json &json) const;
  Status RecoverVarAddrAndTensorDesc(const Json &json) const;
  Status RecoverBroadcastInfo(const Json &json) const;
  Status RecoverTransRoads(const Json &json) const;
  static Status GetNodesNeedRecompile(ComputeGraphPtr &graph, vector<NodePtr> &nodes);
  static Status RecompileNodes(GeModelPtr &ge_model);

  bool IsNodeHashSameAsCache(const map<std::string, size_t> &hash_map) const;
  bool IsMemResourceSameAsCache(Json &json) const;
  bool IsChangedGraphIdSameAsCache(Json &json) const;
  bool IsAllocatedGraphIdSameAsCache(Json &json) const;
  bool IsCurVarTensorDescSameAsCache(Json &json) const;
  bool IsVarAddrMgrMapSameAsCache(Json &json) const;
  bool IsBroadcastInfoSameAsCache(Json &json) const;
  bool IsTransRoadsSameAsCache(Json &json) const;
  bool IsVarManagerSameAsCache(Json &json) const;
  bool IsVarManagerParamSameAsCache(Json &json) const;

  Status SaveJsonToFile(const string &file_name, const Json &json) const;
  Status LoadJsonFromFile(const string &file_name, Json &json) const;

  Status GetNodesHashMapJson(Json &json) const;
  Status GetMemResourceMap(Json &json) const;
  Status GetVarAddrMgrMapJson(Json &json) const;
  Status GetCurVarTensorDescMapJson(Json &json) const;
  Status GetTransRoadsJson(Json &json) const;
  Status GetChangedGraphIdJson(Json &json) const;
  Status GetAllocatedGraphIdJson(Json &json) const;
  Status GetBroadcastInfoJson(Json &json) const;
  Status GetVarResourceJson(Json &json) const;
  Status GetVarManagerJson(Json &json) const;

  static Status TensorDescToJson(const GeTensorDesc &ge_tensor_desc, Json &json);
  static Status JsonToTensorDesc(const Json &json, GeTensorDesc &ge_tensor_desc);
  static Status ParseMemResourceFromJson(const Json &json, map<rtMemType_t, int64_t> &mem_resource);
  static Status ParseVarAddrMgrMapFromJson(const Json &json,
                                           std::vector<std::pair<std::string, VarAddrMgr>> &var_addr_mgr_vector,
                                           std::unordered_set<uint64_t> &var_offset_set);
  static Status ParseCurVarTensorDescMapFromJson(
      const Json &json, std::unordered_map<std::string, ge::GeTensorDesc> &cur_var_tensor_desc_map);
  static Status ParseTransRoadsFromJson(const Json &json,
                                        std::unordered_map<std::string, std::vector<TransNodeInfo>> &trans_roads);
  static Status ParseChangedGraphIdFromJson(const Json &json,
                                            std::unordered_map<std::string, uint32_t> &changed_graph_id);
  static Status ParseAllocatedGraphIdFromJson(const Json &json,
                                              std::unordered_map<std::string, uint32_t> &allocated_graph_id);
  static Status ParseBroadcastInfoFromJson(const Json &json,
                                           std::unordered_map<std::string, VarBroadCastInfo> &var_broadcast_info);
  static Status GetVarNameFromVarKey(const string &var_key, const GeTensorDesc &tensor_desc, string &var_name);

  uint64_t session_id_;
  uint32_t graph_id_;
  string cache_path_;
  ComputeGraphPtr compute_graph_;
  std::set<string> var_names_;
  bool is_cache_path_valid_for_output;
  static map<uint32_t, uint32_t> graph_id_run_times_;
};

using ModelCacheHelperPtr = std::shared_ptr<ModelCacheHelper>;
}  // namespace ge

#endif  // GE_COMMON_HELPER_MODEL_CACHE_HELPER_H_
