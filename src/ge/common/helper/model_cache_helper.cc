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

#include <fcntl.h>
#include <unistd.h>
#include <climits>
#include <cstdio>
#include <fstream>
#include <functional>

#include "common/ge/ge_util.h"
#include "common/helper/model_cache_helper.h"
#include "common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_types.h"
#include "framework/common/helper/model_helper.h"
#include "framework/common/util.h"
#include "graph/detail/attributes_holder.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "graph/model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "init/gelib.h"
#include "proto/ge_ir.pb.h"

using namespace std;

namespace {
const char *const kTbeKernelInfoStoreName = "AIcoreEngine";
const char *const kGraphName = "temp_name";
// Keys of json
const char *const kNodeNum = "nodeNum";
const char *const kEdgeNum = "edgeNum";
const char *const kGraphHash = "graphHash";
const char *const kNodeHash = "nodeHash";
const char *const kHash = "hash";
const char *const kSessionId = "sessionId";
const char *const kDeviceId = "deviceId";
const char *const kJobId = "jobId";
const char *const kGraphMemMaxSize = "graphMemMaxSize";
const char *const kVarMemMaxSize = "varMemMaxSize";
const char *const kVarMemLogicBase = "varMemLogicBase";
const char *const kUseMaxMemSize = "useMaxMemSize";
const char *const kMemResourceMap = "memResourceMap";
const char *const kMemType = "memType";
const char *const kTotalSize = "totalSize";
const char *const kVarMemSize = "varMemSize";
const char *const kVarResource = "varResource";
const char *const kVarAddrMgrMap = "varAddrMgrMap";
const char *const kName = "name";
const char *const kAddress = "address";
const char *const kOffset = "offset";
const char *const kMemoryType = "memoryType";
const char *const kTensorDesc = "tensorDesc";
const char *const kDataType = "dataType";
const char *const kShape = "shape";
const char *const kLayout = "layout";
const char *const kOriginDataType = "originDataType";
const char *const kOriginShape = "originShape";
const char *const kOriginLayout = "originLayout";
const char *const kRealDimCnt = "realDimCnt";
const char *const kCurVarTensorDescMap = "curVarTensorDescMap";
const char *const kTransRoads = "transRoads";
const char *const kTransRoad = "transRoad";
const char *const kNodeType = "nodeType";
const char *const kInputTensorDesc = "inputTensorDesc";
const char *const kOutputTensorDesc = "outputTensorDesc";
const char *const kChangedGraphId = "changedGraphId";
const char *const kAllocatedGraphId = "allocatedGraphId";
const char *const kGraphId = "graphId";
const char *const kVarBroadcastInfo = "varBroadcastInfo";
const char *const kBroadcastName = "broadcastName";
const char *const kIdx = "idx";
const char *const kInputOffset = "inputOffset";
const char *const kInputSize = "inputSize";
const char *const kOutputOffset = "outputOffset";
const char *const kOutputSize = "outputSize";
// Suffix of cache files
const char *const kBeforeVarManagerSuffix = "_before_build_var_manager.json";
const char *const kAfterVarManagerSuffix = "_after_build_var_manager.json";
const char *const kManifestSuffix = ".manifest";
const char *const kOmSuffix = ".om";
}  // namespace

namespace ge {
map<uint32_t, uint32_t> ModelCacheHelper::graph_id_run_times_;
ModelCacheHelper::ModelCacheHelper(uint64_t session_id, uint32_t graph_id, ComputeGraphPtr &compute_graph)
    : session_id_(session_id),
      graph_id_(graph_id),
      compute_graph_(compute_graph),
      is_cache_path_valid_for_output(false) {
  if (graph_id_run_times_.count(graph_id) == 0) {
    graph_id_run_times_[graph_id] = 1;
  } else {
    graph_id_run_times_[graph_id] = graph_id_run_times_[graph_id] + 1;
  }
  for (const auto &node : compute_graph_->GetDirectNode()) {
    bool is_variable = (node->GetType() == VARIABLE) || (node->GetType() == VARIABLEV2) ||
                       (node->GetType() == VARHANDLEOP) || (node->GetType() == CONSTANTOP);
    if (!is_variable) {
      continue;
    }
    var_names_.insert(node->GetName());
  }
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr != nullptr && instance_ptr->IsIncreBuild()) {
    std::string cache_path = instance_ptr->GetIncreBuildCachePath();
    GELOGD("Incre build path conf: %s", cache_path.c_str());
    string fake_file_path = cache_path + to_string(graph_id_) + kManifestSuffix;
    if (CheckOutputPathValid(fake_file_path)) {
      is_cache_path_valid_for_output = true;
    } else {
      GELOGW("Invalid cache path for output.");
    }
    std::string real_cache_path = RealPath(cache_path.c_str());
    if (real_cache_path.empty()) {
      GELOGW("Invalid incre build cache path conf: %s", cache_path.c_str());
      return;
    }
    cache_path_ = real_cache_path + '/';
    GELOGD("Try to use incre build cache path: %s", cache_path_.c_str());
  }
}

ModelCacheHelper::~ModelCacheHelper() { var_names_.clear(); }

bool ModelCacheHelper::IsModelCacheHit() const {
  CacheInfo cache_info;
  if (GetCacheInfo(cache_info) != SUCCESS) {
    GELOGI("Get cache info of graph id[%u] failed.", graph_id_);
    return false;
  }
  // Check number of nodes and edges first.
  if (cache_info.node_num != compute_graph_->GetDirectNodesSize()) {
    GELOGI("Graph id[%u] cache miss: the node number of the graph does not match the cache info.", graph_id_);
    return false;
  }
  size_t edge_num = 0;
  for (const auto &node : compute_graph_->GetDirectNode()) {
    for (const auto &anchor : node->GetAllInAnchors()) {
      edge_num += anchor->GetPeerAnchors().size();
    }
  }
  if (cache_info.edge_num != edge_num) {
    GELOGI("Graph id[%u] cache miss: the edge number of the graph does not match the cache info.", graph_id_);
    return false;
  }
  size_t compute_graph_hash;
  auto ret = GetComputeGraphHash(compute_graph_hash);
  if (ret != SUCCESS || cache_info.graph_hash != compute_graph_hash) {
    GELOGI("Graph id[%u] cache miss: the hash code of the graph does not match the cache info.", graph_id_);
    return false;
  }
  if (!IsNodeHashSameAsCache(cache_info.nodes_hash)) {
    GELOGI("Graph id[%u] cache miss: the hash code of node does not match the cache info.", graph_id_);
    return false;
  }

  string var_manager_cache =
    to_string(graph_id_) + "_" + to_string(graph_id_run_times_[graph_id_]) + kBeforeVarManagerSuffix;
  Json var_manager_json;
  if (LoadJsonFromFile(var_manager_cache, var_manager_json) != SUCCESS) {
    GELOGW("Fail to load json from cache file: %s", var_manager_cache.c_str());
    return false;
  }
  if (!IsVarManagerSameAsCache(var_manager_json)) {
    GELOGI("Graph id[%u] cache miss: the VarManager does not match the cache info.", graph_id_);
    return false;
  }
  GELOGI("Graph id[%u] cache hit.", graph_id_);
  return true;
}

Status ModelCacheHelper::RefreshComputeGraph(const ComputeGraphPtr &compute_graph) {
  if (compute_graph->IsValid()) {
    compute_graph_ = compute_graph;
    var_names_.clear();
    for (const auto &node : compute_graph_->GetDirectNode()) {
      bool is_variable = (node->GetType() == VARIABLE) || (node->GetType() == VARIABLEV2) ||
                         (node->GetType() == VARHANDLEOP) || (node->GetType() == CONSTANTOP);
      if (!is_variable) {
        continue;
      }
      var_names_.insert(node->GetName());
    }
    return SUCCESS;
  } else {
    GELOGW("Invalid compute graph.");
    return FAILED;
  }
}

Status ModelCacheHelper::ClearCache(uint32_t graph_id) const {
  if (!is_cache_path_valid_for_output) {
    GELOGW("Invalid cache path.");
    return SUCCESS;
  }
  string manifest_file = cache_path_ + to_string(graph_id) + kManifestSuffix;
  string manifest_file_path = RealPath(manifest_file.c_str());
  int ret;
  if (!manifest_file_path.empty()) {
    ret = remove(manifest_file_path.c_str());
    // If remove file failed, print the warning log
    if (ret != 0) {
      GELOGW("Clear cache [%s] failed.", manifest_file_path.c_str());
    }
  }
  string before_var_manager_file = cache_path_ + to_string(graph_id) + kManifestSuffix;
  string before_var_manager_file_path = RealPath(before_var_manager_file.c_str());
  if (!before_var_manager_file_path.empty()) {
    ret = remove(before_var_manager_file_path.c_str());
    if (ret != 0) {
      GELOGW("Clear cache [%s] failed.", before_var_manager_file_path.c_str());
    }
  }
  string after_var_manager_file = cache_path_ + to_string(graph_id) + kManifestSuffix;
  string after_var_manager_file_path = RealPath(after_var_manager_file.c_str());
  if (!after_var_manager_file_path.empty()) {
    ret = remove(after_var_manager_file_path.c_str());
    if (ret != 0) {
      GELOGW("Clear cache [%s] failed.", after_var_manager_file_path.c_str());
    }
  }
  string om_file = cache_path_ + to_string(graph_id) + kManifestSuffix;
  string om_file_path = RealPath(om_file.c_str());
  if (!om_file_path.empty()) {
    ret = remove(om_file_path.c_str());
    if (ret != 0) {
      GELOGW("Clear cache [%s] failed.", om_file_path.c_str());
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::RecoverVarManagerFromCache() const {
  string var_manager_cache =
    to_string(graph_id_) + "_" + to_string(graph_id_run_times_[graph_id_]) + kAfterVarManagerSuffix;
  Json var_manager_json;
  if (LoadJsonFromFile(var_manager_cache, var_manager_json) != SUCCESS) {
    GELOGW("Fail to load json from cache file: %s", var_manager_cache.c_str());
    return FAILED;
  }

  Json mem_resource_json = move(var_manager_json[kMemResourceMap]);
  auto ret = RecoverMemResource(mem_resource_json);
  if (ret != SUCCESS) {
    GELOGW("Recover VarManager from cache failed.[MemResource]");
    return FAILED;
  }
  Json var_resource_json = move(var_manager_json[kVarResource]);
  ret = RecoverAllocatedGraphId(var_resource_json[kAllocatedGraphId]);
  if (ret != SUCCESS) {
    GELOGW("Recover VarManager from cache failed.[AllocatedGraphId]");
    return FAILED;
  }
  ret = RecoverChangedGraphId(var_resource_json[kChangedGraphId]);
  if (ret != SUCCESS) {
    GELOGW("Recover VarManager from cache failed.[ChangedGraphId]");
    return FAILED;
  }
  ret = RecoverBroadcastInfo(var_resource_json[kVarBroadcastInfo]);
  if (ret != SUCCESS) {
    GELOGW("Recover VarManager from cache failed.[VarBroadcastInfo]");
    return FAILED;
  }
  ret = RecoverVarAddrAndTensorDesc(var_resource_json[kVarAddrMgrMap]);
  if (ret != SUCCESS) {
    GELOGW("Recover VarManager from cache failed.[VarAddrMgrMap & CurVarTensorDesc]");
    return FAILED;
  }
  ret = RecoverTransRoads(var_resource_json[kTransRoads]);
  if (ret != SUCCESS) {
    GELOGW("Recover VarManager from cache failed.[TransRoads]");
    return FAILED;
  }
  GELOGI("Recover VarManager from cache[%s] success.", cache_path_.c_str());
  return SUCCESS;
}

Status ModelCacheHelper::GetNodesNeedRecompile(ComputeGraphPtr &graph, vector<NodePtr> &nodes) {
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if (instance == nullptr || !instance->InitFlag()) {
    GELOGW("RecompileNodes failed.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }
  // Collect aicore ops for recompile
  for (auto &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    // Get op kernel lib name
    string kernel_lib_name = op_desc->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      // reset op kernel lib
      (void)instance->DNNEngineManagerObj().GetDNNEngineName(node);
      kernel_lib_name = op_desc->GetOpKernelLibName();
      if (kernel_lib_name.empty()) {
        GELOGW("Get node:%s, type:%s kernel lib failed.", node->GetName().c_str(), op_desc->GetType().c_str());
        continue;
      }
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::RecompileNodes(GeModelPtr &ge_model) {
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if (instance == nullptr || !instance->InitFlag()) {
    GELOGW("RecompileNodes failed.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }
  // Get aicore ops kernel info store.
  OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kTbeKernelInfoStoreName);
  if (kernel_info == nullptr) {
    GELOGW("Get %s ops kernel info store failed", kTbeKernelInfoStoreName);
    return INTERNAL_ERROR;
  }

  auto compute_graph = GraphUtils::GetComputeGraph(ge_model->GetGraph());
  vector<NodePtr> node_vec;
  auto ret = GetNodesNeedRecompile(compute_graph, node_vec);
  GE_CHK_BOOL_EXEC_WARN(ret == ge::SUCCESS, return ret, "Get nodes need recompiling failed");
  // Recompile aicore ops
  ret = kernel_info->CompileOp(node_vec);
  GE_CHK_BOOL_EXEC_WARN(ret == ge::SUCCESS, return ret, "Recompile op failed");
  const TBEKernelStore &tbekernel_store = ge_model->GetTBEKernelStore();
  TBEKernelStore tbe_kernel_store;
  for (const ge::NodePtr &n : compute_graph->GetDirectNode()) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    TBEKernelPtr tbe_kernel = node_op_desc->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
    if (tbe_kernel == nullptr) {
      // Load tbe kernel from tbe_kernel_store to op if op was not recompiled
      auto op_desc = n->GetOpDesc();
      tbekernel_store.LoadTBEKernelBinToOpDesc(op_desc);
      GELOGD("LoadOmModelFromCache: Load tbe kernel bin to op desc[%s].", op_desc->GetName().c_str());
    }
    tbe_kernel = node_op_desc->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
    GE_IF_BOOL_EXEC(tbe_kernel == nullptr, continue);
    // Refresh tbe kernel in tbe_kernel_store
    tbe_kernel_store.AddTBEKernel(tbe_kernel);
    GELOGD("Add tbe kernel bin %s", tbe_kernel->GetName().c_str());
  }
  GE_CHK_BOOL_EXEC_WARN(tbe_kernel_store.Build(), return FAILED, "TBE Kernels store build failed!");
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  return SUCCESS;
}

Status ModelCacheHelper::GetNodesHash(map<std::string, size_t> &hash_map) const {
  vector<NodePtr> nodes;
  GraphUtils::TopologicalSortingByName(compute_graph_, nodes);
  ModelSerializeImp model_serialize_imp;
  std::hash<string> node_hash;
  for (const auto &node : nodes) {
    if (node == nullptr) {
      continue;
    }
    proto::OpDef op_def;
    bool is_framework_op = (node->GetType() == FRAMEWORKOP);
    int32_t framework_type = 0;
    if (is_framework_op) {
      AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_FWK_TYPE, framework_type);
      AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_FWK_TYPE, 0);
    }
    bool ret = model_serialize_imp.SerializeNode(node, &op_def, is_framework_op);
    op_def.set_id(0);  // Id of op is not stable because of parallel parsing
    // Clear weights attr in constant.
    auto attr = op_def.mutable_attr();
    if (op_def.type() == CONSTANT || op_def.type() == CONSTANTOP) {
      attr->erase(ATTR_NAME_WEIGHTS);
    }
    if (is_framework_op) {
      AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_FWK_TYPE, framework_type);
    }
    if (!ret) {
      GELOGW("Fail to serialize node[%s].", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    string prototxt;
    ret = google::protobuf::TextFormat::PrintToString(op_def, &prototxt);
    if (!ret) {
      GELOGW("Print OpDef to string failed.");
      hash_map.clear();
      return INTERNAL_ERROR;
    }
    size_t hash_code = node_hash(prototxt);
    hash_map[node->GetName()] = hash_code;
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetComputeGraphHash(size_t &hash) const {
  proto::GraphDef graph_proto;
  ModelSerializeImp model_serialize_imp;
  // The name of compute graph may be generated randomly, so replace it temporarily.
  const string origin_name = compute_graph_->GetName();
  compute_graph_->SetName(kGraphName);
  bool serialize_ret = model_serialize_imp.SerializeGraph(compute_graph_, &graph_proto);
  graph_proto.clear_op();
  if (!serialize_ret) {
    GELOGW("Serialize graph failed.");
    hash = 0;
    return INTERNAL_ERROR;
  }
  compute_graph_->SetName(origin_name);
  // Generate proto text of GraphDef
  string prototxt;
  bool print_ret = google::protobuf::TextFormat::PrintToString(graph_proto, &prototxt);
  if (!print_ret) {
    GELOGW("Print GraphDef to string failed.");
    hash = 0;
    return INTERNAL_ERROR;
  }
  // Get the hash code of proto text
  std::hash<string> graph_hash;
  hash = graph_hash(prototxt);
  return SUCCESS;
}

Status ModelCacheHelper::SaveJsonToFile(const string &file_name, const Json &json) const {
  if (!is_cache_path_valid_for_output) {
    GELOGW("Invalid cache path.");
    return PARAM_INVALID;
  }
  // Check whether the manifest exists, if not, create it.
  string real_path = RealPath(cache_path_.c_str());
  if (real_path.empty()) {
    GELOGW("File path is invalid. please check cache path: %s", cache_path_.c_str());
    return FAILED;
  }
  const string path = cache_path_ + file_name;
  const int FILE_AUTHORITY = 0600;
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, FILE_AUTHORITY);
  if (fd < 0) {
    GELOGW("Fail to open the file: %s.", path.c_str());
    return INTERNAL_ERROR;
  }
  if (close(fd) != 0) {
    GELOGW("Fail to close the file: %s.", path.c_str());
    return INTERNAL_ERROR;
  }

  // Write json into cache file
  ofstream ofs;
  ofs.open(path);
  if (!ofs.is_open()) {
    GELOGW("Fail to open the file: %s.", path.c_str());
    return INTERNAL_ERROR;
  }
  ofs << json << std::endl;
  ofs.close();
  return SUCCESS;
}

Status ModelCacheHelper::LoadJsonFromFile(const string &file_name, Json &json) const {
  if (!json.is_null()) {
    GELOGW("Input param json type should be null.");
    return PARAM_INVALID;
  }
  string real_path = RealPath(cache_path_.c_str());
  if (real_path.empty()) {
    GELOGW("File path is invalid. please check cache path: %s", cache_path_.c_str());
    return FAILED;
  }
  const string path = cache_path_ + file_name;
  if (!CheckInputPathValid(path)) {
    GELOGW("Invalid cache path for input:%s.", path.c_str());
    return FAILED;
  }
  string cache_real_path = RealPath(path.c_str());
  if (cache_real_path.empty()) {
    GELOGI("File[%s] is not found.", path.c_str());
    return FAILED;
  }
  // Read json from cache file
  ifstream ifs;
  ifs.open(path);
  if (!ifs.is_open()) {
    GELOGW("Fail to open the file: %s.", path.c_str());
    return INTERNAL_ERROR;
  }
  try {
    ifs >> json;
  } catch (nlohmann::detail::parse_error e) {
    GELOGW("Fail to load json from file, json throw an error:%s.", e.what());
    return INTERNAL_ERROR;
  } catch (nlohmann::detail::invalid_iterator e) {
    GELOGW("Fail to load json from file, json throw an error:%s.", e.what());
    return INTERNAL_ERROR;
  } catch (nlohmann::detail::type_error e) {
    GELOGW("Fail to load json from file, json throw an error:%s.", e.what());
    return INTERNAL_ERROR;
  } catch (nlohmann::detail::out_of_range e) {
    GELOGW("Fail to load json from file, json throw an error:%s.", e.what());
    return INTERNAL_ERROR;
  } catch (nlohmann::detail::other_error e) {
    GELOGW("Fail to load json from file, json throw an error:%s.", e.what());
    return INTERNAL_ERROR;
  }

  if (!json.is_object()) {
    GELOGW("Fail to load the json file: %s.", path.c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::SaveCacheInfoToCache() const {
  // Generate cache json
  // example: {"edgeNum":6,"nodeNum":7,"graphCache":134714827475991356}
  Json cache_json;
  try {
    cache_json[kNodeNum] = compute_graph_->GetDirectNodesSize();
    size_t edge_num = 0;
    for (const auto &node : compute_graph_->GetDirectNode()) {
      for (const auto &anchor : node->GetAllInAnchors()) {
        edge_num += anchor->GetPeerAnchors().size();
      }
    }
    cache_json[kEdgeNum] = edge_num;
    size_t hash = 0;
    auto ret = GetComputeGraphHash(hash);
    if (ret != SUCCESS) {
      GELOGW("Error occur when generate graph hash code.");
      return ret;
    }
    cache_json[kGraphHash] = hash;
    Json nodes_hash_json;
    ret = GetNodesHashMapJson(nodes_hash_json);
    if (ret != SUCCESS) {
      GELOGW("Error occur when generate nodes hash code.");
      return ret;
    }
    cache_json[kNodeHash] = nodes_hash_json;
  } catch (const std::exception &e) {
    GELOGW("Fail to generate cache info json. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  string cache_manifest = to_string(graph_id_) + "_" + to_string(graph_id_run_times_[graph_id_]) + kManifestSuffix;

  auto ret = SaveJsonToFile(cache_manifest, cache_json);
  if (ret != SUCCESS) {
    GELOGW("Fail to save cache info to json file, path: %s.", cache_path_.c_str());
    return ret;
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetCacheInfo(CacheInfo &cache_info) const {
  string cache_manifest = to_string(graph_id_) + "_" + to_string(graph_id_run_times_[graph_id_]) + kManifestSuffix;
  Json cache_json;
  if (LoadJsonFromFile(cache_manifest, cache_json) != SUCCESS) {
    GELOGW("Fail to load json from cache file: %s", cache_manifest.c_str());
    return INTERNAL_ERROR;
  }
  if (!cache_json.is_object()) {
    GELOGW("Manifest should be a json object");
    return INTERNAL_ERROR;
  }
  try {
    cache_info.node_num = cache_json[kNodeNum];
    cache_info.edge_num = cache_json[kEdgeNum];
    cache_info.graph_hash = cache_json[kGraphHash];
    Json nodes_hash_json = cache_json[kNodeHash];
    if (!(nodes_hash_json.is_null() || nodes_hash_json.is_array())) {
      GELOGW("Nodes hash in cache should be null or array.");
      return FAILED;
    }
    for (const auto &iter : nodes_hash_json) {
      cache_info.nodes_hash[iter[kName].get<std::string>()] = iter[kHash].get<size_t>();
    }
  } catch (const std::exception &e) {
    GELOGW("Fail to get info from json file. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

bool ModelCacheHelper::IsAllocatedGraphIdSameAsCache(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return false;
  }
  // Compare allocated graph id info between json and VarManager
  std::unordered_map<std::string, uint32_t> allocated_graph_id;
  auto ret = ParseAllocatedGraphIdFromJson(json, allocated_graph_id);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse AllocatedGraphId from Json.");
    return false;
  }
  for (const auto &iter : allocated_graph_id) {
    uint32_t graph_id = 0;
    ret = VarManager::Instance(session_id_)->GetAllocatedGraphId(iter.first, graph_id);
    if (ret != SUCCESS) {
      GELOGW("Fail to find allocated graph id of var[%s].", iter.first.c_str());
      return false;
    }
    if (graph_id != iter.second) {
      GELOGW("The allocated graph id of variable[%s] in cache is different from VarManager.", iter.first.c_str());
      return false;
    }
  }
  return true;
}

bool ModelCacheHelper::IsNodeHashSameAsCache(const map<std::string, size_t> &hash_map) const {
  map<std::string, size_t> cur_hash_map;
  GetNodesHash(cur_hash_map);
  if (hash_map.size() != cur_hash_map.size()) {
    GELOGI("The number of hash code is different from cache info.");
    return false;
  }
  for (const auto &iter : cur_hash_map) {
    if (hash_map.count(iter.first) == 0) {
      GELOGI("Node[%s] is not found in cache info.", iter.first.c_str());
      return false;
    }
    if (hash_map.at(iter.first) != iter.second) {
      GELOGI("The hash code of node[%s] is different from cache info.", iter.first.c_str());
      return false;
    }
  }
  return true;
}

bool ModelCacheHelper::IsMemResourceSameAsCache(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return false;
  }
  // Compare var mem size info between json and VarManager
  std::map<rtMemType_t, int64_t> var_mem_size;
  auto ret = ParseMemResourceFromJson(json, var_mem_size);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse MemResource from Json.");
    return false;
  }
  for (const auto &iter : var_mem_size) {
    int64_t mem_size = VarManager::Instance(session_id_)->GetVarMemSize(iter.first);
    if (mem_size != iter.second) {
      GELOGW("The var mem size of memory_type[%u] in cache is different from VarManager.", iter.first);
      return false;
    }
  }
  return true;
}

bool ModelCacheHelper::IsChangedGraphIdSameAsCache(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return false;
  }
  // Compare variable changed graph id info between json and VarManager
  std::unordered_map<std::string, uint32_t> changed_graph_id;
  auto ret = ParseChangedGraphIdFromJson(json, changed_graph_id);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse ChangedGraphId from Json.");
    return false;
  }
  for (const auto &iter : changed_graph_id) {
    uint32_t graph_id = 0;
    ret = VarManager::Instance(session_id_)->GetChangedGraphId(iter.first, graph_id);
    if (ret != SUCCESS) {
      GELOGW("Fail to find changed graph id of var[%s].", iter.first.c_str());
      return false;
    }
    if (graph_id != iter.second) {
      GELOGW("The changed graph id of variable[%s] in cache is different from VarManager.", iter.first.c_str());
      return false;
    }
  }
  return true;
}

bool ModelCacheHelper::IsCurVarTensorDescSameAsCache(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return false;
  }
  // Compare variable tensor desc info between json and VarManager
  std::unordered_map<std::string, ge::GeTensorDesc> cur_var_tensor_desc;
  auto ret = ParseCurVarTensorDescMapFromJson(json, cur_var_tensor_desc);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse CurVarTensorDesc from Json.");
    return false;
  }
  for (const auto &iter : cur_var_tensor_desc) {
    GeTensorDesc tensor_desc;
    ret = VarManager::Instance(session_id_)->GetCurVarDesc(iter.first, tensor_desc);
    if (ret != SUCCESS) {
      GELOGW("Fail to find tensor desc of var[%s].", iter.first.c_str());
      return false;
    }
    uint32_t l_real_dim_cnt = 0;
    uint32_t r_real_dim_cnt = 0;
    TensorUtils::GetRealDimCnt(tensor_desc, l_real_dim_cnt);
    TensorUtils::GetRealDimCnt(iter.second, r_real_dim_cnt);
    if ((tensor_desc.GetDataType() != iter.second.GetDataType()) ||
        (tensor_desc.GetOriginDataType() != iter.second.GetOriginDataType()) ||
        (tensor_desc.GetFormat() != iter.second.GetFormat()) ||
        (tensor_desc.GetOriginFormat() != iter.second.GetOriginFormat()) ||
        (tensor_desc.GetShape().ToString() != iter.second.GetShape().ToString()) ||
        (tensor_desc.GetOriginShape().ToString() != iter.second.GetOriginShape().ToString()) ||
        (l_real_dim_cnt != r_real_dim_cnt)) {
      GELOGW("The var tensor desc of variable[%s] in cache is different from VarManager.", iter.first.c_str());
      return false;
    }
  }
  return true;
}

bool ModelCacheHelper::IsVarAddrMgrMapSameAsCache(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return false;
  }
  // Compare variable address info between json and VarManager
  std::vector<std::pair<std::string, VarAddrMgr>> var_addr_mgr_vector;
  std::unordered_set<uint64_t> var_offset_set;
  auto ret = ParseVarAddrMgrMapFromJson(json, var_addr_mgr_vector, var_offset_set);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse VarAddrMgrMap from Json.");
    return false;
  }
  for (const auto &iter : var_addr_mgr_vector) {
    uint8_t *dev_ptr = nullptr;
    rtMemType_t memory_type;
    ret = VarManager::Instance(session_id_)->GetVarAddr(iter.first, iter.second.tensor_desc, &dev_ptr, memory_type);
    if (ret != SUCCESS) {
      GELOGW("Fail to find tensor desc of var[%s].", iter.first.c_str());
      return false;
    }
    // Compare memory type and logic address
    if (iter.second.memory_type != memory_type || iter.second.address != dev_ptr) {
      GELOGW("The VarAddrMgr of variable[%s] in cache is different from VarManager.", iter.first.c_str());
      return false;
    }
  }
  return true;
}

bool ModelCacheHelper::IsBroadcastInfoSameAsCache(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return false;
  }
  // Compare broadcast info between json and VarManager
  std::unordered_map<std::string, VarBroadCastInfo> var_broadcast_info;
  auto ret = ParseBroadcastInfoFromJson(json, var_broadcast_info);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse BroadcastInfo from Json.");
    return false;
  }
  for (const auto &iter : var_broadcast_info) {
    VarBroadCastInfo broadcast_info;
    if (VarManager::Instance(session_id_)->GetBroadCastInfo(graph_id_, iter.first, broadcast_info) != SUCCESS) {
      GELOGW("Fail to find broadcast info of var[%s].", iter.first.c_str());
      return false;
    }
    if (iter.second.var_name != broadcast_info.var_name || iter.second.idx != broadcast_info.idx ||
        iter.second.input_size != broadcast_info.input_size ||
        iter.second.input_offset != broadcast_info.input_offset ||
        iter.second.output_size != broadcast_info.output_size ||
        iter.second.output_offset != broadcast_info.output_offset) {
      GELOGW("The BroadcastInfo of variable[%s] in cache is different from VarManager.", iter.first.c_str());
      return false;
    }
  }
  return true;
}

bool ModelCacheHelper::IsTransRoadsSameAsCache(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return false;
  }
  // Compare trans road between json and VarManager
  std::unordered_map<std::string, std::vector<TransNodeInfo>> trans_roads;
  auto ret = ParseTransRoadsFromJson(json, trans_roads);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse TransRoads from Json.");
    return false;
  }
  for (const auto &iter : trans_roads) {
    VarTransRoad *trans_road;
    trans_road = VarManager::Instance(session_id_)->GetTransRoad(iter.first);
    if (trans_road == nullptr) {
      GELOGW("Fail to find trans road of var[%s].", iter.first.c_str());
      return false;
    }
    if (trans_road->size() != iter.second.size()) {
      GELOGW("The TransRoad of variable[%s] in cache is different from VarManager.", iter.first.c_str());
      return false;
    }
    // Compare every trans node in trans road.
    for (size_t idx = 0; idx < trans_road->size(); idx += 1) {
      if (!(trans_road->at(idx).node_type == iter.second.at(idx).node_type &&
            trans_road->at(idx).input == iter.second.at(idx).input &&
            trans_road->at(idx).output == iter.second.at(idx).output)) {
        GELOGW("The TransRoad of variable[%s] in cache is different from VarManager.", iter.first.c_str());
        return false;
      }
    }
  }
  return true;
}

bool ModelCacheHelper::IsVarManagerParamSameAsCache(Json &json) const {
  if (!json.is_object()) {
    GELOGW("Input param json type should be object.");
    return false;
  }
  try {
    if (json[kSessionId].get<uint64_t>() != session_id_) {
      GELOGW("Check VarManager cache failed.[sessionId]");
      return false;
    }
    if (json[kDeviceId].get<uint32_t>() != VarManager::Instance(session_id_)->DeviceId()) {
      GELOGW("Check VarManager cache failed.[deviceId]");
      return false;
    }
    if (json[kJobId].get<uint64_t>() != VarManager::Instance(session_id_)->JobId()) {
      GELOGW("Check VarManager cache failed.[jobId]");
      return false;
    }
    if (json[kGraphMemMaxSize].get<size_t>() != VarManager::Instance(session_id_)->GetGraphMemoryMaxSize()) {
      GELOGW("Check VarManager cache failed.[graphMemMaxSize]");
      return false;
    }
    if (json[kVarMemMaxSize].get<size_t>() != VarManager::Instance(session_id_)->GetVarMemMaxSize()) {
      GELOGW("Check VarManager cache failed.[varMemMaxSize]");
      return false;
    }
    if (json[kVarMemLogicBase].get<size_t>() != VarManager::Instance(session_id_)->GetVarMemLogicBase()) {
      GELOGW("Check VarManager cache failed.[varMemLogicBase]");
      return false;
    }
    if (json[kUseMaxMemSize].get<size_t>() != VarManager::Instance(session_id_)->GetUseMaxMemorySize()) {
      GELOGW("Check VarManager cache failed.[useMaxMemSize]");
      return false;
    }
  } catch (const std::exception &e) {
    GELOGW("Fail to check VarManager json. Error message: %s", e.what());
    return false;
  }
  return true;
}

bool ModelCacheHelper::IsVarManagerSameAsCache(Json &json) const {
  if (!json.is_object()) {
    GELOGW("Input param json type should be object.");
    return false;
  }
  try {
    if (!IsVarManagerParamSameAsCache(json)) {
      GELOGW("Check VarManager cache failed.[Param]");
      return false;
    }
    Json mem_resource_json = move(json[kMemResourceMap]);
    auto ret = IsMemResourceSameAsCache(mem_resource_json);
    if (!ret) {
      GELOGW("Check VarManager cache failed.[MemResource]");
      return false;
    }
    Json var_resource_json = move(json[kVarResource]);
    ret = IsAllocatedGraphIdSameAsCache(var_resource_json[kAllocatedGraphId]);
    if (!ret) {
      GELOGW("Check VarManager cache failed.[AllocatedGraphId]");
      return false;
    }
    ret = IsChangedGraphIdSameAsCache(var_resource_json[kChangedGraphId]);
    if (!ret) {
      GELOGW("Check VarManager cache failed.[ChangedGraphId]");
      return false;
    }
    ret = IsBroadcastInfoSameAsCache(var_resource_json[kVarBroadcastInfo]);
    if (!ret) {
      GELOGW("Check VarManager cache failed.[VarBroadcastInfo]");
      return false;
    }
    ret = IsCurVarTensorDescSameAsCache(var_resource_json[kCurVarTensorDescMap]);
    if (!ret) {
      GELOGW("Check VarManager cache failed.[CurVarTensorDesc]");
      return false;
    }
    ret = IsVarAddrMgrMapSameAsCache(var_resource_json[kVarAddrMgrMap]);
    if (!ret) {
      GELOGW("Check VarManager cache failed.[VarAddrMgrMap]");
      return false;
    }
    ret = IsTransRoadsSameAsCache(var_resource_json[kTransRoads]);
    if (!ret) {
      GELOGW("Check VarManager cache failed.[TransRoads]");
      return false;
    }
  } catch (const std::exception &e) {
    GELOGW("Fail to check VarManager json. Error message: %s", e.what());
    return false;
  }
  return true;
}

Status ModelCacheHelper::RecoverMemResource(const Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  std::map<rtMemType_t, int64_t> var_mem_size;
  auto ret = ParseMemResourceFromJson(json, var_mem_size);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse MemResource from Json.");
    return ret;
  }
  for (const auto &iter : var_mem_size) {
    ret = VarManager::Instance(session_id_)->UpdateVarMemSize(iter.first, iter.second);
    if (ret != SUCCESS) {
      GELOGW("Fail to recover var mem size.");
      return ret;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::RecoverAllocatedGraphId(const Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  std::unordered_map<std::string, uint32_t> allocated_graph_id;
  auto ret = ParseAllocatedGraphIdFromJson(json, allocated_graph_id);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse AllocatedGraphId from Json.");
    return ret;
  }
  for (const auto &iter : allocated_graph_id) {
    ret = VarManager::Instance(session_id_)->SetAllocatedGraphId(iter.first, iter.second);
    if (ret != SUCCESS) {
      GELOGW("Fail to recover allocated graph id.");
      return ret;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::RecoverChangedGraphId(const Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  std::unordered_map<std::string, uint32_t> changed_graph_id;
  auto ret = ParseChangedGraphIdFromJson(json, changed_graph_id);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse AllocatedGraphId from Json.");
    return ret;
  }
  for (const auto &iter : changed_graph_id) {
    ret = VarManager::Instance(session_id_)->SetChangedGraphId(iter.first, iter.second);
    if (ret != SUCCESS) {
      GELOGW("Fail to recover changed graph id.");
      return ret;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::RecoverVarAddrAndTensorDesc(const Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  std::vector<std::pair<std::string, VarAddrMgr>> var_addr_mgr_vector;
  std::unordered_set<uint64_t> var_offset_set;
  auto ret = ParseVarAddrMgrMapFromJson(json, var_addr_mgr_vector, var_offset_set);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse VarAddrMgrMap from Json.");
    return ret;
  }
  for (const auto &iter : var_addr_mgr_vector) {
    const VarAddrMgr &tensor_addr_mgr = iter.second;
    const bool var_exist = VarManager::Instance(session_id_)->IsVarExist(iter.first, tensor_addr_mgr.tensor_desc);
    // SaveVarVddr if var does not exist, the logic address will be recorded by VarManager
    if (!var_exist) {
      auto logic_address = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_addr_mgr.address));
      auto offset = (tensor_addr_mgr.offset);
      // Check logic address and offset
      if (logic_address - offset != VarManager::Instance(session_id_)->GetVarMemLogicBase()) {
        GELOGW("Check logic_address[%u] and offset [%u] of %s failed, var mem logic base is %u, abandon", logic_address,
               offset, iter.first.c_str(), VarManager::Instance(session_id_)->GetVarMemLogicBase());
        return PARAM_INVALID;
      }
      // Offset is needed by SaveVarVddr instead of logic address
      ret =
        VarManager::Instance(session_id_)
          ->SaveVarAddr(iter.first, tensor_addr_mgr.tensor_desc,
                        reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(offset)), tensor_addr_mgr.memory_type);
      if (ret != SUCCESS) {
        GELOGW("Fail to recover VarAddr or TensorDesc of var[%s].", iter.first.c_str());
        return ret;
      }
    }
    // SetVarAddr to update cur_var_tensor_desc_map_
    ret = VarManager::Instance(session_id_)
            ->SetVarAddr(iter.first, tensor_addr_mgr.tensor_desc, tensor_addr_mgr.address, tensor_addr_mgr.memory_type);
    if (ret != SUCCESS) {
      GELOGW("Fail to recover VarAddr or TensorDesc desc of var[%s].", iter.first.c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::RecoverBroadcastInfo(const Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  std::unordered_map<std::string, VarBroadCastInfo> var_broadcast_info;
  auto ret = ParseBroadcastInfoFromJson(json, var_broadcast_info);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse BroadcastInfo from Json.");
    return ret;
  }
  for (const auto &iter : var_broadcast_info) {
    VarBroadCastInfo broadcast_info;
    ret = VarManager::Instance(session_id_)->SaveBroadCastInfo(graph_id_, iter.second);
    if (ret != SUCCESS) {
      GELOGW("Fail to recover broadcast info of var[%s].", iter.first.c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::RecoverTransRoads(const Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  std::unordered_map<std::string, std::vector<TransNodeInfo>> trans_roads;
  auto ret = ParseTransRoadsFromJson(json, trans_roads);
  if (ret != SUCCESS) {
    GELOGW("Fail to parse TransRoads from Json.");
    return ret;
  }
  for (const auto &iter : trans_roads) {
    ret = VarManager::Instance(session_id_)->SetTransRoad(iter.first, iter.second);
    if (ret != SUCCESS) {
      GELOGW("Fail to find trans road of var[%s].", iter.first.c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::TensorDescToJson(const GeTensorDesc &ge_tensor_desc, Json &json) {
  if (!(json.is_null() || json.is_object())) {
    GELOGW("Input param json type should be null or object.");
    return PARAM_INVALID;
  }
  try {
    json[kDataType] = static_cast<int>(ge_tensor_desc.GetDataType());
    json[kOriginDataType] = static_cast<int>(ge_tensor_desc.GetOriginDataType());
    json[kLayout] = static_cast<int>(ge_tensor_desc.GetFormat());
    json[kOriginLayout] = static_cast<int>(ge_tensor_desc.GetOriginFormat());
    json[kShape] = ge_tensor_desc.GetShape().GetDims();
    json[kOriginShape] = ge_tensor_desc.GetOriginShape().GetDims();
    uint32_t real_dim_cnt = 0;
    (void)TensorUtils::GetRealDimCnt(ge_tensor_desc, real_dim_cnt);  // [No need to check value]
    json[kRealDimCnt] = real_dim_cnt;
  } catch (const std::exception &e) {
    GELOGW("Fail to trans GeTensorDesc to json. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::JsonToTensorDesc(const Json &json, ge::GeTensorDesc &ge_tensor_desc) {
  if (!json.is_object()) {
    GELOGW("Input param json type should be object.");
    return PARAM_INVALID;
  }
  try {
    ge_tensor_desc.SetDataType(static_cast<DataType>(json[kDataType].get<int>()));
    ge_tensor_desc.SetOriginDataType(static_cast<DataType>(json[kOriginDataType].get<int>()));
    ge_tensor_desc.SetFormat(static_cast<Format>(json[kLayout].get<int>()));
    ge_tensor_desc.SetOriginFormat(static_cast<Format>(json[kOriginLayout].get<int>()));
    GeShape shape(json[kShape].get<std::vector<int64_t>>());
    ge_tensor_desc.SetShape(shape);
    GeShape origin_shape(json[kOriginShape].get<std::vector<int64_t>>());
    ge_tensor_desc.SetOriginShape(origin_shape);
    auto real_dim_cnt = json[kRealDimCnt].get<uint32_t>();
    (void)TensorUtils::SetRealDimCnt(ge_tensor_desc, real_dim_cnt);  // [No need to check value]
  } catch (const std::exception &e) {
    GELOGW("Fail to trans Json to GeTensorDesc. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetNodesHashMapJson(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  map<std::string, size_t> hash_map;
  GetNodesHash(hash_map);
  for (const auto &iter : hash_map) {
    Json node_hash_json;
    try {
      node_hash_json[kName] = iter.first;
      node_hash_json[kHash] = iter.second;
      json.emplace_back(move(node_hash_json));
    } catch (const std::exception &e) {
      GELOGW("Fail to trans node cache to json. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetMemResourceMap(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  const auto total_size = VarManager::Instance(session_id_)->GetVarMemMaxSize();
  const auto var_mem_size = VarManager::Instance(session_id_)->GetVarMemSize(RT_MEMORY_HBM);
  Json mem_resource_json;
  try {
    mem_resource_json[kMemType] = RT_MEMORY_HBM;
    mem_resource_json[kTotalSize] = total_size;
    mem_resource_json[kVarMemSize] = var_mem_size;
    json.emplace_back(move(mem_resource_json));
  } catch (const std::exception &e) {
    GELOGW("Fail to trans MemResourceMap to json. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetVarAddrMgrMapJson(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  std::unordered_map<std::string, VarAddrMgr> var_addr_mgr_map;
  VarManager::Instance(session_id_)->GetAllVarAddrMgr(var_addr_mgr_map);
  try {
    for (const auto &iter : var_addr_mgr_map) {
      Json var_addr_json;
      string name;
      GetVarNameFromVarKey(iter.first, iter.second.tensor_desc, name);
      var_addr_json[kName] = name;
      var_addr_json[kAddress] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(iter.second.address));
      var_addr_json[kMemoryType] = iter.second.memory_type;
      var_addr_json[kOffset] = iter.second.offset;

      // Copy tensor desc to json.
      Json tensor_desc_json;
      auto ret = TensorDescToJson(iter.second.tensor_desc, tensor_desc_json);
      if (ret != SUCCESS) {
        GELOGW("Fail to trans tensor desc to json.");
        return INTERNAL_ERROR;
      }
      var_addr_json[kTensorDesc] = move(tensor_desc_json);

      json.emplace_back(move(var_addr_json));
    }
  } catch (const std::exception &e) {
    GELOGW("Fail to trans VarAddrMgrMap to json. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetCurVarTensorDescMapJson(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  try {
    for (const auto &name : var_names_) {
      Json cur_tensor_desc_json;
      GeTensorDesc tensor_desc;
      auto ret = VarManager::Instance(session_id_)->GetCurVarDesc(name, tensor_desc);
      if (ret != SUCCESS) {
        GELOGI("Get variable[%s] current tensor desc failed. It will be skipped.", name.c_str());
        continue;
      }
      cur_tensor_desc_json[kName] = name;

      Json tensor_desc_json;
      ret = TensorDescToJson(tensor_desc, tensor_desc_json);
      if (ret != SUCCESS) {
        GELOGW("Fail to trans tensor desc to json.");
        return INTERNAL_ERROR;
      }
      cur_tensor_desc_json[kTensorDesc] = move(tensor_desc_json);
      json.emplace_back(move(cur_tensor_desc_json));
    }
  } catch (const std::exception &e) {
    GELOGW("Fail to trans CurVarTensorDescMap to json. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetTransRoadsJson(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  try {
    for (const auto &name : var_names_) {
      auto trans_road = VarManager::Instance(session_id_)->GetTransRoad(name);
      if (trans_road == nullptr) {
        continue;
      }
      // Json object, variable name and trans road
      Json trans_road_map_json;
      trans_road_map_json[kName] = name;

      Json trans_road_json;
      Status ret;
      // Add nodes' info to json
      for (const auto &trans_node_info : *trans_road) {
        Json trans_node_info_json;
        trans_node_info_json[kNodeType] = trans_node_info.node_type;
        Json input_tensor_desc_json;
        ret = TensorDescToJson(trans_node_info.input, input_tensor_desc_json);
        if (ret != SUCCESS) {
          GELOGW("Fail to trans tensor desc to json.");
          return INTERNAL_ERROR;
        }
        trans_node_info_json[kInputTensorDesc] = move(input_tensor_desc_json);
        Json output_tensor_desc_json;
        ret = TensorDescToJson(trans_node_info.output, output_tensor_desc_json);
        if (ret != SUCCESS) {
          GELOGW("Fail to trans tensor desc to json.");
          return INTERNAL_ERROR;
        }
        trans_node_info_json[kOutputTensorDesc] = move(output_tensor_desc_json);
        trans_road_json.emplace_back(move(trans_node_info_json));
      }
      trans_road_map_json[kTransRoad] = move(trans_road_json);
      json.emplace_back(move(trans_road_map_json));
    }
  } catch (const std::exception &e) {
    GELOGW("Fail to trans VarToTransRoad to json. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetChangedGraphIdJson(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  for (const auto &name : var_names_) {
    uint32_t changed_graph_id = 0;
    Status ret = VarManager::Instance(session_id_)->GetChangedGraphId(name, changed_graph_id);
    if (ret != SUCCESS) {
      continue;
    }
    Json name_and_changed_graph_id;
    try {
      name_and_changed_graph_id[kName] = name;
      name_and_changed_graph_id[kGraphId] = changed_graph_id;
      json.emplace_back(move(name_and_changed_graph_id));
    } catch (const std::exception &e) {
      GELOGW("Fail to trans ChangedGraphId to json. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetAllocatedGraphIdJson(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  for (const auto &name : var_names_) {
    uint32_t allocated_graph_id = 0;
    Status ret = VarManager::Instance(session_id_)->GetAllocatedGraphId(name, allocated_graph_id);
    if (ret != SUCCESS) {
      continue;
    }
    Json name_and_allocated_graph_id;
    try {
      name_and_allocated_graph_id[kName] = name;
      name_and_allocated_graph_id[kGraphId] = allocated_graph_id;
      json.emplace_back(move(name_and_allocated_graph_id));
    } catch (const std::exception &e) {
      GELOGW("Fail to trans AllocatedGraphId to json. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetBroadcastInfoJson(Json &json) const {
  if (!(json.is_null() || json.is_array())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  for (const auto &name : var_names_) {
    VarBroadCastInfo var_broadcast_info;
    Status ret = VarManager::Instance(session_id_)->GetBroadCastInfo(graph_id_, name, var_broadcast_info);
    if (ret != SUCCESS) {
      continue;
    }
    Json var_broadcast_info_json;
    try {
      var_broadcast_info_json[kName] = name;
      var_broadcast_info_json[kBroadcastName] = var_broadcast_info.broadcast_name;
      var_broadcast_info_json[kIdx] = var_broadcast_info.idx;
      var_broadcast_info_json[kInputOffset] = var_broadcast_info.input_offset;
      var_broadcast_info_json[kInputSize] = var_broadcast_info.input_size;
      var_broadcast_info_json[kOutputOffset] = var_broadcast_info.output_offset;
      var_broadcast_info_json[kOutputSize] = var_broadcast_info.output_size;
      json.emplace_back(move(var_broadcast_info_json));
    } catch (const std::exception &e) {
      GELOGW("Fail to trans VarBroadcastInfo to json. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetVarResourceJson(Json &json) const {
  if (!(json.is_null() || json.is_object())) {
    GELOGW("Input param json type should be null or object.");
    return PARAM_INVALID;
  }
  Json var_addr_mgr_map_json;
  Status ret = GetVarAddrMgrMapJson(var_addr_mgr_map_json);
  if (ret != SUCCESS) {
    GELOGW("GetVarAddrMgrMapJson failed.");
    return INTERNAL_ERROR;
  }

  Json cur_var_tensor_desc_map_json;
  ret = GetCurVarTensorDescMapJson(cur_var_tensor_desc_map_json);
  if (ret != SUCCESS) {
    GELOGW("GetCurVarTensorDescMapJson failed.");
    return INTERNAL_ERROR;
  }

  Json trans_roads_json;
  ret = GetTransRoadsJson(trans_roads_json);
  if (ret != SUCCESS) {
    GELOGW("GetTransRoadsJson failed.");
    return INTERNAL_ERROR;
  }

  Json changed_graph_id_json;
  ret = GetChangedGraphIdJson(changed_graph_id_json);
  if (ret != SUCCESS) {
    GELOGW("GetChangedGraphIdJson failed.");
    return INTERNAL_ERROR;
  }

  Json allocated_graph_id_json;
  ret = GetAllocatedGraphIdJson(allocated_graph_id_json);
  if (ret != SUCCESS) {
    GELOGW("GetAllocatedGraphIdJson failed.");
    return INTERNAL_ERROR;
  }

  Json var_broadcast_info_json;
  ret = GetBroadcastInfoJson(var_broadcast_info_json);
  if (ret != SUCCESS) {
    GELOGW("GetBroadcastInfoJson failed.");
    return INTERNAL_ERROR;
  }

  try {
    json[kVarAddrMgrMap] = move(var_addr_mgr_map_json);
    json[kCurVarTensorDescMap] = move(cur_var_tensor_desc_map_json);
    json[kTransRoads] = move(trans_roads_json);
    json[kChangedGraphId] = move(changed_graph_id_json);
    json[kAllocatedGraphId] = move(allocated_graph_id_json);
    json[kVarBroadcastInfo] = move(var_broadcast_info_json);
  } catch (const exception &e) {
    GELOGW("Fail to generate VarResource json. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetVarManagerJson(Json &json) const {
  if (!(json.is_null() || json.is_object())) {
    GELOGW("Input param json type should be null or object.");
    return PARAM_INVALID;
  }

  Json mem_resource_map_json;
  auto ret = GetMemResourceMap(mem_resource_map_json);
  if (ret != SUCCESS) {
    GELOGW("GetMemResourceMap failed.");
    return INTERNAL_ERROR;
  }

  Json var_resource_json;
  ret = GetVarResourceJson(var_resource_json);
  if (ret != SUCCESS) {
    GELOGW("GetVarResourceJson failed.");
    return INTERNAL_ERROR;
  }

  try {
    json[kSessionId] = session_id_;
    json[kDeviceId] = VarManager::Instance(session_id_)->DeviceId();
    json[kJobId] = VarManager::Instance(session_id_)->JobId();
    json[kGraphMemMaxSize] = VarManager::Instance(session_id_)->GetGraphMemoryMaxSize();
    json[kVarMemMaxSize] = VarManager::Instance(session_id_)->GetVarMemMaxSize();
    json[kVarMemLogicBase] = VarManager::Instance(session_id_)->GetVarMemLogicBase();
    json[kUseMaxMemSize] = VarManager::Instance(session_id_)->GetUseMaxMemorySize();
    json[kMemResourceMap] = move(mem_resource_map_json);
    json[kVarResource] = move(var_resource_json);
  } catch (const exception &e) {
    GELOGW("Fail to generate VarManager json. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::SaveVarManagerToCache(bool before_build) const {
  if (!is_cache_path_valid_for_output) {
    GELOGW("Invalid cache path.");
    return FAILED;
  }
  Json var_manager_json;
  auto ret = GetVarManagerJson(var_manager_json);
  if (ret != SUCCESS) {
    GELOGW("Fail to generate VarManager json.");
    return FAILED;
  }
  string var_manager_path = to_string(graph_id_) + "_" + to_string(graph_id_run_times_[graph_id_]) +
                            (before_build ? kBeforeVarManagerSuffix : kAfterVarManagerSuffix);
  ret = SaveJsonToFile(var_manager_path, var_manager_json);
  if (ret != SUCCESS) {
    GELOGW("Fail to save VarManager info to json file, path: %s.", cache_path_.c_str());
    return ret;
  }
  return SUCCESS;
}

Status ModelCacheHelper::SaveOmModelToCache(const GeModelPtr &ge_model) const {
  if (!is_cache_path_valid_for_output) {
    GELOGW("Invalid cache path.");
    return FAILED;
  }
  string om_path = RealPath(cache_path_.c_str());
  if (om_path.empty()) {
    GELOGW("file path is invalid. please check path om: %s", cache_path_.c_str());
    return FAILED;
  }
  string cache_om_path = cache_path_;
  cache_om_path += (to_string(graph_id_) + "_" + to_string(graph_id_run_times_[graph_id_]) + kOmSuffix);
  GELOGI("SaveOmModelToCache: start to save om model : %s", cache_om_path.c_str());
  ModelHelper model_helper;
  SaveParam save_param;
  ModelBufferData model;
  Status ret = model_helper.SaveToOmModel(ge_model, save_param, cache_om_path, model);
  if (ret != SUCCESS) {
    GELOGW("SaveOmModelToCache: save mode failed. ret = %u", ret);
    return ret;
  }
  return SUCCESS;
}

Status ModelCacheHelper::ParseMemResourceFromJson(const Json &json, map<rtMemType_t, int64_t> &mem_resource) {
  if (!(json.is_array() || json.is_null())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  mem_resource.clear();
  for (const Json &mem_resource_json : json) {
    try {
      rtMemType_t mem_type = mem_resource_json[kMemType].get<rtMemType_t>();
      uint64_t var_mem_size = mem_resource_json[kVarMemSize].get<int64_t>();
      mem_resource[mem_type] = var_mem_size;
    } catch (const exception &e) {
      GELOGW("Fail to trans Json to MemResource. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::ParseVarAddrMgrMapFromJson(
  const Json &json, std::vector<std::pair<std::string, VarAddrMgr>> &var_addr_mgr_vector,
  std::unordered_set<uint64_t> &var_offset_set) {
  if (!(json.is_array() || json.is_null())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  var_addr_mgr_vector.clear();
  var_offset_set.clear();
  for (const Json &var_addr_json : json) {
    VarAddrMgr var_addr_mgr;
    try {
      auto logic_address = var_addr_json[kAddress].get<uint64_t>();
      auto address = reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(logic_address));
      var_addr_mgr.address = address;
      var_addr_mgr.offset = var_addr_json[kOffset].get<uint64_t>();
      var_addr_mgr.memory_type = var_addr_json[kMemoryType].get<rtMemType_t>();
      auto ret = JsonToTensorDesc(var_addr_json[kTensorDesc], var_addr_mgr.tensor_desc);
      if (ret != SUCCESS) {
        GELOGW("Fail to trans json to tensor desc.");
        return ret;
      }
      var_addr_mgr_vector.emplace_back(var_addr_json[kName].get<string>(), move(var_addr_mgr));
      var_offset_set.insert(logic_address);
    } catch (const exception &e) {
      GELOGW("Fail to trans Json to VarAddrMgr. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::ParseCurVarTensorDescMapFromJson(
  const Json &json, std::unordered_map<std::string, ge::GeTensorDesc> &cur_var_tensor_desc_map) {
  if (!(json.is_array() || json.is_null())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  cur_var_tensor_desc_map.clear();
  for (const Json &tensor_desc_json : json) {
    GeTensorDesc tensor_desc;
    try {
      auto ret = JsonToTensorDesc(tensor_desc_json[kTensorDesc], tensor_desc);
      if (ret != SUCCESS) {
        GELOGW("Fail to trans json to tensor desc.");
        return ret;
      }
      cur_var_tensor_desc_map[tensor_desc_json[kName].get<string>()] = move(tensor_desc);
    } catch (const exception &e) {
      GELOGW("Fail to trans Json to VarAddrMgr. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::ParseTransRoadsFromJson(
  const Json &json, std::unordered_map<std::string, std::vector<TransNodeInfo>> &trans_roads) {
  if (!(json.is_array() || json.is_null())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  trans_roads.clear();
  try {
    for (const Json &name_trans_road_json : json) {
      const Json &trans_road_json = name_trans_road_json[kTransRoad];
      if (!(trans_road_json.is_array() || trans_road_json.is_null())) {
        GELOGW("%s json type should be null or object.", kTransRoad);
        return PARAM_INVALID;
      }
      vector<TransNodeInfo> trans_road;
      for (const Json &trans_node_json : trans_road_json) {
        TransNodeInfo trans_node_info;
        trans_node_info.node_type = trans_node_json[kNodeType];
        GeTensorDesc input_tensor_desc;
        auto ret = JsonToTensorDesc(trans_node_json[kInputTensorDesc], input_tensor_desc);
        if (ret != SUCCESS) {
          GELOGW("Fail to trans json to tensor desc.");
          return ret;
        }
        trans_node_info.input = move(input_tensor_desc);
        GeTensorDesc output_tensor_desc;
        ret = JsonToTensorDesc(trans_node_json[kOutputTensorDesc], output_tensor_desc);
        if (ret != SUCCESS) {
          GELOGW("Fail to trans json to tensor desc.");
          return ret;
        }
        trans_node_info.output = move(output_tensor_desc);
        trans_road.emplace_back(move(trans_node_info));
      }
      trans_roads[name_trans_road_json[kName].get<string>()] = move(trans_road);
    }
  } catch (const exception &e) {
    GELOGW("Fail to trans Json to TransRoads. Error message: %s", e.what());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status ModelCacheHelper::ParseChangedGraphIdFromJson(const Json &json,
                                                     std::unordered_map<std::string, uint32_t> &changed_graph_id) {
  if (!(json.is_array() || json.is_null())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  changed_graph_id.clear();
  for (const Json &name_graph_id_json : json) {
    try {
      changed_graph_id[name_graph_id_json[kName].get<string>()] = name_graph_id_json[kGraphId].get<uint32_t>();
    } catch (const exception &e) {
      GELOGW("Fail to trans Json to changed graph id. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::ParseAllocatedGraphIdFromJson(const Json &json,
                                                       std::unordered_map<std::string, uint32_t> &allocated_graph_id) {
  if (!(json.is_array() || json.is_null())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  allocated_graph_id.clear();
  for (const Json &name_graph_id_json : json) {
    try {
      allocated_graph_id[name_graph_id_json[kName].get<string>()] = name_graph_id_json[kGraphId].get<uint32_t>();
    } catch (const exception &e) {
      GELOGW("Fail to trans Json to allocated graph id. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelCacheHelper::ParseBroadcastInfoFromJson(
  const Json &json, std::unordered_map<std::string, VarBroadCastInfo> &var_broadcast_info) {
  if (!(json.is_array() || json.is_null())) {
    GELOGW("Input param json type should be null or array.");
    return PARAM_INVALID;
  }
  for (const Json &broadcast_info_json : json) {
    VarBroadCastInfo broadcast_info;
    try {
      broadcast_info.var_name = broadcast_info_json[kName].get<string>();
      broadcast_info.broadcast_name = broadcast_info_json[kBroadcastName].get<string>();
      broadcast_info.idx = broadcast_info_json[kIdx].get<int>();
      broadcast_info.input_offset = broadcast_info_json[kInputOffset].get<int64_t>();
      broadcast_info.input_size = broadcast_info_json[kInputSize].get<uint64_t>();
      broadcast_info.output_offset = broadcast_info_json[kOutputOffset].get<int64_t>();
      broadcast_info.output_size = broadcast_info_json[kOutputSize].get<uint64_t>();
    } catch (const exception &e) {
      GELOGW("Fail to trans Json to VarBroadCastInfo. Error message: %s", e.what());
      return INTERNAL_ERROR;
    }
    var_broadcast_info[broadcast_info.var_name] = broadcast_info;
  }
  return SUCCESS;
}

Status ModelCacheHelper::LoadOmModelFromCache(GeModelPtr &ge_model) const {
  string cache_om = cache_path_ + to_string(graph_id_) + "_" + to_string(graph_id_run_times_[graph_id_]) + kOmSuffix;
  if (!CheckInputPathValid(cache_om)) {
    GELOGW("Invalid cache path for input:%s.", cache_om.c_str());
    return FAILED;
  }
  string om_path = RealPath(cache_om.c_str());
  if (om_path.empty()) {
    GELOGW("file path is invalid. please check file om: %s", om_path.c_str());
    return FAILED;
  }
  GELOGI("load model data from file: %s", om_path.c_str());
  Status ret;
  string key_path;
  int32_t priority = 0;
  ModelData model_data;
  ret = DavinciModelParser::LoadFromFile(om_path.c_str(), key_path.c_str(), priority, model_data);
  if (ret != SUCCESS) {
    GELOGW("LoadOmModelFromCache: Load model from file failed. ret = %u", ret);
    return ret;
  }

  ModelHelper model_helper;
  ret = model_helper.LoadModel(model_data);
  if (ret != SUCCESS) {
    GELOGW("LoadOmModelFromCache: Load model from data failed. ret = %u", ret);
    return ret;
  }
  ge_model = model_helper.GetGeModel();
  ret = RecompileNodes(ge_model);
  if (ret != SUCCESS) {
    GELOGW("LoadOmModelFromCache: recompile nodes failed. ret = %u", ret);
    return ret;
  }
  return SUCCESS;
}

Status ModelCacheHelper::GetVarNameFromVarKey(const string &var_key, const GeTensorDesc &tensor_desc,
                                              string &var_name) {
  std::string::size_type underline_idx = var_key.rfind('_');
  if (underline_idx == std::string::npos) {
    GELOGW("Invalid var key: underline not found");
    return FAILED;
  }
  std::string::size_type format_idx =
    var_key.rfind(std::to_string(static_cast<int32_t>(tensor_desc.GetFormat())), underline_idx);
  if (format_idx == std::string::npos) {
    GELOGW("Invalid var key: format not found");
    return FAILED;
  }
  var_name = var_key.substr(0, format_idx);
  return SUCCESS;
}
}  // namespace ge
