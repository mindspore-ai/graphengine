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

#ifndef GE_GRAPH_MANAGER_GRAPH_VAR_MANAGER_H_
#define GE_GRAPH_MANAGER_GRAPH_VAR_MANAGER_H_

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "framework/common/util.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "external/graph/tensor.h"
#include "runtime/mem.h"
#include "graph/manager/memory_manager.h"
#include "proto/var_manager.pb.h"
#include "graph/ge_local_context.h"

namespace ge {
constexpr uint64_t kGraphMemoryManagerMallocMaxSize = 27917287424U; // 26UL * 1024UL * 1024UL * 1024UL;
constexpr uint64_t kMemoryVarManagerMallocSize = 5368709120U; // 5UL * 1024UL * 1024UL * 1024UL;
constexpr uint64_t kMemoryVarLogicBase = 34359738368U; // 32UL * 1024UL * 1024UL * 1024UL;
constexpr uint64_t kMemoryHostFeatureMapLogicBase = 68719476736U; // 64UL * 1024UL * 1024UL * 1024UL;
constexpr uint64_t kMemoryVarAddressSize = kMemoryHostFeatureMapLogicBase - kMemoryVarLogicBase;
constexpr uint64_t kMemoryHostSVMFeatureMapLogicBase = 137438953472U; // 128UL * 1024UL * 1024UL * 1024UL;
constexpr uint64_t kUseMaxMemorySize = kGraphMemoryManagerMallocMaxSize + kMemoryVarManagerMallocSize;
constexpr uint64_t kGraphMemoryBuffer = 34359738368U; // 32UL * 1024UL * 1024UL * 1024UL;
constexpr uint64_t kMaxMemorySize = 274877906944U; // 256UL * 1024UL * 1024UL * 1024UL;
constexpr uint64_t kVarMemoryLogicBase = 137438953472U; // 128UL * 1024UL * 1024UL * 1024UL;
constexpr char_t kEnvGeuseStaticMemory[] = "GE_USE_STATIC_MEMORY";
constexpr uint64_t kSessionMemAlignSize = 512U;
constexpr size_t kSessionMemAlignUnit = 2U;
constexpr float64_t kGraphMemoryManagerMallocRatio = 26.0 / 32.0;
constexpr float64_t kVarMemoryManagerMallocRatio = 5.0 / 32.0;
constexpr float64_t kMaxMemorySizeRatio = (26.0 + 5.0) / 32.0;
constexpr uint32_t kDefaultDeviceId = 0U;
constexpr char_t kFeatureMemoryKey[] = "0_f";

enum class SessionVersion : std::int32_t
{
  ClOUD_VERSION = 0,
  MINI_VERSION = 1,
  OTHER_VERSION = 2,
};

struct VarAddrMgr {
  ge::GeTensorDesc tensor_desc;
  const uint8_t *address;
  uint64_t offset;
  rtMemType_t memory_type;
  OpDescPtr op_desc;
};

struct VarDevAddrMgr {
  ge::GeTensorDesc tensor_desc;
  const uint8_t *logic_addr;
  uint8_t *dev_addr;
};

struct VarBroadCastInfo {
  std::string var_name;
  std::string broadcast_name;
  int32_t idx;
  int64_t input_offset;
  uint64_t input_size;
  int64_t output_offset;
  uint64_t output_size;
};

struct TransNodeInfo {
  std::string node_type;
  GeTensorDesc input;
  GeTensorDesc output;
};

using VarTransRoad = std::vector<TransNodeInfo>;

class VarResource {
 public:
  explicit VarResource(const uint64_t session_id);
  ~VarResource();

  ge::Status GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **const dev_ptr,
                        rtMemType_t &memory_type) const;

  Status GetFileConstantReuseAddr(const OpDescPtr &op_desc, uint8_t **const dev_ptr, rtMemType_t &memory_type) const;

  Status GetReuseAddr(const OpDescPtr &op_desc, uint8_t **const dev_ptr, rtMemType_t &memory_type) const;

  void SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, const uint8_t *const dev_ptr,
                  const rtMemType_t memory_type, const OpDescPtr &op_desc);

  ge::Status SaveVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, const uint8_t *const address,
                         const rtMemType_t memory_type, const OpDescPtr &op_desc);

  ge::Status GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc);

  ge::Status RecordStagedVarDesc(const uint32_t graph_id, const std::string &var_name, const GeTensorDesc &tensor_desc);

  const std::map<std::string, GeTensorDesc> &GetStagedVarDescs(const uint32_t graph_id) const;

  ge::Status RenewCurVarDesc(const std::string &var_name, const ge::OpDescPtr &op_desc);

  Status RenewCurVarDesc(const std::string &var_name, const GeTensorDesc &tensor_desc);

  void SaveBroadCastInfo(const uint32_t graph_id, const VarBroadCastInfo &broad_cast_info);

  Status SetTransRoad(const std::string &var_name, const VarTransRoad &trans_road) {
    const std::string &batch_var_name = GetBatchVarKeyName(var_name);
    if (var_to_trans_road_.find(batch_var_name) != var_to_trans_road_.end()) {
      GELOGW("Var name: %s has already set.", batch_var_name.c_str());
      return GRAPH_SUCCESS;
    }
    var_to_trans_road_[batch_var_name] = trans_road;
    return GRAPH_SUCCESS;
  }

  VarTransRoad *GetTransRoad(const std::string &var_name);

  Status SetChangedGraphId(const std::string &var_name, const uint32_t graph_id) {
    var_names_to_changed_graph_id_[var_name] = graph_id;
    (void) graph_id_to_changed_var_names_[graph_id].emplace(var_name);
    return SUCCESS;
  }

  Status GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) const;

  std::set<std::string> GetChangedVarNames(const uint32_t graph_id) const;

  void RemoveChangedGraphId(const std::string &var_name) {
    const auto graph_id = var_names_to_changed_graph_id_[var_name];
    (void)graph_id_to_changed_var_names_[graph_id].erase(var_name);
    if (graph_id_to_changed_var_names_[graph_id].empty()) {
      (void)graph_id_to_changed_var_names_.erase(graph_id);
    }
    (void)var_names_to_changed_graph_id_.erase(var_name);
    (void)var_names_to_changed_graph_id_.erase(GetBatchVarKeyName(var_name));
  }

  Status SetAllocatedGraphId(const std::string &var_name, uint32_t graph_id);
  Status GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) const;

  bool IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const;

  bool IsVarExist(const std::string &var_name) const;

  bool IsVarAddr(const int64_t offset) const;

  rtMemType_t GetVarMemType(const int64_t offset);

  void UpdateDevVarMgrInfo(const uint32_t device_id);

  VarDevAddrMgr *GetVarMgrInfo(const uint32_t device_id, const int64_t offset);

  const std::map<uint32_t, std::unordered_map<uint64_t, VarDevAddrMgr>> &GetAllDevVarMgrInfo() const {
    return device_id_to_var_dev_addr_mgr_map_;
  }

  ge::Status CheckLogicAddrVaild(const uint32_t device_id,
                                 const uint8_t *const logic_addr,
                                 uint64_t &inner_offset_tmp,
                                 uint64_t &logic_addr_tmp);

  Status SetVarMgrDevAddr(const uint32_t device_id, const int64_t offset, uint8_t *const dev_addr);

  std::unordered_map<std::string, ge::GeTensorDesc> GetAllVarDesc() const { return cur_var_tensor_desc_map_; }

  void SetVarIsReady(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, const uint32_t device_id);

  bool IsVarReady(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, const uint32_t device_id) const;

  Status VarResourceToSerial(deployer::VarResourceInfo *const var_resource_info) const;

  Status VarDescInfoToSerial(deployer::VarDescInfo &desc_info) const;

  Status VarResourceToDeserial(const deployer::VarResourceInfo *const var_resource_info);

  void SetBatchVariablesKeyName(const std::string &batch_var_name, const std::string &key_name);

  bool HasSharedVarMemBetweenBatch() const;

 private:
  std::string VarKey(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const;
  std::string GetBatchVarKeyName(const std::string &var_name) const;
  int32_t GetSizeByTensoDataType(const OpDescPtr &op_desc) const;

  uint64_t session_id_;
  std::unordered_map<uint64_t, rtMemType_t> var_offset_map_;
  std::unordered_map<std::string, VarAddrMgr> var_addr_mgr_map_;
  std::unordered_map<std::string, ge::GeTensorDesc> cur_var_tensor_desc_map_;
  std::unordered_map<std::string, std::vector<TransNodeInfo>> var_to_trans_road_;
  std::unordered_map<uint64_t, VarDevAddrMgr> var_dev_addr_mgr_map_;
  std::map<uint32_t, std::unordered_map<uint64_t, VarDevAddrMgr>> device_id_to_var_dev_addr_mgr_map_;
  std::map<std::string, uint32_t> var_names_to_changed_graph_id_;
  std::map<uint32_t, std::set<std::string>> graph_id_to_changed_var_names_;
  std::map<uint32_t, std::map<std::string, ge::GeTensorDesc>> graph_id_to_staged_var_desc_;
  std::map<std::string, uint32_t> var_names_to_allocated_graph_id_;
  std::map<uint32_t, std::unordered_map<std::string, VarBroadCastInfo>> var_broad_cast_info_;
  std::map<uint32_t, std::set<std::string>> var_is_instance_;
  std::unordered_map<std::string, std::string> batch_var_name_map_;
};

class MemResource {
 public:
  MemResource();
  virtual ~MemResource() = default;
  static std::shared_ptr<MemResource> BuildMemResourceFromType(const rtMemType_t mem_type);

  virtual Status AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                              size_t &mem_offset) = 0;

  uint64_t GetVarMemSize() const;

  void UpdateVarMemSize(const int64_t mem_size);

 private:
  MemResource(MemResource const &) = delete;
  MemResource &operator=(MemResource const &) & = delete;

 protected:
  uint64_t total_size_ = 0U;
  uint64_t var_mem_size_ = 0U;
};

class HbmMemResource : public MemResource {
 public:
  HbmMemResource() = default;
  ~HbmMemResource() override = default;

  Status AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                      size_t &mem_offset) override;
};

class RdmaMemResource : public MemResource {
 public:
  RdmaMemResource() = default;
  ~RdmaMemResource() override = default;

  Status AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                      size_t &mem_offset) override;
};

class HostMemResource : public MemResource {
 public:
  HostMemResource() = default;
  ~HostMemResource() override = default;

  Status AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                      size_t &mem_offset) override;
};

class VarManager {
 public:
  static std::shared_ptr<VarManager> Instance(const uint64_t session_id);
  explicit VarManager(const uint64_t session_id);
  ~VarManager() = default;

  Status Init(const uint32_t version, const uint64_t session_id, const uint32_t device_id, const uint64_t job_id);

  void SetMemManager(MemoryManager *const mem_manager);

  void Destory();

  Status AssignVarMem(const std::string &var_name, const OpDescPtr &op_desc, const GeTensorDesc &tensor_desc,
                      rtMemType_t memory_type);

  Status SetVarAddr(const std::string &var_name, const GeTensorDesc &tensor_desc, const uint8_t *const dev_ptr,
                    const rtMemType_t memory_type, const OpDescPtr &op_desc);

  Status GetVarAddr(const std::string &var_name, const GeTensorDesc &tensor_desc, uint8_t *&dev_ptr,
                    rtMemType_t &memory_type) const;

  Status GetVarAddr(const std::string &var_name, const GeTensorDesc &tensor_desc, uint8_t *&dev_ptr) const;

  Status SaveBroadCastInfo(const uint32_t graph_id, const VarBroadCastInfo &broad_cast_info);

  Status GetCurVarDesc(const std::string &var_name, GeTensorDesc &tensor_desc);

  Status RecordStagedVarDesc(const uint32_t graph_id, const std::string &var_name, const GeTensorDesc &tensor_desc);

  const std::map<std::string, GeTensorDesc> &GetStagedVarDescs(const uint32_t graph_id);

  Status RenewCurVarDesc(const std::string &var_name, OpDescPtr op_desc);

  Status RenewCurVarDesc(const std::string &var_name, const GeTensorDesc &tensor_desc);

  Status MallocVarMemory(const uint64_t memory_size = kMemoryVarManagerMallocSize,
                         const uint32_t device_id = kDefaultDeviceId);

  Status FreeVarMemory();

  Status SetTransRoad(const std::string &var_name, const VarTransRoad &trans_road);

  VarTransRoad *GetTransRoad(const std::string &var_name);

  Status SetChangedGraphId(const std::string &var_name, const uint32_t graph_id);

  Status GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) const;

  std::set<std::string> GetChangedVarNames(const uint32_t graph_id) const;

  Status SetMemoryMallocSize(const std::map<std::string, std::string> &options, const size_t total_mem_size);

  Status SetAllMemoryMaxValue(const std::map<std::string, std::string> &options);

  bool GetEvaluateMode() const {
    std::string option_value;
    if (GetThreadLocalContext().GetOption(EVALUATE_GRAPH_RESOURCE_MODE, option_value) == GRAPH_SUCCESS) {
      // 1: graph resource evaluation
      GELOGI("EvaluateGraphResourceMode is %s", option_value.c_str());
      return (option_value == "1");
    }
    return false;
  }

  uint64_t GetGraphMemoryMaxSize(const bool for_check = false) const {
    if (for_check && GetEvaluateMode()) {
      return std::numeric_limits<uint64_t>::max();
    }
    return graph_mem_max_size_;
  }

  uint64_t GetVarMemMaxSize(const bool for_check = false) const {
    if (for_check && GetEvaluateMode()) {
      return std::numeric_limits<uint64_t>::max();
    }
    return var_mem_max_size_;
  }

  const uint64_t &GetVarMemLogicBase() const { return var_mem_logic_base_; }

  uint64_t GetUseMaxMemorySize() const {
    if (GetEvaluateMode()) {
      return std::numeric_limits<uint64_t>::max();
    }
    return use_max_mem_size_;
  }

  void RemoveChangedGraphId(const std::string &var_name);

  Status SetAllocatedGraphId(const std::string &var_name, const uint32_t graph_id);

  Status GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) const;

  const uint64_t &SessionId() const;

  int64_t GetVarMemSize(const rtMemType_t memory_type) const;

  bool IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const;

  bool IsVarExist(const std::string &var_name) const;

  bool IsVarAddr(const int64_t offset) const;

  rtMemType_t GetVarMemType(const int64_t offset);

  uint8_t *GetVarMemoryBase(const rtMemType_t memory_type, const uint32_t device_id = kDefaultDeviceId);

  uint8_t *GetVarMemoryAddr(uint8_t *const logic_addr, const rtMemType_t memory_type,
                            const uint32_t device_id = kDefaultDeviceId);

  uint8_t *GetAutoMallocVarAddr(const uint8_t *const logic_addr, const rtMemType_t memory_type,
                                const uint32_t device_id);

  uint8_t *GetRdmaPoolMemory(const rtMemType_t memory_type, const size_t mem_size);
  uint8_t *GetHostPoolMemory(const rtMemType_t memory_type, const size_t mem_size);

  Status GetAllVariables(std::map<std::string, GeTensorDesc> &all_variables);

  void SetVarIsReady(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, const uint32_t device_id);

  bool IsVarReady(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, const uint32_t device_id) const;

  Status VarManagerToSerial(const uint64_t session_id, deployer::VarManagerInfo &info) const;

  Status VarDescInfoToSerial(const uint64_t session_id, deployer::VarDescInfo &info) const;

  Status VarManagerToDeserial(const uint64_t session_id, const deployer::VarManagerInfo &info);

  void UpdateMemoryConfig(const size_t graph_mem_max_size, const size_t var_mem_max_size,
                          const size_t var_mem_logic_base, const size_t use_max_mem_size,
                          const bool var_mem_auto_malloc);

  void SetBatchVariablesKeyName(const std::string &batch_var_name, const std::string &key_name);

  bool HasSharedVarMemBetweenBatch() const;

  bool HasMemoryManager() const;

  bool IsVarResourceInited() const { return (var_resource_ != nullptr); }

  void SetVarMemAutoMalloc(const bool is_set_flag) {
    var_mem_auto_malloc_ = is_set_flag;
  }

  bool IsVarMemAutoMalloc() const { return var_mem_auto_malloc_; }

  int64_t GetVarMallocMemSize() const { return var_malloc_mem_size_; }
 private:
  int64_t var_malloc_mem_size_ = 0LL;
  bool var_mem_auto_malloc_ = true;
  SessionVersion version_ = SessionVersion::OTHER_VERSION;
  uint64_t session_id_;
  uint32_t device_id_ = kDefaultDeviceId;
  uint64_t job_id_ = 0U;
  uint64_t graph_mem_max_size_ = kGraphMemoryManagerMallocMaxSize;
  uint64_t var_mem_max_size_ = kMemoryVarManagerMallocSize;
  uint64_t var_mem_logic_base_ = kMemoryVarLogicBase;
  uint64_t use_max_mem_size_ = kUseMaxMemorySize;
  std::shared_ptr<ge::VarResource> var_resource_;
  std::map<rtMemType_t, std::shared_ptr<MemResource>> mem_resource_map_;
  mutable std::recursive_mutex mutex_;
  MemoryManager *mem_manager_{nullptr};
};

class VarManagerPool {
 public:
  virtual ~VarManagerPool();

  static VarManagerPool &Instance();

  std::shared_ptr<VarManager> GetVarManager(const uint64_t session_id);

  void RemoveVarManager(const uint64_t session_id);

  void Destory() noexcept;

 private:
  VarManagerPool() = default;
  std::mutex var_manager_mutex_;
  std::map<uint64_t, std::shared_ptr<VarManager>> var_manager_map_;
};
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_GRAPH_VAR_MANAGER_H_
