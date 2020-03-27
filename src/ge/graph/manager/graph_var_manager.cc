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

#include "graph/manager/graph_var_manager.h"

#include <utility>

#include "common/l2_cache_optimize.h"
#include "graph/debug/ge_attr_define.h"
#include "common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "ge/ge_api_types.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"

using std::string;
using std::vector;
using std::map;

namespace ge {
VarResource::VarResource(uint64_t session_id) : session_id_(session_id) {}

VarResource::~VarResource() {
  var_offset_set_.clear();
  var_addr_mgr_map_.clear();
  cur_var_tensor_desc_map_.clear();
  var_broad_cast_info_.clear();
}

ge::Status VarResource::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **dev_ptr,
                                   rtMemType_t &memory_type) {
  if (dev_ptr == nullptr) {
    GELOGE(FAILED, "[GetVarAddr] dev_ptr is null!");
    return FAILED;
  }
  std::string var_key = VarKey(var_name, tensor_desc);
  GELOGD("VarResource::GetVarAddr , var_key = %s", var_key.c_str());

  auto iter = var_addr_mgr_map_.find(var_key);
  if (iter == var_addr_mgr_map_.end()) {
    GELOGE(FAILED, "VarResource::GetVarAddr failed, var_key %s", var_key.c_str());
    return FAILED;
  }

  *dev_ptr = iter->second.address;
  memory_type = iter->second.memory_type;

  return SUCCESS;
}

void VarResource::SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *dev_ptr,
                             rtMemType_t memory_type) {
  std::string var_key = VarKey(var_name, tensor_desc);
  GELOGI("VarResource::SetVarAddr , var_key = %s, mem_type:%u", var_key.c_str(), memory_type);
  if (var_addr_mgr_map_.count(var_key) == 0) {
    GELOGI("SetVarAddr node_name %s, tensor_desc type %s, format %s", var_name.c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

    VarAddrMgr var_addr_mgr;
    var_addr_mgr.address = dev_ptr;
    var_addr_mgr.tensor_desc = tensor_desc;
    var_addr_mgr_map_[var_key] = var_addr_mgr;
  }

  cur_var_tensor_desc_map_[var_name] = tensor_desc;
}

ge::Status VarResource::SaveVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *address,
                                    rtMemType_t memory_type) {
  std::string var_key = VarKey(var_name, tensor_desc);
  GELOGD("VarResource::SaveVarAddr, var_key = %s", var_key.c_str());
  if (var_addr_mgr_map_.count(var_key) == 0) {
    uint64_t logic_address = VarManager::Instance(0)->GetVarMemLogicBase() +
                             reinterpret_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(address));
    GELOGI("SaveVarAddr node_name %s, tensor_desc format %s, type %s.", var_name.c_str(),
           TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());
    VarAddrMgr var_addr_mgr;
    var_addr_mgr.address = reinterpret_cast<uint8_t *>(reinterpret_cast<std::uintptr_t>(logic_address));
    var_addr_mgr.offset = reinterpret_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(address));
    var_addr_mgr.tensor_desc = tensor_desc;
    var_addr_mgr.memory_type = memory_type;
    var_addr_mgr_map_[var_key] = var_addr_mgr;
    var_offset_set_.insert(logic_address);

    return SUCCESS;
  }

  GELOGE(FAILED, "VarResource::SaveVarAddr, var_key %s save addr conflict", var_key.c_str());
  return FAILED;
}

bool VarResource::IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) {
  std::string var_key = VarKey(var_name, tensor_desc);
  return var_addr_mgr_map_.count(var_key) != 0;
}

bool VarResource::IsVarExist(const std::string &var_name) { return cur_var_tensor_desc_map_.count(var_name) != 0; }

std::string VarResource::VarKey(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) {
  std::string var_key(var_name);
  var_key.append(std::to_string(static_cast<int32_t>(tensor_desc.GetFormat())))
      .append("_")
      .append(std::to_string(static_cast<int32_t>(tensor_desc.GetDataType())));
  return var_key;
}

ge::Status VarResource::GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc) {
  if (cur_var_tensor_desc_map_.count(var_name) == 0) {
    return FAILED;
  }
  tensor_desc = cur_var_tensor_desc_map_[var_name];
  return SUCCESS;
}

ge::Status VarResource::RenewCurVarDesc(const std::string &var_name, const ge::OpDescPtr &op_desc) {
  if (cur_var_tensor_desc_map_.count(var_name) == 0) {
    GELOGI("There is no this node[%s] in var tensor_desc map. so no need renew!", var_name.c_str());
    return SUCCESS;
  }

  if (op_desc == nullptr) {
    GELOGE(FAILED, "[RenewCurVarDesc] renew var desc fail! input opdesc is null!");
    return FAILED;
  }

  ge::GeTensorDesc curr_desc;
  ge::Status ret = GetCurVarDesc(var_name, curr_desc);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[RenewCurVarDesc] Get var desc fail!");
    return FAILED;
  }
  std::string key = VarKey(var_name, curr_desc);
  curr_desc.SetOriginFormat((op_desc->GetOutputDesc(0)).GetOriginFormat());
  curr_desc.SetFormat((op_desc->GetOutputDesc(0)).GetFormat());
  cur_var_tensor_desc_map_[var_name] = curr_desc;
  auto iter = var_addr_mgr_map_.find(key);
  if (iter == var_addr_mgr_map_.end()) {
    GELOGE(FAILED, "[RenewCurVarDesc] can't find ele with key [%s]", key.c_str());
    return FAILED;
  }
  auto val = iter->second;
  val.tensor_desc.SetOriginFormat((op_desc->GetOutputDesc(0)).GetOriginFormat());
  val.tensor_desc.SetFormat((op_desc->GetOutputDesc(0)).GetFormat());
  var_addr_mgr_map_.erase(iter);
  key = VarKey(var_name, curr_desc);
  var_addr_mgr_map_[key] = val;

  return SUCCESS;
}

void VarResource::SaveBroadCastInfo(uint32_t graph_id, const VarBroadCastInfo &broad_cast_info) {
  var_broad_cast_info_[graph_id][broad_cast_info.var_name] = broad_cast_info;
}

ge::Status VarResource::SyncVarData2BroadCast(uint32_t graph_id, const std::string &var_name,
                                              const ge::ConstOpDescPtr &var_op_desc, uint8_t *base_ptr) {
  if (var_op_desc == nullptr) {
    GELOGE(FAILED, "[SyncVarData2BroadCast] var opdesc is null!");
    return FAILED;
  }
  GE_CHECK_NOTNULL(base_ptr);
  GELOGI("SyncVarData2BroadCast graph_id: %u, var_name: %s.", graph_id, var_name.c_str());

  VarBroadCastInfo var_broadcast_info = var_broad_cast_info_[graph_id][var_name];
  uint8_t *dst_addr = base_ptr + var_broadcast_info.input_offset;
  ge::GeTensorDesc var_tensor_desc = var_op_desc->GetOutputDesc(0);

  return ge::TransVarDataUtils::SyncVarData2BroadCast(var_name, var_tensor_desc, dst_addr,
                                                      var_broadcast_info.input_size, session_id_);
}

ge::Status VarResource::SyncBroadCastData2Var(uint32_t graph_id, const std::string &var_name,
                                              const ge::ConstOpDescPtr &var_op_desc, uint8_t *base_ptr) {
  GELOGI("SyncBroadCastData2Var var_name: %s", var_name.c_str());
  GE_CHECK_NOTNULL(var_op_desc);
  string var_is_broadcast;
  bool is_broadcast = AttrUtils::GetStr(var_op_desc, VAR_ATTR_VAR_IS_BROADCAST, var_is_broadcast);
  if (!is_broadcast) {
    return SUCCESS;
  }

  VarBroadCastInfo var_broadcast_info = var_broad_cast_info_[graph_id][var_name];
  // subgraph base_ptr could be nullptr, task it as base 0
  uint8_t *dst_addr = base_ptr + var_broadcast_info.output_offset;
  ge::GeTensorDesc var_tensor_desc = var_op_desc->GetOutputDesc(0);

  return ge::TransVarDataUtils::SyncBroadCastData2Var(dst_addr, var_broadcast_info.output_size, var_name,
                                                      var_tensor_desc, session_id_);
}

ge::Status VarResource::SyncVarData(uint32_t graph_id, const std::string &var_name,
                                    const ge::ConstOpDescPtr &var_op_desc, uint8_t *base_ptr) {
  GE_CHECK_NOTNULL(var_op_desc);
  string var_is_broadcast;
  bool is_broadcast = AttrUtils::GetStr(var_op_desc, VAR_ATTR_VAR_IS_BROADCAST, var_is_broadcast);
  if (!is_broadcast) {
    return SUCCESS;
  }

  return SyncVarData2BroadCast(graph_id, var_name, var_op_desc, base_ptr);
}

bool VarResource::IsVarAddr(const int64_t &offset) { return var_offset_set_.count(offset) > 0; }

VarTransRoad *VarResource::GetTransRoad(const std::string &var_name) {
  auto iter = var_to_trans_road_.find(var_name);
  if (iter == var_to_trans_road_.end()) {
    return nullptr;
  } else {
    return &(iter->second);
  }
}

Status VarResource::GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) {
  auto iter = var_names_to_changed_graph_id_.find(var_name);
  if (iter == var_names_to_changed_graph_id_.end()) {
    return FAILED;
  } else {
    graph_id = iter->second;
    return SUCCESS;
  }
}
Status VarResource::GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) {
  auto iter = var_names_to_allocated_graph_id_.find(var_name);
  if (iter == var_names_to_allocated_graph_id_.end()) {
    return FAILED;
  } else {
    graph_id = iter->second;
    return SUCCESS;
  }
}

Status VarResource::SetAllocatedGraphId(const std::string &var_name, uint32_t graph_id) {
  if (GetAllocatedGraphId(var_name, graph_id) == SUCCESS) {
    GELOGW("VarManager var[%s] has been allocated in graph[%d]", var_name.c_str(), graph_id);
    return SUCCESS;
  }
  var_names_to_allocated_graph_id_[var_name] = graph_id;
  return SUCCESS;
}

MemResource::MemResource() : total_size_(0), var_mem_base_(nullptr), var_mem_size_(0) {}

Status MemResource::AssignVarMem(const std::string &var_name, uint64_t size, uint64_t session_id, size_t &mem_offset) {
  size = (size + kSessionMemAlignSize - 1) / kSessionMemAlignSize * kSessionMemAlignSize;

  total_size_ = VarManager::Instance(0)->GetVarMemMaxSize();
  if (total_size_ < var_mem_size_) {
    GELOGE(PARAM_INVALID, "total_size_: %lu is smaller than var_mem_size_: %lu", total_size_, var_mem_size_);
    return PARAM_INVALID;
  }
  uint64_t free_size = total_size_ - var_mem_size_;
  if (free_size < (size + kSessionMemAlignSize * 2)) {
    GELOGE(PARAM_INVALID, "malloc var mem, size[%lu] > free_size[%lu]", size, free_size);
    return PARAM_INVALID;
  }

  mem_offset = var_mem_size_;

  // offset for next, align 512 BYTE
  size = size + kSessionMemAlignSize;
  var_mem_size_ = var_mem_size_ + size;

  // align 512 BYTE
  var_mem_size_ = var_mem_size_ + kSessionMemAlignSize;
  return SUCCESS;
}

int64_t MemResource::GetVarMemSize() const { return var_mem_size_; }

VarManager::VarManager(uint64_t session_id)
    : version_(SessionVersion::OTHER_VERSION),
      session_id_(session_id),
      device_id_(0),
      job_id_(0),
      graph_mem_max_size_(kGraphMemoryManagerMallocMaxSize),
      var_mem_max_size_(kMemoryVarManagerMallocSize),
      var_mem_logic_base_(kMemoryVarLogicBase),
      use_max_mem_size_(kUseMaxMemorySize) {}

VarManager *VarManager::Instance(uint64_t session_id) {
  GELOGD("VarManager::Instance, session id = %lu", session_id);
  return VarManagerPool::Instance().GetVarManager(session_id);
}

void VarManager::Destory() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::Destory, session id = %lu.", session_id_);
  version_ = SessionVersion::OTHER_VERSION;
  device_id_ = 0;
  session_id_ = 0;
  for (auto &memory_resource : mem_resource_map_) {
    if (memory_resource.second != nullptr) {
      delete memory_resource.second;
      memory_resource.second = nullptr;
    }
  }
  mem_resource_map_.clear();
}

ge::Status VarManager::Init(const uint32_t &version, const uint64_t &session_id, const uint32_t &device_id,
                            const uint64_t &job_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::Init, session id = %lu.", session_id);
  version_ = version;
  device_id_ = device_id;
  session_id_ = session_id;
  job_id_ = job_id;
  var_resource_ = std::unique_ptr<VarResource>(new (std::nothrow) VarResource(session_id_));
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return SUCCESS;
}

const uint64_t &VarManager::SessionId() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return session_id_;
}

const uint32_t &VarManager::DeviceId() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return device_id_;
}

const uint64_t &VarManager::JobId() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return job_id_;
}

ge::Status VarManager::SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *dev_ptr,
                                  rtMemType_t memory_type) {
  GELOGI("VarManager::SetVarAddr var_name = %s, data_type = %s, data_format = %s.", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  var_resource_->SetVarAddr(var_name, tensor_desc, dev_ptr, memory_type);
  return ge::SUCCESS;
}

ge::Status VarManager::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **dev_ptr,
                                  rtMemType_t &memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::GetVarAddr var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  auto ret = var_resource_->GetVarAddr(var_name, tensor_desc, dev_ptr, memory_type);
  if (ret != SUCCESS) {
    GELOGW("GetVarAddr fail.");
    return ge::INTERNAL_ERROR;
  }
  return SUCCESS;
}

ge::Status VarManager::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **dev_ptr) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  rtMemType_t memory_type = RT_MEMORY_HBM;
  return GetVarAddr(var_name, tensor_desc, dev_ptr, memory_type);
}

int64_t VarManager::GetVarMemSize(rtMemType_t memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  MemResource *mem_resource = nullptr;
  auto iter = mem_resource_map_.find(memory_type);
  if (iter == mem_resource_map_.end()) {
    return 0;
  } else {
    mem_resource = iter->second;
  }

  if (mem_resource == nullptr) {
    GELOGE(ge::INTERNAL_ERROR, "MemResource is invalid.");
    return 0;
  }
  return mem_resource->GetVarMemSize();
}

ge::Status VarManager::AssignVarMem(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                    rtMemType_t memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::AssignVarMem var_name = %s, data_type = %s, data_format = %s.", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  uint32_t tensor_desc_size = 0;
  size_t mem_offset = 0;
  ge::Status result = TensorUtils::GetSize(tensor_desc, tensor_desc_size);
  if (result != ge::SUCCESS) {
    GELOGE(result, "get size from TensorDesc failed");
    return result;
  }

  MemResource *mem_resource = nullptr;
  auto it = mem_resource_map_.find(memory_type);
  if (it == mem_resource_map_.end()) {
    mem_resource = new (std::nothrow) MemResource();
    if (mem_resource == nullptr) {
      GELOGE(ge::INTERNAL_ERROR, "Alloc MemResource failed, memory_type = %u.", memory_type);
      return ge::INTERNAL_ERROR;
    } else {
      mem_resource_map_[memory_type] = mem_resource;
    }
  } else {
    mem_resource = it->second;
  }

  if (mem_resource == nullptr) {
    GELOGE(ge::INTERNAL_ERROR, "MemResource is invalid, memory_type = %u.", memory_type);
    return ge::INTERNAL_ERROR;
  }
  result = mem_resource->AssignVarMem(var_name, tensor_desc_size, session_id_, mem_offset);
  if (result != SUCCESS) {
    GELOGE(ge::INTERNAL_ERROR, "AssignVarMem by offset failed.");
    return ge::INTERNAL_ERROR;
  }
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }

  result = var_resource_->SaveVarAddr(
      var_name, tensor_desc, reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(mem_offset)), memory_type);
  if (result != SUCCESS) {
    GELOGE(ge::INTERNAL_ERROR, "AssignVarMem by offset failed.");
    return ge::INTERNAL_ERROR;
  }

  result = var_resource_->GetVarAddr(
      var_name, tensor_desc, reinterpret_cast<uint8_t **>(reinterpret_cast<uintptr_t>(&mem_offset)), memory_type);
  if (result != SUCCESS) {
    GELOGE(ge::INTERNAL_ERROR, "GetVarAddr by offset failed.");
    return ge::INTERNAL_ERROR;
  }

  ge::GeTensorDesc cur_tensor_desc;
  result = var_resource_->GetCurVarDesc(var_name, cur_tensor_desc);
  if (result != SUCCESS) {
    var_resource_->SetVarAddr(var_name, tensor_desc,
                              reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(mem_offset)), memory_type);
    return SUCCESS;
  }

  if (cur_tensor_desc.GetFormat() != tensor_desc.GetFormat() ||
      cur_tensor_desc.GetDataType() != tensor_desc.GetDataType() ||
      cur_tensor_desc.GetShape().GetDims() != tensor_desc.GetShape().GetDims()) {
    GELOGI("var %s assigned new memory (format, data type, shape)  (%s, %s, %zu) from (%s, %s, %zu)", var_name.c_str(),
           ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
           ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
           tensor_desc.GetShape().GetDims().size(),
           ge::TypeUtils::DataTypeToSerialString(cur_tensor_desc.GetDataType()).c_str(),
           ge::TypeUtils::FormatToSerialString(cur_tensor_desc.GetFormat()).c_str(),
           cur_tensor_desc.GetShape().GetDims().size());
    var_resource_->SetVarAddr(var_name, tensor_desc,
                              reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(mem_offset)), memory_type);
  }

  return SUCCESS;
}

bool VarManager::IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::IsVarExist var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarExist(var_name, tensor_desc);
}

bool VarManager::IsVarExist(const std::string &var_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarExist(var_name);
}

ge::Status VarManager::SyncVarData(uint32_t graph_id, const std::string &var_name, ge::ConstOpDescPtr var_op_desc,
                                   uint8_t *base_ptr) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->SyncVarData(graph_id, var_name, std::move(var_op_desc), base_ptr);
}

ge::Status VarManager::GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::GetCurVarDesc var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->GetCurVarDesc(var_name, tensor_desc);
}

ge::Status VarManager::SaveBroadCastInfo(uint32_t graph_id, const VarBroadCastInfo &broad_cast_info) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI(
      "VarManager::SaveBroadCastInfo var_name = %s, broadcast name = %s, "
      "idx = %d, input_offset = %ld, input_size = %lu, output_offset = %ld, "
      "output_size = %lu",
      broad_cast_info.var_name.c_str(), broad_cast_info.broadcast_name.c_str(), broad_cast_info.idx,
      broad_cast_info.input_offset, broad_cast_info.input_size, broad_cast_info.output_offset,
      broad_cast_info.output_size);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  var_resource_->SaveBroadCastInfo(graph_id, broad_cast_info);
  return SUCCESS;
}

ge::Status VarManager::RenewCurVarDesc(const std::string &var_name, ge::OpDescPtr op_desc) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::RenewCurVarDesc var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    GELOGE(ge::INTERNAL_ERROR, "VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->RenewCurVarDesc(var_name, std::move(op_desc));
}

ge::Status VarManager::SyncBroadCastData2Var(uint32_t graph_id, const std::string &var_name,
                                             ge::ConstOpDescPtr var_op_desc, uint8_t *base_ptr) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->SyncBroadCastData2Var(graph_id, var_name, std::move(var_op_desc), base_ptr);
}

bool VarManager::IsVarAddr(const int64_t &offset) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarAddr(offset);
}

ge::Status VarManager::MallocVarMemory(size_t memory_size) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  uint8_t *var_mem_base = nullptr;
  string memory_key = std::to_string(session_id_);

  // malloc variable memory
  size_t var_memory_size = memory_size;

  // align 512 BYTE
  var_memory_size = (var_memory_size + kSessionMemAlignSize - 1) / kSessionMemAlignSize * kSessionMemAlignSize;

  var_mem_base = MemManager::Instance(RT_MEMORY_HBM)->MallocMemory(memory_key, var_memory_size);
  if (var_mem_base == nullptr) {
    GELOGE(ge::INTERNAL_ERROR,
           "VarManager::MallocVarMemory failed "
           "session_id = %s",
           memory_key.c_str());
    return ge::INTERNAL_ERROR;
  }
  return SUCCESS;
}

uint8_t *VarManager::GetVarMemoryBase(rtMemType_t memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  string memory_key = std::to_string(session_id_);
  return MemManager::Instance(memory_type)->GetMemoryAddr(memory_key);
}

uint8_t *VarManager::GetVarMemoryAddr(uint8_t *logic_addr, rtMemType_t memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  string mem_key = std::to_string(session_id_);
  uint8_t *mem_base = MemManager::Instance(memory_type)->GetMemoryAddr(mem_key);
  if (mem_base == nullptr) {
    return nullptr;
  }
  uint8_t *mem_addr = logic_addr + reinterpret_cast<intptr_t>(mem_base) - VarManager::Instance(0)->GetVarMemLogicBase();
  return mem_addr;
}

ge::Status VarManager::FreeVarMemory() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  string memory_key = std::to_string(SessionId());
  return MemManager::Instance(RT_MEMORY_HBM)->FreeMemory(memory_key);
}

ge::Status VarManager::SetTransRoad(const std::string &var_name, const VarTransRoad &trans_road) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->SetTransRoad(var_name, trans_road);
}

VarTransRoad *VarManager::GetTransRoad(const std::string &var_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return nullptr;
  }
  return var_resource_->GetTransRoad(var_name);
}

Status VarManager::SetChangedGraphId(const std::string &var_name, uint32_t graph_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->SetChangedGraphId(var_name, graph_id);
}

Status VarManager::GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->GetChangedGraphId(var_name, graph_id);
}

Status VarManager::SetMemoryMallocSize(const map<string, string> &options) {
  auto it = options.find(GRAPH_MEMORY_MAX_SIZE);
  if (it == options.end()) {
    graph_mem_max_size_ = kGraphMemoryManagerMallocMaxSize;
  } else {
    string graph_memory_manager_malloc_max_size = it->second;
    ge::Status ret = ParseMemoryMallocSize(graph_memory_manager_malloc_max_size, graph_mem_max_size_);
    if (ret != SUCCESS) {
      GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "Parse graph memory manager malloc max size failed.");
      return ge::GE_GRAPH_OPTIONS_INVALID;
    }
  }

  it = options.find(VARIABLE_MEMORY_MAX_SIZE);
  if (it == options.end()) {
    var_mem_max_size_ = kMemoryVarManagerMallocSize;
  } else {
    string memory_var_manager_malloc_size = it->second;
    ge::Status ret = ParseMemoryMallocSize(memory_var_manager_malloc_size, var_mem_max_size_);
    if (ret != SUCCESS) {
      GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "Parse memory var manager malloc size failed.");
      return ge::GE_GRAPH_OPTIONS_INVALID;
    }
  }

  var_mem_logic_base_ = graph_mem_max_size_ + kGraphMemoryBuffer;
  if (var_mem_logic_base_ > kMaxMemorySize) {
    GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "kMemoryVarLogicBase : %zu can not exceed max memory size : %zu.",
           var_mem_logic_base_, kMaxMemorySize);
    return ge::GE_GRAPH_OPTIONS_INVALID;
  }

  use_max_mem_size_ = graph_mem_max_size_ + var_mem_max_size_;
  if (use_max_mem_size_ > kMaxMemorySize) {
    GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "kUseMaxMemorySize : %zu can not exceed max memory size : %zu.",
           use_max_mem_size_, kMaxMemorySize);
    return ge::GE_GRAPH_OPTIONS_INVALID;
  }
  GELOGI("Set memory malloc size successfully");
  return SUCCESS;
}

Status VarManager::ParseMemoryMallocSize(string &memory_size, size_t &result) {
  if (memory_size.empty()) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Memory malloc size input is empty.");
    return GE_GRAPH_OPTIONS_INVALID;
  }
  // split string by '*'
  vector<string> splits;
  std::istringstream str(memory_size);
  string str_split;
  while (getline(str, str_split, '*')) {
    splits.emplace_back(str_split);
  }

  result = 1;
  for (string split : splits) {
    // Trim
    auto it = split.find_first_not_of(" ");
    if (it != string::npos) {
      split.erase(0, it);
    }
    it = split.find_last_not_of(" ");
    if (it != string::npos) {
      split.erase(it + 1);
    }

    for (char c : split) {
      if (!isdigit(c)) {
        GELOGE(GE_GRAPH_OPTIONS_INVALID, "Memory malloc size input contains non digit.");
        return GE_GRAPH_OPTIONS_INVALID;
      }
    }
    uint64_t num = std::strtoul(split.c_str(), nullptr, 0);
    GE_IF_BOOL_EXEC(TypeUtils::CheckUint64MulOverflow(result, static_cast<uint32_t>(num)),
                    GELOGE(FAILED, "Input memory size is out of range.");
                    return FAILED);
    if ((num > kMaxMemorySize) || (result * static_cast<size_t>(num) > kMaxMemorySize)) {
      GELOGE(FAILED, "Input memory size can not exceed max memory size : %zu.", kMaxMemorySize);
      return FAILED;
    }
    result *= static_cast<size_t>(num);
  }

  return SUCCESS;
}

void VarManager::RemoveChangedGraphId(const std::string &var_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return;
  }
  var_resource_->RemoveChangedGraphId(var_name);
}

Status VarManager::SetAllocatedGraphId(const std::string &var_name, uint32_t graph_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->SetAllocatedGraphId(var_name, graph_id);
}

Status VarManager::GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->GetAllocatedGraphId(var_name, graph_id);
}

void VarManager::RemoveAllocatedGraphId(const std::string &var_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return;
  }
  var_resource_->RemoveAllocatedGraphId(var_name);
}

VarManagerPool::~VarManagerPool() { Destory(); }

VarManagerPool &VarManagerPool::Instance() {
  static VarManagerPool var_manager_pool;
  return var_manager_pool;
}

void VarManagerPool::Destory() noexcept {
  std::lock_guard<std::mutex> lock(var_manager_mutex_);
  for (auto &it : var_manager_map_) {
    VarManager *var_manager = it.second;
    if (var_manager != nullptr) {
      var_manager->Destory();
      delete var_manager;
      var_manager = nullptr;
    }
  }
  var_manager_map_.clear();
}

ge::Status VarManagerPool::Init() const { return SUCCESS; }

VarManager *VarManagerPool::GetVarManager(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(var_manager_mutex_);
  auto it = var_manager_map_.find(session_id);
  if (it != var_manager_map_.end()) {
    GELOGD("VarManagerPool::GetVarManager");
    return it->second;
  }

  VarManager *var_manager = new (std::nothrow) VarManager(session_id);
  if (var_manager == nullptr) {
    GELOGE(INTERNAL_ERROR,
           "VarManager::Instance find session by "
           "session_id[%lu] failed.",
           session_id);
    static VarManager new_var_manager(0);
    return &new_var_manager;
  }
  var_manager_map_[session_id] = var_manager;
  return var_manager;
}
}  // namespace ge
