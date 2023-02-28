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
#include "graph/manager/graph_var_manager.h"

#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/tuning_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_context.h"
#include "common/plugin/ge_util.h"
#include "common/math/math_util.h"
#include "runtime/dev.h"

namespace ge {
VarResource::VarResource(const uint64_t session_id) : session_id_(session_id) {}

VarResource::~VarResource() {
  var_offset_map_.clear();
  var_addr_mgr_map_.clear();
  cur_var_tensor_desc_map_.clear();
  var_broad_cast_info_.clear();
}

ge::Status VarResource::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                   uint8_t **const dev_ptr, rtMemType_t &memory_type) const {
  if (dev_ptr == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param dev_ptr is nullptr, var_name:%s, session_id:%" PRIu64 ", "
                       "check invalid", var_name.c_str(), session_id_);
    GELOGE(FAILED, "[Check][Param] Param dev_ptr is nullptr, var_name:%s, session_id:%" PRIu64 "",
           var_name.c_str(), session_id_);
    return FAILED;
  }
  const std::string var_key = VarKey(var_name, tensor_desc);
  GELOGD("VarResource::GetVarAddr , var_key = %s.", var_key.c_str());

  const auto iter = var_addr_mgr_map_.find(var_key);
  if (iter == var_addr_mgr_map_.end()) {
    REPORT_INNER_ERROR("E19999", "var_key:%s can't find in var_addr_mgr_map_, var_name:%s, session_id:%" PRIu64 ", "
                       "check invalid", var_key.c_str(), var_name.c_str(),
                       session_id_);
    GELOGE(FAILED, "[Check][Param] var_key:%s can't find in var_addr_mgr_map_, var_name:%s, session_id:%" PRIu64 "",
           var_key.c_str(), var_name.c_str(), session_id_);
    return FAILED;
  }

  *dev_ptr = const_cast<uint8_t *>(iter->second.address);
  memory_type = iter->second.memory_type;

  return SUCCESS;
}

int32_t VarResource::GetSizeByTensoDataType(const OpDescPtr &op_desc) const {
  const auto &output_tensor = op_desc->MutableOutputDesc(0);
  if (output_tensor == nullptr) {
    GELOGW("The const %s does not have output 0, skip to fusion", op_desc->GetName().c_str());
    return -1;
  }
  return GetSizeByDataType(output_tensor->GetDataType());
}

Status VarResource::GetReuseAddr(const OpDescPtr &op_desc, uint8_t **const dev_ptr, rtMemType_t &memory_type) const {
  GE_CHECK_NOTNULL(op_desc);
  const auto type_size = GetSizeByTensoDataType(op_desc);
  if (type_size <= 0) {
    return FAILED;
  }

  GeTensorPtr weight;
  if (!AttrUtils::MutableTensor(op_desc, ATTR_NAME_WEIGHTS, weight)) {
    GELOGW("The const node %s does not have weight attr, skip it", op_desc->GetName().c_str());
    return FAILED;
  }

  const auto &values = weight->MutableData().GetAlignedPtr();
  if (values == nullptr) {
    GELOGD("aligned_ptr is null.");
    return FAILED;
  }
  const auto weight_size = weight->MutableData().size();
  for (const auto &var_maps : var_addr_mgr_map_) {
    const auto &var_map = var_maps.second;
    const bool skip_var = (var_map.op_desc == nullptr) || (var_map.op_desc->GetType() != CONSTANTOP) ||
                    (GetSizeByTensoDataType(var_map.op_desc) != type_size);
    if (skip_var) {
      continue;
    }

    GeTensorPtr tmp_weight;
    if (!AttrUtils::MutableTensor(var_map.op_desc, ATTR_NAME_WEIGHTS, tmp_weight)) {
      GELOGW("The const node %s does not have weight attr, skip it", var_map.op_desc->GetName().c_str());
      continue;
    }

    if ((tmp_weight->MutableData().size() != weight_size) || (tmp_weight->MutableData().GetAlignedPtr() == nullptr)) {
      continue;
    }

    if (memcmp(values->Get(), tmp_weight->MutableData().GetAlignedPtr()->Get(), weight_size) == 0) {
      const auto real_size = (weight_size + kSessionMemAlignSize - 1U) / kSessionMemAlignSize * kSessionMemAlignSize;
      GELOGD("[IMAS]AssignVarMem Set session_%" PRIu64 " name[%s] output[%d] offset to [%" PRIu64 "] size[%" PRIu64 "]"
             "realsize[%" PRIu64 "], reuse address of node: %s.", session_id_, op_desc->GetName().c_str(), 0,
             var_map.offset, real_size + (kSessionMemAlignSize * kSessionMemAlignUnit), real_size,
             var_map.op_desc->GetName().c_str());
      *dev_ptr = const_cast<uint8_t *>(var_map.address);
      memory_type = var_map.memory_type;
      return SUCCESS;
    }
  }
  return FAILED;
}

void VarResource::SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                             const uint8_t *const dev_ptr, const rtMemType_t memory_type, const OpDescPtr &op_desc) {
  const std::string var_key = VarKey(var_name, tensor_desc);
  GELOGI("VarResource::SetVarAddr , var_key = %s, mem_type:%u.", var_key.c_str(), memory_type);
  if (var_addr_mgr_map_.count(var_key) == 0U) {
    GELOGI("SetVarAddr node_name %s, tensor_desc type %s, format %s", var_name.c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());
    uint64_t offset = 0U;
    if (memory_type != RT_MEMORY_RDMA_HBM) {
      offset = PtrToValue(dev_ptr) - VarManager::Instance(session_id_)->GetVarMemLogicBase();
    }
    var_addr_mgr_map_[var_key] = {tensor_desc, dev_ptr, offset, RT_MEMORY_HBM, op_desc};
  }

  cur_var_tensor_desc_map_[GetBatchVarKeyName(var_name)] = tensor_desc;
}

ge::Status VarResource::SaveVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                    const uint8_t *const address, const rtMemType_t memory_type,
                                    const OpDescPtr &op_desc) {
  const std::string var_key = VarKey(var_name, tensor_desc);
  GE_CHECK_NOTNULL(VarManager::Instance(session_id_));
  GELOGD("VarResource::SaveVarAddr, var_key = %s.", var_key.c_str());
  if (var_addr_mgr_map_.count(var_key) == 0U) {
    uint64_t logic_address = PtrToValue(address);
    if (memory_type == RT_MEMORY_HBM) {
      logic_address += VarManager::Instance(session_id_)->GetVarMemLogicBase();
    }
    GELOGI("SaveVarAddr node_name %s, tensor_desc format %s, type %s.", var_name.c_str(),
           TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());
    var_addr_mgr_map_[var_key] = {tensor_desc, PtrToPtr<void, uint8_t>(ValueToPtr(logic_address)), PtrToValue(address),
                                  memory_type, op_desc};
    var_offset_map_[logic_address] = memory_type;
    return SUCCESS;
  }

  REPORT_INNER_ERROR("E19999", "var_key:%s conflict in var_addr_mgr_map_, var_name:%s, session_id:%" PRIu64 ", "
                     "check invalid", var_key.c_str(), var_name.c_str(),
                     session_id_);
  GELOGE(FAILED, "[Check][Param] var_key:%s conflict in var_addr_mgr_map_, var_name:%s, session_id:%" PRIu64 "",
         var_key.c_str(), var_name.c_str(), session_id_);
  return FAILED;
}

bool VarResource::IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const {
  const std::string var_key = VarKey(var_name, tensor_desc);
  return var_addr_mgr_map_.count(var_key) != 0U;
}

bool VarResource::IsVarExist(const std::string &var_name) const {
  return cur_var_tensor_desc_map_.count(GetBatchVarKeyName(var_name)) != 0U;
}

void VarResource::SetVarIsReady(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) {
  std::string var_key = VarKey(var_name, tensor_desc);
  (void)var_is_instance_.emplace(var_key);
  return;
}

bool VarResource::IsVarReady(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const {
  return var_is_instance_.count(VarKey(var_name, tensor_desc)) != 0U;
}

std::string VarResource::VarKey(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const {
  std::string var_key(GetBatchVarKeyName(var_name));
  (void)var_key.append(std::to_string(static_cast<int32_t>(tensor_desc.GetFormat())))
    .append("_")
    .append(std::to_string(static_cast<int32_t>(tensor_desc.GetDataType())));
  return var_key;
}

std::string VarResource::GetBatchVarKeyName(const std::string &var_name) const {
  const auto iter = batch_var_name_map_.find(var_name);
  return (iter == batch_var_name_map_.end()) ? (var_name) : (iter->second);
}

ge::Status VarResource::GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc) {
  const auto var_key_name = GetBatchVarKeyName(var_name);
  if (cur_var_tensor_desc_map_.count(var_key_name) == 0U) {
    return FAILED;
  }
  tensor_desc = cur_var_tensor_desc_map_[var_key_name];
  return SUCCESS;
}

ge::Status VarResource::RenewCurVarDesc(const std::string &var_name, const ge::OpDescPtr &op_desc) {
  const auto var_key_name = GetBatchVarKeyName(var_name);
  if (cur_var_tensor_desc_map_.count(var_key_name) == 0U) {
    GELOGI("There is no this node[%s] key[%s] in var tensor_desc map. so no need renew!",
           var_name.c_str(), var_key_name.c_str());
    return SUCCESS;
  }

  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param op_desc is nullptr, var_name:%s, session_id:%" PRIu64 ", check invalid",
                       var_name.c_str(), session_id_);
    GELOGE(FAILED, "[Check][Param] input opdesc is nullptr, var_name:%s, session_id:%" PRIu64 "",
           var_name.c_str(), session_id_);
    return FAILED;
  }

  ge::GeTensorDesc curr_desc;
  const ge::Status ret = GetCurVarDesc(var_name, curr_desc);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][CurVarDesc] fail, var_name:%s, session_id:%" PRIu64 "", var_name.c_str(), session_id_);
    return FAILED;
  }
  std::string key = VarKey(var_name, curr_desc);
  curr_desc.SetOriginFormat((op_desc->GetOutputDesc(0U)).GetOriginFormat());
  curr_desc.SetFormat((op_desc->GetOutputDesc(0U)).GetFormat());
  cur_var_tensor_desc_map_[var_key_name] = curr_desc;
  const auto iter = var_addr_mgr_map_.find(key);
  if (iter == var_addr_mgr_map_.end()) {
    REPORT_INNER_ERROR("E19999", "var_key:%s can't find in var_addr_mgr_map_, var_name:%s, "
                       "session_id:%" PRIu64 ", op:%s(%s), check invalid", key.c_str(), var_name.c_str(),
                       session_id_, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] var_key:%s can't find in var_addr_mgr_map_, var_name:%s, "
           "session_id:%" PRIu64 ", op:%s(%s)", key.c_str(), var_name.c_str(), session_id_,
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  auto val = iter->second;
  val.tensor_desc.SetOriginFormat((op_desc->GetOutputDesc(0U)).GetOriginFormat());
  val.tensor_desc.SetFormat((op_desc->GetOutputDesc(0U)).GetFormat());
  (void)var_addr_mgr_map_.erase(iter);
  key = VarKey(var_name, curr_desc);
  var_addr_mgr_map_[key] = val;

  return SUCCESS;
}

void VarResource::SaveBroadCastInfo(const uint32_t graph_id, const VarBroadCastInfo &broad_cast_info) {
  var_broad_cast_info_[graph_id][broad_cast_info.var_name] = broad_cast_info;
}

bool VarResource::IsVarAddr(const int64_t &offset) const {
  return var_offset_map_.count(static_cast<uint64_t>(offset)) > 0U;
}

rtMemType_t VarResource::GetVarMemType(const int64_t &offset) {
  if (var_offset_map_.count(static_cast<uint64_t>(offset)) > 0U) {
    return var_offset_map_[static_cast<uint64_t>(offset)];
  }
  return RT_MEMORY_RESERVED;
}

VarTransRoad *VarResource::GetTransRoad(const std::string &var_name) {
  const auto iter = var_to_trans_road_.find(var_name);
  if (iter == var_to_trans_road_.end()) {
    return nullptr;
  } else {
    return &(iter->second);
  }
}

Status VarResource::GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) const {
  const auto iter = var_names_to_changed_graph_id_.find(GetBatchVarKeyName(var_name));
  if (iter == var_names_to_changed_graph_id_.end()) {
    return FAILED;
  } else {
    graph_id = iter->second;
    return SUCCESS;
  }
}
Status VarResource::GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) const {
  const auto iter = var_names_to_allocated_graph_id_.find(GetBatchVarKeyName(var_name));
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
  var_names_to_allocated_graph_id_[GetBatchVarKeyName(var_name)] = graph_id;
  return SUCCESS;
}

Status VarResource::VarResourceToSerial(deployer::VarResourceInfo *const var_resource_info) const {
  GELOGD("[VarResource] Begin to serial var_resource object.");
  GE_CHECK_NOTNULL(var_resource_info);
  for (auto &info : var_offset_map_) {
    (void)var_resource_info->mutable_var_offset_map()->insert({info.first, info.second});
  }

  for (auto &info : var_addr_mgr_map_) {
    deployer::VarAddrMgrInfo  var_addr_mgr_info;
    GeTensorSerializeUtils::GeTensorDescAsProto(info.second.tensor_desc, var_addr_mgr_info.mutable_desc());
    var_addr_mgr_info.set_address(PtrToValue(info.second.address));
    var_addr_mgr_info.set_offset(info.second.offset);
    var_addr_mgr_info.set_memory_type(static_cast<uint64_t>(info.second.memory_type));
    (void)var_resource_info->mutable_var_addr_mgr_map()->insert({info.first, var_addr_mgr_info});
  }

  for (auto &info : cur_var_tensor_desc_map_) {
    proto::TensorDescriptor tensor_desc_proto;
    GeTensorSerializeUtils::GeTensorDescAsProto(info.second, &tensor_desc_proto);
    (void)var_resource_info->mutable_cur_var_tensor_desc_map()->insert({info.first, tensor_desc_proto});
  }

  for (auto &info : var_to_trans_road_) {
    deployer::TransNodeMultiInfo trans_node_info;
    for (auto &x : info.second) {
      deployer::SingleTransNodeInfo *const single_info = trans_node_info.add_node_info();
      single_info->set_node_type(x.node_type);
      GeTensorSerializeUtils::GeTensorDescAsProto(x.input, single_info->mutable_input());
      GeTensorSerializeUtils::GeTensorDescAsProto(x.output, single_info->mutable_output());
    }
    (void)var_resource_info->mutable_var_to_trans_road()->insert({info.first, trans_node_info});
  }

  for (auto &info : var_names_to_changed_graph_id_) {
    (void)var_resource_info->mutable_var_names_to_changed_graph_id()->insert({info.first, info.second});
  }

  for (auto &info : var_names_to_allocated_graph_id_) {
    (void)var_resource_info->mutable_var_names_to_allocated_graph_id()->insert({info.first, info.second});
  }

  for (auto &info : var_broad_cast_info_) {
    deployer::BroadcastMultiInfo broadcast_multi_info;
    for (auto &x : info.second) {
      deployer::BroadcastInfo broadcast_info;
      broadcast_info.set_var_name(x.second.var_name);
      broadcast_info.set_broadcast_name(x.second.broadcast_name);
      broadcast_info.set_idx(x.second.idx);
      broadcast_info.set_input_offset(x.second.input_offset);
      broadcast_info.set_input_size(x.second.input_size);
      broadcast_info.set_output_offset(x.second.output_offset);
      broadcast_info.set_output_size(x.second.output_size);
      (void)broadcast_multi_info.mutable_broadcast_info()->insert({x.first, broadcast_info});
    }
    (void)var_resource_info->mutable_var_broad_cast_info()->insert({info.first, broadcast_multi_info});
  }
  GELOGD("[VarResource] Success to serial var_resource object.");
  return SUCCESS;
}

Status VarResource::VarResourceToDeserial(const deployer::VarResourceInfo *const var_resource_info) {
  GELOGD("[VarResource] Begin to deserial var_resource object.");
  GE_CHECK_NOTNULL(var_resource_info);
  auto name_changed_graph_id_map = var_resource_info->var_names_to_changed_graph_id();
  auto name_alloc_graph_id_map = var_resource_info->var_names_to_allocated_graph_id();
  for (const auto &x : var_resource_info->var_offset_map()) {
    (void)var_offset_map_.insert(std::pair<uint64_t, rtMemType_t>(x.first, x.second));
  }

  for (const auto &x : var_resource_info->var_addr_mgr_map()) {
    GeTensorDesc tensor_desc;
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&x.second.desc(), tensor_desc);
    const struct VarAddrMgr addr_mgr = {tensor_desc, PtrToPtr<void, uint8_t>(ValueToPtr(x.second.address())),
        static_cast<uint64_t>(x.second.offset()), static_cast<rtMemType_t>(x.second.memory_type()), nullptr};
    (void)var_addr_mgr_map_.insert(std::pair<std::string, VarAddrMgr>(x.first, addr_mgr));
  }

  for (const auto &x : var_resource_info->cur_var_tensor_desc_map()) {
    GeTensorDesc tensor_desc;
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&x.second, tensor_desc);
    (void)cur_var_tensor_desc_map_.insert(std::pair<std::string, GeTensorDesc>(x.first, tensor_desc));
  }

  for (const auto &x : var_resource_info->var_to_trans_road()) {
    std::vector<TransNodeInfo> trans_node_info_vec;
    for (auto i = 0; i < x.second.node_info_size(); i++) {
      TransNodeInfo trans_node_info;
      trans_node_info.node_type = x.second.node_info(i).node_type();
      const proto::TensorDescriptor &input_tensor_desc = x.second.node_info(i).input();
      const proto::TensorDescriptor &output_tensor_desc = x.second.node_info(i).output();
      GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&input_tensor_desc, trans_node_info.input);
      GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&output_tensor_desc, trans_node_info.output);
      trans_node_info_vec.emplace_back(trans_node_info);
    }
    (void)var_to_trans_road_.insert(std::pair<std::string, std::vector<TransNodeInfo>>(x.first, trans_node_info_vec));
  }
  var_names_to_changed_graph_id_.insert(name_changed_graph_id_map.begin(), name_changed_graph_id_map.end());
  var_names_to_allocated_graph_id_.insert(name_alloc_graph_id_map.begin(), name_alloc_graph_id_map.end());
  for (const auto &x : var_resource_info->var_broad_cast_info()) {
    std::unordered_map<std::string, VarBroadCastInfo> var_broadcast_info;
    const deployer::BroadcastMultiInfo &boardcast_multi_info = x.second;
    for (const auto &broadcast_info : boardcast_multi_info.broadcast_info()) {
      const auto &bc = broadcast_info.second;
      const struct VarBroadCastInfo info = {bc.var_name(),   bc.broadcast_name(), bc.idx(),        bc.input_offset(),
                                            bc.input_size(), bc.output_offset(),  bc.output_size()};
      (void)var_broadcast_info.insert(std::pair<std::string, VarBroadCastInfo>(broadcast_info.first, info));
    }
    (void)var_broad_cast_info_.insert(
        std::pair<uint32_t, std::unordered_map<std::string, VarBroadCastInfo>>(x.first, var_broadcast_info));
  }
  GELOGD("[VarResource] Success to deserial var_resource object.");
  return SUCCESS;
}

void VarResource::SetBatchVariablesKeyName(const std::string &batch_var_name, const std::string &key_name) {
  batch_var_name_map_[batch_var_name] = key_name;
}

bool VarResource::HasSharedVarMemBetweenBatch() const {
  return !batch_var_name_map_.empty();
}

MemResource::MemResource() : total_size_(0U), var_mem_size_(0U) {}

std::shared_ptr<MemResource> MemResource::BuildMemResourceFromType(const rtMemType_t mem_type) {
  std::shared_ptr<MemResource> resource = nullptr;
  switch (mem_type) {
    case RT_MEMORY_HBM:
      resource = MakeShared<HbmMemResource>();
      break;
    case RT_MEMORY_RDMA_HBM:
      resource = MakeShared<RdmaMemResource>();
      break;
    case RT_MEMORY_HOST:
      resource = MakeShared<HostMemResource>();
      break;
    default:
      break;
  }
  return resource;
}

Status HbmMemResource::AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                                    size_t &mem_offset) {
  FMK_UINT64_ADDCHECK(size, kSessionMemAlignSize);
  uint64_t align_size = (size + kSessionMemAlignSize - 1U) / kSessionMemAlignSize * kSessionMemAlignSize;
  const uint64_t real_size = align_size;
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  total_size_ = VarManager::Instance(session_id)->GetVarMemMaxSize(true);
  if (total_size_ < var_mem_size_) {
    REPORT_INNER_ERROR("E19999", "VarMemMaxSize:%" PRIu64 " < var_mem_size_:%" PRIu64 ", var_size:"
                       "%" PRIu64 ", var_name:%s, check invalid"
                       "", total_size_, var_mem_size_, align_size, var_name.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] total_size_:%" PRIu64 " is smaller than var_mem_size_:"
           "%" PRIu64 ", var_name:%s", total_size_, var_mem_size_, var_name.c_str());
    return PARAM_INVALID;
  }
  const uint64_t free_size = total_size_ - var_mem_size_;
  FMK_UINT64_ADDCHECK(align_size, (kSessionMemAlignSize * kSessionMemAlignUnit));
  if (free_size < (align_size + (kSessionMemAlignSize * kSessionMemAlignUnit))) {
    REPORT_INNER_ERROR("E19999", "VarMemMaxSize:%" PRIu64" free_size:%" PRIu64 " not enough, var_align_size:%" PRIu64 ""
                       ", var_name:%s, check invalid", total_size_, free_size, align_size, var_name.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Out of memory: current var size["
           "%" PRIu64 "] exceeds total var size[%" PRIu64 "]",
           align_size + (kSessionMemAlignSize * kSessionMemAlignUnit) + var_mem_size_, total_size_);
    return PARAM_INVALID;
  }

  mem_offset = var_mem_size_;

  // offset for next, align 512 BYTE
  FMK_UINT64_ADDCHECK(align_size, kSessionMemAlignSize);
  align_size = align_size + kSessionMemAlignSize;
  FMK_UINT64_ADDCHECK(var_mem_size_, align_size);
  var_mem_size_ = var_mem_size_ + align_size;

  // align 512 BYTE
  FMK_UINT64_ADDCHECK(var_mem_size_, kSessionMemAlignSize);
  var_mem_size_ = var_mem_size_ + kSessionMemAlignSize;
  GELOGI("[IMAS]AssignVarMem Set session_%" PRIu64 " name[%s] output[%d] offset to [%zu] size["
         "%" PRIu64 "] realsize[%" PRIu64 "].", session_id, var_name.c_str(), 0,
	 mem_offset, (var_mem_size_ - mem_offset), real_size);
  return SUCCESS;
}

Status RdmaMemResource::AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                                     size_t &mem_offset) {
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  uint8_t *const buffer = VarManager::Instance(session_id)->GetRdmaPoolMemory(RT_MEMORY_HBM, size);
  if (buffer == nullptr) {
    REPORT_CALL_ERROR("E19999", "malloc rdma memory fail, var_size:%" PRIu64 ", var_name:%s",
                      size, var_name.c_str());
    GELOGE(MEMALLOC_FAILED, "[Malloc][RdmaMemory] for node %s failed, size = %" PRIu64 "", var_name.c_str(), size);
    return MEMALLOC_FAILED;
  }
  mem_offset = static_cast<size_t>(PtrToValue(buffer));
  FMK_UINT64_ADDCHECK(var_mem_size_, size);
  var_mem_size_ += size;
  GELOGI("[IMAS]AssignVarMem Set session_%" PRIu64 " name[%s] output[%d] addr to [%p] size[%" PRIu64 "].",
         session_id, var_name.c_str(), 0, buffer, size);
  return SUCCESS;
}

Status HostMemResource::AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                                     size_t &mem_offset) {
  GELOGD("Start to malloc host memory, size=%zu.", size);
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  uint8_t *const buffer = VarManager::Instance(session_id)->GetHostPoolMemory(RT_MEMORY_HBM, size);
  if (buffer == nullptr) {
    REPORT_CALL_ERROR("E19999", "malloc host memory fail, var_size:%" PRIu64 ", var_name:%s",
                      size, var_name.c_str());
    GELOGE(MEMALLOC_FAILED, "[Malloc][HostMemory] for node %s failed, size = %" PRIu64 "", var_name.c_str(), size);
    return MEMALLOC_FAILED;
  }
  mem_offset = static_cast<size_t>(PtrToValue(buffer));
  FMK_UINT64_ADDCHECK(var_mem_size_, size);
  var_mem_size_ += size;
  GELOGI("[IMAS]AssignVarMem Set session_%" PRIu64 " name[%s] output[%zu] size[%lu]",
         session_id, var_name.c_str(), mem_offset, size);
  return SUCCESS;
}

uint64_t MemResource::GetVarMemSize() const { return var_mem_size_; }

void MemResource::UpdateVarMemSize(const int64_t mem_size) { var_mem_size_ = static_cast<uint64_t>(mem_size); };

VarManager::VarManager(const uint64_t session_id)
    : version_(SessionVersion::OTHER_VERSION),
      session_id_(session_id),
      device_id_(kDefaultDeviceId),
      job_id_(0U),
      graph_mem_max_size_(kGraphMemoryManagerMallocMaxSize),
      var_mem_max_size_(kMemoryVarManagerMallocSize),
      var_mem_logic_base_(kMemoryVarLogicBase),
      use_max_mem_size_(kUseMaxMemorySize) {}

std::shared_ptr<VarManager> VarManager::Instance(const uint64_t session_id) {
  GELOGD("VarManager::Instance, session id = %" PRIu64 ".", session_id);
  return VarManagerPool::Instance().GetVarManager(session_id);
}

void VarManager::Destory() {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::Destory, session id = %" PRIu64 ".", session_id_);
  version_ = SessionVersion::OTHER_VERSION;
  device_id_ = kDefaultDeviceId;
  session_id_ = 0U;
  mem_resource_map_.clear();
}

Status VarManager::Init(const uint32_t version, const uint64_t session_id, const uint32_t device_id,
                        const uint64_t job_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::Init, session id = %" PRIu64 ".", session_id);
  if (var_resource_ == nullptr) {
    version_ = static_cast<SessionVersion>(version);
    device_id_ = device_id;
    session_id_ = session_id;
    job_id_ = job_id;
    var_resource_ = MakeShared<VarResource>(session_id_);
    if (var_resource_ == nullptr) {
      GELOGW("VarManager init failed session id = %" PRIu64 ".", session_id);
      return ge::INTERNAL_ERROR;
    }
  } else {
    GELOGW("VarManager::has been inited, session id = %" PRIu64 ".", session_id);
  }
  return SUCCESS;
}

const uint64_t &VarManager::SessionId() const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  return session_id_;
}

ge::Status VarManager::SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                  const uint8_t *const dev_ptr, const rtMemType_t memory_type,
                                  const OpDescPtr &op_desc) {
  GELOGI("VarManager::SetVarAddr var_name = %s, data_type = %s, data_format = %s.", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  var_resource_->SetVarAddr(var_name, tensor_desc, dev_ptr, memory_type, op_desc);
  return ge::SUCCESS;
}

ge::Status VarManager::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                  uint8_t *&dev_ptr, rtMemType_t &memory_type) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::GetVarAddr var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  const auto ret = var_resource_->GetVarAddr(var_name, tensor_desc, &dev_ptr, memory_type);
  if (ret != SUCCESS) {
    GELOGW("GetVarAddr fail.");
    return ge::INTERNAL_ERROR;
  }
  return SUCCESS;
}

ge::Status VarManager::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                  uint8_t *&dev_ptr) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  rtMemType_t memory_type = RT_MEMORY_HBM;
  return GetVarAddr(var_name, tensor_desc, dev_ptr, memory_type);
}

int64_t VarManager::GetVarMemSize(const rtMemType_t memory_type) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::shared_ptr<MemResource> mem_resource = nullptr;
  const auto iter = mem_resource_map_.find(memory_type);
  if (iter == mem_resource_map_.end()) {
    return 0;
  } else {
    mem_resource = iter->second;
  }

  if (mem_resource == nullptr) {
    REPORT_INNER_ERROR("E19999", "Find no mem_resource in map, memory_type:%d, session_id:%" PRIu64 "",
                       memory_type, session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] MemResource is invalid, memory_type:%d, session_id:%" PRIu64 "",
           memory_type, session_id_);
    return 0;
  }
  return static_cast<int64_t>(mem_resource->GetVarMemSize());
}

ge::Status VarManager::AssignVarMem(const std::string &var_name, const OpDescPtr &op_desc,
                                    const ge::GeTensorDesc &tensor_desc, rtMemType_t memory_type) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::AssignVarMem var_name = %s, data_type = %s, data_format = %s.", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  int64_t tensor_desc_size = 0;
  ge::Status result = TensorUtils::GetSize(tensor_desc, tensor_desc_size);
  if (result != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get size from tensor fail, var_name:%s, memory_type:%d, session_id:%" PRIu64 "",
                      var_name.c_str(), memory_type, session_id_);
    GELOGE(result, "[Get][Size] from tensor fail, var_name:%s, memory_type:%u, session_id:%" PRIu64 "",
           var_name.c_str(), memory_type, session_id_);
    return result;
  }

  std::shared_ptr<MemResource> mem_resource = nullptr;
  const auto it = mem_resource_map_.find(memory_type);
  if (it == mem_resource_map_.end()) {
    mem_resource = MemResource::BuildMemResourceFromType(memory_type);
    GE_CHECK_NOTNULL(mem_resource);
    mem_resource_map_[memory_type] = mem_resource;
  } else {
    mem_resource = it->second;
  }

  GE_CHECK_NOTNULL(mem_resource);
  GE_CHECK_NOTNULL(var_resource_);

  ge::GeTensorDesc cur_tensor_desc;
  int64_t cur_tensor_desc_size = 0;
  uint8_t *mem_offset = nullptr;
  result = var_resource_->GetCurVarDesc(var_name, cur_tensor_desc);
  // reuse old format variable memory
  if (result == SUCCESS) {
    result = var_resource_->GetVarAddr(var_name, cur_tensor_desc, &mem_offset, memory_type);
    if (result == SUCCESS) {
      result = TensorUtils::GetSize(cur_tensor_desc, cur_tensor_desc_size);
      GELOGD("tensor_desc_size is %ld, cur_tensor_desc_size is %ld, memoffset is %" PRIu64 "", tensor_desc_size,
             cur_tensor_desc_size, PtrToValue(mem_offset));
    }
  } else {
    result = var_resource_->GetReuseAddr(op_desc, &mem_offset, memory_type);
    if (result == SUCCESS) {
      cur_tensor_desc_size = tensor_desc_size;
    }
  }

  const bool can_not_reuse_old_memory = (result != SUCCESS) || (tensor_desc_size > cur_tensor_desc_size);
  if (can_not_reuse_old_memory) {
    size_t tmp_mem_offset = 0UL;
    result = mem_resource->AssignVarMem(var_name, static_cast<uint64_t>(tensor_desc_size), session_id_, tmp_mem_offset);
    if (result != SUCCESS) {
      GELOGE(ge::INTERNAL_ERROR, "[Assign][VarMem] by offset failed, session_id:%" PRIu64 ".", session_id_);
      return ge::INTERNAL_ERROR;
    }

    mem_offset = PtrToPtr<void, uint8_t>(ValueToPtr(tmp_mem_offset));
    result = var_resource_->SaveVarAddr(var_name, tensor_desc, mem_offset, memory_type, op_desc);
    if (result != SUCCESS) {
      GELOGE(ge::INTERNAL_ERROR, "[Save][VarAddr] by offset failed, memory type:%u, session_id:%" PRIu64 ".",
             memory_type, session_id_);
      return ge::INTERNAL_ERROR;
    }
  }
  // old not exist only save new tensor
  result = var_resource_->GetCurVarDesc(var_name, cur_tensor_desc);
  if (result != SUCCESS) {
    var_resource_->SetVarAddr(var_name, tensor_desc, mem_offset, memory_type, op_desc);
    return SUCCESS;
  }
  const bool format_changed = (cur_tensor_desc.GetFormat() != tensor_desc.GetFormat()) ||
                              (cur_tensor_desc.GetDataType() != tensor_desc.GetDataType()) ||
                              (cur_tensor_desc.GetShape().GetDims() != tensor_desc.GetShape().GetDims());
  if (format_changed) {
    GELOGI("var %s assigned new memory (format, data type, shape)  (%s, %s, %zu) from (%s, %s, %zu)", var_name.c_str(),
           ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
           ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
           tensor_desc.GetShape().GetDims().size(),
           ge::TypeUtils::DataTypeToSerialString(cur_tensor_desc.GetDataType()).c_str(),
           ge::TypeUtils::FormatToSerialString(cur_tensor_desc.GetFormat()).c_str(),
           cur_tensor_desc.GetShape().GetDims().size());
    var_resource_->SetVarAddr(var_name, tensor_desc, mem_offset, memory_type, op_desc);
  }

  return SUCCESS;
}

void VarManager::SetVarIsReady(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::SetVarIsReady var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return;
  }
  var_resource_->SetVarIsReady(var_name, tensor_desc);
  return;
}

bool VarManager::IsVarReady(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::IsVarReady var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarReady(var_name, tensor_desc);
}

bool VarManager::IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::IsVarExist var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarExist(var_name, tensor_desc);
}

bool VarManager::IsVarExist(const std::string &var_name) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarExist(var_name);
}


ge::Status VarManager::VarManagerToSerial(const uint64_t session_id, deployer::VarManagerInfo &info) const {
  GELOGD("[VarManager] Begin to serial var manager objection, the session id is %" PRIu64 ".", session_id);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "Var manager has not been inited.");
    return INTERNAL_ERROR;
  }

  info.set_version(static_cast<uint32_t>(version_));
  info.set_session_id(session_id_);
  info.set_device_id(device_id_);
  info.set_job_id(job_id_);
  info.set_graph_mem_max_size(graph_mem_max_size_);
  info.set_var_mem_max_size(var_mem_max_size_);
  info.set_var_mem_logic_base(var_mem_logic_base_);
  info.set_use_max_mem_size(use_max_mem_size_);
  deployer::VarResourceInfo *const var_resource_info = info.mutable_var_resource();
  (void)var_resource_->VarResourceToSerial(var_resource_info);

  auto const resource_map = info.mutable_mem_resource_map();
  for (auto &mem_resource : mem_resource_map_) {
    deployer::MemResourceInfo source_info;
    if (mem_resource.second == nullptr) {
      REPORT_INNER_ERROR("E19999", "Find no mem_resource in map, memory_type:%d, session_id:"
		         "%" PRIu64 ".", mem_resource.first, session_id_);
      GELOGE(ge::INTERNAL_ERROR, "[Check][Param] MemResource is invalid, memory_type:%d, "
		                 "session_id:%" PRIu64 ".", mem_resource.first, session_id_);
      return INTERNAL_ERROR;
    }
    source_info.set_var_mem_size(mem_resource.second->GetVarMemSize());
    (void)resource_map->insert({mem_resource.first, source_info});
  }
  GELOGD("[VarManager] Success to serial var manager objection, the session id is %" PRIu64 ".", session_id);
  return SUCCESS;
}

ge::Status VarManager::VarManagerToDeserial(const uint64_t session_id, const deployer::VarManagerInfo &info) {
  GELOGD("[VarManager] Begin to deserial var manager objection, the session id is %" PRIu64 ".", session_id);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    version_ = static_cast<SessionVersion>(info.version());
    int32_t device_id = -1;
    GE_CHK_RT_RET(rtGetDevice(&device_id));
    device_id_ = static_cast<uint32_t>(device_id);
    GELOGD("[VarManager] Success to get device id = %u", device_id_);
    session_id_ = info.session_id();
    job_id_ = info.job_id();
    UpdateMemoryConfig(info.graph_mem_max_size(), info.var_mem_max_size(), info.var_mem_logic_base(),
                       info.use_max_mem_size());
    var_resource_ = MakeShared<VarResource>(session_id_);
    if (var_resource_ == nullptr) {
      GELOGE(ge::INTERNAL_ERROR, "VarManager init failed session id = %" PRIu64 ".", session_id);
      return ge::INTERNAL_ERROR;
    }
  }

  (void)var_resource_->VarResourceToDeserial(&info.var_resource());

  for (const auto &x : info.mem_resource_map()) {
    const rtMemType_t memory_type = x.first;
    std::shared_ptr<MemResource> mem_resource = nullptr;
    const auto it = mem_resource_map_.find(memory_type);
    if (it == mem_resource_map_.end()) {
      mem_resource = MemResource::BuildMemResourceFromType(memory_type);
      if (mem_resource == nullptr) {
        REPORT_INNER_ERROR("E19999", "Failed to build mem_resource, memory_type:%d, session_id:"
			   "%" PRIu64 ".", memory_type, session_id_);
        GELOGE(ge::INTERNAL_ERROR, "Failed to build mem_resource, memory_type:%d, session_id:"
			           "%" PRIu64 ".", memory_type, session_id_);
        return INTERNAL_ERROR;
      } else {
        mem_resource_map_[memory_type] = mem_resource;
      }
    } else {
      mem_resource = it->second;
    }
    mem_resource->UpdateVarMemSize(static_cast<int64_t>(x.second.var_mem_size()));
  }
  GELOGD("[VarManager] Success to deserial var manager objection, the session id is "
         "%" PRIu64 ".", session_id);
  return SUCCESS;
}

ge::Status VarManager::GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::GetCurVarDesc var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->GetCurVarDesc(var_name, tensor_desc);
}

ge::Status VarManager::SaveBroadCastInfo(const uint32_t graph_id, const VarBroadCastInfo &broad_cast_info) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::SaveBroadCastInfo var_name = %s, broadcast name = %s, "
         "idx = %d, input_offset = %ld, input_size = %" PRIu64 ", output_offset = %ld, output_size = %" PRIu64 "",
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
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::RenewCurVarDesc var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "VarManager has not been init, op:%s(%s), session_id:%" PRIu64 ", check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] VarManager has not been init, op:%s(%s), session_id:%" PRIu64 "",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), session_id_);
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->RenewCurVarDesc(var_name, std::move(op_desc));
}

bool VarManager::IsVarAddr(const int64_t &offset) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGD("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarAddr(offset);
}

rtMemType_t VarManager::GetVarMemType(const int64_t &offset) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return RT_MEMORY_RESERVED;
  }
  return var_resource_->GetVarMemType(offset);
}

void VarManager::SetMemManager(MemoryManager *const mem_manager) {
  // Better use shared_ptr instead, reconsitution later.
  GELOGI("Set MemManager to VarManager.");
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  mem_manager_ = mem_manager;
}

ge::Status VarManager::MallocVarMemory(const uint64_t memory_size, const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGE(FAILED, "[Check][Param] MemManager has not been init.");
    REPORT_INNER_ERROR("E19999", "MemManager has not been init, session_id: %" PRIu64 "", session_id_);
    return FAILED;
  }
  uint8_t *var_mem_base = nullptr;
  const std::string memory_key = std::to_string(session_id_);

  // malloc variable memory
  size_t var_memory_size = memory_size;

  // align 512 BYTE
  FMK_SIZET_ADDCHECK(var_memory_size, kSessionMemAlignSize);
  var_memory_size = (var_memory_size + kSessionMemAlignSize - 1U) / kSessionMemAlignSize * kSessionMemAlignSize;
  const std::string purpose("variables and constant op memory in training network.");
  device_id_ = device_id;
  GELOGI("Start malloc var mem on device %u", device_id_);
  var_mem_base = mem_manager_->MallocMemory(RT_MEMORY_HBM, purpose, memory_key, var_memory_size, device_id_);
  if (var_mem_base == nullptr) {
    GELOGE(ge::INTERNAL_ERROR, "[Malloc][VarMemory] failed, size:%zu, session_id:%s",
           var_memory_size, memory_key.c_str());
    return ge::INTERNAL_ERROR;
  }
  return SUCCESS;
}

uint8_t *VarManager::GetVarMemoryBase(const rtMemType_t memory_type, const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGE(FAILED, "[Check][Param] MemManager has not been init.");
    REPORT_INNER_ERROR("E19999", "MemManager has not been init, session_id: %" PRIu64 "", session_id_);
    return nullptr;
  }
  const std::string memory_key = std::to_string(session_id_);
  return mem_manager_->GetMemoryBase(memory_type, memory_key, device_id);
}

uint8_t *VarManager::GetVarMemoryAddr(uint8_t *const logic_addr, const rtMemType_t memory_type,
                                      const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGE(FAILED, "[Check][Param] MemManager has not been init.");
    REPORT_INNER_ERROR("E19999", "MemManager has not been init, session_id: %" PRIu64 "", session_id_);
    return nullptr;
  }

  if (memory_type == RT_MEMORY_RDMA_HBM) {
    return logic_addr;
  }
  const std::string mem_key = std::to_string(session_id_);
  uint8_t *const mem_base = mem_manager_->GetMemoryAddr(memory_type, mem_key, device_id);
  if (mem_base == nullptr) {
    return nullptr;
  }
  const uint64_t mem_addr =
      PtrToValue(logic_addr) + (PtrToValue(mem_base) - VarManager::Instance(session_id_)->GetVarMemLogicBase());
  return PtrToPtr<void, uint8_t>(ValueToPtr(mem_addr));
}

ge::Status VarManager::FreeVarMemory() {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGW("MemManager is nullptr please check if it has been initialized.");
    return FAILED;
  }

  const std::string memory_key = std::to_string(SessionId());
  return mem_manager_->FreeMemory(RT_MEMORY_HBM, memory_key, device_id_);
}

uint8_t *VarManager::GetRdmaPoolMemory(const rtMemType_t memory_type, const size_t mem_size) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGE(FAILED, "[Check][Param] MemManager has not been init.");
    REPORT_INNER_ERROR("E19999", "MemManager has not been init, session_id: %" PRIu64 "", session_id_);
    return nullptr;
  }

  return mem_manager_->GetRdmaPoolMemory(memory_type, mem_size, device_id_);
}

uint8_t *VarManager::GetHostPoolMemory(const rtMemType_t memory_type, const size_t mem_size) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGE(FAILED, "[Check][Param] MemManager has not been init.");
    REPORT_INNER_ERROR("E19999", "MemManager has not been init, session_id: %" PRIu64 "", session_id_);
    return nullptr;
  }

  return mem_manager_->GetHostPoolMemory(memory_type, mem_size);
}

ge::Status VarManager::SetTransRoad(const std::string &var_name, const VarTransRoad &trans_road) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->SetTransRoad(var_name, trans_road);
}

VarTransRoad *VarManager::GetTransRoad(const std::string &var_name) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return nullptr;
  }
  return var_resource_->GetTransRoad(var_name);
}

Status VarManager::SetChangedGraphId(const std::string &var_name, const uint32_t graph_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->SetChangedGraphId(var_name, graph_id);
}

Status VarManager::GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->GetChangedGraphId(var_name, graph_id);
}

void VarManager::UpdateMemoryConfig(const size_t graph_mem_max_size, const size_t var_mem_max_size,
                                    const size_t var_mem_logic_base, const size_t use_max_mem_size) {
  graph_mem_max_size_ = graph_mem_max_size;
  var_mem_max_size_ = var_mem_max_size;
  var_mem_logic_base_ = var_mem_logic_base;
  use_max_mem_size_ = use_max_mem_size;
}

Status VarManager::SetAllMemoryMaxValue(const std::map<std::string, std::string> &options) {
  const auto it1 = options.find(GRAPH_MEMORY_MAX_SIZE);
  if (it1 != options.end()) {
    const std::string graph_memory_manager_malloc_max_size = it1->second;
    const ge::Status ret = ParseMemoryMallocSize(graph_memory_manager_malloc_max_size, graph_mem_max_size_);
    if (ret != SUCCESS) {
      GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Call][ParseMemoryMallocSize] failed, session id:"
		                           "%" PRIu64 ".", session_id_);
      return ge::GE_GRAPH_OPTIONS_INVALID;
    }
  }

  const auto it2 = options.find(VARIABLE_MEMORY_MAX_SIZE);
  if (it2 != options.end()) {
    const std::string memory_var_manager_malloc_size = it2->second;
    const ge::Status ret = ParseMemoryMallocSize(memory_var_manager_malloc_size, var_mem_max_size_);
    if (ret != SUCCESS) {
      GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Call][ParseMemoryMallocSize] failed, session id:"
		                           "%" PRIu64 ".", session_id_);
      return ge::GE_GRAPH_OPTIONS_INVALID;
    }
  }

  GEEVENT("The graph_mem_max_size is %zu and the var_mem_max_size is %zu", graph_mem_max_size_, var_mem_max_size_);

  FMK_SIZET_ADDCHECK(graph_mem_max_size_, kGraphMemoryBuffer);
  var_mem_logic_base_ = graph_mem_max_size_ + kGraphMemoryBuffer;
  if (var_mem_logic_base_ > kMaxMemorySize) {
    REPORT_INNER_ERROR("E19999", "var_login_base:%" PRIu64 " can not exeed limit:%" PRIu64 ", "
		       "session_id:%" PRIu64 ", check invalid", var_mem_logic_base_, kMaxMemorySize, session_id_);
    GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Check][Param] kMemoryVarLogicBase:%zu can not exceed "
           "max memory size:%zu, session_id:%" PRIu64 ".", var_mem_logic_base_, kMaxMemorySize, session_id_);
    return ge::GE_GRAPH_OPTIONS_INVALID;
  }

  FMK_SIZET_ADDCHECK(graph_mem_max_size_, var_mem_max_size_);
  use_max_mem_size_ = graph_mem_max_size_ + var_mem_max_size_;
  if (use_max_mem_size_ > kMaxMemorySize) {
    REPORT_INNER_ERROR("E19999", "all mem_use size:%" PRIu64 " can not exeed limit:%" PRIu64 ", "
		       "session_id:%" PRIu64 ", check invalid", use_max_mem_size_, kMaxMemorySize, session_id_);
    GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Check][Param] kUseMaxMemorySize:%zu can not exceed "
           "max memory size:%zu, session_id:%" PRIu64 ".", use_max_mem_size_, kMaxMemorySize, session_id_);
    return ge::GE_GRAPH_OPTIONS_INVALID;
  }
  GELOGI("Set memory malloc size successfully");
  return SUCCESS;
}

Status VarManager::SetMemoryMallocSize(const std::map<std::string, std::string> &options, const size_t total_mem_size) {
  GEEVENT("Total memory size is %zu", total_mem_size);

  graph_mem_max_size_ = static_cast<size_t>(
      floor(static_cast<float64_t>(total_mem_size) * kGraphMemoryManagerMallocRatio));
  var_mem_max_size_ = static_cast<size_t>(floor(static_cast<float64_t>(total_mem_size) * kVarMemoryManagerMallocRatio));

  return SetAllMemoryMaxValue(options);
}

Status VarManager::ParseMemoryMallocSize(const std::string &memory_size, uint64_t &target_size) const {
  if (memory_size.empty()) {
    REPORT_INNER_ERROR("E19999", "Param memory_size is empty, session_id:%" PRIu64 ", check invalid",
                       session_id_);
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] Memory malloc size input is empty, session_id:"
		                     "%" PRIu64 ".", session_id_);
    return GE_GRAPH_OPTIONS_INVALID;
  }
  // split std::string by '*'
  std::vector<std::string> splits;
  std::istringstream str(memory_size);
  std::string str_split;
  while (getline(str, str_split, '*')) {
    splits.emplace_back(str_split);
  }

  target_size = 1U;
  for (std::string split : splits) {
    // Trim
    auto it = split.find_first_not_of(" ");
    if (it != std::string::npos) {
      (void)split.erase(0U, it);
    }
    it = split.find_last_not_of(" ");
    if (it != std::string::npos) {
      (void)split.erase(it + 1U);
    }

    for (const char_t c : split) {
      if (isdigit(static_cast<int32_t>(c)) == 0) {
        REPORT_INNER_ERROR("E19999", "Param memory_size:%s contains non digit, session_id:"
			   "%" PRIu64 ", check invalid", memory_size.c_str(), session_id_);
        GELOGE(GE_GRAPH_OPTIONS_INVALID,
               "[Check][Param] Memory malloc size:%s input contains non digit, session_id:%" PRIu64 ".",
               memory_size.c_str(), session_id_);
        return GE_GRAPH_OPTIONS_INVALID;
      }
    }
    const uint64_t num = std::strtoul(split.c_str(), nullptr, 0);
    if (TypeUtils::CheckUint64MulOverflow(target_size, static_cast<uint32_t>(num))) {
      REPORT_INNER_ERROR("E19999", "Param memory_size:%s will overflow after multi all, session_id:"
		         "%" PRIu64 ", check invalid", memory_size.c_str(), session_id_);
      GELOGE(FAILED, "[Check][Param] Param memory_size:%s will overflow after multi all, session_id:"
		     "%" PRIu64 "", memory_size.c_str(), session_id_);
      return FAILED;
    }

    if ((num > kMaxMemorySize) || ((target_size * static_cast<uint64_t>(num)) > kMaxMemorySize)) {
      REPORT_INNER_ERROR("E19999", "Param memory_size:%s after multi will exceed limit:%" PRIu64 ", "
		         "session_id:%" PRIu64 ", check invalid", memory_size.c_str(), kMaxMemorySize,
                         session_id_);
      GELOGE(FAILED, "[Check][Param] Input memory size can not exceed max memory size:%zu, session_id:"
		     "%" PRIu64 ".", kMaxMemorySize, session_id_);
      return FAILED;
    }
    target_size *= static_cast<size_t>(num);
  }

  return SUCCESS;
}

void VarManager::RemoveChangedGraphId(const std::string &var_name) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return;
  }
  var_resource_->RemoveChangedGraphId(var_name);
}

Status VarManager::SetAllocatedGraphId(const std::string &var_name, const uint32_t graph_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->SetAllocatedGraphId(var_name, graph_id);
}

Status VarManager::GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->GetAllocatedGraphId(var_name, graph_id);
}

Status VarManager::GetAllVariables(std::map<std::string, GeTensorDesc> &all_variables) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been inited.");
    return INTERNAL_ERROR;
  }
  auto new_variable_desc = var_resource_->GetAllVarDesc();
  if (new_variable_desc.size() == 0U) {
    GELOGW("VarManager don't have variables.");
    return INTERNAL_ERROR;
  }

  for (auto iter = new_variable_desc.begin(); iter != new_variable_desc.end(); ++iter) {
    const auto trans_road = var_resource_->GetTransRoad(iter->first);
    if ((trans_road == nullptr) || trans_road->empty()) {
      GELOGI("The variable %s does not have any trans road", iter->first.c_str());
      all_variables[iter->first] = iter->second;
    } else {
      // get origin trans info : the first trans node info
      all_variables[iter->first] = trans_road->at(0U).input;
    }
  }
  return SUCCESS;
}

void VarManager::SetBatchVariablesKeyName(const std::string &batch_var_name, const std::string &key_name) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been inited.");
    return;
  }
  var_resource_->SetBatchVariablesKeyName(batch_var_name, key_name);
}

bool VarManager::HasSharedVarMemBetweenBatch() const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been inited.");
    return false;
  }
  return var_resource_->HasSharedVarMemBetweenBatch();
}

bool VarManager::HasMemoryManager() const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  return mem_manager_ != nullptr;
}

VarManagerPool::~VarManagerPool() { Destory(); }

VarManagerPool &VarManagerPool::Instance() {
  static VarManagerPool var_manager_pool;
  return var_manager_pool;
}

void VarManagerPool::Destory() noexcept {
  const std::lock_guard<std::mutex> lock(var_manager_mutex_);
  for (auto &it : var_manager_map_) {
    if (it.second != nullptr) {
      it.second->Destory();
    }
  }
  var_manager_map_.clear();
}

std::shared_ptr<VarManager> VarManagerPool::GetVarManager(const uint64_t session_id) {
  const std::lock_guard<std::mutex> lock(var_manager_mutex_);
  const std::map<uint64_t, std::shared_ptr<VarManager>>::const_iterator it = var_manager_map_.find(session_id);
  if (it != var_manager_map_.end()) {
    GELOGD("VarManagerPool::GetVarManager");
    return it->second;
  }

  const std::shared_ptr<VarManager> var_manager = MakeShared<VarManager>(session_id);
  if (var_manager == nullptr) {
    REPORT_INNER_ERROR("E19999", "New VarManager fail, session_id:%" PRIu64 "", session_id);
    GELOGE(INTERNAL_ERROR, "[New][VarManager] fail, session_id:%" PRIu64 "", session_id);
    return nullptr;
  }
  var_manager_map_[session_id] = var_manager;
  return var_manager;
}

void VarManagerPool::RemoveVarManager(const uint64_t session_id) {
  std::shared_ptr<VarManager> var_manager = nullptr;
  {
    const std::lock_guard<std::mutex> lock(var_manager_mutex_);
    const auto it = var_manager_map_.find(session_id);
    if (it != var_manager_map_.end()) {
      var_manager = it->second;
      (void)var_manager_map_.erase(it);
    }
  }

  if (var_manager != nullptr) {
    var_manager->Destory();
  }
}
}  // namespace ge
