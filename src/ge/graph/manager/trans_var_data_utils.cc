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

#include "graph/manager/trans_var_data_utils.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/formats/formats.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"

namespace ge {
Status TransVarDataUtils::SyncVarData2BroadCast(const string &var_name, const ge::GeTensorDesc &src_tensor_desc,
                                                uint8_t *dst_addr, int64_t dst_addr_size, uint64_t session_id) {
  GE_CHK_BOOL_RET_STATUS(dst_addr != nullptr, FAILED, "dst addr is null. ");
  uint8_t *src_host_addr = nullptr;
  int64_t src_addr_size = 0;
  GE_MAKE_GUARD_RTMEM(src_host_addr);
  GE_CHK_STATUS_RET(SyncTensorToHost(var_name, src_tensor_desc, &src_host_addr, src_addr_size, session_id));

  GELOGI("src_addr_size: %u, dst_addr_size: %u", src_addr_size, dst_addr_size);
  GE_CHK_BOOL_RET_STATUS(src_addr_size == dst_addr_size, FAILED, "var data size is not equal broadcast ");

  GE_CHK_RT_RET(rtMemcpy(dst_addr, dst_addr_size, src_host_addr, src_addr_size, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status TransVarDataUtils::SyncBroadCastData2Var(uint8_t *src_addr, int64_t src_addr_size, const string &var_name,
                                                const ge::GeTensorDesc &dst_tensor_desc, uint64_t session_id) {
  GE_CHK_BOOL_RET_STATUS(src_addr != nullptr, FAILED, "src addr is null. ");
  uint8_t *host_addr = nullptr;
  GE_MAKE_GUARD_RTMEM(host_addr);
  GE_CHK_RT_RET(rtMallocHost(reinterpret_cast<void **>(&host_addr), src_addr_size));
  GE_CHK_RT_RET(rtMemcpy(host_addr, src_addr_size, src_addr, src_addr_size, RT_MEMCPY_DEVICE_TO_HOST));

  GE_CHK_STATUS_RET(
    SyncTensorToDevice(var_name, reinterpret_cast<uint8_t *>(host_addr), src_addr_size, dst_tensor_desc, session_id));

  return SUCCESS;
}

Status TransVarDataUtils::SyncTensorToHost(const string &var_name, const ge::GeTensorDesc &src_tensor_desc,
                                           uint8_t **host_addr, int64_t &src_tensor_size, uint64_t session_id) {
  GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(src_tensor_desc, src_tensor_size), "get size from TensorDesc failed");

  uint8_t *src_addr = nullptr;
  GE_CHK_STATUS_RET(VarManager::Instance(session_id)->GetVarAddr(var_name, src_tensor_desc, &src_addr));
  uint8_t *mem_addr = src_addr -
                      static_cast<int64_t>(reinterpret_cast<uintptr_t>(VarManager::Instance(0)->GetVarMemLogicBase())) +
                      static_cast<int64_t>(
                        reinterpret_cast<uintptr_t>(VarManager::Instance(session_id)->GetVarMemoryBase(RT_MEMORY_HBM)));
  GE_CHK_RT_RET(rtMallocHost(reinterpret_cast<void **>(host_addr), src_tensor_size));

  GE_CHK_RT_RET(rtMemcpy(*host_addr, src_tensor_size, mem_addr, src_tensor_size, RT_MEMCPY_DEVICE_TO_HOST));

  GELOGI("SyncTensorToHost var_name %s, src_tensor_size %ld", var_name.c_str(), src_tensor_size);
  return SUCCESS;
}

Status TransVarDataUtils::SyncTensorToDevice(const string &var_name, const uint8_t *host_addr, uint32_t addr_size,
                                             const ge::GeTensorDesc &dst_tensor_desc, uint64_t session_id) {
  uint8_t *dst_addr = nullptr;
  GE_CHK_STATUS_RET(VarManager::Instance(session_id)->GetVarAddr(var_name, dst_tensor_desc, &dst_addr));
  uint8_t *mem_addr = dst_addr -
                      static_cast<int64_t>(reinterpret_cast<uintptr_t>(VarManager::Instance(0)->GetVarMemLogicBase())) +
                      static_cast<int64_t>(
                        reinterpret_cast<uintptr_t>(VarManager::Instance(session_id)->GetVarMemoryBase(RT_MEMORY_HBM)));
  GE_CHK_RT_RET(rtMemcpy(mem_addr, addr_size, host_addr, addr_size, RT_MEMCPY_HOST_TO_DEVICE));

  GELOGI("SyncTensorToDevice var_name %s, addr_size %u", var_name.c_str(), addr_size);

  return SUCCESS;
}
}  // namespace ge
