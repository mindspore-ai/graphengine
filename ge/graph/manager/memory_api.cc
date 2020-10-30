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

#include "framework/memory/memory_api.h"

#include <memory>

#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/host_mem_manager.h"
#include "graph/manager/rdma_pool_allocator.h"
#include "hccl/base.h"
#include "hccl/hcom.h"

namespace ge {
Status InitRdmaPool(size_t size, rtMemType_t mem_type) {
  GELOGD("InitRdmaPool in");
  return MemManager::Instance().RdmaPoolInstance(mem_type).InitMemory(size);
}

Status RdmaRemoteRegister(const std::vector<HostVarInfo> &var_info, rtMemType_t mem_type) {
  GELOGD("Start to register rdma memory with host var size %zu", var_info.size());
  uint64_t device_base = 0;
  uint64_t device_size = 0;
  GE_CHK_STATUS_RET(MemManager::Instance().RdmaPoolInstance(mem_type).GetBaseAddr(device_base, device_size));
  return SUCCESS;
}

Status GetVarBaseAddrAndSize(const string &var_name, uint64_t &base_addr, uint64_t &var_size) {
  GELOGD("GetVarBaseAddrAndSize in");
  return HostMemManager::Instance().QueryVarMemInfo(var_name, base_addr, var_size);
}
}  // namespace ge