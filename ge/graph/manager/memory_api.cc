/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "common/ge/plugin_manager.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/host_mem_manager.h"
#include "graph/manager/rdma_pool_allocator.h"
#include "graph/utils/type_utils.h"
#include "hccl/base.h"
#include "hccl/hccl_types.h"

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

Status MallocSharedMemory(const TensorInfo &tensor_info, uint64_t &dev_addr, uint64_t &memory_size) {
  GELOGD("MallocSharedMemory in");
  uint32_t type_size = 0;
  bool result = TypeUtils::GetDataTypeLength(tensor_info.data_type, type_size);
  if (!result) {
    GELOGE(GRAPH_FAILED, "GetDataTypeLength failed, data_type=(%s).",
           TypeUtils::DataTypeToSerialString(tensor_info.data_type).c_str());
    return GRAPH_FAILED;
  }
  memory_size = type_size;
  for (auto dim : tensor_info.dims) {
    if (dim <= 0) {
      GELOGE(GRAPH_FAILED, "Tensor dims should be positive");
      return GRAPH_FAILED;
    }
    memory_size *= dim;
  }
  SharedMemInfo mem_info(tensor_info.var_name, memory_size);
  Status ret = HostMemManager::Instance().MallocSharedMemory(mem_info);
  if (ret != SUCCESS) {
    GELOGE(GRAPH_FAILED, "MallocSharedMemory failed op name [%s]", tensor_info.var_name.c_str());
    return GRAPH_FAILED;
  }
  dev_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(mem_info.device_address));
  GELOGD("MallocSharedMemory Succeeded");
  return SUCCESS;
}

Status GetVarBaseAddrAndSize(const string &var_name, uint64_t &base_addr, uint64_t &var_size) {
  GELOGD("GetVarBaseAddrAndSize in");
  return HostMemManager::Instance().QueryVarMemInfo(var_name, base_addr, var_size);
}
}  // namespace ge
