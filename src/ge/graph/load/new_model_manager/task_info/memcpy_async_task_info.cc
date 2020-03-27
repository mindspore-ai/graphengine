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

#include "graph/load/new_model_manager/task_info/memcpy_async_task_info.h"

#include "framework/common/debug/ge_log.h"
#include "graph/load/new_model_manager/davinci_model.h"

namespace ge {
Status MemcpyAsyncTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("MemcpyAsyncTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  auto memcpy_async_def = task_def.memcpy_async();

  GELOGI("InitMemcpyAsyncTaskInfo start.");

  uint64_t logic_dst = memcpy_async_def.dst();
  uint64_t logic_src = memcpy_async_def.src();

  dst_max_ = memcpy_async_def.dst_max();

  uint64_t update_base_addr = 0;
  ret = GetUpdateBaseAddr(davinci_model, logic_src, update_base_addr);
  if (ret != SUCCESS) {
    return ret;
  }
  src_ = reinterpret_cast<uint8_t *>(update_base_addr + logic_src);

  uint64_t mem_base = reinterpret_cast<uint64_t>(davinci_model->MemBase());
  uint64_t logic_mem_base = davinci_model->GetRtBaseAddr();
  dst_ = reinterpret_cast<uint8_t *>(mem_base + (logic_dst - logic_mem_base));

  count_ = memcpy_async_def.count();
  kind_ = memcpy_async_def.kind();

  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::Distribute() {
  GELOGI("MemcpyAsyncTaskInfo Distribute Start.");
  GELOGI("Distribute MemcpyAsync, dst_max:%lu, count:%lu, kind:%u.", dst_max_, count_, kind_);

  rtError_t rt_ret = rtMemcpyAsync(dst_, dst_max_, src_, count_, static_cast<rtMemcpyKind_t>(kind_), stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::GetUpdateBaseAddr(DavinciModel *davinci_model, uint64_t update_addr, uint64_t &base_addr) {
  GE_CHECK_NOTNULL(davinci_model);
  uint64_t data_base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(davinci_model->MemBase())) -
                            davinci_model->GetRtBaseAddr();
  uint64_t weight_base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(davinci_model->WeightsMemBase())) -
                              davinci_model->GetRtWeightAddr();
  uint64_t var_base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(davinci_model->VarMemBase())) -
                           davinci_model->GetRtVarAddr();

  uint64_t data_base_addr_start = davinci_model->GetRtBaseAddr();
  uint64_t data_base_addr_end = davinci_model->GetRtBaseAddr() + davinci_model->TotalMemSize();
  uint64_t wight_base_addr_start = davinci_model->GetRtWeightAddr();
  uint64_t wight_base_addr_end = davinci_model->GetRtWeightAddr() + davinci_model->TotalWeightsMemSize();
  uint64_t varible_base_addr_start = davinci_model->GetRtVarAddr();
  uint64_t varible_base_addr_end = davinci_model->GetRtVarAddr() + davinci_model->TotalVarMemSize();

  if ((data_base_addr_start <= update_addr) && (update_addr <= data_base_addr_end)) {
    base_addr = data_base_addr;
    GELOGI("The update_addr is data address.");
  } else if ((wight_base_addr_start <= update_addr) && (update_addr <= wight_base_addr_end)) {
    base_addr = weight_base_addr;
    GELOGI("The update_addr is weight address.");
  } else if ((varible_base_addr_start <= update_addr) && (update_addr <= varible_base_addr_end)) {
    base_addr = var_base_addr;
    GELOGI("The update_addr is variable address.");
  } else if (update_addr != 0) {
    base_addr = 0;
    GELOGE(PARAM_INVALID, "The update_addr is abnormal.");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_MEMCPY_ASYNC, MemcpyAsyncTaskInfo);
}  // namespace ge
