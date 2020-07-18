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

#include "graph/load/new_model_manager/task_info/memcpy_addr_async_task_info.h"

#include "framework/common/debug/ge_log.h"
#include "graph/load/new_model_manager/davinci_model.h"

namespace ge {
Status MemcpyAddrAsyncTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("MemcpyAddrAsyncTaskInfo Init Start");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null");
    return PARAM_INVALID;
  }

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  const auto &memcpy_async = task_def.memcpy_async();
  OpDescPtr op_desc = davinci_model->GetOpByIndex(memcpy_async.op_index());
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Task op index:%u out of range", memcpy_async.op_index());
    return INTERNAL_ERROR;
  }

  ret = ModelUtils::GetRtAddress(davinci_model->GetRuntimeParam(), memcpy_async.src(), src_);
  if (ret != SUCCESS) {
    return ret;
  }

  ret = ModelUtils::GetRtAddress(davinci_model->GetRuntimeParam(), memcpy_async.dst(), dst_);
  if (ret != SUCCESS) {
    return ret;
  }

  vector<void *> io_addrs;
  io_addrs.emplace_back(src_);
  io_addrs.emplace_back(dst_);

  // malloc args memory
  size_t args_size = sizeof(void *) * io_addrs.size();
  rtError_t rt_ret = rtMalloc(&args_, args_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  // copy orign src/dst
  GELOGI("src_args:%p, destMax:%zu, src_:%p, dst_args:%p, dst_:%p, count=%zu", args_, args_size, src_,
         static_cast<uint8_t *>(args_) + args_size, dst_, io_addrs.size());
  rt_ret = rtMemcpy(args_, args_size, io_addrs.data(), args_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api for src failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  count_ = memcpy_async.count();
  kind_ = memcpy_async.kind();
  dst_max_ = memcpy_async.dst_max();
  GELOGI("InitMemcpyAddrAsyncTaskInfo, logic[0x%lx, 0x%lx], src:%p, dst:%p, max:%lu, count:%lu, args:%p, size:%zu",
         memcpy_async.src(), memcpy_async.dst(), src_, dst_, dst_max_, count_, args_, args_size);

  davinci_model->SetZeroCopyAddr(op_desc, io_addrs, io_addrs.data(), args_, args_size, 0);
  return SUCCESS;
}

Status MemcpyAddrAsyncTaskInfo::Distribute() {
  GELOGI("MemcpyAddrAsyncTaskInfo Distribute Start, dst_max:%lu, count:%lu, kind:%u", dst_max_, count_, kind_);

  rtError_t rt_ret = rtMemcpyAsync(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(args_) + sizeof(void *)),
                                   dst_max_, args_, count_, static_cast<rtMemcpyKind_t>(kind_), stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_MEMCPY_ADDR_ASYNC, MemcpyAddrAsyncTaskInfo);
}  // namespace ge
