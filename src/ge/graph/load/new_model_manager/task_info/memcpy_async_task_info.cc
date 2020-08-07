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
  GELOGI("MemcpyAsyncTaskInfo Init Start");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null");
    return PARAM_INVALID;
  }

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  memcpy_async = task_def.memcpy_async();
  count_ = memcpy_async.count();
  kind_ = memcpy_async.kind();
  dst_max_ = memcpy_async.dst_max();
  if (davinci_model->IsKnownNode()) {
    src_ = reinterpret_cast<uint8_t *>(davinci_model_->GetCurrentArgsAddr(args_offset_));
    dst_ = reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(src_) + sizeof(void *));
    // for zero copy
    kind_ = RT_MEMCPY_ADDR_DEVICE_TO_DEVICE;
    GELOGI("MemcpyAsyncTaskInfo src_ %p, dst_ %p, args_offset %u.", src_, dst_, args_offset_);
    return SUCCESS;
  }
  ret = ModelUtils::GetRtAddress(davinci_model->GetRuntimeParam(), memcpy_async.src(), src_);
  if (ret != SUCCESS) {
    return ret;
  }

  ret = ModelUtils::GetRtAddress(davinci_model->GetRuntimeParam(), memcpy_async.dst(), dst_);
  if (ret != SUCCESS) {
    return ret;
  }

  GELOGI("MemcpyAsyncTaskInfo Init Success, logic[0x%lx, 0x%lx], src:%p, dst:%p, max:%lu, count:%lu",
         memcpy_async.src(), memcpy_async.dst(), src_, dst_, dst_max_, count_);

  davinci_model->DisableZeroCopy(src_);
  davinci_model->DisableZeroCopy(dst_);
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::Distribute() {
  GELOGI("MemcpyAsyncTaskInfo Distribute Start. dst_max:%lu, count:%lu, kind:%u", dst_max_, count_, kind_);

  rtError_t rt_ret = rtMemcpyAsync(dst_, dst_max_, src_, count_, static_cast<rtMemcpyKind_t>(kind_), stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("MemcpyAsyncTaskInfo Distribute Success");
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  // the num of src and dst size is 2
  uint32_t args_size = sizeof(void *) * 2;
  args_offset_ = davinci_model->GetTotalArgsSize();
  davinci_model->SetTotalArgsSize(args_size);
  davinci_model_ = davinci_model;
  GELOGI("MemcpyAsyncTaskInfo kernel args_size %u, args_offset %u", args_size, args_offset_);
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::UpdateArgs() {
  GELOGI("MemcpyAsyncTaskInfo::UpdateArgs in.");
  GE_CHECK_NOTNULL(davinci_model_);
  Status ret = ModelUtils::GetRtAddress(davinci_model_->GetRuntimeParam(), memcpy_async.src(), src_);
  if (ret != SUCCESS) {
    return ret;
  }

  ret = ModelUtils::GetRtAddress(davinci_model_->GetRuntimeParam(), memcpy_async.dst(), dst_);
  if (ret != SUCCESS) {
    return ret;
  }

  vector<void *> io_addrs;
  io_addrs.emplace_back(reinterpret_cast<void *>(src_));
  io_addrs.emplace_back(reinterpret_cast<void *>(dst_));

  davinci_model_->SetTotalIOAddrs(io_addrs);

  GELOGI("MemcpyAsyncTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_MEMCPY_ASYNC, MemcpyAsyncTaskInfo);
}  // namespace ge
