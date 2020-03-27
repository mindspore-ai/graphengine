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

#include "ge_runtime/task/hccl_task.h"

#include "ge_runtime/task/task_factory.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/opskernel/ge_task_info.h"

namespace ge {
namespace model_runner {
HcclTask::HcclTask(const ModelContext &model_context, const std::shared_ptr<HcclTaskInfo> &task_info)
    : TaskRepeater<HcclTaskInfo>(model_context, task_info), task_info_(task_info), stream_(nullptr),
      rt_model_handle_(nullptr), priority_(0), slave_stream_list_(), hcom_bind_model_(nullptr),
      hcom_unbind_model_(nullptr), hcom_distribute_task_(nullptr) {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
  }

  hcom_bind_model_ = task_info->hcom_bind_model();
  hcom_unbind_model_ = task_info->hcom_unbind_model();

  priority_ = model_context.priority();
  rt_model_handle_ = model_context.rt_model_handle();
  auto stream_list = model_context.stream_list();

  if (hcom_bind_model_ != nullptr) {
    if (rt_model_handle_list_.insert(rt_model_handle_).second) {
      for (auto stream : stream_list) {
        (void) hcom_bind_model_(rt_model_handle_, stream);
      }
    }
  }

  if (stream_list.size() == 1) {
    stream_ = stream_list[0];
  } else if (stream_list.size() > task_info->stream_id()) {
    stream_ = stream_list[task_info->stream_id()];
  } else {
    GELOGW("index: %u >= stream_list.size(): %zu.", task_info->stream_id(), stream_list.size());
  }
}

HcclTask::~HcclTask() {
  for (size_t i = 0; i < slave_stream_list_.size(); ++i) {
    rtError_t rt_ret = rtModelUnbindStream(rt_model_handle_, slave_stream_list_[i]);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Unbind stream from model failed! Index: %zu", i);
    }
  }

  for (size_t i = 0; i < slave_stream_list_.size(); ++i) {
    rtError_t rt_ret = rtStreamDestroy(slave_stream_list_[i]);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Destroy stream failed! Index: %zu", i);
    }
  }

  if (hcom_unbind_model_ != nullptr) {
    if (rt_model_handle_list_.find(rt_model_handle_) != rt_model_handle_list_.end()) {
      (void) hcom_unbind_model_(rt_model_handle_);
      (void)rt_model_handle_list_.erase(rt_model_handle_);
    }
  }
}

bool HcclTask::Distribute() {
  // No ops kernel info store
  hcom_distribute_task_ = task_info_->hcom_distribute_task();
  if (hcom_distribute_task_ != nullptr) {
    return hcom_distribute_task_(task_info_, stream_);
  }

  // Ops kernel info store
  // Get private_def and ops_kernel_store_ptr
  GELOGI("get custom info in modelTaskDef");
  void *ops_kernel_store = task_info_->ops_kernel_store();
  OpsKernelInfoStore* ops_kernel_info_store = reinterpret_cast<OpsKernelInfoStore*> (ops_kernel_store);
  if (ops_kernel_store == nullptr) {
    GELOGE(PARAM_INVALID, "No hcom distribute function ptr and no ops kernel store.");
    return false;
  }

  char *private_def = reinterpret_cast<char *>(const_cast<char unsigned *>(task_info_->private_def().data()));
  auto private_def_len = static_cast<uint32_t>(task_info_->private_def().size());

  GELOGI("hcclStreamNum =%ld", task_info_->hccl_stream_num());
  for (int64_t i = 0; i < task_info_->hccl_stream_num(); ++i) {
    rtStream_t stream = nullptr;
    rtError_t rt_ret = rtStreamCreateWithFlags(&stream, priority_, RT_STREAM_PERSISTENT | RT_STREAM_FORCE_COPY);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }

    rt_ret = rtModelBindStream(rt_model_handle_, stream, RT_HEAD_STREAM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }

    slave_stream_list_.push_back(stream);
  }

  GELOGI("HcclTaskInfo Distribute Start. begin to call function LoadTask in hccl.");
  GETaskInfo ge_task;
  ge_task.id = 0;
  ge_task.type = static_cast<uint16_t>(RT_MODEL_TASK_HCCL);
  ge_task.stream = stream_;

  ge_task.kernelHcclInfo.hccl_type = task_info_->hccl_type();
  ge_task.kernelHcclInfo.inputDataAddr = task_info_->input_data_addr();
  ge_task.kernelHcclInfo.outputDataAddr = task_info_->output_data_addr();
  ge_task.kernelHcclInfo.workSpaceAddr = task_info_->workspace_addr();
  ge_task.kernelHcclInfo.workSpaceMemSize = task_info_->workspace_size();
  ge_task.kernelHcclInfo.count = task_info_->count();
  ge_task.kernelHcclInfo.dataType = static_cast<int32_t>(task_info_->data_type());
  ge_task.kernelHcclInfo.opType = static_cast<int32_t>(task_info_->op_type());
  ge_task.kernelHcclInfo.rootId = task_info_->root_id();

  ge_task.kernelHcclInfo.hcclStreamList = slave_stream_list_;

  ge_task.privateDef = private_def;
  ge_task.privateDefLen = private_def_len;
  ge_task.opsKernelStorePtr = ops_kernel_store;

  auto result = ops_kernel_info_store->LoadTask(ge_task);
  // tagHcclResult::HCCL_SUCCESS is 0
  if (result != 0) {
    GELOGE(INTERNAL_ERROR, "davinci_model : load task fail, return ret: %u", result);
    return false;
  }

  GELOGI("call function LoadTask end.");
  return true;
}

REGISTER_TASK(TaskInfoType::kHccl, HcclTask, HcclTaskInfo);
}  // namespace model_runner
}  // namespace ge
