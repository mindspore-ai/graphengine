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

#include "graph/load/new_model_manager/task_info/hccl_task_info.h"

#include <utility>

#include "common/opskernel/ops_kernel_info_store.h"
#include "framework/common/debug/ge_log.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/model_utils.h"

namespace ge {
HcclTaskInfo::~HcclTaskInfo() {
  if (private_def_ != nullptr) {
    rtError_t ret = rtFreeHost(private_def_);
    if (ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rtFree Fail, ret = 0x%X.", ret);
    }

    private_def_ = nullptr;
  }
  input_data_addr_ = nullptr;
  davinci_model_ = nullptr;
  ops_kernel_store_ = nullptr;
  output_data_addr_ = nullptr;
  workspace_addr_ = nullptr;
}

Status HcclTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("HcclTaskInfo Init Start.");
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }
  davinci_model_ = davinci_model;

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  GetPrivateDefByTaskDef(task_def);

  auto hccl_def = task_def.kernel_hccl();
  hcclDataType_t data_type;
  int32_t count;
  uint32_t op_index = hccl_def.op_index();
  GELOGI("HcclTaskInfo Init, op_index is: %u", op_index);
  std::string hccl_type = hccl_def.hccl_type();

  // Get HCCL op
  auto op_desc = davinci_model->GetOpList()[op_index];
  GE_CHECK_NOTNULL(op_desc);

  Status dmrt = HcomOmeUtil::GetHcomDataType(op_desc, data_type);
  if (dmrt != SUCCESS) {
    GELOGE(FAILED, "davinci_model: GetHcomDataType fail! domi error: %u", dmrt);
    return FAILED;
  }

  dmrt = HcomOmeUtil::GetHcomCount(op_desc, data_type, (hccl_type == HCOMALLGATHER), count);
  if (dmrt != SUCCESS) {
    GELOGE(FAILED, "davinci_model: GetHcomCount fail! domi error: %u", dmrt);
    return FAILED;
  }

  ret = SetAddrs(hccl_type, op_desc);
  if (ret != SUCCESS) {
    GELOGE(ret, "Setaddrs Fail.");
    return ret;
  }

  count_ = count;
  hccl_type_ = hccl_type;
  data_type_ = data_type;

  // GE's new process: hccl declares the need for Workspace size, and GE allocates Workspace
  auto workspace_bytes = op_desc->GetWorkspaceBytes();
  if (!workspace_bytes.empty()) {
    uint64_t workspace_mem_size_tmp = workspace_bytes[0];
    GELOGI("hccl need work_space_mem_size=%lu", workspace_mem_size_tmp);
    if (workspace_mem_size_tmp != 0) {
      workspace_mem_size_ = workspace_mem_size_tmp;
      vector<void *> workspace_data_addrs =
          ModelUtils::GetWorkspaceDataAddrs(davinci_model->GetRuntimeParam(), op_desc);
      if (!workspace_data_addrs.empty()) {
        GELOGI("Get work_space_addr");
        workspace_addr_ = workspace_data_addrs[0];
      }
    }
  }
  // GE's new process: hccl declares the number of streams required, creates a stream by GE, and sends it to hccl
  int64_t hccl_stream_num = 0;
  if (!ge::AttrUtils::GetInt(op_desc, "used_stream_num", hccl_stream_num)) {
    GELOGW("op_desc has no attr used_stream_num!");
  }

  GELOGI("hcclStreamNum =%ld", hccl_stream_num);

  for (int64_t i = 0; i < hccl_stream_num; ++i) {
    rtStream_t stream = nullptr;
    rtError_t rt_ret =
        rtStreamCreateWithFlags(&stream, davinci_model->Priority(), RT_STREAM_PERSISTENT | RT_STREAM_FORCE_COPY);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }

    // Create slave stream, inactive by default, activated by hccl
    rt_ret = rtModelBindStream(davinci_model->GetRtModelHandle(), stream, RT_MODEL_WAIT_ACTIVE_STREAM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }

    hccl_stream_list_.push_back(stream);
    davinci_model->PushHcclStream(stream);
  }

  return SUCCESS;
}

Status HcclTaskInfo::Distribute() {
  GELOGI("HcclTaskInfo Distribute Start. begin to call function LoadTask in hccl.");
  if (ops_kernel_store_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "ops kernel store is null.");
    return INTERNAL_ERROR;
  }

  OpsKernelInfoStore *ops_kernel_info_store = reinterpret_cast<OpsKernelInfoStore *>(ops_kernel_store_);
  GE_CHECK_NOTNULL(ops_kernel_info_store);
  GETaskInfo ge_task;
  TransToGETaskInfo(ge_task);
  auto result = ops_kernel_info_store->LoadTask(ge_task);
  if (result != HCCL_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "davinci_model : load task fail, return ret: %u", result);
    return INTERNAL_ERROR;
  }

  GELOGI("Call function LoadTask end.");
  return SUCCESS;
}

Status HcclTaskInfo::SetAddrs(const std::string &hccl_type, const std::shared_ptr<OpDesc> &op_desc) {
  domi::Status dmrt;
  hcclRedOp_t op_type;
  GE_CHECK_NOTNULL(davinci_model_);
  auto input_data_addr_list = ModelUtils::GetInputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
  if (!input_data_addr_list.empty()) {
    input_data_addr_ = input_data_addr_list[0];
  }

  void *output_data_addr = nullptr;
  auto output_data_addr_list = ModelUtils::GetOutputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
  if (!output_data_addr_list.empty()) {
    output_data_addr = output_data_addr_list[0];
  }

  if (hccl_type == HCOMBROADCAST) {
    int64_t root_id;
    dmrt = HcomOmeUtil::GetHcomRootId(op_desc, root_id);
    if (dmrt != SUCCESS) {
      GELOGE(FAILED, "davinci_model: GetHcomRootId fail! domi error: %u", dmrt);
      return FAILED;
    }
    root_id_ = root_id;
  } else if (hccl_type == HCOMALLGATHER || hccl_type == HCOMRECEIVE) {
    output_data_addr_ = output_data_addr;
  } else if (hccl_type == HCOMALLREDUCE) {
    dmrt = HcomOmeUtil::GetHcomOperationType(op_desc, op_type);
    if (dmrt != SUCCESS) {
      GELOGE(FAILED, "davinci_model: GetHcomOperationType fail! domi error: %u", dmrt);
      return FAILED;
    }

    output_data_addr_ = output_data_addr;
    op_type_ = op_type;
  } else if (hccl_type == HCOMREDUCESCATTER) {
    dmrt = HcomOmeUtil::GetHcomOperationType(op_desc, op_type);
    if (dmrt != SUCCESS) {
      GELOGE(FAILED, "davinci_model: GetHcomOperationType fail! domi error: %u", dmrt);
      return FAILED;
    }

    output_data_addr_ = output_data_addr;
    op_type_ = op_type;
  }

  return SUCCESS;
}

void HcclTaskInfo::TransToGETaskInfo(GETaskInfo &ge_task) {
  ge_task.id = id_;
  ge_task.type = static_cast<uint16_t>(RT_MODEL_TASK_HCCL);
  ge_task.stream = stream_;

  ge_task.kernelHcclInfo.hccl_type = hccl_type_;
  ge_task.kernelHcclInfo.inputDataAddr = input_data_addr_;
  ge_task.kernelHcclInfo.outputDataAddr = output_data_addr_;
  ge_task.kernelHcclInfo.workSpaceAddr = workspace_addr_;
  ge_task.kernelHcclInfo.count = count_;
  ge_task.kernelHcclInfo.dataType = data_type_;
  ge_task.kernelHcclInfo.opType = op_type_;
  ge_task.kernelHcclInfo.rootId = root_id_;
  ge_task.kernelHcclInfo.workSpaceMemSize = workspace_mem_size_;
  ge_task.kernelHcclInfo.hcclStreamList = hccl_stream_list_;

  ge_task.privateDef = private_def_;
  ge_task.privateDefLen = private_def_len_;
  ge_task.opsKernelStorePtr = ops_kernel_store_;
}

void HcclTaskInfo::GetPrivateDefByTaskDef(const domi::TaskDef &task) {
  // Get privateDef and opsKernelStorePtr from taskDef and save them in taskInfo
  GELOGI("get custom info in modelTaskDef.");
  ops_kernel_store_ = nullptr;
  void *ops_kernel_store_name_temp = reinterpret_cast<void *>(task.ops_kernel_store_ptr());
  if (ops_kernel_store_name_temp != nullptr) {
    ops_kernel_store_ = std::move(ops_kernel_store_name_temp);
    std::string private_def_temp = task.private_def();
    if (!private_def_temp.empty()) {
      private_def_len_ = private_def_temp.size();
      rtError_t ret = rtMallocHost(&private_def_, private_def_len_);
      if (ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "Call rtMallocHost Fail, ret = 0x%X.", ret);
        return;
      }

      ret = rtMemcpy(private_def_, private_def_len_, task.private_def().c_str(), private_def_len_,
                     RT_MEMCPY_HOST_TO_HOST);
      if (ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "Call rtMemcpy Fail, ret = 0x%X.", ret);
        return;
      }
    }
  }
}

REGISTER_TASK_INFO(RT_MODEL_TASK_HCCL, HcclTaskInfo);
}  // namespace ge
