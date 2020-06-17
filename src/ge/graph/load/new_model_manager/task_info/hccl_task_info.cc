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
namespace {
const uint32_t kMaxTaskOfStream = 200;
}

uint32_t HcclTaskInfo::max_node_of_hccl_stream_ = 0;
std::mutex HcclTaskInfo::hccl_follow_stream_mutex_;

HcclTaskInfo::~HcclTaskInfo() {
  if (private_def_ != nullptr) {
    rtError_t ret = rtFreeHost(private_def_);
    if (ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rtFree Fail, ret = 0x%X.", ret);
    }
    private_def_ = nullptr;
  }
  davinci_model_ = nullptr;
  ops_kernel_store_ = nullptr;
  max_node_of_hccl_stream_ = 0;
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
  uint32_t op_index = hccl_def.op_index();
  GELOGI("HcclTaskInfo Init, op_index is: %u", op_index);

  // Get HCCL op
  OpDescPtr op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);

  // Create the kernel hccl infos
  CreateKernelHcclInfo(op_desc);

  // Initialize the hccl_type of all kernel hccl info
  HcomOmeUtil::GetHcclType(task_def, kernel_hccl_infos_);

  // Only in Horovod scenario should get the inputName and GeShape
  ret = HcomOmeUtil::GetHorovodInputs(op_desc, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "davinci_model: GetHorovodInputs fail! domi error: %u", ret);
    return FAILED;
  }
  Status dmrt = HcomOmeUtil::GetHcclDataType(op_desc, kernel_hccl_infos_);
  if (dmrt != SUCCESS) {
    GELOGE(FAILED, "davinci_model: GetHcomDataType fail! domi error: %u", dmrt);
    return FAILED;
  }
  dmrt = HcomOmeUtil::GetHcclCount(op_desc, kernel_hccl_infos_);
  if (dmrt != SUCCESS) {
    GELOGE(FAILED, "davinci_model: GetHcomCount fail! domi error: %u", dmrt);
    return FAILED;
  }
  // Only HCOMBROADCAST and HVDCALLBACKBROADCAST need to get the rootId
  dmrt = HcomOmeUtil::GetAllRootId(op_desc, kernel_hccl_infos_);
  if (dmrt != SUCCESS) {
    GELOGE(FAILED, "davinci_model: Get rootId fail! domi error: %u", dmrt);
    return FAILED;
  }
  ret = SetAddrs(op_desc, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Setaddrs Fail.");
    return ret;
  }
  // GE's new process: hccl declares the need for Workspace size, and GE allocates Workspace
  ret = SetWorkspace(op_desc, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    GELOGE(ret, "SetWorkspace Fail.");
    return ret;
  }
  // GE's new process: hccl declares the number of streams required, creates a stream by GE, and sends it to hccl
  ret = SetFollowStream(op_desc, davinci_model);
  if (ret != SUCCESS) {
    GELOGE(ret, "SetStream Fail.");
    return ret;
  }

  GELOGI("HcclTaskInfo Init Success");
  return SUCCESS;
}

Status HcclTaskInfo::SetFollowStream(const ge::ConstOpDescPtr &op_desc, DavinciModel *davinci_model) {
  if (!HcomOmeUtil::IsHCOMOp(op_desc->GetType())) {
    GELOGI("Node %s Optye %s no need to create slave streams.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return SUCCESS;
  }
  Status ret;
  int64_t hccl_stream_num = 0;
  if (!ge::AttrUtils::GetInt(op_desc, "used_stream_num", hccl_stream_num)) {
    GELOGI("op_desc has no attr used_stream_num!");
  }

  std::lock_guard<std::mutex> lock(hccl_follow_stream_mutex_);
  if (max_node_of_hccl_stream_ == 0) {
    uint32_t max_stream_count;
    uint32_t max_task_count;
    ret = rtGetMaxStreamAndTask(RT_NORMAL_STREAM, &max_stream_count, &max_task_count);
    if (ret != RT_ERROR_NONE) {
      GELOGE(FAILED, "Get max stream and task count by rts failed.");
      return FAILED;
    }
    max_node_of_hccl_stream_ = max_task_count / kMaxTaskOfStream;
  }

  if (static_cast<size_t>(hccl_stream_num) <= davinci_model->GetHcclFolowStream().size()) {
    GELOGI("capacity of follow stream is enough to be reused.");
    ReuseStream(hccl_stream_num, davinci_model);
  } else {
    GELOGI("need to reuse follow stream and create new follow stream.");
    size_t created_stream_num = davinci_model->GetHcclFolowStream().size();
    ReuseStream(created_stream_num, davinci_model);
    ret = CreateStream(hccl_stream_num - created_stream_num, davinci_model);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Create hccl stream failed.");
      return FAILED;
    }
  }
  GELOGI("Initialize hccl slave stream success, hcclStreamNum =%ld", hccl_stream_num);
  return SUCCESS;
}

void HcclTaskInfo::ReuseStream(int64_t stream_num, DavinciModel *davinci_model) {
  GELOGI("Start to reuse %ld follow stream.", stream_num);
  int64_t index = 0;
  for (int64_t i = 0; i < stream_num; i++) {
    hccl_stream_list_.emplace_back(davinci_model->GetHcclFolowStream().at(index).first);
    int64_t remain_cap = davinci_model->GetHcclFolowStream().at(index).second - 1;
    davinci_model->ReuseHcclFollowStream(remain_cap, index);
  }
}

Status HcclTaskInfo::CreateStream(int64_t stream_num, DavinciModel *davinci_model) {
  GELOGI("Start to create %ld hccl stream.", stream_num);
  for (int64_t i = 0; i < stream_num; ++i) {
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
    GELOGD("hccl_stream addr is=%p", stream);
    int64_t remain_cap = max_node_of_hccl_stream_ - 1;
    davinci_model->CreateHcclFollowStream(stream, remain_cap);

    hccl_stream_list_.emplace_back(stream);
    davinci_model->PushHcclStream(stream);
  }
  GELOGI("CreateStream success.");
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
  GELOGI("HcclTaskInfo Distribute Success.");
  return SUCCESS;
}
Status HcclTaskInfo::SetAddrs(const std::shared_ptr<OpDesc> &op_desc,
                              std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (HcomOmeUtil::CheckKernelHcclInfo(op_desc, kernel_hccl_infos) != SUCCESS) {
    GELOGE(PARAM_INVALID, "HcomOmeUtil:: the number of GETaskKernelHcclInfo is invalid.");
    return PARAM_INVALID;
  }
  GELOGI("Set hccl task input output address, node[%s}, type[%s] kernel_hccl_infos.size[%zu].",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), kernel_hccl_infos.size());
  if (op_desc->GetType() == HVDWAIT) {
    return SUCCESS;
  }
  domi::Status dmrt;
  hcclRedOp_t op_type = HCCL_REP_OP_SUM;
  GE_CHECK_NOTNULL(davinci_model_);
  GELOGI("Calc opType[%s] input address before. Node name[%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  auto input_data_addr_list = ModelUtils::GetInputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);

  auto output_data_addr_list = ModelUtils::GetOutputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
  // initialize every kernel_hccl_info inputDataAddr
  for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
    std::string hccl_type = kernel_hccl_infos[i].hccl_type;
    void *input_data_addr = input_data_addr_list.empty() ? nullptr : input_data_addr_list[i];
    kernel_hccl_infos[i].inputDataAddr = input_data_addr;

    void *output_data_addr = output_data_addr_list.empty() ? nullptr : output_data_addr_list[i];
    if (hccl_type == HCOMALLGATHER || hccl_type == HCOMRECEIVE || hccl_type == HVDCALLBACKALLGATHER) {
      kernel_hccl_infos[i].outputDataAddr = output_data_addr;
    } else if (hccl_type == HCOMALLREDUCE || hccl_type == HCOMREDUCESCATTER || hccl_type == HVDCALLBACKALLREDUCE) {
      dmrt = HcomOmeUtil::GetHcclOperationType(op_desc, op_type);
      if (dmrt != SUCCESS) {
        GELOGE(FAILED, "davinci_model: GetHcomOperationType fail! domi error: %u", dmrt);
        return FAILED;
      }
      kernel_hccl_infos[i].outputDataAddr = output_data_addr;
      kernel_hccl_infos[i].opType = op_type;
    }
    davinci_model_->DisableZeroCopy(input_data_addr);
  }
  return SUCCESS;
}
void HcclTaskInfo::TransToGETaskInfo(GETaskInfo &ge_task) {
  ge_task.id = id_;
  ge_task.type = static_cast<uint16_t>(RT_MODEL_TASK_HCCL);
  ge_task.stream = stream_;
  ge_task.kernelHcclInfo = kernel_hccl_infos_;
  ge_task.privateDef = private_def_;
  ge_task.privateDefLen = private_def_len_;
  ge_task.opsKernelStorePtr = ops_kernel_store_;
  for (size_t i = 0; i < ge_task.kernelHcclInfo.size(); i++) {
    ge_task.kernelHcclInfo[i].hcclStreamList = hccl_stream_list_;
  }
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

      ret =
        rtMemcpy(private_def_, private_def_len_, task.private_def().c_str(), private_def_len_, RT_MEMCPY_HOST_TO_HOST);
      if (ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "Call rtMemcpy Fail, ret = 0x%X.", ret);
        return;
      }
      GELOGI("The first address of the custom info, privateDef=%p.", private_def_);
    }
  }
}
void HcclTaskInfo::CreateKernelHcclInfo(const ge::ConstOpDescPtr &op_desc) {
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
  if (HcomOmeUtil::IsHCOMOp(op_desc->GetType())) {
    GETaskKernelHcclInfo kernel_hccl_info;
    kernel_hccl_infos_.emplace_back(kernel_hccl_info);
  } else if (HcomOmeUtil::IsHorovodOp(op_desc->GetType())) {
    // Horovod wait do not have any input, but create a GETaskKernelHcclInfo to record hccl_type.
    // Other Operator need to check that the number of GETaskKernelHcclInfo must equals to number of inputs
    if (op_desc->GetType() == HVDWAIT) {
      GETaskKernelHcclInfo kernel_hccl_info;
      kernel_hccl_infos_.emplace_back(kernel_hccl_info);
      return;
    }
    for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
      GETaskKernelHcclInfo kernel_hccl_info;
      kernel_hccl_infos_.emplace_back(kernel_hccl_info);
    }
  }
}
Status HcclTaskInfo::SetWorkspace(const std::shared_ptr<OpDesc> &op_desc,
                                  std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("SetWorkspace Node[%s] opType[%s] set workspace.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  uint64_t workspace_mem_size = 0;
  void *workspace_addr = nullptr;
  auto workspace_bytes = op_desc->GetWorkspaceBytes();
  if (!workspace_bytes.empty()) {
    uint64_t workspace_mem_size_tmp = workspace_bytes[0];
    GELOGI("hccl need workSpaceMemSize=%lu", workspace_mem_size_tmp);
    if (workspace_mem_size_tmp != 0) {
      workspace_mem_size = workspace_mem_size_tmp;
      vector<void *> workspace_data_addrs =
        ModelUtils::GetWorkspaceDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
      if (!workspace_data_addrs.empty()) {
        GELOGI("Get workSpaceAddr");
        workspace_addr = workspace_data_addrs[0];
      }
    }
  }
  for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
    kernel_hccl_infos[i].workSpaceMemSize = workspace_mem_size;
    kernel_hccl_infos[i].workSpaceAddr = workspace_addr;
  }
  return SUCCESS;
}
REGISTER_TASK_INFO(RT_MODEL_TASK_HCCL, HcclTaskInfo);
}  // namespace ge
