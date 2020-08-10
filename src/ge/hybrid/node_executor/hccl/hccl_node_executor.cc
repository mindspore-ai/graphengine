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

#include "hybrid/node_executor/hccl/hccl_node_executor.h"
#include "graph/manager/util/hcom_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "common/ge/ge_util.h"
#include "common/ge/plugin_manager.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "hccl/hcom.h"

namespace ge {
namespace hybrid {

REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HCCL, HcclNodeExecutor);

Status HcclNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGI("[%s] HcclNodeTask::ExecuteAsync in.", context.GetNodeName());
  if (context.handle_ == nullptr) {
    GELOGE(FAILED, "hccl handle is nullptr! ");
    return FAILED;
  }
  auto EnqueueHcomOpertion = (hcclResult_t(*)(HcomOpertion, std::function<void(hcclResult_t status)>))dlsym(
    context.handle_, "EnqueueHcomOpertion");
  if (EnqueueHcomOpertion == nullptr) {
    GELOGE(FAILED, "Failed to invoke EnqueueHcomOpertion hcom unknown node function.");
    if (dlclose(context.handle_) != 0) {
      GELOGW("Failed to close handle %s", dlerror());
    }
    return FAILED;
  }

  vector<void *> inputs;
  for (int i = 0; i < context.NumInputs(); ++i) {
    TensorValue *tv = context.MutableInput(i);
    GE_CHECK_NOTNULL(tv);
    inputs.emplace_back(tv->MutableData());
  }

  vector<void *> outputs;
  for (int i = 0; i < context.NumOutputs(); ++i) {
    TensorValue *tv = context.MutableOutput(i);
    GE_CHECK_NOTNULL(tv);
    outputs.emplace_back(tv->MutableData());
  }

  const NodeItem &node_item = context.GetNodeItem();
  const OpDescPtr op_desc = MakeShared<OpDesc>(*(node_item.op_desc));
  GE_CHECK_NOTNULL(op_desc);

  HcomOpertion op_info;
  op_info.hcclType = op_desc->GetType();
  op_info.inputPtr = inputs.empty() ? nullptr : inputs[0];
  op_info.outputPtr = outputs.empty() ? nullptr : outputs[0];
  ge::DataType src_data_type = op_desc->GetInputDescPtr(0)->GetDataType();
  auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
  if (iter == kConstOpHcclDataType.end()) {
    GELOGE(PARAM_INVALID, "kConstOpHcclDataType find failed.");
    return PARAM_INVALID;
  }
  op_info.dataType = iter->second;
  hcclRedOp_t op_type = HCCL_REP_OP_SUM;
  if (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HCOMREDUCESCATTER ||
      op_desc->GetType() == HVDCALLBACKALLREDUCE) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclOperationType(op_desc, op_type), "GetHcclOperationType failed");
    op_info.opType = op_type;
  }
  int64_t root_id = 0;
  if (op_desc->GetType() == HCOMBROADCAST) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclRootId(op_desc, root_id), "GetHcclRootId failed");
  }
  op_info.root = root_id;
  auto callback = [this](hcclResult_t status) {
    if (status != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "Call HcomExcutorInitialize failed, ret: 0x%X", status);
    }
    std::lock_guard<std::mutex> lock(this->hccl_mutex_);
    this->cond_.notify_all();
    GELOGI("hccl callback success.");
  };
  int32_t count = 0;
  GE_CHK_STATUS_RET(HcomOmeUtil::GetHcomCount(op_desc, static_cast<hcclDataType_t>(op_info.dataType), false, count),
                    "GetHcomCount failed");
  GELOGI("[%s] HcclNodeTask::ExecuteAsync hccl_type %s, count %d, data_type %d, op_type %d, root %d.",
         context.GetNodeName(), op_info.hcclType.c_str(), count, op_info.dataType, op_info.opType, op_info.root);
  op_info.count = count;

  hcclResult_t hccl_ret = EnqueueHcomOpertion(op_info, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    GELOGE(HCCL_E_INTERNAL, "Call HcomExcutorInitialize failed, ret: 0x%X", hccl_ret);
    return HCCL_E_INTERNAL;
  }

  // pending until hccl finished
  std::unique_lock<std::mutex> ulock(hccl_mutex_);
  cond_.wait(ulock);

  context.RegisterCallback(done_callback);
  GELOGI("[%s] HcclNodeTask::ExecuteAsync success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeTask::UpdateArgs(TaskContext &context) { return SUCCESS; }

Status HcclNodeTask::Init(TaskContext &context) {
  GELOGI("[%s] HcclNodeExecutor::Init success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GELOGI("[%s] HcclNodeExecutor::PrepareTask in.", context.GetNodeName());

  GE_CHK_STATUS_RET(task.Init(context), "hccl node load hccl so failed.");
  // allocate output mem
  GE_CHK_STATUS_RET(context.AllocateOutputs(), "hccl node task allocate output failed.");

  GE_CHK_STATUS_RET(task.UpdateArgs(context), "hccl node task update args failed.");
  GELOGI("[%s] HcclNodeExecutor::PrepareTask success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GELOGI("[%s] HcclNodeExecutor::LoadTask in.", node->GetName().c_str());
  GE_CHECK_NOTNULL(node);

  task = MakeShared<HcclNodeTask>();
  GE_CHECK_NOTNULL(task);
  GELOGI("[%s] HcclNodeExecutor::LoadTask success.", node->GetName().c_str());
  return SUCCESS;
}

Status HcclNodeExecutor::ExecuteTask(NodeTask &task, TaskContext &context,
                                     const std::function<void()> &callback) const {
  context.handle_ = handle_;
  GE_CHK_STATUS_RET(task.ExecuteAsync(context, callback), "Failed to execute task. node = %s",
                    context.GetNodeItem().NodeName().c_str());
  return SUCCESS;
}

Status HcclNodeExecutor::Initialize() {
  std::string file_name = "libhcom_graph_adaptor.so";
  std::string path = PluginManager::GetPath();
  path.append(file_name);
  string canonical_path = RealPath(path.c_str());
  if (canonical_path.empty()) {
    GELOGW("failed to get realpath of %s", path.c_str());
    return FAILED;
  }

  GELOGI("FileName:%s, Path:%s.", file_name.c_str(), canonical_path.c_str());
  handle_ = dlopen(canonical_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle_ == nullptr) {
    GELOGE(GE_PLGMGR_SO_NOT_EXIST, "Failed in dlopen %s! ", dlerror());
    return FAILED;
  }
  auto HcomExcutorInitialize = (hcclResult_t(*)())dlsym(handle_, "HcomExcutorInitialize");
  if (HcomExcutorInitialize == nullptr) {
    GELOGE(FAILED, "Failed to invoke HcomExcutorInitialize hcom unknown node function.");
    return FAILED;
  }
  hcclResult_t hccl_ret = HcomExcutorInitialize();
  if (hccl_ret == HCCL_E_PTR) {
    GELOGI("Hccl comm is null, hcom executor initialize is not required.");
  } else if (hccl_ret == HCCL_SUCCESS) {
    GELOGI("Hcom executor initialize success.");
  } else {
    GELOGE(FAILED, "Call HcomExcutorInitialize failed, ret: 0x%X", hccl_ret);
    return FAILED;
  }
  return SUCCESS;
}

Status HcclNodeExecutor::Finalize() {
  auto HcomExcutorFinalize = (hcclResult_t(*)())dlsym(handle_, "HcomExcutorFinalize");
  if (HcomExcutorFinalize == nullptr) {
    GELOGE(FAILED, "Failed to invoke HcomExcutorFinalize hcom unknown node function.");
    return FAILED;
  }
  hcclResult_t hccl_ret = HcomExcutorFinalize();
  if (hccl_ret != HCCL_SUCCESS) {
    GELOGE(FAILED, "Call HcomExcutorFinalize failed, ret: 0x%X", hccl_ret);
    return FAILED;
  }
  // dlclose file handle
  if (dlclose(handle_) != 0) {
    GELOGW("Failed to close handle %s", dlerror());
  }
  GELOGI("Hcom executor finalize success.");
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
