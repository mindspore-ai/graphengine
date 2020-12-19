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

#include "hybrid/node_executor/hccl/hccl_node_executor.h"
#include "common/ge/ge_util.h"
#include "common/ge/plugin_manager.h"
#include "common/math/math_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/util/hcom_util.h"
#include "graph/runtime_inference_context.h"
#include "hccl/hcom.h"

namespace {
const size_t kVarTableDims = 2;
const size_t kVarTableRowCnt = 3;
const size_t kVarTableIdxAddr = 1;
const size_t kVarTableIdxLen = 2;
}  // namespace
namespace ge {
namespace hybrid {

REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HCCL, HcclNodeExecutor);

Status HcclNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGI("[%s] HcclNodeTask::ExecuteAsync in.", context.GetNodeName());
  if (context.handle_ == nullptr) {
    GELOGE(FAILED, "hccl handle is nullptr! ");
    return FAILED;
  }
  auto HcomExecEnqueueOperation = (HcclResult(*)(HcomOperation, std::function<void(HcclResult status)>))dlsym(
      context.handle_, "HcomExecEnqueueOperation");
  if (HcomExecEnqueueOperation == nullptr) {
    GELOGE(FAILED, "Failed to invoke HcomExecEnqueueOperation hcom unknown node function.");
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
  const OpDescPtr op_desc = node_item.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  HcomOperation op_info;
  op_info.hcclType = op_desc->GetType();
  op_info.inputPtr = inputs.empty() ? nullptr : inputs[0];
  op_info.outputPtr = outputs.empty() ? nullptr : outputs[0];
  auto input_desc = node_item.MutableInputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  ge::DataType src_data_type = input_desc->GetDataType();
  auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
  if (iter == kConstOpHcclDataType.end()) {
    GELOGE(PARAM_INVALID, "kConstOpHcclDataType find failed.");
    return PARAM_INVALID;
  }
  op_info.dataType = iter->second;
  HcclReduceOp op_type = HCCL_REDUCE_SUM;
  if (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HCOMREDUCESCATTER ||
      op_desc->GetType() == HVDCALLBACKALLREDUCE || op_desc->GetType() == HCOMREDUCE) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclOperationType(op_desc, op_type), "GetHcclOperationType failed");
    op_info.opType = op_type;
  }
  int64_t root_id = 0;
  if (op_desc->GetType() == HCOMBROADCAST) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclRootId(op_desc, root_id), "GetHcclRootId failed");
  }
  op_info.root = root_id;
  auto callback = [this, op_desc](HcclResult status) {
    if (status != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "node %s call HcomExecEnqueueOperation failed, ret: 0x%X", op_desc->GetName().c_str(), status);
    }
    std::lock_guard<std::mutex> lock(this->hccl_mutex_);
    this->cond_.notify_all();
    GELOGI("node %s hccl callback success.", op_desc->GetName().c_str());
  };
  int32_t count = 0;
  GE_CHK_STATUS_RET(HcomOmeUtil::GetHcomCount(op_desc, static_cast<HcclDataType>(op_info.dataType),
                                              op_desc->GetType() == HCOMALLGATHER, count),
                    "GetHcomCount failed");
  GELOGI("[%s] HcclNodeTask::ExecuteAsync hccl_type %s, count %d, data_type %d, op_type %d, root %d.",
         context.GetNodeName(), op_info.hcclType.c_str(), count, op_info.dataType, op_info.opType, op_info.root);
  op_info.count = count;

  HcclResult hccl_ret = HcomExecEnqueueOperation(op_info, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    GELOGE(HCCL_E_INTERNAL, "Call HcomExecInitialize failed, ret: 0x%X", hccl_ret);
    return HCCL_E_INTERNAL;
  }

  // pending until hccl finished
  std::unique_lock<std::mutex> ulock(hccl_mutex_);
  cond_.wait(ulock);

  GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
  GELOGI("[%s] HcclNodeTask::ExecuteAsync success.", context.GetNodeName());
  return SUCCESS;
}

Status RdmaNodeTask::UpdateArgs(TaskContext &context) { return SUCCESS; }

Status RdmaNodeTask::Init(TaskContext &context) {
  GELOGI("[%s] RdmaNodeTask::Init in.", context.GetNodeName());
  const NodeItem &node_item = context.GetNodeItem();
  auto op_desc = node_item.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto remote_idx = op_desc->GetInputIndexByName("remote");
  auto in_data_anchor = node_item.node->GetInDataAnchor(remote_idx);
  GE_CHECK_NOTNULL(in_data_anchor);
  auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(out_data_anchor);
  auto peer_node = out_data_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(peer_node->GetOpDesc());

  remote_index_ = {peer_node->GetOpDesc()->GetId(), out_data_anchor->GetIdx()};
  if (node_item.node->GetType() == HCOMREMOTEREAD) {
    local_index_ = 0;
  } else {
    local_index_ = op_desc->GetInputIndexByName("local");
  }
  return SUCCESS;
}

Status RdmaNodeTask::ExtractTensor(TaskContext &context, vector<HcomRemoteAccessAddrInfo> &addr_infos) {
  RuntimeInferenceContext *ctx = nullptr;
  GE_CHK_STATUS_RET(RuntimeInferenceContext::GetContext(std::to_string(context.GetSessionId()), &ctx));

  ge::Tensor remote_tensor;
  GE_CHK_STATUS_RET(ctx->GetTensor(remote_index_.first, remote_index_.second, remote_tensor));
  auto data = reinterpret_cast<uint64_t *>(remote_tensor.GetData());
  if (data == nullptr) {
    GELOGE(FAILED, "Tensor data is nullptr.");
    return FAILED;
  }
  auto dims = remote_tensor.GetTensorDesc().GetShape().GetDims();
  if (dims.size() != kVarTableDims && dims.back() != kVarTableRowCnt) {
    GELOGE(PARAM_INVALID, "Variable table shape check failed");
    return PARAM_INVALID;
  }

  if (context.GetNodeItem().NodeType() == HCOMREMOTEREAD) {
    size_t remote_size = 0;
    for (auto idx = 0; idx < dims.front(); ++idx) {
      FMK_INT64_MULCHECK(idx, kVarTableRowCnt);
      auto line_idx = idx * kVarTableRowCnt;
      remote_size += data[line_idx + kVarTableIdxLen];
    }
    auto allocator = NpuMemoryAllocator::GetAllocator();
    GE_CHECK_NOTNULL(allocator);
    AllocationAttr attr;
    attr.SetMemType(RDMA_HBM);
    for (auto i = 0; i < context.NumOutputs(); ++i) {
      GELOGD("Allocate rdma memory for node %s, size: %zu", context.GetNodeName(), remote_size);
      auto tensor_buffer = TensorBuffer::Create(allocator, remote_size, &attr);
      GE_CHK_STATUS_RET(context.SetOutput(i, TensorValue(std::shared_ptr<TensorBuffer>(tensor_buffer.release()))));
    }
  }

  TensorValue *tv;
  if (context.GetNodeItem().NodeType() == HCOMREMOTEREAD) {
    tv = context.MutableOutput(0);
  } else {
    tv = context.MutableInput(local_index_);
  }
  GE_CHECK_NOTNULL(tv);
  auto local_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(tv->MutableData()));
  auto row_num = dims.front();
  addr_infos.resize(row_num);
  auto device_len = tv->GetSize() / row_num;
  if (device_len <= 0 || device_len > data[kVarTableIdxLen]) {
    GELOGE(FAILED, "Local embedding length is out of range.");
    return FAILED;
  }

  for (auto idx = 0; idx < row_num; ++idx) {
    FMK_INT64_MULCHECK(idx, kVarTableRowCnt);
    auto line_idx = idx * kVarTableRowCnt;
    addr_infos[idx] = {static_cast<uint32_t>(data[line_idx]), data[line_idx + kVarTableIdxAddr], local_addr,
                       device_len};
    local_addr += device_len;
  }

  return SUCCESS;
}

Status RdmaNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGI("[%s] RdmaNodeTask::ExecuteAsync in.", context.GetNodeName());
  auto HcomExecEnqueueRemoteAccess =
      (HcclResult(*)(const string &, const vector<HcomRemoteAccessAddrInfo> &,
                     std::function<void(HcclResult status)>))dlsym(context.handle_, "HcomExecEnqueueRemoteAccess");
  if (HcomExecEnqueueRemoteAccess == nullptr) {
    GELOGE(FAILED, "Failed to invoke HcomExecEnqueueRemoteAccess hcom unknown node function.");
    if (dlclose(context.handle_) != 0) {
      GELOGW("Failed to close handle %s", dlerror());
    }
    return FAILED;
  }
  vector<HcomRemoteAccessAddrInfo> addr_infos;
  GE_CHK_STATUS_RET(ExtractTensor(context, addr_infos));

  auto callback = [this](HcclResult status) {
    if (status != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "Call HcomExecInitialize failed, ret: 0x%X", status);
    }
    std::lock_guard<std::mutex> lock(this->hccl_mutex_);
    this->cond_.notify_all();
    GELOGI("rdma callback success.");
  };
  HcclResult hccl_ret = HcomExecEnqueueRemoteAccess(context.GetNodeItem().NodeType(), addr_infos, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    GELOGE(HCCL_E_INTERNAL, "Call HcomExecInitialize failed, ret: 0x%X", hccl_ret);
    return HCCL_E_INTERNAL;
  }

  // pending until hccl finished
  std::unique_lock<std::mutex> ulock(hccl_mutex_);
  cond_.wait(ulock);

  (void)context.RegisterCallback(done_callback);
  GELOGI("[%s] RdmaNodeTask::ExecuteAsync success.", context.GetNodeName());
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
  // allocate output mem, output mem or remote read will be calculated when node execute.
  if (context.GetNodeItem().NodeType() != HCOMREMOTEREAD) {
    GE_CHK_STATUS_RET(context.AllocateOutputs(), "hccl node task allocate output failed.");
  }

  GE_CHK_STATUS_RET(task.UpdateArgs(context), "hccl node task update args failed.");
  GELOGI("[%s] HcclNodeExecutor::PrepareTask success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GELOGI("[%s] HcclNodeExecutor::LoadTask in.", node->GetName().c_str());
  GE_CHECK_NOTNULL(node);
  if (node->GetType() == HCOMREMOTEREAD || node->GetType() == HCOMREMOTEWRITE) {
    task = MakeShared<RdmaNodeTask>();
  } else {
    task = MakeShared<HcclNodeTask>();
  }
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
  auto HcomExecInitialize = (HcclResult(*)())dlsym(handle_, "HcomExecInitialize");
  if (HcomExecInitialize == nullptr) {
    GELOGE(FAILED, "Failed to invoke HcomExecInitialize hcom unknown node function.");
    return FAILED;
  }
  HcclResult hccl_ret = HcomExecInitialize();
  if (hccl_ret == HCCL_E_PTR) {
    GELOGI("Hccl comm is null, hcom executor initialize is not required.");
  } else if (hccl_ret == HCCL_SUCCESS) {
    GELOGI("Hcom executor initialize success.");
  } else {
    GELOGE(FAILED, "Call HcomExecInitialize failed, ret: 0x%X", hccl_ret);
    return FAILED;
  }
  return SUCCESS;
}

Status HcclNodeExecutor::Finalize() {
  auto HcomExecFinalize = (HcclResult(*)())dlsym(handle_, "HcomExecFinalize");
  if (HcomExecFinalize == nullptr) {
    GELOGE(FAILED, "Failed to invoke HcomExecFinalize hcom unknown node function.");
    return FAILED;
  }
  HcclResult hccl_ret = HcomExecFinalize();
  if (hccl_ret != HCCL_SUCCESS) {
    GELOGE(FAILED, "Call HcomExecFinalize failed, ret: 0x%X", hccl_ret);
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
