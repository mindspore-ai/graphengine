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

#include "hybrid/node_executor/node_executor.h"
#include "framework/common/debug/log.h"
#include "graph/utils/node_utils.h"
#include "init/gelib.h"
#include "hybrid/model/hybrid_model.h"

namespace ge {
namespace hybrid {
namespace {
const char *const kEngineNameAiCore = "AIcoreEngine";
const char *const kEngineNameGeLocal = "DNN_VM_GE_LOCAL_OP_STORE";
const char *const kEngineNameAiCpu = "aicpu_kernel";
const char *const kEngineNameHccl = "ops_kernel_info_hccl";
}  // namespace
Status NodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
  GE_CHK_STATUS_RET_NOLOG(task.UpdateTilingData(context));  // update op_desc before alloc ws
  GE_CHK_STATUS_RET_NOLOG(context.AllocateWorkspaces());
  GE_CHK_STATUS_RET_NOLOG(task.UpdateArgs(context));
  return SUCCESS;
}

Status NodeExecutor::ExecuteTask(NodeTask &task, TaskContext &context, const std::function<void()> &callback) const {
  GE_CHK_STATUS_RET(task.ExecuteAsync(context, callback), "Failed to execute task. node = %s",
                    context.GetNodeItem().NodeName().c_str());
  return SUCCESS;
}

Status NodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  return UNSUPPORTED;
}

Status NodeExecutor::CompileTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  return UNSUPPORTED;
}

Status NodeExecutorManager::EnsureInitialized() {
  GE_CHK_STATUS_RET(InitializeExecutors());
  std::lock_guard<std::mutex> lk(mu_);
  if (initialized_) {
    return SUCCESS;
  }

  engine_mapping_.emplace(kEngineNameAiCore, NodeExecutorManager::ExecutorType::AICORE);
  engine_mapping_.emplace(kEngineNameGeLocal, NodeExecutorManager::ExecutorType::GE_LOCAL);
  engine_mapping_.emplace(kEngineNameAiCpu, NodeExecutorManager::ExecutorType::AICPU_TF);
  engine_mapping_.emplace(kEngineNameHccl, NodeExecutorManager::ExecutorType::HCCL);

  std::shared_ptr<GELib> instance_ptr = GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGW("GELib not initialized");
    return FAILED;
  }

  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  for (auto &it : ops_kernel_manager.GetAllOpsKernelInfoStores()) {
    GELOGD("add kernel store: %s", it.first.c_str());
    kernel_stores_.emplace(it.first, it.second);
  }

  initialized_ = true;
  GELOGI("Initializing NodeExecutors successfully");
  return SUCCESS;
}

NodeExecutorManager::ExecutorType NodeExecutorManager::ResolveExecutorType(Node &node) const {
  auto op_type = node.GetType();
  if (op_type == PARTITIONEDCALL) {
    bool is_dynamic = false;
    (void)NodeUtils::GetNodeUnknownShapeStatus(node, is_dynamic);
    if (is_dynamic) {
      return ExecutorType::DYNAMIC_SUBGRAPH;
    }
    return ExecutorType::COMPILED_SUBGRAPH;
  }

  // rts kernel store is assigned to NetOutput
  if (op_type == NETOUTPUT || op_type == VARIABLE) {
    return ExecutorType::GE_LOCAL;
  }

  if (op_type == IF || op_type == CASE || op_type == WHILE) {
    return ExecutorType::CONTROL_OP;
  }

  auto op_desc = node.GetOpDesc();  // checked before
  const auto &lib_name = op_desc->GetOpKernelLibName();
  auto it = engine_mapping_.find(lib_name);
  if (it == engine_mapping_.end()) {
    GELOGE(UNSUPPORTED, "KernelLib not supported. node = %s, lib_name = %s", node.GetName().c_str(), lib_name.c_str());
    return ExecutorType::RESERVED;
  }

  return it->second;
}

Status NodeExecutorManager::GetExecutor(Node &node, const NodeExecutor **executor) const {
  auto executor_type = ResolveExecutorType(node);
  const auto it = executors_.find(executor_type);
  if (it == executors_.end()) {
    GELOGE(INTERNAL_ERROR, "Failed to get executor by type: %d.", executor_type);
    return INTERNAL_ERROR;
  }

  GELOGD("[%s] Set node executor by type: %d.", node.GetName().c_str(), executor_type);
  *executor = it->second.get();
  return SUCCESS;
}

void NodeExecutorManager::RegisterExecutorBuilder(NodeExecutorManager::ExecutorType executor_type,
                                                  const std::function<NodeExecutor *()> &builder) {
  builders_.emplace(executor_type, builder);
}

Status NodeExecutorManager::CalcOpRunningParam(Node &node) const {
  auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetType() == PARTITIONEDCALL) {
    GELOGD("[%s] Skipping CalcOpRunningParam for PartitionedCall.", node.GetName().c_str());
    return SUCCESS;
  }

  auto it = kernel_stores_.find(op_desc->GetOpKernelLibName());
  if (it == kernel_stores_.end()) {
    GELOGE(INTERNAL_ERROR, "Failed to get OpKernelStore. libName = %s, node = %s",
           op_desc->GetOpKernelLibName().c_str(), op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  // calc hccl output size independent, hccl ops kernel manager should GetSize for
  // input which is the output size of input-op, but sometimes return error
  // when multi-thread
  if (op_desc->GetOpKernelLibName() == kEngineNameHccl) {
    for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
      GeTensorDesc output_tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
      Format format = output_tensor.GetFormat();
      DataType data_type = output_tensor.GetDataType();
      GeShape output_shape = output_tensor.GetShape();
      int64_t output_mem_size = 0;
      GE_CHK_STATUS_RET(TensorUtils::CalcTensorMemSize(output_shape, format, data_type, output_mem_size),
                        "hccl calc tensor mem size failed.");
      output_mem_size =
        ((output_mem_size + MEMORY_ALIGN_RATIO * MEMORY_ALIGN_SIZE - 1) / MEMORY_ALIGN_SIZE) * MEMORY_ALIGN_SIZE;
      TensorUtils::SetSize(output_tensor, output_mem_size);
      GE_CHK_STATUS_RET(op_desc->UpdateOutputDesc(static_cast<uint32_t>(i), output_tensor),
                        "hccl update output size failed.");
      GELOGD("%s output desc[%u], dim_size: %zu, mem_size: %ld.", node.GetName().c_str(), i,
             output_tensor.GetShape().GetDimNum(), output_mem_size);
    }
    return SUCCESS;
  }
  return it->second->CalcOpRunningParam(node);
}

Status NodeExecutorManager::InitializeExecutors() {
  std::lock_guard<std::mutex> lk(mu_);
  if (executor_initialized_) {
    ++ref_count_;
    GELOGI("Executor is already initialized. add ref count to [%d]", ref_count_);
    return SUCCESS;
  }

  GELOGI("Start to Initialize NodeExecutors");
  for (auto &it : builders_) {
    auto engine_type = it.first;
    auto build_fn = it.second;
    GE_CHECK_NOTNULL(build_fn);
    auto executor = std::unique_ptr<NodeExecutor>(build_fn());
    if (executor == nullptr) {
      GELOGE(INTERNAL_ERROR, "Failed to create executor for engine type = %d", engine_type);
      return INTERNAL_ERROR;
    }

    GELOGD("Executor of engine type = %d was created successfully", engine_type);
    auto ret = executor->Initialize();
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to initialize NodeExecutor of type = %d, clear executors", engine_type);
      for (auto &executor_it : executors_) {
        executor_it.second->Finalize();
      }
      executors_.clear();
      return ret;
    }

    executors_.emplace(engine_type, std::move(executor));
  }

  ++ref_count_;
  executor_initialized_ = true;
  GELOGI("Initializing NodeExecutors successfully.");
  return SUCCESS;
}

void NodeExecutorManager::FinalizeExecutors() {
  std::lock_guard<std::mutex> lk(mu_);
  if (!executor_initialized_) {
    GELOGD("No need for finalizing for not initialized.");
    return;
  }

  if (--ref_count_ > 0) {
    GELOGD("Ref count = %d, do not finalize executors.", ref_count_);
    return;
  }

  GELOGD("Start to invoke Finalize on executors.");
  for (auto &it : executors_) {
    it.second->Finalize();
  }
  executors_.clear();
  executor_initialized_ = false;
  GELOGD("Done invoking Finalize successfully.");
}

NodeExecutorRegistrar::NodeExecutorRegistrar(NodeExecutorManager::ExecutorType executor_type,
                                             NodeExecutor *(*builder)()) {
  NodeExecutorManager::GetInstance().RegisterExecutorBuilder(executor_type, builder);
}
}  // namespace hybrid
}  // namespace ge
