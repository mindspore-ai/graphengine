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
#include "init/gelib.h"
#include "hybrid/model/hybrid_model.h"

namespace ge {
namespace hybrid {
namespace {
const char *const kEngineNameAiCore = "AIcoreEngine";
const char *const kEngineNameGeLocal = "DNN_VM_GE_LOCAL_OP_STORE";
const char *const kEngineNameAiCpu = "aicpu_kernel";
}  // namespace
Status NodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
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
  std::lock_guard<std::mutex> lk(mu_);
  if (initialized_) {
    return SUCCESS;
  }

  engine_mapping_.emplace(kEngineNameAiCore, NodeExecutorManager::ExecutorType::AICORE);
  engine_mapping_.emplace(kEngineNameGeLocal, NodeExecutorManager::ExecutorType::GE_LOCAL);
  engine_mapping_.emplace(kEngineNameAiCpu, NodeExecutorManager::ExecutorType::AICPU_TF);

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
    GE_CHK_STATUS_RET(executor->Initialize(), "Failed to initialize NodeExecutor of type = %d", engine_type);
    executors_.emplace(engine_type, std::move(executor));
  }

  initialized_ = true;
  GELOGI("Initializing NodeExecutors successfully");
  return SUCCESS;
}

NodeExecutorManager::ExecutorType NodeExecutorManager::ResolveExecutorType(Node &node) const {
  auto op_type = node.GetType();
  if (op_type == PARTITIONEDCALL) {
    return ExecutorType::COMPILED_SUBGRAPH;
  }

  // rts kernel store is assigned to NetOutput
  if (op_type == NETOUTPUT || op_type == VARIABLE) {
    return ExecutorType::GE_LOCAL;
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
    GELOGE(INTERNAL_ERROR, "Failed to get executor by type: %d", executor_type);
    return INTERNAL_ERROR;
  }

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
  auto it = kernel_stores_.find(op_desc->GetOpKernelLibName());
  if (it == kernel_stores_.end()) {
    GELOGE(INTERNAL_ERROR, "Failed to get OpKernelStore. libName = %s, node = %s",
           op_desc->GetOpKernelLibName().c_str(), op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  return it->second->CalcOpRunningParam(node);
}

NodeExecutorRegistrar::NodeExecutorRegistrar(NodeExecutorManager::ExecutorType executor_type,
                                             NodeExecutor *(*builder)()) {
  NodeExecutorManager::GetInstance().RegisterExecutorBuilder(executor_type, builder);
}
}  // namespace hybrid
}  // namespace ge