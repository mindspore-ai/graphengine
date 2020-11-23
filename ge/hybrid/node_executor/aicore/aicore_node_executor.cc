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

#include "aicore_node_executor.h"
#include "cce/taskdown_common.hpp"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICORE, AiCoreNodeExecutor);

AiCoreNodeTask::AiCoreNodeTask(std::vector<std::unique_ptr<AiCoreOpTask>> &&tasks) : tasks_(std::move(tasks)) {
}

Status AiCoreNodeExecutor::Initialize() {
  compiler_ = TaskCompilerFactory::GetInstance().GetTaskCompiler();
  if (compiler_ != nullptr) {
    GE_CHK_STATUS_RET(compiler_->Initialize(), "Failed to init aicore task compiler.");
  }
  return SUCCESS;
}

Status AiCoreNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  GELOGI("AiCoreNodeExecutor(%s) LoadTask Start.", node->GetName().c_str());

  auto *task_defs = model.GetTaskDefs(node);
  if (task_defs == nullptr || task_defs->empty()) {
    bool dynamic_flag = false;
    if (!AttrUtils::GetBool(node->GetOpDesc(), "support_dynamicshape", dynamic_flag) || !dynamic_flag) {
      GELOGD("Skip create task of node (%s) as 'support_dynamicshape' is false and cann't get task_defs.",
             node->GetName().c_str());
      return SUCCESS;
    } else {
      GELOGE(FAILED, "Task_defs is empty for node (%s) which 'support_dynamicshape' is true, failed.",
             node->GetName().c_str());
      return FAILED;
    }
  }

  AiCoreTaskBuilder builder(node->GetOpDesc(), *task_defs);
  std::unique_ptr<NodeTask> node_task;
  GE_CHK_STATUS_RET(builder.BuildTask(node_task, true), "[%s] Failed to build op tasks.", node->GetName().c_str());
  task = std::move(node_task);
  GELOGI("AiCoreNodeExecutor(%s) LoadTask End.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreNodeExecutor::GenNodeKey(const NodePtr &node, std::string &node_key) {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  // make sure unique, (op_id + input_shape) is unique
  node_key = std::to_string(op_desc->GetId()) + "-";
  node_key.append(std::to_string(op_desc->GetInputsSize()));
  auto input_descs = op_desc->GetAllInputsDescPtr();
  for (auto &input_desc : input_descs) {
    node_key.push_back('-');
    auto &shape = input_desc->MutableShape();
    auto num_dims = shape.GetDimNum();
    if (num_dims == 0) {
      continue;
    } // scalar
    for (std::size_t i = 0; i < num_dims - 1; i++) {
      node_key.append(std::to_string(shape.GetDim(i)));
      node_key.push_back('_');
    }
    node_key.append(std::to_string(shape.GetDim(num_dims - 1)));
  }
  return SUCCESS;
}

bool AiCoreNodeTaskRegistry::AddTask(const std::string &node_key, const std::shared_ptr<NodeTask> task) {
  GE_CHECK_NOTNULL(task);
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = reg_node_tasks_.find(node_key);
  if (iter != reg_node_tasks_.end()) {
    GELOGE(FAILED, "AiCoreNodeTaskRegistry(%s) AddTask failed, key already exist.", node_key.c_str());
    return false;
  }
  auto ret = reg_node_tasks_.emplace(node_key, task);
  return ret.second;
}

std::shared_ptr<NodeTask> AiCoreNodeTaskRegistry::GetTask(const std::string &node_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = reg_node_tasks_.find(node_key);
  return (iter != reg_node_tasks_.end()) ? iter->second : nullptr;
}

Status AiCoreNodeExecutor::CompileTask(const HybridModel &model,
                                       const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("AiCoreNodeExecutor(%s) CompileTask Start.", node->GetName().c_str());

  auto ori_node_name = node->GetName();
  if (compiler_ == nullptr) {
    GELOGE(FAILED, "[%s] Can not find any valid aicore task compiler.", ori_node_name.c_str());
    return FAILED;
  }

  AiCoreNodeTaskRegistry &registry = AiCoreNodeTaskRegistry::GetInstance();
  std::string shape_key;
  GE_CHK_STATUS_RET(GenNodeKey(node, shape_key), "GenNodeKey failed, op name = %s.", node->GetName().c_str());

  auto node_key = std::to_string(model.GetModelId()) + "/" + shape_key;
  GELOGD("NodeKey for %s = %s", node->GetName().c_str(), node_key.c_str());
  task = registry.GetTask(node_key);
  if (task != nullptr) {
    GELOGI("AiCoreNodeExecutor(%s) CompileTask Skip.", node->GetName().c_str());
    return SUCCESS;
  }

  std::vector<domi::TaskDef> task_defs;
  op_desc->SetName(ori_node_name + "_" + shape_key);
  GE_CHK_STATUS_RET(compiler_->CompileOp(node, task_defs), "Compile op(%s) failed.", ori_node_name.c_str());
  op_desc->SetName(ori_node_name);
  GELOGD("successfully generated task_defs: %s", node->GetName().c_str());

  AiCoreTaskBuilder builder(node->GetOpDesc(), task_defs);
  std::unique_ptr<NodeTask> node_task;
  GE_CHK_STATUS_RET(builder.BuildTask(node_task, false), "[%s] Failed to build op tasks.", node->GetName().c_str());
  task = std::move(node_task);
  GELOGD("successfully created node task: %s", node->GetName().c_str());

  if (!registry.AddTask(node_key, task)) {
    GELOGE(INTERNAL_ERROR, "Add NodeTask failed, op name = %s.", node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGI("AiCoreNodeExecutor(%s) CompileTask End.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeTaskExecuteAsync] Start");
  if (IsNoOp(context)) {
    GELOGD("[%s] Skipping execution for op with empty outputs", context.GetNodeName());
    auto ret = context.TryExecuteCallback(done_callback);
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeTaskExecuteAsync] End");
    return ret;
  }

  auto op_desc = context.GetNodeItem().op_desc;
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("[%s] ExecuteAsync Start.", op_desc->GetName().c_str());
  for (auto it = tasks_.begin(); it != tasks_.end(); ++it) {
    // AtomicAddrClean has 2 tasks
    if (tasks_.size() == 2 && it == tasks_.begin() && !(*(tasks_.rbegin()))->GetClearAtomic()) {
      continue;
    }
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeLaunchKernel] Start");
    GE_CHK_STATUS_RET_NOLOG((*it)->LaunchKernel(context.GetStream()));
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeLaunchKernel] End");
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeLaunchKernel] End");
  }

  if (done_callback != nullptr) {
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeRegisterCallback] Start");
    GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeRegisterCallback] End");
  }

  GELOGD("[%s] ExecuteAsync End.", op_desc->GetName().c_str());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeTaskExecuteAsync] End");
  return SUCCESS;
}

Status AiCoreNodeTask::UpdateArgs(TaskContext &context) {
  auto op_desc = context.GetNodeItem().op_desc;
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("[%s] AiCoreNodeTask UpdateArgs Start.", op_desc->GetName().c_str());
  for (auto it = tasks_.rbegin(); it != tasks_.rend(); ++it) {
    GE_CHK_STATUS_RET_NOLOG((*it)->UpdateArgs(context));
    // AtomicAddrClean has 2 tasks
    if (tasks_.size() == 2 && it == tasks_.rbegin() && !(*it)->GetClearAtomic()) {
      break;
    }
  }
  GELOGI("[%s] AiCoreNodeTask UpdateArgs End.", op_desc->GetName().c_str());
  return SUCCESS;
}

Status AiCoreNodeTask::UpdateTilingData(TaskContext &context) {
  GELOGD("[%s] PrepareWithShape started", context.GetNodeName());
  for (auto it = tasks_.rbegin(); it != tasks_.rend(); ++it) {
    GE_CHK_STATUS_RET_NOLOG((*it)->PrepareWithShape(context));
    // AtomicAddrClean has 2 tasks
    if (tasks_.size() == 2 && it == tasks_.rbegin() && !(*it)->GetClearAtomic()) {
      break;
    }
  }
  GELOGD("[%s] Done PrepareWithShape successfully.", context.GetNodeName());
  return SUCCESS;
}

bool AiCoreNodeTask::IsSupportDynamicShape() {
  for (size_t i = 0; i < tasks_.size(); ++i) {
    if (!tasks_[i]->IsDynamicShapeSupported()) {
      GELOGD("[%s] Task does not support dynamic shape.", tasks_[i]->GetName().c_str());
      return false;
    }
  }

  return true;
}

bool AiCoreNodeTask::IsNoOp(TaskContext &task_context) {
  for (int i = 0; i < task_context.NumOutputs(); ++i) {
    const auto &tensor_desc = task_context.MutableOutputDesc(i);
    GE_CHECK_NOTNULL(tensor_desc);
    const auto &shape = tensor_desc->MutableShape();
    if (shape.IsScalar() || shape.GetShapeSize() > 0) {
      return false;
    }
  }

  return true;
}

TaskCompilerFactory &TaskCompilerFactory::GetInstance() {
  static TaskCompilerFactory instance;
  return instance;
}

void TaskCompilerFactory::Register(CreateFn fn) {
  compiler_func_ = fn;
}

std::unique_ptr<TaskCompiler> TaskCompilerFactory::GetTaskCompiler() {
  if (compiler_func_ == nullptr) {
    return nullptr;
  }
  auto compiler_instance = std::unique_ptr<TaskCompiler>(compiler_func_());
  return compiler_instance;
}

CompilerFunctionRegistrar::CompilerFunctionRegistrar(CreateFn fn) {
  TaskCompilerFactory::GetInstance().Register(fn);
}
}  // namespace hybrid
}  // namespace ge
