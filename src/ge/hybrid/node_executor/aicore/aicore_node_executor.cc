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
#include "graph/debug/ge_attr_define.h"
#include "hybrid/model/hybrid_model.h"
#include "init/gelib.h"
#include "framework/common/debug/log.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICORE, AiCoreNodeExecutor);

AiCoreNodeTask::AiCoreNodeTask(std::vector<std::unique_ptr<AiCoreOpTask>> &&tasks) : tasks_(std::move(tasks)) {}

Status AiCoreNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  GELOGI("AiCoreNodeExecutor[%s] LoadTask Start.", node->GetName().c_str());

  auto *task_defs = model.GetTaskDefs(node);
  Status ret = SUCCESS;
  GE_IF_BOOL_EXEC(task_defs != nullptr && !task_defs->empty(), ret = CreateTask(model, *task_defs, node, task));

  GELOGI("AiCoreNodeExecutor[%s] LoadTask End, ret[%u].", node->GetName().c_str(), ret);
  return ret;
}

Status AiCoreNodeExecutor::GenNodeKey(const NodePtr &node, std::string &node_key) {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  // make sure unique, (op_id + input_shape) is unique
  node_key = std::to_string(op_desc->GetId()) + "/";
  node_key.append(std::to_string(op_desc->GetInputsSize()));
  auto input_descs = op_desc->GetAllInputsDesc();
  for (auto input_desc : input_descs) {
    node_key.push_back('/');
    std::vector<int64_t> dims = input_desc.GetShape().GetDims();
    GE_IF_BOOL_EXEC(dims.size() == 0, continue);  // scalar
    for (std::size_t i = 0; i < dims.size() - 1; i++) {
      node_key.append(std::to_string(dims[i]));
      node_key.push_back(',');
    }
    node_key.append(std::to_string(dims[dims.size() - 1]));
  }
  return SUCCESS;
}

bool AiCoreNodeTaskRegistry::AddTask(const std::string &node_key, const std::shared_ptr<NodeTask> task) {
  GE_CHECK_NOTNULL(task);
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = reg_node_tasks_.find(node_key);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(iter != reg_node_tasks_.end(), return false,
                                 "AiCoreNodeTaskRegistry[%s] AddTask failed, key already exist.", node_key.c_str());
  auto ret = reg_node_tasks_.emplace(node_key, task);
  return ret.second;
}

std::shared_ptr<NodeTask> AiCoreNodeTaskRegistry::GetTask(const std::string &node_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = reg_node_tasks_.find(node_key);
  return (iter != reg_node_tasks_.end()) ? iter->second : nullptr;
}

Status AiCoreNodeExecutor::CompileTask(const HybridModel &model, const NodePtr &node,
                                       shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  GELOGI("AiCoreNodeExecutor[%s] CompileTask Start.", node->GetName().c_str());

  AiCoreNodeTaskRegistry &registry = AiCoreNodeTaskRegistry::GetInstance();
  std::string node_key;
  GE_CHK_STATUS_RET(GenNodeKey(node, node_key), "GenNodeKey failed. op name = %s", node->GetName().c_str());

  GELOGD("NodeKey for %s = %s", node->GetName().c_str(), node_key.c_str());
  task = registry.GetTask(node_key);
  GE_CHK_TRUE_EXEC_INFO(task != nullptr, return SUCCESS, "AiCoreNodeExecutor[%s] CompileTask Skip.",
                        node->GetName().c_str());

  std::vector<domi::TaskDef> task_defs;
  GE_CHK_STATUS_RET_NOLOG(compiler_->CompileOp(node, task_defs));
  GELOGD("successfully generated task_defs: %s", node->GetName().c_str());

  GE_CHK_STATUS_RET_NOLOG(CreateTask(model, task_defs, node, task));
  GELOGD("successfully created node task: %s", node->GetName().c_str());

  GE_CHK_BOOL_EXEC(registry.AddTask(node_key, task), return INTERNAL_ERROR, "Add NodeTask failed. op name = %s",
                   node->GetName().c_str());  // should not happen.
  GELOGI("AiCoreNodeExecutor[%s] CompileTask End.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreNodeExecutor::BuildAiCoreTask(const domi::KernelDef &kernel_def, const OpDescPtr &op_desc,
                                           AiCoreOpTask **task) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(task);

  const auto &context = kernel_def.context();
  auto kernel_type = static_cast<cce::ccKernelType>(context.kernel_type());
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(kernel_type != cce::ccKernelType::TE, return UNSUPPORTED,
                                 "Only TBE kernel is supported, but [%s] got %u", op_desc->GetName().c_str(),
                                 context.kernel_type());

  auto *aicore_task = new (std::nothrow) AiCoreOpTask();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(aicore_task == nullptr, return MEMALLOC_FAILED, "Create AiCore op task failed.");

  auto builder = AiCoreTaskBuilder(op_desc, kernel_def);
  auto ret = builder.BuildTask(*aicore_task);
  GE_IF_BOOL_EXEC(ret != SUCCESS, delete aicore_task; aicore_task = nullptr; return ret);

  *task = aicore_task;
  return SUCCESS;
}

Status AiCoreNodeExecutor::CreateTask(const HybridModel &model, const std::vector<domi::TaskDef> &task_defs,
                                      const NodePtr &node, std::shared_ptr<NodeTask> &task) {
  GE_CHECK_NOTNULL(node);
  GELOGD("To CreateTask, task def size = %zu", task_defs.size());
  std::vector<std::unique_ptr<AiCoreOpTask>> aicore_op_tasks;
  aicore_op_tasks.reserve(task_defs.size());
  for (size_t i = 0; i < task_defs.size(); ++i) {
    const domi::TaskDef &task_def = task_defs[i];
    GELOGD("Op[%s] Task[%d], type = %u, DebugString = %s", node->GetName().c_str(), i, task_def.type(),
           task_def.DebugString().c_str());
    auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(task_type == RT_MODEL_TASK_KERNEL_EX, return UNSUPPORTED,
                                   "BuildKernelExTask is not supported");
    GE_CHK_BOOL_TRUE_EXEC_INFO(task_type != RT_MODEL_TASK_KERNEL, continue, "Skip task type %d",
                               static_cast<int>(task_type));

    const domi::KernelDef &kernel_def = task_def.kernel();
    AiCoreOpTask *aicore_op_task = nullptr;
    // not use hybrid model now
    GE_CHK_STATUS_RET_NOLOG(BuildAiCoreTask(kernel_def, node->GetOpDesc(), &aicore_op_task));
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(aicore_op_task == nullptr, return FAILED, "BuildAiCoreTask[%s] failed.",
                                   node->GetName().c_str());

    aicore_op_tasks.emplace_back(std::unique_ptr<AiCoreOpTask>(aicore_op_task));
  }

  if (!aicore_op_tasks.empty()) {
    auto aic_task = std::shared_ptr<NodeTask>(new AiCoreNodeTask(std::move(aicore_op_tasks)));
    task = std::move(aic_task);
    GELOGD("Generate AiCoreOpTask success");
    return SUCCESS;
  }

  GELOGE(INTERNAL_ERROR, "Failed to build task. node = %s", node->GetName().c_str());
  return INTERNAL_ERROR;
}

Status AiCoreNodeExecutor::Initialize() {
  std::shared_ptr<GELib> ge_lib = GELib::GetInstance();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((ge_lib == nullptr) || !ge_lib->InitFlag(), return GE_CLI_GE_NOT_INITIALIZED,
                                 "Get ge_lib failed.");

  auto &kernel_manager = ge_lib->OpsKernelManagerObj();
  auto aic_ops_store = kernel_manager.GetOpsKernelInfoStore("AIcoreEngine");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(aic_ops_store == nullptr, return GE_CLI_GE_NOT_INITIALIZED,
                                 "Failed to get kernel info store for AIcoreEngine.");

  compiler_.reset(new (std::nothrow) AiCoreTaskCompiler(aic_ops_store));
  GE_CHECK_NOTNULL(compiler_);
  return SUCCESS;
}

Status AiCoreNodeExecutor::Finalize() { return NodeExecutor::Finalize(); }

Status AiCoreNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  auto op_desc = context.GetNodeItem().op_desc;
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("AiCoreNodeTask[%s] ExecuteAsync Start.", op_desc->GetName().c_str());
  for (size_t i = 0; i < tasks_.size(); i++) {
    GE_CHECK_NOTNULL(tasks_[i]);
    GE_CHK_STATUS_RET_NOLOG(tasks_[i]->LaunchKernel(context.GetStream()));
  }

  if (done_callback != nullptr) {
    GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
  }

  GELOGI("AiCoreNodeTask[%s] ExecuteAsync End.", op_desc->GetName().c_str());
  return SUCCESS;
}

Status AiCoreNodeTask::UpdateAtomicArgs(TaskContext &context, std::unique_ptr<AiCoreOpTask> &task) {
  GE_CHECK_NOTNULL(task);
  auto op_desc = context.GetNodeItem().op_desc;
  GE_CHECK_NOTNULL(op_desc);

  // refresh atomic output addr
  std::vector<int64_t> atomic_output_indexes;  // here atomic just clean output
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indexes);
  GE_RETURN_WITH_LOG_IF_TRUE(atomic_output_indexes.size() > static_cast<size_t>(context.NumOutputs()),
                             "AtomicAddrClean op's arg_size error.");
  auto *arg_off = reinterpret_cast<uint8_t *>(task->args_.get()) + task->offset_;
  auto *arg_base = reinterpret_cast<uintptr_t *>(arg_off);
  int index = 0;
  for (size_t i = 0; i < atomic_output_indexes.size(); ++i) {
    const auto output = context.GetOutput(atomic_output_indexes[i]);
    GE_CHECK_NOTNULL(output);
    arg_base[index++] = reinterpret_cast<uintptr_t>(output->GetData());
  }

  // refresh atomic workspace addr
  auto workspace_sizes = op_desc->GetWorkspaceBytes();
  uint64_t ops_workspace_num = static_cast<uint64_t>(workspace_sizes.size());
  uint64_t workspace_num = static_cast<uint64_t>(context.NumWorkspaces());
  GE_CHK_BOOL_EXEC(ops_workspace_num == workspace_num, return PARAM_INVALID,
                   "The workspace_num in op_desc %lu is not equal to it %lu in context.", ops_workspace_num,
                   workspace_num);
  GE_IF_BOOL_EXEC(workspace_num == 0, return SUCCESS);

  map<string, map<int64_t, int64_t>> workspace_info;
  workspace_info = op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);
  if (!workspace_info.empty()) {
    bool is_fusion_node = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(is_fusion_node, return PARAM_INVALID,
                                   "Atomic desc[%s] shouldn't be fusion_node in AiCoreNodeTask",
                                   op_desc->GetName().c_str());

    for (auto iter = workspace_info.begin(); iter != workspace_info.end(); ++iter) {
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(op_desc->GetName() != iter->first, return PARAM_INVALID,
                                     "The node name %s and the node name %s in workspace info are inconsistent.",
                                     op_desc->GetName().c_str(), iter->first.c_str());
      GE_IF_BOOL_EXEC(iter->second.empty(), continue);

      for (auto &info_iter : iter->second) {
        auto workspace_index = static_cast<uint64_t>(info_iter.first);

        GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(workspace_index >= workspace_num, return PARAM_INVALID,
                                       "The workspace index %lu is more than the size %lu of workspace vector.",
                                       workspace_index, workspace_num);

        const auto workspace = context.MutableWorkspace(workspace_index);
        arg_base[index++] = reinterpret_cast<uintptr_t>(workspace);
      }
    }
  }

  return SUCCESS;
}

Status AiCoreNodeTask::UpdateAllArgs(TaskContext &context, std::unique_ptr<AiCoreOpTask> &task) {
  GE_CHECK_NOTNULL(task);
  auto *arg_off = reinterpret_cast<uint8_t *>(task->args_.get()) + task->offset_;
  auto *arg_base = reinterpret_cast<uintptr_t *>(arg_off);
  int index = 0;
  for (int i = 0; i < context.NumInputs(); ++i) {
    const auto input = context.GetInput(i);
    GE_CHECK_NOTNULL(input);
    arg_base[index++] = reinterpret_cast<uintptr_t>(input->GetData());
  }

  for (int i = 0; i < context.NumOutputs(); ++i) {
    const auto output = context.GetOutput(i);
    GE_CHECK_NOTNULL(output);
    arg_base[index++] = reinterpret_cast<uintptr_t>(output->GetData());
  }

  auto op_desc = context.GetNodeItem().op_desc;
  GE_CHECK_NOTNULL(op_desc);
  auto workspace_sizes = op_desc->GetWorkspaceBytes();
  int ops_workspace_num = static_cast<int>(workspace_sizes.size());
  int workspace_num = static_cast<int>(context.NumWorkspaces());
  GE_CHK_BOOL_EXEC(ops_workspace_num == workspace_num, return PARAM_INVALID,
                   "The workspace_num in op_desc %lu is not equal to it %lu in context.", ops_workspace_num,
                   workspace_num);
  for (int i = 0; i < workspace_num; ++i) {
    const auto workspace = context.MutableWorkspace(i);
    arg_base[index++] = reinterpret_cast<uintptr_t>(workspace);
  }

  return SUCCESS;
}

Status AiCoreNodeTask::UpdateArgs(TaskContext &context) {
  auto op_desc = context.GetNodeItem().op_desc;
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("AiCoreNodeTask[%s] UpdateArgs Start.", op_desc->GetName().c_str());
  GE_IF_BOOL_EXEC(tasks_.size() == 1, return UpdateAllArgs(context, tasks_[0]));

  std::vector<int64_t> atomic_output_indexes;  // here atomic just clean output
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indexes);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(atomic_output_indexes.empty(), return FAILED, "ATOMIC_ATTR_OUTPUT_INDEX is empty.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(tasks_.size() != 2, return FAILED, "AtomicAddrClean op task num != 2.");

  GE_CHK_STATUS_RET_NOLOG(UpdateAtomicArgs(context, tasks_[0]));
  GE_CHK_STATUS_RET_NOLOG(UpdateAllArgs(context, tasks_[1]));

  GELOGI("AiCoreNodeTask[%s] UpdateArgs End.", op_desc->GetName().c_str());
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
