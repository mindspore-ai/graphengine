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

#include "hybrid/node_executor/compiledsubgraph/known_node_executor.h"
#include "cce/aicpu_engine_struct.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "common/ge/ge_util.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::COMPILED_SUBGRAPH, KnownNodeExecutor);

Status KnownNodeTask::   ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeTaskExecuteAsync] Start");
  GELOGD("[%s] KnownNodeTask::ExecuteAsync in.", context.GetNodeName());
  if (davinci_model_->GetTaskList().empty()) {
    GELOGW("KnownNodeExecutor::ExecuteAsync davinci model has no taskinfo.");

    // todo if data is connected to netoutput, forward address ? copy data?
    if (context.NumInputs() == context.NumOutputs()){
      for (int i = 0; i < context.NumInputs(); ++i) {
        auto tensor = context.MutableInput(i);
        GE_CHECK_NOTNULL(tensor);
        GE_CHK_STATUS_RET(context.SetOutput(i, *tensor),
                          "[%s] Failed to set output[%d]",
                          context.GetNodeName(),
                          i);
      }
    }

    GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
    return SUCCESS;
  }

  rtError_t rt_ret;
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodertModelExecute] Start");
  rt_ret = rtModelExecute(davinci_model_->GetRtModelHandle(), context.GetStream(), 0);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  GELOGE(rt_ret, "rtModelExecute error, ret: hybrid_model_executorOx%X", rt_ret); return FAILED;);
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodertModelExecute] End");

  GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
  GELOGD("[%s] KnownNodeTask::ExecuteAsync success.", context.GetNodeName());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeTaskExecuteAsync] End");
  return SUCCESS;
}

Status KnownNodeTask::UpdateArgs(TaskContext &context) {
  GELOGD("[%s] KnownNodeExecutor::UpdateArgs in.", context.GetNodeName());
  if (davinci_model_->GetTaskList().empty()) {
    GELOGW("KnownNodeExecutor::UpdateArgs davinci model has no taskinfo.");
    return SUCCESS;
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

  GE_CHK_STATUS_RET(davinci_model_->UpdateKnownNodeArgs(inputs, outputs),
                    "known node task update known node args failed.");
  GELOGD("[%s] KnownNodeExecutor::UpdateArgs success, task_size = %zu", context.GetNodeName(),
         davinci_model_->GetTaskList().size());
  return SUCCESS;
}

Status KnownNodeTask::Init(TaskContext &context) {
  // allocate output mem
  GE_CHK_STATUS_RET(context.AllocateOutputs(), "known node task allocate output failed.");
  // allocate mem base
  void *buffer = nullptr;
  if (davinci_model_->TotalMemSize() != 0) {
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                           "[KnownNodeTask_AllocateWorkspace] Start");
    GE_CHK_STATUS_RET(
        context.AllocateWorkspace(davinci_model_->TotalMemSize(), &buffer, davinci_model_->GetRuntimeParam().mem_base),
        "known node task allocate workspace failed.");
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                           "[KnownNodeTask_AllocateWorkspace] End, size %zu", davinci_model_->TotalMemSize());
    // update mem base
    davinci_model_->UpdateMemBase(static_cast<uint8_t *>(buffer));
    GELOGI("KnownNodeTask::Init mem base is %p, size %lu.",
           davinci_model_->GetRuntimeParam().mem_base, davinci_model_->GetRuntimeParam().mem_size);
  }
  if (!load_flag_) {
    auto dump_properties = context.GetDumpProperties();
    if (dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen()) {
      davinci_model_->SetDumpProperties(dump_properties);
      void *global_step = context.GetExecutionContext()->global_step;
      davinci_model_->SetKnownShapeGlobalStep(global_step);
    }
    load_flag_ = true;
  }
  GE_CHK_STATUS_RET(ModelManager::GetInstance()->DestroyAicpuKernel(davinci_model_->GetSessionId(),
                                                                    davinci_model_->Id(), davinci_model_->SubModelId()),
                    "KnownNodeTask::Init destroy aicpu kernel failed.");
  GELOGI("[%s] KnownNodeExecutor::Init success.", context.GetNodeName());
  return SUCCESS;
}

Status KnownNodeTask::InitDavinciModel() {
  GELOGD("[Init][Model] start");
  davinci_model_->InitRuntimeParams();
  GE_CHK_STATUS_RET(davinci_model_->InitVariableMem(), "init variable mem failed");
  int32_t device_id = 0;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  davinci_model_->SetDeviceId(static_cast<uint32_t>(device_id));
  GE_CHK_STATUS_RET(DoInitDavinciModel(), "[Init][Model] Failed to init davinci model.");
  GELOGD("[Init][Model] success");
  return SUCCESS;
}

Status KnownNodeTask::DoInitDavinciModel() {
  return davinci_model_->Init();
}

Status KnownNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GELOGD("[%s] KnownNodeExecutor::PrepareTask in.", context.GetNodeName());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorPrepareTask] Start");
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorTaskInit] Start");
  GE_CHK_STATUS_RET(task.Init(context), "known node init davinci model failed.");
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorTaskInit] End");

  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorUpdateArgs] Start");
  GE_CHK_STATUS_RET(task.UpdateArgs(context), "known node task update args failed.");
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorUpdateArgs] End");
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorPrepareTask] End");
  GELOGD("[%s] KnownNodeExecutor::PrepareTask success.", context.GetNodeName());
  return SUCCESS;
}

Status KnownNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node,
                                   shared_ptr<NodeTask> &task) const {
  GELOGI("[%s] KnownNodeExecutor::LoadTask in.", node->GetName().c_str());
  GE_CHECK_NOTNULL(node);

  const GeModelPtr ge_model = model.GetGeModel(node);
  GE_CHECK_NOTNULL(ge_model);

  std::shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(0, nullptr);
  GE_CHECK_NOTNULL(davinci_model);

  // set known node flag as true
  davinci_model->SetKnownNode(true);
  davinci_model->SetId(model.GetModelId());
  davinci_model->SetOmName(model.GetOmName());
  // set model id as root node's node id
  davinci_model->SetSubModelId(node->GetOpDesc()->GetId());
  GELOGD("KnownNodeExecutor::LoadTask node id %ld.", node->GetOpDesc()->GetId());

  GE_CHK_STATUS_RET(davinci_model->Assign(ge_model), "KnownNodeExecutor::LoadTask davincimodel assign failed.");

  auto known_node_task = MakeShared<KnownNodeTask>(davinci_model);
  GE_CHECK_NOTNULL(known_node_task);
  GE_CHK_STATUS_RET_NOLOG(known_node_task->InitDavinciModel());
  GELOGI("[%s] KnownNodeExecutor::LoadTask success.", node->GetName().c_str());
  task = std::move(known_node_task);
  return SUCCESS;
}

Status KnownNodeExecutor::ExecuteTask(NodeTask &task, TaskContext &context,
                                      const std::function<void()> &callback) const {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorExecuteTask] Start");
  GE_CHK_STATUS_RET(task.ExecuteAsync(context, callback),
                    "Failed to execute task. node = %s",
                    context.GetNodeItem().NodeName().c_str());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorExecuteTask] End");
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
