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

#include "hybrid/node_executor/compiledsubgraph/known_node_executor.h"
#include "cce/aicpu_engine_struct.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "common/ge/ge_util.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/load/new_model_manager/model_manager.h"

namespace ge {
namespace hybrid {

REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::COMPILED_SUBGRAPH, KnownNodeExecutor);

Status KnownNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGI("[%s] KnownNodeTask::ExecuteAsync in.", context.GetNodeName());
  if (davinci_model_->GetTaskList().size() == 0) {
    GELOGW("KnownNodeExecutor::ExecuteAsync davinci moel has no taskinfo.");

    // todo if data is connected to netoutput, forward address ? copy data?
    if (context.NumInputs() == context.NumOutputs()) {
      GELOGW("[%s] KnownNodeExecutor::ExecuteAsync davinci moel has no taskinfo.", context.GetNodeName());
      for (int i = 0; i < context.NumInputs(); ++i) {
        auto tensor = context.MutableInput(i);
        GE_CHK_STATUS_RET(context.SetOutput(i, *tensor), "[%s] Failed to set output[%d]", context.GetNodeName(), i);
      }
    }

    context.RegisterCallback(done_callback);
    return SUCCESS;
  }

  rtError_t rt_ret;
  GELOGI("rtModelExecute start.");
  rt_ret = rtModelExecute(davinci_model_->GetRtModelHandle(), davinci_model_->GetRtModelStream(), 0);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtModelExecute error, ret: Ox%X", rt_ret); return FAILED;);
  GELOGI("rtModelExecute end");

  GELOGI("rtStreamSynchronize start.");
  rt_ret = rtStreamSynchronize(davinci_model_->GetRtModelStream());
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtStreamSynchronize error, ret: Ox%X", rt_ret);
                  return FAILED;);
  GELOGI("rtStreamSynchronize end.");

  context.RegisterCallback(done_callback);
  GELOGI("[%s] KnownNodeTask::ExecuteAsync success.", context.GetNodeName());

  return SUCCESS;
}

Status KnownNodeTask::UpdateArgs(TaskContext &context) {
  GELOGI("[%s] KnownNodeExecutor::UpdateArgs in.", context.GetNodeName());
  if (davinci_model_->GetTaskList().size() == 0) {
    GELOGW("KnownNodeExecutor::UpdateArgs davinci moel has no taskinfo.");
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
  GELOGI("[%s] KnownNodeExecutor::UpdateArgs success.", context.GetNodeName());
  return SUCCESS;
}

Status KnownNodeTask::Init(TaskContext &context) {
  // allocate output mem
  GE_CHK_STATUS_RET(context.AllocateOutputs(), "known node task allocate output failed.");

  // init davinicmodel
  davinci_model_->InitRuntimeParams();
  GE_CHK_STATUS_RET(davinci_model_->InitVariableMem(), "init variable mem failed.");
  // allocate mem base
  void *buffer = nullptr;
  if (davinci_model_->TotalMemSize() != 0) {
    GE_CHK_STATUS_RET(context.AllocateWorkspace(davinci_model_->TotalMemSize(), &buffer),
                      "known node task allocate workspace failed.");
    // update mem base
    davinci_model_->UpdateMemBase(static_cast<uint8_t *>(buffer));
    GELOGI("KnownNodeTask::Init mem base is %p, size %u.", davinci_model_->GetRuntimeParam().mem_base,
           davinci_model_->GetRuntimeParam().mem_size);
  }
  if (!load_flag_) {
    GE_CHK_STATUS_RET(davinci_model_->Init(), "KnownNodeExecutor::InitDavinciModel failed.");
    load_flag_ = true;
  } else {
    GE_CHK_STATUS_RET(
      ModelManager::GetInstance()->DestroyAicpuKernel(davinci_model_->GetSessionId(), davinci_model_->Id()),
      "KnownNodeTask::Init destroy aicpu kernel failed.");
  }
  GELOGI("[%s] KnownNodeExecutor::Init success.", context.GetNodeName());
  return SUCCESS;
}

Status KnownNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GELOGI("[%s] KnownNodeExecutor::PrepareTask in.", context.GetNodeName());

  GE_CHK_STATUS_RET(task.Init(context), "known node init davinci model failed.");

  GE_CHK_STATUS_RET(task.UpdateArgs(context), "known node task update args failed.");
  GELOGI("[%s] KnownNodeExecutor::PrepareTask success.", context.GetNodeName());
  return SUCCESS;
}

Status KnownNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GELOGI("[%s] KnownNodeExecutor::LoadTask in.", node->GetName().c_str());
  GE_CHECK_NOTNULL(node);

  const GeModelPtr ge_model = model.GetGeModel(node);
  GE_CHECK_NOTNULL(ge_model);

  std::shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(0, nullptr);
  GE_CHECK_NOTNULL(davinci_model);

  // set known node flag as true
  davinci_model->SetKnownNode(true);
  // set model id
  davinci_model->SetId(model.GetModelId());

  GE_CHK_STATUS_RET(davinci_model->Assign(ge_model), "KnownNodeExecutor::LoadTask davincimodel assign failed.");

  task = MakeShared<KnownNodeTask>(davinci_model);
  GE_CHECK_NOTNULL(task);
  GELOGI("[%s] KnownNodeExecutor::LoadTask success.", node->GetName().c_str());
  return SUCCESS;
}

Status KnownNodeExecutor::ExecuteTask(NodeTask &task, TaskContext &context,
                                      const std::function<void()> &callback) const {
  GE_CHK_STATUS_RET(task.ExecuteAsync(context, callback), "Failed to execute task. node = %s",
                    context.GetNodeItem().NodeName().c_str());
  return SUCCESS;
}

}  // namespace hybrid
}  // namespace ge
