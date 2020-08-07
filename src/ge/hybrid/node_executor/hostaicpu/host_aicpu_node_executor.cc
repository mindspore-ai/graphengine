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

#include "host_aicpu_engine/ops_kernel_store/op/op_factory.h"
#include "hybrid/node_executor/hostaicpu/host_aicpu_node_executor.h"
#include "graph/passes/folding_pass.h"
#include "hybrid/model/hybrid_model.h"
#include "inc/kernel_factory.h"
#include "ge_local_engine/engine/host_cpu_engine.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HOST_AICPU, HostAiCpuNodeExecutor);

Status HostCpuNodeTaskBase::UpdateArgs(TaskContext &) {
  // no need update args
  return SUCCESS;
}

Status HostCpuNodeTaskBase::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGD("[%s] Start execute.", context.GetNodeName());

  std::vector<ConstGeTensorPtr> inputs;
  std::vector<GeTensorPtr> outputs;
  GE_CHK_STATUS_RET(ProcessInputs(context, inputs), "node:%s type:%s, process inputs failed.", node_->GetName().c_str(),
                    node_->GetType().c_str());
  GE_CHK_STATUS_RET(Execute(context, inputs, outputs), "node:%s type:%s, task execute failed.",
                    node_->GetName().c_str(), node_->GetType().c_str());
  GE_CHK_STATUS_RET(ProcessOutputs(context, outputs), "node:%s type:%s, process outputs failed.",
                    node_->GetName().c_str(), node_->GetType().c_str());

  if (done_callback) {
    GELOGD("[%s] Start invoke callback.", context.GetNodeName());
    done_callback();
  }
  GELOGD("[%s] Done execute successfully.", context.GetNodeName());
  return SUCCESS;
}

Status HostCpuNodeTaskBase::ProcessInputs(TaskContext &context, std::vector<ConstGeTensorPtr> &inputs) {
  // Constant & Variable
  auto tensor = context.GetVariable(node_->GetName());
  if (tensor != nullptr) {
    // Constant & Variable Op has and only has one output
    GeTensorPtr input_ptr = MakeShared<GeTensor>(
      node_->GetOpDesc()->GetOutputDesc(0), reinterpret_cast<const uint8_t *>(tensor->GetData()), tensor->GetSize());
    GE_CHECK_NOTNULL(input_ptr);
    inputs.push_back(input_ptr);
    return SUCCESS;
  }

  int32_t input_num = context.NumInputs();
  for (auto i = 0; i < input_num; ++i) {
    auto tensor_value = context.GetInput(i);
    GE_CHECK_NOTNULL(tensor_value);
    GeTensorPtr input_ptr =
      MakeShared<GeTensor>(node_->GetOpDesc()->GetInputDesc(i),
                           reinterpret_cast<const uint8_t *>(tensor_value->GetData()), tensor_value->GetSize());
    if (input_ptr == nullptr) {
      GELOGE(MEMALLOC_FAILED, "Make shared failed");
      return MEMALLOC_FAILED;
    }
    inputs.push_back(input_ptr);
  }
  return SUCCESS;
}

Status HostCpuNodeTaskBase::ProcessOutputs(TaskContext &context, std::vector<GeTensorPtr> &outputs) {
  int32_t output_num = context.NumOutputs();
  if (static_cast<size_t>(output_num) != outputs.size()) {
    GELOGE(INTERNAL_ERROR, "node %s type %s has %d output, but kernel compute only has %zu output.",
           node_->GetName().c_str(), node_->GetType().c_str(), output_num, outputs.size());
    return INTERNAL_ERROR;
  }

  // alloc output
  GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());

  // copy data to output
  for (auto i = 0; i < output_num; ++i) {
    GeTensorPtr &tensor = outputs[i];
    GE_CHECK_NOTNULL(tensor);
    auto tensor_data = tensor->GetData();
    auto tensor_value = context.MutableOutput(i);
    GE_CHECK_NOTNULL(tensor_value);
    if (tensor_data.GetSize() > tensor_value->GetSize()) {
      GELOGE(INTERNAL_ERROR, "node:%s type:%s [%d]th compute data size=%zu, but context data size=%zu.",
             node_->GetName().c_str(), node_->GetType().c_str(), i, tensor_data.GetSize(), tensor_value->GetSize());
      return INTERNAL_ERROR;
    }

    GELOGI("node:%s type:%s [%d]th output data=%p, out size=%zu, data size=%zu.", node_->GetName().c_str(),
           node_->GetType().c_str(), i, tensor_value->GetData(), tensor_value->GetSize(), tensor_data.GetSize());
    if (tensor_data.GetSize() > 0) {
      GE_CHK_RT_RET(rtMemcpy(tensor_value->MutableData(), tensor_value->GetSize(), tensor_data.GetData(),
                             tensor_data.GetSize(), RT_MEMCPY_HOST_TO_HOST));
    }
    GELOGI("node:%s type:%s [%d]th set data success, data size=%zu.", node_->GetName().c_str(),
           node_->GetType().c_str(), i, tensor_data.GetSize());
  }

  return SUCCESS;
}

Status CpuKernelNodeTask::Execute(TaskContext &context, const std::vector<GeTensorPtr> &inputs,
                                  std::vector<GeTensorPtr> &outputs) {
  std::vector<ConstGeTensorPtr> const_inputs;
  for (const auto &input : inputs) {
    const_inputs.emplace_back(input);
  }
  return FoldingPass::RunOpKernel(node_, const_inputs, outputs);
}

Status HostKernelNodeTask::Execute(TaskContext &context, const std::vector<GeTensorPtr> &inputs,
                                   std::vector<GeTensorPtr> &outputs) {
  auto kernel = KernelFactory::Instance().Create(node_->GetType());
  if (kernel == nullptr) {
    GELOGE(UNSUPPORTED, "node %s type %s is not supported by host kernel.", node_->GetName().c_str(),
           node_->GetType().c_str());
    return UNSUPPORTED;
  }

  std::vector<ConstGeTensorPtr> const_inputs;
  for (const auto &input : inputs) {
    const_inputs.emplace_back(input);
  }
  Status compute_ret = kernel->Compute(node_->GetOpDesc(), const_inputs, outputs);
  if (compute_ret != SUCCESS) {
    GELOGE(compute_ret, "node %s type %s compute failed or not imply.", node_->GetName().c_str(),
           node_->GetType().c_str());
    return compute_ret;
  }

  return SUCCESS;
}

Status HostAiCpuNodeTask::Execute(TaskContext &context, const std::vector<GeTensorPtr> &inputs,
                                  std::vector<GeTensorPtr> &outputs) {
  RunContext run_context;
  auto host_op = host_aicpu::OpFactory::Instance().CreateOp(*node_, run_context);
  if (host_op == nullptr) {
    GELOGE(UNSUPPORTED, "node %s type %s is not supported by host kernel.", node_->GetName().c_str(),
           node_->GetType().c_str());
    return UNSUPPORTED;
  }

  Status compute_ret = host_op->Compute(node_->GetOpDesc(), inputs, outputs);
  if (compute_ret != SUCCESS) {
    GELOGE(compute_ret, "node %s type %s compute failed or not imply.", node_->GetName().c_str(),
           node_->GetType().c_str());
    return compute_ret;
  }

  return SUCCESS;
}

Status HostAiCpuNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  return task.UpdateArgs(context);
}

Status HostAiCpuNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node,
                                       std::shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  const std::string &name = node->GetName();
  const std::string &type = node->GetType();
  if (HostCpuEngine::GetInstance().CheckSupported(type)) {
    GELOGI("create CpuKernelNodeTask for node %s, type %s.", name.c_str(), type.c_str());
    task = MakeShared<CpuKernelNodeTask>(node);
    GE_CHECK_NOTNULL(task);
  } else if (KernelFactory::Instance().Create(type) != nullptr) {
    GELOGI("create HostKernelNodeTask for node %s, type %s.", name.c_str(), type.c_str());
    task = MakeShared<HostKernelNodeTask>(node);
    GE_CHECK_NOTNULL(task);
  } else if (host_aicpu::OpFactory::Instance().CheckSupported(type)) {
    GELOGI("create HostAiCpuNodeTask for node %s, type %s.", name.c_str(), type.c_str());
    task = MakeShared<HostAiCpuNodeTask>(node);
    GE_CHECK_NOTNULL(task);
  } else {
    GELOGE(UNSUPPORTED, "node %s type %s is not support in HostAiCpuNodeExecutor now.", name.c_str(), type.c_str());
    return UNSUPPORTED;
  }
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge