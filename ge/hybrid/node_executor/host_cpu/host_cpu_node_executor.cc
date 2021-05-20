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

#include "hybrid/node_executor/host_cpu/host_cpu_node_executor.h"
#include "hybrid/node_executor/host_cpu/kernel_factory.h"
#include "graph/passes/folding_pass.h"
#include "hybrid/model/hybrid_model.h"
#include "graph/manager/graph_mem_manager.h"
#include "ge_local_engine/engine/host_cpu_engine.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HOST_CPU, HostCpuNodeExecutor);

Status HostNodeTaskBase::UpdateArgs(TaskContext &) {
  // no need update args
  return SUCCESS;
}

Status HostNodeTaskBase::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGD("[%s] Start execute.", context.GetNodeName());
  GE_CHK_STATUS_RET(Execute(context), "[Invoke][Execute] failed for node:%s type:%s.",
                    node_->GetName().c_str(), node_->GetType().c_str())
  if (done_callback) {
    GELOGD("[%s] Start invoke callback.", context.GetNodeName());
    done_callback();
  }
  GELOGD("[%s] Done execute successfully.", context.GetNodeName());
  return SUCCESS;
}

Status CpuKernelNodeTask::Execute(TaskContext &context) {
  const auto &op_desc = node_->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  std::vector<ConstGeTensorPtr> inputs;
  for (int32_t i = 0; i < context.NumInputs(); ++i) {
    auto input_desc_ptr = context.GetInputDesc(i);
    GE_CHECK_NOTNULL(input_desc_ptr);
    const auto &input_desc = *input_desc_ptr;
    auto tensor = context.GetInput(i);
    GE_CHECK_NOTNULL(tensor);
    auto item = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).GetAlignedPtr(tensor->GetData());
    GE_CHECK_NOTNULL(item.second);
    auto in_tensor = MakeShared<GeTensor>(input_desc, item.second, item.first);
    GE_CHECK_NOTNULL(in_tensor);
    in_tensor->MutableTensorDesc().SetDataType(input_desc.GetDataType());
    in_tensor->MutableTensorDesc().SetShape(input_desc.GetShape());
    inputs.emplace_back(in_tensor);
    GELOGD("node:%s allocate input %d, size=%zu", op_desc->GetName().c_str(), i, in_tensor->GetData().size());
  }

  std::vector<GeTensorPtr> outputs;
  for (int32_t i = 0; i < context.NumOutputs(); ++i) {
    const auto &output_desc = op_desc->GetOutputDesc(i);
    AllocationAttr attr;
    attr.SetMemType(HOST_DDR);
    if (context.AllocateOutput(i, output_desc, nullptr, &attr) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "node:%s Failed to allocate output %d", context.GetNodeName(), i);
      GELOGE(FAILED, "[Invoke][AllocateOutput]node:%s Failed to allocate output %d", context.GetNodeName(), i);
      return FAILED;
    }
    auto tensor = context.GetOutput(i);
    GE_CHECK_NOTNULL(tensor);
    auto item = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).GetAlignedPtr(tensor->GetData());
    GE_CHECK_NOTNULL(item.second);
    auto out_tensor = MakeShared<GeTensor>(output_desc, item.second, item.first);
    GE_CHECK_NOTNULL(out_tensor);
    out_tensor->MutableTensorDesc().SetDataType(output_desc.GetDataType());
    out_tensor->MutableTensorDesc().SetShape(output_desc.GetShape());
    outputs.emplace_back(out_tensor);
    GELOGD("node:%s allocate output %d, size=%zu", op_desc->GetName().c_str(), i, out_tensor->GetData().size());
  }

  return HostCpuEngine::GetInstance().Run(node_, inputs, outputs);
}

Status HostCpuNodeTask::Execute(TaskContext &context) {
  RunContext run_context;
  auto host_kernel = hybrid::host_cpu::KernelFactory::Instance().CreateKernel(node_);
  if (host_kernel == nullptr) {
    REPORT_CALL_ERROR("E19999", "CreateKernel failed for node %s type %s is not supported by host kernel.",
                      node_->GetName().c_str(), node_->GetType().c_str());
    GELOGE(UNSUPPORTED, "[Create][Kernel]node %s type %s is not supported by host kernel.",
           node_->GetName().c_str(), node_->GetType().c_str());
    return UNSUPPORTED;
  }

  Status compute_ret = host_kernel->Compute(context);
  if (compute_ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "node %s type %s compute failed.",
                      node_->GetName().c_str(), node_->GetType().c_str());
    GELOGE(compute_ret, "[Invoke][Compute]node %s type %s compute failed or not imply.",
           node_->GetName().c_str(), node_->GetType().c_str());
    return compute_ret;
  }

  return SUCCESS;
}

Status HostCpuNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  return task.UpdateArgs(context);
}

Status HostCpuNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node,
                                     std::shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto mem_type = static_cast<uint32_t>(HOST_DDR);
  for (size_t i = 0; i < op_desc->GetOutputsSize(); i++) {
    (void)AttrUtils::SetInt(op_desc->MutableOutputDesc(i), ATTR_OUTPUT_MEMORY_TYPE, mem_type);
  }
  const std::string &name = node->GetName();
  const std::string &type = node->GetType();
  if (HostCpuEngine::GetInstance().CheckSupported(type)) {
    GELOGI("create CpuKernelNodeTask for node %s, type %s.", name.c_str(), type.c_str());
    task = MakeShared<CpuKernelNodeTask>(node);
    GE_CHECK_NOTNULL(task);
  } else if (hybrid::host_cpu::KernelFactory::Instance().CreateKernel(node) != nullptr) {
    GELOGI("create HostCpuNodeTask for node %s, type %s.", name.c_str(), type.c_str());
    task = MakeShared<HostCpuNodeTask>(node);
    GE_CHECK_NOTNULL(task);
  } else {
    REPORT_INNER_ERROR("E19999", "Create NodeTask failed for node %s type %s.",
                       name.c_str(), type.c_str());
    GELOGE(UNSUPPORTED, "[Create][NodeTask]node %s type %s is not support in HostCpuNodeExecutor now.",
           name.c_str(), type.c_str());
    return UNSUPPORTED;
  }
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
