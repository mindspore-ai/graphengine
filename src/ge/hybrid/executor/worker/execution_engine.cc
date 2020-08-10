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

#include "hybrid/executor/worker/execution_engine.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int64_t kMaxPadding = 63;

Status LogInputs(const NodeItem &node_item, const TaskContext &task_context) {
  for (auto i = 0; i < task_context.NumInputs(); ++i) {
    const auto &input_tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input_tensor);
    const auto &tensor_desc = node_item.op_desc->MutableInputDesc(i);
    GE_CHECK_NOTNULL(tensor_desc);
    GELOGD("[%s] Print task args. input[%d] = %s, shape = [%s]", node_item.NodeName().c_str(), i,
           input_tensor->DebugString().c_str(), tensor_desc->MutableShape().ToString().c_str());
  }

  return SUCCESS;
}

Status LogOutputs(const NodeItem &node_item, const TaskContext &task_context) {
  for (auto i = 0; i < task_context.NumOutputs(); ++i) {
    const auto &output_tensor = task_context.GetOutput(i);
    GE_CHECK_NOTNULL(output_tensor);
    const auto &tensor_desc = node_item.op_desc->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(tensor_desc);
    GELOGD("[%s] Print task args. output[%d] = %s, shape = [%s]", node_item.NodeName().c_str(), i,
           output_tensor->DebugString().c_str(), tensor_desc->MutableShape().ToString().c_str());
  }

  return SUCCESS;
}
}  // namespace
class NodeDoneCallback {
 public:
  NodeDoneCallback(GraphExecutionContext *graph_context, std::shared_ptr<TaskContext> task_context);
  ~NodeDoneCallback() = default;
  Status OnNodeDone();

 private:
  Status PrepareConstInputs(const NodeItem &node_item);
  GraphExecutionContext *graph_context_;
  std::shared_ptr<TaskContext> context_;
};

NodeDoneCallback::NodeDoneCallback(GraphExecutionContext *graph_context, std::shared_ptr<TaskContext> task_context)
    : graph_context_(graph_context), context_(std::move(task_context)) {}

Status NodeDoneCallback::PrepareConstInputs(const NodeItem &node_item) {
  for (auto output_idx : node_item.to_const_output_id_list) {
    RECORD_CALLBACK_EVENT(graph_context_, node_item.NodeName().c_str(), "[PrepareConstInputs] [index = %d] Start",
                          output_idx);

    auto output_tensor = context_->GetOutput(output_idx);
    GE_CHECK_NOTNULL(output_tensor);

    Tensor tensor;
    auto ge_tensor_desc = node_item.op_desc->MutableOutputDesc(output_idx);
    GE_CHECK_NOTNULL(ge_tensor_desc);
    tensor.SetTensorDesc(TensorAdapter::GeTensorDesc2TensorDesc(*ge_tensor_desc));

    int64_t tensor_size;
    GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*ge_tensor_desc, tensor_size),
                            "Failed to invoke GetTensorSizeInBytes");

    if (output_tensor->GetSize() < static_cast<size_t>(tensor_size)) {
      GELOGE(INTERNAL_ERROR, "[%s] Tensor size is not enough. output index = %d, required size = %zu, tensor = %s",
             node_item.NodeName().c_str(), output_idx, tensor_size, output_tensor->DebugString().c_str());
      return INTERNAL_ERROR;
    }

    vector<uint8_t> host_buffer(tensor_size);
    GELOGD("[%s] To cache output[%d] to host, size = %zu", node_item.NodeName().c_str(), output_idx,
           output_tensor->GetSize());
    GE_CHK_RT_RET(
      rtMemcpy(host_buffer.data(), tensor_size, output_tensor->GetData(), tensor_size, RT_MEMCPY_DEVICE_TO_HOST));
    tensor.SetData(host_buffer);

    string session_id = std::to_string(context_->GetSessionId());
    RuntimeInferenceContext *runtime_infer_ctx = nullptr;
    GE_CHK_GRAPH_STATUS_RET(RuntimeInferenceContext::GetContext(session_id, &runtime_infer_ctx),
                            "Failed to get RuntimeInferenceContext, session_id = %s", session_id.c_str());
    GE_CHK_STATUS_RET(runtime_infer_ctx->SetTensor(node_item.node_id, output_idx, std::move(tensor)),
                      "Failed to SetTensor, node = %s, output_index = %d", node_item.NodeName().c_str(), output_idx);
    GELOGD("[%s] Output[%d] cached successfully in session: %s. node_id = %d, shape = [%s]",
           node_item.NodeName().c_str(), output_idx, session_id.c_str(), node_item.node_id,
           ge_tensor_desc->GetShape().ToString().c_str());

    RECORD_CALLBACK_EVENT(graph_context_, node_item.NodeName().c_str(), "[PrepareConstInputs] [index = %d] End",
                          output_idx);
  }

  return SUCCESS;
}

Status NodeDoneCallback::OnNodeDone() {
  auto &node_item = context_->GetNodeItem();
  GELOGI("[%s] Start callback process.", node_item.NodeName().c_str());
  RECORD_CALLBACK_EVENT(graph_context_, context_->GetNodeName(), "Start");

  // release inputs
  for (int i = 0; i < context_->NumInputs(); ++i) {
    context_->ReleaseInput(i);
  }

  GE_CHK_STATUS_RET_NOLOG(PrepareConstInputs(node_item));
  // PropagateOutputs for type == DEPEND_COMPUTE
  if (node_item.shape_inference_type == DEPEND_COMPUTE) {
    if (graph_context_->trace_enabled) {
      (void)LogOutputs(node_item, *context_);
    }

    GE_CHK_STATUS_RET(context_->PropagateOutputs(), "[%s] Failed to propagate outputs failed",
                      node_item.NodeName().c_str());

    RECORD_CALLBACK_EVENT(graph_context_, context_->GetNodeName(), "[PropagateOutputs] End");
  }

  // release condition variable
  if (node_item.has_observer) {
    GELOGI("[%s] Notify observer. node_id = %d", node_item.NodeName().c_str(), node_item.node_id);
    context_->NodeDone();
  }

  RECORD_CALLBACK_EVENT(graph_context_, context_->GetNodeName(), "[Callback] End");
  return SUCCESS;
}

Status ExecutionEngine::ExecuteAsync(NodeState &node_state, const std::shared_ptr<TaskContext> &task_context,
                                     GraphExecutionContext &execution_context) {
  GELOGI("[%s] Node is ready for execution", task_context->GetNodeName());
  RECORD_EXECUTION_EVENT(&execution_context, task_context->GetNodeName(), "Start");
  auto cb = std::shared_ptr<NodeDoneCallback>(new (std::nothrow) NodeDoneCallback(&execution_context, task_context));
  GE_CHECK_NOTNULL(cb);
  auto callback = [&, cb]() {
    auto ret = cb->OnNodeDone();
    if (ret != SUCCESS) {
      task_context->OnError(ret);
    }
  };

  GE_CHK_STATUS_RET_NOLOG(DoExecuteAsync(node_state, *task_context, execution_context, callback));
  GE_CHK_STATUS_RET_NOLOG(PropagateOutputs(*node_state.GetNodeItem(), *task_context, execution_context));
  return SUCCESS;
}

Status ExecutionEngine::DoExecuteAsync(NodeState &node_state, TaskContext &task_context, GraphExecutionContext &context,
                                       const std::function<void()> &callback) {
  const auto &task = node_state.GetKernelTask();
  if (task == nullptr) {
    GELOGE(INTERNAL_ERROR, "[%s] NodeTask is null.", node_state.GetName().c_str());
    return INTERNAL_ERROR;
  }

  // Wait for dependent nodes(DEPEND_COMPUTE), so that the input tensors are valid.
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[AwaitDependents] Start");
  GE_CHK_STATUS_RET(node_state.AwaitInputTensors(context), "[%s] Failed to wait for dependent nodes.",
                    node_state.GetName().c_str());

  const auto &node_item = *node_state.GetNodeItem();
  auto executor = node_item.node_executor;
  GE_CHECK_NOTNULL(executor);
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[PrepareTask] Start");
  GE_CHK_STATUS_RET(executor->PrepareTask(*task, task_context), "[%s] Failed to prepare task",
                    node_state.GetName().c_str());
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[PrepareTask] End");
  GELOGD("[%s] Done task preparation successfully.", node_state.GetName().c_str());

  if (context.trace_enabled) {
    LogInputs(node_item, task_context);
    if (node_item.shape_inference_type != DEPEND_COMPUTE) {
      LogOutputs(node_item, task_context);
    }
  }

  GE_CHK_STATUS_RET(ValidateInputTensors(node_state, task_context), "Failed to validate input tensors.");
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[ValidateInputTensors] End");

  GE_CHK_STATUS_RET(executor->ExecuteTask(*task, task_context, callback), "[%s] Failed to execute task",
                    node_state.GetName().c_str());
  RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[ExecuteTask] End");

  GELOGD("[%s] Done task launch successfully.", node_state.GetName().c_str());
  return SUCCESS;
}

Status ExecutionEngine::ValidateInputTensors(const NodeState &node_state, const TaskContext &task_context) {
  for (auto i = 0; i < task_context.NumInputs(); ++i) {
    const auto &input_tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input_tensor);
    const auto &tensor_desc = node_state.GetOpDesc()->MutableInputDesc(i);
    GE_CHECK_NOTNULL(tensor_desc);
    int64_t expected_size;
    GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorMemorySizeInBytes(*tensor_desc, expected_size));
    GELOGD("[%s] Input[%d] expects [%ld] bytes.", task_context.GetNodeName(), i, expected_size);
    auto size_diff = expected_size - static_cast<int64_t>(input_tensor->GetSize());
    if (size_diff > 0) {
      if (size_diff <= kMaxPadding) {
        GELOGW("[%s] Input[%d]: tensor size mismatches. expected: %ld, but given %zu", task_context.GetNodeName(), i,
               expected_size, input_tensor->GetSize());
      } else {
        GELOGE(INTERNAL_ERROR, "[%s] Input[%d]: tensor size mismatches. expected: %ld, but given %zu",
               task_context.GetNodeName(), i, expected_size, input_tensor->GetSize());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status ExecutionEngine::PropagateOutputs(const NodeItem &node_item, TaskContext &task_context,
                                         GraphExecutionContext &context) {
  if (node_item.shape_inference_type != DEPEND_COMPUTE) {
    GE_CHK_STATUS_RET(task_context.PropagateOutputs(), "[%s] Failed to propagate outputs.",
                      node_item.NodeName().c_str());
    RECORD_EXECUTION_EVENT(&context, task_context.GetNodeName(), "[PropagateOutputs] End");
    GELOGD("[%s] Done propagating outputs successfully.", node_item.NodeName().c_str());
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
