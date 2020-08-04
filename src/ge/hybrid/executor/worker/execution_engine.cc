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
#include <sstream>
#include "graph/runtime_inference_context.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
class NodeDoneCallback {
 public:
  NodeDoneCallback(GraphExecutionContext *graph_context, std::shared_ptr<TaskContext> &task_context);
  ~NodeDoneCallback() = default;
  Status OnNodeDone();

 private:
  Status PrepareConstInputs(const NodeItem &node_item);
  GraphExecutionContext *graph_context_;
  std::shared_ptr<TaskContext> context_;
};

NodeDoneCallback::NodeDoneCallback(GraphExecutionContext *graph_context, std::shared_ptr<TaskContext> &task_context)
    : graph_context_(graph_context), context_(task_context) {}

Status NodeDoneCallback::PrepareConstInputs(const NodeItem &node_item) {
  for (auto output_idx : node_item.to_const_output_id_list) {
    RECORD_CALLBACK_EVENT(graph_context_, node_item.NodeName().c_str(), "[PrepareConstInputs] [index = %d] Start",
                          output_idx);

    auto output_tensor = context_->GetOutput(output_idx);
    GE_CHECK_NOTNULL(output_tensor);

    vector<uint8_t> host_buffer(output_tensor->GetSize());
    GELOGD("[%s] To cache output[%d] to host, size = %zu", node_item.NodeName().c_str(), output_idx,
           output_tensor->GetSize());
    GE_CHK_RT_RET(rtMemcpy(host_buffer.data(), host_buffer.size(), output_tensor->GetData(), output_tensor->GetSize(),
                           RT_MEMCPY_HOST_TO_DEVICE));
    Tensor tensor;
    tensor.SetData(host_buffer);
    auto ge_tensor_desc = node_item.op_desc->MutableOutputDesc(output_idx);
    GE_CHECK_NOTNULL(ge_tensor_desc);
    tensor.SetTensorDesc(TensorAdapter::GeTensorDesc2TensorDesc(*ge_tensor_desc));

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
    GE_CHK_STATUS_RET(context_->PropagateOutputs(), "[%s] Failed to propagate outputs failed",
                      node_item.NodeName().c_str());

    RECORD_CALLBACK_EVENT(graph_context_, context_->GetNodeName(), "[PropagateOutputs] End");
  }

  // release
  if (node_item.has_observer) {
    GELOGI("[%s] Notify observer. node_id = %d", node_item.NodeName().c_str(), node_item.node_id);
    graph_context_->cv_manager.NodeDone(node_item.node);
  }

  RECORD_CALLBACK_EVENT(graph_context_, context_->GetNodeName(), "[Callback] End");
  return SUCCESS;
}

ExecutionEngine::ExecutionEngine(GraphExecutionContext *context, CallbackManager *callback_manager)
    : context_(context), callback_manager_(callback_manager) {}

Status ExecutionEngine::Start() {
  GE_CHK_STATUS_RET_NOLOG(ExecutionProcess());
  return SUCCESS;
}

Status ExecutionEngine::ExecutionProcess() {
  GELOGI("ExecutorEngine worker started");
  auto &ready_queue = context_->execution_queue;
  while (true) {
    NodeStatePtr node_state = nullptr;
    if (!ready_queue.Pop(node_state)) {
      GELOGE(FAILED, "Pop task failed");
      return FAILED;
    }

    // EOF
    if (node_state == nullptr) {
      break;
    }

    RECORD_EXECUTION_EVENT(context_, node_state->GetName().c_str(), "Start");
    GELOGI("[%s] Node is ready for execution", node_state->GetName().c_str());
    auto *node_item = node_state->node_item;
    auto task_context = TaskContext::Create(*node_item, context_);
    GE_CHECK_NOTNULL(task_context);
    auto shared_task_context = shared_ptr<TaskContext>(task_context.release());

    auto cb = std::shared_ptr<NodeDoneCallback>(new (std::nothrow) NodeDoneCallback(context_, shared_task_context));
    GE_CHECK_NOTNULL(cb);
    auto callback = [&, cb]() {
      auto ret = cb->OnNodeDone();
      if (ret != SUCCESS) {
        context_->OnError(ret);
      }
    };

    GE_CHK_STATUS_RET_NOLOG(ExecuteAsync(*node_state, *shared_task_context, callback));
    GE_CHK_STATUS_RET_NOLOG(PropagateOutputs(*node_item, *shared_task_context));
  }

  GELOGI("ExecutorEngine worker ended.");
  return SUCCESS;
}

Status ExecutionEngine::ExecuteAsync(NodeState &node_state, TaskContext &task_context,
                                     const std::function<void()> &callback) {
  const auto &task = node_state.kernel_task;
  if (task == nullptr) {
    GELOGE(INTERNAL_ERROR, "[%s] NodeTask is null.", node_state.GetName().c_str());
    return INTERNAL_ERROR;
  }

  RECORD_EXECUTION_EVENT(context_, task_context.GetNodeName(), "[PrepareTask] Start");
  auto executor = node_state.node_item->node_executor;
  GE_CHK_STATUS_RET(executor->PrepareTask(*task, task_context), "[%s] Failed to prepare task",
                    node_state.GetName().c_str());
  RECORD_EXECUTION_EVENT(context_, task_context.GetNodeName(), "[PrepareTask] End");
  GELOGD("[%s] Done task preparation successfully.", node_state.GetName().c_str());

  if (context_->trace_enabled) {
    for (auto i = 0; i < task_context.NumInputs(); ++i) {
      const auto &input_tensor = task_context.GetInput(i);
      GE_CHECK_NOTNULL(input_tensor);
      GELOGD("[%s] Tensor of input[%d] = %s", node_state.GetName().c_str(), i, input_tensor->DebugString().c_str());
    }

    for (auto i = 0; i < task_context.NumOutputs(); ++i) {
      const auto &output_tensor = task_context.GetOutput(i);
      GE_CHECK_NOTNULL(output_tensor);
      GELOGD("[%s] Tensor of output[%d] = %s", node_state.GetName().c_str(), i, output_tensor->DebugString().c_str());
    }
  }

  RECORD_EXECUTION_EVENT(context_, task_context.GetNodeName(), "[ExecuteTask] Start");
  GE_CHK_STATUS_RET(executor->ExecuteTask(*task, task_context, callback), "[%s] Failed to execute task",
                    node_state.GetName().c_str());
  RECORD_EXECUTION_EVENT(context_, task_context.GetNodeName(), "[ExecuteTask] End");

  GELOGD("[%s] Done task launch successfully.", node_state.GetName().c_str());
  return SUCCESS;
}

Status ExecutionEngine::PropagateOutputs(const NodeItem &node_item, TaskContext &task_context) {
  if (node_item.shape_inference_type != DEPEND_COMPUTE) {
    GE_CHK_STATUS_RET(task_context.PropagateOutputs(), "[%s] Failed to propagate outputs.",
                      node_item.NodeName().c_str());
    RECORD_EXECUTION_EVENT(context_, task_context.GetNodeName(), "[PropagateOutputs] End");
  }

  GELOGD("[%s] Done propagating outputs successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
