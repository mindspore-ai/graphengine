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

#include "hybrid/executor/subgraph_executor.h"
#include "hybrid/executor/worker/task_compile_engine.h"
#include "hybrid/executor/worker/execution_engine.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int kDefaultThreadNum = 4;
constexpr int kDataInputIndex = 0;
}  // namespace

SubgraphExecutor::SubgraphExecutor(const GraphItem *graph_item, GraphExecutionContext *context, bool force_infer_shape)
    : graph_item_(graph_item),
      context_(context),
      force_infer_shape_(force_infer_shape),
      pre_run_pool_(kDefaultThreadNum) {}

SubgraphExecutor::~SubgraphExecutor() { GELOGD("[%s] SubgraphExecutor destroyed.", graph_item_->GetName().c_str()); }

Status SubgraphExecutor::Init(const std::vector<TensorValue> &inputs,
                              const std::vector<ConstGeTensorDescPtr> &input_desc) {
  subgraph_context_.reset(new (std::nothrow) SubgraphContext(graph_item_));
  GE_CHECK_NOTNULL(subgraph_context_);
  GE_CHK_STATUS_RET(subgraph_context_->Init(), "[%s] Failed to init subgraph context.", graph_item_->GetName().c_str());

  shape_inference_engine_.reset(new (std::nothrow) ShapeInferenceEngine(context_, subgraph_context_.get()));
  GE_CHECK_NOTNULL(shape_inference_engine_);

  if (graph_item_->IsDynamic()) {
    GE_CHK_STATUS_RET(InitInputsForUnknownShape(inputs, input_desc), "[%s] Failed to set inputs.",
                      graph_item_->GetName().c_str());
  } else {
    GE_CHK_STATUS_RET(InitInputsForKnownShape(inputs),
                      "[%s] Failed to init subgraph executor for known shape subgraph.",
                      graph_item_->GetName().c_str());
  }

  return SUCCESS;
}

Status SubgraphExecutor::InitInputsForUnknownShape(const std::vector<TensorValue> &inputs,
                                                   const std::vector<ConstGeTensorDescPtr> &input_desc) {
  // Number of inputs of parent node should be greater or equal than that of subgraph
  auto input_nodes = graph_item_->GetInputNodes();
  if (inputs.size() < input_nodes.size()) {
    GELOGE(INTERNAL_ERROR, "[%s] Number of inputs [%zu] is not sufficient for subgraph which needs [%zu] inputs.",
           graph_item_->GetName().c_str(), inputs.size(), input_nodes.size());
    return INTERNAL_ERROR;
  }

  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto &input_node = input_nodes[i];
    if (input_node == nullptr) {
      GELOGD("[%s] Input[%zu] is not needed by subgraph, skip it.", graph_item_->GetName().c_str(), i);
      continue;
    }

    auto &input_tensor = inputs[i];
    GELOGD("[%s] Set input tensor[%zu] to inputs with index = %d, tensor = %s", graph_item_->GetName().c_str(), i,
           input_node->input_start, input_tensor.DebugString().c_str());

    GE_CHK_STATUS_RET(subgraph_context_->SetInput(*input_node, kDataInputIndex, input_tensor),
                      "[%s] Failed to set input tensor[%zu]", graph_item_->GetName().c_str(), i);

    if (force_infer_shape_ || input_node->is_dynamic) {
      GELOGD("[%s] Start to update input[%zu] for subgraph data node.", graph_item_->GetName().c_str(), i);
      GE_CHECK_LE(i + 1, input_desc.size());
      const auto &tensor_desc = input_desc[i];
      auto node_state = subgraph_context_->GetOrCreateNodeState(input_node);
      GE_CHECK_NOTNULL(node_state);
      node_state->GetShapeInferenceState().UpdateInputShape(0, tensor_desc->GetOriginShape(), tensor_desc->GetShape());
    }
  }

  GELOGD("[%s] Done setting inputs.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::InitInputsForKnownShape(const std::vector<TensorValue> &inputs) {
  auto &input_index_mapping = graph_item_->GetInputIndexMapping();
  for (size_t i = 0; i < input_index_mapping.size(); ++i) {
    auto &parent_input_index = input_index_mapping[i];
    if (static_cast<size_t>(parent_input_index) >= inputs.size()) {
      GELOGE(INTERNAL_ERROR,
             "[%s] Number of inputs [%zu] is not sufficient for subgraph which needs at lease [%d] inputs",
             graph_item_->GetName().c_str(), inputs.size(), parent_input_index + 1);

      return INTERNAL_ERROR;
    }

    auto &input_tensor = inputs[parent_input_index];
    subgraph_context_->SetInput(i, input_tensor);
    GELOGD("[%s] Set input tensor[%zu] with inputs with index = %d, tensor = %s", graph_item_->GetName().c_str(), i,
           parent_input_index, input_tensor.DebugString().c_str());
  }

  return SUCCESS;
}

Status SubgraphExecutor::ExecuteAsync(const std::vector<TensorValue> &inputs,
                                      const std::vector<ConstGeTensorDescPtr> &input_desc) {
  GELOGD("[%s] is dynamic = %s", graph_item_->GetName().c_str(), graph_item_->IsDynamic() ? "true" : "false");
  GE_CHK_STATUS_RET(Init(inputs, input_desc), "[%s] Failed to init executor.", graph_item_->GetName().c_str());

  if (!graph_item_->IsDynamic()) {
    return ExecuteAsyncForKnownShape(inputs);
  }

  GE_CHK_STATUS_RET(ScheduleTasks(), "[%s] Failed to execute tasks.", graph_item_->GetName().c_str());
  GELOGD("[%s] Done executing subgraph successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::ExecuteAsyncForKnownShape(const std::vector<TensorValue> &inputs) {
  GELOGD("[%s] subgraph is not dynamic.", graph_item_->GetName().c_str());
  if (graph_item_->GetAllNodes().size() != 1) {
    GELOGE(INTERNAL_ERROR, "[%s] Invalid known shape subgraph. node size = %zu", graph_item_->GetName().c_str(),
           graph_item_->GetAllNodes().size());
    return INTERNAL_ERROR;
  }

  auto node_item = graph_item_->GetAllNodes()[0];
  GE_CHECK_NOTNULL(node_item);
  auto node_state = subgraph_context_->GetOrCreateNodeState(node_item);
  GE_CHECK_NOTNULL(node_state);
  node_state->SetKernelTask(node_item->kernel_task);

  known_shape_task_context_ = TaskContext::Create(*node_item, context_, subgraph_context_.get());
  GE_CHECK_NOTNULL(known_shape_task_context_);

  GE_CHK_STATUS_RET(ExecutionEngine::ExecuteAsync(*node_state, known_shape_task_context_, *context_),
                    "[%s] Failed to execute node [%s] for known subgraph.", graph_item_->GetName().c_str(),
                    known_shape_task_context_->GetNodeName());

  GELOGD("[%s] Done execute non-dynamic subgraph successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::ExecuteAsync(TaskContext &task_context) {
  std::vector<TensorValue> inputs;
  std::vector<ConstGeTensorDescPtr> input_desc;
  for (int i = 0; i < task_context.NumInputs(); ++i) {
    auto tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(tensor);
    inputs.emplace_back(*tensor);
    input_desc.emplace_back(task_context.GetInputDesc(i));
  }

  GE_CHK_STATUS_RET(ExecuteAsync(inputs, input_desc), "[%s] Failed to execute subgraph.",
                    graph_item_->GetName().c_str());

  GE_CHK_STATUS_RET(SetOutputsToParentNode(task_context), "[%s] Failed to set output shapes to parent node.",
                    graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::PrepareNodes() {
  GELOGD("[%s] Start to prepare nodes. force infer shape = %s.", graph_item_->GetName().c_str(),
         force_infer_shape_ ? "true" : "false");
  auto &all_nodes = graph_item_->GetAllNodes();
  for (size_t i = 0; i < all_nodes.size(); ++i) {
    auto &node_item = *all_nodes[i];
    // for while op
    if (force_infer_shape_ && !node_item.is_dynamic) {
      GELOGD("[%s] Force infer shape is set, updating node to dynamic.", node_item.NodeName().c_str());
      auto &mutable_node_item = const_cast<NodeItem &>(node_item);
      mutable_node_item.SetToDynamic();
    }

    GELOGD("[%s] Start to prepare node [%s].", graph_item_->GetName().c_str(), node_item.NodeName().c_str());
    auto node_state = subgraph_context_->GetOrCreateNodeState(&node_item);
    GE_CHECK_NOTNULL(node_state);
    auto p_node_state = node_state.get();

    if (node_item.node_type == NETOUTPUT) {
      // Wait for all inputs become valid
      // after PrepareNodes returned. all output tensors and shapes are valid
      GE_CHK_STATUS_RET_NOLOG(p_node_state->GetShapeInferenceState().AwaitShapesReady(*context_));
      GE_CHK_STATUS_RET_NOLOG(p_node_state->AwaitInputTensors(*context_));
      continue;
    }

    // only do shape inference and compilation for nodes with dynamic shapes.
    if (node_item.is_dynamic) {
      auto prepare_future = pre_run_pool_.commit([this, p_node_state]() -> Status {
        GE_CHK_STATUS_RET_NOLOG(InferShape(shape_inference_engine_.get(), *p_node_state));
        return PrepareForExecution(context_, *p_node_state);
      });

      p_node_state->SetPrepareFuture(std::move(prepare_future));
    } else {
      GELOGD("[%s] Skipping shape inference and compilation for node with static shape.", node_item.NodeName().c_str());
      if (node_item.kernel_task == nullptr) {
        GELOGW("[%s] Node of static shape got no task.", node_item.NodeName().c_str());
        GE_CHK_STATUS_RET(TaskCompileEngine::Compile(*p_node_state, context_), "[%s] Failed to create task.",
                          p_node_state->GetName().c_str());
      } else {
        node_state->SetKernelTask(node_item.kernel_task);
      }
    }

    if (!ready_queue_.Push(p_node_state)) {
      GELOGE(INTERNAL_ERROR, "[%s] Error occurs while launching tasks. quit from preparing nodes.",
             graph_item_->GetName().c_str());
      return INTERNAL_ERROR;
    }

    GELOGD("[%s] Push node [%s] to queue.", graph_item_->GetName().c_str(), node_item.NodeName().c_str());
  }

  return SUCCESS;
}

Status SubgraphExecutor::InferShape(ShapeInferenceEngine *shape_inference_engine, NodeState &node_state) {
  const auto &node_item = *node_state.GetNodeItem();
  GE_CHK_STATUS_RET(shape_inference_engine->InferShape(node_state), "[%s] Failed to InferShape.",
                    node_state.GetName().c_str());
  GE_CHK_STATUS_RET(shape_inference_engine->PropagateOutputShapes(node_item), "[%s] Failed to PropagateOutputShapes.",
                    node_state.GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::PrepareForExecution(GraphExecutionContext *ctx, NodeState &node_state) {
  auto &node_item = *node_state.GetNodeItem();
  if (node_item.kernel_task == nullptr) {
    GE_CHK_STATUS_RET(TaskCompileEngine::Compile(node_state, ctx), "Failed to create task for node[%s]",
                      node_state.GetName().c_str());
  } else {
    node_state.SetKernelTask(node_item.kernel_task);
  }

  GELOGD("[%s] Start to invoke CalcOpRunningParam.", node_item.NodeName().c_str());
  RECORD_COMPILE_EVENT(ctx, node_item.NodeName().c_str(), "[CalcOpRunningParam] Start");
  GE_CHK_STATUS_RET(NodeExecutorManager::GetInstance().CalcOpRunningParam(*node_item.node),
                    "[%s] Failed to invoke CalcOpRunningParam.", node_item.NodeName().c_str());
  RECORD_COMPILE_EVENT(ctx, node_item.NodeName().c_str(), "[CalcOpRunningParam] End");
  GELOGD("[%s] Done invoking CalcOpRunningParam successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::LaunchTasks() {
  while (true) {
    NodeState *node_state = nullptr;
    if (!ready_queue_.Pop(node_state)) {
      GELOGE(INTERNAL_ERROR, "[%s] Failed to pop node.", graph_item_->GetName().c_str());
      return INTERNAL_ERROR;
    }

    if (node_state == nullptr) {
      GELOGD("[%s] Got EOF from queue.", graph_item_->GetName().c_str());
      return SUCCESS;
    }

    GE_CHK_STATUS_RET_NOLOG(node_state->WaitForPrepareDone());

    GELOGD("[%s] Start to execute.", node_state->GetName().c_str());
    auto task_context = TaskContext::Create(*node_state->GetNodeItem(), context_, subgraph_context_.get());
    GE_CHECK_NOTNULL(task_context);
    task_context->SetForceInferShape(force_infer_shape_);
    auto shared_task_context = std::shared_ptr<TaskContext>(task_context.release());
    GE_CHK_STATUS_RET(ExecutionEngine::ExecuteAsync(*node_state, shared_task_context, *context_),
                      "[%s] Execute node failed.", node_state->GetName().c_str());

    GELOGD("[%s] Done executing node successfully.", node_state->GetName().c_str());
  }
}

Status SubgraphExecutor::ScheduleTasks() {
  GELOGD("[%s] Start to schedule prepare workers.", graph_item_->GetName().c_str());
  auto prepare_future = std::async([&]() -> Status {
    auto ret = PrepareNodes();
    ready_queue_.Push(nullptr);
    return ret;
  });

  GELOGD("[%s] Start to execute subgraph.", graph_item_->GetName().c_str());
  auto ret = LaunchTasks();
  if (ret != SUCCESS) {
    GELOGE(ret, "[%s] Failed to execute subgraph.", graph_item_->GetName().c_str());
    subgraph_context_->OnError(ret);
    ready_queue_.Stop();
    prepare_future.wait();
    return ret;
  }

  GE_CHK_STATUS_RET(prepare_future.get(), "[%s] Error occurred in task preparation.", graph_item_->GetName().c_str());

  GELOGD("[%s] Done launching all tasks successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::GetOutputs(vector<TensorValue> &outputs) { return subgraph_context_->GetOutputs(outputs); }

Status SubgraphExecutor::GetOutputs(vector<TensorValue> &outputs, std::vector<ConstGeTensorDescPtr> &output_desc) {
  GE_CHK_STATUS_RET(GetOutputs(outputs), "[%s] Failed to get output tensors.", graph_item_->GetName().c_str());

  // copy output data from op to designated position
  std::vector<GeTensorDescPtr> output_tensor_desc_list;
  GE_CHK_STATUS_RET(graph_item_->GetOutputDescList(output_desc), "[%s] Failed to get output tensor desc.",
                    graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::Synchronize() {
  GELOGD("[%s] Synchronize start.", graph_item_->GetName().c_str());
  GE_CHK_RT_RET(rtStreamSynchronize(context_->stream));
  GELOGD("[%s] Done synchronizing successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::SetOutputsToParentNode(TaskContext &task_context) {
  // get output tensors and tensor desc list
  std::vector<TensorValue> outputs;
  std::vector<ConstGeTensorDescPtr> output_desc_list;
  GE_CHK_STATUS_RET(subgraph_context_->GetOutputs(outputs), "[%s] Failed to get output tensors.",
                    graph_item_->GetName().c_str());
  GE_CHK_STATUS_RET(graph_item_->GetOutputDescList(output_desc_list), "[%s] Failed to get output tensor desc.",
                    graph_item_->GetName().c_str());

  if (outputs.size() != output_desc_list.size()) {
    GELOGE(INTERNAL_ERROR, "[%s] num output tensors = %zu, num output tensor desc = %zu",
           graph_item_->GetName().c_str(), outputs.size(), output_desc_list.size());
    return INTERNAL_ERROR;
  }

  // mapping to parent task context
  for (size_t i = 0; i < outputs.size(); ++i) {
    int parent_output_index = graph_item_->GetParentOutputIndex(i);
    GE_CHECK_GE(parent_output_index, 0);
    // update tensor
    GELOGD("[%s] Updating output[%zu] to parent output[%d]", graph_item_->GetName().c_str(), i, parent_output_index);

    GELOGD("[%s] Updating output tensor, index = %d, tensor = %s", graph_item_->GetName().c_str(), parent_output_index,
           outputs[i].DebugString().c_str());
    GE_CHK_STATUS_RET(task_context.SetOutput(parent_output_index, outputs[i]));

    // updating shapes. dynamic format/dtype is not supported.
    // It should be noted that even the subgraph is of known shape, it is also necessary to update parent output desc,
    // for instance, IfOp may have two known-shaped subgraphs of different output shapes
    const auto &output_desc = output_desc_list[i];
    auto parent_output_desc = task_context.MutableOutputDesc(parent_output_index);
    GE_CHECK_NOTNULL(parent_output_desc);
    GELOGD("[%s] Updating output shape[%d] from [%s] to [%s]", graph_item_->GetName().c_str(), parent_output_index,
           parent_output_desc->MutableShape().ToString().c_str(), output_desc->GetShape().ToString().c_str());
    parent_output_desc->SetShape(output_desc->GetShape());

    GELOGD("[%s] Updating output original shape[%d] from [%s] to [%s]", graph_item_->GetName().c_str(),
           parent_output_index, parent_output_desc->GetOriginShape().ToString().c_str(),
           output_desc->GetOriginShape().ToString().c_str());
    parent_output_desc->SetOriginShape(output_desc->GetOriginShape());
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
