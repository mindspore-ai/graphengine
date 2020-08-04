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

#include "task_context.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/log.h"
#include "graph/utils/tensor_utils.h"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
TaskContext::TaskContext(GraphExecutionContext *execution_context) : execution_context_(execution_context) {}
TaskContext::~TaskContext() {
  GELOGD("To execute ~TaskContext(). node = %s", node_item_->NodeName().c_str());
  for (auto ws_addr : workspaces_) {
    execution_context_->allocator->Deallocate(ws_addr);
  }

  // release output
  for (int i = 0; i < NumOutputs(); ++i) {
    auto output_tensor = MutableOutput(i);
    if (output_tensor != nullptr) {
      output_tensor->Destroy();
    }
  }
}

std::unique_ptr<TaskContext> TaskContext::Create(const NodeItem &node_item, GraphExecutionContext *graph_context) {
  GELOGI("To create task context for node %s, input start = %d, num_inputs = %d, output start = %d, num_outputs = %d",
         node_item.NodeName().c_str(), node_item.input_start, node_item.num_inputs, node_item.output_start,
         node_item.num_outputs);
  auto task_context = std::unique_ptr<TaskContext>(new (std::nothrow) TaskContext(graph_context));
  if (task_context == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to create instance of TaskContext. node = %s", node_item.NodeName().c_str());
    return nullptr;
  }

  task_context->node_item_ = &node_item;
  task_context->inputs_start_ = graph_context->all_inputs.data() + node_item.input_start;
  task_context->outputs_start_ = graph_context->all_outputs.data() + node_item.output_start;
  return task_context;
}

int TaskContext::NumInputs() const { return node_item_->num_inputs; }

int TaskContext::NumOutputs() const { return node_item_->num_outputs; }

TensorValue *TaskContext::MutableInput(int index) {
  if (index < 0 || index > node_item_->num_inputs) {
    GELOGE(PARAM_INVALID, "Index out of range. index = %d, num_inputs = %d", index, node_item_->num_inputs);
    return nullptr;
  }

  return inputs_start_ + index;
}

const TensorValue *TaskContext::GetOutput(int index) const {
  if (index < 0 || index > node_item_->num_outputs) {
    GELOGE(PARAM_INVALID, "Index out of range. index = %d, num_outputs = %d", index, node_item_->num_outputs);
    return nullptr;
  }

  return outputs_start_ + index;
}

TensorValue *TaskContext::MutableOutput(int index) {
  if (index < 0 || index > node_item_->num_outputs) {
    GELOGE(PARAM_INVALID, "Index out of range. index = %d, num_outputs = %d", index, node_item_->num_outputs);
    return nullptr;
  }

  return outputs_start_ + index;
}

std::size_t TaskContext::NumWorkspaces() const { return workspaces_.size(); }

void *TaskContext::MutableWorkspace(int index) {
  if (index < 0 || static_cast<size_t>(index) >= workspaces_.size()) {
    GELOGE(PARAM_INVALID, "Index out of range. index = %d, num_workspaces = %d", index, node_item_->num_outputs);
    return nullptr;
  }

  return workspaces_[index];
}

const TensorValue *TaskContext::GetInput(int index) const {
  if (index < 0 || index > node_item_->num_inputs) {
    GELOGE(PARAM_INVALID, "Index out of range. index = %d, num_inputs = %d", index, node_item_->num_inputs);
    return nullptr;
  }

  return inputs_start_ + index;
}

Status TaskContext::AllocateWorkspaces() {
  auto workspace_sizes = node_item_->node->GetOpDesc()->GetWorkspaceBytes();
  for (auto size : workspace_sizes) {
    void *workspace = execution_context_->allocator->Allocate(size);
    if (workspace == nullptr) {
      GELOGE(MEMALLOC_FAILED, "Failed to allocate workspace of size: %ld", size);
      return MEMALLOC_FAILED;
    }

    workspaces_.emplace_back(workspace);
  }
  return SUCCESS;
}

Status TaskContext::RegisterCallback(const std::function<void()> &callback_fun) const {
  return execution_context_->callback_manager->RegisterCallback(callback_fun);
}

string TaskContext::TensorDesc2String(const GeTensorDesc &desc) {
  std::stringstream ss;
  ss << "[TensorDesc] ";
  ss << "DataType = " << desc.GetDataType();
  ss << ", Format = " << desc.GetFormat();
  ss << ", Shape = [";
  for (auto dim : desc.GetShape().GetDims()) {
    ss << dim << ", ";
  }
  ss << "]";

  return ss.str();
}

Status TaskContext::AllocateTensor(const GeTensorDesc &tensor_desc, TensorValue &tensor) {
  int64_t size = 0;
  if (ge::TensorUtils::GetSize(tensor_desc, size) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to get tensor size");
    return INTERNAL_ERROR;
  }

  if (size == 0) {
    GELOGW("size from tensor_desc == 0");
  }

  auto buffer = TensorBuffer::Create(execution_context_->allocator, size);
  GE_CHECK_NOTNULL(buffer);
  tensor = TensorValue(shared_ptr<TensorBuffer>(buffer.release()));
  return SUCCESS;
}

Status TaskContext::AllocateOutput(int index, const GeTensorDesc &tensor_desc, TensorValue **tensor) {
  GELOGI("To allocate output for node: %s. index = %d, tensor desc = %s", node_item_->NodeName().c_str(), index,
         TensorDesc2String(tensor_desc).c_str());

  if (index < 0 || index >= node_item_->num_outputs) {
    GELOGE(PARAM_INVALID, "output index out of range. num_output = %d, index = %d", node_item_->num_outputs, index);
    return PARAM_INVALID;
  }

  if (outputs_start_[index].GetData() != nullptr) {
    GELOGI("already allocated as net output");
    return SUCCESS;
  }

  auto it = node_item_->ref_outputs.find(index);
  if (it != node_item_->ref_outputs.end()) {
    auto &ref_node = it->second;
    GELOGD("source node of %s:%d = %s, op_type = %s", node_item_->NodeName().c_str(), index,
           ref_node->GetName().c_str(), ref_node->GetType().c_str());

    TensorValue *ref_tensor = execution_context_->model->GetVariable(ref_node->GetName());
    GE_CHECK_NOTNULL(ref_tensor);
    outputs_start_[index] = *ref_tensor;
  } else {
    GE_CHK_STATUS_RET_NOLOG(AllocateTensor(tensor_desc, outputs_start_[index]));
    GELOGD("Allocating output successfully. node: %s. index = %d, size = %zu", node_item_->NodeName().c_str(), index,
           outputs_start_[index].GetSize());
  }

  if (execution_context_->trace_enabled) {
    outputs_start_[index].SetName(node_item_->NodeName() + "_out_" + std::to_string(index));
  }

  if (tensor != nullptr) {
    *tensor = outputs_start_ + index;
  }

  return SUCCESS;
}

Status TaskContext::AllocateOutputs() {
  for (int i = 0; i < node_item_->num_outputs; ++i) {
    const auto &output_desc = node_item_->op_desc->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    GE_CHK_STATUS_RET_NOLOG(AllocateOutput(i, *output_desc, nullptr));
  }

  return SUCCESS;
}

Status TaskContext::AllocateTemp(size_t size, TensorValue &tensor) {
  auto buffer = TensorBuffer::Create(execution_context_->allocator, size);
  if (buffer == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to allocate buffer of size: %zu", size);
    return MEMALLOC_FAILED;
  }

  tensor = TensorValue(shared_ptr<TensorBuffer>(buffer.release()));
  return SUCCESS;
}

const NodeItem &TaskContext::GetNodeItem() const { return *node_item_; }

Status TaskContext::SetOutput(int index, const TensorValue &tensor) {
  if (index < 0 || index >= node_item_->num_outputs) {
    GELOGE(PARAM_INVALID, "output index out of range. num_output = %d, index = %d", node_item_->num_outputs, index);
    return PARAM_INVALID;
  }

  GELOGD("Set %s:%d with tensor: %s", node_item_->NodeName().c_str(), index, tensor.DebugString().c_str());
  outputs_start_[index] = tensor;
  return SUCCESS;
}

rtStream_t TaskContext::GetStream() { return execution_context_->stream; }

int64_t TaskContext::GetSessionId() { return execution_context_->session_id; }

Status TaskContext::GetStatus() const { return status_; }

void TaskContext::SetStatus(Status status) { status_ = status; }

Status TaskContext::AllocateWorkspace(size_t size, void **buffer, void *ori_addr) {
  GE_CHECK_NOTNULL(buffer);
  *buffer = execution_context_->allocator->Allocate(size, ori_addr);
  if (*buffer == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to allocate workspace of size = %zu", size);
    return MEMALLOC_FAILED;
  }

  GELOGD("Allocating workspace of size = %zu successfully", size);
  workspaces_.emplace_back(*buffer);
  return SUCCESS;
}

Status TaskContext::PropagateOutputs() {
  // propagate outputs
  for (int i = 0; i < NumOutputs(); ++i) {
    auto tensor = MutableOutput(i);
    GE_CHECK_NOTNULL(tensor);
    if (tensor->GetData() == nullptr) {
      GELOGD("[%s] Node output[%d] is null.", node_item_->NodeName().c_str(), i);
    }
    auto &output_nodes = node_item_->outputs[i];
    for (auto &dst_input_index_and_node : output_nodes) {
      auto dst_input_idx = dst_input_index_and_node.first;
      auto dst_node_item = dst_input_index_and_node.second;
      GELOGI(
        "Propagate output of node %s, output index = %d, dst node = %s, "
        "dst_input_index = %d, dst_input_offset = %d, addr = %p",
        node_item_->NodeName().c_str(), i, dst_node_item->NodeName().c_str(), dst_input_idx,
        dst_node_item->input_start + dst_input_idx,
        execution_context_->all_inputs.data() + dst_node_item->input_start + dst_input_idx);
      execution_context_->all_inputs[dst_node_item->input_start + dst_input_idx] = *tensor;
      if (execution_context_->trace_enabled) {
        execution_context_->all_inputs[dst_node_item->input_start + dst_input_idx].SetName(node_item_->NodeName() +
                                                                                           "_in_" + std::to_string(i));
      }
    }
  }

  return SUCCESS;
}

const void *TaskContext::GetVarBaseAddr() { return execution_context_->model->GetVarMemBase(); }

const char *TaskContext::GetNodeName() const { return node_item_->NodeName().c_str(); }

void TaskContext::ReleaseInput(int index) {
  auto input_tensor = MutableInput(index);
  if (input_tensor != nullptr) {
    input_tensor->Destroy();
    GELOGD("[%s] Tensor of input[%d] released", GetNodeName(), index);
  }
}
}  // namespace hybrid
}  // namespace ge
