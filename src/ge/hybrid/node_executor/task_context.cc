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
#include "graph/debug/ge_attr_define.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/subgraph_executor.h"

namespace ge {
namespace hybrid {
TaskContext::TaskContext(GraphExecutionContext *execution_context, const NodeItem *node_item,
                         SubgraphContext *subgraph_context)
    : node_item_(node_item), execution_context_(execution_context), subgraph_context_(subgraph_context) {}

TaskContext::~TaskContext() {
  GELOGD("[%s] TaskContext destroyed.", node_item_->NodeName().c_str());
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

std::unique_ptr<TaskContext> TaskContext::Create(const NodeItem &node_item, GraphExecutionContext *execution_context,
                                                 SubgraphContext *subgraph_context) {
  GELOGI("[%s] To create task context, input start = %d, num_inputs = %d, output start = %d, num_outputs = %d.",
         node_item.NodeName().c_str(), node_item.input_start, node_item.num_inputs, node_item.output_start,
         node_item.num_outputs);
  if (node_item.input_start < 0 || node_item.output_start < 0) {
    GELOGE(INTERNAL_ERROR, "NodeItem not property initialized. input_start = %d, output_start = %d",
           node_item.input_start, node_item.output_start);
    return nullptr;
  }

  auto task_context =
    std::unique_ptr<TaskContext>(new (std::nothrow) TaskContext(execution_context, &node_item, subgraph_context));
  if (task_context == nullptr) {
    GELOGE(MEMALLOC_FAILED, "[%s] Failed to create instance of TaskContext.", node_item.NodeName().c_str());
    return nullptr;
  }

  task_context->node_item_ = &node_item;
  task_context->inputs_start_ = subgraph_context->all_inputs_.data() + node_item.input_start;
  task_context->outputs_start_ = subgraph_context->all_outputs_.data() + node_item.output_start;
  task_context->iteration_ = execution_context->iteration;
  return task_context;
}

int TaskContext::NumInputs() const { return node_item_->num_inputs; }

int TaskContext::NumOutputs() const { return node_item_->num_outputs; }

TensorValue *TaskContext::MutableInput(int index) {
  if (index < 0 || index >= node_item_->num_inputs) {
    GELOGE(PARAM_INVALID, "Index out of range. index = %d, num_inputs = %d", index, node_item_->num_inputs);
    return nullptr;
  }

  return inputs_start_ + index;
}

const TensorValue *TaskContext::GetOutput(int index) const {
  if (index < 0 || index >= node_item_->num_outputs) {
    GELOGE(PARAM_INVALID, "Index out of range. index = %d, num_outputs = %d", index, node_item_->num_outputs);
    return nullptr;
  }

  return outputs_start_ + index;
}

TensorValue *TaskContext::MutableOutput(int index) {
  if (index < 0 || index >= node_item_->num_outputs) {
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
  if (index < 0 || index >= node_item_->num_inputs) {
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
  auto ret = execution_context_->callback_manager->RegisterCallback(callback_fun);
  if (ret != SUCCESS) {
    GELOGE(ret, "[%s] Failed to register callback", GetNodeName());
    execution_context_->callback_manager->Destroy();
    return ret;
  }

  return SUCCESS;
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

Status TaskContext::AllocateTensor(const GeTensorDesc &tensor_desc, TensorValue &tensor, AllocationAttr *attr) {
  int64_t size = 0;
  if (ge::TensorUtils::GetSize(tensor_desc, size) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to get tensor size");
    return INTERNAL_ERROR;
  }

  if (size == 0) {
    GELOGW("size from tensor_desc == 0");
  }

  auto buffer = TensorBuffer::Create(execution_context_->allocator, size, attr);
  GE_CHECK_NOTNULL(buffer);
  tensor = TensorValue(shared_ptr<TensorBuffer>(buffer.release()));
  return SUCCESS;
}

Status TaskContext::AllocateOutput(int index, const GeTensorDesc &tensor_desc, TensorValue **tensor,
                                   AllocationAttr *attr) {
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
    auto reuse_input = node_item_->reuse_inputs.find(index);
    if (reuse_input != node_item_->reuse_inputs.end()) {
      GELOGD("[%s] Output[%d] is referenced to input[%d]", GetNodeName(), index, reuse_input->second);
      outputs_start_[index] = inputs_start_[reuse_input->second];
    } else {
      GE_CHK_STATUS_RET_NOLOG(AllocateTensor(tensor_desc, outputs_start_[index], attr));
      GELOGD("Allocating output successfully. node: %s. index = %d, size = %zu", node_item_->NodeName().c_str(), index,
             outputs_start_[index].GetSize());
    }
  }

  if (execution_context_->trace_enabled) {
    outputs_start_[index].SetName(node_item_->NodeName() + "_out_" + std::to_string(index));
  }

  if (tensor != nullptr) {
    *tensor = outputs_start_ + index;
  }

  return SUCCESS;
}

Status TaskContext::AllocateOutputs(AllocationAttr *attr) {
  for (int i = 0; i < node_item_->num_outputs; ++i) {
    const auto &output_desc = node_item_->op_desc->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    uint32_t mem_type = 0;
    (void)AttrUtils::GetInt(output_desc, ATTR_OUTPUT_MEMORY_TYPE, mem_type);
    if (attr == nullptr) {
      auto tmp_attr = AllocationAttr(0, nullptr, static_cast<MemStorageType>(mem_type));
      GE_CHK_STATUS_RET_NOLOG(AllocateOutput(i, *output_desc, nullptr, &tmp_attr));
    } else {
      attr->SetMemType(static_cast<MemStorageType>(mem_type));
      GE_CHK_STATUS_RET_NOLOG(AllocateOutput(i, *output_desc, nullptr, attr));
    }
  }

  return SUCCESS;
}

Status TaskContext::AllocateTensor(size_t size, TensorValue &tensor, AllocationAttr *attr) {
  auto buffer = TensorBuffer::Create(execution_context_->allocator, size, attr);
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

int64_t TaskContext::GetSessionId() const { return execution_context_->session_id; }

Status TaskContext::GetStatus() const { return status_; }

void TaskContext::SetStatus(Status status) {
  status_ = status;
  if (status != SUCCESS) {
    execution_context_->SetErrorCode(status);
  }
}

Status TaskContext::AllocateWorkspace(size_t size, void **buffer, void *ori_addr) {
  GE_CHECK_NOTNULL(buffer);
  if (ori_addr == nullptr) {
    *buffer = execution_context_->allocator->Allocate(size, nullptr);
  } else {
    AllocationAttr attr(ori_addr);
    *buffer = execution_context_->allocator->Allocate(size, &attr);
  }

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
      auto input_offset = dst_node_item->input_start + dst_input_idx;
      GELOGI(
        "Propagate output of node %s, output index = %d, dst node = %s, "
        "dst_input_index = %d, dst_input_offset = %d.",
        node_item_->NodeName().c_str(), i, dst_node_item->NodeName().c_str(), dst_input_idx, input_offset);

      if (subgraph_context_->all_inputs_.size() <= static_cast<size_t>(input_offset)) {
        GELOGE(INTERNAL_ERROR, "[%s] input index out of range. index = %d, total input num = %zu", GetNodeName(),
               input_offset, subgraph_context_->all_inputs_.size());
        return INTERNAL_ERROR;
      }

      subgraph_context_->all_inputs_[input_offset] = *tensor;
      if (execution_context_->trace_enabled) {
        subgraph_context_->all_inputs_[input_offset].SetName(node_item_->NodeName() + "_in_" +
                                                             std::to_string(dst_input_idx));
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

ConstGeTensorDescPtr TaskContext::GetOutputDesc(int index) {
  return node_item_->op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
}

ConstGeTensorDescPtr TaskContext::GetInputDesc(int index) {
  return node_item_->op_desc->MutableInputDesc(static_cast<uint32_t>(index));
}

GeTensorDescPtr TaskContext::MutableInputDesc(int index) {
  return node_item_->op_desc->MutableInputDesc(static_cast<uint32_t>(index));
}

GeTensorDescPtr TaskContext::MutableOutputDesc(int index) {
  return node_item_->op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
}

bool TaskContext::IsForceInferShape() const { return force_infer_shape_; }

void TaskContext::SetForceInferShape(bool force_infer_shape) { force_infer_shape_ = force_infer_shape; }

void TaskContext::NodeDone() { subgraph_context_->NodeDone(node_item_->node); }

void TaskContext::OnError(Status error) {
  subgraph_context_->OnError(error);
  execution_context_->SetErrorCode(error);
}

bool TaskContext::IsTraceEnabled() const { return execution_context_->trace_enabled; }

TensorValue *TaskContext::GetVariable(const std::string &name) { return execution_context_->model->GetVariable(name); }

uint64_t TaskContext::GetIterationNumber() const { return iteration_; }

bool TaskContext::IsDumpEnabled() const { return execution_context_->dump_enabled; }

Status TaskContext::TryExecuteCallback(const function<void()> &callback_fun) const {
  if (!callback_fun) {
    return SUCCESS;
  }

  if (node_item_->has_observer) {
    return RegisterCallback(callback_fun);
  }

  callback_fun();
  return SUCCESS;
}
const DumpProperties &TaskContext::GetDumpProperties() const { return execution_context_->dump_properties; }
}  // namespace hybrid
}  // namespace ge
