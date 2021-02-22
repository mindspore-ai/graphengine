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
#include "graph/types.h"
#include "graph/debug/ge_attr_define.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/subgraph_executor.h"
#include "common/profiling/profiling_manager.h"

namespace ge {
namespace hybrid {
TaskContext::TaskContext(GraphExecutionContext *execution_context,
                         NodeState *node_state,
                         SubgraphContext *subgraph_context)
    : node_state_(node_state),
      node_item_(node_state->GetNodeItem()),
      execution_context_(execution_context),
      subgraph_context_(subgraph_context) {}

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

std::unique_ptr<TaskContext> TaskContext::Create(NodeState *node_state,
                                                 GraphExecutionContext *execution_context,
                                                 SubgraphContext *subgraph_context) {
  const NodeItem &node_item = *node_state->GetNodeItem();
  GELOGI("[%s] To create task context, input start = %d, num_inputs = %d, output start = %d, num_outputs = %d.",
         node_item.NodeName().c_str(),
         node_item.input_start,
         node_item.num_inputs,
         node_item.output_start,
         node_item.num_outputs);
  if (node_item.input_start < 0 || node_item.output_start < 0) {
    GELOGE(INTERNAL_ERROR,
           "NodeItem not property initialized. input_start = %d, output_start = %d",
           node_item.input_start,
           node_item.output_start);
    return nullptr;
  }

  auto task_context = std::unique_ptr<TaskContext>(
      new(std::nothrow)TaskContext(execution_context, node_state, subgraph_context));
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

int TaskContext::NumInputs() const {
  return node_item_->num_inputs;
}

int TaskContext::NumOutputs() const {
  return node_item_->num_outputs;
}

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

std::size_t TaskContext::NumWorkspaces() const {
  return workspaces_.size();
}

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
  if (callback_fun == nullptr) {
    GELOGW("[%s] Callback is NULL", GetNodeName());
    return SUCCESS;
  }
  auto ret = execution_context_->callback_manager->RegisterCallback(GetStream(), callback_fun);
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

Status TaskContext::AllocateOutput(int index,
                                   const GeTensorDesc &tensor_desc,
                                   TensorValue **tensor,
                                   AllocationAttr *attr) {
  GELOGI("To allocate output for node: %s. index = %d, tensor desc = %s",
         node_item_->NodeName().c_str(),
         index,
         TensorDesc2String(tensor_desc).c_str());

  if (index < 0 || index >= node_item_->num_outputs) {
    GELOGE(PARAM_INVALID, "output index out of range. num_output = %d, index = %d", node_item_->num_outputs, index);
    return PARAM_INVALID;
  }

  if (outputs_start_[index].GetData() != nullptr) {
    GELOGI("already allocated as net output");
    return SUCCESS;
  }

  int32_t calc_type = 0;
  bool ret = ge::AttrUtils::GetInt(tensor_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
  if (ret && (calc_type == static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY))) {
    outputs_start_[index] = TensorValue();
    return SUCCESS;
  }

  auto it = node_item_->ref_outputs.find(index);
  if (it != node_item_->ref_outputs.end()) {
    auto &ref_node = it->second;
    GELOGD("source node of %s:%d = %s, op_type = %s",
           node_item_->NodeName().c_str(),
           index,
           ref_node->GetName().c_str(),
           ref_node->GetType().c_str());

    TensorValue *ref_tensor = execution_context_->model->GetVariable(ref_node->GetName());
    GE_CHECK_NOTNULL(ref_tensor);
    outputs_start_[index] = *ref_tensor;
  } else {
    auto reuse_output_it = node_item_->reuse_outputs.find(index);
    if (reuse_output_it != node_item_->reuse_outputs.end()) {
      GELOGD("[%s] reuse output [%d] with output [%d]", GetNodeName(), index, reuse_output_it->second);
      outputs_start_[index] = outputs_start_[reuse_output_it->second];
    } else {
      auto reuse_input = node_item_->reuse_inputs.find(index);
      if (reuse_input != node_item_->reuse_inputs.end()) {
        GELOGD("[%s] Output[%d] is referenced to input[%d]", GetNodeName(), index, reuse_input->second);
        outputs_start_[index] = inputs_start_[reuse_input->second];
      } else {
        GE_CHK_STATUS_RET_NOLOG(AllocateTensor(tensor_desc, outputs_start_[index], attr));
        GELOGD("Allocating output successfully. node: %s. index = %d, size = %zu",
               node_item_->NodeName().c_str(), index, outputs_start_[index].GetSize());
      }
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
    const auto &output_desc = node_item_->MutableOutputDesc(i);
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

const NodeItem &TaskContext::GetNodeItem() const {
  return *node_item_;
}

Status TaskContext::SetOutput(int index, const TensorValue &tensor) {
  if (index < 0 || index >= node_item_->num_outputs) {
    GELOGE(PARAM_INVALID, "output index out of range. num_output = %d, index = %d", node_item_->num_outputs, index);
    return PARAM_INVALID;
  }

  GELOGD("Set %s:%d with tensor: %s",
         node_item_->NodeName().c_str(),
         index,
         tensor.DebugString().c_str());
  outputs_start_[index] = tensor;
  return SUCCESS;
}

rtStream_t TaskContext::GetStream() const {
  return execution_context_->stream;
}

int64_t TaskContext::GetSessionId() const {
  return execution_context_->session_id;
}

Status TaskContext::GetStatus() const {
  return status_;
}

void TaskContext::SetStatus(Status status) {
  status_ = status;
  if (status != SUCCESS) {
    execution_context_->SetErrorCode(status);
  }
}

uint32_t TaskContext::GetTaskId() const {
  return task_id_;
}

void TaskContext::SetTaskId(uint32_t task_id) {
  task_id_ = task_id;
}

uint32_t TaskContext::GetStreamId() const {
  return stream_id_;
}

void TaskContext::SetStreamId(uint32_t stream_id) {
  stream_id_ = stream_id;
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
      GELOGD(
          "Propagate output of node %s, output index = %d, dst node = %s, "
          "dst_input_index = %d, dst_input_offset = %d.",
          node_item_->NodeName().c_str(),
          i,
          dst_node_item->NodeName().c_str(),
          dst_input_idx,
          input_offset);

      if (subgraph_context_->all_inputs_.size() <= static_cast<size_t>(input_offset)) {
        GELOGE(INTERNAL_ERROR,
               "[%s] input index out of range. index = %d, total input num = %zu",
               GetNodeName(),
               input_offset,
               subgraph_context_->all_inputs_.size());
        return INTERNAL_ERROR;
      }

      subgraph_context_->all_inputs_[input_offset] = *tensor;
      if (execution_context_->trace_enabled) {
        subgraph_context_->all_inputs_[input_offset].SetName(
            node_item_->NodeName() + "_in_" + std::to_string(dst_input_idx));
      }
    }
  }

  return SUCCESS;
}

const void *TaskContext::GetVarBaseAddr() {
  return execution_context_->model->GetVarMemBase();
}

const char *TaskContext::GetNodeName() const {
  return node_item_->NodeName().c_str();
}

void TaskContext::ReleaseInputsAndOutputs() {
  for (int i = 0; i < node_item_->num_inputs; ++i) {
    auto tensor = inputs_start_ + i;
    tensor->Destroy();
    GELOGD("[%s] Tensor of input[%d] released", GetNodeName(), i);
  }

  for (int i = 0; i < node_item_->num_outputs; ++i) {
    auto tensor = outputs_start_ + i;
    tensor->Destroy();
    GELOGD("[%s] Tensor of output[%d] released", GetNodeName(), i);
  }
}

void TaskContext::ReleaseInput(int index) {
  auto input_tensor = MutableInput(index);
  if (input_tensor != nullptr) {
    input_tensor->Destroy();
    GELOGD("[%s] Tensor of input[%d] released", GetNodeName(), index);
  }
}

ConstGeTensorDescPtr TaskContext::GetOutputDesc(int index) const {
  return node_item_->MutableOutputDesc(static_cast<uint32_t>(index));
}

ConstGeTensorDescPtr TaskContext::GetInputDesc(int index) const {
  return node_item_->MutableInputDesc(index);
}

GeTensorDescPtr TaskContext::MutableInputDesc(int index) const {
  return node_item_->MutableInputDesc(index);
}

GeTensorDescPtr TaskContext::MutableOutputDesc(int index) const {
  return node_item_->MutableOutputDesc(static_cast<uint32_t>(index));
}

bool TaskContext::IsForceInferShape() const {
  return force_infer_shape_;
}

void TaskContext::SetForceInferShape(bool force_infer_shape) {
  force_infer_shape_ = force_infer_shape;
}

void TaskContext::NodeDone() {
  subgraph_context_->NodeDone(node_item_->node);
}

void TaskContext::OnError(Status error) {
  subgraph_context_->OnError(error);
  execution_context_->SetErrorCode(error);
}

bool TaskContext::IsTraceEnabled() const {
  return execution_context_->trace_enabled;
}

TensorValue *TaskContext::GetVariable(const std::string &name) {
  return execution_context_->model->GetVariable(name);
}

uint64_t TaskContext::GetIterationNumber() const {
  return iteration_;
}

bool TaskContext::IsDumpEnabled() const {
  return execution_context_->dump_enabled;
}

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
const DumpProperties &TaskContext::GetDumpProperties() const {
  return execution_context_->dump_properties;
}

bool TaskContext::NeedCallback() {
  return node_item_->has_observer || IsDumpEnabled() || execution_context_->profiling_level > 0;
}

Status TaskContext::Synchronize() {
  return execution_context_->Synchronize(GetStream());
}

Status TaskContext::SaveProfilingTaskDescInfo(uint32_t task_id, uint32_t  stream_id,
                                              uint32_t task_type, uint32_t block_dim) {
  if (ProfilingManager::Instance().ProfilingModelExecuteOn()) {
    const NodeItem &node_item = GetNodeItem();
    auto op_desc = node_item.GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const GraphExecutionContext * graph_context = GetExecutionContext();
    GE_CHECK_NOTNULL(graph_context);
    const HybridModel *model = graph_context->model;
    GE_CHECK_NOTNULL(model);

    std::string op_name = op_desc->GetName();
    std::string dynamic_model_name = model->GetModelName();
    TaskDescInfo tmp_task_desc_info;
    tmp_task_desc_info.model_name = dynamic_model_name;
    tmp_task_desc_info.op_name = op_name;
    tmp_task_desc_info.block_dim = block_dim;
    tmp_task_desc_info.task_type = task_type;
    tmp_task_desc_info.task_id = task_id;
    tmp_task_desc_info.stream_id = stream_id;
    tmp_task_desc_info.shape_type = "dynamic";
    tmp_task_desc_info.cur_iter_num = iteration_ + 1;
    task_desc_info.emplace_back(tmp_task_desc_info);
  }

  return SUCCESS;
}

NodeState *TaskContext::GetNodeState() const {
  return node_state_;
}

Status TaskContext::SaveProfilingGraphDescInfo(uint32_t task_id, uint32_t stream_id) {
  if (ProfilingManager::Instance().ProfilingModelExecuteOn()) {
    const NodeItem &node_item = GetNodeItem();
    auto op_desc = node_item.GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const GraphExecutionContext * graph_context = GetExecutionContext();
    GE_CHECK_NOTNULL(graph_context);
    const HybridModel *model = graph_context->model;
    GE_CHECK_NOTNULL(model);

    std::string dynamic_model_name = model->GetModelName();
    auto op_mode = static_cast<uint32_t>(domi::ImplyType::INVALID);
    if (AttrUtils::GetInt(op_desc, ATTR_NAME_IMPLY_TYPE, op_mode) &&
        op_mode == static_cast<uint32_t>(domi::ImplyType::TVM)) {
      ComputeGraphDescInfo tmp_compute_graph_info;
      tmp_compute_graph_info.model_name = dynamic_model_name;
      tmp_compute_graph_info.op_name = op_desc->GetName();
      tmp_compute_graph_info.op_type = op_desc->GetType();
      tmp_compute_graph_info.task_id = task_id;
      tmp_compute_graph_info.stream_id = stream_id;
      compute_graph_info.emplace_back(tmp_compute_graph_info);
    }
  }
  return SUCCESS;
}

}  // namespace hybrid
}  // namespace ge
