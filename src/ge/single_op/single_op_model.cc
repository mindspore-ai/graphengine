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

#include "single_op/single_op_model.h"

#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "runtime/rt.h"
#include "task/tbe_task_builder.h"

using domi::TaskDef;
using std::unique_ptr;
using std::vector;

namespace ge {
namespace {
const size_t kDataOutputNum = 1;
}  // namespace
SingleOpModel::SingleOpModel(const std::string &model_name, const void *model_data, uint32_t model_size)
    : model_name_(model_name), ori_model_data_(model_data), ori_model_size_(model_size) {}

Status SingleOpModel::Init() {
  auto ret = InitModel();
  if (ret != SUCCESS) {
    return ret;
  }

  return ParseInputsAndOutputs();
}

Status SingleOpModel::InitModel() {
  ge::ModelData model;
  model.model_len = ori_model_size_;
  model.model_data = const_cast<void *>(ori_model_data_);

  auto ret = model_helper_.LoadModel(model);
  if (ret != SUCCESS) {
    GELOGE(ret, "LoadModel failed");
    return ret;
  }

  return SUCCESS;
}

void SingleOpModel::ParseOpModelParams(ModelHelper &model_helper, SingleOpModelParam &param) {
  int64_t value = 0;
  bool ret = false;
  std::shared_ptr<ge::GeModel> model = model_helper.GetGeModel();
  GE_CHECK_NOTNULL_JUST_RETURN(model);
  ret = ge::AttrUtils::GetInt(model, ATTR_MODEL_MEMORY_SIZE, value);
  param.memory_size = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(model, ATTR_MODEL_WEIGHT_SIZE, value);
  param.weight_size = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(model, MODEL_ATTR_TASK_GEN_BASE_ADDR, value);
  param.base_addr = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, value);
  param.weight_addr = ret ? static_cast<uint64_t>(value) : 0;

  GELOGI("ParseOpModelParams(), memory_size:%lu, weight_size:%lu.", param.memory_size, param.weight_size);
}

Status SingleOpModel::InitModelMem(StreamResource &res) {
  ParseOpModelParams(model_helper_, model_params_);
  if (model_params_.memory_size > ALLOC_MEMORY_MAX_SIZE || model_params_.weight_size > ALLOC_MEMORY_MAX_SIZE) {
    GELOGE(PARAM_INVALID, "Can not alloc memory larger than %lu. memory size = %lu, weight size = %lu",
           ALLOC_MEMORY_MAX_SIZE, model_params_.memory_size, model_params_.weight_size);
    return PARAM_INVALID;
  }

  if (model_params_.memory_size > 0) {
    model_params_.mem_base = res.MallocMemory(model_params_.memory_size);
    if (model_params_.mem_base == nullptr) {
      return RT_FAILED;
    }
  }

  if (model_params_.weight_size > 0) {
    model_params_.weight_base = res.MallocWeight(model_params_.weight_size);
    if (model_params_.weight_base == nullptr) {
      // no need to free memory, for that was handled by StreamResources
      return RT_FAILED;
    }
  }

  return SUCCESS;
}

Status SingleOpModel::ParseInputNode(const OpDescPtr &op_desc) {
  vector<int64_t> offsets = op_desc->GetOutputOffset();
  if (offsets.size() != kDataOutputNum) {
    GELOGE(PARAM_INVALID, "Data op should have only one output, but got %zu", op_desc->GetOutputOffset().size());
    return PARAM_INVALID;
  }

  auto output_desc = op_desc->GetOutputDescPtr(0);
  GE_CHECK_NOTNULL(output_desc);
  int64_t tensor_size = 0;
  (void)TensorUtils::GetSize(*output_desc, tensor_size);
  input_offset_list_.emplace_back(offsets[0]);
  input_sizes_.emplace_back(tensor_size);
  GELOGI("[%s] parse input node: %s, size = %ld, offset = %u", model_name_.c_str(), op_desc->GetName().c_str(),
         tensor_size, static_cast<uint32_t>(offsets[0]));
  return SUCCESS;
}

void SingleOpModel::ParseOutputNode(const OpDescPtr &op_desc) {
  vector<int64_t> offsets = op_desc->GetInputOffset();
  for (uint32_t k = 0; k < static_cast<uint32_t>(offsets.size()); ++k) {
    auto input_desc = op_desc->GetInputDescPtr(k);
    if (input_desc == nullptr) {
      continue;
    }
    int64_t tensor_size = 0;
    (void)TensorUtils::GetSize(*input_desc, tensor_size);
    output_offset_list_.emplace_back(offsets[k]);
    output_sizes_.emplace_back(tensor_size);
    GELOGI("[%s] parse output node: %s, size = %ld, offset = %u", model_name_.c_str(), op_desc->GetName().c_str(),
           tensor_size, static_cast<uint32_t>(offsets[k]));
  }
}

Status SingleOpModel::ParseInputsAndOutputs() {
  auto ge_model = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);
  Graph graph = ge_model->GetGraph();
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGE(PARAM_INVALID, "[%s] compute_graph is null", model_name_.c_str());
    return PARAM_INVALID;
  }

  auto nodes = compute_graph->GetDirectNode();
  size_t model_op_size = nodes.size();
  GELOGI("[%s] node size = %zu", model_name_.c_str(), model_op_size);

  for (size_t i = 0; i < model_op_size; ++i) {
    auto node = nodes.at(i);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    op_list_[i] = op_desc;
    auto op_type = op_desc->GetType();
    GELOGI("[%s] node[%zu] = %s, type = %s", model_name_.c_str(), i, node->GetName().c_str(), op_type.c_str());

    if (op_type == DATA_TYPE || op_type == AIPP_DATA_TYPE) {
      auto ret = ParseInputNode(op_desc);
      if (ret != SUCCESS) {
        return ret;
      }
    }

    if (op_type == NETOUTPUT) {
      ParseOutputNode(op_desc);
    }

    ge_model->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(op_desc);
  }

  return SUCCESS;
}

Status SingleOpModel::SetInputsAndOutputs(SingleOp &single_op) {
  // for lhisi
  const char *use_physical_address = std::getenv("GE_USE_PHYSICAL_ADDRESS");
  if (use_physical_address != nullptr) {
    single_op.use_physical_addr_ = true;
  }

  int arg_index = 0;
  for (size_t i = 0; i < input_offset_list_.size(); ++i) {
    auto *addr = model_params_.mem_base + input_offset_list_[i];
    auto ret = ModelUtils::ConvertVirtualAddressToPhysical(addr, input_sizes_[i], addr);
    if (ret != SUCCESS) {
      GELOGE(ret, "ConvertVirtualAddressToPhysical failed. Input index = %zu", i);
      return ret;
    }
    model_params_.addr_mapping_.emplace(reinterpret_cast<uintptr_t>(addr), arg_index++);
    single_op.input_sizes_.emplace_back(input_sizes_[i]);
    single_op.input_addr_list_.emplace_back(addr);
  }

  for (size_t i = 0; i < output_offset_list_.size(); ++i) {
    auto *addr = model_params_.mem_base + output_offset_list_[i];
    auto ret = ModelUtils::ConvertVirtualAddressToPhysical(addr, output_sizes_[i], addr);
    if (ret != SUCCESS) {
      GELOGE(ret, "ConvertVirtualAddressToPhysical failed. Output index = %zu", i);
      return ret;
    }
    model_params_.addr_mapping_.emplace(reinterpret_cast<uintptr_t>(addr), arg_index++);
    single_op.output_sizes_.emplace_back(output_sizes_[i]);
    single_op.output_addr_list_.emplace_back(addr);
  }

  single_op.args_.resize(arg_index);
  return SUCCESS;
}

Status SingleOpModel::BuildTaskList(SingleOp &single_op) {
  auto ge_model = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);
  auto tasks = ge_model->GetModelTaskDefPtr()->task();
  for (int i = 0; i < tasks.size(); ++i) {
    const TaskDef &task_def = tasks[i];
    GELOGI("[%s] Task[%d], type = %u, DebugString = %s", model_name_.c_str(), i, task_def.type(),
           task_def.DebugString().c_str());
    auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
    if (task_type == RT_MODEL_TASK_KERNEL) {
      OpTask *task = nullptr;
      auto ret = BuildKernelTask(task_def.kernel(), single_op, &task);
      if (ret != SUCCESS) {
        return ret;
      }

      single_op.tasks_.emplace_back(task);
    } else if (task_type == RT_MODEL_TASK_KERNEL_EX) {
      GELOGD("BuildKernelExTask is not supported. modelName = %s", model_name_.c_str());
      return UNSUPPORTED;
    } else {
      // skip
      GELOGD("Skip task type: %d", static_cast<int>(task_type));
    }
  }
  return SUCCESS;
}

void SingleOpModel::ParseArgTable(TbeOpTask *task, SingleOp &op) {
  if (task == nullptr) {
    GELOGE(PARAM_INVALID, "tbe op task is nullptr");
    return;
  }
  // args: addr1, addr2, addr3 ...
  auto *args = const_cast<uintptr_t *>(reinterpret_cast<const uintptr_t *>(task->GetArgs()));
  size_t arg_size = task->GetArgSize();
  for (size_t i = 0; i < arg_size / sizeof(void *); ++i) {
    uintptr_t *ptr_to_addr = args + i;
    uintptr_t addr = *ptr_to_addr;
    auto iter = model_params_.addr_mapping_.find(addr);
    if (iter != model_params_.addr_mapping_.end()) {
      int arg_index = iter->second;
      GELOGI("%s args[%zu] mapped to user designated args[%d]", task->GetStubName().c_str(), i, arg_index);
      op.arg_table_[iter->second].emplace_back(ptr_to_addr);
    }
  }
}

Status SingleOpModel::BuildKernelTask(const domi::KernelDef &kernel_def, SingleOp &single_op, OpTask **task) {
  GE_CHECK_NOTNULL(task);
  const auto &context = kernel_def.context();
  auto kernel_type = static_cast<cce::ccKernelType>(context.kernel_type());
  if (kernel_type != cce::ccKernelType::TE) {
    GELOGE(UNSUPPORTED, "Only TBE kernel is supported, but got %u", context.kernel_type());
    return UNSUPPORTED;
  }

  auto iter = op_list_.find(context.op_index());
  if (iter == op_list_.end()) {
    GELOGE(INTERNAL_ERROR, "op desc not found. op index = %u", context.op_index());
    return INTERNAL_ERROR;
  }

  auto *tbe_task = new (std::nothrow) TbeOpTask();
  if (tbe_task == nullptr) {
    GELOGE(MEMALLOC_FAILED, "create tbe op task failed");
    return MEMALLOC_FAILED;
  }

  auto builder = TbeTaskBuilder(model_name_, iter->second, kernel_def);
  auto ret = builder.BuildTask(*tbe_task, model_params_);
  if (ret != SUCCESS) {
    delete tbe_task;
    tbe_task = nullptr;
    return ret;
  }

  single_op.arg_table_.resize(single_op.input_sizes_.size() + single_op.output_sizes_.size());
  ParseArgTable(tbe_task, single_op);

  *task = tbe_task;
  return SUCCESS;
}

Status SingleOpModel::BuildOp(StreamResource &resource, SingleOp &single_op) {
  auto ret = InitModelMem(resource);
  if (ret != SUCCESS) {
    return ret;
  }

  ret = SetInputsAndOutputs(single_op);
  if (ret != SUCCESS) {
    return ret;
  }
  return BuildTaskList(single_op);
}
}  // namespace ge
