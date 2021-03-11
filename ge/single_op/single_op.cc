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

#include "single_op/single_op.h"

#include "common/fmk_types.h"
#include "common/ge_types.h"
#include "common/math/math_util.h"
#include "common/profiling/profiling_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/load/model_manager/model_utils.h"
#include "runtime/mem.h"
#include "single_op/single_op_manager.h"
#include "single_op/task/build_task_utils.h"
#include "graph/load/model_manager/model_manager.h"

namespace ge {
namespace {
const size_t kDataMemAlignSize = 32;
const size_t kDataMemAlignUnit = 2;
const string kShapeTypeDynamic = "dynamic";
const string kShapeTypeStatic = "static";

size_t GetAlignedSize(size_t size) {
  size_t aligned_size = (size + kDataMemAlignUnit * kDataMemAlignSize - 1) / kDataMemAlignSize * kDataMemAlignSize;
  return aligned_size;
}

Status ProfilingTaskInfo(OpTask *op_task, const string &shape_type) {
  if (!ProfilingManager::Instance().ProfilingModelLoadOn()) {
    return SUCCESS;
  }

  TaskDescInfo tmp_task_desc_info;
  uint32_t model_id;
  if (op_task->GetProfilingArgs(tmp_task_desc_info, model_id) != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Get profiling data of task failed");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GELOGD("ProfilingReport of op[%s] model[%s] start.",
         tmp_task_desc_info.op_name.c_str(), tmp_task_desc_info.model_name.c_str());

  tmp_task_desc_info.shape_type = shape_type;
  tmp_task_desc_info.cur_iter_num = 0;
  tmp_task_desc_info.task_type = op_task->GetTaskType();

  std::vector<TaskDescInfo> task_desc_info;
  task_desc_info.emplace_back(tmp_task_desc_info);

  auto &profiling_manager = ProfilingManager::Instance();
  profiling_manager.ReportProfilingData(model_id, task_desc_info);
  return SUCCESS;
}
}  // namespace

SingleOp::SingleOp(StreamResource *stream_resource, std::mutex *stream_mutex, rtStream_t stream)
    : stream_resource_(stream_resource), stream_mutex_(stream_mutex), stream_(stream) {
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY SingleOp::~SingleOp() {
  for (auto task : tasks_) {
    delete task;
    task = nullptr;
  }
}

Status SingleOp::ValidateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  auto num_inputs = inputs.size();
  if (num_inputs != input_sizes_.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Input num mismatch. model expect %zu, but given %zu", input_addr_list_.size(),
           inputs.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    // preventing from read out of bound
    size_t aligned_size = GetAlignedSize(inputs[i].length);
    GELOGI("Input [%zu], aligned_size:%zu, inputs.length:%lu, input_sizes_:%zu",
           i, aligned_size, inputs[i].length, input_sizes_[i]);
    if (aligned_size < input_sizes_[i]) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Input size mismatch. index = %zu, model expect %zu,"
                            " but given %zu(after align)", i, input_sizes_[i], aligned_size);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
  }

  auto num_outputs = outputs.size();
  if (num_outputs != output_sizes_.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "output num mismatch. model expect %zu, but given %zu",
           output_sizes_.size(), outputs.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    // preventing from write out of bound
    size_t aligned_size = GetAlignedSize(outputs[i].length);
    GELOGI("Output [%zu], aligned_size:%zu, outputs.length:%lu, output_sizes_:%zu",
           i, aligned_size, outputs[i].length, output_sizes_[i]);
    if (aligned_size < output_sizes_[i]) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Output size mismatch. index = %zu, model expect %zu,"
                            "but given %zu(after align)", i, output_sizes_[i], aligned_size);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status SingleOp::GetArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  size_t arg_index = 0;
  for (auto &input : inputs) {
    args_[arg_index++] = reinterpret_cast<uintptr_t>(input.data);
  }

  for (auto &output : outputs) {
    args_[arg_index++] = reinterpret_cast<uintptr_t>(output.data);
  }
  return SUCCESS;
}

Status SingleOp::UpdateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  Status ret = GetArgs(inputs, outputs);
  if (ret != SUCCESS) {
    return ret;
  }
  // update tbe task args
  size_t num_args = arg_table_.size();
  for (size_t i = 0; i < num_args; ++i) {
    std::vector<uintptr_t *> &ptr_to_arg_in_tasks = arg_table_[i];
    if (ptr_to_arg_in_tasks.empty()) {
      GELOGW("found NO arg address to update for arg[%lu]", i);
      continue;
    }

    for (uintptr_t *arg_addr : ptr_to_arg_in_tasks) {
      *arg_addr = args_[i];
    }
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status SingleOp::ExecuteAsync(const std::vector<DataBuffer> &inputs,
                                                                               const std::vector<DataBuffer> &outputs) {
  Status ret = ValidateArgs(inputs, outputs);
  if (ret != SUCCESS) {
    return ret;
  }

  GE_CHECK_NOTNULL(stream_resource_);
  std::lock_guard<std::mutex> lk(*stream_mutex_);
  auto current_mem_base = stream_resource_->GetMemoryBase();
  if (running_param_->mem_base != current_mem_base) {
    running_param_->mem_base = const_cast<uint8_t *>(current_mem_base);
    GELOGD("Memory base changed, new memory base = %p", current_mem_base);
    for (auto &task : tasks_) {
      auto new_address = BuildTaskUtils::GetAddresses(task->GetOpdesc(), *running_param_);
      GE_CHK_STATUS_RET(task->UpdateArgTable(*running_param_),
                        "[%s] Failed to update arg table",
                        task->GetOpdesc()->GetName().c_str());
    }
  }
  ret = UpdateArgs(inputs, outputs);
  if (ret != SUCCESS) {
    return ret;
  }

  for (auto &task : tasks_) {
    ret = task->LaunchKernel(stream_);
    if (ret != SUCCESS) {
      return ret;
    }
    GE_CHK_STATUS_RET(task->OpenDump(stream_), "Open single op %s dump filed",task->GetOpdesc()->GetName().c_str());
    GE_CHK_STATUS_RET_NOLOG(ProfilingTaskInfo(task, kShapeTypeStatic));
  }

  return ret;
}

void SingleOp::SetStream(rtStream_t stream) {
  stream_ = stream;
}

DynamicSingleOp::DynamicSingleOp(uintptr_t resource_id, std::mutex *stream_mutex, rtStream_t stream)
    : resource_id_(resource_id), stream_mutex_(stream_mutex), stream_(stream) {
}

Status DynamicSingleOp::ValidateParams(const vector<GeTensorDesc> &input_desc,
                                       const std::vector<DataBuffer> &inputs,
                                       std::vector<GeTensorDesc> &output_desc,
                                       std::vector<DataBuffer> &outputs) const {
  if (inputs.size() != input_desc.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "Input number mismatches input desc number. Input num = %zu, input desc num = %zu",
           inputs.size(),
           input_desc.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (outputs.size() != output_desc.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "Output number mismatches output desc number. Output num = %zu, output desc num = %zu",
           outputs.size(),
           output_desc.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (input_desc.size() != num_inputs_) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "Input number mismatches. expect %zu, but given %zu",
           num_inputs_,
           input_desc.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (output_desc.size() != num_outputs_) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "Output number mismatches. expect %zu, but given %zu",
           num_outputs_,
           output_desc.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  return SUCCESS;
}

Status DynamicSingleOp::ExecuteAsync(const vector<GeTensorDesc> &input_desc,
                                     const vector<DataBuffer> &input_buffers,
                                     vector<GeTensorDesc> &output_desc,
                                     vector<DataBuffer> &output_buffers) {
  GE_CHK_STATUS_RET_NOLOG(ValidateParams(input_desc, input_buffers, output_desc, output_buffers));
  if (hybrid_model_executor_ != nullptr) {
    GELOGD("Execute multi-task dynamic single op by hybrid model executor");
    hybrid::HybridModelExecutor::ExecuteArgs args;
    for (auto &input : input_buffers) {
      args.inputs.emplace_back(hybrid::TensorValue(input.data, input.length));
    }
    for (auto &output : output_buffers) {
      args.outputs.emplace_back(hybrid::TensorValue(output.data, output.length));
    }
    for (auto &tensor_desc : input_desc) {
      auto desc = MakeShared<GeTensorDesc>(tensor_desc);
      GE_CHECK_NOTNULL(desc);
      args.input_desc.emplace_back(desc);
    }

    return hybrid_model_executor_->Execute(args);
  }

  std::lock_guard<std::mutex> lk(*stream_mutex_);
  GE_CHECK_NOTNULL(op_task_);

  GE_CHK_STATUS_RET_NOLOG(op_task_->LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream_));
  GE_CHK_STATUS_RET_NOLOG(op_task_->OpenDump(stream_));
  GE_CHK_STATUS_RET_NOLOG(ProfilingTaskInfo(op_task_.get(), kShapeTypeDynamic));
  return SUCCESS;
}
}  // namespace ge
