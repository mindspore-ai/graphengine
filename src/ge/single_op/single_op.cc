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
#include "common/math/math_util.h"
#include "common/profiling/profiling_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "runtime/mem.h"
#include "single_op/single_op_manager.h"
#include "graph/load/new_model_manager/model_manager.h"

namespace ge {
namespace {
const size_t kDataMemAlignSize = 32;

size_t GetAlignedSize(size_t size) {
  size_t aligned_size = (size + 2 * kDataMemAlignSize - 1) / kDataMemAlignSize * kDataMemAlignSize;
  return aligned_size;
}
}  // namespace

SingleOp::SingleOp(std::mutex *stream_mutex, rtStream_t stream) : stream_mutex_(stream_mutex), stream_(stream) {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY SingleOp::~SingleOp() {
  for (auto task : tasks_) {
    delete task;
    task = nullptr;
  }
  GELOGI("SingleOp destory sessionId = %lu", aicpu_session_id_);
  ModelManager::GetInstance()->DestroyAicpuSession(aicpu_session_id_);
}

Status SingleOp::ValidateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  auto num_inputs = inputs.size();
  if (num_inputs != input_sizes_.size()) {
    GELOGE(PARAM_INVALID, "Input num mismatch. model expect %zu, but given %zu", input_addr_list_.size(),
           inputs.size());
    return PARAM_INVALID;
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    // preventing from read out of bound
    size_t aligned_size = GetAlignedSize(inputs[i].length);
    GELOGI("Input [%zu], aligned_size:%zu, inputs.length:%lu, input_sizes_:%lu", i, aligned_size, inputs[i].length,
           input_sizes_[i]);
    if (aligned_size < input_sizes_[i]) {
      GELOGE(PARAM_INVALID,
             "Input size mismatch. index = %zu, model expect %zu,"
             " but given %zu(after align)",
             i, input_sizes_[i], aligned_size);
      return PARAM_INVALID;
    }
  }

  auto num_outputs = outputs.size();
  if (num_outputs != output_sizes_.size()) {
    GELOGE(PARAM_INVALID, "output num mismatch. model expect %zu, but given %zu", output_sizes_.size(), outputs.size());
    return PARAM_INVALID;
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    // preventing from write out of bound
    size_t aligned_size = GetAlignedSize(outputs[i].length);
    GELOGI("Output [%zu], aligned_size:%zu, outputs.length:%lu, output_sizes_:%lu", i, aligned_size, outputs[i].length,
           output_sizes_[i]);
    if (aligned_size < output_sizes_[i]) {
      GELOGE(PARAM_INVALID,
             "Output size mismatch. index = %zu, model expect %zu,"
             "but given %zu(after align)",
             i, output_sizes_[i], aligned_size);
      return PARAM_INVALID;
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
  // update aicpu_TF or aicpu_CC args
  for (auto &task : tasks_) {
    size_t io_addr_num = args_.size();
    if (task->GetOpTaskType() == OP_TASK_AICPU) {
      GELOGD("Update aicpu_TF task args");
      auto *dst_io_addr = const_cast<uintptr_t *>(reinterpret_cast<const uintptr_t *>(task->GetIOAddr()));
      GE_CHECK_NOTNULL(dst_io_addr);
      auto rt_ret = rtMemcpyAsync(dst_io_addr, sizeof(uint64_t) * args_.size(), &args_[0],
                                  sizeof(uint64_t) * args_.size(), RT_MEMCPY_HOST_TO_DEVICE_EX, stream_);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "rtMemcpyAsync addresses failed, ret = %d", rt_ret);
        return RT_FAILED;
      }
    } else if (task->GetOpTaskType() == OP_TASK_AICPUCC) {
      GELOGD("Update aicpu_CC task args");
      const uintptr_t *task_io_addr = reinterpret_cast<const uintptr_t *>(task->GetIOAddr());
      GE_CHECK_NOTNULL(task_io_addr);
      auto io_addr = reinterpret_cast<uint64_t *>(const_cast<uintptr_t *>(task_io_addr));
      for (size_t i = 0; i < io_addr_num; ++i) {
        io_addr[i] = reinterpret_cast<uintptr_t>(args_[i]);
      }
    } else {
      GELOGW("Only TF_kernel aicpu and aicpu_CC are supported, but got %u", task->GetOpTaskType());
      continue;
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

  std::lock_guard<std::mutex> lk(*stream_mutex_);
  ret = UpdateArgs(inputs, outputs);
  if (ret != SUCCESS) {
    return ret;
  }

  for (auto &task : tasks_) {
    ret = task->LaunchKernel(stream_);
    if (ret != SUCCESS) {
      return ret;
    }
    ret = task->OpenDump(args_, stream_);
    if (ret != SUCCESS) {
      GELOGE(ret, "Open dump failed");
      return ret;
    }
  }

  return ret;
}

void SingleOp::SetStream(rtStream_t stream) { stream_ = stream; }

void SingleOp::SetSessionID(uint64_t session_id) { aicpu_session_id_ = session_id; }

DynamicSingleOp::DynamicSingleOp(uintptr_t resource_id, std::mutex *stream_mutex, rtStream_t stream)
    : resource_id_(resource_id), stream_mutex_(stream_mutex), stream_(stream) {}

DynamicSingleOp::~DynamicSingleOp() {
  GELOGI("DynamicSingleOp destory sessionId = %lu", aicpu_session_id_);
  ModelManager::GetInstance()->DestroyAicpuSession(aicpu_session_id_);
}

Status DynamicSingleOp::ValidateParams(const vector<GeTensorDesc> &input_desc, const std::vector<DataBuffer> &inputs,
                                       std::vector<GeTensorDesc> &output_desc, std::vector<DataBuffer> &outputs) const {
  if (inputs.size() != input_desc.size()) {
    GELOGE(PARAM_INVALID, "Input number mismatches input desc number. Input num = %zu, input desc num = %zu",
           inputs.size(), input_desc.size());
    return PARAM_INVALID;
  }

  if (outputs.size() != output_desc.size()) {
    GELOGE(PARAM_INVALID, "Output number mismatches output desc number. Output num = %zu, output desc num = %zu",
           outputs.size(), output_desc.size());
    return PARAM_INVALID;
  }

  if (input_desc.size() != num_inputs_) {
    GELOGE(PARAM_INVALID, "Input number mismatches. expect %zu, but given %zu", num_inputs_, input_desc.size());
    return PARAM_INVALID;
  }

  if (output_desc.size() != num_outputs_) {
    GELOGE(PARAM_INVALID, "Output number mismatches. expect %zu, but given %zu", num_outputs_, output_desc.size());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status DynamicSingleOp::AllocateWorkspaces(const std::vector<int64_t> &workspace_sizes,
                                           std::vector<void *> &workspaces) {
  static const std::string kPurpose("malloc workspace memory for dynamic op.");
  if (workspace_sizes.empty()) {
    GELOGD("No need to allocate workspace.");
    return SUCCESS;
  }
  int64_t total_size = 0;
  std::vector<int64_t> ws_offsets;
  for (auto ws_size : workspace_sizes) {
    // alignment and padding should be done in OpParaCalculate
    GE_CHK_STATUS_RET_NOLOG(CheckInt64AddOverflow(total_size, ws_size));
    ws_offsets.emplace_back(total_size);
    total_size += ws_size;
  }

  GELOGD("Total workspace size is %ld", total_size);
  StreamResource *stream_resource = SingleOpManager::GetInstance().GetResource(resource_id_, stream_);
  GE_CHECK_NOTNULL(stream_resource);
  auto ws_base = stream_resource->MallocMemory(kPurpose, static_cast<size_t>(total_size));
  if (ws_base == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to allocate memory of size: %ld", total_size);
    return MEMALLOC_FAILED;
  }
  GELOGD("Done allocating workspace memory successfully.");

  for (auto ws_offset : ws_offsets) {
    workspaces.emplace_back(ws_base + ws_offset);
  }

  return SUCCESS;
}

Status DynamicSingleOp::ExecuteTbeTask(const vector<GeTensorDesc> &input_desc, const vector<void *> &inputs,
                                       vector<GeTensorDesc> &output_desc, vector<void *> &outputs) {
  GE_CHK_STATUS_RET_NOLOG(op_task_->UpdateRunInfo(input_desc, output_desc));

  std::vector<void *> workspace_buffers;
  GE_CHK_STATUS_RET_NOLOG(AllocateWorkspaces(op_task_->GetWorkspaceSizes(), workspace_buffers));

  return op_task_->LaunchKernel(inputs, outputs, workspace_buffers, stream_);
}

Status DynamicSingleOp::ExecuteAsync(const vector<GeTensorDesc> &input_desc, const vector<DataBuffer> &input_buffers,
                                     vector<GeTensorDesc> &output_desc, vector<DataBuffer> &output_buffers) {
  GE_CHECK_NOTNULL(op_task_);
  GE_CHK_STATUS_RET_NOLOG(ValidateParams(input_desc, input_buffers, output_desc, output_buffers));
  std::lock_guard<std::mutex> lk(*stream_mutex_);

  std::vector<void *> inputs;
  std::vector<void *> outputs;
  for (auto &buffer : input_buffers) {
    inputs.emplace_back(buffer.data);
  }
  for (auto &buffer : output_buffers) {
    outputs.emplace_back(buffer.data);
  }

  if (op_task_->GetOpTaskType() == OP_TASK_TBE) {
    return ExecuteTbeTask(input_desc, inputs, output_desc, outputs);
  } else if (op_task_->GetOpTaskType() == OP_TASK_AICPU || op_task_->GetOpTaskType() == OP_TASK_AICPUCC) {
    return op_task_->LaunchKernel(input_desc, inputs, output_desc, outputs, stream_);
  } else {
    GELOGE(UNSUPPORTED, "Only TBE_Task, AI_CPU_Task and AI_CPUCC_Task are supported, but got %u",
           op_task_->GetOpTaskType());
    return UNSUPPORTED;
  }
}

void DynamicSingleOp::SetSessionID(uint64_t session_id) { aicpu_session_id_ = session_id; }
}  // namespace ge
