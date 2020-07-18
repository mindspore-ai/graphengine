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
#include "common/profiling/profiling_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "runtime/mem.h"

namespace ge {
namespace {
const size_t kDataMemAlignSize = 32;

size_t GetAlignedSize(uint32_t size) {
  size_t aligned_size = (size + 2 * kDataMemAlignSize - 1) / kDataMemAlignSize * kDataMemAlignSize;
  return aligned_size;
}
}  // namespace
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY SingleOp::~SingleOp() {
  for (auto task : tasks_) {
    delete task;
    task = nullptr;
  }
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
    GELOGI("Input [%zu], aligned_size:%zu, inputs.length:%u, input_sizes_:%u", i, aligned_size, inputs[i].length,
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
    GELOGI("Output [%zu], aligned_size:%zu, outputs.length:%u, output_sizes_:%u", i, aligned_size, outputs[i].length,
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
  if (use_physical_addr_) {
    for (auto &input : inputs) {
      auto *addr = reinterpret_cast<uint8_t *>(input.data);
      args_[arg_index++] = reinterpret_cast<uintptr_t>(addr);
    }

    for (auto &output : outputs) {
      auto *addr = reinterpret_cast<uint8_t *>(output.data);
      args_[arg_index++] = reinterpret_cast<uintptr_t>(addr);
    }
  } else {
    for (auto &input : inputs) {
      args_[arg_index++] = reinterpret_cast<uintptr_t>(input.data);
    }

    for (auto &output : outputs) {
      args_[arg_index++] = reinterpret_cast<uintptr_t>(output.data);
    }
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
      AiCpuTask *task_aicpu = dynamic_cast<AiCpuTask *>(task);
      GE_CHECK_NOTNULL(task_aicpu);
      auto *dst_io_addr = const_cast<uintptr_t *>(reinterpret_cast<const uintptr_t *>(task_aicpu->GetIOAddr()));
      GE_CHECK_NOTNULL(dst_io_addr);
      auto rt_ret = rtMemcpyAsync(dst_io_addr, sizeof(uint64_t) * args_.size(), &args_[0],
                                  sizeof(uint64_t) * args_.size(), RT_MEMCPY_HOST_TO_DEVICE_EX, stream_);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "rtMemcpyAsync addresses failed, ret = %d", rt_ret);
        return RT_FAILED;
      }
    } else if (task->GetOpTaskType() == OP_TASK_AICPUCC) {
      GELOGD("Update aicpu_CC task args");
      AiCpuCCTask *task_aicpu_cc = dynamic_cast<AiCpuCCTask *>(task);
      GE_CHECK_NOTNULL(task_aicpu_cc);
      const uintptr_t *task_io_addr = reinterpret_cast<const uintptr_t *>(task_aicpu_cc->GetIOAddr());
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

  ret = UpdateArgs(inputs, outputs);
  if (ret != SUCCESS) {
    return ret;
  }

  for (auto &task : tasks_) {
    ret = task->LaunchKernel(stream_);
    if (ret != SUCCESS) {
      return ret;
    }
  }

  return ret;
}

void SingleOp::SetStream(rtStream_t stream) { stream_ = stream; }
}  // namespace ge
