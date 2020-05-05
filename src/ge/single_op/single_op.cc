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
    if (aligned_size < input_sizes_[i]) {
      GELOGE(PARAM_INVALID, "Input size mismatch. index = %zu, model expect %zu, but given %zu(after align)", i,
             input_sizes_[i], aligned_size);
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
    if (aligned_size < output_sizes_[i]) {
      GELOGE(PARAM_INVALID, "Output size mismatch. index = %zu, model expect %zu, but given %zu(after align)", i,
             output_sizes_[i], aligned_size);
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status SingleOp::UpdateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  size_t arg_index = 0;
  if (use_physical_addr_) {
    for (auto &input : inputs) {
      auto *addr = reinterpret_cast<uint8_t *>(input.data);
      size_t aligned_size = GetAlignedSize(input.length);
      auto ret = ModelUtils::ConvertVirtualAddressToPhysical(addr, aligned_size, addr);
      if (ret != SUCCESS) {
        GELOGE(ret, "ConvertVirtualAddressToPhysical failed. Arg index = %zu", arg_index);
        return ret;
      }
      args_[arg_index++] = reinterpret_cast<uintptr_t>(addr);
    }

    for (auto &output : outputs) {
      auto *addr = reinterpret_cast<uint8_t *>(output.data);
      size_t aligned_size = GetAlignedSize(output.length);
      auto ret = ModelUtils::ConvertVirtualAddressToPhysical(addr, aligned_size, addr);
      if (ret != SUCCESS) {
        GELOGE(ret, "ConvertVirtualAddressToPhysical failed. Arg index = %zu", arg_index);
        return ret;
      }
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
