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

#include "common/dump/dump_op.h"

#include "common/dump/dump_manager.h"
#include "common/ge/datatype_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/anchor.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"
#include "proto/ge_ir.pb.h"
#include "proto/op_mapping_info.pb.h"
#include "runtime/mem.h"
#include "aicpu/common/aicpu_task_struct.h"

namespace {
const uint32_t kAicpuLoadFlag = 1;
const char *const kDumpOutput = "output";
const char *const kDumpInput = "input";
const char *const kDumpAll = "all";
const char *const kDumpKernelsDumpOp = "DumpDataInfo";
}  // namespace

namespace ge {
DumpOp::~DumpOp() {
  if (proto_dev_mem_ != nullptr) {
    (void)rtFree(proto_dev_mem_);
  }
  if (proto_size_dev_mem_ != nullptr) {
    (void)rtFree(proto_size_dev_mem_);
  }
  proto_dev_mem_ = nullptr;
  proto_size_dev_mem_ = nullptr;
}

void DumpOp::SetLoopAddr(void *global_step, void *loop_per_iter, void *loop_cond) {
  global_step_ = reinterpret_cast<uintptr_t>(global_step);
  loop_per_iter_ = reinterpret_cast<uintptr_t>(loop_per_iter);
  loop_cond_ = reinterpret_cast<uintptr_t>(loop_cond);
}

void DumpOp::SetDynamicModelInfo(const string &dynamic_model_name, uint32_t dynamic_model_id) {
  dynamic_model_name_ = dynamic_model_name;
  dynamic_model_id_ = dynamic_model_id;
}

static void SetOpMappingLoopAddr(uintptr_t step_id, uintptr_t loop_per_iter, uintptr_t loop_cond,
                                 aicpu::dump::OpMappingInfo &op_mapping_info) {
  if (step_id != 0) {
    GELOGI("step_id exists.");
    op_mapping_info.set_step_id_addr(static_cast<uint64_t>(step_id));
  } else {
    GELOGI("step_id is null.");
  }

  if (loop_per_iter != 0) {
    GELOGI("loop_per_iter exists.");
    op_mapping_info.set_iterations_per_loop_addr(static_cast<uint64_t>(loop_per_iter));
  } else {
    GELOGI("loop_per_iter is null.");
  }

  if (loop_cond != 0) {
    GELOGI("loop_cond exists.");
    op_mapping_info.set_loop_cond_addr(static_cast<uint64_t>(loop_cond));
  } else {
    GELOGI("loop_cond is null.");
  }
}

Status DumpOp::DumpOutput(aicpu::dump::Task &task) {
  GELOGI("Start dump output in Launch dump op");
  const auto &output_descs = op_desc_->GetAllOutputsDesc();
  for (size_t i = 0; i < output_descs.size(); ++i) {
    aicpu::dump::Output output;
    output.set_data_type(static_cast<int32_t>(DataTypeUtil::GetIrDataType(output_descs.at(i).GetDataType())));
    output.set_format(static_cast<int32_t>(output_descs.at(i).GetFormat()));
    for (auto dim : output_descs.at(i).GetShape().GetDims()) {
      output.mutable_shape()->add_dim(dim);
    }
    int64_t output_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(output_descs.at(i), output_size) != SUCCESS) {
      GELOGE(PARAM_INVALID, "Get output size filed");
      return PARAM_INVALID;
    }
    GELOGD("Get output size in lanch dump op is %ld", output_size);
    output.set_size(output_size);
    output.set_address(static_cast<uint64_t>(output_addrs_[i]));
    task.mutable_output()->Add(std::move(output));
  }
  return SUCCESS;
}

Status DumpOp::DumpInput(aicpu::dump::Task &task) {
  GELOGI("Start dump input in Launch dump op");
  const auto &input_descs = op_desc_->GetAllInputsDesc();
  for (size_t i = 0; i < input_descs.size(); ++i) {
    aicpu::dump::Input input;
    input.set_data_type(static_cast<int32_t>(DataTypeUtil::GetIrDataType(input_descs.at(i).GetDataType())));
    input.set_format(static_cast<int32_t>(input_descs.at(i).GetFormat()));

    for (auto dim : input_descs.at(i).GetShape().GetDims()) {
      input.mutable_shape()->add_dim(dim);
    }
    int64_t input_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(input_descs.at(i), input_size) != SUCCESS) {
      GELOGE(PARAM_INVALID, "Get output size filed");
      return PARAM_INVALID;
    }
    GELOGD("Get input size in lanch dump op is %ld", input_size);
    input.set_size(input_size);
    input.set_address(static_cast<uint64_t>(input_addrs_[i]));
    task.mutable_input()->Add(std::move(input));
  }
  return SUCCESS;
}

void DumpOp::SetDumpInfo(const DumpProperties &dump_properties, const OpDescPtr &op_desc, vector<uintptr_t> input_addrs,
                         vector<uintptr_t> output_addrs, rtStream_t stream) {
  dump_properties_ = dump_properties;
  op_desc_ = op_desc;
  input_addrs_ = input_addrs;
  output_addrs_ = output_addrs;
  stream_ = stream;
}

Status DumpOp::ExecutorDumpOp(aicpu::dump::OpMappingInfo &op_mapping_info) {
  std::string proto_msg;
  size_t proto_size = op_mapping_info.ByteSizeLong();
  bool ret = op_mapping_info.SerializeToString(&proto_msg);
  if (!ret || proto_size == 0) {
    GELOGE(FAILED, "Protobuf serialize failed,proto_size is %zu", proto_size);
    return FAILED;
  }

  rtError_t rt_ret = rtMalloc(&proto_dev_mem_, proto_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMalloc failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtMemcpy(proto_dev_mem_, proto_size, proto_msg.c_str(), proto_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMemcpy failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtMalloc(&proto_size_dev_mem_, sizeof(size_t), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMalloc failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  rt_ret = rtMemcpy(proto_size_dev_mem_, sizeof(size_t), &proto_size, sizeof(size_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMemcpy failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  constexpr int32_t ioAddrNum = 2;
  constexpr uint32_t argsSize = sizeof(aicpu::AicpuParamHead) + ioAddrNum * sizeof(uint64_t);
  char args[argsSize] = {0};
  auto paramHead = reinterpret_cast<aicpu::AicpuParamHead *>(args);
  paramHead->length = argsSize;
  paramHead->ioAddrNum = ioAddrNum;
  auto ioAddr = reinterpret_cast<uint64_t *>(args + sizeof(aicpu::AicpuParamHead));
  ioAddr[0] = reinterpret_cast<uintptr_t>(proto_dev_mem_);
  ioAddr[1] = reinterpret_cast<uintptr_t>(proto_size_dev_mem_);
  rt_ret = rtCpuKernelLaunch(nullptr, kDumpKernelsDumpOp,
                             1,  // blockDim default 1
                             args, argsSize,
                             nullptr,  // no need smDesc
                             stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtCpuKernelLaunch failed,rt_ret:0x%X", rt_ret);
    return rt_ret;
  }
  GELOGI("Kernel launch dump op success");
  return SUCCESS;
}

Status DumpOp::LaunchDumpOp() {
  GELOGI("Start to launch dump op %s", op_desc_->GetName().c_str());
  int32_t device_id = 0;
  rtError_t rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE || device_id < 0) {
    GELOGE(RT_FAILED, "Call rtGetDevice failed, ret = 0x%X, device_id = %d.", rt_ret, device_id);
    return RT_FAILED;
  }
  aicpu::dump::OpMappingInfo op_mapping_info;
  auto dump_path = dump_properties_.GetDumpPath() + std::to_string(device_id) + "/";
  op_mapping_info.set_dump_path(dump_path);
  op_mapping_info.set_flag(kAicpuLoadFlag);
  op_mapping_info.set_dump_step(dump_properties_.GetDumpStep());
  if (!dynamic_model_name_.empty()) {
    op_mapping_info.set_model_name(dynamic_model_name_);
    op_mapping_info.set_model_id(dynamic_model_id_);
  }
  SetOpMappingLoopAddr(global_step_, loop_per_iter_, loop_cond_, op_mapping_info);
  GELOGI("Dump step is %s ,dump path is %s ,in Launch dump op", dump_properties_.GetDumpStep().c_str(),
         dump_path.c_str());

  aicpu::dump::Task task;
  task.mutable_op()->set_op_name(op_desc_->GetName());
  task.mutable_op()->set_op_type(op_desc_->GetType());
  if (dump_properties_.GetDumpMode() == kDumpOutput) {
    if (DumpOutput(task) != SUCCESS) {
      GELOGE(FAILED, "Dump output failed");
      return FAILED;
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  if (dump_properties_.GetDumpMode() == kDumpInput) {
    if (DumpInput(task) != SUCCESS) {
      GELOGE(FAILED, "Dump input failed");
      return FAILED;
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  if (dump_properties_.GetDumpMode() == kDumpAll) {
    auto ret = DumpOutput(task);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Dump output failed when in dumping all");
      return FAILED;
    }
    ret = DumpInput(task);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Dump input failed when in dumping all");
      return FAILED;
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  auto ret = ExecutorDumpOp(op_mapping_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "Executor dump op failed");
    return ret;
  }
  return SUCCESS;
}
}  // namesapce ge
