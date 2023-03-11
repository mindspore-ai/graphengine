/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "common/plugin/datatype_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "graph/anchor.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"
#include "proto/ge_ir.pb.h"
#include "proto/op_mapping.pb.h"
#include "runtime/rt.h"
#include "aicpu/common/aicpu_task_struct.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

namespace {
const uint32_t kAiCpuLoadFlag = 1U;
const ge::char_t *const kDumpModeOutput = "output";
const ge::char_t *const kDumpModeInput = "input";
const ge::char_t *const kDumpModeAll = "all";
const ge::char_t *const kDumpKernelsDumpOp = "DumpDataInfo";
constexpr uint32_t k16BitsMask = 0x0000FFFFU;  // 16 bits, 1111,1111,1111,1111
const ge::char_t *const kDumpDataDefaultValue = "stats";
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

void DumpOp::SetLoopAddr(const uintptr_t global_step, const uintptr_t loop_per_iter, const uintptr_t loop_cond) {
  global_step_ = global_step;
  loop_per_iter_ = loop_per_iter;
  loop_cond_ = loop_cond;
}

void DumpOp::SetDynamicModelInfo(const std::string &dynamic_model_name, const std::string &dynamic_om_name,
                                 const uint32_t dynamic_model_id) {
  dynamic_model_name_ = dynamic_model_name;
  dynamic_om_name_ = dynamic_om_name;
  dynamic_model_id_ = dynamic_model_id;
}

static void SetLoopAddrToOpMapping(const uintptr_t step_id, const uintptr_t loop_per_iter, const uintptr_t loop_cond,
                                   toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  if (step_id != 0U) {
    GELOGI("Exists step_id.");
    op_mapping_info.set_step_id_addr(static_cast<uint64_t>(step_id));
  } else {
    GELOGI("step_id is null.");
  }

  if (loop_per_iter != 0U) {
    GELOGI("Exists loop_per_iter.");
    op_mapping_info.set_iterations_per_loop_addr(static_cast<uint64_t>(loop_per_iter));
  } else {
    GELOGI("loop_per_iter is null.");
  }

  if (loop_cond != 0U) {
    GELOGI("Exists loop_cond.");
    op_mapping_info.set_loop_cond_addr(static_cast<uint64_t>(loop_cond));
  } else {
    GELOGI("loop_cond is null.");
  }
}

void DumpOp::DumpWorkspace(toolkit::aicpu::dump::Task &task) {
  std::vector<int64_t> space_type;
  bool has_space_type = ge::AttrUtils::GetListInt(op_desc_, TVM_ATTR_NAME_WORKSPACE_TYPE, space_type);
  bool has_memory_log = false;
  if (has_space_type) {
    auto result = std::find(space_type.begin(), space_type.end(), RT_MEMORY_CUST_AICPU_LOG);
    if (result != space_type.end()) {
      has_memory_log = true;
    }
  }
  const auto v_workspace_size = op_desc_->GetWorkspaceBytes();
  for (size_t i = 0U; has_memory_log && i < v_workspace_size.size() && i < space_addrs_.size(); ++i) {
    GELOGI("workspace_info addr=:%lu  %zu", space_addrs_[i], v_workspace_size[i]);
    toolkit::aicpu::dump::Workspace space;
    space.set_type(toolkit::aicpu::dump::Workspace::LOG);
    space.set_size(static_cast<uint64_t>(v_workspace_size[i]));
    space.set_data_addr(space_addrs_[i]);
    task.mutable_space()->Add(std::move(space));
  }
}

Status DumpOp::DumpOutput(toolkit::aicpu::dump::Task &task) {
  GELOGI("Start dump output in Launch dump op");
  const auto &output_descs = op_desc_->GetAllOutputsDesc();
  for (size_t i = 0U; i < output_descs.size(); ++i) {
    if ((i >= output_addrs_.size()) || (output_addrs_[i] == reinterpret_cast<uintptr_t>(nullptr))) {
      GELOGW("[Dumper] Node name %s, i is %zu, output addrs size is %zu", op_desc_->GetName().c_str(), i,
             output_addrs_.size());
      continue;
    }
    toolkit::aicpu::dump::Output output;
    output.set_data_type(static_cast<int32_t>(DataTypeUtil::GetIrDataType(output_descs.at(i).GetDataType())));
    output.set_format(static_cast<int32_t>(output_descs.at(i).GetFormat()));
    for (const auto dim : output_descs.at(i).GetShape().GetDims()) {
      output.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    for (const auto dim : output_descs.at(i).GetOriginShape().GetDims()) {
      output.mutable_origin_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    int64_t output_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(output_descs.at(i), output_size) != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TensorSize]Failed, output %zu, node %s(%s),",
             i, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      REPORT_CALL_ERROR("E19999", "Get output %zu tensor size of node %s(%s) failed",
                        i, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    GELOGD("Get output size in lanch dump op is %ld", output_size);
    output.set_size(static_cast<uint64_t>(output_size));
    GELOGD("[Dumper] Node [%s] output %zu addr is %p.", op_desc_->GetName().c_str(), i,
           reinterpret_cast<void *>(output_addrs_[i]));
    output.set_address(static_cast<uint64_t>(output_addrs_[i]));
    bool no_tiling_mem_type = false;
    (void)AttrUtils::GetBool(output_descs.at(i), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, no_tiling_mem_type);
    output.set_addr_type(no_tiling_mem_type ?
        toolkit::aicpu::dump::AddressType::NOTILING_ADDR : toolkit::aicpu::dump::AddressType::TRADITIONAL_ADDR);
    task.mutable_output()->Add(std::move(output));
  }
  return SUCCESS;
}

Status DumpOp::DumpInput(toolkit::aicpu::dump::Task &task) {
  GELOGI("Start dump input in launch dump op");
  const auto &input_descs = op_desc_->GetAllInputsDesc();
  auto idx = 0UL;
  for (size_t i = 0U; i < input_descs.size(); ++i) {
    if (input_descs.at(i).GetShape().IsUnknownShape()) {
      continue;
    }
    if ((idx >= input_addrs_.size()) || (input_addrs_[idx] == reinterpret_cast<uintptr_t>(nullptr))) {
      GELOGW("[Dumper] Node name %s, idx is %zu, input addrs size is %zu", op_desc_->GetName().c_str(), idx,
             input_addrs_.size());
      ++idx;
      continue;
    }
    toolkit::aicpu::dump::Input input;
    input.set_data_type(static_cast<int32_t>(DataTypeUtil::GetIrDataType(input_descs.at(i).GetDataType())));
    input.set_format(static_cast<int32_t>(input_descs.at(i).GetFormat()));

    for (const auto dim : input_descs.at(i).GetShape().GetDims()) {
      input.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    for (const auto dim : input_descs.at(i).GetOriginShape().GetDims()) {
      input.mutable_origin_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    int64_t input_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(input_descs.at(i), input_size) != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TensorSize]Failed, input %zu, node %s(%s)",
             i, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      REPORT_CALL_ERROR("E19999", "Get input %zu tensor size of node %s(%s) failed",
                        i, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    GELOGD("Get input size in launch dump op is %ld.", input_size);
    input.set_size(static_cast<uint64_t>(input_size));
    GELOGD("[Dumper] Node [%s] input %zu addr is %p.", op_desc_->GetName().c_str(), idx,
           reinterpret_cast<void *>(input_addrs_[idx]));
    input.set_address(static_cast<uint64_t>(input_addrs_[idx]));
    bool no_tiling_mem_type = false;
    (void)AttrUtils::GetBool(input_descs.at(i), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, no_tiling_mem_type);
    input.set_addr_type(no_tiling_mem_type ?
        toolkit::aicpu::dump::AddressType::NOTILING_ADDR : toolkit::aicpu::dump::AddressType::TRADITIONAL_ADDR);
    task.mutable_input()->Add(std::move(input));
    ++idx;
  }
  return SUCCESS;
}

void DumpOp::SetDumpInfo(const DumpProperties &dump_properties, const OpDescPtr &op_desc,
                         const std::vector<uintptr_t> &input_addrs, const std::vector<uintptr_t> &output_addrs,
                         const rtStream_t stream) {
  dump_properties_ = dump_properties;
  op_desc_ = op_desc;
  input_addrs_ = input_addrs;
  output_addrs_ = output_addrs;
  stream_ = stream;
}

Status DumpOp::ExecutorDumpOp(const toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  std::string proto_msg;
  const size_t proto_size = op_mapping_info.ByteSizeLong();
  const bool ret = op_mapping_info.SerializeToString(&proto_msg);
  if ((!ret) || (proto_size == 0U)) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Serialize][Protobuf]Failed, proto_size is %zu",
           proto_size);
    REPORT_CALL_ERROR("E19999", "[Serialize][Protobuf]Failed, proto_size is %zu", proto_size);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  rtError_t rt_ret = rtMalloc(&proto_dev_mem_, proto_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][rtMalloc]Failed, ret: 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtMemcpy(proto_dev_mem_, proto_size, proto_msg.c_str(), proto_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][rtMemcpy]Failed, ret: 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtMalloc(&proto_size_dev_mem_, sizeof(size_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][rtMalloc]Failed, ret: 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtMemcpy(proto_size_dev_mem_, sizeof(size_t), &proto_size, sizeof(size_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][rtMemcpy]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  constexpr uint32_t io_addr_num = 2U;
  constexpr uint32_t args_size = sizeof(aicpu::AicpuParamHead) + (io_addr_num * sizeof(uint64_t));
  uint8_t args[args_size] = {};
  size_t args_pos = 0U;
  aicpu::AicpuParamHead &param_head = *(static_cast<aicpu::AicpuParamHead *>(static_cast<void *>(&args[args_pos])));
  args_pos += sizeof(aicpu::AicpuParamHead);
  param_head.length = args_size;
  param_head.ioAddrNum = io_addr_num;
  *(static_cast<uint64_t *>(static_cast<void *>(&args[args_pos]))) = PtrToValue(proto_dev_mem_);
  args_pos += sizeof(uint64_t);
  *(static_cast<uint64_t *>(static_cast<void *>(&args[args_pos]))) = PtrToValue(proto_size_dev_mem_);
  rt_ret = rtCpuKernelLaunch(nullptr, kDumpKernelsDumpOp,
                             1U,  // blockDim default 1
                             &args[0], args_size,
                             nullptr,  // no need smDesc
                             stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][rtCpuKernelLaunch]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtCpuKernelLaunch failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGI("Kernel launch dump op %s success", op_desc_->GetName().c_str());
  return SUCCESS;
}

Status DumpOp::SetDumpModelName(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  if (dynamic_model_name_.empty() && dynamic_om_name_.empty()) {
    GELOGI("Single op dump, no need set model name");
    return SUCCESS;
  }
  std::set<std::string> model_list = dump_properties_.GetAllDumpModel();
  const bool not_find_by_omname = model_list.find(dynamic_om_name_) == model_list.end();
  const bool not_find_by_modelname = model_list.find(dynamic_model_name_) == model_list.cend();
  const std::string dump_model_name = not_find_by_omname ? dynamic_model_name_ : dynamic_om_name_;
  if ((!dump_model_name.empty()) && (!dump_properties_.GetEnableDumpDebug().empty())) {
    GELOGI("Debug dump model name is %s", dump_model_name.c_str());
    op_mapping_info.set_model_name(dump_model_name);
    return SUCCESS;
  }
  if (model_list.find(DUMP_ALL_MODEL) == model_list.end()) {
    if (not_find_by_omname && not_find_by_modelname) {
      std::string model_list_str;
      for (auto &model : model_list) {
        model_list_str += "[" + model + "].";
      }
      GELOGW("Model %s will not be set to dump, dump list: %s", dump_model_name.c_str(), model_list_str.c_str());
      return FAILED;
    }
  }
  if ((!dump_model_name.empty()) && dump_properties_.IsDumpOpen()) {
    GELOGI("Dump model name is %s", dump_model_name.c_str());
    op_mapping_info.set_model_name(dump_model_name);
  }
  return SUCCESS;
}

Status DumpOp::LaunchDumpOp(bool is_single_op_dump) {
  GELOGI("Start to launch dump op %s, is single op dump %d.", op_desc_->GetName().c_str(),
         static_cast<int32_t>(is_single_op_dump));
  int32_t device_id = 0;
  rtError_t rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][rtGetDevice]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "[Call][rtGetDevice]Failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  if (device_id < 0) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][DeviceId]Failed, device_id %d", device_id);
    REPORT_INNER_ERROR("E19999", "Check device_id %d failed", device_id);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  const auto dump_path = dump_properties_.GetDumpPath() + std::to_string(device_id) + "/";
  op_mapping_info.set_dump_path(dump_path);
  op_mapping_info.set_flag(kAiCpuLoadFlag);
  op_mapping_info.set_dump_step(dump_properties_.GetDumpStep());
  op_mapping_info.set_model_id(dynamic_model_id_);
  const auto dump_data = (dump_properties_.GetDumpData() == kDumpDataDefaultValue)
                             ? toolkit::aicpu::dump::DumpData::STATS_DUMP_DATA
                             : toolkit::aicpu::dump::DumpData::TENSOR_DUMP_DATA;
  op_mapping_info.set_dump_data(dump_data);

  if (!is_single_op_dump && (SetDumpModelName(op_mapping_info) != SUCCESS)) {
    return SUCCESS;
  }
  SetLoopAddrToOpMapping(global_step_, loop_per_iter_, loop_cond_, op_mapping_info);
  GELOGI("Dump step is %s ,dump path is %s in Launch dump op", dump_properties_.GetDumpStep().c_str(),
         dump_path.c_str());
  if (task_id_ == 0U || stream_id_ == 0U) {
    GE_CHK_RT(rtGetTaskIdAndStreamID(&task_id_, &stream_id_));
  }
  const auto task_id = task_id_ & k16BitsMask;
  toolkit::aicpu::dump::Task task;
  GELOGW("Task id is %u, stream id is %u", task_id, stream_id_);
  task.set_task_id(task_id);
  task.set_stream_id(stream_id_);
  task.mutable_op()->set_op_name(op_desc_->GetName());
  task.mutable_op()->set_op_type(op_desc_->GetType());
  if (dump_properties_.GetDumpMode() == kDumpModeOutput) {
    const auto ret = DumpOutput(task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Output]Failed, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_CALL_ERROR("E19999", "Dump Output failed, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  if (dump_properties_.GetDumpMode() == kDumpModeInput) {
    const auto ret = DumpInput(task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Input]Failed, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_CALL_ERROR("E19999", "Dump Input failed, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  if ((dump_properties_.GetDumpMode() == kDumpModeAll) || dump_properties_.IsOpDebugOpen()) {
    DumpWorkspace(task);
    auto ret = DumpOutput(task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Output]Failed when in dumping all, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_CALL_ERROR("E19999", "Dump Output failed when in dumping all, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
    ret = DumpInput(task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Input]Failed when in dumping all, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_CALL_ERROR("E19999", "Dump Input failed when in dumping all, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  const auto ret = ExecutorDumpOp(op_mapping_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Dump][Op]Failed, ret 0x%X", ret);
    REPORT_CALL_ERROR("E19999", "Executor dump op failed, ret 0x%X", ret);
    return ret;
  }
  GELOGI("Dump %s success", op_desc_->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge
