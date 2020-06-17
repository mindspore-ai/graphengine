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

#include "graph/load/new_model_manager/data_dumper.h"

#include <ctime>
#include <map>
#include <utility>
#include <vector>

#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/anchor.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/utils/attr_utils.h"
#include "proto/ge_ir.pb.h"
#include "proto/op_mapping_info.pb.h"
#include "runtime/mem.h"

namespace {
const uint32_t kAicpuLoadFlag = 1;
const uint32_t kAicpuUnloadFlag = 0;
const uint32_t kTimeBufferLen = 80;
const char *const kDumpOutput = "output";
const char *const kDumpInput = "input";
const char *const kDumpAll = "all";
}  // namespace

static int32_t GetIrDataType(ge::DataType data_type) {
  static const std::map<ge::DataType, ge::proto::DataType> data_type_map = {
    {ge::DT_UNDEFINED, ge::proto::DT_UNDEFINED},
    {ge::DT_FLOAT, ge::proto::DT_FLOAT},
    {ge::DT_FLOAT16, ge::proto::DT_FLOAT16},
    {ge::DT_INT8, ge::proto::DT_INT8},
    {ge::DT_UINT8, ge::proto::DT_UINT8},
    {ge::DT_INT16, ge::proto::DT_INT16},
    {ge::DT_UINT16, ge::proto::DT_UINT16},
    {ge::DT_INT32, ge::proto::DT_INT32},
    {ge::DT_INT64, ge::proto::DT_INT64},
    {ge::DT_UINT32, ge::proto::DT_UINT32},
    {ge::DT_UINT64, ge::proto::DT_UINT64},
    {ge::DT_BOOL, ge::proto::DT_BOOL},
    {ge::DT_DOUBLE, ge::proto::DT_DOUBLE},
    {ge::DT_DUAL, ge::proto::DT_DUAL},
    {ge::DT_DUAL_SUB_INT8, ge::proto::DT_DUAL_SUB_INT8},
    {ge::DT_DUAL_SUB_UINT8, ge::proto::DT_DUAL_SUB_UINT8},
    {ge::DT_COMPLEX64, ge::proto::DT_COMPLEX64},
    {ge::DT_COMPLEX128, ge::proto::DT_COMPLEX128},
    {ge::DT_QINT8, ge::proto::DT_QINT8},
    {ge::DT_QINT16, ge::proto::DT_QINT16},
    {ge::DT_QINT32, ge::proto::DT_QINT32},
    {ge::DT_QUINT8, ge::proto::DT_QUINT8},
    {ge::DT_QUINT16, ge::proto::DT_QUINT16},
    {ge::DT_RESOURCE, ge::proto::DT_RESOURCE},
    {ge::DT_STRING_REF, ge::proto::DT_STRING_REF},
    {ge::DT_STRING, ge::proto::DT_STRING},
  };

  auto iter = data_type_map.find(data_type);
  if (iter == data_type_map.end()) {
    return static_cast<int32_t>(ge::proto::DT_UNDEFINED);
  }

  return static_cast<int32_t>(iter->second);
}

namespace ge {
DataDumper::~DataDumper() {
  ReleaseDevMem(&dev_mem_load_);
  ReleaseDevMem(&dev_mem_unload_);
}

void DataDumper::ReleaseDevMem(void **ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }

  if (*ptr != nullptr) {
    rtError_t rt_ret = rtFree(*ptr);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rtFree failed, ret: 0x%X", rt_ret);
    }

    *ptr = nullptr;
  }
}

void DataDumper::SetLoopAddr(void *global_step, void *loop_per_iter, void *loop_cond) {
  global_step_ = reinterpret_cast<uintptr_t>(global_step);
  loop_per_iter_ = reinterpret_cast<uintptr_t>(loop_per_iter);
  loop_cond_ = reinterpret_cast<uintptr_t>(loop_cond);
}

void DataDumper::SaveDumpInput(const std::shared_ptr<Node> &node) {
  if (node != nullptr) {
    auto input_op_desc = node->GetOpDesc();
    if (input_op_desc == nullptr) {
      GELOGE(PARAM_INVALID, "input op desc is null.");
      return;
    }

    for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (auto &dst_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        ge::NodePtr dst_node = dst_in_data_anchor->GetOwnerNode();
        auto op_desc = dst_node->GetOpDesc();
        if (op_desc == nullptr) {
          GELOGE(PARAM_INVALID, "input op desc is null.");
          return;
        }

        input_map_.insert(
          {op_desc->GetName(), {input_op_desc, dst_in_data_anchor->GetIdx(), out_data_anchor->GetIdx()}});
      }
    }
  }
}

void DataDumper::SaveEndGraphId(uint32_t task_id, uint32_t stream_id) {
  end_graph_task_id_ = task_id;
  end_graph_stream_id_ = stream_id;
}

void DataDumper::SaveDumpTask(uint32_t task_id, uint32_t stream_id, const std::shared_ptr<OpDesc> &op_desc,
                              uintptr_t args) {
  if (op_desc == nullptr) {
    GELOGE(PARAM_INVALID, "Opdesc is nullptr");
    return;
  }

  GELOGI("Save dump task %s, task id: %u, stream id: %u", op_desc->GetName().c_str(), task_id, stream_id);
  op_list_.push_back({task_id, stream_id, op_desc, args, true});

  for (auto iter = input_map_.equal_range(op_desc->GetName()); iter.first != iter.second; ++iter.first) {
    InnerInputMapping &inner_input_mapping = iter.first->second;
    auto &data_op = inner_input_mapping.data_op;
    if (data_op == nullptr) {
      GELOGE(PARAM_INVALID, "data_op is null.");
      return;
    }

    auto input_tensor = op_desc->GetInputDescPtr(inner_input_mapping.input_anchor_index);
    if (input_tensor == nullptr) {
      GELOGE(PARAM_INVALID, "input_tensor is null, index: %d, size: %zu.", inner_input_mapping.input_anchor_index,
             op_desc->GetInputsSize());
      return;
    }

    GELOGI("Save input dump task %s, id: %u.", data_op->GetName().c_str(), task_id);
    op_list_.push_back({task_id, stream_id, data_op, args, false, inner_input_mapping.input_anchor_index,
                        inner_input_mapping.output_anchor_index, input_tensor->GetShape().GetDims()});
  }
}

static void SetOpMappingLoopAddr(uintptr_t step_id, uintptr_t loop_per_iter, uintptr_t loop_cond,
                                 aicpu::dump::OpMappingInfo &op_mapping_info) {
  if (step_id != 0) {
    GELOGI("step_id exist.");
    op_mapping_info.set_step_id_addr(static_cast<uint64_t>(step_id));
  } else {
    GELOGI("step_id is null.");
  }

  if (loop_per_iter != 0) {
    GELOGI("loop_per_iter exist.");
    op_mapping_info.set_iterations_per_loop_addr(static_cast<uint64_t>(loop_per_iter));
  } else {
    GELOGI("loop_per_iter is null.");
  }

  if (loop_cond != 0) {
    GELOGI("loop_cond exist.");
    op_mapping_info.set_loop_cond_addr(static_cast<uint64_t>(loop_cond));
  } else {
    GELOGI("loop_cond is null.");
  }
}

static std::string GetCurrentTime() {
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);
  if (ptm == nullptr) {
    return "";
  }
  char buffer[kTimeBufferLen] = {0};
  // format: 20171122042550
  std::strftime(buffer, kTimeBufferLen, "%Y%m%d%H%M%S", ptm);
  return std::string(buffer);
}

Status DataDumper::DumpOutput(const InnerDumpInfo &inner_dump_info, aicpu::dump::Task &task) {
  GELOGI("Start dump output");
  if (inner_dump_info.is_task) {
    // tbe or aicpu op
    const auto &output_descs = inner_dump_info.op->GetAllOutputsDesc();
    const auto input_size = inner_dump_info.op->GetAllInputsDesc().size();
    const std::vector<void *> output_addrs = ModelUtils::GetOutputDataAddrs(runtime_param_, inner_dump_info.op, false);
    if (output_descs.size() != output_addrs.size()) {
      GELOGE(PARAM_INVALID, "Invalid output desc addrs size %zu, op %s has %zu output desc.", output_addrs.size(),
             inner_dump_info.op->GetName().c_str(), output_descs.size());
      return PARAM_INVALID;
    }

    for (size_t i = 0; i < output_descs.size(); ++i) {
      aicpu::dump::Output output;
      output.set_data_type(static_cast<int32_t>(GetIrDataType(output_descs.at(i).GetDataType())));
      output.set_format(static_cast<int32_t>(output_descs.at(i).GetFormat()));

      for (auto dim : output_descs.at(i).GetShape().GetDims()) {
        output.mutable_shape()->add_dim(dim);
      }

      std::string origin_name;
      int32_t origin_output_index = -1;
      (void)AttrUtils::GetStr(&output_descs.at(i), ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_name);
      (void)AttrUtils::GetInt(&output_descs.at(i), ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index);
      output.set_original_name(origin_name);
      output.set_original_output_index(origin_output_index);
      output.set_original_output_format(static_cast<int32_t>(output_descs.at(i).GetOriginFormat()));
      output.set_original_output_data_type(static_cast<int32_t>(output_descs.at(i).GetOriginDataType()));
      output.set_address(static_cast<uint64_t>(inner_dump_info.args + (i + input_size) * sizeof(void *)));

      task.mutable_output()->Add(std::move(output));
    }
    return SUCCESS;
  }

  // else data, const or variable op
  aicpu::dump::Output output;
  auto output_tensor = inner_dump_info.op->GetOutputDescPtr(inner_dump_info.output_anchor_index);
  const std::vector<void *> output_addrs = ModelUtils::GetOutputDataAddrs(runtime_param_, inner_dump_info.op, false);
  if (output_tensor == nullptr) {
    GELOGE(PARAM_INVALID, "output_tensor is null, index: %d, size: %zu.", inner_dump_info.output_anchor_index,
           inner_dump_info.op->GetOutputsSize());
    return PARAM_INVALID;
  }

  output.set_data_type(static_cast<int32_t>(GetIrDataType(output_tensor->GetDataType())));
  output.set_format(static_cast<int32_t>(output_tensor->GetFormat()));

  for (auto dim : inner_dump_info.dims) {
    output.mutable_shape()->add_dim(dim);
  }

  std::string origin_name;
  int32_t origin_output_index = -1;
  (void)AttrUtils::GetStr(output_tensor, ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_name);
  (void)AttrUtils::GetInt(output_tensor, ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index);
  output.set_original_name(origin_name);
  output.set_original_output_index(origin_output_index);
  output.set_original_output_format(static_cast<int32_t>(output_tensor->GetOriginFormat()));
  output.set_original_output_data_type(static_cast<int32_t>(output_tensor->GetOriginDataType()));
  // due to lhisi virtual addr bug, cannot use args now
  if (inner_dump_info.output_anchor_index >= static_cast<int>(output_addrs.size())) {
    GELOGE(FAILED, "Index is out of range.");
    return FAILED;
  }
  auto data_addr = inner_dump_info.args + sizeof(void *) * static_cast<uint32_t>(inner_dump_info.input_anchor_index);
  output.set_address(static_cast<uint64_t>(data_addr));

  task.mutable_output()->Add(std::move(output));

  return SUCCESS;
}

Status DataDumper::DumpInput(const InnerDumpInfo &inner_dump_info, aicpu::dump::Task &task) {
  GELOGI("Start dump input");
  const auto &input_descs = inner_dump_info.op->GetAllInputsDesc();
  const std::vector<void *> input_addrs = ModelUtils::GetInputDataAddrs(runtime_param_, inner_dump_info.op, false);
  if (input_descs.size() != input_addrs.size()) {
    GELOGE(PARAM_INVALID, "Invalid input desc addrs size %zu, op %s has %zu input desc.", input_addrs.size(),
           inner_dump_info.op->GetName().c_str(), input_descs.size());
    return PARAM_INVALID;
  }

  for (size_t i = 0; i < input_descs.size(); ++i) {
    aicpu::dump::Input input;
    input.set_data_type(static_cast<int32_t>(GetIrDataType(input_descs.at(i).GetDataType())));
    input.set_format(static_cast<int32_t>(input_descs.at(i).GetFormat()));

    for (auto dim : input_descs.at(i).GetShape().GetDims()) {
      input.mutable_shape()->add_dim(dim);
    }

    input.set_address(static_cast<uint64_t>(inner_dump_info.args + sizeof(void *) * i));
    task.mutable_input()->Add(std::move(input));
  }
  return SUCCESS;
}

Status DataDumper::ExecuteLoadDumpInfo(aicpu::dump::OpMappingInfo &op_mapping_info) {
  std::string proto_str;
  size_t proto_size = op_mapping_info.ByteSizeLong();
  bool ret = op_mapping_info.SerializeToString(&proto_str);
  if (!ret || proto_size == 0) {
    GELOGE(FAILED, "Protobuf SerializeToString failed, proto size %zu.", proto_size);
    return FAILED;
  }

  if (dev_mem_load_ != nullptr) {
    GELOGW("dev_mem_load_ has been used.");
    ReleaseDevMem(&dev_mem_load_);
  }

  rtError_t rt_ret = rtMalloc(&dev_mem_load_, proto_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMalloc failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "load dump information.", proto_size)

  rt_ret = rtMemcpy(dev_mem_load_, proto_size, proto_str.c_str(), proto_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMemcpy failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtDatadumpInfoLoad(dev_mem_load_, proto_size);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtDatadumpInfoLoad failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  load_flag_ = true;
  GELOGI("LoadDumpInfo success, proto size: %zu.", proto_size);
  return SUCCESS;
}

Status DataDumper::ExecuteUnLoadDumpInfo(aicpu::dump::OpMappingInfo &op_mapping_info) {
  std::string proto_str;
  size_t proto_size = op_mapping_info.ByteSizeLong();
  bool ret = op_mapping_info.SerializeToString(&proto_str);
  if (!ret || proto_size == 0) {
    GELOGE(FAILED, "Protobuf SerializeToString failed, proto size %zu.", proto_size);
    return FAILED;
  }

  if (dev_mem_unload_ != nullptr) {
    GELOGW("dev_mem_unload_ has been used.");
    ReleaseDevMem(&dev_mem_unload_);
  }

  rtError_t rt_ret = rtMalloc(&dev_mem_unload_, proto_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMalloc failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "unload dump information.", proto_size)

  rt_ret = rtMemcpy(dev_mem_unload_, proto_size, proto_str.c_str(), proto_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMemcpy failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtDatadumpInfoLoad(dev_mem_unload_, proto_size);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtDatadumpInfoLoad failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  load_flag_ = false;
  GELOGI("UnloadDumpInfo success, proto size: %zu.", proto_size);
  return SUCCESS;
}
Status DataDumper::LoadDumpInfo() {
  PrintCheckLog();

  if (op_list_.empty()) {
    return SUCCESS;
  }

  aicpu::dump::OpMappingInfo op_mapping_info;
  std::string time_now = GetCurrentTime();
  GELOGI("Time is %s now", time_now.c_str());
  op_mapping_info.set_dump_path(PropertiesManager::Instance().GetDumpOutputPath() + time_now + "/" +
                                std::to_string(device_id_) + "/");
  op_mapping_info.set_model_name(model_name_);
  op_mapping_info.set_model_id(model_id_);
  op_mapping_info.set_flag(kAicpuLoadFlag);
  op_mapping_info.set_dump_step(PropertiesManager::Instance().GetDumpStep());
  SetOpMappingLoopAddr(global_step_, loop_per_iter_, loop_cond_, op_mapping_info);
  GELOGD("Dump step in load dump info is %s", PropertiesManager::Instance().GetDumpStep().c_str());

  for (const auto &op_iter : op_list_) {
    aicpu::dump::Task task;
    auto op_desc = op_iter.op;
    task.set_end_graph(false);
    task.set_task_id(op_iter.task_id);
    task.set_stream_id(op_iter.stream_id);
    task.mutable_op()->set_op_name(op_desc->GetName());
    task.mutable_op()->set_op_type(op_desc->GetType());

    if (PropertiesManager::Instance().GetDumpMode() == kDumpOutput) {
      if (DumpOutput(op_iter, task) != SUCCESS) {
        GELOGE(FAILED, "Dump output failed");
        return FAILED;
      }
      op_mapping_info.mutable_task()->Add(std::move(task));
      continue;
    }
    if (PropertiesManager::Instance().GetDumpMode() == kDumpInput) {
      if (op_iter.is_task) {
        if (DumpInput(op_iter, task) != SUCCESS) {
          GELOGE(FAILED, "Dump input failed");
          return FAILED;
        }
      }
      op_mapping_info.mutable_task()->Add(std::move(task));
      continue;
    }
    if (PropertiesManager::Instance().GetDumpMode() == kDumpAll) {
      auto ret = DumpOutput(op_iter, task);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "Dump output failed when in dumping all");
        return FAILED;
      }
      if (op_iter.is_task) {
        ret = DumpInput(op_iter, task);
        if (ret != SUCCESS) {
          GELOGE(FAILED, "Dump input failed when in dumping all");
          return FAILED;
        }
      }
      op_mapping_info.mutable_task()->Add(std::move(task));
      continue;
    }
  }

  SetEndGraphIdToAicpu(end_graph_task_id_, end_graph_stream_id_, op_mapping_info);

  auto ret = ExecuteLoadDumpInfo(op_mapping_info);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Execute load dump info failed");
    return FAILED;
  }
  return SUCCESS;
}

void DataDumper::SetEndGraphIdToAicpu(uint32_t task_id, uint32_t stream_id,
                                      aicpu::dump::OpMappingInfo &op_mapping_info) {
  if (PropertiesManager::Instance().GetDumpMode() == kDumpOutput ||
      PropertiesManager::Instance().GetDumpMode() == kDumpInput ||
      PropertiesManager::Instance().GetDumpMode() == kDumpAll) {
    GELOGI("add end_graph_info to aicpu, task_id is %u, stream_id is %u", end_graph_task_id_, end_graph_stream_id_);
    aicpu::dump::Task task;
    task.set_end_graph(true);
    task.set_task_id(end_graph_task_id_);
    task.set_stream_id(end_graph_stream_id_);
    task.mutable_op()->set_op_name(NODE_NAME_END_GRAPH);
    task.mutable_op()->set_op_type(ENDGRAPH);
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
}

Status DataDumper::UnloadDumpInfo() {
  if (!load_flag_) {
    GELOGI("No need to UnloadDumpInfo.");
    load_flag_ = false;
    return SUCCESS;
  }

  GELOGI("UnloadDumpInfo start.");
  aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.set_model_id(model_id_);
  op_mapping_info.set_flag(kAicpuUnloadFlag);

  for (const auto &op_iter : op_list_) {
    aicpu::dump::Task task;
    task.set_task_id(op_iter.task_id);
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  auto ret = ExecuteUnLoadDumpInfo(op_mapping_info);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Execute unload dump info failed");
    return FAILED;
  }
  return SUCCESS;
}

void DataDumper::PrintCheckLog() {
  std::set<std::string> model_list = PropertiesManager::Instance().GetAllDumpModel();
  if (model_list.empty()) {
    GELOGI("No model need dump.");
    return;
  }

  GELOGI("%zu op need dump in %s.", op_list_.size(), model_name_.c_str());
  if (model_list.find(ge::DUMP_ALL_MODEL) == model_list.end()) {
    if (model_list.find(model_name_) == model_list.end()) {
      std::string model_list_str;
      for (auto &model : model_list) {
        model_list_str += "[" + model + "].";
      }

      GELOGW("Model %s not be set to dump, dump list: %s", model_name_.c_str(), model_list_str.c_str());
      return;
    }
  }

  std::set<std::string> config_dump_op_list = PropertiesManager::Instance().GetDumpPropertyValue(model_name_);
  std::set<std::string> dump_op_list;
  for (auto &inner_dump_info : op_list_) {
    // oplist value OpDescPtr is not nullptr
    dump_op_list.insert(inner_dump_info.op->GetName());
  }

  for (auto &dump_op : config_dump_op_list) {
    if (dump_op_list.find(dump_op) == dump_op_list.end()) {
      GELOGW("Op %s set to dump but not exist in model %s or not a valid op.", dump_op.c_str(), model_name_.c_str());
    }
  }
}
}  // namespace ge
