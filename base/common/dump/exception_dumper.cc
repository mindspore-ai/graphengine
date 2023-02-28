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
#include "common/dump/exception_dumper.h"

#ifdef __GNUC__
#include <sys/types.h>
#include <unistd.h>
#endif

#include "mmpa/mmpa_api.h"
#include "common/plugin/datatype_util.h"
#include "common/debug/memory_dumper.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"
#include "proto/dump_task.pb.h"
#include "graph/ge_context.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/attr_utils.h"
#include "framework/common/util.h"
#include "common/plugin/ge_util.h"
#include "runtime/mem.h"
#include "exception_dumper.h"

namespace ge {
namespace {
const char_t *const kEnvCollectPath = "NPU_COLLECT_PATH";
constexpr const char_t *kExceptionDumpPath = "NPU_COLLECT_PATH_EXE";
const std::string kExtraPath = "/extra-info/data-dump/";
constexpr size_t kMaxOpDescInfoNum = 2048UL * 2048UL;

static uint64_t GetNowTime() {
  uint64_t ret = 0U;
  mmTimeval tv;
  if (mmGetTimeOfDay(&tv, nullptr) == 0) {
    ret = (static_cast<uint64_t>(tv.tv_sec) * 1000000UL) + static_cast<uint64_t>(tv.tv_usec);
  }

  return ret;
}

static void ReplaceStringElem(std::string &str) {
  (void)for_each(str.begin(), str.end(), [](ge::char_t &ch) {
    if ((ch == ' ') || (ch == '.') || (ch == '/') || (ch == '\\')) {
      ch = '_';
    }
  });
}

static void SetDumpData(const ge::OpDescInfo &op_desc_info, toolkit::dump::DumpData &dump_data) {
  dump_data.set_version("2.0");
  dump_data.set_dump_time(GetNowTime());
  dump_data.set_op_name(op_desc_info.op_name);
  for (size_t i = 0U; i < op_desc_info.input_format.size(); ++i) {
    toolkit::dump::OpInput input;
    input.set_data_type(
        static_cast<toolkit::dump::OutputDataType>(ge::DataTypeUtil::GetIrDataType(op_desc_info.input_data_type[i])));
    input.set_format(static_cast<toolkit::dump::OutputFormat>(op_desc_info.input_format[i]));
    for (const auto dim : op_desc_info.input_shape[i]) {
      input.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    input.set_size(static_cast<uint64_t>(op_desc_info.input_size[i]));
    GELOGI("[Set][DumpData] The input size exception is %ld", op_desc_info.input_size[i]);
    dump_data.mutable_input()->Add(std::move(input));
  }

  for (size_t j = 0U; j < op_desc_info.output_format.size(); ++j) {
    toolkit::dump::OpOutput output;
    output.set_data_type(
        static_cast<toolkit::dump::OutputDataType>(ge::DataTypeUtil::GetIrDataType(op_desc_info.output_data_type[j])));
    output.set_format(static_cast<toolkit::dump::OutputFormat>(op_desc_info.output_format[j]));
    for (const auto dim : op_desc_info.output_shape[j]) {
      output.mutable_shape()->add_dim(static_cast<uint64_t>(dim));
    }
    output.set_size(static_cast<uint64_t>(op_desc_info.output_size[j]));
    GELOGI("[Set][DumpData] The output size exception is %ld", op_desc_info.output_size[j]);
    dump_data.mutable_output()->Add(std::move(output));
  }
  if (op_desc_info.is_mem_log) {
    GELOGI("workspace_info size=: %zu  %zu", op_desc_info.workspace_bytes.size(),
           op_desc_info.space_addrs.size());
    for (size_t i = 0U; i < op_desc_info.workspace_bytes.size(); ++i) {
      toolkit::dump::Workspace space;
      GELOGI("workspace_info add to dump_data");
      space.set_size(static_cast<uint64_t>(op_desc_info.workspace_bytes[i]));
      space.set_type(toolkit::dump::Workspace::LOG);
      dump_data.mutable_space()->Add(std::move(space));
    }
  }
}
}  // namespace

ExceptionDumper::~ExceptionDumper() {}

void ExceptionDumper::SaveDumpOpInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id,
                                     const ExtraOpInfo &extra_op_info) {
  OpDescInfo op_desc_info;
  if (extra_op_info.has_memory_log) {
    op_desc_info.is_mem_log = true;
    op_desc_info.space_addrs = extra_op_info.space_addrs;
  }
  SaveOpDescInfo(op, task_id, stream_id, op_desc_info);
  op_desc_info.args = extra_op_info.args;
  op_desc_info.input_addrs = extra_op_info.input_addrs;
  op_desc_info.output_addrs = extra_op_info.output_addrs;
  op_desc_info.tiling_key = extra_op_info.tiling_key;
  op_desc_info.tiling_data = extra_op_info.tiling_data;
  op_desc_info.node_info = extra_op_info.node_info;
  const std::lock_guard<std::mutex> lock(mutex_);
  ++op_desc_info_idx_;
  if (op_desc_info_.size() < kMaxOpDescInfoNum) {
    op_desc_info_.emplace_back(std::move(op_desc_info));
  } else {
    op_desc_info_[op_desc_info_idx_ % kMaxOpDescInfoNum] = op_desc_info;
  }
}

void ExceptionDumper::SaveDumpOpInfo(const OpDescPtr &op, const std::vector<void *> input_addrs,
                                     const std::vector<void *> output_addrs, const uint32_t task_id,
                                     const uint32_t stream_id) {
  OpDescInfo op_desc_info;
  SaveOpDescInfo(op, task_id, stream_id, op_desc_info);
  op_desc_info.input_addrs = input_addrs;
  op_desc_info.output_addrs = output_addrs;
  op_desc_info_.emplace_back(std::move(op_desc_info));
}

void ExceptionDumper::SaveOpDescInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id,
                                     OpDescInfo &op_desc_info) const {
  if (op == nullptr) {
    GELOGW("[Save][OpExceptionInfo] op desc ptr is null.");
    return;
  }
  GELOGD("[Save][OpExceptionInfo] Start to save dump op [%s] info of task_id: %u, stream_id: %u", op->GetName().c_str(),
         task_id, stream_id);
  op_desc_info.op_name = op->GetName();
  op_desc_info.op_type = op->GetType();
  op_desc_info.task_id = task_id;
  op_desc_info.stream_id = stream_id;
  (void)AttrUtils::GetInt(op, ATTR_NAME_IMPLY_TYPE, op_desc_info.imply_type);
  (void)AttrUtils::GetInt(op, TVM_ATTR_NAME_BLOCKDIM, op_desc_info.block_dim);
  (void)AttrUtils::GetStr(op, op->GetName() + "_kernelname", op_desc_info.dev_func);
  (void)AttrUtils::GetStr(op, TVM_ATTR_NAME_MAGIC, op_desc_info.tvm_magic);

  op_desc_info.workspace_bytes = op->GetWorkspaceBytes();

  op_desc_info.op_file_path = op->TryGetExtAttr(ATTR_NAME_OP_FILE_PATH, std::string("./kernel_meta"));
  for (size_t i = 0U; i < op->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr input_tensor_desc = op->MutableInputDesc(static_cast<uint32_t>(i));
    if (input_tensor_desc == nullptr) {
      continue;
    }
    op_desc_info.input_format.emplace_back(input_tensor_desc->GetFormat());
    op_desc_info.input_shape.emplace_back(input_tensor_desc->GetShape().GetDims());
    op_desc_info.input_data_type.emplace_back(input_tensor_desc->GetDataType());
    int64_t input_size = 0;

    if (TensorUtils::GetTensorSizeInBytes(*input_tensor_desc, input_size) != SUCCESS) {
      GELOGW("[Save][OpExceptionInfo] Op [%s] get input size failed.", op->GetName().c_str());
      return;
    }
    GELOGD("[Save][OpExceptionInfo] Save dump op info, the input size is %ld", input_size);
    op_desc_info.input_size.emplace_back(input_size);
  }
  for (size_t j = 0U; j < op->GetOutputsSize(); ++j) {
    const GeTensorDescPtr output_tensor_desc = op->MutableOutputDesc(static_cast<uint32_t>(j));
    if (output_tensor_desc == nullptr) {
      continue;
    }
    op_desc_info.output_format.emplace_back(output_tensor_desc->GetFormat());
    op_desc_info.output_shape.emplace_back(output_tensor_desc->GetShape().GetDims());
    op_desc_info.output_data_type.emplace_back(output_tensor_desc->GetDataType());
    int64_t output_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(*output_tensor_desc, output_size) != SUCCESS) {
      GELOGW("[Save][OpExceptionInfo] Op [%s] get output size failed.", op->GetName().c_str());
      return;
    }
    GELOGD("[Save][OpExceptionInfo] Save dump op info, the output size is %ld.", output_size);
    op_desc_info.output_size.emplace_back(output_size);
  }
}

void ExceptionDumper::LogExceptionTvmOpInfo(const OpDescInfo &op_desc_info) const {
  if (static_cast<domi::ImplyType>(op_desc_info.imply_type) != domi::ImplyType::TVM) {
    GELOGI("exception op:%s(%s) imply_type:%s not tvm", op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str(),
           TypeUtils::ImplyTypeToSerialString(static_cast<domi::ImplyType>(op_desc_info.imply_type)).c_str());
    return;
  }

  if ((op_desc_info.input_format.size() != op_desc_info.input_shape.size()) ||
      (op_desc_info.input_format.size() != op_desc_info.input_data_type.size())) {
    GELOGW("exception op:%s(%s) input format size:%zu, shape size:%zu, dtype size:%zu not equal, skip log op info",
           op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str(), op_desc_info.input_format.size(),
           op_desc_info.input_shape.size(), op_desc_info.input_data_type.size());
    return;
  }

  if ((op_desc_info.output_format.size() != op_desc_info.output_shape.size()) ||
      (op_desc_info.output_format.size() != op_desc_info.output_data_type.size())) {
    GELOGW("exception op:%s(%s) output format size:%zu, shape size:%zu, dtype size:%zu not equal, skip log op info",
           op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str(), op_desc_info.output_format.size(),
           op_desc_info.output_shape.size(), op_desc_info.output_data_type.size());
    return;
  }

  GEEVENT("[AIC_INFO] node_name:%s, node_type:%s, stream_id:%u, task_id:%u", op_desc_info.op_name.c_str(),
          op_desc_info.op_type.c_str(), op_desc_info.stream_id, op_desc_info.task_id);
  for (size_t i = 0U; i < op_desc_info.input_format.size(); i++) {
    const std::string content = "input:" + std::to_string(i) +
                                ";shape:" + ToString(op_desc_info.input_shape[i]) +
                                ";format:" + TypeUtils::FormatToSerialString(op_desc_info.input_format[i]) +
                                ";dtype:" + TypeUtils::DataTypeToSerialString(op_desc_info.input_data_type[i]) +
                                ";addr:" + std::to_string(PtrToValue(op_desc_info.input_addrs[i]));
    GEEVENT("[AIC_INFO] %s", content.c_str());
  }

  for (size_t i = 0U; i < op_desc_info.output_format.size(); i++) {
    const std::string content = "output:" + std::to_string(i) +
                                ";shape:" + ToString(op_desc_info.output_shape[i]) +
                                ";format:" + TypeUtils::FormatToSerialString(op_desc_info.output_format[i]) +
                                ";dtype:" + TypeUtils::DataTypeToSerialString(op_desc_info.output_data_type[i]) +
                                ";addr:" + std::to_string(PtrToValue(op_desc_info.output_addrs[i]));
    GEEVENT("[AIC_INFO] %s", content.c_str());
  }

  GEEVENT("[AIC_INFO] block_dim:%u", op_desc_info.block_dim);
  if (op_desc_info.is_mem_log) {
    GEEVENT("[AICpu_INFO] workspace_size:%s", ToString(op_desc_info.workspace_bytes).c_str());
    if (!op_desc_info.space_addrs.empty()) {
      GEEVENT("[AICpu_INFO] workspace_addr:%s", std::to_string(PtrToValue(op_desc_info.space_addrs[0])).c_str());
    }
  } else {
    GEEVENT("[AIC_INFO] workspace_bytes:%s", ToString(op_desc_info.workspace_bytes).c_str());
  }
  GEEVENT("[AIC_INFO] dev_func:%s", op_desc_info.dev_func.c_str());
  GEEVENT("[AIC_INFO] tvm_magic:%s", op_desc_info.tvm_magic.c_str());
  GEEVENT("[AIC_INFO] kernel_info:%s/%u", op_desc_info.node_info.c_str(), op_desc_info.tiling_key);
  GEEVENT("[AIC_INFO] tiling_key:%u", op_desc_info.tiling_key);
  std::string log_tiling_data = "";
  if (!op_desc_info.tiling_data.empty()) {
    log_tiling_data = google::protobuf::CEscape(op_desc_info.tiling_data);
  }
  GEEVENT("[AIC_INFO] tiling_data:%s", log_tiling_data.c_str());

  ge::char_t curr_path[MMPA_MAX_PATH] = {};
  if (mmGetCwd(&curr_path[0], MMPA_MAX_PATH) != EN_OK) {
    GELOGW("get current path failed when do aicerror info record");
    return;
  }

  ge::char_t real_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(op_desc_info.op_file_path.c_str(), &real_path[0], MMPA_MAX_PATH) != EN_OK) {
    GELOGW("real path for %s failed when do aicerror info record", op_desc_info.op_file_path.c_str());
    return;
  }
  const std::string file_prefix = op_desc_info.dev_func.substr(0U, op_desc_info.dev_func.rfind("__"));
  const std::string src_file = std::string(real_path) + "/" + file_prefix + ".o";
  const std::string dst_path = std::string(curr_path);

#ifdef __GNUC__
  const uint32_t pid = static_cast<uint32_t>(fork());
  if (pid == 0U) {
    (void)execlp("cp", "cp", src_file.c_str(), dst_path.c_str(), nullptr);
  }
#endif

  GEEVENT("[AIC_INFO] op_file_path:%s", dst_path.c_str());
}

Status ExceptionDumper::DumpExceptionInfo(const std::vector<rtExceptionInfo> &exception_infos) {
  GELOGI("[Dump][Exception] Start to dump exception info");
  std::string env_record_path = "./";
  char_t record_path[MMPA_MAX_PATH]{};
  const auto env_ret =
      (mmGetEnv(kExceptionDumpPath, &record_path[0U], static_cast<uint32_t>(MMPA_MAX_PATH)) == EN_OK) ||
      (mmGetEnv(kEnvCollectPath, &record_path[0U], static_cast<uint32_t>(MMPA_MAX_PATH)) == EN_OK);
  if (env_ret) {
    env_record_path = std::string(&record_path[0U]) + kExtraPath + std::to_string(ge::GetContext().DeviceId()) + "/";
    const int32_t directory_ret = CreateDirectory(env_record_path);
    if (directory_ret != 0) {
        GELOGW("Can not create directory[%s].", env_record_path.c_str());
        return PARAM_INVALID;
    }
  }
  for (const rtExceptionInfo &iter : exception_infos) {
    OpDescInfo op_desc_info;
    if (GetOpDescInfo(iter.streamid, iter.taskid, op_desc_info)) {
      toolkit::dump::DumpData dump_data;
      SetDumpData(op_desc_info, dump_data);
      const uint64_t now_time = GetNowTime();
      std::string op_name = op_desc_info.op_name;
      std::string op_type = op_desc_info.op_type;
      ReplaceStringElem(op_name);
      ReplaceStringElem(op_type);
      const std::string dump_file_path = env_record_path + op_type + "." + op_name + "." +
        std::to_string(op_desc_info.task_id) + "." + std::to_string(now_time);
      GELOGI("[Dump][Exception] The exception dump file path is %s", dump_file_path.c_str());

      uint64_t proto_size = dump_data.ByteSizeLong();
      const std::unique_ptr<char[]> proto_msg = MakeUnique<char[]>(proto_size);
      GE_CHECK_NOTNULL(proto_msg);
      const bool ret = dump_data.SerializeToArray(proto_msg.get(), static_cast<int32_t>(proto_size));
      if ((!ret) || (proto_size == 0U)) {
        REPORT_INNER_ERROR("E19999", "Serialize proto to std::string fail");
        GELOGE(PARAM_INVALID, "[Dump][Exception] Dump data proto serialize failed");
        return PARAM_INVALID;
      }

      auto dump_size = static_cast<int64_t>(sizeof(uint64_t));
      GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(dump_file_path.c_str(), &proto_size, dump_size),
                        "Failed to dump proto size");
      dump_size = static_cast<int64_t>(proto_size);
      GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(dump_file_path.c_str(), proto_msg.get(), dump_size),
                        "Failed to dump proto msg");
      if (DumpExceptionInput(op_desc_info, dump_file_path) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Dump][Exception] Dump exception input failed");
        return PARAM_INVALID;
      }

      if (DumpExceptionOutput(op_desc_info, dump_file_path) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Dump][Exception] Dump exception output failed");
        return PARAM_INVALID;
      }
      for (size_t i = 0U; i < op_desc_info.space_addrs.size(); ++i) {
        GELOGI("workspace_info addr&size=:%lu %lu", PtrToValue(op_desc_info.space_addrs.at(i)),
               op_desc_info.workspace_bytes.at(i));
        if (DumpDevMem(dump_file_path.data(), op_desc_info.space_addrs.at(i),
                       op_desc_info.workspace_bytes.at(i)) != SUCCESS) {
          GELOGE(PARAM_INVALID, "[Dump][ExceptionWorkspace] Dump the %zu workspace data of op [%s] failed",
                 i, op_desc_info.op_name.c_str());
          return PARAM_INVALID;
        }
      }
      GELOGI("[Dump][Exception] Dump exception info SUCCESS");
    } else {
      GELOGW("[Dump][Exception] Get op desc info failed,task id:%u,stream id:%u", iter.taskid, iter.streamid);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

void ExceptionDumper::RefreshAddrs(OpDescInfo &op_desc_info) const {
  if (op_desc_info.args == 0U) {
    GELOGI("op:%s(%s) store args is empty, skip refresh addr",
           op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str());
    return;
  }

  const size_t offset = sizeof(void *);

  const size_t target_size = (op_desc_info.input_shape.size() + op_desc_info.output_shape.size()) * offset;
  std::vector<void *> host_addr(op_desc_info.input_shape.size() + op_desc_info.output_shape.size());

  const auto rt_ret = rtMemcpy(host_addr.data(), target_size, ValueToPtr(op_desc_info.args), target_size,
                               RT_MEMCPY_DEVICE_TO_HOST);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGI("op:%s(%s) can't rtMemcpy to host, store args:%zu, memcpy size:%zu, skip refresh addr",
           op_desc_info.op_name.c_str(), op_desc_info.op_type.c_str(), op_desc_info.args, target_size);
    return;
  }

  for (size_t i = 0U; i < op_desc_info.input_shape.size(); i++) {
    void *target_addr = host_addr[i];
    GELOGI("op:%s(%s) input index:%zu addr:%p refresh to addr:%p", op_desc_info.op_name.c_str(),
           op_desc_info.op_type.c_str(), i, op_desc_info.input_addrs[i], target_addr);
    op_desc_info.input_addrs[i] = target_addr;
  }

  for (size_t i = 0U; i < op_desc_info.output_shape.size(); i++) {
    void *target_addr = host_addr[i + op_desc_info.input_shape.size()];
    GELOGI("op:%s(%s) output index:%zu addr:%p refresh to addr:%p", op_desc_info.op_name.c_str(),
           op_desc_info.op_type.c_str(), i, op_desc_info.output_addrs[i], target_addr);
    op_desc_info.output_addrs[i] = target_addr;
  }
}

bool ExceptionDumper::GetOpDescInfo(const uint32_t stream_id, const uint32_t task_id, OpDescInfo &op_desc_info) {
  GELOGI("[Get][OpDescInfo] There are %zu op info saved, target stream_id:%u, task_id:%u.", op_desc_info_.size(),
         stream_id, task_id);
  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto &dump_op_info : op_desc_info_) {
    if ((dump_op_info.task_id == task_id) && (dump_op_info.stream_id == stream_id)) {
      GELOGI("[Get][OpDescInfo] Find exception op [%s] of task_id: %u, stream_id: %u.", dump_op_info.op_name.c_str(),
             task_id, stream_id);
      op_desc_info = dump_op_info;
      RefreshAddrs(op_desc_info);
      LogExceptionTvmOpInfo(op_desc_info);
      return true;
    }
  }
  return false;
}

Status ExceptionDumper::DumpDevMem(const ge::char_t *const file, const void *const addr, const int64_t size) {
  if (size == 0) {
    GELOGI("No need to dump data, because the size is 0.");
    return SUCCESS;
  }
  uint8_t *host_addr = nullptr;
  rtError_t ret = rtMallocHost(PtrToPtr<uint8_t *, void *>(&host_addr), static_cast<uint64_t>(size),
                               GE_MODULE_NAME_U16);
  if (ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMallocHost failed, size:%" PRIu64 ", ret:0x%" PRIxLEAST8 "", size, ret);
    GELOGE(FAILED, "[Call][RtMallocHost] failed, size:%zu, ret:0x%X", size, ret);
    return FAILED;
  }
  GE_MAKE_GUARD_RTMEM(host_addr);
  ret = rtMemcpy(host_addr, static_cast<uint64_t>(size), addr, static_cast<uint64_t>(size), RT_MEMCPY_DEVICE_TO_HOST);
  if (ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%" PRIu64 ", ret:0x%" PRIxLEAST8 "", size, ret);
    GELOGE(FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", size, ret);
    return FAILED;
  }

  GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(file, host_addr, size));
  return SUCCESS;
}

Status ExceptionDumper::DumpExceptionInput(const OpDescInfo &op_desc_info, const std::string &dump_file) const {
  GELOGI("[Dump][ExceptionInput] Start to dump exception input");
  for (size_t i = 0U; i < op_desc_info.input_addrs.size(); i++) {
    if (DumpDevMem(dump_file.data(), op_desc_info.input_addrs.at(i), op_desc_info.input_size.at(i)) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Dump][ExceptionInput] Dump the %zu input data of op [%s] failed",
             i, op_desc_info.op_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status ExceptionDumper::DumpExceptionOutput(const OpDescInfo &op_desc_info, const std::string &dump_file) const {
  GELOGI("[Dump][ExceptionOutput] Start to dump exception output");
  for (size_t i = 0U; i < op_desc_info.output_addrs.size(); i++) {
    if (DumpDevMem(dump_file.data(), op_desc_info.output_addrs.at(i), op_desc_info.output_size.at(i)) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Dump][ExceptionInput] Dump the %zu input data of op [%s] failed",
             i, op_desc_info.op_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

OpDescInfo *ExceptionDumper::MutableOpDescInfo(const uint32_t task_id, const uint32_t stream_id) {
  for (OpDescInfo &op_desc_info : op_desc_info_) {
    if ((op_desc_info.task_id == task_id) && (op_desc_info.stream_id == stream_id)) {
      return &op_desc_info;
    }
  }
  return nullptr;
}

void ExceptionDumper::Reset(ExtraOpInfo &extra_op_info) {
  extra_op_info.input_addrs.clear();
  extra_op_info.output_addrs.clear();
}
}  // namespace ge
