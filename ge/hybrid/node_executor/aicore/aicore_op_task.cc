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

#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "framework/common/taskdown_common.h"
#include "framework/common/debug/log.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "graph/load/new_model_manager/tbe_handle_store.h"

using optiling::OpRunInfo;

namespace ge {
namespace hybrid {
namespace {
constexpr char const *kAttrSupportDynamicShape = "support_dynamicshape";
constexpr char const *kAttrOpParamSize = "op_para_size";
constexpr char const *kAttrAtomicOpParamSize = "atomic_op_para_size";
}  // namespace

Status AiCoreOpTask::Init(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  GE_CHK_STATUS_RET_NOLOG(InitWithTaskDef(op_desc, task_def));
  GE_CHK_STATUS_RET_NOLOG(InitTilingInfo(op_desc));
  return SUCCESS;
}

Status AiCoreOpTask::RegisterTbeHandle(const OpDesc &op_desc) {
  auto op_desc_ptr = std::make_shared<OpDesc>(op_desc);
  GE_CHECK_NOTNULL(op_desc_ptr);
  auto tbe_kernel = op_desc_ptr->TryGetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
  if (tbe_kernel == nullptr) {
    GELOGE(INTERNAL_ERROR, "TBE: %s can't find tvm bin file!", op_desc_ptr->GetName().c_str());
    return INTERNAL_ERROR;
  }
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
  rtError_t rt_ret = rtQueryFunctionRegistered(stub_name_.c_str());
  if (rt_ret != RT_ERROR_NONE) {
    void *bin_handle = nullptr;
    if (!kernel_store.FindTBEHandle(stub_name_.c_str(), bin_handle)) {
      GELOGI("TBE: can't find the kernel_name[%s] in HandleMap", stub_name_.c_str());
      rtDevBinary_t binary;
      std::string json_string;
      GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_ptr, TVM_ATTR_NAME_MAGIC, json_string),
                      GELOGI("Get original type of session_graph_id."));
      if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AICPU") {
        binary.magic = RT_DEV_BINARY_MAGIC_ELF_AICPU;
      } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF") {
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
      } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
        binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
      } else {
        GELOGE(PARAM_INVALID, "TBE: Invalid parameter magic number! json: %s", json_string.c_str());
        return PARAM_INVALID;
      }
      binary.version = 0;
      binary.data = tbe_kernel->GetBinData();
      binary.length = tbe_kernel->GetBinDataSize();
      GELOGI("TBE: binary.length: %lu", binary.length);
      GE_CHK_RT_RET(rtDevBinaryRegister(&binary, &bin_handle));
      std::string meta_data;
      GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_ptr, TVM_ATTR_NAME_METADATA, meta_data),
                      GELOGI("Get original type of json_string"));
      GELOGI("TBE: meta data: %s", meta_data.empty() ? "null" : meta_data.c_str());
      GE_IF_BOOL_EXEC(!meta_data.empty(), GE_CHK_RT_RET(rtMetadataRegister(bin_handle, meta_data.c_str())));
      kernel_store.StoreTBEHandle(stub_name_.c_str(), bin_handle, tbe_kernel);
    } else {
      GELOGI("TBE: find the kernel_name[%s] in HandleMap", stub_name_.c_str());
      kernel_store.ReferTBEHandle(stub_name_.c_str());
    }
    std::string kernel_name;
    GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_ptr, op_desc_ptr->GetName() + "_kernelname", kernel_name),
                    GELOGI("Get original type of kernel_name"));
    GELOGI("TBE: binfile_key=%s, kernel_name=%s", stub_name_.c_str(), kernel_name.c_str());
    GE_CHK_RT_RET(rtFunctionRegister(bin_handle, stub_name_.c_str(), stub_name_.c_str(), kernel_name.c_str(), 0));
  }
  return SUCCESS;
}

Status AiCoreOpTask::InitWithTaskDef(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  GE_CHK_STATUS_RET(ValidateTaskDef(task_def),
                    "[%s] Failed to validate task def: [%s]",
                    op_desc.GetName().c_str(),
                    task_def.DebugString().c_str());

  const domi::KernelDef &kernel_def = task_def.kernel();
  const domi::KernelContext &context = kernel_def.context();
  stub_name_ = kernel_def.stub_func();

  GE_CHK_STATUS_RET(RegisterTbeHandle(op_desc));

  GE_CHK_RT_RET(rtGetFunctionByName(stub_name_.c_str(), &stub_func_));
  args_size_ = kernel_def.args_size();
  block_dim_ = kernel_def.block_dim();

  // malloc args memory
  args_.reset(new(std::nothrow) uint8_t[args_size_]);
  GE_CHECK_NOTNULL(args_);
  errno_t err = memcpy_s(args_.get(), args_size_, kernel_def.args().data(), args_size_);
  if (err != EOK) {
    GELOGE(INTERNAL_ERROR, "AiCoreTask memcpy args failed.");
    return INTERNAL_ERROR;
  }

  if (context.args_offset().size() < sizeof(uint16_t)) {
    GELOGE(INTERNAL_ERROR, "Invalid args_offset, size = %zu.", context.args_offset().size());
    return INTERNAL_ERROR;
  }

  const auto *args_offset_buffer = reinterpret_cast<const uint16_t *>(context.args_offset().data());
  uint32_t offset = *args_offset_buffer;
  if (offset > args_size_) {
    GELOGE(INTERNAL_ERROR,
           "[%s] Arg offset out of range. offset = %u, arg size = %u",
           GetName().c_str(),
           offset,
           args_size_);
    return INTERNAL_ERROR;
  }

  arg_base_ = reinterpret_cast<uintptr_t *>(args_.get() + offset);
  max_arg_count_ = (args_size_ - offset) / sizeof(void *);
  GELOGD("[%s] Done setting kernel args successfully. stub_func = %s, block_dim = %d, arg base = %p, arg size = %u",
         op_desc.GetName().c_str(),
         stub_name_.c_str(),
         block_dim_,
         arg_base_,
         args_size_);

  return SUCCESS;
}

Status AiCoreOpTask::ValidateTaskDef(const domi::TaskDef &task_def) {
  auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
  if (task_type != RT_MODEL_TASK_KERNEL) {
    GELOGE(INTERNAL_ERROR, "Invalid task type (%d) in AiCore CreateTask.", static_cast<int>(task_type));
    return INTERNAL_ERROR;
  }

  const domi::KernelDef &kernel_def = task_def.kernel();
  const domi::KernelContext &context = kernel_def.context();
  auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
  if (kernel_type != ccKernelType::TE) {
    GELOGE(INTERNAL_ERROR, "Invalid kernel type(%d) in AiCore TaskDef.", static_cast<int>(kernel_type));
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status AiCoreOpTask::PrepareWithShape(TaskContext &context) {
  if (tiling_buffer_ != nullptr) {
    return UpdateTilingInfo(context);
  }

  return SUCCESS;
}

Status AiCoreOpTask::UpdateTilingInfo(TaskContext &context) {
  auto node = context.GetNodeItem().node;
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  GELOGD("[%s] Start to update tiling info for task: [%s]", node->GetName().c_str(), stub_name_.c_str());
  OpRunInfo tiling_info;
  tiling_info.block_dim = -1; // codex: Using uninitialized value
  tiling_info.clear_atomic = true;

  auto execution_context = context.GetExecutionContext();
  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CalcTilingInfo] Start");
  GE_CHK_STATUS_RET(CalcTilingInfo(node, tiling_info));
  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CalcTilingInfo] End");

  // update op args by tiling info
  block_dim_ = static_cast<uint32_t>(tiling_info.block_dim);
  op_desc->SetWorkspaceBytes(tiling_info.workspaces);
  clear_atomic_ = tiling_info.clear_atomic;

  tiling_data_ = tiling_info.tiling_data.str();
  if (tiling_data_.empty()) {
    GELOGE(INTERNAL_ERROR, "[%s] Tiling data is empty.", stub_name_.c_str());
    return INTERNAL_ERROR;
  }

  if (tiling_data_.size() > tiling_buffer_->GetSize()) {
    GELOGE(INTERNAL_ERROR, "[%s] Tiling data size now (%zu) shouldn't larger than we alloc before (%zu).",
           stub_name_.c_str(), tiling_data_.size(), tiling_buffer_->GetSize());
    return INTERNAL_ERROR;
  }

  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CopyTilingInfo] Start");
  GE_CHK_RT_RET(rtMemcpy(tiling_buffer_->GetData(), tiling_buffer_->GetSize(),
                         tiling_data_.c_str(), tiling_data_.size(),
                         RT_MEMCPY_HOST_TO_DEVICE));
  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CopyTilingInfo] End");

  GELOGD("[%s] Done updating tiling info for task: [%s]", node->GetName().c_str(), stub_name_.c_str());
  return SUCCESS;
}

Status AiCoreOpTask::CalcTilingInfo(const NodePtr &node, OpRunInfo &tiling_info) {
  GELOGD("[%s] Start to invoke OpParaCalculate.", node->GetName().c_str());
  GE_CHK_STATUS_RET(OpParaCalculate(*node, tiling_info),
                    "Failed calc tiling data of node %s.",
                    node->GetName().c_str());
  GELOGD("[%s] Done invoking OpParaCalculate successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreOpTask::UpdateArgs(TaskContext &task_context) {
  size_t expected_arg_count = task_context.NumInputs() + task_context.NumOutputs() + task_context.NumWorkspaces();
  if (tiling_buffer_ != nullptr) {
    ++expected_arg_count;
  }
  if (expected_arg_count > max_arg_count_) {
    GELOGE(INTERNAL_ERROR,
           "[%s] Invalid arg memory, max arg count = %u, but expect = %zu",
           GetName().c_str(),
           max_arg_count_,
           expected_arg_count);
    return INTERNAL_ERROR;
  }

  int index = 0;
  for (int i = 0; i < task_context.NumInputs(); ++i) {
    const auto input = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input);
    arg_base_[index++] = reinterpret_cast<uintptr_t>(input->GetData());
  }

  for (int i = 0; i < task_context.NumOutputs(); ++i) {
    const auto output = task_context.GetOutput(i);
    GE_CHECK_NOTNULL(output);
    arg_base_[index++] = reinterpret_cast<uintptr_t>(output->GetData());
  }

  int workspace_num = static_cast<int>(task_context.NumWorkspaces());
  for (int i = 0; i < workspace_num; ++i) {
    const auto workspace = task_context.MutableWorkspace(i);
    GE_CHECK_NOTNULL(workspace);
    arg_base_[index++] = reinterpret_cast<uintptr_t>(workspace);
  }

  if (tiling_buffer_ != nullptr) {
    arg_base_[index++] = reinterpret_cast<uintptr_t>(tiling_buffer_->GetData());
  }

  if (task_context.IsTraceEnabled()) {
    for (int i = 0; i < index; ++i) {
      GELOGD("[%s] Arg[%d] = %lu", stub_name_.c_str(), i, arg_base_[i]);
    }
  }

  return SUCCESS;
}

Status AiCoreOpTask::LaunchKernel(rtStream_t stream) {
  GELOGD("AiCoreOpTask LaunchKernel Start (task = %s, block_dim = %u).", stub_name_.c_str(), block_dim_);
  GE_CHK_RT_RET(rtKernelLaunch(stub_func_, block_dim_, args_.get(), args_size_, nullptr, stream));
  GELOGD("AiCoreOpTask LaunchKernel End (task = %s, block_dim = %u).", stub_name_.c_str(), block_dim_);
  return SUCCESS;
}

Status AiCoreOpTask::InitTilingInfo(const OpDesc &op_desc) {
  bool dynamic_supported = false;
  (void) AttrUtils::GetBool(op_desc, kAttrSupportDynamicShape, dynamic_supported);
  if (!dynamic_supported) {
    GELOGD("[%s] Dynamic shape is not supported.", op_desc.GetName().c_str());
    return SUCCESS;
  }

  GELOGD("Start alloc tiling data of node %s.", op_desc.GetName().c_str());
  int64_t max_size = -1;
  (void) AttrUtils::GetInt(op_desc, GetKeyForOpParamSize(), max_size);
  GELOGD("Got op param size by key: %s, ret = %ld", GetKeyForOpParamSize().c_str(), max_size);
  if (max_size <= 0) {
    GELOGE(PARAM_INVALID, "[%s] Invalid op_param_size: %ld.", op_desc.GetName().c_str(), max_size);
    return PARAM_INVALID;
  }

  auto allocator = NpuMemoryAllocator::GetAllocator();
  GE_CHECK_NOTNULL(allocator);
  tiling_buffer_ = TensorBuffer::Create(allocator, static_cast<size_t>(max_size));
  GE_CHECK_NOTNULL(tiling_buffer_);

  GELOGD("[%s] Done allocating tiling buffer, size=%ld.", op_desc.GetName().c_str(), max_size);
  return SUCCESS;
}

bool AiCoreOpTask::IsDynamicShapeSupported() {
  return tiling_buffer_ != nullptr;
}

const std::string &AiCoreOpTask::GetName() const {
  return stub_name_;
}

std::string AiCoreOpTask::GetKeyForOpParamSize() const {
  return kAttrOpParamSize;
}

Status AtomicAddrCleanOpTask::Init(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  GE_CHK_STATUS_RET_NOLOG(AiCoreOpTask::Init(op_desc, task_def));
  return InitAtomicAddrCleanIndices(op_desc);
}

Status AtomicAddrCleanOpTask::InitAtomicAddrCleanIndices(const OpDesc &op_desc) {
  GELOGD("[%s] Start to setup AtomicAddrClean task.", op_desc.GetName().c_str());
  std::vector<int64_t> atomic_output_indices;
  (void) ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  map<string, map<int64_t, int64_t>> workspace_info; // op_name, ws_index, ws_offset
  workspace_info = op_desc.TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);
  if (atomic_output_indices.empty() && workspace_info.empty()) {
    GELOGE(INTERNAL_ERROR,
           "[%s] Neither ATOMIC_ATTR_OUTPUT_INDEX nor EXT_ATTR_ATOMIC_WORKSPACE_INFO is empty.",
           op_desc.GetName().c_str());
    return INTERNAL_ERROR;
  }

  for (auto output_index : atomic_output_indices) {
    GELOGD("[%s] Adding output index [%ld]", op_desc.GetName().c_str(), output_index);
    GE_CHECK_GE(output_index, 0);
    GE_CHECK_LE(output_index, INT32_MAX);
    atomic_output_indices_.emplace_back(static_cast<int>(output_index));
  }

  for (auto &iter : workspace_info) {
    for (auto &info_iter : iter.second) {
      auto workspace_index = info_iter.first;
      GELOGD("[%s] Adding workspace index [%ld]", op_desc.GetName().c_str(), workspace_index);
      GE_CHECK_GE(workspace_index, 0);
      GE_CHECK_LE(workspace_index, INT32_MAX);
      atomic_workspace_indices_.emplace_back(static_cast<int>(workspace_index));
    }
  }

  size_t arg_count = atomic_workspace_indices_.size() + atomic_output_indices_.size();
  if (tiling_buffer_ != nullptr) {
    arg_count += 1;
  }

  if (arg_count > max_arg_count_) {
    GELOGE(INTERNAL_ERROR,
           "[%s] Invalid arg memory, max arg count = %u, but expect = %zu",
           GetName().c_str(),
           max_arg_count_,
           arg_count);
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

std::string AtomicAddrCleanOpTask::GetKeyForOpParamSize() const {
  return kAttrAtomicOpParamSize;
}

Status AtomicAddrCleanOpTask::CalcTilingInfo(const NodePtr &node, OpRunInfo &tiling_info) {
  GELOGD("[%s] Start to invoke OpAtomicCalculate.", node->GetName().c_str());
  GE_CHK_STATUS_RET(OpAtomicCalculate(*node, tiling_info),
                    "Failed calc tiling data of node %s.",
                    node->GetName().c_str());
  GELOGD("[%s] Done invoking OpAtomicCalculate successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status AtomicAddrCleanOpTask::UpdateArgs(TaskContext &task_context) {
  // refresh atomic output addr
  int index = 0;
  for (auto atomic_output_index : atomic_output_indices_) {
    const auto output_tensor = task_context.GetOutput(atomic_output_index);
    GE_CHECK_NOTNULL(output_tensor);
    arg_base_[index++] = reinterpret_cast<uintptr_t>(output_tensor->GetData());
  }

  // refresh atomic workspace addr
  for (auto atomic_ws_index : atomic_workspace_indices_) {
    const auto workspace_tensor = task_context.GetOutput(atomic_ws_index);
    GE_CHECK_NOTNULL(workspace_tensor);
    arg_base_[index++] = reinterpret_cast<uintptr_t>(workspace_tensor->GetData());
  }

  if (tiling_buffer_ != nullptr) {
    arg_base_[index++] = reinterpret_cast<uintptr_t>(tiling_buffer_->GetData());
  } else {
    GELOGD("[%s] Not a dynamic op", GetName().c_str());
  }

  if (task_context.IsTraceEnabled()) {
    for (int i = 0; i < index; ++i) {
      GELOGD("[%s] Arg[%d] = %lu", GetName().c_str(), i, arg_base_[i]);
    }
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
