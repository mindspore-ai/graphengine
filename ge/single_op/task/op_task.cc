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

#include "single_op/task/op_task.h"

#include <google/protobuf/extension_set.h>
#include <chrono>
#include <thread>

#include "aicpu/common/aicpu_task_struct.h"
#include "common/dump/dump_manager.h"
#include "common/dump/dump_op.h"
#include "common/formats/formats.h"
#include "common/math/math_util.h"
#include "framework/common/debug/log.h"
#include "register/op_tiling.h"
#include "runtime/rt.h"
#include "build_task_utils.h"

namespace ge {
namespace {
constexpr int kLaunchRetryTimes = 1000;
constexpr int kSleepTime = 10;
constexpr uint64_t kReleaseFlag = 1;
constexpr int kCopyNum = 2;
void FreeHbm(void *var) {
  if (var) {
    (void)rtFree(var);
  }
}
}  // namespace

Status OpTask::OpenDump(rtStream_t stream) {
  if (DumpManager::GetInstance().GetDumpProperties().IsSingleOpNeedDump()) {
    GELOGI("Dump is open in single op,start to set dump info");
    std::vector<uint64_t> input_addrs;
    std::vector<uint64_t> output_adds;
    auto input_size = op_desc_->GetInputsSize();
    auto output_size = op_desc_->GetOutputsSize();
    uintptr_t *arg_base = nullptr;
    size_t arg_num = 0;
    GetIoAddr(arg_base, arg_num);
    if (arg_num < input_size + output_size) {
      GELOGE(FAILED, "io_addrs_for_dump_ size %zu is not equal input and output size %zu",
             arg_num,
             input_size + output_size);
      return FAILED;
    }

    for (size_t i = 0; i < input_size; i++) {
      uint64_t input_addr = arg_base[i];
      input_addrs.emplace_back(input_addr);
    }
    for (size_t j = 0; j < output_size; j++) {
      uint64_t output_addr = arg_base[input_size + j];
      output_adds.emplace_back(output_addr);
    }
    dump_op_.SetDumpInfo(DumpManager::GetInstance().GetDumpProperties(), op_desc_, input_addrs, output_adds, stream);
    auto status = dump_op_.LaunchDumpOp();
    if (status != SUCCESS) {
      GELOGE(status, "Launch dump op failed in single op");
      return status;
    }
    return SUCCESS;
  }
  GELOGI("Dump is not open in single op");
  return SUCCESS;
}

void TbeOpTask::SetStubFunc(const std::string &name, const void *stub_func) {
  this->stub_name_ = name;
  this->stub_func_ = stub_func;
}

void TbeOpTask::SetKernelArgs(std::unique_ptr<uint8_t[]> &&args, size_t arg_size, uint32_t block_dim,
                              const OpDescPtr &op_desc) {
  args_ = std::move(args);
  arg_size_ = arg_size;
  block_dim_ = block_dim;
  op_desc_ = op_desc;
}

void TbeOpTask::SetSmDesc(void *sm_desc) { sm_desc_ = sm_desc; }

void OpTask::SetModelArgs(std::string model_name, uint32_t model_id) {
  model_name_ = model_name;
  model_id_ = model_id;
}

Status OpTask::GetProfilingArgs(std::string &model_name, std::string &op_name, uint32_t &model_id,
                                uint32_t &block_dim) {
  model_name = model_name_;
  model_id = model_id_;
  block_dim = block_dim_;
  GE_CHECK_NOTNULL(op_desc_);
  op_name = op_desc_->GetName();
  return SUCCESS;
}
Status OpTask::UpdateRunInfo(const vector<GeTensorDesc> &input_desc, const vector<GeTensorDesc> &output_desc) {
  return UNSUPPORTED;
}

Status OpTask::DoUpdateArgTable(const SingleOpModelParam &param, bool keep_workspace) {
  auto addresses = BuildTaskUtils::GetAddresses(op_desc_, param, keep_workspace);
  auto all_addresses = BuildTaskUtils::JoinAddresses(addresses);
  uintptr_t *arg_base = nullptr;
  size_t arg_num = 0;
  GetIoAddr(arg_base, arg_num);
  if (arg_num < all_addresses.size()) {
    GELOGE(INTERNAL_ERROR, "[%s] arg number mismatches, expect at least = %zu, but got = %zu",
           op_desc_->GetName().c_str(),
           all_addresses.size(),
           arg_num);
    return INTERNAL_ERROR;
  }

  for (void *addr : all_addresses) {
    *arg_base++ = reinterpret_cast<uintptr_t >(addr);
  }
  return SUCCESS;
}

Status OpTask::UpdateArgTable(const SingleOpModelParam &param) {
  return DoUpdateArgTable(param, true);
}

Status OpTask::LaunchKernel(const vector<GeTensorDesc> &input_desc,
                            const vector<DataBuffer> &input_buffers,
                            vector<GeTensorDesc> &output_desc,
                            vector<DataBuffer> &output_buffers,
                            rtStream_t stream) {
  return UNSUPPORTED;
}

uint32_t OpTask::GetTaskType() const { return kTaskTypeInvalid; }

TbeOpTask::~TbeOpTask() {
  if (sm_desc_ != nullptr) {
    (void)rtMemFreeManaged(sm_desc_);
  }

  if (tiling_buffer_ != nullptr) {
    (void)rtFree(tiling_buffer_);
  }
}

const void *TbeOpTask::GetArgs() const { return args_.get(); }

size_t TbeOpTask::GetArgSize() const { return arg_size_; }

const std::string &TbeOpTask::GetStubName() const { return stub_name_; }

uint32_t TbeOpTask::GetTaskType() const { return kTaskTypeAicore; }

Status TbeOpTask::LaunchKernel(rtStream_t stream) {
  GELOGD("To invoke rtKernelLaunch. task = %s, block_dim = %u", this->stub_name_.c_str(), block_dim_);
  auto *sm_desc = reinterpret_cast<rtSmDesc_t *>(sm_desc_);
  auto ret = rtKernelLaunch(stub_func_, block_dim_, args_.get(), static_cast<uint32_t>(arg_size_), sm_desc, stream);
  int retry_times = 0;
  while (ret != RT_ERROR_NONE && retry_times < kLaunchRetryTimes) {
    retry_times++;
    GELOGW("Retry after %d ms, retry_times: %d", kSleepTime, retry_times);
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepTime));
    ret = rtKernelLaunch(stub_func_, block_dim_, args_.get(), arg_size_, sm_desc, stream);
  }

  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Invoke rtKernelLaunch failed. ret = %d, task = %s", ret, this->stub_name_.c_str());
    return RT_FAILED;
  }
  GELOGI("[TASK_INFO] %s", this->stub_name_.c_str());
  auto status = OpenDump(stream);
  if (status != SUCCESS) {
    GELOGE(status, "Open dump failed in the tbe single op %s", this->stub_name_.c_str());
    return status;
  }

  return SUCCESS;
}

Status TbeOpTask::UpdateRunInfo(const vector<GeTensorDesc> &input_desc, const vector<GeTensorDesc> &output_desc) {
  GE_CHK_STATUS_RET_NOLOG(UpdateNodeByShape(input_desc, output_desc));
  // invoke OpParaCalculate
  GELOGD("Start to invoke OpParaCalculate.");
  optiling::OpRunInfo run_info;
  run_info.block_dim = 0;
  auto ret = optiling::OpParaCalculate(*node_, run_info);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to invoke OpParaCalculate. ret = %u", ret);
    return FAILED;
  }
  block_dim_ = run_info.block_dim;
  tiling_data_ = run_info.tiling_data.str();
  GELOGD("Done invoking OpParaCalculate successfully. block_dim = %u, tiling size = %zu", block_dim_,
         tiling_data_.size());

  GE_CHK_STATUS_RET(AllocateWorkspaces(run_info.workspaces), "Failed to allocate workspaces");
  return SUCCESS;
}

Status TbeOpTask::UpdateTensorDesc(const GeTensorDesc &src_tensor, GeTensorDesc &dst_tensor) {
  int64_t storage_format_val = static_cast<Format>(FORMAT_RESERVED);
  (void)AttrUtils::GetInt(src_tensor, ge::ATTR_NAME_STORAGE_FORMAT, storage_format_val);
  auto storage_format = static_cast<Format>(storage_format_val);
  if (storage_format == FORMAT_RESERVED) {
    GELOGD("Storage format not set. update shape to [%s], and original shape to [%s]",
           src_tensor.GetShape().ToString().c_str(), src_tensor.GetOriginShape().ToString().c_str());
    dst_tensor.SetShape(src_tensor.GetShape());
    dst_tensor.SetOriginShape(src_tensor.GetOriginShape());
  } else {
    std::vector<int64_t> storage_shape;
    if (!AttrUtils::GetListInt(src_tensor, ge::ATTR_NAME_STORAGE_SHAPE, storage_shape)) {
      GELOGE(PARAM_INVALID, "Failed to get storage_shape while storage_format was set");
      return PARAM_INVALID;
    }

    GELOGD("Storage format set. update shape to [%s], and original shape to [%s]",
           GeShape(storage_shape).ToString().c_str(), src_tensor.GetShape().ToString().c_str());
    dst_tensor.SetShape(GeShape(std::move(storage_shape)));
    dst_tensor.SetOriginShape(src_tensor.GetShape());
  }

  return SUCCESS;
}

Status TbeOpTask::UpdateNodeByShape(const vector<GeTensorDesc> &input_desc, const vector<GeTensorDesc> &output_desc) {
  auto op_desc = node_->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // Set runtime shape to node
  for (size_t i = 0; i < input_desc.size(); ++i) {
    auto tensor_desc = op_desc->MutableInputDesc(i);
    auto &runtime_tensor_desc = input_desc[i];
    GE_CHECK_NOTNULL(tensor_desc);
    GE_CHK_STATUS_RET(UpdateTensorDesc(runtime_tensor_desc, *tensor_desc));
  }

  for (size_t i = 0; i < output_desc.size(); ++i) {
    auto tensor_desc = op_desc->MutableOutputDesc(i);
    auto &runtime_tensor_desc = output_desc[i];
    GE_CHECK_NOTNULL(tensor_desc);
    GE_CHK_STATUS_RET(UpdateTensorDesc(runtime_tensor_desc, *tensor_desc));
  }

  return SUCCESS;
}

void TbeOpTask::EnableDynamicSupport(const NodePtr &node, void *tiling_buffer, size_t max_tiling_size) {
  node_ = node;
  tiling_buffer_ = tiling_buffer;
  max_tiling_size_ = max_tiling_size;
}

Status TbeOpTask::AllocateWorkspaces(const vector<int64_t> &workspace_sizes) {
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
  GE_CHECK_NOTNULL(stream_resource_);
  auto ws_base = stream_resource_->MallocMemory(kPurpose, static_cast<size_t>(total_size));
  if (ws_base == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Failed to allocate memory of size: %ld", total_size);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  GELOGD("Done allocating workspace memory successfully.");

  for (auto ws_offset : ws_offsets) {
    workspaces_.emplace_back(ws_base + ws_offset);
  }

  return SUCCESS;
}

Status TbeOpTask::LaunchKernel(const vector<GeTensorDesc> &input_desc,
                               const vector<DataBuffer> &input_buffers,
                               vector<GeTensorDesc> &output_desc,
                               vector<DataBuffer> &output_buffers,
                               rtStream_t stream) {
  GE_CHK_STATUS_RET_NOLOG(UpdateRunInfo(input_desc, output_desc));
  GELOGD("[%s] Start to launch kernel", node_->GetName().c_str());
  std::vector<void *> args;
  for (auto &buffer : input_buffers) {
    args.emplace_back(buffer.data);
  }
  for (auto &buffer : output_buffers) {
    args.emplace_back(buffer.data);
  }
  for (auto &buffer : workspaces_) {
    args.emplace_back(buffer);
  }

  if (tiling_buffer_ != nullptr) {
    GELOGD("[%s] Start to copy tiling info. size = %zu", node_->GetName().c_str(), tiling_data_.size());
    GE_CHK_RT_RET(rtMemcpyAsync(tiling_buffer_, max_tiling_size_, tiling_data_.data(), tiling_data_.size(),
                                RT_MEMCPY_HOST_TO_DEVICE_EX, stream));

    args.emplace_back(tiling_buffer_);
  }

  if (memcpy_s(args_.get(), arg_size_, args.data(), args.size() * sizeof(void *)) != EOK) {
    GELOGE(INTERNAL_ERROR, "[%s] Failed to update kernel args.", node_->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("[%s] Start to invoke rtKernelLaunch", node_->GetName().c_str());
  GE_CHK_RT_RET(rtKernelLaunch(stub_func_, block_dim_, args_.get(), arg_size_, nullptr, stream));
  GELOGD("[%s] Done invoking rtKernelLaunch successfully", node_->GetName().c_str());
  return SUCCESS;
}

void TbeOpTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = reinterpret_cast<uintptr_t *>(args_.get());
  arg_count = arg_size_ / sizeof(void *);
  if (tiling_buffer_ != nullptr) {
    --arg_count;
  }
}

AiCpuBaseTask::~AiCpuBaseTask() {
  if (ext_info_addr_dev_ != nullptr) {
    (void)rtFree(ext_info_addr_dev_);
  }
}

Status AiCpuBaseTask::SetExtInfoAndType(const std::string &kernel_ext_info, uint64_t kernel_id) {
  if (kernel_ext_info.empty()) {
    GELOGI("Kernel_ext_info is empty, no need copy to device.");
    return SUCCESS;
  }

  int32_t unknown_shape_type_val = 0;
  (void) AttrUtils::GetInt(op_desc_, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  GELOGD("Get unknown_type is %d.", unknown_shape_type_val);
  unknown_type_ = static_cast<UnknowShapeOpType>(unknown_shape_type_val);

  aicpu_ext_handle_.reset(new(std::nothrow) ::ge::hybrid::AicpuExtInfoHandler(op_desc_->GetName(),
                                                                              num_inputs_,
                                                                              num_outputs_,
                                                                              unknown_type_));
  GE_CHK_BOOL_RET_STATUS(aicpu_ext_handle_ != nullptr, FAILED, "Malloc aicpu_ext_handle mem failed!");

  Status ret = aicpu_ext_handle_->Parse(kernel_ext_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "Parse kernel ext info failed, kernel_ext_info_size=%zu.", kernel_ext_info.size());
    return ret;
  }

  GE_CHK_STATUS_RET(aicpu_ext_handle_->UpdateSessionInfo(ULLONG_MAX, kernel_id, false),
                    "UpdateSessionInfo failed.");

  GE_CHK_RT_RET(rtMalloc(&ext_info_addr_dev_, aicpu_ext_handle_->GetExtInfoLen(), RT_MEMORY_HBM));
  GE_CHK_RT_RET(rtMemcpy(ext_info_addr_dev_, aicpu_ext_handle_->GetExtInfoLen(),
                         aicpu_ext_handle_->GetExtInfo(), aicpu_ext_handle_->GetExtInfoLen(),
                         RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AiCpuBaseTask::SetInputConst() {
  input_is_const_.clear();
  const vector<bool> v_is_input_const = op_desc_->GetIsInputConst();
  for (size_t i = 0; i < op_desc_->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr tensor_desc = op_desc_->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGD("SingleOp: %s, Index: %zu, has no input", op_desc_->GetName().c_str(), i);
      continue;
    }
    if (i < v_is_input_const.size() && v_is_input_const[i]) {
      GELOGD("SingleOp: %s, Index: %zu, input is const", op_desc_->GetName().c_str(), i);
      input_is_const_.push_back(true);
      continue;
    }
    input_is_const_.push_back(false);
  }
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateExtInfo(const std::vector<GeTensorDesc> &input_desc, 
                                    std::vector<GeTensorDesc> &output_desc,
                                    rtStream_t stream) {
  GELOGI("Update ext info begin, unknown_type=%d.", unknown_type_);
  if (num_inputs_ == 0 && num_outputs_ == 0) {
    GELOGI("No input and output, no need update ext info.");
    return SUCCESS;
  }

  GE_CHECK_NOTNULL(aicpu_ext_handle_);

  size_t non_const_index = 0;
  for (size_t input_index = 0; input_index < num_inputs_; input_index++) {
    if (input_index < input_is_const_.size() && input_is_const_[input_index]) {
      // get input_desc from op_desc if const input, num_inputs_ is op_desc_ input_size
      auto const_input_desc = op_desc_->MutableInputDesc(static_cast<uint32_t>(input_index));
      GE_CHECK_NOTNULL(const_input_desc);
      GE_CHK_STATUS_RET(aicpu_ext_handle_->UpdateInputShapeAndType(input_index, *const_input_desc),
                        "Input[%zu] update input shape failed.", input_index);
      continue;
    }
    GE_CHK_BOOL_RET_STATUS(non_const_index < input_desc.size(), PARAM_INVALID,
                           "Input_desc size is %zu, but get non_const_index is %zu",
                           input_desc.size(), non_const_index);
    GE_CHK_STATUS_RET(aicpu_ext_handle_->UpdateInputShapeAndType(input_index, input_desc[non_const_index]),
                      "Input[%zu] update input shape failed.", input_index);
    non_const_index++;
  }

  if (unknown_type_ != DEPEND_COMPUTE) {
    for (size_t j = 0; j < num_outputs_; ++j) {
      GE_CHK_STATUS_RET(aicpu_ext_handle_->UpdateOutputShapeAndType(j, output_desc[j]),
                        "Output[%zu] UpdateOutputShapeAndType failed.", j);
    }
  }

  GE_CHK_RT_RET(rtMemcpyAsync(ext_info_addr_dev_,
                              aicpu_ext_handle_->GetExtInfoLen(), // check size
                              aicpu_ext_handle_->GetExtInfo(),
                              aicpu_ext_handle_->GetExtInfoLen(),
                              RT_MEMCPY_HOST_TO_DEVICE_EX,
                              stream));

  GELOGI("Update ext info end.");
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateOutputShape(vector<GeTensorDesc> &output_desc) {
  if (num_outputs_ == 0) {
    GELOGD("AiCpuBaseTask output_num is 0, no need update output shape.");
    return SUCCESS;
  }
  GELOGD("Start to update DEPEND_SHAPE_RANGE AiCpuBaseTask outputshape.");

  GE_CHK_RT_RET(rtMemcpy(aicpu_ext_handle_->GetExtInfo(),
                         aicpu_ext_handle_->GetExtInfoLen(),
                         ext_info_addr_dev_,
                         aicpu_ext_handle_->GetExtInfoLen(),
                         RT_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < num_outputs_; ++i) {
    GeShape shape;
    DataType data_type;
    aicpu_ext_handle_->GetOutputShapeAndType(i, shape, data_type);
    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(shape, output_desc[i]),
                      "AiCpuCCTask Update [%zu]th output shape failed.", i);
  }
  GELOGD("Update DEPEND_SHAPE_RANGE AiCpuBaseTask outputshape finished.");
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateShapeToOutputDesc(const GeShape &shape_new, GeTensorDesc &output_desc) {
  auto shape_old = output_desc.GetShape();
  output_desc.SetShape(shape_new);
  GELOGD("Update AiCpuBaseTask shape from %s to %s", shape_old.ToString().c_str(), shape_new.ToString().c_str());

  auto origin_shape_old = output_desc.GetOriginShape();
  auto origin_format = output_desc.GetOriginFormat();
  auto format = output_desc.GetFormat();
  if (origin_format == format) {
    output_desc.SetOriginShape(shape_new);
    return SUCCESS;
  }

  std::vector<int64_t> origin_dims_new;

  auto trans_ret = formats::TransShape(format, shape_new.GetDims(),
                                       output_desc.GetDataType(), origin_format, origin_dims_new);
  GE_CHK_STATUS_RET(trans_ret,
                    "AiCpuTask originFormat[%d] is not same as format[%d], but TransShape failed, shape=%s.",
                    origin_format, format, shape_new.ToString().c_str());

  auto origin_shape_new = GeShape(origin_dims_new);
  output_desc.SetOriginShape(origin_shape_new);
  GELOGD("AiCpuTask originFormat[%d] is not same as format[%d], need update from %s ro %s.",
         origin_format, format, origin_shape_old.ToString().c_str(), origin_shape_new.ToString().c_str());
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateIoAddr(const vector<DataBuffer> &inputs, const vector<DataBuffer> &outputs) {
  uintptr_t *arg_base = nullptr;
  size_t arg_num = 0;
  GetIoAddr(arg_base, arg_num);

  // input number and output number was check in ValidateParams
  size_t non_const_index = 0;
  for (size_t input_index = 0; input_index < num_inputs_; input_index++) {
    if (input_index < input_is_const_.size() && input_is_const_[input_index]) {
      // const input no need update addr
      GE_CHECK_NOTNULL(arg_base);
      GELOGD("AICpuTask input[%zu] addr = %lu", input_index, *arg_base);
      arg_base++;
      continue;
    }
    GE_CHK_BOOL_RET_STATUS(non_const_index < inputs.size(), PARAM_INVALID,
                           "Input size is %zu, but get non_const_index is %zu",
                           inputs.size(), non_const_index);
    auto addr = inputs[non_const_index].data;
    GE_CHECK_NOTNULL(addr);
    GELOGD("AICpuTask input[%zu] addr = %p", input_index, addr);
    *arg_base++ = reinterpret_cast<uintptr_t>(addr);
    non_const_index++;
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto addr = outputs[i].data;
    GE_CHECK_NOTNULL(addr);
    GELOGD("AICpuTask output[%zu] addr = %p", i, addr);
    *arg_base++ = reinterpret_cast<uintptr_t>(addr);
  }

  return SUCCESS;
}

AiCpuTask::~AiCpuTask() {
  FreeHbm(args_);
  FreeHbm(io_addr_);
  if (dynamic_flag_) {
    FreeHbm(workspace_addr_);
  }
  FreeHbm(copy_workspace_buf_);
  FreeHbm(copy_ioaddr_dev_);
  FreeHbm(copy_input_release_flag_dev_);
  FreeHbm(copy_input_data_size_dev_);
  FreeHbm(copy_input_src_dev_);
  FreeHbm(copy_input_dst_dev_);
  FreeHbm(copy_task_args_buf_);
  for (auto summary : output_summary_) {
    FreeHbm(summary);
  }
  for (auto out_shape : out_shape_hbm_) {
    FreeHbm(out_shape);
  }
}

Status AiCpuTask::LaunchKernel(rtStream_t stream) {
  GELOGD("Start to launch kernel. task = %s", this->op_type_.c_str());
  auto ret = rtMemcpyAsync(io_addr_,
                           io_addr_size_,
                           io_addr_host_.data(),
                           io_addr_host_.size() * sizeof(void *),
                           RT_MEMCPY_HOST_TO_DEVICE_EX,
                           stream);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMemcpyAsync workspace data failed. ret = %d, task = %s", ret, this->op_type_.c_str());
    return RT_FAILED;
  }

  GELOGI("To invoke rtKernelLaunchEx. task = %s", this->op_type_.c_str());
  ret = rtKernelLaunchEx(args_, arg_size_, 0, stream);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Invoke rtKernelLaunch failed. ret = %d, task = %s", ret, this->op_type_.c_str());
    return RT_FAILED;
  }
  GELOGI("[TASK_INFO] %lu/%s", kernel_id_, op_type_.c_str());

  auto status = OpenDump(stream);
  if (status != SUCCESS) {
    GELOGE(status, "Open dump failed in aicpu single op %s", this->op_type_.c_str());
    return status;
  }

  GELOGD("Done launch kernel successfully. task = %s", this->op_type_.c_str());
  return SUCCESS;
}

Status AiCpuTask::PrepareCopyInputs(vector<DataBuffer> &outputs) {
  std::vector<uint64_t> copy_input_release_flag;
  std::vector<uint64_t> copy_input_data_size;
  std::vector<uint64_t> copy_input_src;
  std::vector<uint64_t> copy_input_dst;

  for (size_t i = 0; i < num_outputs_; ++i) {
    const auto &summary = output_summary_host_[i];
    GELOGI("Node out[%zu] summary, shape data=0x%lx, shape data size=%lu, raw data=0x%lx, raw data size=%lu.",
           i, summary.shape_data_ptr, summary.shape_data_size,
           summary.raw_data_ptr, summary.raw_data_size);
    auto output = outputs[i];
    copy_input_release_flag.emplace_back(kReleaseFlag);
    if (summary.raw_data_size > 0) {
      copy_input_data_size.emplace_back(output.length);
    } else {
      copy_input_data_size.emplace_back(summary.raw_data_size);
    }
    copy_input_src.emplace_back(summary.raw_data_ptr);
    copy_input_dst.emplace_back(reinterpret_cast<uintptr_t>(output.data));

    const auto &shape_buffer = out_shape_hbm_[i];
    copy_input_release_flag.emplace_back(kReleaseFlag);
    copy_input_data_size.emplace_back(summary.shape_data_size);
    copy_input_src.emplace_back(summary.shape_data_ptr);
    copy_input_dst.emplace_back(reinterpret_cast<uintptr_t>(shape_buffer));
  }

  const size_t copy_input_buf_len = num_outputs_ * kCopyNum * sizeof(uint64_t);

  GE_CHK_RT_RET(rtMemcpy(copy_input_release_flag_dev_, copy_input_buf_len,
                         copy_input_release_flag.data(), copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_data_size_dev_, copy_input_buf_len,
                         copy_input_data_size.data(), copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_src_dev_, copy_input_buf_len,
                         copy_input_src.data(), copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_dst_dev_, copy_input_buf_len,
                         copy_input_dst.data(), copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AiCpuTask::ReadResultSummaryAndPrepareMemory() {
  for (size_t i = 0; i < num_outputs_; ++i) {
    auto &result_summary = output_summary_host_[i];

    GE_CHK_RT_RET(rtMemcpy(&result_summary, sizeof(aicpu::FWKAdapter::ResultSummary),
                           output_summary_[i], sizeof(aicpu::FWKAdapter::ResultSummary),
                           RT_MEMCPY_DEVICE_TO_HOST));
    auto shape_data_size = result_summary.shape_data_size;
    void *shape_buffer = nullptr;
    if (shape_data_size > 0) {
      GE_CHK_RT_RET(rtMalloc(&shape_buffer, shape_data_size, RT_MEMORY_HBM));
    }
    out_shape_hbm_.emplace_back(shape_buffer);
  }
  return SUCCESS;
}

Status AiCpuTask::CopyDataToHbm(vector<DataBuffer> &outputs,
                                rtStream_t stream) {
  GE_CHK_STATUS_RET_NOLOG(PrepareCopyInputs(outputs));

  GE_CHK_RT_RET(rtKernelLaunchEx(copy_task_args_buf_, sizeof(STR_FWK_OP_KERNEL),
                                 RT_KERNEL_DEFAULT, stream));
  GE_CHK_RT_RET(rtStreamSynchronize(stream));
  return SUCCESS;
}

Status AiCpuTask::UpdateShapeByHbmBuffer(vector<GeTensorDesc> &output_desc) {
  for (size_t i = 0; i < num_outputs_; ++i) {
    const auto &result_summary = output_summary_host_[i];
    std::vector<int64_t> shape_dims;
    if (result_summary.shape_data_size > 0) {
      const auto &shape_hbm = out_shape_hbm_[i];

      uint32_t dim_num = result_summary.shape_data_size / sizeof(int64_t);
      std::unique_ptr<int64_t[]> shape_addr(new(std::nothrow) int64_t[dim_num]());
      GE_CHECK_NOTNULL(shape_addr);
      GE_CHK_RT_RET(rtMemcpy(shape_addr.get(), result_summary.shape_data_size,
                             shape_hbm, result_summary.shape_data_size, RT_MEMCPY_DEVICE_TO_HOST));

      for (uint32_t dim_idx = 0; dim_idx < dim_num; ++dim_idx) {
        shape_dims.emplace_back(shape_addr[dim_idx]);
        GELOGD("Node [%zu]th output dim[%u]=%ld.", i, dim_idx, shape_addr[dim_idx]);
      }
    }

    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(GeShape(shape_dims), output_desc[i]),
                      "AiCpuTask update [%zu]th output shape failed.", i);
  }
  return SUCCESS;
}

Status AiCpuTask::UpdateShapeAndDataByResultSummary(vector<GeTensorDesc> &output_desc,
                                                    vector<DataBuffer> &outputs,
                                                    rtStream_t stream) {
  if (num_outputs_ == 0) {
    GELOGI("Output num is 0, there is no need to update the output and size.");
    return SUCCESS;
  }

  GELOGI("Update shape and data by result summary begin.");

  for (auto out_shape : out_shape_hbm_) {
    FreeHbm(out_shape);
  }
  out_shape_hbm_.clear();
  GE_CHK_STATUS_RET(ReadResultSummaryAndPrepareMemory(),
                    "Read ResultSummary and update output shape failed.");

  GE_CHK_STATUS_RET(CopyDataToHbm(outputs, stream),
                    "Copy data to output failed.");

  GE_CHK_STATUS_RET(UpdateShapeByHbmBuffer(output_desc),
                    "Update shape by hbm buffer failed.");

  for (auto out_shape : out_shape_hbm_) {
    FreeHbm(out_shape);
  }
  out_shape_hbm_.clear();

  GELOGI("Update shape and data by result summary end.");
  return SUCCESS;
}

Status AiCpuTask::InitForSummaryAndCopy() {
  if (unknown_type_ != DEPEND_COMPUTE || num_outputs_ == 0) {
    GELOGI("Unknown_type is %d, output num is %zu.", unknown_type_, num_outputs_);
    return SUCCESS;
  }

  output_summary_.resize(num_outputs_);
  constexpr auto result_summary_size = sizeof(aicpu::FWKAdapter::ResultSummary);
  for (size_t i = 0; i < num_outputs_; ++i) {
    GE_CHK_RT_RET(rtMalloc(&output_summary_[i], result_summary_size, RT_MEMORY_HBM));
  }
  output_summary_host_.resize(num_outputs_);

  const size_t copy_input_buf_len = num_outputs_ * kCopyNum * sizeof(uint64_t);

  GE_CHK_RT_RET(rtMalloc(&copy_input_release_flag_dev_, copy_input_buf_len, RT_MEMORY_HBM));
  GE_CHK_RT_RET(rtMalloc(&copy_input_data_size_dev_, copy_input_buf_len, RT_MEMORY_HBM));
  GE_CHK_RT_RET(rtMalloc(&copy_input_src_dev_, copy_input_buf_len, RT_MEMORY_HBM));
  GE_CHK_RT_RET(rtMalloc(&copy_input_dst_dev_, copy_input_buf_len, RT_MEMORY_HBM));

  GE_CHK_RT_RET(rtMalloc(&copy_task_args_buf_, sizeof(STR_FWK_OP_KERNEL), RT_MEMORY_HBM));

  std::vector<uint64_t> copy_io_addr;
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_release_flag_dev_));
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_data_size_dev_));
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_src_dev_));
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_dst_dev_));

  const auto copy_io_addr_size = sizeof(uint64_t) * copy_io_addr.size();

  GE_CHK_RT_RET(rtMalloc(&copy_ioaddr_dev_, copy_io_addr_size, RT_MEMORY_HBM));

  GE_CHK_RT_RET(rtMemcpy(copy_ioaddr_dev_, copy_io_addr_size,
                         copy_io_addr.data(), copy_io_addr_size, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AiCpuTask::SetMemCopyTask(const domi::KernelExDef &kernel_def) {
  if (kernel_def.args_size() > sizeof(STR_FWK_OP_KERNEL)) {
    GELOGE(PARAM_INVALID, "sizeof STR_FWK_OP_KERNEL is: %lu, but args_size is: %d",
           sizeof(STR_FWK_OP_KERNEL), kernel_def.args_size());
    return PARAM_INVALID;
  }
  GE_CHK_RT_RET(rtMalloc(&copy_workspace_buf_, kernel_def.task_info_size(), RT_MEMORY_HBM));
  GE_CHK_RT_RET(rtMemcpy(copy_workspace_buf_, kernel_def.task_info_size(),
                         kernel_def.task_info().data(), kernel_def.task_info_size(), RT_MEMCPY_HOST_TO_DEVICE));

  STR_FWK_OP_KERNEL aicpu_task = {0};
  auto sec_ret = memcpy_s(&aicpu_task, sizeof(STR_FWK_OP_KERNEL),
                          kernel_def.args().data(), kernel_def.args().size());
  if (sec_ret != EOK) {
    GELOGE(FAILED, "memcpy failed, ret: %d", sec_ret);
    return FAILED;
  }

  aicpu_task.fwkKernelBase.fwk_kernel.inputOutputAddr = reinterpret_cast<uintptr_t>(copy_ioaddr_dev_);
  aicpu_task.fwkKernelBase.fwk_kernel.workspaceBaseAddr = reinterpret_cast<uintptr_t>(copy_workspace_buf_);
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoAddr = 0;
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoLen = 0;

  GE_CHK_RT_RET(rtMemcpy(copy_task_args_buf_, sizeof(STR_FWK_OP_KERNEL),
                         &aicpu_task, sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AiCpuTask::LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                               const std::vector<DataBuffer> &input_buffers,
                               std::vector<GeTensorDesc> &output_desc,
                               std::vector<DataBuffer> &output_buffers,
                               rtStream_t stream) {
  GE_CHK_STATUS_RET_NOLOG(UpdateExtInfo(input_desc, output_desc, stream));
  if (unknown_type_ == DEPEND_COMPUTE) {
    std::vector<DataBuffer> summary_buffers;
    for (size_t i = 0; i < num_outputs_; ++i) {
      summary_buffers.emplace_back(output_summary_[i], sizeof(aicpu::FWKAdapter::ResultSummary), false);
    }
    GE_CHK_STATUS_RET_NOLOG(UpdateIoAddr(input_buffers, summary_buffers));
  } else {
    GE_CHK_STATUS_RET_NOLOG(UpdateIoAddr(input_buffers, output_buffers));
  }

  GE_CHK_STATUS_RET_NOLOG(LaunchKernel(stream));
  if (unknown_type_ == DEPEND_SHAPE_RANGE) {
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GE_CHK_STATUS_RET_NOLOG(UpdateOutputShape(output_desc));
  } else if (unknown_type_ == DEPEND_COMPUTE) {
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GE_CHK_STATUS_RET_NOLOG(UpdateShapeAndDataByResultSummary(output_desc, output_buffers, stream));
  }

  return SUCCESS;
}

Status AiCpuBaseTask::UpdateArgTable(const SingleOpModelParam &param) {
  // aicpu do not have workspace, for now
  return DoUpdateArgTable(param, false);
}

uint32_t AiCpuBaseTask::GetTaskType() const { return kTaskTypeAicpu; }

void AiCpuTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = reinterpret_cast<uintptr_t *>(io_addr_host_.data());
  arg_count = io_addr_host_.size();
}

void AiCpuCCTask::SetKernelArgs(std::unique_ptr<uint8_t[]> args, size_t arg_size) {
  args_ = std::move(args);
  arg_size_ = arg_size;
  // The blockdim value is defult "1" for rtCpuKernelLaunch
  block_dim_ = 1;
}

void AiCpuCCTask::SetSoName(const std::string &so_name) { so_name_ = so_name; }

void AiCpuCCTask::SetkernelName(const std::string &kernel_Name) { kernel_name_ = kernel_Name; }

void AiCpuCCTask::SetIoAddr(uintptr_t *io_addr) { io_addr_ = io_addr; }

const void *AiCpuCCTask::GetArgs() const { return args_.get(); }

size_t AiCpuCCTask::GetArgSize() const { return arg_size_; }

AiCpuCCTask::~AiCpuCCTask() {
}

Status AiCpuCCTask::LaunchKernel(rtStream_t stream) {
  GELOGI("To invoke rtCpuKernelLaunch. block_dim = %u, so_name is %s, kernel_name is %s", block_dim_, so_name_.data(),
         kernel_name_.data());
  // sm_desc is nullptr, because l2 buffer does not support
  auto *sm_desc = reinterpret_cast<rtSmDesc_t *>(sm_desc_);
  auto ret = rtCpuKernelLaunchWithFlag(static_cast<const void *>(so_name_.data()),
                                       static_cast<const void *>(kernel_name_.data()),
                                       block_dim_, args_.get(), static_cast<uint32_t>(arg_size_),
                                       sm_desc, stream, dump_flag_);
  if (ret != RT_ERROR_NONE) {
    GELOGE(ret, "Invoke rtCpuKernelLaunch failed. ret = %d", ret);
    return ret;
  }
  GELOGI("[TASK_INFO] %lu/%s", kernel_id_, op_type_.c_str());
  GELOGD("Invoke rtCpuKernelLaunch succeeded");
  auto status = OpenDump(stream);
  if (status != SUCCESS) {
    GELOGE(status, "Open dump failed in the aicpucc single op %s", this->kernel_name_.c_str());
    return status;
  }

  return SUCCESS;
}

Status AiCpuCCTask::LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                                 const std::vector<DataBuffer> &input_buffers,
                                 std::vector<GeTensorDesc> &output_desc,
                                 std::vector<DataBuffer> &output_buffers,
                                 rtStream_t stream) {
  GE_CHK_STATUS_RET_NOLOG(UpdateExtInfo(input_desc, output_desc, stream));
  GE_CHK_STATUS_RET_NOLOG(UpdateIoAddr(input_buffers, output_buffers));
  GE_CHK_STATUS_RET_NOLOG(LaunchKernel(stream));
  if (unknown_type_ == DEPEND_SHAPE_RANGE) {
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GE_CHK_STATUS_RET_NOLOG(UpdateOutputShape(output_desc));
  }

  return SUCCESS;
}

void AiCpuCCTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = io_addr_;
  arg_count = io_addr_num_;
}
}  // namespace ge
