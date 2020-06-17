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

#include "hybrid/node_executor/aicpu/aicpu_node_executor.h"
#include "common/formats/formats.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/model/hybrid_model.h"
#include "init/gelib.h"

namespace ge {
namespace hybrid {
using aicpu::FWKAdapter::ExtInfo;
namespace {
// mem need release
constexpr uint64_t kReleaseFlag = 1;

// max dim count is 8.
constexpr uint32_t kMaxDimCount = 8;

// if dim count is not reach kMaxDimCount, use INT64_MIN to mark dim end.
constexpr int64_t kDimEndFlag = INT64_MIN;

struct MaxShape {
  int64_t dims[kMaxDimCount] = {0};
};
}  // namespace
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICPU_TF, AiCpuNodeExecutor);

Status AicpuTfNodeTask::AllocTensorBuffer(size_t size, std::unique_ptr<TensorBuffer> &tensor_buffer) {
  auto allocator = NpuMemoryAllocator::GetAllocator();
  GE_CHECK_NOTNULL(allocator);
  tensor_buffer = TensorBuffer::Create(allocator, size);
  GE_CHECK_NOTNULL(tensor_buffer);
  return SUCCESS;
}

Status AicpuTfNodeTask::InitExtInfo() {
  // exit info, 0: op type
  size_t ext_info_size = sizeof(ExtInfo) + sizeof(uint32_t);
  ext_info_num_ = 1;
  // exit info 1:input shape, 2:output shape
  if (input_num_ > 0) {
    ext_info_size += sizeof(ExtInfo) + input_num_ * sizeof(MaxShape);
    ++ext_info_num_;
  }

  // exit info 2:output shape
  if ((unknown_type_ != DEPEND_COMPUTE) && (output_num_ > 0)) {
    ext_info_size += sizeof(ExtInfo) + output_num_ * sizeof(MaxShape);
    ++ext_info_num_;
  }

  GE_CHK_STATUS_RET(AllocTensorBuffer(ext_info_size, ext_info_addr_dev_),
                    "Node %s alloc buffer for ext info failed, size=%zu.", node_->GetName().c_str(), ext_info_size);

  auto ext_info_dev_base = reinterpret_cast<uintptr_t>(ext_info_addr_dev_->GetData());
  ext_info_addr_host_.reset(new (std::nothrow) uint8_t[ext_info_size]);
  GE_CHECK_NOTNULL(ext_info_addr_host_);

  size_t ext_info_type_offset = ext_info_num_ * sizeof(ExtInfo);
  size_t ext_info_input_shape_offset = ext_info_type_offset + sizeof(uint32_t);

  auto ext_info_host_buf = ext_info_addr_host_.get();

  auto ext_info_type = reinterpret_cast<ExtInfo *>(ext_info_host_buf);
  ext_info_type->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  ext_info_type->infoLen = sizeof(uint32_t);
  ext_info_type->infoAddr = ext_info_dev_base + ext_info_type_offset;
  // set unknown shape type
  auto unkonw_shape_type_addr = reinterpret_cast<uint32_t *>(ext_info_host_buf + ext_info_type_offset);
  *unkonw_shape_type_addr = unknown_type_;

  if (input_num_ > 0) {
    auto ext_info_input = reinterpret_cast<ExtInfo *>(ext_info_host_buf + sizeof(ExtInfo));
    ext_info_input->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
    ext_info_input->infoLen = input_num_ * sizeof(MaxShape);
    ext_info_input->infoAddr = ext_info_dev_base + ext_info_input_shape_offset;
  }
  if ((unknown_type_ != DEPEND_COMPUTE) && (output_num_ > 0)) {
    size_t ext_info_output_shape_offset = ext_info_input_shape_offset + input_num_ * sizeof(MaxShape);
    auto ext_info_output = reinterpret_cast<ExtInfo *>(ext_info_host_buf + (ext_info_num_ - 1) * sizeof(ExtInfo));
    ext_info_output->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE;
    ext_info_output->infoLen = output_num_ * sizeof(MaxShape);
    ext_info_output->infoAddr = ext_info_dev_base + ext_info_output_shape_offset;
  }

  GE_CHK_RT_RET(rtMemcpy(ext_info_addr_dev_->GetData(), ext_info_addr_dev_->GetSize(), ext_info_host_buf, ext_info_size,
                         RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AicpuTfNodeTask::InitForDependComputeTask() {
  if ((unknown_type_ != DEPEND_COMPUTE) || (output_num_ == 0)) {
    GELOGI("node %s type %s unknown_type is %d, output num is %zu.", node_->GetName().c_str(), node_->GetType().c_str(),
           unknown_type_, output_num_);
    return SUCCESS;
  }

  output_summary_.resize(output_num_);
  constexpr auto result_summary_size = sizeof(aicpu::FWKAdapter::ResultSummary);
  for (size_t i = 0; i < output_num_; ++i) {
    GE_CHK_STATUS_RET(AllocTensorBuffer(result_summary_size, output_summary_[i]),
                      "Node %s alloc buffer for ext info failed, size=%zu.", node_->GetName().c_str(),
                      result_summary_size);
  }
  output_summary_host_.resize(output_num_);

  // init for mem copy task
  // copy task need copy output_data and output_shape, max len is 2 * output_num
  const size_t copy_input_buf_len = output_num_ * 2 * sizeof(uint64_t);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_release_flag_dev_),
                    "Node %s alloc copy task input release_flag failed, size=%zu", node_->GetName().c_str(),
                    copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_data_size_dev_),
                    "Node %s alloc copy task input data_size failed, size=%zu", node_->GetName().c_str(),
                    copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_src_dev_),
                    "Node %s alloc copy task input src failed, size=%zu", node_->GetName().c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_dst_dev_),
                    "Node %s alloc copy task input dst failed, size=%zu", node_->GetName().c_str(), copy_input_buf_len);

  // copy task args buf
  GE_CHK_STATUS_RET(AllocTensorBuffer(sizeof(STR_FWK_OP_KERNEL), copy_task_args_buf_),
                    "Node %s alloc copy task args buf failed, size=%zu", node_->GetName().c_str(),
                    sizeof(STR_FWK_OP_KERNEL));

  std::vector<uint64_t> copy_io_addr;
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_release_flag_dev_->GetData()));
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_data_size_dev_->GetData()));
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_src_dev_->GetData()));
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_dst_dev_->GetData()));

  // mem copy op has 4 inputs and 0 output.
  const auto copy_io_addr_size = sizeof(uint64_t) * copy_io_addr.size();

  // can alloc in init, it can reuse
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_io_addr_size, copy_ioaddr_dev_),
                    "Node %s alloc copy task io buf failed, size=%zu", node_->GetName().c_str(), copy_io_addr_size);

  GE_CHK_RT_RET(rtMemcpy(copy_ioaddr_dev_->GetData(), copy_io_addr_size, &copy_io_addr[0], copy_io_addr_size,
                         RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AicpuTfNodeTask::Init(const HybridModel &model) {
  auto node_name = node_->GetName();
  GELOGI("AicpuTfNodeTask[%s] Init Start.", node_name.c_str());
  auto op_desc = node_->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  const auto node_item = model.GetNodeItem(node_);
  GE_CHECK_NOTNULL(node_item);
  unknown_type_ = node_item->shape_inference_type;

  auto &kernel_ex_def = task_def_.kernel_ex();

  auto kernel_workspace_size = static_cast<size_t>(kernel_ex_def.task_info_size());
  GE_CHK_STATUS_RET(AllocTensorBuffer(kernel_workspace_size, kernel_workspace_),
                    "Node %s alloc buffer for kernel workspace failed, size=%zu.", node_name.c_str(),
                    kernel_workspace_size);

  GE_CHK_RT_RET(rtMemcpy(kernel_workspace_->GetData(), kernel_workspace_size, kernel_ex_def.task_info().data(),
                         static_cast<uint64_t>(kernel_ex_def.task_info_size()), RT_MEMCPY_HOST_TO_DEVICE));
  input_num_ = op_desc->GetInputsSize();
  output_num_ = op_desc->GetOutputsSize();
  size_t input_output_size = (input_num_ + output_num_) * sizeof(uint64_t);
  if (input_output_size > 0) {
    // alloc input output addr buf
    GE_CHK_STATUS_RET(AllocTensorBuffer(input_output_size, input_output_addr_),
                      "Node %s alloc buffer for input output addr failed, size=%zu.", node_name.c_str(),
                      input_output_size);
  }

  // init ext info
  GE_CHK_STATUS_RET(InitExtInfo(), "Task %s init ext info failed.", node_name.c_str());
  GE_CHK_STATUS_RET(InitForDependComputeTask(), "Task %s init for depend compute task failed.", node_name.c_str());

  // build fwk_op_kernel.
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(sizeof(STR_FWK_OP_KERNEL) < kernel_ex_def.args_size(), return FAILED,
                                 "sizeof STR_FWK_OP_KERNEL is: %zu, but args_size is: %u", sizeof(STR_FWK_OP_KERNEL),
                                 kernel_ex_def.args_size());

  STR_FWK_OP_KERNEL fwk_op_kernel = {0};
  errno_t sec_ret = memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL), kernel_ex_def.args().data(),
                             static_cast<size_t>(kernel_ex_def.args_size()));
  GE_CHK_BOOL_EXEC(sec_ret == EOK, return INTERNAL_ERROR, "memcpy fwk_op_kernel failed, ret: %d", sec_ret);

  fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr = reinterpret_cast<uintptr_t>(kernel_workspace_->GetData());
  fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = reinterpret_cast<uintptr_t>(input_output_addr_->GetData());
  // set ext info addr and ext info num
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = reinterpret_cast<uintptr_t>(ext_info_addr_dev_->GetData());
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoNum = ext_info_num_;

  // get step_id_addr
  auto var_tensor = model.GetVariable(NODE_NAME_GLOBAL_STEP);
  uint64_t step_id_addr = 0;
  if (var_tensor != nullptr) {
    step_id_addr = reinterpret_cast<uintptr_t>(var_tensor->GetData());
  }

  fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = step_id_addr;

  auto session_id = fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID;
  GE_CHK_STATUS_RET(EnsureSessionCreated(session_id), "session id %lu create failed.", session_id);

  // alloc kernel_buf_ and copy to device.
  GE_CHK_STATUS_RET(AllocTensorBuffer(sizeof(STR_FWK_OP_KERNEL), kernel_buf_),
                    "Node %s alloc buffer for kernel buf failed, size=%zu.", node_name.c_str(),
                    sizeof(STR_FWK_OP_KERNEL));

  GE_CHK_RT_RET(rtMemcpy(kernel_buf_->GetData(), sizeof(STR_FWK_OP_KERNEL), &fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL),
                         RT_MEMCPY_HOST_TO_DEVICE));

  GELOGI("AicpuTfNodeTask[%s] init end.", node_name.c_str());
  return SUCCESS;
}

Status AicpuTfNodeTask::EnsureSessionCreated(uint64_t session_id) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  GE_CHK_STATUS_RET(model_manager->CreateAicpuSession(session_id), "Create aicpu session %u failed", session_id);
  return SUCCESS;
}

Status AicpuTfNodeTask::SetShapeToBuf(const GeShape &shape, int64_t buf[], uint32_t buf_size) {
  auto node_name = node_->GetName();
  uint32_t index = 0;
  int64_t shape_size = shape.GetDimNum();
  if (shape_size > buf_size) {
    GELOGI("SetShapeToBuf[%s] failed, as shape size %ld is over %u.", node_name.c_str(), shape_size, buf_size);
    return PARAM_INVALID;
  }
  for (; index < shape_size; ++index) {
    buf[index] = shape.GetDim(index);
  }
  if (index < buf_size) {
    buf[index] = kDimEndFlag;
  }
  return SUCCESS;
}

Status AicpuTfNodeTask::UpdateShapeToOutputDesc(const GeShape &shape_new, size_t output_index,
                                                GeTensorDescPtr &output_desc) {
  auto node_name = node_->GetName();
  auto shape_old = output_desc->GetShape();
  output_desc->SetShape(shape_new);
  GELOGI("Update node[%s] out[%zu] shape from %s to %s.", node_name.c_str(), output_index, shape_old.ToString().c_str(),
         shape_new.ToString().c_str());

  auto origin_shape_old = output_desc->GetOriginShape();
  auto origin_format = output_desc->GetOriginFormat();
  auto format = output_desc->GetFormat();
  if (origin_format == format) {
    output_desc->SetOriginShape(shape_new);
    return SUCCESS;
  }
  // if format is not same need convert shape
  std::vector<int64_t> origin_dims_new;
  auto trans_ret =
    formats::TransShape(format, shape_new.GetDims(), output_desc->GetDataType(), origin_format, origin_dims_new);
  GE_CHK_STATUS_RET(trans_ret,
                    "Node[%s] out[%zu] originFormat[%d] is not same as format[%d], but TransShape failed, shape=%s.",
                    node_name.c_str(), output_index, origin_format, format, shape_new.ToString().c_str());
  auto origin_shape_new = GeShape(origin_dims_new);
  output_desc->SetOriginShape(origin_shape_new);
  GELOGI("Node[%s] out[%zu] originFormat[%d] is not same as format[%d], need update from %s ro %s.", node_name.c_str(),
         output_index, origin_format, format, origin_shape_old.ToString().c_str(), origin_shape_new.ToString().c_str());
  return SUCCESS;
}

Status AicpuTfNodeTask::ReadResultSummaryAndPrepareMemory(TaskContext &context,
                                                          std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  for (size_t i = 0; i < output_num_; ++i) {
    auto &result_summary = output_summary_host_[i];
    GE_CHK_RT_RET(rtMemcpy(&result_summary, sizeof(aicpu::FWKAdapter::ResultSummary), output_summary_[i]->GetData(),
                           output_summary_[i]->GetSize(), RT_MEMCPY_DEVICE_TO_HOST));

    GELOGI(
      "Node[%s] out[%zu] result summary addr=%p,"
      " shape_data_ptr=0x%lx, shape_data_size=%lu, raw_data_ptr=0x%lx, raw_data_size=%lu.",
      node_->GetName().c_str(), i, output_summary_[i]->GetData(), result_summary.shape_data_ptr,
      result_summary.shape_data_size, result_summary.raw_data_ptr, result_summary.raw_data_size);

    auto raw_data_size = result_summary.raw_data_size;
    std::unique_ptr<TensorBuffer> tensor_buffer;
    GE_CHK_STATUS_RET(AllocTensorBuffer(raw_data_size, tensor_buffer), "alloc tensor buffer failed, raw_data_size=%lu",
                      raw_data_size);
    auto status = context.SetOutput(i, TensorValue(std::shared_ptr<TensorBuffer>(tensor_buffer.release())));
    GE_CHK_STATUS_RET(status, "SetOutput %zu failed.", i);

    auto shape_data_size = result_summary.shape_data_size;
    std::unique_ptr<TensorBuffer> shape_buffer;
    GE_CHK_STATUS_RET(AllocTensorBuffer(shape_data_size, shape_buffer),
                      "alloc shape buffer failed, shape_data_size=%lu", shape_data_size);
    out_shape_hbm.emplace_back(std::move(shape_buffer));
  }
  return SUCCESS;
}

Status AicpuTfNodeTask::CopyDataToHbm(TaskContext &context,
                                      const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  GE_CHK_BOOL_RET_STATUS(out_shape_hbm.size() == output_num_, INTERNAL_ERROR,
                         "Node %s has %zu outputs but out shape is %zu", node_->GetName().c_str(), output_num_,
                         out_shape_hbm.size());

  std::vector<uint64_t> copy_input_release_flag;
  std::vector<uint64_t> copy_input_data_size;
  std::vector<uint64_t> copy_input_src;
  std::vector<uint64_t> copy_input_dst;

  for (size_t i = 0; i < output_num_; ++i) {
    const auto &summary = output_summary_host_[i];
    GELOGI("node[%s] [%zu]th output summary, shape data=%lx, shape data size=%lu, raw data=%lx, raw data size=%lu.",
           node_->GetName().c_str(), i, summary.shape_data_ptr, summary.shape_data_size, summary.raw_data_ptr,
           summary.raw_data_size);
    if (summary.raw_data_size > 0) {
      auto output = context.GetOutput(i);
      GE_CHECK_NOTNULL(output);
      GE_CHECK_NOTNULL(output->GetData());
      copy_input_release_flag.emplace_back(kReleaseFlag);
      copy_input_data_size.emplace_back(summary.raw_data_size);
      copy_input_src.emplace_back(summary.raw_data_ptr);
      copy_input_dst.emplace_back(reinterpret_cast<uintptr_t>(output->GetData()));
    }

    if (summary.shape_data_size > 0) {
      const auto &shape_buffer = out_shape_hbm[i];
      GE_CHECK_NOTNULL(shape_buffer);
      GE_CHECK_NOTNULL(shape_buffer->GetData());
      copy_input_release_flag.emplace_back(kReleaseFlag);
      copy_input_data_size.emplace_back(summary.shape_data_size);
      copy_input_src.emplace_back(summary.shape_data_ptr);
      copy_input_dst.emplace_back(reinterpret_cast<uintptr_t>(shape_buffer->GetData()));
    }
  }

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(copy_input_release_flag.empty(), return INTERNAL_ERROR, "Node %s need copy num is 0",
                                 node_->GetName().c_str());

  auto copy_num = copy_input_release_flag.size();
  STR_FWK_OP_KERNEL aicpu_task = {0};
  std::string task_info;
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_->GetName().c_str(), "[GenMemCopyTask] Start");
  GE_CHK_STATUS_RET_NOLOG(GenMemCopyTask(copy_num, aicpu_task, task_info));
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_->GetName().c_str(), "[GenMemCopyTask] End");

  // copy task need copy output and output shape
  const size_t copy_input_buf_len = copy_num * sizeof(uint64_t);

  GE_CHK_RT_RET(rtMemcpy(copy_input_release_flag_dev_->GetData(), copy_input_release_flag_dev_->GetSize(),
                         &copy_input_release_flag[0], copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_data_size_dev_->GetData(), copy_input_data_size_dev_->GetSize(),
                         &copy_input_data_size[0], copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_src_dev_->GetData(), copy_input_src_dev_->GetSize(), &copy_input_src[0],
                         copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_dst_dev_->GetData(), copy_input_dst_dev_->GetSize(), &copy_input_dst[0],
                         copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));

  std::unique_ptr<TensorBuffer> kernel_workspace_buf;
  GE_CHK_STATUS_RET(AllocTensorBuffer(task_info.size(), kernel_workspace_buf),
                    "Node %s alloc copy task workspace buf failed, size=%zu", node_->GetName().c_str(),
                    task_info.size());

  GE_CHK_RT_RET(rtMemcpy(kernel_workspace_buf->GetData(), task_info.size(), task_info.data(), task_info.size(),
                         RT_MEMCPY_HOST_TO_DEVICE));

  aicpu_task.fwkKernelBase.fwk_kernel.inputOutputAddr = reinterpret_cast<uintptr_t>(copy_ioaddr_dev_->GetData());
  aicpu_task.fwkKernelBase.fwk_kernel.workspaceBaseAddr = reinterpret_cast<uintptr_t>(kernel_workspace_buf->GetData());
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoAddr = 0;
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoNum = 0;

  GE_CHK_RT_RET(rtMemcpy(copy_task_args_buf_->GetData(), sizeof(STR_FWK_OP_KERNEL), &aicpu_task,
                         sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE));

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_->GetName().c_str(), "[LaunchCopy] Start");
  GE_CHK_RT_RET(rtKernelLaunchEx(copy_task_args_buf_->GetData(), sizeof(STR_FWK_OP_KERNEL), RT_KERNEL_DEFAULT,
                                 context.GetStream()));
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_->GetName().c_str(), "[LaunchCopy] End");

  GE_CHK_RT_RET(rtStreamSynchronize(context.GetStream()));
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_->GetName().c_str(), "[SynchronizeCopy] End");
  return SUCCESS;
}

Status AicpuTfNodeTask::GenMemCopyTask(uint64_t copy_num, STR_FWK_OP_KERNEL &task, string &task_info) {
  auto instance_ptr = ge::GELib::GetInstance();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(instance_ptr == nullptr || !instance_ptr->InitFlag(), return GE_CLI_GE_NOT_INITIALIZED,
                                 "GE is not initialized");

  static constexpr const char *const kKernelLibName = "aicpu_kernel";
  OpsKernelInfoStorePtr kernel_info = instance_ptr->OpsKernelManagerObj().GetOpsKernelInfoStore(kKernelLibName);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(kernel_info == nullptr, return FAILED, "Get op kernel info store failed");
  auto ret = kernel_info->GenMemCopyTask(copy_num, task, task_info);
  GE_CHK_STATUS_RET(ret, "call aicpu GenMemCopyTask failed, copy_num=%lu, ret=%u", copy_num, ret);
  return SUCCESS;
}

Status AicpuTfNodeTask::UpdateShapeByHbmBuffer(TaskContext &context,
                                               const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  GE_CHK_BOOL_RET_STATUS(out_shape_hbm.size() == output_num_, INTERNAL_ERROR,
                         "Node %s has %zu outputs but out shape is %zu", node_->GetName().c_str(), output_num_,
                         out_shape_hbm.size());
  auto op_desc = node_->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (size_t i = 0; i < output_num_; ++i) {
    const auto &result_summary = output_summary_host_[i];
    auto output_desc = op_desc->MutableOutputDesc(i);
    std::vector<int64_t> shape_dims;
    if (result_summary.shape_data_size > 0) {
      const auto &shape_hbm = out_shape_hbm[i];
      GE_CHK_BOOL_RET_STATUS((result_summary.shape_data_size % sizeof(int64_t) == 0), INTERNAL_ERROR,
                             "node %s %zuth output shape data size is %lu is not divided by int64_t.",
                             node_->GetName().c_str(), i, result_summary.shape_data_size);
      uint32_t dim_num = result_summary.shape_data_size / sizeof(int64_t);
      GELOGI("node %s %zuth output dim num=%lu.", node_->GetName().c_str(), i, dim_num);
      std::unique_ptr<int64_t[]> shape_addr(new (std::nothrow) int64_t[dim_num]());
      GE_CHECK_NOTNULL(shape_addr);
      GE_CHK_RT_RET(rtMemcpy(shape_addr.get(), result_summary.shape_data_size, shape_hbm->GetData(),
                             shape_hbm->GetSize(), RT_MEMCPY_DEVICE_TO_HOST));
      for (uint32_t dim_idx = 0; dim_idx < dim_num; ++dim_idx) {
        shape_dims.emplace_back(shape_addr[dim_idx]);
        GELOGD("node %s %zuth output dim[%u]=%lu.", node_->GetName().c_str(), i, dim_idx, shape_addr[dim_idx]);
      }
    }
    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(GeShape(shape_dims), i, output_desc),
                      "update node %s %uth output shape failed.", node_->GetName().c_str(), i);
  }
  return SUCCESS;
}

Status AicpuTfNodeTask::UpdateOutputShapeFromExtInfo() {
  auto node_name = node_->GetName();
  if (output_num_ == 0) {
    GELOGI("Task [%s] output_num is 0, no need reset output shape.", node_name.c_str());
    return SUCCESS;
  }

  auto ext_output_shape_offset = ext_info_num_ * sizeof(ExtInfo) + sizeof(uint32_t) + input_num_ * sizeof(MaxShape);
  size_t ext_info_output_shape_len = output_num_ * sizeof(MaxShape);
  auto output_shape_host_buf = ext_info_addr_host_.get() + ext_output_shape_offset;
  auto output_shape_dev_buf = reinterpret_cast<uint8_t *>(ext_info_addr_dev_->GetData()) + ext_output_shape_offset;

  GE_CHK_RT_RET(rtMemcpy(output_shape_host_buf, ext_info_output_shape_len, output_shape_dev_buf,
                         ext_info_output_shape_len, RT_MEMCPY_DEVICE_TO_HOST));

  auto op_desc = node_->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  auto shapeBuf = reinterpret_cast<int64_t *>(output_shape_host_buf);
  for (uint32_t i = 0; i < output_num_; ++i) {
    std::vector<int64_t> dims;
    GetShapeFromBuf(shapeBuf + i * kMaxDimCount, kMaxDimCount, dims);
    auto output_desc = op_desc->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(GeShape(dims), i, output_desc),
                      "update node %s %uth output shape failed.", node_name.c_str(), i);
  }

  return SUCCESS;
}

Status AicpuTfNodeTask::UpdateShapeAndDataByResultSummary(TaskContext &context) {
  GELOGI("Task [%s] update shape and data by result summary begin.", node_->GetName().c_str());

  std::vector<std::unique_ptr<TensorBuffer>> out_shape_hbm;
  GE_CHK_STATUS_RET(ReadResultSummaryAndPrepareMemory(context, out_shape_hbm),
                    "node %s read ResultSummary and update output shape failed.", node_->GetName().c_str());

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_->GetName().c_str(),
                        "[ReadResultSummaryAndPrepareMemory] End");

  GE_CHK_STATUS_RET(CopyDataToHbm(context, out_shape_hbm), "node %s copy data to output failed.",
                    node_->GetName().c_str());

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_->GetName().c_str(), "[CopyDataToHbm] End");

  GE_CHK_STATUS_RET(UpdateShapeByHbmBuffer(context, out_shape_hbm), "node %s update shape by hbm buffer failed.",
                    node_->GetName().c_str());

  GELOGI("Task [%s] update shape and data by result summary end.", node_->GetName().c_str());
  return SUCCESS;
}

void AicpuTfNodeTask::GetShapeFromBuf(const int64_t buf[], uint32_t buf_size, std::vector<int64_t> &dims) {
  for (uint32_t index = 0; index < buf_size; ++index) {
    auto tmpDim = buf[index];
    if (tmpDim == kDimEndFlag) {
      break;
    }
    dims.emplace_back(tmpDim);
  }
}

Status AicpuTfNodeTask::UpdateArgs(TaskContext &context) {
  auto node_name = node_->GetName();
  GELOGI("AicpuTfNodeTask[%s] UpdateArgs begin. unknown_type=%d", node_name.c_str(), unknown_type_);
  auto op_desc = node_->GetOpDesc();
  auto io_nums = input_num_ + output_num_;
  if (io_nums == 0) {
    GELOGI("Node %s has no input and output, no need update args.", node_name.c_str());
    return SUCCESS;
  }

  vector<uint64_t> io_addrs(io_nums, 0UL);
  size_t ext_shape_nums = (unknown_type_ == DEPEND_COMPUTE) ? input_num_ : io_nums;
  vector<MaxShape> io_shapes(ext_shape_nums);

  uint32_t index = 0;
  for (size_t i = 0; i < input_num_; ++i, ++index) {
    auto inputData = context.GetInput(i);
    GE_CHECK_NOTNULL(inputData);
    auto input_desc = op_desc->MutableInputDesc(i);
    GE_CHECK_NOTNULL(input_desc);
    auto &shape = input_desc->MutableShape();

    GELOGD("io_addr[%u] = %p, size = %zu", index, inputData->GetData(), inputData->GetSize());
    io_addrs[index] = reinterpret_cast<uintptr_t>(inputData->GetData());
    GE_CHK_STATUS_RET(SetShapeToBuf(shape, io_shapes[index].dims, kMaxDimCount),
                      "task %s input[%zu] SetShapeToBuf failed.", node_name.c_str(), i);
  }

  if (unknown_type_ != DEPEND_COMPUTE) {
    // unknown type 4 do this in call back.
    GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
    for (size_t j = 0; j < output_num_; ++j, ++index) {
      auto outputData = context.GetOutput(j);
      GE_CHECK_NOTNULL(outputData);
      auto output_desc = op_desc->MutableOutputDesc(j);
      GE_CHECK_NOTNULL(output_desc);
      auto shape = output_desc->GetShape();

      // shape range need use range update shape
      if (unknown_type_ == DEPEND_SHAPE_RANGE) {
        std::vector<std::pair<int64_t, int64_t>> range;
        auto range_ret = output_desc->GetShapeRange(range);
        GE_CHK_BOOL_RET_STATUS(range_ret == GRAPH_SUCCESS, INTERNAL_ERROR,
                               "node %s has is shape range but get GetShapeRange failed, ret=%u.", node_name.c_str(),
                               range_ret);
        for (size_t k = 0; k < range.size(); ++k) {
          if (shape.GetDim(k) < 0 && k < range.size()) {
            GELOGD("node %s output[%zu] update dim[%zu] from %lu to range max %lu.", node_name.c_str(), j, k,
                   shape.GetDim(k), range[k].second);
            shape.SetDim(k, range[k].second);
          }
        }
      }

      GELOGD("io_addr[%u] = %p, size = %zu", index, outputData->GetData(), outputData->GetSize());
      io_addrs[index] = reinterpret_cast<uintptr_t>(outputData->GetData());
      GE_CHK_STATUS_RET(SetShapeToBuf(shape, io_shapes[index].dims, kMaxDimCount),
                        "task %s output[%zu] SetShapeToBuf failed.", node_name.c_str(), j);
    }
  } else {
    // unknown type 4 use result summary update ioaddr.
    GELOGI("AicpuTfNodeTask[%s] is unknown-shape, use ResultSummary as out-addr.", node_name.c_str());
    GE_CHK_BOOL_RET_STATUS(output_summary_.size() == output_num_, INTERNAL_ERROR,
                           "node %s has %zu output but %zu output summary.", node_name.c_str(), output_num_,
                           output_summary_.size());

    for (size_t j = 0; j < output_num_; ++j, ++index) {
      void *summary_addr = output_summary_[j]->GetData();
      io_addrs[index] = reinterpret_cast<uintptr_t>(summary_addr);
    }
  }

  // if has input and output, need copy to ioaddr
  if (io_nums > 0) {
    // copy input and output to device
    GE_CHK_RT_RET(rtMemcpy(input_output_addr_->GetData(), input_output_addr_->GetSize(), &io_addrs[0],
                           sizeof(uint64_t) * io_addrs.size(), RT_MEMCPY_HOST_TO_DEVICE));
  }

  // if has shape ext info, need copy to ext addr
  if (ext_shape_nums > 0) {
    uint32_t offset = ext_info_num_ * sizeof(ExtInfo) + sizeof(uint32_t);
    uint32_t len = sizeof(MaxShape) * ext_shape_nums;
    auto ext_addr_dev_base = reinterpret_cast<uint8_t *>(ext_info_addr_dev_->GetData()) + offset;
    // copy input and output shapes to device
    GE_CHK_RT_RET(rtMemcpy(ext_addr_dev_base, ext_info_addr_dev_->GetSize() - offset, &io_shapes[0], len,
                           RT_MEMCPY_HOST_TO_DEVICE));
  }

  GELOGI("AicpuTfNodeTask[%s] UpdateArgs end.", node_name.c_str());
  return SUCCESS;
}

Status AicpuTfNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  auto node_name = node_->GetName();
  GELOGI("AicpuTfNodeTask[%s] ExecuteAsync Start. unknown_type=%d.", node_name.c_str(), unknown_type_);

  uint32_t flag = RT_KERNEL_DEFAULT;
  GE_CHK_RT_RET(rtKernelLaunchEx(kernel_buf_->GetData(), kernel_buf_->GetSize(), flag, context.GetStream()));

  auto callback = [=, &context]() {
    GELOGI("AicpuTfNodeTask[%s] callback start.", node_->GetName().c_str());
    RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_->GetName().c_str(), "[TaskCallback] Start");
    Status callback_ret = SUCCESS;
    // check need update shape, call update shape.
    if (unknown_type_ == DEPEND_SHAPE_RANGE) {
      // check result
      callback_ret = UpdateOutputShapeFromExtInfo();
    } else if (unknown_type_ == DEPEND_COMPUTE) {
      callback_ret = UpdateShapeAndDataByResultSummary(context);
    }

    GELOGI("AicpuTfNodeTask[%s] refresh output complete, ret = %d.", node_->GetName().c_str(), callback_ret);
    RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_->GetName().c_str(), "[TaskCallback] End");

    if (done_callback != nullptr) {
      context.SetStatus(callback_ret);
      done_callback();
    }

    GELOGI("AicpuTfNodeTask[%s] callback end.", node_->GetName().c_str());
  };

  GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(callback));

  GELOGI("AicpuTfNodeTask[%s] ExecuteAsync end.", node_name.c_str());
  return SUCCESS;
}

Status AiCpuNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  // malloc HBM memory at Init, here just update them
  return task.UpdateArgs(context);
}

Status AiCpuNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node,
                                   std::shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  GELOGI("Node[%s] create task start.", node->GetName().c_str());
  auto task_defs = model.GetTaskDefs(node);
  GE_CHECK_NOTNULL(task_defs);
  GE_CHK_BOOL_EXEC((*task_defs).size() == 1, return PARAM_INVALID, "aicpu op[%s] task_def num[%zu] != 1",
                   node->GetName().c_str(), (*task_defs).size());
  auto aicpu_task = MakeShared<AicpuTfNodeTask>(node, (*task_defs)[0]);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(aicpu_task == nullptr, return MEMALLOC_FAILED,
                                 "create aicpuTfNodeTask for node %s failed", node->GetName().c_str());

  GE_CHK_STATUS_RET(aicpu_task->Init(model), "AicpuTfNodeTask %s Init failed.", node->GetName().c_str());

  task = std::move(aicpu_task);
  GELOGI("Node[%s] create task end.", node->GetName().c_str());
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
