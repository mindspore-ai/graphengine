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
#include "framework/common/taskdown_common.h"
#include "common/formats/formats.h"
#include "aicpu/common/aicpu_task_struct.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "graph/utils/node_utils.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/model/hybrid_model.h"
#include "opskernel_manager/ops_kernel_builder_manager.h"

namespace ge {
namespace hybrid {
namespace {
// mem need release
constexpr uint64_t kReleaseFlag = 1;
}
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICPU_TF, AiCpuNodeExecutor);
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICPU_CUSTOM, AiCpuNodeExecutor);

Status AicpuNodeTaskBase::AllocTensorBuffer(size_t size, std::unique_ptr<TensorBuffer> &tensor_buffer) {
  auto allocator = NpuMemoryAllocator::GetAllocator();
  GE_CHECK_NOTNULL(allocator);
  tensor_buffer = TensorBuffer::Create(allocator, size);
  GE_CHECK_NOTNULL(tensor_buffer);
  return SUCCESS;
}

Status AicpuNodeTaskBase::InitExtInfo(const std::string &kernel_ext_info, int64_t session_id) {
  if (kernel_ext_info.empty()) {
    if (node_item_->is_dynamic) {
      // dynamic node must have ext info
      GELOGE(PARAM_INVALID, "Node[%s] parse ext info failed as ext info is empty.", node_name_.c_str());
      return PARAM_INVALID;
    } else {
      // if no ext info no need copy to device.
      GELOGD("Node[%s] kernel_ext_info is empty, no need copy to device, is_dynamic=%s.",
             node_name_.c_str(), node_item_->is_dynamic ? "true" : "false");
      return SUCCESS;
    }
  }

  GE_CHK_STATUS_RET(aicpu_ext_handle_.Parse(kernel_ext_info),
                    "Node[%s] parse kernel ext info failed, kernel_ext_info_size=%zu.",
                    node_name_.c_str(), kernel_ext_info.size());
  GELOGD("To update aicpu_task ext_info session_info session_id to %lu", session_id);
  GE_CHK_STATUS_RET(aicpu_ext_handle_.UpdateSessionInfoSessionId(session_id),
                    "UpdateSessionInfoSessionId failed.");

  // copy task args buf
  GE_CHK_STATUS_RET(AllocTensorBuffer(aicpu_ext_handle_.GetExtInfoLen(), ext_info_addr_dev_),
                    "Node[%s] alloc kernel_ext_info buf failed, size=%zu",
                    node_name_.c_str(), aicpu_ext_handle_.GetExtInfoLen());

  // copy default ext info to device
  GE_CHK_RT_RET(rtMemcpy(ext_info_addr_dev_->GetData(), ext_info_addr_dev_->GetSize(),
                         aicpu_ext_handle_.GetExtInfo(), aicpu_ext_handle_.GetExtInfoLen(),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateOutputShapeFromExtInfo() {
  if (node_item_->num_outputs == 0) {
    GELOGD("Task [%s] output_num is 0, no need update output shape.", node_name_.c_str());
    return SUCCESS;
  }
  // copy to host buf
  GE_CHK_RT_RET(rtMemcpy(aicpu_ext_handle_.GetExtInfo(),
                         aicpu_ext_handle_.GetExtInfoLen(),
                         ext_info_addr_dev_->GetData(),
                         ext_info_addr_dev_->GetSize(),
                         RT_MEMCPY_DEVICE_TO_HOST));

  for (auto i = 0; i < node_item_->num_outputs; ++i) {
    GeShape shape;
    // not support update data type now, just for param
    DataType data_type;
    aicpu_ext_handle_.GetOutputShapeAndType(i, shape, data_type);
    auto output_desc = node_item_->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(shape, i, output_desc),
                      "Update node %s [%d]th output shape failed.",
                      node_name_.c_str(), i);
  }
  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateShapeToOutputDesc(const GeShape &shape_new,
                                                  int32_t output_index, GeTensorDescPtr &output_desc) {
  auto shape_old = output_desc->GetShape();
  output_desc->SetShape(shape_new);
  GELOGD("Update node[%s] out[%d] shape from %s to %s.", node_name_.c_str(), output_index,
         shape_old.ToString().c_str(), shape_new.ToString().c_str());

  auto origin_shape_old = output_desc->GetOriginShape();
  auto origin_format = output_desc->GetOriginFormat();
  auto format = output_desc->GetFormat();
  if (origin_format == format) {
    output_desc->SetOriginShape(shape_new);
    return SUCCESS;
  }
  // if format is not same need convert shape
  std::vector<int64_t> origin_dims_new;
  auto trans_ret = formats::TransShape(format, shape_new.GetDims(),
                                       output_desc->GetDataType(), origin_format, origin_dims_new);
  GE_CHK_STATUS_RET(trans_ret,
                    "Node[%s] out[%d] originFormat[%d] is not same as format[%d], but TransShape failed, shape=%s.",
                    node_name_.c_str(), output_index, origin_format, format, shape_new.ToString().c_str());
  auto origin_shape_new = GeShape(origin_dims_new);
  output_desc->SetOriginShape(origin_shape_new);
  GELOGD("Node[%s] out[%d] originFormat[%d] is not same as format[%d], need update from %s ro %s.",
         node_name_.c_str(), output_index, origin_format, format,
         origin_shape_old.ToString().c_str(), origin_shape_new.ToString().c_str());
  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateExtInfo() {
  GELOGI("Node[%s] update ext info begin, unknown_type=%d.", node_name_.c_str(), unknown_type_);
  if (node_item_->num_inputs == 0 && node_item_->num_outputs == 0) {
    GELOGD("Node[%s] has no input and output, no need update ext info.", node_name_.c_str());
    return SUCCESS;
  }

  for (auto i = 0; i < node_item_->num_inputs; ++i) {
    auto input_desc = node_item_->MutableInputDesc(i);
    GE_CHECK_NOTNULL(input_desc);
    GE_CHK_STATUS_RET(aicpu_ext_handle_.UpdateInputShapeAndType(i, *input_desc),
                      "Node[%s] input[%d] update input shape failed.",
                      node_name_.c_str(), i);
  }

  if (unknown_type_ != DEPEND_COMPUTE) {
    for (auto j = 0; j < node_item_->num_outputs; ++j) {
      auto output_desc = node_item_->MutableOutputDesc(j);
      GE_CHECK_NOTNULL(output_desc);

      GE_CHK_STATUS_RET(aicpu_ext_handle_.UpdateOutputShapeAndType(j, *output_desc),
                        "Node[%s] output[%d] UpdateOutputShapeAndType failed.",
                        node_name_.c_str(), j);
    }
  }

  // copy input and output shapes to device
  GE_CHK_RT_RET(rtMemcpy(ext_info_addr_dev_->GetData(),
                         ext_info_addr_dev_->GetSize(),
                         aicpu_ext_handle_.GetExtInfo(),
                         aicpu_ext_handle_.GetExtInfoLen(),
                         RT_MEMCPY_HOST_TO_DEVICE));

  GELOGD("Node[%s] update ext info end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateArgs(TaskContext &context) {
  GELOGD("Node[%s] update args begin. is_dynamic=%s, unknown_type=%d",
         node_name_.c_str(), node_item_->is_dynamic ? "true" : "false", unknown_type_);
  if (node_item_->num_inputs == 0 && node_item_->num_outputs == 0) {
    GELOGD("Node[%s] has no input and output, no need update args.", node_name_.c_str());
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(UpdateIoAddr(context), "Node[%s] update io addr failed.", node_name_.c_str());
  if (node_item_->is_dynamic) {
    // dynamic node need update ext info.
    GE_CHK_STATUS_RET(UpdateExtInfo(), "Node[%s] update ext info failed.", node_name_.c_str());
  }
  GELOGD("Node[%s] update args end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuNodeTaskBase::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AicpuNodeTaskBaseExecuteAsync] Start");
  GELOGD("Node[%s] execute async start. unknown_type=%d.", node_name_.c_str(), unknown_type_);

  HYBRID_CHK_STATUS_RET(LaunchTask(context), "[%s] Failed to launch task", node_name_.c_str());

  // save profiling data
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  rtError_t rt_ret = rtGetTaskIdAndStreamID(&task_id, &stream_id); // must be called after Launch kernel
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "Get task_id and stream_id failed.");
    return FAILED;
  }
  GELOGD("Aicpu node[%s] task_id: %u, stream_id: %u.", context.GetNodeName(), task_id, stream_id);
  (void)context.SaveProfilingTaskDescInfo(task_id, stream_id, kTaskTypeAicpu, 0);
  (void)context.SaveProfilingGraphDescInfo(task_id, stream_id);
  auto callback = [=, &context]() {
    GELOGD("Node[%s] callback start.", node_name_.c_str());
    RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[TaskCallback] Start");
    Status callback_ret = TaskCallback(context);
    RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[TaskCallback] End");

    GELOGD("Node[%s] task callBack ret = %u.", node_name_.c_str(), callback_ret);
    if (done_callback != nullptr) {
      context.SetStatus(callback_ret);
      done_callback();
    }

    GELOGD("Node[%s] callback end.", node_name_.c_str());
  };

  GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(callback));

  GELOGD("Node[%s] execute async end.", node_name_.c_str());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AicpuNodeTaskBaseExecuteAsync] End");
  return SUCCESS;
}

Status AicpuTfNodeTask::InitForDependComputeTask() {
  if ((unknown_type_ != DEPEND_COMPUTE) || (node_item_->num_outputs == 0)) {
    GELOGD("Node[%s] type[%s] unknown_type is %d, output num is %d.",
           node_name_.c_str(), node_item_->node_type.c_str(), unknown_type_, node_item_->num_outputs);
    return SUCCESS;
  }

  output_summary_.resize(node_item_->num_outputs);
  constexpr auto result_summary_size = sizeof(aicpu::FWKAdapter::ResultSummary);
  for (auto i = 0; i < node_item_->num_outputs; ++i) {
    GE_CHK_STATUS_RET(AllocTensorBuffer(result_summary_size, output_summary_[i]),
                      "Node[%s] alloc buffer for result summary info failed, size=%zu.",
                      node_name_.c_str(), result_summary_size);
  }
  output_summary_host_.resize(node_item_->num_outputs);

  // init for mem copy task
  // copy task need copy output_data and output_shape, max len is 2 * output_num
  const size_t copy_input_buf_len = node_item_->num_outputs * 2 * sizeof(uint64_t);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_release_flag_dev_),
                    "Node[%s] alloc copy task input release_flag failed, size=%zu",
                    node_name_.c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_data_size_dev_),
                    "Node[%s] alloc copy task input data_size failed, size=%zu",
                    node_name_.c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_src_dev_),
                    "Node[%s] alloc copy task input src failed, size=%zu",
                    node_name_.c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_dst_dev_),
                    "Node[%s] alloc copy task input dst failed, size=%zu",
                    node_name_.c_str(), copy_input_buf_len);

  // copy task args buf
  GE_CHK_STATUS_RET(AllocTensorBuffer(sizeof(STR_FWK_OP_KERNEL), copy_task_args_buf_),
                    "Node[%s] alloc copy task args buf failed, size=%zu",
                    node_name_.c_str(), sizeof(STR_FWK_OP_KERNEL));

  std::vector<uint64_t> copy_io_addr;
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_release_flag_dev_->GetData()));
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_data_size_dev_->GetData()));
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_src_dev_->GetData()));
  copy_io_addr.emplace_back(reinterpret_cast<uintptr_t>(copy_input_dst_dev_->GetData()));

  // mem copy op has 4 inputs and 0 output.
  const auto copy_io_addr_size = sizeof(uint64_t) * copy_io_addr.size();

  // can alloc in init, it can reuse
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_io_addr_size, copy_ioaddr_dev_),
                    "Node[%s] alloc copy task io buf failed, size=%zu",
                    node_name_.c_str(), copy_io_addr_size);

  GE_CHK_RT_RET(rtMemcpy(copy_ioaddr_dev_->GetData(), copy_io_addr_size,
                         &copy_io_addr[0], copy_io_addr_size, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AicpuTfNodeTask::Init(const HybridModel &model) {
  GELOGI("Node[%s] init start.", node_name_.c_str());

  GE_CHK_BOOL_RET_STATUS(task_def_.has_kernel_ex(), FAILED,
                         "Node[%s] is tf node but task def does not has kernel ex.",
                         node_name_.c_str());

  auto &kernel_ex_def = task_def_.kernel_ex();
  auto kernel_workspace_size = kernel_ex_def.task_info().size();
  GE_CHK_STATUS_RET(AllocTensorBuffer(kernel_workspace_size, kernel_workspace_),
                    "Node[%s] alloc buffer for kernel workspace failed, size=%zu.",
                    node_name_.c_str(), kernel_workspace_size);

  GE_CHK_RT_RET(rtMemcpy(kernel_workspace_->GetData(), kernel_workspace_size,
                         kernel_ex_def.task_info().data(), kernel_workspace_size,
                         RT_MEMCPY_HOST_TO_DEVICE));

  auto input_output_size = (node_item_->num_inputs + node_item_->num_outputs) * sizeof(uint64_t);
  // alloc input output addr buf, allow alloc size 0
  GE_CHK_STATUS_RET(AllocTensorBuffer(input_output_size, input_output_addr_),
                    "Node[%s] alloc buffer for io addr failed, size=%zu.",
                    node_name_.c_str(), input_output_size);

  auto &kernel_ext_info = kernel_ex_def.kernel_ext_info();
  auto kernel_ext_info_size = kernel_ex_def.kernel_ext_info_size();
  GE_CHK_BOOL_RET_STATUS(kernel_ext_info.size() == kernel_ext_info_size, FAILED,
                         "Node[%s] task def kernel_ext_info.size=%zu, but kernel_ext_info_size=%u.",
                         node_name_.c_str(), kernel_ext_info.size(), kernel_ext_info_size);

  // init ext info
  uint64_t ext_session_id = model.GetSessionId();
  GE_CHK_STATUS_RET(InitExtInfo(kernel_ext_info, ext_session_id), "Node[%s] init ext info failed.", node_name_.c_str());
  GE_CHK_STATUS_RET(InitForDependComputeTask(), "Node[%s] init for depend compute task failed.", node_name_.c_str());

  // build fwk_op_kernel.
  GE_CHK_BOOL_RET_STATUS(sizeof(STR_FWK_OP_KERNEL) >= kernel_ex_def.args_size(), FAILED,
                         "Node[%s] sizeof STR_FWK_OP_KERNEL is: %zu, but args_size is: %u",
                         node_name_.c_str(), sizeof(STR_FWK_OP_KERNEL), kernel_ex_def.args_size());

  STR_FWK_OP_KERNEL fwk_op_kernel = {0};
  errno_t sec_ret = memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL),
                             kernel_ex_def.args().data(), kernel_ex_def.args_size());
  GE_CHK_BOOL_RET_STATUS(sec_ret == EOK, INTERNAL_ERROR,
                         "Node[%s] memcpy fwk_op_kernel failed, ret: %d.", node_name_.c_str(), sec_ret);

  fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr = reinterpret_cast<uintptr_t>(kernel_workspace_->GetData());
  fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = reinterpret_cast<uintptr_t>(input_output_addr_->GetData());

  if (ext_info_addr_dev_ != nullptr) {
    // set ext info addr and ext info num
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = reinterpret_cast<uintptr_t>(ext_info_addr_dev_->GetData());
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = ext_info_addr_dev_->GetSize();
  }

  fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = GetStepIdAddr(model);

  auto session_id = fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID;
  GE_CHK_STATUS_RET(EnsureSessionCreated(session_id), "Node[%s] create session id %lu failed.",
                    node_name_.c_str(), session_id);

  // alloc kernel_buf_ and copy to device.
  GE_CHK_STATUS_RET(AllocTensorBuffer(sizeof(STR_FWK_OP_KERNEL), kernel_buf_),
                    "Node[%s] alloc buffer for kernel buf failed, size=%zu.",
                    node_name_.c_str(), sizeof(STR_FWK_OP_KERNEL));

  GE_CHK_RT_RET(rtMemcpy(kernel_buf_->GetData(), sizeof(STR_FWK_OP_KERNEL),
                         &fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL),
                         RT_MEMCPY_HOST_TO_DEVICE));
  auto node_type = NodeUtils::GetNodeType(node_item_->node);
  if (node_type.find(GETNEXT) != string::npos) {
    GELOGD("[%s] Is GetNext, set need sync to true, node type = %s", node_name_.c_str(), node_type.c_str());
    need_sync_ = true;
  }
  GELOGI("Node[%s] init end.", node_name_.c_str());
  return SUCCESS;
}

uint64_t AicpuTfNodeTask::GetStepIdAddr(const HybridModel &model) {
  // get step_id_addr
  auto var_tensor = model.GetVariable(NODE_NAME_GLOBAL_STEP);
  uint64_t step_id_addr = 0;
  if (var_tensor != nullptr) {
    step_id_addr = reinterpret_cast<uintptr_t>(var_tensor->GetData());
  }
  return step_id_addr;
}

Status AicpuTfNodeTask::EnsureSessionCreated(uint64_t session_id) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  GE_CHK_STATUS_RET(model_manager->CreateAicpuSession(session_id),
                    "Create aicpu session %lu failed", session_id);
  return SUCCESS;
}

Status AicpuTfNodeTask::ReadResultSummaryAndPrepareMemory(TaskContext &context,
                                                          std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  for (auto i = 0; i < node_item_->num_outputs; ++i) {
    auto &result_summary = output_summary_host_[i];
    GE_CHK_RT_RET(rtMemcpy(&result_summary, sizeof(aicpu::FWKAdapter::ResultSummary),
                           output_summary_[i]->GetData(), output_summary_[i]->GetSize(),
                           RT_MEMCPY_DEVICE_TO_HOST));

    auto raw_data_size = result_summary.raw_data_size;
    std::unique_ptr<TensorBuffer> tensor_buffer;
    GE_CHK_STATUS_RET(AllocTensorBuffer(raw_data_size, tensor_buffer),
                      "Node[%s] out[%d] alloc tensor buffer failed, raw_data_size=%lu",
                      node_name_.c_str(), i, raw_data_size);
    auto status = context.SetOutput(i, TensorValue(std::shared_ptr<TensorBuffer>(tensor_buffer.release())));
    GE_CHK_STATUS_RET(status, "Node[%s] set output %d failed.", node_name_.c_str(), i);

    auto shape_data_size = result_summary.shape_data_size;
    std::unique_ptr<TensorBuffer> shape_buffer;
    GE_CHK_STATUS_RET(AllocTensorBuffer(shape_data_size, shape_buffer),
                      "Node[%s] out[%d] alloc shape buffer failed, shape_data_size=%lu",
                      node_name_.c_str(), i, shape_data_size);
    out_shape_hbm.emplace_back(std::move(shape_buffer));
  }
  return SUCCESS;
}

Status AicpuTfNodeTask::CopyDataToHbm(TaskContext &context,
                                      const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  GE_CHK_BOOL_RET_STATUS(out_shape_hbm.size() == static_cast<std::size_t>(node_item_->num_outputs),
                         INTERNAL_ERROR,
                         "Node[%s] has %d outputs but out shape is %zu.",
                         node_name_.c_str(), node_item_->num_outputs, out_shape_hbm.size());

  uint64_t copy_num = 0;
  GE_CHK_STATUS_RET_NOLOG(PrepareCopyInputs(context, out_shape_hbm, copy_num));

  STR_FWK_OP_KERNEL aicpu_task = {0};
  std::string task_info;
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(),
                        "[GenMemCopyTask] Start");
  GE_CHK_STATUS_RET_NOLOG(GenMemCopyTask(copy_num, aicpu_task, task_info));
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(),
                        "[GenMemCopyTask] End");

  std::unique_ptr<TensorBuffer> kernel_workspace_buf;
  GE_CHK_STATUS_RET(AllocTensorBuffer(task_info.size(), kernel_workspace_buf),
                    "Node[%s] alloc copy task workspace buf failed, size=%zu.",
                    node_name_.c_str(), task_info.size());

  GE_CHK_RT_RET(rtMemcpy(kernel_workspace_buf->GetData(), task_info.size(),
                         task_info.data(), task_info.size(), RT_MEMCPY_HOST_TO_DEVICE));

  aicpu_task.fwkKernelBase.fwk_kernel.inputOutputAddr = reinterpret_cast<uintptr_t>(copy_ioaddr_dev_->GetData());
  aicpu_task.fwkKernelBase.fwk_kernel.workspaceBaseAddr = reinterpret_cast<uintptr_t>(kernel_workspace_buf->GetData());
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoAddr = 0;
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoLen = 0;

  GE_CHK_RT_RET(rtMemcpy(copy_task_args_buf_->GetData(), sizeof(STR_FWK_OP_KERNEL),
                         &aicpu_task, sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE));

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[LaunchCopy] Start");
  GE_CHK_RT_RET(rtKernelLaunchEx(copy_task_args_buf_->GetData(), sizeof(STR_FWK_OP_KERNEL),
                                 RT_KERNEL_DEFAULT, context.GetStream()));
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[LaunchCopy] End");

  GE_CHK_RT_RET(rtStreamSynchronize(context.GetStream()));
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[SynchronizeCopy] End");
  return SUCCESS;
}

Status AicpuTfNodeTask::PrepareCopyInputs(const TaskContext &context,
                                          const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm,
                                          uint64_t &copy_num) {
  std::vector<uint64_t> copy_input_release_flag;
  std::vector<uint64_t> copy_input_data_size;
  std::vector<uint64_t> copy_input_src;
  std::vector<uint64_t> copy_input_dst;

  for (auto i = 0; i < node_item_->num_outputs; ++i) {
    const auto &summary = output_summary_host_[i];
    GELOGD("Node[%s] out[%d] summary, shape data=0x%lx, shape data size=%lu, raw data=0x%lx, raw data size=%lu.",
           node_name_.c_str(), i,
           summary.shape_data_ptr, summary.shape_data_size,
           summary.raw_data_ptr, summary.raw_data_size);
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

  copy_num = copy_input_release_flag.size();

  GE_CHK_BOOL_RET_STATUS(copy_num > 0, INTERNAL_ERROR,
                         "Node[%s] need copy num is 0", node_name_.c_str());

  // copy task need copy output and output shape
  const size_t copy_input_buf_len = copy_num * sizeof(uint64_t);

  GE_CHK_RT_RET(rtMemcpy(copy_input_release_flag_dev_->GetData(), copy_input_release_flag_dev_->GetSize(),
                         &copy_input_release_flag[0], copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_data_size_dev_->GetData(), copy_input_data_size_dev_->GetSize(),
                         &copy_input_data_size[0], copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_src_dev_->GetData(), copy_input_src_dev_->GetSize(),
                         &copy_input_src[0], copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_dst_dev_->GetData(), copy_input_dst_dev_->GetSize(),
                         &copy_input_dst[0], copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AicpuTfNodeTask::GenMemCopyTask(uint64_t copy_num, STR_FWK_OP_KERNEL &task, std::string &task_info) {
  static constexpr const char *const kKernelLibName = "aicpu_tf_kernel";
  auto kernel_builder = OpsKernelBuilderManager::Instance().GetOpsKernelBuilder(kKernelLibName);
  GE_CHK_BOOL_RET_STATUS(kernel_builder != nullptr, FAILED, "Get op kernel info store[%s] failed", kKernelLibName);
  auto ret = kernel_builder->GenMemCopyTask(copy_num, task, task_info);
  GE_CHK_STATUS_RET(ret, "Call aicpu GenMemCopyTask failed, copy_num=%lu, ret=%u", copy_num, ret);
  return SUCCESS;
}

Status AicpuTfNodeTask::UpdateShapeByHbmBuffer(TaskContext &context,
                                               const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  GE_CHK_BOOL_RET_STATUS(out_shape_hbm.size() == static_cast<std::size_t>(node_item_->num_outputs),
                         INTERNAL_ERROR,
                         "Node[%s] has %d outputs but out shape is %zu",
                         node_name_.c_str(), node_item_->num_outputs, out_shape_hbm.size());
  for (auto i = 0; i < node_item_->num_outputs; ++i) {
    const auto &result_summary = output_summary_host_[i];
    auto output_desc = node_item_->MutableOutputDesc(i);
    std::vector<int64_t> shape_dims;
    if (result_summary.shape_data_size > 0) {
      const auto &shape_hbm = out_shape_hbm[i];
      GE_CHK_BOOL_RET_STATUS((result_summary.shape_data_size % sizeof(int64_t) == 0), INTERNAL_ERROR,
                             "Node[%s] [%d]th output shape data size is %lu is not divided by int64_t.",
                             node_name_.c_str(), i, result_summary.shape_data_size);
      uint32_t dim_num = result_summary.shape_data_size / sizeof(int64_t);
      GELOGD("Node[%s] [%d]th output dim num=%u.", node_name_.c_str(), i, dim_num);
      std::unique_ptr<int64_t[]> shape_addr(new(std::nothrow) int64_t[dim_num]());
      GE_CHECK_NOTNULL(shape_addr);
      GE_CHK_RT_RET(rtMemcpy(shape_addr.get(), result_summary.shape_data_size,
                             shape_hbm->GetData(), shape_hbm->GetSize(), RT_MEMCPY_DEVICE_TO_HOST));
      for (uint32_t dim_idx = 0; dim_idx < dim_num; ++dim_idx) {
        shape_dims.emplace_back(shape_addr[dim_idx]);
        GELOGD("Node[%s] [%d]th output dim[%u]=%ld.", node_name_.c_str(), i, dim_idx, shape_addr[dim_idx]);
      }
    }
    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(GeShape(shape_dims), i, output_desc),
                      "Node[%s] update [%d]th output shape failed.",
                      node_name_.c_str(), i);
  }
  return SUCCESS;
}

Status AicpuTfNodeTask::UpdateShapeAndDataByResultSummary(TaskContext &context) {
  GELOGD("Node[%s] update shape and data by result summary begin.", node_name_.c_str());

  std::vector<std::unique_ptr<TensorBuffer>> out_shape_hbm;
  GE_CHK_STATUS_RET(ReadResultSummaryAndPrepareMemory(context, out_shape_hbm),
                    "Node[%s] read ResultSummary and update output shape failed.",
                    node_name_.c_str());

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(),
                        "[ReadResultSummaryAndPrepareMemory] End");

  GE_CHK_STATUS_RET(CopyDataToHbm(context, out_shape_hbm),
                    "Node[%s] copy data to output failed.",
                    node_name_.c_str());

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[CopyDataToHbm] End");

  GE_CHK_STATUS_RET(UpdateShapeByHbmBuffer(context, out_shape_hbm),
                    "Node[%s] update shape by hbm buffer failed.",
                    node_name_.c_str());

  GELOGD("Node[%s] update shape and data by result summary end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuTfNodeTask::UpdateIoAddr(TaskContext &context) {
  vector<uint64_t> io_addrs;
  io_addrs.reserve(node_item_->num_inputs + node_item_->num_outputs);
  for (auto i = 0; i < node_item_->num_inputs; ++i) {
    auto inputData = context.GetInput(i);
    GE_CHECK_NOTNULL(inputData);
    GELOGD("Node[%s] input[%d] addr = %p, size = %zu", node_name_.c_str(), i,
           inputData->GetData(), inputData->GetSize());
    io_addrs.emplace_back(reinterpret_cast<uintptr_t>(inputData->GetData()));
  }

  // known shape or not depend compute
  if (!node_item_->is_dynamic || unknown_type_ != DEPEND_COMPUTE) {
    // unknown type 4 do this in call back.
    GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
    for (auto j = 0; j < node_item_->num_outputs; ++j) {
      auto outputData = context.GetOutput(j);
      GE_CHECK_NOTNULL(outputData);

      GELOGD("Node[%s] output[%d] addr = %p, size = %zu",
             node_name_.c_str(), j, outputData->GetData(), outputData->GetSize());
      io_addrs.emplace_back(reinterpret_cast<uintptr_t>(outputData->GetData()));
    }
  } else {
    // unknown type 4 use result summary update ioaddr.
    GELOGD("Node[%s] is depend compute node, use result summary as out addr.", node_name_.c_str());
    GE_CHK_BOOL_RET_STATUS(output_summary_.size() == static_cast<std::size_t>(node_item_->num_outputs),
                           INTERNAL_ERROR,
                           "Node[%s] has %d output but %zu output summary.",
                           node_name_.c_str(), node_item_->num_outputs, output_summary_.size());

    for (auto j = 0; j < node_item_->num_outputs; ++j) {
      void *summary_addr = output_summary_[j]->GetData();
      io_addrs.emplace_back(reinterpret_cast<uintptr_t>(summary_addr));
    }
  }

  // if has input and output, need copy to ioaddr
  if (!io_addrs.empty()) {
    // copy input and output to device
    GE_CHK_RT_RET(rtMemcpy(input_output_addr_->GetData(),
                           input_output_addr_->GetSize(),
                           &io_addrs[0],
                           sizeof(uint64_t) * io_addrs.size(),
                           RT_MEMCPY_HOST_TO_DEVICE));
  }
  return SUCCESS;
}

Status AicpuTfNodeTask::LaunchTask(TaskContext &context) {
  GELOGD("Node[%s] launch task start, unknown_type=%d.", node_name_.c_str(), unknown_type_);
  uint32_t flag = RT_KERNEL_DEFAULT;
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[AicpuTfNodertKernelLaunchEx] Start");
  GE_CHK_RT_RET(rtKernelLaunchEx(kernel_buf_->GetData(), kernel_buf_->GetSize(), flag, context.GetStream()));
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[AicpuTfNodertKernelLaunchEx] End");
  GELOGD("Node[%s] launch end.", node_name_.c_str());
  if (need_sync_) {
    GELOGD("[%s] Task needs sync", node_name_.c_str());
    GE_CHK_STATUS_RET_NOLOG(context.Synchronize());
  }
  return SUCCESS;
}

Status AicpuTfNodeTask::TaskCallback(TaskContext &context) {
  GELOGD("Node[%s] task callback start. is_dynamic=%s, unknown_type=%d.",
         node_name_.c_str(), node_item_->is_dynamic ? "true" : "false", unknown_type_);
  Status callback_ret = SUCCESS;
  if (node_item_->is_dynamic) {
    // check need update shape, call update shape.
    if (unknown_type_ == DEPEND_SHAPE_RANGE) {
      // check result
      callback_ret = UpdateOutputShapeFromExtInfo();
    } else if (unknown_type_ == DEPEND_COMPUTE) {
      callback_ret = UpdateShapeAndDataByResultSummary(context);
    }
  }
  GELOGD("Node[%s] task callback end.", node_name_.c_str());
  return callback_ret;
}

Status AicpuNodeTask::Init(const HybridModel &model) {
  auto node_name = node_name_;
  GELOGD("Node[%s] init start.", node_name.c_str());

  GE_CHK_BOOL_RET_STATUS(unknown_type_ != DEPEND_COMPUTE, FAILED,
                         "Node[%s] unknown type[%d] is depend compute, it's not supported now.",
                         node_name.c_str(), unknown_type_);

  GE_CHK_BOOL_RET_STATUS(task_def_.has_kernel(), FAILED, "Node[%s] task def does not has kernel.", node_name.c_str());
  auto &kernel_def = task_def_.kernel();

  auto &args = kernel_def.args();
  args_size_ = kernel_def.args_size();

  const std::string &so_name = kernel_def.so_name();
  const OpDescPtr op_desc = node_item_->GetOpDesc();
  const auto &context = kernel_def.context();
  auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
  if (kernel_type == ccKernelType::CUST_AI_CPU) {
    bool loaded = false;
    GE_CHK_STATUS_RET(ModelManager::GetInstance()->LoadCustAicpuSo(op_desc, so_name, loaded), 
                      "load cust aicpu so failed.");
    if (!loaded) {
      GE_CHK_STATUS_RET(ModelManager::GetInstance()->LaunchCustAicpuSo(), "Launch cust aicpu so failed.");
    }
  }

  GE_CHK_BOOL_RET_STATUS(args.size() == args_size_, FAILED,
                         "Node[%s] task def args.size=%zu, but args_size=%u.",
                         node_name.c_str(), args.size(), args_size_);

  GE_CHK_BOOL_RET_STATUS(args_size_ >= sizeof(aicpu::AicpuParamHead), FAILED,
                         "Node[%s] task def args_size=%u is less than aicpu param head len=%zu.",
                         node_name.c_str(), args_size_, sizeof(aicpu::AicpuParamHead));

  args_.reset(new(std::nothrow) uint8_t[args_size_]());
  GE_CHK_BOOL_RET_STATUS(args_ != nullptr, FAILED,
                         "Node[%s] malloc args mem failed, args_size_=%u.",
                         node_name.c_str(), args_size_);

  errno_t sec_ret = memcpy_s(args_.get(), args_size_, args.c_str(), args.size());
  GE_CHK_BOOL_RET_STATUS(sec_ret == EOK, INTERNAL_ERROR,
                         "Node[%s] copy args failed, ret: %d", node_name_.c_str(), sec_ret);

  auto aicpu_param_head = reinterpret_cast<aicpu::AicpuParamHead *>(args_.get());
  auto io_num = node_item_->num_inputs + node_item_->num_outputs;

  // check AicpuParamHead ioAddrNum is right.
  GE_CHK_BOOL_RET_STATUS((aicpu_param_head->ioAddrNum == static_cast<uint32_t>(io_num)), PARAM_INVALID,
                         "Node[%s] param head ioAddrNum=%u, but node has %d inputs and %d outputs.",
                         node_name.c_str(), aicpu_param_head->ioAddrNum,
                         node_item_->num_inputs, node_item_->num_outputs);

  auto mini_len = sizeof(aicpu::AicpuParamHead) + io_num * sizeof(uint64_t);
  // check args len must over mini len.
  GE_CHK_BOOL_RET_STATUS((mini_len <= aicpu_param_head->length), PARAM_INVALID,
                         "Node[%s] param head length=%u, but min len need %zu.",
                         node_name.c_str(), aicpu_param_head->length, mini_len);

  auto &kernel_ext_info = kernel_def.kernel_ext_info();
  auto kernel_ext_info_size = kernel_def.kernel_ext_info_size();
  GE_CHK_BOOL_RET_STATUS(kernel_ext_info.size() == kernel_ext_info_size, FAILED,
                         "Node[%s] task def kernel_ext_info.size=%zu, but kernel_ext_info_size=%u.",
                         node_name.c_str(), kernel_ext_info.size(), kernel_ext_info_size);

  uint64_t ext_session_id = model.GetSessionId();
  GE_CHK_STATUS_RET(InitExtInfo(kernel_ext_info, ext_session_id), "Node[%s] init ext info failed.", node_name.c_str());

  if (ext_info_addr_dev_ == nullptr) {
    aicpu_param_head->extInfoLength = 0;
    aicpu_param_head->extInfoAddr = 0;
  } else {
    aicpu_param_head->extInfoLength = ext_info_addr_dev_->GetSize();
    aicpu_param_head->extInfoAddr = reinterpret_cast<uintptr_t>(ext_info_addr_dev_->GetData());
  }

  GELOGD("Node[%s] init end.", node_name.c_str());
  return SUCCESS;
}

Status AicpuNodeTask::UpdateIoAddr(TaskContext &context) {
  vector<uint64_t> io_addrs;
  io_addrs.reserve(node_item_->num_inputs + node_item_->num_outputs);
  for (auto i = 0; i < node_item_->num_inputs; ++i) {
    auto inputData = context.GetInput(i);
    GE_CHECK_NOTNULL(inputData);

    GELOGD("Node[%s] input[%d] = %p, size = %zu", node_name_.c_str(), i, inputData->GetData(), inputData->GetSize());
    io_addrs.emplace_back(reinterpret_cast<uintptr_t>(inputData->GetData()));
  }

  GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
  for (auto j = 0; j < node_item_->num_outputs; ++j) {
    auto outputData = context.GetOutput(j);
    GE_CHECK_NOTNULL(outputData);
    GELOGD("Node[%s] output[%d] addr = %p, size = %zu", node_name_.c_str(), j,
           outputData->GetData(), outputData->GetSize());
    io_addrs.emplace_back(reinterpret_cast<uintptr_t>(outputData->GetData()));
  }

  auto io_addr = args_.get() + sizeof(aicpu::AicpuParamHead);
  // if has input and output, need copy to ioaddr
  int cpy_ret = memcpy_s(io_addr, args_size_ - sizeof(aicpu::AicpuParamHead),
                         &io_addrs[0], sizeof(uint64_t) * io_addrs.size());
  GE_CHK_BOOL_RET_STATUS(cpy_ret == 0, INTERNAL_ERROR,
                         "Node[%s] memcpy io addr to AicpuParamHead failed, ret=%d, args_size=%u, io nums=%zu.",
                         node_name_.c_str(), cpy_ret, args_size_, io_addrs.size());
  return SUCCESS;
}

Status AicpuNodeTask::LaunchTask(TaskContext &context) {
  GELOGD("Node[%s] launch task start. unknown_type=%d.", node_name_.c_str(), unknown_type_);
  const auto &so_name = task_def_.kernel().so_name();
  const auto &kernel_name = task_def_.kernel().kernel_name();
  const auto &kcontext = task_def_.kernel().context();
  auto kernel_type = static_cast<ccKernelType>(kcontext.kernel_type());
  uint32_t flag = RT_KERNEL_DEFAULT;
  if (kernel_type == ccKernelType::CUST_AI_CPU) {
    flag |= static_cast<uint32_t>(RT_KERNEL_CUSTOM_AICPU);
  }
  auto rt_ret = rtCpuKernelLaunchWithFlag(reinterpret_cast<const void *>(so_name.c_str()),
                                          reinterpret_cast<const void *>(kernel_name.c_str()),
                                          1, // default core dim is 1
                                          args_.get(), args_size_,
                                          nullptr, context.GetStream(), flag);
  GE_CHK_RT_RET(rt_ret);
  GELOGD("Node[%s] launch task end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuNodeTask::TaskCallback(TaskContext &context) {
  GELOGD("Node[%s] task callback start, is_dynamic = %s, unknown_type=%d.",
         node_name_.c_str(), node_item_->is_dynamic ? "true" : "false", unknown_type_);
  Status callback_ret = SUCCESS;

  // check need update shape, call update shape.
  if (node_item_->is_dynamic && unknown_type_ == DEPEND_SHAPE_RANGE) {
    // check result
    callback_ret = UpdateOutputShapeFromExtInfo();
  } else {
    GELOGD("Node[%s] unknown shape type is %d no need update output shape.",
           node_name_.c_str(), unknown_type_);
  }
  GELOGD("Node[%s] task callback end.", node_name_.c_str());
  return callback_ret;
}

Status AiCpuNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  // malloc HBM memory at Init, here just update them
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCpuNodeExecutorPrepareTask] Start");
  Status status = task.UpdateArgs(context);
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCpuNodeExecutorPrepareTask] End");
  return status;
}

Status AiCpuNodeExecutor::LoadTask(const HybridModel &model,
                                   const NodePtr &node,
                                   std::shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  GELOGD("Node[%s] load task start.", node->GetName().c_str());
  auto node_item = model.GetNodeItem(node);
  GE_CHECK_NOTNULL(node_item);
  auto task_defs = model.GetTaskDefs(node);
  GE_CHECK_NOTNULL(task_defs);
  if (node_item->shape_inference_type != DEPEND_COMPUTE) {
    GE_CHK_BOOL_RET_STATUS((*task_defs).size() == 1, PARAM_INVALID,
                           "Node[%s] task_def num[%zu] != 1", node->GetName().c_str(), (*task_defs).size());
  } else {
    // The number of tasks of the fourth type operator may be 2
    GE_CHK_BOOL_RET_STATUS((*task_defs).size() == 1 || (*task_defs).size() == 2, PARAM_INVALID,
                           "Node[%s] DEPEND_COMPUTE task_def num[%zu] != 1 or 2",
                           node->GetName().c_str(), (*task_defs).size());
  }
  const auto &task_def = (*task_defs)[0];
  std::shared_ptr<AicpuNodeTaskBase> aicpu_task;
  if (task_def.type() == RT_MODEL_TASK_KERNEL_EX) {
    GELOGI("Node[%s] task type=%u is AicpuTfNodeTask.", node->GetName().c_str(), task_def.type());
    aicpu_task = MakeShared<AicpuTfNodeTask>(node_item, task_def);
  } else if (task_def.type() == RT_MODEL_TASK_KERNEL) {
    GELOGI("Node[%s] task type=%u is AicpuNodeTask.", node->GetName().c_str(), task_def.type());
    aicpu_task = MakeShared<AicpuNodeTask>(node_item, task_def);
  } else {
    GELOGE(UNSUPPORTED, "Node[%s] task type=%u is not supported by aicpu node executor.",
           node->GetName().c_str(), task_def.type());
    return UNSUPPORTED;
  }

  GE_CHK_BOOL_RET_STATUS(aicpu_task != nullptr, MEMALLOC_FAILED,
                         "Load task for node %s failed.", node->GetName().c_str());

  GE_CHK_STATUS_RET(aicpu_task->Init(model), "Node[%s] task init failed.", node->GetName().c_str());

  task = std::move(aicpu_task);
  GELOGD("Node[%s] load task end.", node->GetName().c_str());
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
