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

#include "graph/load/new_model_manager/task_info/kernel_task_info.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "aicpu/common/aicpu_task_struct.h"
#include "common/ge/plugin_manager.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/l2_cache_optimize.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "runtime/kernel.h"
#include "super_kernel/super_kernel.h"
#include "super_kernel/super_kernel_factory.h"

namespace {
const uint8_t kL2LoadToDdr = 1;
const uint8_t kL2NotLoadToDdr = 0;
// for skt
constexpr int64_t kInvalidGroupKey = -1;
constexpr uint32_t kSKTSingleSize = 1;
constexpr uint32_t kSKTMaxSizeLimit = 20000;
const char *kIsLastNode = "is_last_node";
const char *kIsFirstNode = "is_first_node";
const int64_t kCloseSkt = 100;
const uint32_t kAddrLen = sizeof(void *);
}  // namespace

namespace ge {
KernelTaskInfo::SuperKernelTaskInfo KernelTaskInfo::skt_info_ = {
  0, 0, 0, 0, nullptr, nullptr, {}, {}, RT_KERNEL_DEFAULT, kInvalidGroupKey, 0, nullptr};

Status KernelTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  if (davinci_model == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }
  davinci_model_ = davinci_model;
  is_l1_fusion_enable_ = davinci_model_->GetL1FusionEnableOption();
  GELOGD("KernelTaskInfo Init Start, ge.enableL1Fusion in davinci model is %d.", is_l1_fusion_enable_);

  Status ret = SetStream(task_def.stream_id(), davinci_model_->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  const domi::KernelDef &kernel_def = task_def.kernel();
  block_dim_ = kernel_def.block_dim();
  args_size_ = kernel_def.args_size();
  // get opcontext stored in model
  const domi::KernelContext &context = kernel_def.context();
  // get kernel_type
  kernel_type_ = static_cast<cce::ccKernelType>(context.kernel_type());
  // get opdesc
  op_desc_ = davinci_model_->GetOpByIndex(context.op_index());
  if (op_desc_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "Get op_desc failed, index is out of range!");
    return INTERNAL_ERROR;
  }
  (void)AttrUtils::GetBool(*op_desc_, ATTR_N_BATCH_SPILT, is_n_batch_spilt_);
  GELOGD("node[%s] is_n_batch_spilt %d", op_desc_->GetName().c_str(), is_n_batch_spilt_);
  (void)AttrUtils::GetInt(*op_desc_, ATTR_NAME_FUSION_GROUP_KEY, group_key_);
  has_group_key_ = (group_key_ != kInvalidGroupKey);
  GELOGD("node[%s] has_group_key_ %ld, group key is [%ld]", op_desc_->GetName().c_str(), has_group_key_, group_key_);

  // fusion_op_info
  vector<std::string> original_op_names;
  bool result = AttrUtils::GetListStr(op_desc_, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);
  GE_IF_BOOL_EXEC(result, fusion_op_info_.stream_id = task_def.stream_id();
                  fusion_op_info_.op_index = context.op_index(); fusion_op_info_.original_op_names = original_op_names;
                  fusion_op_info_.op_name = op_desc_->GetName());

  string session_graph_model_id;
  davinci_model_->GetUniqueId(op_desc_, session_graph_model_id);
  // get bin_file_key
  const char *bin_file_key = davinci_model_->GetRegisterStub(op_desc_->GetName(), session_graph_model_id);
  // new aicpu kernel(rtCpuKernelLaunch) no need to check function
  if (kernel_type_ == cce::ccKernelType::CCE_AI_CORE) {
    rtError_t rt_ret;
    rt_ret = rtGetFunctionByName(const_cast<char *>(kernel_def.stub_func().c_str()), &stub_func_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "execute rtGetFunctionByName failed. stub_func: %s",
                                                    kernel_def.stub_func().c_str());
                    return RT_FAILED;);
  } else if (kernel_type_ != cce::ccKernelType::AI_CPU) {
    rtError_t rt_ret;
    rt_ret = rtGetFunctionByName(bin_file_key, &stub_func_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    GELOGE(RT_FAILED, "execute rtGetFunctionByName failed. bin_file_key: %s", bin_file_key);
                    return RT_FAILED;);
  }

  if (context.origin_op_index_size() > CC_FUSION_OP_MAX) {
    GELOGE(PARAM_INVALID, "context.origin_op_index_size() is more than CC_FUSION_OP_MAX(%d)", CC_FUSION_OP_MAX);
    return PARAM_INVALID;
  }

  for (int32_t i = 0; i < context.origin_op_index_size(); ++i) {
    ctx_.opIndex2[i] = context.origin_op_index(i);
  }
  ctx_.opCount = context.origin_op_index_size();
  if (kernel_type_ == cce::ccKernelType::TE) {
    ctx_.opIndex = context.op_index();
    uint16_t *args_offset_tmp = reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data()));
    if (context.args_offset().size() / sizeof(uint16_t) < 1) {
      GELOGE(FAILED, "context.args_offset().size() / sizeof(uint16_t) less than 1");
      return FAILED;
    }

    ret = InitTVMTask(args_offset_tmp[0], kernel_def);
  } else if (kernel_type_ == cce::ccKernelType::CUSTOMIZED) {
    ret = InitAICPUCustomTask(context.op_index(), kernel_def);
  } else if (kernel_type_ == cce::ccKernelType::AI_CPU) {
    ret = InitAicpuTask(context.op_index(), kernel_def);
  } else {
    if (kernel_def.args().empty() || args_size_ == 0) {
      GELOGE(FAILED, "args is null.");
      return FAILED;
    }
    ret = InitCceTask(kernel_def);
  }

  GELOGD("KernelTaskInfo Init finish, result=%u.", ret);
  return ret;
}

Status KernelTaskInfo::SaveSKTDumpInfo() {
  GE_CHECK_NOTNULL(davinci_model_);
  davinci_model_->SaveDumpTask(skt_info_.last_task_id, skt_info_.last_stream_id, skt_info_.last_op,
                               skt_info_.last_dump_args);
  return SUCCESS;
}

void KernelTaskInfo::UpdateSKTTaskId() {
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  if (davinci_model_ != nullptr) {
    rtError_t rt_ret = rtModelGetTaskId(davinci_model_->GetRtModelHandle(), &task_id, &stream_id);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return;
    }
    skt_info_.last_task_id = task_id;
    skt_info_.last_stream_id = stream_id;
    skt_id_ = skt_info_.last_task_id;

    GELOGI("UpdateTaskId:UpdateSKTTaskId [%u],stream id [%u]", task_id, stream_id);
  }
}

void KernelTaskInfo::UpdateTaskId() {
  uint32_t task_id = 0;
  uint32_t stream_id = 0;  //  for profiling
  if (davinci_model_ != nullptr) {
    rtError_t rt_ret = rtModelGetTaskId(davinci_model_->GetRtModelHandle(), &task_id, &stream_id);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return;
    }
    task_id_ = task_id;
    stream_id_ = stream_id;
    GELOGI("UpdateTaskId:UpdateTaskId [%u], stream id [%u]:", task_id, stream_id);
  }
}

Status KernelTaskInfo::SKTFinalize() {
  UpdateSKTTaskId();
  GE_CHK_STATUS_RET(SaveSKTDumpInfo(), "skt save dump info failed");
  GELOGI("SuperKernel Distribute [skt_id:%u]", skt_id_);
  skt_info_.kernel_list.clear();
  skt_info_.arg_list.clear();
  skt_info_.last_stream = nullptr;
  skt_info_.last_block_dim = 0;
  skt_info_.last_sm_desc = sm_desc_;
  skt_info_.last_group_key = kInvalidGroupKey;
  skt_info_.last_dump_flag = RT_KERNEL_DEFAULT;
  skt_info_.last_dump_args = 0;
  skt_info_.last_op = nullptr;
  return SUCCESS;
}

Status KernelTaskInfo::SuperKernelLaunch() {
  if (skt_info_.kernel_list.empty()) {
    GELOGI("SuperKernelLaunch: Skt_kernel_list has no task, just return");
    return SUCCESS;
  }
  rtError_t rt_ret;
  auto &skt_kernel_list = skt_info_.kernel_list;
  auto &skt_arg_list = skt_info_.arg_list;
  GELOGI("SuperKernelLaunch: Skt_kernel_list size[%d] skt_arg_list[%d]", skt_kernel_list.size(), skt_arg_list.size());
  if (skt_kernel_list.size() == kSKTSingleSize) {
    rt_ret = rtKernelLaunchWithFlag(skt_info_.kernel_list[0], static_cast<uint32_t>(skt_info_.last_block_dim),
                                    skt_info_.arg_list[0], skt_info_.last_args_size,
                                    static_cast<rtSmDesc_t *>(skt_info_.last_sm_desc), skt_info_.last_stream,
                                    skt_info_.last_dump_flag);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "SuperKernelLaunch: Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
    GE_CHK_STATUS_RET(SKTFinalize(), "Skt finalize failed");
    return SUCCESS;
  }
  // Create super kernel factory
  skt::SuperKernelFactory *factory = &skt::SuperKernelFactory::GetInstance();
  // Init super kernel factory
  if (factory->Init() != SUCCESS) {
    GELOGE(RT_FAILED, "SuperKernelLaunch: SuperKernelFactory init failed");
    return RT_FAILED;
  }
  // Call the fuse API
  skt::SuperKernel *superKernel = nullptr;
  if (factory->FuseKernels(skt_kernel_list, skt_arg_list, skt_info_.last_block_dim, superKernel) != SUCCESS) {
    GELOGE(RT_FAILED, "SuperKernelLaunch: fuse call failed");
    return RT_FAILED;
  }
  // Launch a super kernel
  if (superKernel->Launch(skt_info_.last_stream, RT_KERNEL_DUMPFLAG) != SUCCESS) {
    GELOGE(RT_FAILED, "SuperKernelLaunch: launch failed");
    return RT_FAILED;
  }
  GELOGI("SuperKernelLaunch: success[skt_kernel_list size[%zu] skt_arg_list[%zu]]", skt_kernel_list.size(),
         skt_arg_list.size());
  GE_CHK_STATUS_RET(SKTFinalize(), "Skt finalize failed");
  return SUCCESS;
}

Status KernelTaskInfo::SaveSuperKernelInfo() {
  skt_info_.kernel_list.push_back(stub_func_);
  skt_info_.arg_list.push_back(args_);
  skt_info_.last_stream = stream_;
  skt_info_.last_block_dim = block_dim_;
  skt_info_.last_args_size = args_size_;
  skt_info_.last_sm_desc = sm_desc_;
  skt_info_.last_dump_flag = dump_flag_;
  skt_info_.last_group_key = group_key_;
  skt_info_.last_dump_args = reinterpret_cast<uintptr_t>(dump_args_);
  skt_info_.last_op = op_desc_;
  // last node in a stream, just launch
  if (IsMarkedLastNode()) {
    return SuperKernelLaunch();
  }
  return SUCCESS;
}

bool KernelTaskInfo::IsMarkedLastNode() {
  if (davinci_model_ == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return false;
  }
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(ctx_.opIndex);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "InitTVMTaskInfo error, index is out of range!");
    return false;
  }
  bool is_last_node = false;
  (void)AttrUtils::GetBool(*op_desc, kIsLastNode, is_last_node);
  return is_last_node;
}

bool KernelTaskInfo::IsMarkedFirstNode() {
  if (davinci_model_ == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return false;
  }
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(ctx_.opIndex);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "InitTVMTaskInfo error, index is out of range!");
    return false;
  }
  bool is_first_node = false;
  (void)AttrUtils::GetBool(*op_desc, kIsFirstNode, is_first_node);
  return is_first_node;
}
// current task 's block dim and stream and grouo key (if have) must same with last task,
// then may be saved to skt task list; else
// call skt launch those saved tasks before
bool KernelTaskInfo::FirstCallSKTLaunchCheck() {
  return ((block_dim_ != skt_info_.last_block_dim) || (stream_ != skt_info_.last_stream) ||
          (has_group_key_ && (group_key_ != skt_info_.last_group_key)));
}

// current task has group_id or has n ATTR_N_BATCH_SPLIT then save it to skt task list; else
// call skt launch those saved tasks and call rtlaunch for current task
bool KernelTaskInfo::DoubleCallSKTSaveCheck() { return (!is_n_batch_spilt_ && !has_group_key_); }

Status KernelTaskInfo::SuperKernelDistribute() {
  Status ret;
  char *skt_task_num = getenv("SKT_TASK_NUM");
  auto task_num = static_cast<uint64_t>((skt_task_num != nullptr) ? strtol(skt_task_num, nullptr, 10)
                                                                  : kSKTMaxSizeLimit);  // 10 for decimal number
  GELOGI("SKT: SuperKernel Distribute Task num[skt_id:%lu]", task_num);
  if (FirstCallSKTLaunchCheck()) {
    ret = SuperKernelLaunch();
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Call SuperKernelLaunch failed!");
      return FAILED;
    }
  }
  if (DoubleCallSKTSaveCheck()) {
    // 1.launch before
    ret = SuperKernelLaunch();
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Call SuperKernelLaunch failed!");
      return FAILED;
    }
    // 2.launch current
    rtError_t rt_ret = rtKernelLaunchWithFlag(stub_func_, block_dim_, args_, args_size_,
                                              static_cast<rtSmDesc_t *>(sm_desc_), stream_, dump_flag_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return FAILED;
    }
    UpdateTaskId();
    GELOGI("Current Common Task Distribute [taskid:%u]", task_id_);
  } else {
    ret = SaveSuperKernelInfo();
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Call SuperKernelLaunch failed!");
      return FAILED;
    }
    GELOGI("Save Current task [block_dim:%u, size:%zu].", block_dim_, skt_info_.kernel_list.size());
  }
  return SUCCESS;
}

Status KernelTaskInfo::Distribute() {
  GELOGD("KernelTaskInfo Distribute Start.");
  if (davinci_model_->IsKnownNode()) {
    args_ = davinci_model_->GetCurrentArgsAddr(args_offset_);
    GELOGI("Known node %s args addr %p, offset %u.", op_desc_->GetName().c_str(), args_, args_offset_);
  }
  rtError_t rt_ret = RT_ERROR_NONE;
  char *skt_enable_env = getenv("SKT_ENABLE");
  int64_t env_flag = (skt_enable_env != nullptr) ? strtol(skt_enable_env, nullptr, 10) : 0;
  bool call_skt = ((env_flag != 0) || is_l1_fusion_enable_);
  if (kernel_type_ == cce::ccKernelType::AI_CPU) {
    // blockDim is reserved parameter, set to 1
    rt_ret = rtCpuKernelLaunchWithFlag(reinterpret_cast<const void *>(so_name_.c_str()),
                                       reinterpret_cast<const void *>(kernel_name_.c_str()), 1, args_, args_size_,
                                       nullptr, stream_, dump_flag_);
  } else {
    /* default: not skt launch */
    GELOGI(
      "KernelTaskInfo Distribute Start, sktenable:%d taskid:%u sktid:%u last_sktid:%u stubfunc_name:%s "
      "stubfunc:%p blockdim:%u stream:%p",
      call_skt, task_id_, skt_id_, skt_info_.last_task_id, stub_func_name_.c_str(), stub_func_, block_dim_, stream_);
    // l1 fusion enable and env flag open (kCloseSkt for skt debug)
    if (call_skt && (env_flag != kCloseSkt)) {
      GE_RETURN_IF_ERROR(SuperKernelDistribute());
    } else {
      // call rtKernelLaunch for current task
      rt_ret = rtKernelLaunchWithFlag(stub_func_, block_dim_, args_, args_size_, static_cast<rtSmDesc_t *>(sm_desc_),
                                      stream_, dump_flag_);
    }
  }
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  // set for task_id_
  UpdateTaskId();
  GELOGI(
    "KernelTaskInfo Distribute Success. sktenable:%d taskid:%d sktid:%d stubfunc_name:%s stubfunc:%p "
    "blockdim:%d stream:%p",
    call_skt, task_id_, skt_id_, stub_func_name_.c_str(), stub_func_, block_dim_, stream_);
  return SUCCESS;
}

Status KernelTaskInfo::UpdateArgs() {
  GELOGI("KernelTaskInfo::UpdateArgs in.");
  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();
  vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc_);
  vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc_);
  vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc_);

  vector<void *> io_addrs;
  io_addrs.insert(io_addrs.end(), input_data_addrs.begin(), input_data_addrs.end());
  io_addrs.insert(io_addrs.end(), output_data_addrs.begin(), output_data_addrs.end());
  io_addrs.insert(io_addrs.end(), workspace_data_addrs.begin(), workspace_data_addrs.end());

  GE_CHK_STATUS_RET(davinci_model_->UpdateKnownZeroCopyAddr(io_addrs, args_offset_),
                    "update known node %s zero copy addr failed.", op_desc_->GetName().c_str());

  GELOGI("KernelTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

Status KernelTaskInfo::Release() {
  if (davinci_model_ != nullptr && davinci_model_->IsKnownNode()) {
    return SUCCESS;
  }
  FreeRtMem(&args_);
  FreeRtMem(&flowtable_);
  FreeRtMem(&custom_info_.input_descs);
  FreeRtMem(&custom_info_.input_addrs);
  FreeRtMem(&custom_info_.output_descs);
  FreeRtMem(&custom_info_.output_addrs);
  FreeRtMem(&custom_info_.attr_handle);
  FreeRtMem(&aicpu_ext_info_addr_);

  if (ctx_.argsOffset != nullptr) {
    delete[] ctx_.argsOffset;
    ctx_.argsOffset = nullptr;
  }

  rtError_t ret = (sm_desc_ != nullptr) ? rtMemFreeManaged(sm_desc_) : RT_ERROR_NONE;
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", static_cast<int>(ret));
    return FAILED;
  }
  sm_desc_ = nullptr;

  return SUCCESS;
}

Status KernelTaskInfo::UpdateL2Data(const domi::KernelDef &kernel_def) {
  string sm_desc = kernel_def.sm_desc();
  if (sm_desc.empty()) {
    return SUCCESS;
  }

  char *sm_contrl = const_cast<char *>(sm_desc.data());
  rtL2Ctrl_t *l2_ctrl_info = reinterpret_cast<rtL2Ctrl_t *>(sm_contrl);
  uint64_t gen_base_addr = davinci_model_->GetRtBaseAddr();

  // There is no weight for te op now. Update L2_mirror_addr by data memory base.
  uint64_t data_base_addr = (uint64_t)(uintptr_t)davinci_model_->MemBase() - (uint64_t)gen_base_addr;
  const uint32_t l2_ctrl_info_data_count = 8;
  for (uint32_t data_index = 0; data_index < l2_ctrl_info_data_count; ++data_index) {
    if (l2_ctrl_info->data[data_index].L2_mirror_addr != 0) {
      l2_ctrl_info->data[data_index].L2_mirror_addr += data_base_addr;
      l2_ctrl_info->data[data_index].L2_load_to_ddr = IsL2CpToDDR(l2_ctrl_info->data[data_index].L2_load_to_ddr);
    }
  }

  rtError_t rt_ret = rtMemAllocManaged(&sm_desc_, sm_desc.size(), RT_MEMORY_SPM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtMemcpy(sm_desc_, sm_desc.size(), sm_desc.data(), sm_desc.size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  return SUCCESS;
}

Status KernelTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  domi::KernelDef kernel_def = task_def.kernel();
  uint32_t args_size = kernel_def.args_size();
  args_offset_ = davinci_model->GetTotalArgsSize();
  davinci_model->SetTotalArgsSize(args_size);
  GELOGI("kernel task name , args_size %u, args_offset %u", args_size, args_offset_);
  return SUCCESS;
}

Status KernelTaskInfo::InitTVMTask(uint16_t offset, const domi::KernelDef &kernel_def) {
  GELOGD("Do InitTVMTask.");
  GE_CHECK_NOTNULL(davinci_model_);
  // get tvm op desc
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(ctx_.opIndex);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "InitTVMTaskInfo error, index:%u out of range!", ctx_.opIndex);
    return INTERNAL_ERROR;
  }
  if (davinci_model_->IsKnownNode()) {
    return SUCCESS;
  }

  // Update Stub
  // When training, when the the second call to DavinciModel::init() comes here, stub_func_ is already valid,
  // and does not need to be modified.
  // When inferencing, stub_func_ is different from dynamic-registration to runtime, and needs to be modified.
  string session_graph_model_id;
  davinci_model_->GetUniqueId(op_desc, session_graph_model_id);
  const char *bin_file_key = davinci_model_->GetRegisterStub(op_desc->GetName(), session_graph_model_id);
  rtError_t rt_ret = rtQueryFunctionRegistered(const_cast<char *>(bin_file_key));
  if (rt_ret != RT_ERROR_NONE) {
    stub_func_ = const_cast<char *>(bin_file_key);
  }

  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();
  const vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  const vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  const vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);

  vector<void *> tensor_device_addrs;
  tensor_device_addrs.insert(tensor_device_addrs.end(), input_data_addrs.begin(), input_data_addrs.end());
  tensor_device_addrs.insert(tensor_device_addrs.end(), output_data_addrs.begin(), output_data_addrs.end());
  tensor_device_addrs.insert(tensor_device_addrs.end(), workspace_data_addrs.begin(), workspace_data_addrs.end());

  // malloc args memory
  rt_ret = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  // copy orign args
  rt_ret = rtMemcpy(args_, args_size_, kernel_def.args().data(), args_size_, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  vector<uint8_t> args_info(args_size_);
  errno_t sec_ret = memcpy_s(args_info.data(), args_size_, kernel_def.args().data(), args_size_);
  if (sec_ret != EOK) {
    GELOGE(FAILED, "memcpy failed, ret: %d", sec_ret);
    return FAILED;
  }

  if ((args_size_ <= offset) || (args_size_ - offset < kAddrLen * tensor_device_addrs.size())) {
    GELOGE(FAILED, "offset >= kernelInfo.argsSize or copy content beyond applied memory.");
    return FAILED;
  }

  // copy args
  rt_ret = rtMemcpy(static_cast<char *>(args_) + offset, args_size_ - offset, tensor_device_addrs.data(),
                    kAddrLen * tensor_device_addrs.size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  sec_ret = memcpy_s(args_info.data() + offset, args_size_ - offset, tensor_device_addrs.data(),
                     kAddrLen * tensor_device_addrs.size());
  if (sec_ret != EOK) {
    GELOGE(FAILED, "memcpy failed, ret: %d", sec_ret);
    return FAILED;
  }

  if (PropertiesManager::Instance().IsLayerNeedDump(davinci_model_->Name(), davinci_model_->OmName(),
                                                    op_desc->GetName())) {
    dump_flag_ = RT_KERNEL_DUMPFLAG;
    dump_args_ = static_cast<char *>(args_) + offset;
  }

  // update origin l2 data
  if (UpdateL2Data(kernel_def) != SUCCESS) {
    return RT_FAILED;
  }

  vector<void *> virtual_io_addrs;  // use virtual address for zero copy key.
  const vector<void *> virtual_in_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc, false);
  const vector<void *> virtual_out_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc, false);
  virtual_io_addrs.insert(virtual_io_addrs.end(), virtual_in_addrs.begin(), virtual_in_addrs.end());
  virtual_io_addrs.insert(virtual_io_addrs.end(), virtual_out_addrs.begin(), virtual_out_addrs.end());
  davinci_model_->SetZeroCopyAddr(op_desc, virtual_io_addrs, args_info.data(), args_, args_size_, offset);

  GELOGD("Do InitTVMTask end");
  return SUCCESS;
}

Status KernelTaskInfo::InitAICPUCustomTask(uint32_t op_index, const domi::KernelDef &kernel_def) {
  GELOGI("Do InitAICPUCustomTask");
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "index is out of range, index: %u", op_index);
    return INTERNAL_ERROR;
  }

  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();

  const domi::KernelContext &context = kernel_def.context();
  const uint32_t kCustomAicpuArgsLen = 5;
  ctx_.argsOffset = new (std::nothrow) uint16_t[kCustomAicpuArgsLen]();
  if (ctx_.argsOffset == nullptr) {
    GELOGE(PARAM_INVALID, "ctx_.argsOffset is null!");
    return PARAM_INVALID;
  }

  if (context.args_offset().size() / sizeof(uint16_t) < kCustomAicpuArgsLen) {
    GELOGE(PARAM_INVALID, "context.args_offset().size() / sizeof(uint16_t) is less than kCustomAicpuArgsLen");
    return PARAM_INVALID;
  }

  for (uint32_t i = 0; i < kCustomAicpuArgsLen; ++i) {
    ctx_.argsOffset[i] = (reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[i];
  }

  const std::vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  const std::vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  Status ret = StoreInputOutputTensor(input_data_addrs, output_data_addrs, ModelUtils::GetInputDescs(op_desc),
                                      ModelUtils::GetOutputDescs(op_desc));

  if (ret != SUCCESS) {
    GELOGE(ret, "StoreInputOutputTensor Failed");
    return ret;
  }

  // attrHandle
  Buffer buffer;
  if (!AttrUtils::GetBytes(op_desc, ATTR_NAME_OPATTR, buffer)) {
    GELOGE(FAILED, "can't find opattr bytes!.");
    return FAILED;
  }

  uint32_t op_attr_size = buffer.GetSize();
  if (op_attr_size == 0) {
    GELOGE(PARAM_INVALID, "param op_attr_size is out of range");
    return PARAM_INVALID;
  }

  rtError_t rt_ret = rtMalloc(&custom_info_.attr_handle, op_attr_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret = rtMemcpy(custom_info_.attr_handle, op_attr_size, buffer.GetData(), op_attr_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  // args
  char *args = const_cast<char *>(kernel_def.args().data());

  for (uint32_t i = 0; i < kCustomAicpuArgsLen; ++i) {
    if (kernel_def.args().size() < ((size_t)ctx_.argsOffset[i] + sizeof(uint64_t))) {
      GELOGE(FAILED, "ctx.argsOffset[%u]: %u + sizeof(uint64_t): %zu >= kernelDef.args().size():%zu", i,
             (uint32_t)ctx_.argsOffset[i], sizeof(uint64_t), kernel_def.args().size());
      return FAILED;
    }
  }
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[0])) =
    reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.input_descs));  // arg 0
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[1])) =
    reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.input_addrs));  // arg 1
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[2])) =
    reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.output_descs));  // arg 2
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[3])) =
    reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.output_addrs));  // arg 3
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[4])) =
    reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.attr_handle));  // arg 4

  rt_ret = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  rt_ret =
    rtMemcpy(args_, kernel_def.args_size(), kernel_def.args().data(), kernel_def.args_size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  const vector<void *> virtual_in_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc, false);
  const vector<void *> virtual_out_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc, false);
  davinci_model_->SetZeroCopyAddr(op_desc, virtual_in_addrs, input_data_addrs.data(), custom_info_.input_addrs,
                                  virtual_in_addrs.size() * kAddrLen, 0);
  davinci_model_->SetZeroCopyAddr(op_desc, virtual_out_addrs, output_data_addrs.data(), custom_info_.output_addrs,
                                  output_data_addrs.size() * kAddrLen, 0);
  return SUCCESS;
}

Status KernelTaskInfo::InitCceTask(const domi::KernelDef &kernel_def) {
  GELOGI("Do InitCCETask");
  if (davinci_model_ == nullptr) {
    GELOGE(PARAM_INVALID, "davinci_model is null!");
    return PARAM_INVALID;
  }
  Status ret = SetContext(kernel_def);
  if (ret != SUCCESS) {
    GELOGE(ret, "SetContext Fail.");
    return ret;
  }

  string flowtable = kernel_def.flowtable();
  const domi::KernelContext &context = kernel_def.context();

  if (context.is_flowtable()) {
    if (flowtable.empty()) {
      GELOGE(FAILED, "flowtable is null.");
      return FAILED;
    }
    flowtable_size_ = flowtable.size();
  }

  // get smDesc stored in model
  string sm_desc = kernel_def.sm_desc();
  uint64_t sm_contrl_size = sm_desc.empty() ? 0 : sizeof(rtSmDesc_t);

  // Passing the memory info when the offline-model-generated to the CCE, which uses this info for address refresh
  ctx_.genDataBaseAddr = davinci_model_->GetRtBaseAddr();
  ctx_.genDataBaseSize = davinci_model_->TotalMemSize();
  ctx_.genWeightBaseAddr = davinci_model_->GetRtWeightAddr();
  ctx_.genWeightBaseSize = davinci_model_->TotalWeightsMemSize();
  ctx_.genVariableBaseAddr = davinci_model_->GetRtVarAddr();
  ctx_.genVariableBaseSize = davinci_model_->TotalVarMemSize();
  ctx_.l2ctrlSize = sm_contrl_size;

  if (UpdateCceArgs(sm_desc, flowtable, kernel_def) != SUCCESS) {
    GELOGE(ret, "update cce args fail");
    return ret;
  }

  // flowtable
  ret = SetFlowtable(flowtable, kernel_def);
  if (ret != SUCCESS) {
    GELOGE(ret, "SetFlowtable Fail");
    return ret;
  }

  // args
  rtError_t rt_ret = rtMalloc(&args_, kernel_def.args_size(), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "cce task physical memory.", kernel_def.args_size())

  rt_ret =
    rtMemcpy(args_, kernel_def.args_size(), kernel_def.args().data(), kernel_def.args_size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  // L2
  if (!sm_desc.empty()) {
    rt_ret = rtMemAllocManaged(&sm_desc_, sm_desc.size(), RT_MEMORY_SPM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }

    rt_ret = rtMemcpy(sm_desc_, sm_desc.size(), sm_desc.data(), sm_desc.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }
  return SUCCESS;
}

Status KernelTaskInfo::InitAicpuTask(uint32_t op_index, const domi::KernelDef &kernel_def) {
  GELOGI("Do InitAicpuTask");
  so_name_ = kernel_def.so_name();
  kernel_name_ = kernel_def.kernel_name();

  OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "index is out of range, index: %u", op_index);
    return INTERNAL_ERROR;
  }

  // copy args to new host memory
  std::unique_ptr<uint8_t[]> args_addr(new (std::nothrow) uint8_t[args_size_]);
  GE_PRINT_DYNAMIC_MEMORY(new, "cce task physical memory.", sizeof(uint8_t) * args_size_)
  errno_t sec_ret = memcpy_s(args_addr.get(), args_size_, kernel_def.args().data(), args_size_);
  if (sec_ret != EOK) {
    GELOGE(FAILED, "memcpy failed, ret: %d", sec_ret);
    return FAILED;
  }

  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();

  vector<void *> input_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  vector<void *> output_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  vector<void *> io_addrs;
  io_addrs.insert(io_addrs.end(), input_addrs.begin(), input_addrs.end());
  io_addrs.insert(io_addrs.end(), output_addrs.begin(), output_addrs.end());
  if (!io_addrs.empty()) {
    // refresh io addrs
    uintptr_t io_addr = reinterpret_cast<uintptr_t>(args_addr.get()) + sizeof(aicpu::AicpuParamHead);
    auto addrs_size = sizeof(uint64_t) * io_addrs.size();
    sec_ret = memcpy_s(reinterpret_cast<void *>(io_addr), addrs_size, io_addrs.data(), addrs_size);
    if (sec_ret != EOK) {
      GELOGE(FAILED, "memcpy failed, ret: %d", sec_ret);
      return FAILED;
    }
  }

  auto aicpu_param_head = reinterpret_cast<aicpu::AicpuParamHead *>(args_addr.get());
  const auto &ext_info = kernel_def.kernel_ext_info();
  auto init_ret = InitAicpuTaskExtInfo(ext_info);
  if (init_ret != SUCCESS) {
    GELOGE(init_ret, "Init aicpu task ext info failed, ext_info size=%zu", ext_info.size());
    return init_ret;
  }
  aicpu_param_head->extInfoAddr = reinterpret_cast<uintptr_t>(aicpu_ext_info_addr_);
  aicpu_param_head->extInfoLength = reinterpret_cast<uintptr_t>(ext_info.size());

  // malloc device memory for args
  rtError_t rt_ret = rtMalloc(static_cast<void **>(&args_), args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api(rtMalloc) failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "cce task physical memory.", args_size_)

  // copy args to device
  rt_ret = rtMemcpy(args_, args_size_, args_addr.get(), args_size_, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api(rtMemcpy) failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  if (PropertiesManager::Instance().IsLayerNeedDump(davinci_model_->Name(), davinci_model_->OmName(),
                                                    op_desc->GetName())) {
    dump_flag_ = RT_KERNEL_DUMPFLAG;
    dump_args_ = static_cast<char *>(args_) + sizeof(aicpu::AicpuParamHead);
  }

  vector<void *> virtual_io_addrs;  // use virtual address for zero copy key.
  const vector<void *> virtual_in_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc, false);
  const vector<void *> virtual_out_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc, false);
  virtual_io_addrs.insert(virtual_io_addrs.end(), virtual_in_addrs.begin(), virtual_in_addrs.end());
  virtual_io_addrs.insert(virtual_io_addrs.end(), virtual_out_addrs.begin(), virtual_out_addrs.end());
  davinci_model_->SetZeroCopyAddr(op_desc, virtual_io_addrs, args_addr.get(), args_, args_size_,
                                  sizeof(aicpu::AicpuParamHead));

  return SUCCESS;
}

Status KernelTaskInfo::InitAicpuTaskExtInfo(const std::string &ext_info) {
  if (ext_info.empty()) {
    return SUCCESS;
  }
  auto rt_ret = rtMalloc(&aicpu_ext_info_addr_, ext_info.size(), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMalloc ext_info error: 0x%X, size=%zu", rt_ret, ext_info.size());
    return FAILED;
  }
  rt_ret = rtMemcpy(aicpu_ext_info_addr_, ext_info.size(), ext_info.c_str(), ext_info.size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMemcpy ext_info error: 0x%X, size=%zu", rt_ret, ext_info.size());
    return FAILED;
  }

  return SUCCESS;
}

Status KernelTaskInfo::StoreInputOutputTensor(const std::vector<void *> &input_data_addrs,
                                              const std::vector<void *> &output_data_addrs,
                                              const std::vector<::tagCcAICPUTensor> &input_descs,
                                              const std::vector<::tagCcAICPUTensor> &output_descs) {
  auto input_size = input_descs.size();
  auto output_size = output_descs.size();

  // inputDescs
  rtError_t rt_ret = rtMalloc(&custom_info_.input_descs, sizeof(opTensor_t) * input_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  for (std::size_t i = 0; i < input_size; ++i) {
    rt_ret = rtMemcpy(static_cast<opTensor_t *>(custom_info_.input_descs) + i, sizeof(opTensor_t),
                      const_cast<tagOpTensor *>(&input_descs[i]), sizeof(opTensor_t), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  // inputAddrs
  rt_ret = rtMalloc(&custom_info_.input_addrs, sizeof(opTensor_t) * input_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  if (!input_data_addrs.empty()) {
    rt_ret = rtMemcpy(custom_info_.input_addrs, kAddrLen * input_size, &input_data_addrs[0], kAddrLen * input_size,
                      RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  // outputDescs
  rt_ret = rtMalloc(&custom_info_.output_descs, sizeof(opTensor_t) * output_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }
  for (std::size_t i = 0; i < output_size; ++i) {
    rt_ret = rtMemcpy(static_cast<opTensor_t *>(custom_info_.output_descs) + i, sizeof(opTensor_t),
                      const_cast<tagOpTensor *>(&input_descs[i]), sizeof(opTensor_t), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  // outputAddrs
  rt_ret = rtMalloc(&custom_info_.output_addrs, sizeof(opTensor_t) * output_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return RT_FAILED;
  }

  if (!output_data_addrs.empty()) {
    rt_ret = rtMemcpy(custom_info_.output_addrs, kAddrLen * output_size, &output_data_addrs[0], kAddrLen * output_size,
                      RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  return SUCCESS;
}

Status KernelTaskInfo::SetContext(const domi::KernelDef &kernel_def) {
  const domi::KernelContext &context = kernel_def.context();
  ctx_.kernelType = static_cast<cce::ccKernelType>(context.kernel_type());
  ctx_.opId = context.op_id();
  ctx_.kernelFuncId = context.kernel_func_id();
  ctx_.isFlowtable = context.is_flowtable();
  ctx_.argsCount = context.args_count();
  if (ctx_.argsCount == 0) {
    GELOGE(INTERNAL_ERROR, "check argsCount fail:%u.", ctx_.argsCount);
    return INTERNAL_ERROR;
  }

  if (context.args_offset().size() / sizeof(uint16_t) < ctx_.argsCount) {
    GELOGE(PARAM_INVALID, "param [context.args_offset().size() / sizeof(uint16_t)] is less than [ctx_.argsCount]");
    return PARAM_INVALID;
  }

  // ctx_.argsOffset stores the offset of the internal information of agrs_, equal to the ctx_.argsCount
  ctx_.argsOffset = new (std::nothrow) uint16_t[ctx_.argsCount]();
  if (ctx_.argsOffset == nullptr) {
    GELOGE(PARAM_INVALID, "(param [ctx_.argsOffset] must not be null.");
    return PARAM_INVALID;
  }

  for (uint32_t i = 0; i < ctx_.argsCount; ++i) {
    ctx_.argsOffset[i] = (reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[i];
  }

  return SUCCESS;
}

void KernelTaskInfo::FreeRtMem(void **ptr) {
  if (ptr == nullptr || *ptr == nullptr) {
    return;
  }
  rtError_t ret = rtFree(*ptr);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", ret);
  }

  *ptr = nullptr;
}

Status KernelTaskInfo::UpdateCceArgs(std::string &sm_desc, std::string &flowtable, const domi::KernelDef &kernel_def) {
  GE_CHECK_NOTNULL(davinci_model_);
  const domi::KernelContext &context = kernel_def.context();

  uint64_t data_base_addr = reinterpret_cast<uintptr_t>(davinci_model_->MemBase()) - davinci_model_->GetRtBaseAddr();
  uint64_t weight_base_addr =
    reinterpret_cast<uintptr_t>(davinci_model_->WeightsMemBase()) - davinci_model_->GetRtWeightAddr();
  uint64_t var_base_addr = reinterpret_cast<uintptr_t>(davinci_model_->VarMemBase()) - davinci_model_->GetRtVarAddr();

  Status status =
    CceUpdateKernelArgs(context, data_base_addr, weight_base_addr, var_base_addr, sm_desc, flowtable, kernel_def);
  if (status != SUCCESS) {
    GELOGE(FAILED, "Call cce api failed");
    return FAILED;
  }
  return SUCCESS;
}

Status KernelTaskInfo::CceUpdateKernelArgs(const domi::KernelContext &context, uint64_t &data_base_addr,
                                           uint64_t &weight_base_addr, uint64_t &var_base_addr, std::string &sm_desc,
                                           std::string &flowtable, const domi::KernelDef &kernel_def) {
  char *sm_contrl = nullptr;
  if (!sm_desc.empty()) {
    sm_contrl = const_cast<char *>(sm_desc.data());
  }

  std::string file_name = "libcce.so";
  std::string path = PluginManager::GetPath();
  path.append(file_name);
  string canonicalPath = RealPath(path.c_str());
  if (canonicalPath.empty()) {
    GELOGW("failed to get realpath of %s", path.c_str());
    return FAILED;
  }

  GELOGI("FileName:%s, Path:%s.", file_name.c_str(), canonicalPath.c_str());
  auto handle = dlopen(canonicalPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    GELOGE(GE_PLGMGR_SO_NOT_EXIST, "Failed in dlopen %s! ", dlerror());
    return FAILED;
  }
  cce::ccStatus_t cc_ret;
  auto cceUpdateKernelArgs = (cce::ccStatus_t(*)(cce::ccOpContext &, uint64_t, uint64_t, uint64_t, void *, uint64_t,
                                                 void *))dlsym(handle, "ccUpdateKernelArgs");
  if (cceUpdateKernelArgs == nullptr) {
    GELOGE(FAILED, "Failed to invoke function ccUpdateKernelArgs");
    if (dlclose(handle) != 0) {
      GELOGW("Failed to close handle %s", dlerror());
    }
    return FAILED;
  } else {
    GELOGI("Libcce.so has been opened");
    if (context.is_flowtable()) {
      cc_ret = cceUpdateKernelArgs(ctx_, data_base_addr, weight_base_addr, var_base_addr,
                                   const_cast<char *>(flowtable.data()), kernel_def.flowtable().size(), sm_contrl);
    } else {
      cc_ret = cceUpdateKernelArgs(ctx_, data_base_addr, weight_base_addr, var_base_addr,
                                   const_cast<char *>(kernel_def.args().data()), args_size_, sm_contrl);
    }
  }
  if (dlclose(handle) != 0) {
    GELOGW("Failed to close handle %s", dlerror());
    return FAILED;
  }
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    return CCE_FAILED;
  }

  GELOGI("CceUpdateKernelArgs success!");
  return SUCCESS;
}

Status KernelTaskInfo::SetFlowtable(std::string &flowtable, const domi::KernelDef &kernel_def) {
  const domi::KernelContext &context = kernel_def.context();
  if (context.is_flowtable()) {
    rtError_t rt_ret = rtMalloc(&flowtable_, flowtable.size(), RT_MEMORY_HBM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
    GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "flowtable refresh of cce scence.", flowtable.size())

    rt_ret = rtMemcpy(flowtable_, flowtable.size(), flowtable.data(), flowtable.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }

    // modify flowtable addr in args
    char *args = const_cast<char *>(kernel_def.args().data());

    if (kernel_def.args().size() <
        ((reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0] + sizeof(uint64_t))) {
      GELOGE(FAILED, "(context.args_offset().data()))[0]:%u + sizeof(uint64_t):%zu > kernelDef.args().size():%zu",
             (uint32_t)((reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0]),
             sizeof(uint64_t), kernel_def.args().size());
      return FAILED;
    }

    *(reinterpret_cast<uint64_t *>(
      args + (reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0])) =
      reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(flowtable_));
  }
  return SUCCESS;
}

uint8_t KernelTaskInfo::IsL2CpToDDR(uint8_t origain_L2_load_to_ddr) {
  if (origain_L2_load_to_ddr == kL2LoadToDdr) {
    return kL2LoadToDdr;
  }

  if (dump_flag_ == RT_KERNEL_DUMPFLAG) {
    return kL2LoadToDdr;
  }

  static char *ge_dump_env = std::getenv("DUMP_OP");
  if (ge_dump_env != nullptr) {
    static std::string ge_dump_str(ge_dump_env);
    static std::string open_ge_dump("1");
    if (ge_dump_str == open_ge_dump) {
      return kL2LoadToDdr;
    }
  }

  return kL2NotLoadToDdr;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_KERNEL, KernelTaskInfo);
}  // namespace ge
