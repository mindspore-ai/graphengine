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

#include "graph/load/new_model_manager/davinci_model.h"

#include <cce/dnn.h>
#include <dlfcn.h>
#include <graph/utils/node_utils.h>
#include <pthread.h>
#include <sched.h>
#include <sys/prctl.h>
#include <algorithm>
#include <map>
#include <utility>

#include "common/debug/log.h"
#include "common/formats/formats.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/math/math_util.h"
#include "common/op/ge_op_utils.h"
#include "common/profiling/profiling_manager.h"
#include "common/properties_manager.h"
#include "common/scope_guard.h"
#include "common/thread_pool.h"
#include "framework/common/debug/ge_log.h"
#include "graph/common/ge_call_wrapper.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/graph.h"
#include "graph/load/new_model_manager/cpu_queue_schedule.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "graph/load/new_model_manager/tbe_handle_store.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/manager/util/debug.h"
#include "graph/model_serialize.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "mmpa/mmpa_api.h"
#include "omm/csa_interact.h"
#include "runtime/base.h"
#include "runtime/dev.h"
#include "runtime/event.h"
#include "runtime/mem.h"
#include "runtime/rt_model.h"
#include "runtime/stream.h"
#include "securec.h"

// create std::thread, catch exceptions using try/catch
#define CREATE_STD_THREAD(thread_id, func, args)                                                  \
  do {                                                                                            \
    try {                                                                                         \
      thread_id = std::thread(func, args);                                                        \
    } catch (const std::system_error &e) {                                                        \
      GELOGE(FAILED, "Caught system_error with code:%d, meaning:%s", e.code().value(), e.what()); \
      GELOGE(FAILED, "Thread creat FAIL, Please check the left resource!");                       \
      return FAILED;                                                                              \
    }                                                                                             \
  } while (0)

namespace ge {
namespace {
const uint32_t kDataIndex = 0;
const uint32_t kOutputNum = 1;
const uint32_t kTrueBranchStreamNum = 1;
const uint32_t kThreadNum = 16;
const uint32_t kAddrLen = sizeof(void *);
const int kDecimal = 10;
const int kBytes = 8;
const uint32_t kDataMemAlignSizeCompare = 64;
const uint32_t kDumpL1FusionOpMByteSize = 2 * 1024 * 1024;
const uint32_t kDumpFlagOfL1Fusion = 0;
const char *const kDefaultBatchLable = "Batch_default";

inline bool IsDataOp(const std::string &node_type) {
  return node_type == DATA_TYPE || node_type == AIPP_DATA_TYPE || node_type == ANN_DATA_TYPE;
}
inline bool IsNoTaskAndDumpNeeded(const OpDescPtr &op_desc) {
  bool save_dump_info = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NO_TASK_AND_DUMP_NEEDED, save_dump_info);
  return save_dump_info;
}
}  // namespace

std::mutex DavinciModel::tvm_bin_mutex_;

DavinciModel::DavinciModel(int32_t priority, const std::shared_ptr<ModelListener> &listener)
    : weights_mem_base_(nullptr),
      var_mem_base_(nullptr),
      mem_base_(nullptr),
      is_inner_mem_base_(false),
      is_inner_weight_base_(false),
      is_inner_p2p_mem_base_(false),
      data_inputer_(nullptr),
      load_begin_time_(0),
      load_end_time_(0),
      time_info_(),
      dataInputTid(0),
      is_model_has_inited_(false),
      model_id_(0),
      runtime_model_id_(0),
      version_(0),
      ge_model_(nullptr),
      thread_id_(),
      listener_(listener),
      run_flg_(false),
      priority_(priority),
      rt_model_handle_(nullptr),
      rt_model_stream_(nullptr),
      is_inner_model_stream_(false),
      is_async_mode_(false),
      last_execute_mode_(INITIALIZATION),
      session_id_(0),
      device_id_(0),
      maxDumpOpNum_(0), data_dumper_(runtime_param_),
      iterator_count_(0),
      is_l1_fusion_enable_(false),
      is_first_execute_(true) {
  op_list_.clear();
}

DavinciModel::~DavinciModel() {
  try {
    Status ret = data_dumper_.UnloadDumpInfo();
    if (ret != SUCCESS) {
      GELOGW("UnloadDumpInfo failed, ret: %u.", ret);
    }

    for (const auto &op_and_addr : saved_task_addrs_) {
      auto addr = op_and_addr.second;
      if (addr != nullptr) {
        GE_CHK_RT(rtFree(addr));
      }
      addr = nullptr;
    }
    saved_task_addrs_.clear();

    GE_CHK_STATUS(ModelRunStop());

    op_list_.clear();
    data_op_list_.clear();
    output_op_list_.clear();
    tensor_name_to_fixed_addr_size_.clear();
    tensor_name_to_peer_output_index_.clear();
    GE_DELETE_NEW_SINGLE(data_inputer_);
    // check rt ctx is exist. rt api call will cause error log when ctx not exist
    rtContext_t ctx = nullptr;
    rtError_t rt_ret = rtCtxGetCurrent(&ctx);
    if (rt_ret == RT_ERROR_NONE) {
      UnbindTaskSinkStream();
      for (size_t i = 0; i < label_list_.size(); ++i) {
        if (label_list_[i] != nullptr) {
          GE_LOGW_IF(rtLabelDestroy(label_list_[i]) != RT_ERROR_NONE, "Destroy label failed, index: %zu", i);
        }
      }

      for (size_t i = 0; i < stream_list_.size(); ++i) {
        GE_LOGW_IF(rtStreamDestroy(stream_list_[i]) != RT_ERROR_NONE, "Destroy stream failed, index: %zu", i);
      }

      for (size_t i = 0; i < event_list_.size(); ++i) {
        GE_LOGW_IF(rtEventDestroy(event_list_[i]) != RT_ERROR_NONE, "Destroy event failed, index: %zu", i);
      }

      FreeWeightsMem();

      FreeFeatureMapMem();

      FreeP2PMem();

      if (l1_fusion_addr_ != nullptr) {
        GE_CHK_RT(rtFree(l1_fusion_addr_));
      }

      if (rt_model_handle_ != nullptr) {
        GE_CHK_RT(rtModelDestroy(rt_model_handle_));
        rt_model_handle_ = nullptr;
      }
    }

    OpDebugUnRegister();

    GELOGI("do ReleaseTask");
    ReleaseTask();
    CleanTbeHandle();

    var_mem_base_ = nullptr;
    if (known_node_) {
      if (args_ != nullptr) {
        GE_CHK_RT(rtFree(args_));
      }
      total_io_addrs_.clear();
      if (fixed_addrs_ != nullptr) {
        GE_CHK_RT(rtFree(fixed_addrs_));
      }
    }
  } catch (...) {
    GELOGW("DavinciModel::~DavinciModel: clear op_list catch exception.");
  }
}

void DavinciModel::UnbindHcomStream() {
  if (!all_hccl_stream_list_.empty()) {
    for (size_t i = 0; i < all_hccl_stream_list_.size(); i++) {
      GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, all_hccl_stream_list_[i]) != RT_ERROR_NONE,
                 "Unbind hccl stream from model failed! Index: %zu", i);
      GE_LOGW_IF(rtStreamDestroy(all_hccl_stream_list_[i]) != RT_ERROR_NONE, "Destroy hccl stream for rt_model failed!")
    }
  }
  return;
}

void DavinciModel::ReleaseTask() {
  for (const auto &task : cpu_task_list_) {
    if (task != nullptr) {
      GE_CHK_STATUS(task->Release(), "Release task failed.");
    }
  }
  cpu_task_list_.clear();

  for (const auto &task : task_list_) {
    if (task != nullptr) {
      GE_CHK_STATUS(task->Release(), "Release task failed.");
    }
  }
}

Status DavinciModel::Assign(const GeModelPtr &ge_model) {
  if (ge_model == nullptr) {
    GELOGI("can't assign null ge_model");
    return FAILED;
  }
  ge_model_ = ge_model;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Reduce memory usage after task sink.
/// @return: void
///
void DavinciModel::Shrink() {
  ge_model_.reset();  // delete object.
}

Status DavinciModel::InitModelMem(void *dev_ptr, size_t mem_size, void *weight_ptr, size_t weight_size) {
  if (is_model_has_inited_) {
    GELOGE(FAILED, "call InitModelMem more than once .");
    return FAILED;
  }
  is_model_has_inited_ = true;

  std::size_t data_size = TotalMemSize();
  std::size_t p2p_data_size = P2PMemInfos().at(RT_MEMORY_P2P_DDR).memory_size;
  const Buffer &weights = ge_model_->GetWeight();
  std::size_t weights_size = weights.GetSize();
  GE_CHECK_LE(weights_size, ALLOC_MEMORY_MAX_SIZE);

  if ((dev_ptr != nullptr) && (mem_size < TotalMemSize())) {
    GELOGE(FAILED, "Invalid mem param: mem_size=%zu totalsize=%zu.", mem_size, TotalMemSize());
    return FAILED;
  }

  if ((weight_ptr != nullptr) && (weight_size < weights_size)) {
    GELOGE(FAILED, "Invalid mem param: weight_size=%zu totalsize=%zu.", weight_size, weights_size);
    return FAILED;
  }

  mem_base_ = static_cast<uint8_t *>(dev_ptr);
  p2p_mem_base_ = static_cast<uint8_t *>(dev_ptr);
  weights_mem_base_ = static_cast<uint8_t *>(dev_ptr);
  is_inner_mem_base_ = false;
  is_inner_weight_base_ = false;

  if (TotalMemSize() && mem_base_ == nullptr) {
    mem_base_ = MallocFeatureMapMem(data_size);
    if (mem_base_ == nullptr) {
      GELOGE(GE_EXEC_ALLOC_FEATURE_MAP_MEM_FAILED, "Alloc feature map memory failed. size: %zu", data_size);
      return GE_EXEC_ALLOC_FEATURE_MAP_MEM_FAILED;
    }
    GEEVENT("[IMAS]InitModelMem graph_%u MallocMemory type[F] memaddr[%p] mem_size[%zu]", runtime_param_.graph_id,
            mem_base_, data_size);
    weights_mem_base_ = mem_base_;

    is_inner_mem_base_ = true;
    is_inner_weight_base_ = true;
  }

  if (p2p_data_size != 0) {
    p2p_mem_base_ = MallocP2PMem(p2p_data_size);
    if (p2p_mem_base_ == nullptr) {
      GELOGE(GE_EXEC_ALLOC_P2P_MEM_FAILED, "Alloc p2p memory failed,size: %zu", p2p_data_size);
      return GE_EXEC_ALLOC_P2P_MEM_FAILED;
    }
    GELOGI("InitModelMem graph_%u MallocMemory type[P] memaddr[%p] mem_size[%zu]", runtime_param_.graph_id,
           p2p_mem_base_, p2p_data_size);
    is_inner_p2p_mem_base_ = true;
  }

  if (weights_size != 0) {
    weights_mem_base_ = static_cast<uint8_t *>(weight_ptr);
    is_inner_weight_base_ = false;
    if (weight_ptr == nullptr) {
      weights_mem_base_ = MallocWeightsMem(weights_size);
      if (weights_mem_base_ == nullptr) {
        GELOGE(GE_EXEC_ALLOC_WEIGHT_MEM_FAILED, "Alloc weight memory failed. size: %zu", weights_size);
        return GE_EXEC_ALLOC_WEIGHT_MEM_FAILED;
      }
      is_inner_weight_base_ = true;
    }
    GELOGI("[IMAS]InitModelMem graph_%u MallocMemory type[W] memaddr[%p] mem_size[%zu]", runtime_param_.graph_id,
           weights_mem_base_, weights_size);
    GE_CHK_RT_RET(rtMemcpy(weights_mem_base_, weights_size, weights.GetData(), weights_size, RT_MEMCPY_HOST_TO_DEVICE));
    GELOGI("copy weights data to device");
  }

  GE_CHK_STATUS_RET(InitVariableMem(), "Init variable memory failed.");
  runtime_param_.mem_base = mem_base_;
  runtime_param_.weight_base = weights_mem_base_;
  runtime_param_.memory_infos[RT_MEMORY_P2P_DDR].memory_base = p2p_mem_base_;
  return SUCCESS;
}

Status DavinciModel::InitVariableMem() {
  // malloc variable memory base
  var_mem_base_ = VarManager::Instance(session_id_)->GetVarMemoryBase(RT_MEMORY_HBM);
  if (TotalVarMemSize() && var_mem_base_ == nullptr) {
    Status ret = VarManager::Instance(session_id_)->MallocVarMemory(TotalVarMemSize());
    if (ret != SUCCESS) {
      GELOGE(ret, "Malloc variable memory failed.");
      return ret;
    }
    var_mem_base_ = VarManager::Instance(session_id_)->GetVarMemoryBase(RT_MEMORY_HBM);
    GEEVENT("[IMAS]InitVariableMem graph_%u MallocMemory type[V] memaddr[%p] mem_size[%zu]", runtime_param_.graph_id,
            var_mem_base_, TotalVarMemSize());
  }
  runtime_param_.var_base = var_mem_base_;
  return SUCCESS;
}

void DavinciModel::InitRuntimeParams() {
  int64_t value = 0;
  bool ret;
  MemInfo p2p_mem_info;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_MEMORY_SIZE, value);
  runtime_param_.mem_size = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_WEIGHT_SIZE, value);
  runtime_param_.weight_size = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_STREAM_NUM, value);
  runtime_param_.stream_num = ret ? (uint32_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_EVENT_NUM, value);
  runtime_param_.event_num = ret ? (uint32_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_LABEL_NUM, value);
  runtime_param_.label_num = ret ? (uint32_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_BATCH_NUM, value);
  runtime_param_.batch_num = ret ? (uint32_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, MODEL_ATTR_TASK_GEN_BASE_ADDR, value);
  runtime_param_.logic_mem_base = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, value);
  runtime_param_.logic_weight_base = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ge::MODEL_ATTR_SESSION_ID, value);
  runtime_param_.session_id = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_TASK_GEN_VAR_ADDR, value);
  runtime_param_.logic_var_base = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_VAR_SIZE, value);
  runtime_param_.var_size = ret ? (uint64_t)value : 0;
  session_id_ = runtime_param_.session_id;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_P2P_MEMORY_SIZE, value);
  p2p_mem_info.memory_size = ret ? (uint64_t)value : 0;
  runtime_param_.memory_infos[RT_MEMORY_P2P_DDR] = std::move(p2p_mem_info);

  GELOGI(
      "InitRuntimeParams(), session_id:%lu, stream_num:%u, event_num:%u, label_num:%u, "
      "logic_mem_base:0x%lx, logic_weight_base:0x%lx, logic_var_base:0x%lx, "
      "memory_size:%lu, weight_size:%lu, var_size:%lu",
      runtime_param_.session_id, runtime_param_.stream_num, runtime_param_.event_num, runtime_param_.label_num,
      runtime_param_.logic_mem_base, runtime_param_.logic_weight_base, runtime_param_.logic_var_base,
      runtime_param_.mem_size, runtime_param_.weight_size, runtime_param_.var_size);
}

void DavinciModel::CheckHasHcomOp() {
  Graph graph = ge_model_->GetGraph();
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    return;
  }
  for (const auto &node : compute_graph->GetAllNodes()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGW("Node OpDesc is nullptr"); continue);
    GE_IF_BOOL_EXEC(((op_desc->GetType() == HCOMBROADCAST) || (op_desc->GetType() == HCOMALLGATHER) ||
                     (op_desc->GetType() == HCOMALLREDUCE) || (op_desc->GetType() == HCOMSEND) ||
                     (op_desc->GetType() == HCOMRECEIVE) || (op_desc->GetType() == HCOMREDUCESCATTER) ||
                     (op_desc->GetType() == HVDCALLBACKALLREDUCE) || (op_desc->GetType() == HVDCALLBACKALLGATHER) ||
                     (op_desc->GetType() == HVDCALLBACKBROADCAST) || (op_desc->GetType() == HVDWAIT)),
                    uint32_t stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
                    (void)hcom_streams_.emplace(stream_id); GELOGD("hcom stream: %u.", stream_id); continue);
  }
}

///
/// @ingroup ge
/// @brief Make active stream list and bind to model.
/// @return: 0 for success / others for fail
///
Status DavinciModel::BindModelStream() {
  // Stream not in active_stream_indication_ is active stream.
  is_stream_list_bind_ = false;
  if ((!input_queue_ids_.empty() || !output_queue_ids_.empty()) || (deploy_type_ == AICPU_DEPLOY_CROSS_THREAD)) {
    for (size_t i = 0; i < stream_list_.size(); ++i) {
      if (active_stream_indication_.count(i) == 0) {
        active_stream_list_.push_back(stream_list_[i]);
        active_stream_indication_.insert(i);  // deactive all model stream.
      }
    }
  }

  for (size_t i = 0; i < stream_list_.size(); ++i) {
    if (active_stream_indication_.count(i) > 0) {
      GELOGI("rtModelBindStream[%zu]", i);
      GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, stream_list_[i], RT_INVALID_FLAG));
    } else {
      // bind rt_model_handel to all streams that relates to op
      GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, stream_list_[i], RT_HEAD_STREAM));
    }
  }
  is_stream_list_bind_ = true;
  return SUCCESS;
}

Status DavinciModel::DoTaskSink() {
  // task sink is supported as model_task_def is set
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    return SUCCESS;
  }

  GE_CHK_RT_RET(rtGetAicpuDeploy(&deploy_type_));
  GELOGI("do task_sink. AiCpu deploy type is: %x.", deploy_type_);

  GE_CHK_STATUS_RET(BindModelStream(), "Bind model stream failed.");

  if (known_node_) {
    GE_CHK_STATUS_RET(MallocKnownArgs(), "Mallloc known node args failed.");
  }

  GE_CHK_STATUS_RET(InitTaskInfo(*model_task_def.get()), "InitTaskInfo failed.");

  GE_CHK_STATUS_RET(ModelManager::GetInstance()->LaunchCustAicpuSo(), "Launch cust aicpu so failed.");

  GE_CHK_STATUS_RET(InitEntryTask(), "InitEntryTask failed.");

  GE_CHK_STATUS_RET(DistributeTask(), "Distribute failed.");

  GE_CHK_RT_RET(rtModelLoadComplete(rt_model_handle_));

  SetCopyOnlyOutput();
  return SUCCESS;
}

// set device use aicore(0) or vectorcore(1)
Status DavinciModel::SetTSDevice() {
  int64_t value = 0;
  bool ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_CORE_TYPE, value);
  uint32_t core_type = ret ? static_cast<uint32_t>(value) : 0;
  GELOGI("SetTSDevice: %u", core_type);
  rtError_t rt_ret = rtSetTSDevice(core_type);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "SetTSDevice failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

Status DavinciModel::OpDebugRegister() {
  bool is_op_debug = false;
  (void)ge::AttrUtils::GetBool(ge_model_, ATTR_OP_DEBUG_FLAG, is_op_debug);
  GELOGD("The value of op_debug in ge_model_ is %d.", is_op_debug);
  if (is_op_debug) {
    debug_reg_mutex_.lock();
    rtError_t rt_ret = rtMalloc(&op_debug_addr_, kOpDebugMemorySize, RT_MEMORY_DDR);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtMalloc error, ret: 0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    uint64_t debug_addrs_tmp = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(op_debug_addr_));

    // For data dump, aicpu needs the pointer to pointer that save the real debug address.
    rt_ret = rtMalloc(&p2p_debug_addr_, kDebugP2pSize, RT_MEMORY_HBM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtMalloc error, ret: 0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    rt_ret = rtMemcpy(p2p_debug_addr_, sizeof(uint64_t), &debug_addrs_tmp, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtMemcpy to p2p_addr error: 0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    uint32_t op_debug_mode = 0;
    (void)ge::AttrUtils::GetInt(ge_model_, ATTR_OP_DEBUG_MODE, op_debug_mode);
    GELOGD("The value of op_debug_mode in ge_model_ is %u.", op_debug_mode);
    uint32_t debug_task_id = 0;
    uint32_t debug_stream_id = 0;
    rt_ret = rtDebugRegister(rt_model_handle_, op_debug_mode, op_debug_addr_, &debug_stream_id, &debug_task_id);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtDebugRegister error, ret: 0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    GELOGI("debug_task_id:%d, debug_stream_id:%u", debug_task_id, debug_stream_id);
    is_op_debug_reg_ = true;

    data_dumper_.SaveOpDebugId(debug_task_id, debug_stream_id, p2p_debug_addr_, is_op_debug);
  }

  return SUCCESS;
}

void DavinciModel::OpDebugUnRegister() {
  GELOGI("OpDebugUnRegister, is_op_debug_reg_ = %d", is_op_debug_reg_);
  if (is_op_debug_reg_) {
    debug_reg_mutex_.unlock();
    rtError_t rt_ret = RT_ERROR_NONE;
    if (rt_model_handle_ != nullptr) {
      GELOGD("start call debug_unregister.");
      rt_ret = rtDebugUnRegister(rt_model_handle_);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGW("rtDebugUnRegister failed, ret: 0x%X", rt_ret);
      }
    }

    if (op_debug_addr_ != nullptr) {
      rt_ret = rtFree(op_debug_addr_);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGW("rtFree failed, ret: 0x%X", rt_ret);
      }
      op_debug_addr_ = nullptr;
    }

    if (p2p_debug_addr_ != nullptr) {
      rt_ret = rtFree(p2p_debug_addr_);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGW("rtFree failed, ret: 0x%X", rt_ret);
      }
      p2p_debug_addr_ = nullptr;
    }
    is_op_debug_reg_ = false;
  }
  return;
}

// initialize op sequence and call initialization function of each op respectively
Status DavinciModel::Init(void *dev_ptr, size_t mem_size, void *weight_ptr, size_t weight_size) {
  // validating params
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(priority_ < 0 || priority_ > 7, return PARAM_INVALID,
                                 "Priority must between 0-7, now is %d", priority_);
  GE_CHK_BOOL_RET_STATUS(ge_model_ != nullptr, PARAM_INVALID, "GeModel is null.");
  Graph graph = ge_model_->GetGraph();
  ComputeGraphPtr compute_graph = GraphUtils::GetComputeGraph(graph);
  GE_CHK_BOOL_RET_STATUS(compute_graph != nullptr, INTERNAL_ERROR, "Get compute graph is nullptr.");

  // Initializing runtime_param_
  InitRuntimeParams();

  // RTS set aicore or vectorcore
  GE_CHK_STATUS_RET(SetTSDevice(), "SetTSDevice failed");

  version_ = ge_model_->GetVersion();
  name_ = ge_model_->GetName();
  (void)ge::AttrUtils::GetBool(ge_model_, ATTR_NAME_SWITCH_FOR_L1_FUSION, is_l1_fusion_enable_);
  GELOGD("The value of ge.l1Fusion in ge_model_ is %d.", is_l1_fusion_enable_);
  CheckHasHcomOp();

  vector<int64_t> huge_stream_list;
  (void)ge::AttrUtils::GetListInt(ge_model_, ATTR_MODEL_HUGE_STREAM_LIST, huge_stream_list);
  std::set<int64_t> huge_streams(huge_stream_list.begin(), huge_stream_list.end());

  for (uint32_t i = 0; i < StreamNum(); i++) {
    rtStream_t stream = nullptr;
    GE_MAKE_GUARD_RTSTREAM(stream);

    uint32_t stream_flags = RT_STREAM_PERSISTENT;
    if (huge_streams.find(i) != huge_streams.end()) {
      GELOGI("Stream %u is huge stream.", i);
      stream_flags |= RT_STREAM_HUGE;
    }

    if (hcom_streams_.find(i) != hcom_streams_.end()) {
      GE_CHK_RT_RET(rtStreamCreateWithFlags(&stream, priority_, stream_flags | RT_STREAM_FORCE_COPY));
    } else {
      GE_CHK_RT_RET(rtStreamCreateWithFlags(&stream, priority_, stream_flags));
    }

    GE_DISMISS_GUARD(stream);
    stream_list_.push_back(stream);
    GELOGD("Stream index:%u, stream:%p.", i, stream);
  }

  for (uint32_t i = 0; i < EventNum(); i++) {
    rtEvent_t rt_event;
    GE_CHK_RT_RET(rtEventCreate(&rt_event));
    event_list_.push_back(rt_event);
  }

  label_list_.resize(LabelNum(), nullptr);

  // create model_handle to load model
  GE_CHK_RT_RET(rtModelCreate(&rt_model_handle_, 0));
  GE_CHK_RT_RET(rtModelGetId(rt_model_handle_, &runtime_model_id_));

  // inference will use default graph_id 0;
  runtime_param_.graph_id = compute_graph->GetGraphID();

  // op debug register
  GE_CHK_STATUS_RET(OpDebugRegister(), "OpDebugRegister failed");

  GE_TIMESTAMP_START(TransAllVarData);
  GE_CHK_STATUS_RET(TransAllVarData(compute_graph, runtime_param_.graph_id), "TransAllVarData failed.");
  GE_TIMESTAMP_END(TransAllVarData, "GraphLoader::TransAllVarData");
  GE_CHK_STATUS_RET(TransVarDataUtils::CopyVarData(compute_graph, session_id_, device_id_), "copy var data failed.");

  GE_TIMESTAMP_START(InitModelMem);
  GELOGI("Known node is %d", known_node_);
  if (!known_node_) {
    GE_CHK_STATUS_RET_NOLOG(InitModelMem(dev_ptr, mem_size, weight_ptr, weight_size));
    data_inputer_ = new (std::nothrow) DataInputer();
    GE_CHK_BOOL_RET_STATUS(data_inputer_ != nullptr, MEMALLOC_FAILED, "data_inputer_ is nullptr.");
  }
  GE_TIMESTAMP_END(InitModelMem, "GraphLoader::InitModelMem");

  for (const ge::NodePtr &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
    GE_IF_BOOL_EXEC(op_desc->GetType() != VARIABLE, continue);
    GE_IF_BOOL_EXEC(IsBroadCastOpData(node),
                    (void)ge::AttrUtils::SetStr(op_desc, VAR_ATTR_VAR_IS_BROADCAST, "var_is_restore"););
  }

  GE_CHK_STATUS_RET(InitNodes(compute_graph), "Init nodes failed");

  SetDataDumperArgs(compute_graph);
  GE_TIMESTAMP_START(DoTaskSink);
  auto ret = DoTaskSink();
  GE_TIMESTAMP_END(DoTaskSink, "GraphLoader::DoTaskSink");

  auto all_dump_model = GetDumpProperties().GetAllDumpModel();
  bool findByOmName = all_dump_model.find(om_name_) != all_dump_model.end();
  bool findByModelName = all_dump_model.find(name_) != all_dump_model.end();
  if (all_dump_model.find(ge::DUMP_ALL_MODEL) != all_dump_model.end() || findByOmName || findByModelName) {
    // malloc 2M for dump l1fusion op
    GE_CHK_RT_RET(rtMalloc(&l1_fusion_addr_, kDumpL1FusionOpMByteSize, RT_MEMORY_DDR));

    // send l1fusion dump addr to rts
    GE_CHK_RT_RET(rtDumpAddrSet(rt_model_handle_, l1_fusion_addr_, kDumpL1FusionOpMByteSize, kDumpFlagOfL1Fusion));
  }

  /// In zero copy model, if a aicpu operator is connected to the first or last layer, before model execution,
  /// the aicpu opertor needs to destroy history record, and update operator memory address.
  /// The model with specified aicpu operators is only marked here, and destruction is in ModelManager::ExecuteModel().
  need_destroy_aicpu_kernel_ = IsAicpuKernelConnectSpecifiedLayer();
  (void)ge::AttrUtils::GetListStr(ge_model_, ATTR_MODEL_OUT_NODES_NAME, out_node_name_);

  // collect profiling for ge
  auto &profiling_manager = ProfilingManager::Instance();
  if (profiling_manager.ProfilingModelLoadOn()) {
    Status p_ret = ReportProfilingData(!profiling_manager.IsAclApiMode());
    if (p_ret != SUCCESS) {
      GELOGE(p_ret, "Report profiling data failed.");
      return p_ret;
    }
  }

  Shrink();
  GELOGI("Davinci model init success.");
  return ret;
}

Status DavinciModel::ReportProfilingData(bool check_device) {
  std::vector<ComputeGraphDescInfo> compute_graph_desc_info;
  Status ret = GetComputeGraphInfo(compute_graph_desc_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "GetComputeGraphInfo failed.");
    return ret;
  }
  ProfilingManager::Instance().ReportProfilingData(model_id_, GetTaskDescInfo(), compute_graph_desc_info, check_device);
  GE_CHK_STATUS(SinkModelProfile(), "Sink model profiler failed.");
  op_list_.clear();

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Travel all nodes and determine if destruction is required.
/// @return bool
///
bool DavinciModel::IsAicpuKernelConnectSpecifiedLayer() {
  Graph graph = ge_model_->GetGraph();
  ComputeGraphPtr compute_graph = GraphUtils::GetComputeGraph(graph);
  auto all_nodes = compute_graph->GetAllNodes();
  for (auto &node : all_nodes) {
    GE_IF_BOOL_EXEC(node == nullptr, continue);
    OpDescPtr op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);

    int64_t imply_type = -1;
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_IMPLY_TYPE, imply_type);
    if (imply_type != static_cast<int64_t>(domi::ImplyType::AI_CPU)) {
      continue;
    }
    GELOGD("Current operator imply type is %ld, name is %s.", imply_type, op_desc->GetName().c_str());

    for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
      GE_IF_BOOL_EXEC(in_data_anchor == nullptr, continue);
      auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_data_anchor == nullptr, continue);
      auto peer_node = peer_out_data_anchor->GetOwnerNode();
      GE_IF_BOOL_EXEC(peer_node == nullptr, continue);
      auto peer_op_desc = peer_node->GetOpDesc();
      GE_IF_BOOL_EXEC(peer_op_desc == nullptr, continue);
      if (IsDataOp(peer_op_desc->GetType())) {
        GELOGI("Mark specified aicpu operator connected to data.");
        return true;
      }
    }
    for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      GE_IF_BOOL_EXEC(out_data_anchor == nullptr, continue);
      auto peer_in_data_anchors = out_data_anchor->GetPeerInDataAnchors();
      for (auto &peer_in_data_anchor : peer_in_data_anchors) {
        GE_IF_BOOL_EXEC(peer_in_data_anchor == nullptr, continue);
        auto peer_node = peer_in_data_anchor->GetOwnerNode();
        GE_IF_BOOL_EXEC(peer_node == nullptr, continue);
        auto peer_op_desc = peer_node->GetOpDesc();
        GE_IF_BOOL_EXEC(peer_op_desc == nullptr, continue);
        if (peer_op_desc->GetType() == NETOUTPUT) {
          GELOGI("Mark specified aicpu operator connected to netoutput.");
          return true;
        }
      }
    }
  }

  return false;
}

Status DavinciModel::UpdateSessionId(uint64_t session_id) {
  GE_CHECK_NOTNULL(ge_model_);
  if (!AttrUtils::SetInt(ge_model_, MODEL_ATTR_SESSION_ID, static_cast<int64_t>(session_id))) {
    GELOGW("Set attr[%s] failed in updating session_id.", MODEL_ATTR_SESSION_ID.c_str());
  }

  GELOGD("Update session id: %lu.", session_id);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Travel all nodes and do some init.
/// @param [in] compute_graph: ComputeGraph to load.
/// @return Status
///
Status DavinciModel::InitNodes(const ComputeGraphPtr &compute_graph) {
  uint32_t data_op_index = 0;
  GE_TIMESTAMP_CALLNUM_START(LoadTBEKernelBinToOpDesc);
  GE_TIMESTAMP_CALLNUM_START(InitTbeHandle);

  typedef Status (DavinciModel::*OpDescCall)(const OpDescPtr &);
  static std::map<std::string, OpDescCall> op_desc_handle = {
      {VARIABLE, &DavinciModel::InitVariable},
      {CONSTANTOP, &DavinciModel::InitConstant},
      {STREAMACTIVE, &DavinciModel::InitStreamActive},
      {STREAMSWITCH, &DavinciModel::InitStreamSwitch},
      {STREAMSWITCHN, &DavinciModel::InitStreamSwitchN},
      {LABELSET, &DavinciModel::InitLabelSet},
      {CASE, &DavinciModel::InitCase},
  };

  GE_CHK_STATUS_RET(InitInputOutputForDynamic(compute_graph), "InitInputOutputForDynamic failed.");

  map<uint32_t, OpDescPtr> data_by_index;
  auto nodes = compute_graph->GetAllNodes();
  const CustAICPUKernelStore &aicpu_kernel_store = ge_model_->GetCustAICPUKernelStore();
  for (size_t i = 0; i < nodes.size(); i++) {
    auto node = nodes.at(i);
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGE(PARAM_INVALID, "op_desc is null.");
      return PARAM_INVALID;
    }

    op_list_[op_desc->GetId()] = op_desc;

    GE_TIMESTAMP_RESTART(LoadTBEKernelBinToOpDesc);
    aicpu_kernel_store.LoadCustAICPUKernelBinToOpDesc(op_desc);
    GE_TIMESTAMP_ADD(LoadTBEKernelBinToOpDesc);

    if (IsDataOp(op_desc->GetType())) {
      if (InitDataOp(node, data_op_index, data_by_index) != SUCCESS) {
        GELOGE(PARAM_INVALID, "Data init failed, Name: %s", op_desc->GetName().c_str());
        return PARAM_INVALID;
      }
      data_dumper_.SaveDumpInput(node);
      continue;
    }

    if (op_desc->GetType() == NETOUTPUT) {
      if (InitNetOutput(node) != SUCCESS) {
        GELOGE(PARAM_INVALID, "NetOutput init failed, Name: %s", op_desc->GetName().c_str());
        return PARAM_INVALID;
      }
      continue;
    }

    auto it = op_desc_handle.find(op_desc->GetType());
    if (it != op_desc_handle.end()) {
      if ((this->*it->second)(op_desc) != SUCCESS) {
        GELOGE(PARAM_INVALID, "NetOutput init failed, Name: %s", op_desc->GetName().c_str());
        return PARAM_INVALID;
      }
      continue;
    }
    // for dynamic shape with control flow
    SetLabelForDynamic(node);
    if (IsNoTaskAndDumpNeeded(op_desc)) {
      GELOGD("node[%s] without task, and save op_desc and addr for dump", op_desc->GetName().c_str());
      const RuntimeParam &rts_param = GetRuntimeParam();
      const vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
      const vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
      const vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);
      vector<void *> tensor_device_addrs;
      tensor_device_addrs.insert(tensor_device_addrs.end(), input_data_addrs.begin(), input_data_addrs.end());
      tensor_device_addrs.insert(tensor_device_addrs.end(), output_data_addrs.begin(), output_data_addrs.end());
      tensor_device_addrs.insert(tensor_device_addrs.end(), workspace_data_addrs.begin(), workspace_data_addrs.end());
      void *addr = nullptr;
      auto size = kAddrLen * tensor_device_addrs.size();
      GE_CHK_RT_RET(rtMalloc(&addr, size, RT_MEMORY_HBM));

      rtError_t rt_ret = rtMemcpy(addr, size, tensor_device_addrs.data(), size, RT_MEMCPY_HOST_TO_DEVICE);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "rtMemcpy error, ret: 0x%X", rt_ret);
        GE_CHK_RT(rtFree(addr));
        return RT_ERROR_TO_GE_STATUS(rt_ret);
      }
      saved_task_addrs_.emplace(op_desc, addr);
    }

    GE_TIMESTAMP_RESTART(InitTbeHandle);
    uint32_t run_mode = static_cast<uint32_t>(domi::ImplyType::INVALID);
    if (AttrUtils::GetInt(op_desc, ATTR_NAME_IMPLY_TYPE, run_mode) &&
        run_mode == static_cast<uint32_t>(domi::ImplyType::TVM)) {
      // Skip no_task operator, such as concat and split.
      bool attr_notask = false;
      bool get_attr_notask_flag = ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOTASK, attr_notask);
      GE_IF_BOOL_EXEC(get_attr_notask_flag && attr_notask,
                      GELOGI("Node[name:%s, type:%s] does not generate task, skip initialization.",
                             op_desc->GetName().c_str(), op_desc->GetType().c_str());
                      continue;);

      Status status = InitTbeHandle(op_desc);
      if (status != SUCCESS) {
        GELOGE(status, "TBE init failed. %s", op_desc->GetName().c_str());
        return status;
      }
    }
    GE_TIMESTAMP_ADD(InitTbeHandle);
  }

  AdjustDataOpList(data_by_index);
  GE_TIMESTAMP_CALLNUM_END(LoadTBEKernelBinToOpDesc, "GraphLoader::LoadTBEKernelBinToOpDesc.");
  GE_TIMESTAMP_CALLNUM_END(InitTbeHandle, "GraphLoader::InitTbeHandle.");
  return SUCCESS;
}

Status DavinciModel::InitInputOutputForDynamic(const ComputeGraphPtr &compute_graph) {
  if (!known_node_) return SUCCESS;
  // for dynamic shape
  auto direct_nodes = compute_graph->GetDirectNode();
  for (size_t i = 0; i < direct_nodes.size(); i++) {
    auto node = direct_nodes.at(i);
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGE(PARAM_INVALID, "op_desc is null.");
      return PARAM_INVALID;
    }
    if (IsDataOp(op_desc->GetType())) {
      GELOGD("init data op %s", op_desc->GetName().c_str());
      data_op_list_.push_back(op_desc);
    }
    if (op_desc->GetType() == NETOUTPUT) {
      GELOGD("init netouput op %s", op_desc->GetName().c_str());
      output_op_list_.push_back(op_desc);
    }
  }
  return SUCCESS;
}

void DavinciModel::SetLabelForDynamic(const NodePtr &node) {
  if (known_node_ && node->GetOpDesc()->GetType() == LABELSWITCHBYINDEX) {
    for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
      auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      if (peer_out_data_anchor != nullptr) {
        string tensor_name = node->GetName();
        auto peer_node = peer_out_data_anchor->GetOwnerNode();
        (void)AttrUtils::SetStr(peer_node->GetOpDesc(), ATTR_DYNAMIC_SHAPE_FIXED_ADDR, tensor_name);
        (void)AttrUtils::SetInt(peer_node->GetOpDesc(), ATTR_DYNAMIC_SHAPE_FIXED_ADDR_INDEX, 0);
        tensor_name_to_peer_output_index_[tensor_name] = 0;
      }
    }
  }
}

/// @ingroup ge
/// @brief Data Op Initialize.
/// @param [in] NodePtr: Data Op.
/// @param [in/out] data_op_index: NetOutput addr size info.
/// @return Status
Status DavinciModel::InitDataOp(const NodePtr &node, uint32_t &data_op_index, map<uint32_t, OpDescPtr> &data_by_index) {
  // op_desc Checked by Init: Data, valid.
  auto op_desc = node->GetOpDesc();
  if (known_node_) {
    return SUCCESS;
  }
  uint32_t parent_index = 0;  // Ignore subgraph Data Node.
  if (AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGI("Init zero copy by subgraph Data node: %s.", op_desc->GetName().c_str());
    return InitInputBatchLabel(node);
  }

  data_op_list_.push_back(op_desc);

  // Make information for copy input data.
  const vector<int64_t> output_size_list = ModelUtils::GetOutputSize(op_desc);
  const vector<void *> virtual_addr_list = ModelUtils::GetOutputDataAddrs(runtime_param_, op_desc);
  const vector<int64_t> output_offset_list = op_desc->GetOutputOffset();
  if (output_offset_list.size() != virtual_addr_list.size()) {
    GELOGE(PARAM_INVALID, "virtual_addr size:%zu should be equal to offset size:%zu.", virtual_addr_list.size(),
           output_offset_list.size());
    return PARAM_INVALID;
  }
  auto data_index = data_op_index;
  if (AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, data_index)) {
    GELOGI("ge_train: get new index %u, old %u", data_index, data_op_index);
  }
  bool fusion_flag = false;
  ZeroCopyOffset zero_copy_offset;
  Status ret = zero_copy_offset.InitInputDataInfo(output_size_list, virtual_addr_list, op_desc, fusion_flag);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "InitDataInfo of input_info %s failed.", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  new_input_data_info_[data_index] = zero_copy_offset;
  data_by_index[data_index] = op_desc;

  for (size_t index = 0; index < virtual_addr_list.size(); ++index) {
    void *addr = virtual_addr_list.at(index);
    if (new_input_outside_addrs_.find(addr) != new_input_outside_addrs_.end()) {
      continue;
    }
    zero_copy_offset.SetInputOutsideAddrs(output_offset_list, addr, index, fusion_flag, real_virtual_addrs_);
    new_input_outside_addrs_[addr] = zero_copy_offset;
  }

  GELOGI("SetInputOutsideAddr success.");
  data_op_index++;
  if (InitInputZeroCopyNodes(node) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Input zero copy nodes init failed!");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Sort Data op list by index.
/// @param [in] data_by_index: map of Data Op.
/// @return
///
void DavinciModel::AdjustDataOpList(const map<uint32_t, OpDescPtr> &data_by_index) {
  if (data_by_index.size() != data_op_list_.size()) {
    GELOGW("Data map size: %zu, Data list size: %zu.", data_by_index.size(), data_op_list_.size());
    return;
  }

  data_op_list_.clear();
  for (auto &item : data_by_index) {
    data_op_list_.emplace_back(item.second);
  }
}

///
/// @ingroup ge
/// @brief input zero copy node Initialize.
/// @param [in] NodePtr: Data Op.
/// @return Status
///
Status DavinciModel::InitInputZeroCopyNodes(const NodePtr &node) {
  auto out_data_anchor = node->GetOutDataAnchor(kDataIndex);
  if (out_data_anchor == nullptr) {
    GELOGE(FAILED, "Out data anchor is nullptr");
    return FAILED;
  }
  for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    auto node = peer_in_data_anchor->GetOwnerNode();
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGE(FAILED, "Op desc is nullptr");
      return FAILED;
    }
    string batch_label;
    (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
    if (batch_label.empty()) {
      batch_label = kDefaultBatchLable;
    }
    if (zero_copy_op_id_batch_label_.find(op_desc->GetId()) == zero_copy_op_id_batch_label_.end()) {
      zero_copy_op_id_batch_label_.emplace(pair<int64_t, string>(op_desc->GetId(), batch_label));
      GELOGD("Init input zero copy nodes success, op name:%s, op id: %ld, batch label: %s.", op_desc->GetName().c_str(),
             op_desc->GetId(), batch_label.c_str());
    }
  }
  return SUCCESS;
}

/// @ingroup ge
/// @brief NetOutput Op Initialize.
/// @param [in] NodePtr: NetOutput Op.
/// @return Status
Status DavinciModel::InitNetOutput(const NodePtr &node) {
  // node->GetOpDesc Checked by Init: NetOutput, valid.
  auto op_desc = node->GetOpDesc();
  // excludes the function op sub graph, e.g. case,if
  if (known_node_) {
    return SUCCESS;
  }
  ComputeGraphPtr owner_graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(owner_graph);
  if (owner_graph->GetParentGraph() != nullptr) {
    GELOGI("Init zero copy by subgraph NetOutput node: %s.", op_desc->GetName().c_str());
    op_list_.erase(op_desc->GetId());
    return InitOutputBatchLabel(node);
  }

  output_op_list_.push_back(op_desc);
  // Make information for copy output data.
  const vector<int64_t> input_size_list = ModelUtils::GetInputSize(op_desc);
  const vector<void *> virtual_addr_list = ModelUtils::GetInputDataAddrs(runtime_param_, op_desc);
  const vector<int64_t> input_offset_list = op_desc->GetInputOffset();
  if (input_offset_list.size() != virtual_addr_list.size()) {
    GELOGE(PARAM_INVALID, "virtual_addr size should be equal to offset size.");
    return PARAM_INVALID;
  }
  if (input_size_list.empty() && virtual_addr_list.empty()) {
    GELOGI("NetOutput[%s] is empty.", op_desc->GetName().c_str());
    return SUCCESS;
  }
  if (input_size_list.empty() || input_size_list.size() != virtual_addr_list.size()) {
    GELOGE(PARAM_INVALID, "NetOutput[%s] init failed: Input size is %zu, Input addr is %zu", op_desc->GetName().c_str(),
           input_size_list.size(), virtual_addr_list.size());
    return PARAM_INVALID;
  }

  size_t num = new_output_data_info_.size();
  bool fusion_flag = false;

  for (size_t idx = 0; idx < input_size_list.size(); ++idx) {
    ZeroCopyOffset zero_copy_offset;
    Status ret = zero_copy_offset.InitOutputDataInfo(input_size_list, virtual_addr_list, op_desc, idx, fusion_flag);
    if (ret != SUCCESS) {
      GELOGE(PARAM_INVALID, "InitDataInfo of input_info %s failed.", op_desc->GetName().c_str());
      return PARAM_INVALID;
    }
    new_output_data_info_[num + idx] = zero_copy_offset;
    void *addr = virtual_addr_list.at(idx);
    int64_t input_offset = input_offset_list.at(idx);
    vector<void *> tensor_addrs;
    zero_copy_offset.SetOutputOutsideAddrs(input_offset, fusion_flag, addr, tensor_addrs);
    auto rslt = new_output_outside_addrs_.insert(std::pair<void *, ZeroCopyOffset>(addr, zero_copy_offset));
    if (!rslt.second) {
      GELOGI("same output_tensor_addr %p to different input_tensor of %s", addr, op_desc->GetName().c_str());
      DisableZeroCopy(addr);
    }

    for (size_t i = 0; i < tensor_addrs.size(); ++i) {
      void *real_addr = tensor_addrs.at(i);
      DisableZeroCopy(real_addr);
      real_virtual_addrs_.insert(real_addr);
    }
    GELOGI("SetOutputOutsideAddr success.");
  }

  if (InitOutputZeroCopyNodes(node) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Output zero copy nodes init failed!");
    return PARAM_INVALID;
  }
  GELOGI("DavinciModel::InitNetoutput success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief output zero copy node Initialize.
/// @param [in] NodePtr: netoutput Op.
/// @return Status
///
Status DavinciModel::InitOutputZeroCopyNodes(const NodePtr &node) {
  set<NodePtr> nodes_need_record;
  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_data_anchor == nullptr) {
      continue;
    }
    auto peer_node = peer_out_data_anchor->GetOwnerNode();
    nodes_need_record.emplace(peer_node);

    // Merge node output multiplexed input, upstream nodes need to be considered in multiple batch scenarios
    if (peer_node->GetType() == MERGE) {
      for (const auto &merge_peer_in_data_anchor : peer_node->GetAllInDataAnchors()) {
        auto merge_peer_out_data_anchor = merge_peer_in_data_anchor->GetPeerOutAnchor();
        if (merge_peer_out_data_anchor == nullptr) {
          continue;
        }
        auto merge_peer_node = merge_peer_out_data_anchor->GetOwnerNode();
        nodes_need_record.emplace(merge_peer_node);
      }
    } else {
      for (const auto &other_in_data_anchor : peer_out_data_anchor->GetPeerInDataAnchors()) {
        auto other_in_node = other_in_data_anchor->GetOwnerNode();
        if (other_in_node->GetType() != NETOUTPUT) {
          nodes_need_record.emplace(other_in_node);
        }
      }
    }
  }

  for (const auto &node_need_record : nodes_need_record) {
    auto op_desc = node_need_record->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    string batch_label;
    (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
    if (batch_label.empty()) {
      batch_label = kDefaultBatchLable;
    }
    if (zero_copy_op_id_batch_label_.find(op_desc->GetId()) == zero_copy_op_id_batch_label_.end()) {
      zero_copy_op_id_batch_label_.emplace(pair<int64_t, string>(op_desc->GetId(), batch_label));
      GELOGD("Init Output zero copy nodes success, op name:%s, op id: %ld, batch label: %s.",
             op_desc->GetName().c_str(), op_desc->GetId(), batch_label.c_str());
    }
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief input zero copy node Initialize.
/// @param [in] NodePtr: Data Op.
/// @return Status
///
Status DavinciModel::InitInputBatchLabel(const NodePtr &node) {
  string batch_label;
  if (!AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label)) {
    return SUCCESS;  // Not Multi-batch.
  }

  const auto &out_data_anchor = node->GetOutDataAnchor(kDataIndex);
  GE_CHECK_NOTNULL(out_data_anchor);

  for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    const auto &node = peer_in_data_anchor->GetOwnerNode();
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    if (zero_copy_op_id_batch_label_.find(op_desc->GetId()) == zero_copy_op_id_batch_label_.end()) {
      zero_copy_op_id_batch_label_[op_desc->GetId()] = batch_label;
      GELOGD("Init input zero copy nodes success, op name: %s, op id: %ld, batch label: %s", op_desc->GetName().c_str(),
             op_desc->GetId(), batch_label.c_str());
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief output zero copy node Initialize for Case.
/// @param [in] NodePtr: netoutput Op.
/// @return Status
///
Status DavinciModel::InitOutputBatchLabel(const NodePtr &node) {
  string batch_label;
  if (!AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label)) {
    return SUCCESS;  // Not Multi-batch.
  }

  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    const auto &peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_data_anchor == nullptr) {
      continue;
    }

    const auto &peer_node = peer_out_data_anchor->GetOwnerNode();
    const auto &op_desc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    if (zero_copy_op_id_batch_label_.find(op_desc->GetId()) == zero_copy_op_id_batch_label_.end()) {
      zero_copy_op_id_batch_label_[op_desc->GetId()] = batch_label;
      GELOGD("Init Output zero copy nodes success, op name: %s, op id: %ld, batch label: %s",
             op_desc->GetName().c_str(), op_desc->GetId(), batch_label.c_str());
    }
  }

  return SUCCESS;
}

/// @ingroup ge
/// @brief LabelSet Op Initialize.
/// @param [in] op_desc: LabelSet Op descriptor.
/// @return Status
Status DavinciModel::InitLabelSet(const OpDescPtr &op_desc) {
  uint32_t label_index = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, label_index)) {
    GELOGE(INTERNAL_ERROR, "InitLabelSet: %s attr [%s] not exist.", op_desc->GetName().c_str(),
           ATTR_NAME_LABEL_SWITCH_INDEX.c_str());
    return INTERNAL_ERROR;
  }
  if (label_index >= LabelNum()) {
    GELOGE(INTERNAL_ERROR, "InitLabelSet: label index: %u >= label size: %zu.", label_index, LabelNum());
    return INTERNAL_ERROR;
  }
  if (label_id_indication_.count(label_index) > 0) {
    GELOGE(INTERNAL_ERROR, "InitLabelSet: %s label index: %u already used.", op_desc->GetName().c_str(), label_index);
    return INTERNAL_ERROR;
  }

  rtStream_t stream = nullptr;
  uint32_t stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
  if (stream_list_.size() == 1) {
    stream = stream_list_[0];
  } else if (stream_list_.size() > stream_id) {
    stream = stream_list_[stream_id];
  } else {
    GELOGE(INTERNAL_ERROR, "InitLabelSet: stream index: %u >= stream size: %zu.", stream_id, stream_list_.size());
    return INTERNAL_ERROR;
  }

  rtLabel_t rt_label = nullptr;
  rtError_t rt_error = rtLabelCreateEx(&rt_label, stream);
  if (rt_error != RT_ERROR_NONE || rt_label == nullptr) {
    GELOGE(INTERNAL_ERROR, "InitLabelSet: %s create label failed, error=0x%x.", op_desc->GetName().c_str(), rt_error);
    return INTERNAL_ERROR;
  }

  GELOGI("InitLabelSet: label[%u]=%p stream[%u]=%p.", label_index, rt_label, stream_id, stream);
  label_id_indication_.insert(label_index);
  label_list_[label_index] = rt_label;
  return SUCCESS;
}

Status DavinciModel::InitVariable(const OpDescPtr &op_desc) {
  variable_op_list_.push_back(op_desc);
  return SUCCESS;
}

/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [in] input_queue_ids: input queue ids from user, nums equal Data Op.
/// @param [in] output_queue_ids: input queue ids from user, nums equal NetOutput Op.
/// @return: 0 for success / others for failed
Status DavinciModel::SetQueIds(const std::vector<uint32_t> &input_queue_ids,
                               const std::vector<uint32_t> &output_queue_ids) {
  if (input_queue_ids.empty() && output_queue_ids.empty()) {
    GELOGE(GE_EXEC_MODEL_QUEUE_ID_INVALID, "Param is empty");
    return GE_EXEC_MODEL_QUEUE_ID_INVALID;
  }

  input_queue_ids_ = input_queue_ids;
  output_queue_ids_ = output_queue_ids;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [in] input_que_ids: input queue ids from user, nums equal Data Op.
/// @param [in] output_que_ids: input queue ids from user, nums equal NetOutput Op.
/// @return: 0 for success / others for failed
///
Status DavinciModel::LoadWithQueue() {
  if (input_queue_ids_.empty() && output_queue_ids_.empty()) {
    return SUCCESS;
  }

  if (input_queue_ids_.size() != new_input_data_info_.size()) {
    GELOGE(GE_EXEC_MODEL_QUEUE_ID_INVALID, "Input queue ids not match model: input_queue=%zu input_data=%zu",
           input_queue_ids_.size(), new_input_data_info_.size());
    return GE_EXEC_MODEL_QUEUE_ID_INVALID;
  }

  if (output_queue_ids_.size() != new_output_data_info_.size()) {
    GELOGE(GE_EXEC_MODEL_QUEUE_ID_INVALID, "Output queue ids not match model: output_queue=%zu output_data=%zu",
           output_queue_ids_.size(), new_output_data_info_.size());
    return GE_EXEC_MODEL_QUEUE_ID_INVALID;
  }

  GE_CHK_STATUS_RET(AddHeadStream(), "Add head stream failed.");
  // Binding input_queue and Data Op.
  GE_CHK_STATUS_RET(BindInputQueue(), "Launch bind input queue failed.");
  GE_CHK_STATUS_RET(CpuTaskModelZeroCopy(input_mbuf_list_, new_input_outside_addrs_), "Launch zero copy failed.");

  // Binding output_queue and NetOutput Op.
  GE_CHK_STATUS_RET(BindOutputQueue(), "Launch bind output queue failed.");
  GE_CHK_STATUS_RET(CpuTaskModelZeroCopy(output_mbuf_list_, new_output_outside_addrs_), "Launch zero copy failed.");

  GE_CHK_STATUS_RET(CpuActiveStream(), "Launch active entry stream failed.");
  GE_CHK_STATUS_RET(CpuWaitEndGraph(), "Launch wait end graph failed.");
  GE_CHK_STATUS_RET(BindEnqueue(), "Launch enqueue failed.");
  GE_CHK_STATUS_RET(CpuModelRepeat(), "Launch model repeat failed.");

  return SUCCESS;
}

/// @ingroup ge
/// @brief queue schedule, Bind  input queue to Data output address.
/// @return: 0 for success / others for failed
Status DavinciModel::BindInputQueue() {
  // Caller checked: input_queue_ids_.size() == input_size_list_.size() != input_addr_list_.size()
  for (size_t i = 0; i < input_queue_ids_.size(); ++i) {
    auto it = new_input_data_info_.find(i);
    if (it == new_input_data_info_.end()) {
      GELOGE(FAILED, "Input not match: tensor num=%zu, Queue id index=%zu", new_input_data_info_.size(), i);
      return FAILED;
    }

    uint32_t queue_id = input_queue_ids_[i];
    if (it->second.GetDataInfo().empty()) {
      GELOGE(INTERNAL_ERROR, "the %zu input_queue not set data_info.", i);
      return INTERNAL_ERROR;
    }
    uint32_t data_size = static_cast<uint32_t>(it->second.GetDataInfo().at(0).first);
    uintptr_t data_addr = reinterpret_cast<uintptr_t>(it->second.GetDataInfo().at(0).second);
    GELOGI("BindInputToQueue: graph_%u index[%zu] queue id[%u] output addr[0x%lx] output size[%u]",
           runtime_param_.graph_id, i, queue_id, data_addr, data_size);

    rtError_t rt_ret = rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_INPUT_QUEUE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rtModelBindQueue failed, ret: 0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    if (CpuModelDequeue(queue_id) != SUCCESS) {
      return INTERNAL_ERROR;
    }
  }

  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, bind input queue to task.
/// @param [in] queue_id: input queue id from user.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuModelDequeue(uint32_t queue_id) {
  GELOGI("Set CpuKernel model dequeue task enter.");
  std::shared_ptr<CpuTaskModelDequeue> dequeue_task = MakeShared<CpuTaskModelDequeue>(rt_entry_stream_);
  if (dequeue_task == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make CpuTaskModelDequeue task failed.");
    return MEMALLOC_FAILED;
  }

  // Get DataOp Output address and bind to queue.
  uintptr_t in_mbuf = 0;
  Status status = dequeue_task->Init(queue_id, in_mbuf);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(dequeue_task);
  input_mbuf_list_.push_back(in_mbuf);
  GELOGI("Set CpuKernel model dequeue task success.");
  return SUCCESS;
}

Status DavinciModel::CpuTaskModelZeroCopy(std::vector<uintptr_t> &mbuf_list,
                                          std::map<const void *, ZeroCopyOffset> &outside_addrs) {
  GELOGI("Set CpuKernel model zero_copy task enter.");
  std::shared_ptr<CpuTaskZeroCopy> zero_copy = MakeShared<CpuTaskZeroCopy>(rt_entry_stream_);
  if (zero_copy == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make CpuTaskZeroCopy task failed.");
    return MEMALLOC_FAILED;
  }

  // mdc zero_copy not support l2 fusion
  Status status = zero_copy->Init(mbuf_list, outside_addrs);
  if (status != SUCCESS) {
    return status;
  }
  cpu_task_list_.push_back(zero_copy);
  GELOGI("Set CpuKernel model zero_copy task success.");
  return SUCCESS;
}

/// @ingroup ge
/// @brief queue schedule, bind output queue to NetOutput input address.
/// @return: 0 for success / others for failed
Status DavinciModel::BindOutputQueue() {
  // Caller checked: input_queue_ids_.size() == input_size_list_.size() != input_addr_list_.size()
  for (size_t i = 0; i < output_queue_ids_.size(); ++i) {
    auto it = new_output_data_info_.find(i);
    if (it == new_output_data_info_.end()) {
      GELOGE(FAILED, "Output not match: tensor num=%zu, Queue id index=%zu", new_output_data_info_.size(), i);
      return FAILED;
    }

    uint32_t queue_id = output_queue_ids_[i];
    if (it->second.GetDataInfo().empty()) {
      GELOGE(INTERNAL_ERROR, "the %zu output_queue not set data_info.", i);
      return INTERNAL_ERROR;
    }
    uint32_t data_size = static_cast<uint32_t>(it->second.GetDataInfo().at(0).first);
    uintptr_t data_addr = reinterpret_cast<uintptr_t>(it->second.GetDataInfo().at(0).second);
    GELOGI("BindOutputToQueue: graph_%u index[%zu] queue id[%u] input addr[0x%lx] input size[%u]",
           runtime_param_.graph_id, i, queue_id, data_addr, data_size);

    rtError_t rt_ret = rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_OUTPUT_QUEUE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rtModelBindQueue failed, ret: 0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    Status status = CpuModelPrepareOutput(data_addr, data_size);
    if (status != SUCCESS) {
      return status;
    }
  }

  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, bind output queue to task.
/// @param [in] addr: NetOutput Op input tensor address.
/// @param [in] size: NetOutput Op input tensor size.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuModelPrepareOutput(uintptr_t addr, uint32_t size) {
  GELOGI("Set CpuKernel model enqueue task enter.");
  if (input_mbuf_list_.empty()) {
    GELOGE(FAILED, "Need input mbuf for fill output mbuf head info.");
    return FAILED;
  }

  std::shared_ptr<CpuTaskPrepareOutput> prepare_output = MakeShared<CpuTaskPrepareOutput>(rt_entry_stream_);
  if (prepare_output == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make CpuTaskPrepareOutput task failed.");
    return MEMALLOC_FAILED;
  }

  uintptr_t out_mbuf = 0;
  if (prepare_output->Init(addr, size, input_mbuf_list_.back(), out_mbuf) != SUCCESS) {
    return FAILED;
  }

  cpu_task_list_.push_back(prepare_output);
  output_mbuf_list_.push_back(out_mbuf);
  GELOGI("Set CpuKernel model enqueue task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, active original model stream.
/// @return: 0 for success / others for failed
///
Status DavinciModel::CpuActiveStream() {
  GELOGI("Set CpuKernel active stream task enter.");
  std::shared_ptr<CpuTaskActiveEntry> active_entry = MakeShared<CpuTaskActiveEntry>(rt_entry_stream_);
  if (active_entry == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make CpuTaskActiveEntry task failed.");
    return MEMALLOC_FAILED;
  }

  Status status = active_entry->Init(rt_head_stream_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(active_entry);
  GELOGI("Set CpuKernel active stream task success.");
  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, wait for end graph.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuWaitEndGraph() {
  GELOGI("Set CpuKernel wait end graph task enter.");
  std::shared_ptr<CpuTaskWaitEndGraph> wait_endgraph = MakeShared<CpuTaskWaitEndGraph>(rt_entry_stream_);
  if (wait_endgraph == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make CpuTaskWaitEndGraph task failed.");
    return MEMALLOC_FAILED;
  }

  Status status = wait_endgraph->Init(runtime_model_id_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(wait_endgraph);
  GELOGI("Set CpuKernel wait end graph task success.");
  return SUCCESS;
}

Status DavinciModel::BindEnqueue() {
  for (size_t i = 0; i < output_queue_ids_.size(); ++i) {
    auto it = new_output_data_info_.find(i);
    if (it == new_output_data_info_.end()) {
      GELOGE(FAILED, "Output not match: tensor num=%zu, Queue id index=%zu", new_output_data_info_.size(), i);
      return FAILED;
    }

    uint32_t queue_id = output_queue_ids_[i];
    if (CpuModelEnqueue(queue_id, output_mbuf_list_[i]) != SUCCESS) {
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status DavinciModel::CpuModelEnqueue(uint32_t queue_id, uintptr_t out_mbuf) {
  GELOGI("Set CpuKernel model enqueue task enter.");
  std::shared_ptr<CpuTaskModelEnqueue> model_enqueue = MakeShared<CpuTaskModelEnqueue>(rt_entry_stream_);
  if (model_enqueue == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make CpuTaskModelEnqueue task failed.");
    return MEMALLOC_FAILED;
  }

  Status status = model_enqueue->Init(queue_id, out_mbuf);
  if (status != SUCCESS) {
    return status;
  }
  cpu_task_list_.push_back(model_enqueue);
  GELOGI("Set CpuKernel model enqueue task enter.");
  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, repeat run model.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuModelRepeat() {
  GELOGI("Set CpuKernel repeat task enter.");
  std::shared_ptr<CpuTaskModelRepeat> model_repeat = MakeShared<CpuTaskModelRepeat>(rt_entry_stream_);
  if (model_repeat == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make CpuTaskModelRepeat task failed.");
    return MEMALLOC_FAILED;
  }

  Status status = model_repeat->Init(runtime_model_id_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(model_repeat);
  GELOGI("Set CpuKernel repeat task success.");
  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc) {
  if ((data_op_list_.empty()) || (data_op_list_[0]->GetInputsSize()) != 1) {
    GELOGI("data_op_list_ is empty or input_desc size is not 1.");
  } else {
    std::vector<uint32_t> input_formats;
    GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats), "get input desc info failed.");
  }

  std::vector<uint32_t> outputFormats;
  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, outputFormats), "get output desc info failed.");

  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc,
                                            std::vector<uint32_t> &input_formats,
                                            std::vector<uint32_t> &outputFormats) {
  if ((data_op_list_.empty()) || (data_op_list_[0]->GetInputsSize()) != 1) {
    GELOGE(FAILED, "OP List Pointer is null or input_desc size is not 1!");
    return FAILED;
  }

  GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats), "get input desc info failed");

  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, outputFormats), "get ouput desc info failed");

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [out] batch_info
/// @param [out] dynamic_type
/// @return execute result
///
Status DavinciModel::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const {
  dynamic_type = dynamic_type_;
  batch_info = batch_info_;

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get combined dynamic dims info
/// @param [out] batch_info
/// @return None
///
void DavinciModel::GetCombinedDynamicDims(std::vector<std::vector<int64_t>> &batch_info) const {
  batch_info.clear();
  batch_info = combined_batch_info_;
}

///
/// @ingroup ge
/// @brief Get user designate shape order
/// @param [out] user_input_shape_order
/// @return None
///
void DavinciModel::GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) const {
  user_input_shape_order.clear();
  user_input_shape_order = user_designate_shape_order_;
}

///
/// @ingroup ge
/// @brief Get AIPP input info
/// @param [in] index
/// @param [out] aipp_info
/// @return execute result
///
Status DavinciModel::GetAIPPInfo(uint32_t index, AippConfigInfo &aipp_info) {
  GE_CHK_BOOL_RET_STATUS(index < data_op_list_.size(), PARAM_INVALID, "Index %u is invalid.", index);
  OpDescPtr data_op = data_op_list_[index];
  if (!data_op->HasAttr(ATTR_NAME_AIPP)) {
    GELOGW("GetAIPPInfo: there is not AIPP related with index %u.", index);
    return GE_AIPP_NOT_EXIST;
  }

  std::unique_ptr<domi::AippOpParams> aipp_params(new (std::nothrow) domi::AippOpParams());
  GE_CHECK_NOTNULL(aipp_params);

  ge::GeAttrValue::NAMED_ATTRS aipp_attr;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetNamedAttrs(data_op, ATTR_NAME_AIPP, aipp_attr), GE_AIPP_NOT_EXIST,
                         "Data node do not contain param aipp!");
  GE_CHK_STATUS_RET(OpUtils::ConvertAippParams(aipp_attr, aipp_params.get()), "get aipp params failed");
  GELOGI("GetAIPPInfo: node data: %s, type: %s, current index: %u, current node related input rank: %u",
         data_op->GetName().c_str(), data_op->GetType().c_str(), index, aipp_params->related_input_rank());

  GE_CHK_STATUS_RET(AippUtils::ConvertAippParams2AippInfo(aipp_params.get(), aipp_info),
                    "convert aipp params to aipp config info failed");

  return SUCCESS;
}

Status DavinciModel::GetAippType(uint32_t index, InputAippType &type, size_t &aipp_index) {
  GE_CHK_BOOL_RET_STATUS(index < data_op_list_.size(), PARAM_INVALID, "Index %u is invalid.", index);
  // Set default value
  type = DATA_WITHOUT_AIPP;
  aipp_index = 0xFFFFFFFF;  // default invalid value
  OpDescPtr data_op = data_op_list_[index];
  GE_CHECK_NOTNULL(data_op);
  if (!data_op->HasAttr(ATTR_DATA_RELATED_AIPP_MODE)) {
    GELOGW("There is no aipp releated info with index %u.", index);
    return SUCCESS;
  }
  std::string data_mode;
  (void)AttrUtils::GetStr(data_op, ATTR_DATA_RELATED_AIPP_MODE, data_mode);
  if (data_mode == "static_aipp") {
    type = DATA_WITH_STATIC_AIPP;
  } else if (data_mode == "dynamic_aipp") {
    type = DATA_WITH_DYNAMIC_AIPP;
  } else if (data_mode == "dynamic_aipp_conf") {
    type = DYNAMIC_AIPP_NODE;
  } else {
    GELOGE(INTERNAL_ERROR, "The info of aipp releated info %s is invalid with index %u.", data_mode.c_str(), index);
    return INTERNAL_ERROR;
  }

  if (type == DATA_WITH_DYNAMIC_AIPP) {
    string releated_name;
    (void)AttrUtils::GetStr(data_op, ATTR_DATA_AIPP_DATA_NAME_MAP, releated_name);
    for (size_t i = 0; i < data_op_list_.size(); ++i) {
      GE_CHECK_NOTNULL(data_op_list_[i]);
      if (data_op_list_[i]->GetName() == releated_name) {
        GELOGI("Find aipp_data [%s] index %zu from index %u", releated_name.c_str(), i, index);
        aipp_index = i;
      }
    }
    if (aipp_index == 0xFFFFFFFF) {
      GELOGE(INTERNAL_ERROR, "Can not find aipp data node from index %u", index);
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

void DavinciModel::SetDynamicSize(const std::vector<uint64_t> &batch_num, int32_t dynamic_type) {
  batch_size_.clear();
  if (batch_num.empty()) {
    GELOGD("User has not set dynammic data");
  }
  for (size_t i = 0; i < batch_num.size(); i++) {
    batch_size_.emplace_back(batch_num[i]);
  }

  dynamic_type_ = dynamic_type;
}

void DavinciModel::GetCurShape(std::vector<int64_t> &batch_info, int32_t &dynamic_type) {
  if (batch_size_.empty()) {
    GELOGD("User does not set dynamic size");
  }
  for (size_t i = 0; i < batch_size_.size(); i++) {
    GELOGI("Start to get current shape");
    batch_info.emplace_back(batch_size_[i]);
  }

  dynamic_type = dynamic_type_;
}

void DavinciModel::GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) {
  for (auto &op : output_op_list_) {
    if (op->GetType() != NETOUTPUT) {
      continue;
    }
    GELOGI("Start to get dynamic output dims attr");
    if (!AttrUtils::GetListStr(op, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, dynamic_output_shape_info)) {
      GELOGD("Can not get dynamic output dims attr");
    }
  }
}

Status DavinciModel::GetInputOutputDescInfoForZeroCopy(vector<InputOutputDescInfo> &input_desc,
                                                       vector<InputOutputDescInfo> &output_desc,
                                                       std::vector<uint32_t> &input_formats,
                                                       std::vector<uint32_t> &outputFormats) {
  if ((data_op_list_.empty()) || (1 != data_op_list_[0]->GetInputsSize())) {
    GELOGE(FAILED, "OP List Pointer is null or input_desc size is not 1!");
    return FAILED;
  }

  GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats), "get input desc info failed");

  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, outputFormats), "get ouput desc info failed");

  GE_CHK_BOOL_RET_STATUS(output_desc.size() == output_memory_size_list_.size(), INTERNAL_ERROR,
                         "output_desc size[%zu] not equal output_size_list_[%zu] size!", output_desc.size(),
                         output_memory_size_list_.size());

  /// For function zero copy,the momery should be aligned by 512 bytes.
  /// And, because of the cce op limit, size should be lager than the real shape size. The memory should be padded by 32
  /// bytes.
  /// *size equals to ((tensorDesc->dataSize + 2 * 32 - 1) / 32) * 32;
  for (size_t i = 0; i < output_memory_size_list_.size(); i++) {
    output_desc[i].size = output_memory_size_list_[i];
  }

  return SUCCESS;
}

void DavinciModel::SetInputDimsInfo(const vector<int64_t> &model_input_dims, Format &format,
                                    InputOutputDescInfo &input) {
  uint32_t n, c, h, w;
  n = format == FORMAT_NHWC ? NHWC_DIM_N : NCHW_DIM_N;
  c = format == FORMAT_NHWC ? NHWC_DIM_C : NCHW_DIM_C;
  h = format == FORMAT_NHWC ? NHWC_DIM_H : NCHW_DIM_H;
  w = format == FORMAT_NHWC ? NHWC_DIM_W : NCHW_DIM_W;

  if (model_input_dims.size() == static_cast<size_t>(NORMAL_TENSOR_SIZE)) {
    input.shape_info.num = model_input_dims[n];
    input.shape_info.height = model_input_dims[h];
    input.shape_info.width = model_input_dims[w];
    input.shape_info.channel = model_input_dims[c];
  }
  for (size_t k = 0; k < model_input_dims.size(); ++k) {
    input.shape_info.dims.push_back(model_input_dims[k]);
  }
  return;
}

void DavinciModel::CreateInputDimsInfo(const OpDescPtr &op_desc, Format format, InputOutputDescInfo &input) {
  if (is_new_model_desc_ && op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
    // When static aipp is set, need to get the model input dims which processed by aipp
    vector<int64_t> model_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, model_input_dims);
    SetInputDimsInfo(model_input_dims, format, input);
    return;
  }
  // judge if this data is linked dynamic aipp first, multiply batch has been considered
  if (op_desc->HasAttr(ATTR_DYNAMIC_AIPP_INPUT_DIMS)) {
    vector<int64_t> dynamic_aipp_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_DYNAMIC_AIPP_INPUT_DIMS, dynamic_aipp_input_dims);
    SetInputDimsInfo(dynamic_aipp_input_dims, format, input);
    return;
  } else {
    // judge if this data is multiply batch
    if (!op_desc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS)) {
      vector<int64_t> input_dims = op_desc->GetInputDescPtr(0)->GetShape().GetDims();
      SetInputDimsInfo(input_dims, format, input);
      return;
    } else {
      vector<int64_t> origin_input_dims;
      (void)AttrUtils::GetListInt(op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);
      SetInputDimsInfo(origin_input_dims, format, input);
      return;
    }
  }
}

Status DavinciModel::GetInputDescInfo(vector<InputOutputDescInfo> &input_desc, std::vector<uint32_t> &formats) {
  for (size_t index = 0; index < data_op_list_.size(); ++index) {
    InputOutputDescInfo input;
    GE_CHECK_NOTNULL(data_op_list_[index]);
    GE_CHECK_NOTNULL(data_op_list_[index]->GetInputDescPtr(0));

    Format format = data_op_list_[index]->GetInputDescPtr(0)->GetFormat();
    CreateInputDimsInfo(data_op_list_[index], format, input);

    input.data_type = data_op_list_[index]->GetInputDescPtr(0)->GetDataType();
    input.name = data_op_list_[index]->GetName();
    int64_t input_size = 0;
    GE_CHK_STATUS_RET(TensorUtils::GetSize(*data_op_list_[index]->GetInputDescPtr(0), input_size),
                      "get input size failed.");
    input.size = input_size;
    formats.push_back(format);
    input_desc.push_back(input);
  }
  // cause GetInputDescInfo called not only once, set is_new_model_desc_ to false after calc the model input dims
  is_new_model_desc_ = false;
  return SUCCESS;
}

void DavinciModel::CreateOutput(uint32_t index, OpDescPtr &op_desc, InputOutputDescInfo &output,
                                uint32_t &format_result) {
  /// netoutput input tensor desc
  GE_IF_BOOL_EXEC(op_desc->GetInputDescPtr(index) == nullptr, GELOGE(FAILED, "OpDesc GetInputDescPtr is nullptr");
                  return );
  Format format = op_desc->GetInputDescPtr(index)->GetFormat();
  GeShape shape = op_desc->GetInputDescPtr(index)->GetShape();
  DataType data_type = op_desc->GetInputDescPtr(index)->GetDataType();

  int64_t dims[] = {1, 1, 1, 1};
  format_result = format;
  if (format == FORMAT_ND) {  // for ND tensor
    for (size_t i = 0; i < shape.GetDimNum() && i < (sizeof(dims) / sizeof(dims[0])); i++) {
      dims[i] = shape.GetDim(i);
    }
  } else {                                                                    // FOR FORMAT_NHWC or FORMAT_NCHW
    dims[0] = shape.GetDim(format == FORMAT_NHWC ? NHWC_DIM_N : NCHW_DIM_N);  // 0: first dim
    dims[1] = shape.GetDim(format == FORMAT_NHWC ? NHWC_DIM_C : NCHW_DIM_C);  // 1: second dim
    dims[2] = shape.GetDim(format == FORMAT_NHWC ? NHWC_DIM_H : NCHW_DIM_H);  // 2: third dim
    dims[3] = shape.GetDim(format == FORMAT_NHWC ? NHWC_DIM_W : NCHW_DIM_W);  // 3: forth dim
  }
  output.shape_info.num = dims[0];      // 0: first dim
  output.shape_info.channel = dims[1];  // 1: second dim
  output.shape_info.height = dims[2];   // 2: third dim
  output.shape_info.width = dims[3];    // 3: forth dim

  if (op_desc->GetInputDescPtr(index)->GetFormat() == FORMAT_FRACTAL_Z) {  // FraczToHWCK
    int64_t k = shape.GetDim(0);                                           // 0: first dim
    int64_t c = shape.GetDim(1);                                           // 1: second dim
    int64_t h = shape.GetDim(2);                                           // 2: third dim
    int64_t w = shape.GetDim(3);                                           // 3: forth dim
    output.shape_info.dims.push_back(h);
    output.shape_info.dims.push_back(w);
    output.shape_info.dims.push_back(c);
    output.shape_info.dims.push_back(k);
    format_result = FORMAT_HWCN;
  } else {
    for (size_t j = 0; j < shape.GetDimNum(); j++) {
      output.shape_info.dims.push_back(shape.GetDim(j));
    }
  }

  int64_t tensor_size = 0;
  (void)TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);  // no need to check value
  output.size = static_cast<uint64_t>(tensor_size);
  output.data_type = op_desc->GetInputDescPtr(index)->GetDataType();
}

Status DavinciModel::GetOutputDescInfo(vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &formats) {
  GELOGI("Output node size: %zu", output_op_list_.size());
  for (size_t i = 0; i < output_op_list_.size(); i++) {
    auto &op_desc = output_op_list_[i];
    uint32_t out_size = static_cast<uint32_t>(op_desc->GetInputsSize());
    for (uint32_t index = 0; index < out_size; index++) {
      string output_name;
      InputOutputDescInfo output;
      uint32_t format_result;
      CreateOutput(index, op_desc, output, format_result);

      std::vector<std::string> src_name = op_desc->GetSrcName();
      std::vector<int64_t> src_index = op_desc->GetSrcIndex();
      GE_CHK_BOOL_RET_STATUS(src_name.size() > index && src_index.size() > index, INTERNAL_ERROR,
                             "construct output_name failed.");
      // forward compatbility, if old om has no out_node_name, need to return output follow origin way
      if (out_size == out_node_name_.size()) {
        // neweast plan, the index will add to name during generate model.
        bool contains_colon = out_node_name_[index].find(":") != std::string::npos;
        output_name =
            contains_colon ? out_node_name_[index] : out_node_name_[index] + ":" + std::to_string(src_index[index]);
      } else {
        output_name = std::string("output_") + std::to_string(index) + "_" + src_name[index] + "_" +
                      std::to_string(src_index[index]);
      }
      output.name = output_name;
      output_desc.push_back(output);
      formats.push_back(format_result);
    }
  }
  return SUCCESS;
}

ge::Format DavinciModel::GetFormat() {
  if ((data_op_list_.empty()) || data_op_list_[0] == nullptr || data_op_list_[0]->GetInputDescPtr(0) == nullptr) {
    GELOGW("OP List Pointer is null or input_desc size is not 1!");
    return FORMAT_NCHW;
  }

  return data_op_list_[0]->GetInputDescPtr(0)->GetFormat();
}

Status DavinciModel::CopyInputData(const InputData &input_data, bool device_data) {
  rtMemcpyKind_t kind = device_data ? RT_MEMCPY_DEVICE_TO_DEVICE : RT_MEMCPY_HOST_TO_DEVICE;
  const std::vector<DataBuffer> &blobs = input_data.blobs;
  for (const auto &data : new_input_data_info_) {
    if (data.first >= blobs.size()) {
      GELOGE(FAILED, "Blobs not match: blobs=%zu, tensor=%zu, index=%u, size=%ld", blobs.size(),
             new_input_data_info_.size(), data.first, data.second.GetDataInfo().at(0).first);
      return FAILED;
    }

    const DataBuffer &data_buf = blobs[data.first];
    if (data_buf.length == 0) {
      GELOGW("No data need to memcpy!");
      return SUCCESS;
    }
    uint64_t data_size = data.second.GetDataSize();
    GE_CHK_BOOL_RET_STATUS(data_size >= data_buf.length, PARAM_INVALID,
                           "input data size(%lu) does not match model required size(%lu), ret failed.", data_buf.length,
                           data_size);
    void *mem_addr = data.second.GetBasicAddr();
    void *data_buf_addr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(data_buf.data));
    uint64_t data_buf_length = data_buf.length;
    GELOGI("[IMAS]CopyPlainData memcpy graph_%lu type[F] input[%lu] dst[%p] src[%p] mem_size[%lu] datasize[%lu]",
           runtime_param_.graph_id, data.first, mem_addr, data_buf_addr, data_size, data_buf_length);
    GE_CHK_RT_RET(rtMemcpy(mem_addr, data_size, data_buf_addr, data_buf_length, kind));
  }

  return SUCCESS;
}

Status DavinciModel::SyncVarData() {
  GELOGI("Sync var data, model id:%u", model_id_);
  Status ret = SUCCESS;

  OpDescPtr global_step = GetVariableOp(NODE_NAME_GLOBAL_STEP);
  if (global_step != nullptr) {
    auto v_output_size = ModelUtils::GetOutputSize(global_step);
    auto v_output_addr = ModelUtils::GetOutputDataAddrs(runtime_param_, global_step);
    if (v_output_size.empty() || v_output_addr.empty()) {
      GELOGE(PARAM_INVALID, "global step op:%s not set output", global_step->GetName().c_str());
      return PARAM_INVALID;
    }
    std::vector<uint64_t> v_step;
    v_step.push_back(iterator_count_);
    GE_CHK_RT_RET(rtMemcpy(v_output_addr[0], v_output_size[0], v_step.data(), v_step.size() * sizeof(uint64_t),
                           RT_MEMCPY_HOST_TO_DEVICE));
  }

  for (auto op_desc : variable_op_list_) {
    ret =
        VarManager::Instance(session_id_)->SyncVarData(runtime_param_.graph_id, op_desc->GetName(), op_desc, mem_base_);
    GE_CHK_BOOL_EXEC(ret == SUCCESS, break, "sync var data ret failed, model id:%u, op name:%s.", model_id_,
                     op_desc->GetName().c_str());
  }
  return ret;
}

inline int64_t SumSize(const vector<int64_t> &size_list) {
  int64_t sum_size = 0;
  for (const int64_t &size : size_list) {
    sum_size += size;
  }
  return sum_size;
}

Status DavinciModel::SinkModelProfile() {
  // profiling plugin must be registered
  Msprof::Engine::Reporter *reporter = PluginImpl::GetPluginReporter();
  GE_IF_BOOL_EXEC(reporter == nullptr, GELOGI("Profiling report is nullptr!"); return SUCCESS);

  GELOGI("Start collect model load profiling data.");

  Msprof::Engine::ReporterData reporter_data{};
  // report model data tag name
  std::string tag_name;
  tag_name.append("model_load_info_").append(std::to_string(this->Id()));
  GE_CHK_BOOL_EXEC(memcpy_s(reporter_data.tag, MSPROF_ENGINE_MAX_TAG_LEN, tag_name.c_str(), tag_name.size()) == EOK,
                   return FAILED, "Sink model tag memcpy error.");

  // Model Header
  string name;
  if (!om_name_.empty()) {
    name = om_name_;
  } else {
    name = name_;
  }
  size_t name_len = name.size();
  reporter_data.deviceId = device_id_;
  reporter_data.data = (unsigned char *)&name_len;
  reporter_data.dataLen = sizeof(int32_t);
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                   this->Id());

  reporter_data.data = (unsigned char *)name.c_str();
  reporter_data.dataLen = name.size();
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                   this->Id());

  uint32_t model_id = this->Id();
  reporter_data.data = (unsigned char *)&model_id;
  reporter_data.dataLen = sizeof(uint32_t);
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                   this->Id());

  // Load Start/End Time
  int64_t start_time = this->GetLoadBeginTime();
  reporter_data.data = (unsigned char *)&start_time;
  reporter_data.dataLen = sizeof(int64_t);
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                   this->Id());

  int64_t end_time = this->GetLoadEndTime();
  reporter_data.data = (unsigned char *)&end_time;
  reporter_data.dataLen = sizeof(int64_t);
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                   this->Id());

  int32_t task_num = task_list_.size();
  std::multimap<uint32_t, uint32_t> op_id_map;
  std::set<uint32_t> task_id_set;
  for (int32_t i = 0; i < task_num; i++) {
    auto task = task_list_[i];
    GE_CHECK_NOTNULL(task);
    auto fusion_op_info = task->GetFusionOpInfo();
    // when type is RT_MODEL_TASK_KERNEL, ctx is not null
    if (fusion_op_info != nullptr) {
      uint32_t op_num = fusion_op_info->original_op_names.size();
      uint32_t task_id = task->GetTaskID();
      if (op_num > 0) {
        GELOGI("task.id = %u, opNum = %u", task_id, op_num);
        op_id_map.insert(std::make_pair(fusion_op_info->op_index, task_id));
      }
    }
  }

  struct memoryInfo {
    int64_t input_size;
    int64_t output_size;
    int64_t weight_size;
    int64_t workspace_size;
    int64_t total_size;

    memoryInfo() : input_size(0), output_size(0), weight_size(0), workspace_size(0), total_size(0) {}
  };

  using CIT = std::multimap<uint32_t, uint32_t>::const_iterator;
  using Range = std::pair<CIT, CIT>;
  for (int32_t i = 0; i < task_num; i++) {
    auto task = task_list_[i];
    GE_CHECK_NOTNULL(task);
    auto fusion_op_info = task->GetFusionOpInfo();
    if (fusion_op_info != nullptr && fusion_op_info->original_op_names.size() > 0) {
      uint32_t task_id = task->GetTaskID();
      uint32_t op_num = fusion_op_info->original_op_names.size();
      uint32_t task_count = 0;
      if (task_id_set.count(task_id) != 0) {
        continue;
      }

      uint32_t op_id = fusion_op_info->op_index;
      Range range = op_id_map.equal_range(op_id);
      for (CIT range_idx = range.first; range_idx != range.second; ++range_idx) {
        task_count++;
        uint32_t task_id = range_idx->second;
        task_id_set.insert(task_id);
      }

      // op name after fusion
      string fusion_op_name = fusion_op_info->op_name;
      int32_t fusion_op_name_len = fusion_op_name.size() == 0 ? 1 : fusion_op_name.size();
      reporter_data.data = (unsigned char *)&fusion_op_name_len;
      reporter_data.dataLen = sizeof(int32_t);
      GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                       this->Id());

      reporter_data.data = (unsigned char *)fusion_op_name.c_str();
      reporter_data.dataLen = fusion_op_name_len;
      GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                       this->Id());

      // original op name before fusion
      reporter_data.data = (unsigned char *)&op_num;
      reporter_data.dataLen = sizeof(int32_t);
      GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                       this->Id());

      for (uint32_t k = 0; k < op_num; k++) {
        std::string op_name = fusion_op_info->original_op_names[k];
        int32_t op_name_len = op_name.size() == 0 ? 1 : op_name.size();
        reporter_data.data = (unsigned char *)&op_name_len;
        reporter_data.dataLen = sizeof(int32_t);
        GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                         this->Id());
        reporter_data.data = (unsigned char *)op_name.c_str();
        reporter_data.dataLen = op_name_len;
        GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                         this->Id());
      }

      // stream id info
      uint32_t streamId = task->GetStreamId();
      reporter_data.data = (unsigned char *)&streamId;
      reporter_data.dataLen = sizeof(int32_t);
      GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                       this->Id());

      // memory info
      struct memoryInfo memory_info;
      uint32_t op_index = fusion_op_info->op_index;
      auto iter = op_list_.find(op_index);
      GE_CHK_BOOL_EXEC(iter != op_list_.end(), return FAILED, "index is out of range, index: %u", op_index);
      auto op_desc = iter->second;
      memory_info.input_size = SumSize(ModelUtils::GetInputSize(op_desc));
      memory_info.output_size = SumSize(ModelUtils::GetOutputSize(op_desc));
      memory_info.workspace_size = SumSize(ModelUtils::GetWorkspaceSize(op_desc));
      memory_info.weight_size = SumSize(ModelUtils::GetWeightSize(op_desc));
      memory_info.total_size =
          memory_info.weight_size + memory_info.input_size + memory_info.output_size + memory_info.workspace_size;
      reporter_data.data = (unsigned char *)&memory_info;
      reporter_data.dataLen = sizeof(struct memoryInfo);
      GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                       this->Id());

      // task info
      reporter_data.data = (unsigned char *)&task_count;
      reporter_data.dataLen = sizeof(uint32_t);
      GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                       this->Id());

      Range task_range = op_id_map.equal_range(op_id);
      for (CIT idx = task_range.first; idx != task_range.second; ++idx) {
        uint32_t task_id = idx->second;
        reporter_data.data = (unsigned char *)&task_id;
        reporter_data.dataLen = sizeof(uint32_t);
        GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                         this->Id());
      }
    }
  }
  return SUCCESS;
}

Status DavinciModel::SinkTimeProfile(const InputData &current_data) {
  // profiling plugin must be registered
  Msprof::Engine::Reporter *reporter = PluginImpl::GetPluginReporter();
  GE_IF_BOOL_EXEC(reporter == nullptr, GELOGI("Profiling report is nullptr!"); return SUCCESS);

  Msprof::Engine::ReporterData reporter_data{};
  // report model data tag name
  std::string tag_name;
  tag_name.append("model_time_info_")
      .append(std::to_string(this->Id()))
      .append("_")
      .append(std::to_string(current_data.index));

  GE_CHK_BOOL_EXEC(memcpy_s(reporter_data.tag, MSPROF_ENGINE_MAX_TAG_LEN, tag_name.c_str(), tag_name.size()) == EOK,
                   return FAILED, "Sink model tag memcpy error.");
  // device id
  reporter_data.deviceId = device_id_;

  // Model Header
  string name;
  if (!om_name_.empty()) {
    name = om_name_;
  } else {
    name = name_;
  }
  size_t name_len = name.size();
  reporter_data.data = (unsigned char *)&name_len;
  reporter_data.dataLen = sizeof(int32_t);
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                   this->Id());

  reporter_data.data = (unsigned char *)name.c_str();
  reporter_data.dataLen = name.size();
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED, "Reporter data fail, model id:%u.",
                   this->Id());

  // request id
  uint64_t request_id = current_data.request_id;
  reporter_data.data = (unsigned char *)&request_id;
  reporter_data.dataLen = sizeof(uint32_t);
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED,
                   "Reporter data fail, model id:%u, data index:%u.", this->Id(), current_data.index);

  // thread id
  int32_t thread_id = GetDataInputTid();
  reporter_data.data = (unsigned char *)&thread_id;
  reporter_data.dataLen = sizeof(int32_t);
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED,
                   "Reporter data fail, model id:%u, data index:%u.", this->Id(), current_data.index);

  // time info
  time_info_.modelId = this->Id();
  reporter_data.data = (unsigned char *)&time_info_;
  reporter_data.dataLen = sizeof(struct timeInfo);
  GE_CHK_BOOL_EXEC(reporter->Report(&reporter_data) == SUCCESS, return FAILED,
                   "Reporter data fail, model id:%u, data index:%u.", this->Id(), current_data.index);

  return SUCCESS;
}

void DavinciModel::SetProfileTime(ModelProcStage stage, int64_t endTime) {
  int64_t time = endTime;

  if (time == 0) {
    mmTimespec timespec = mmGetTickCount();
    time = timespec.tv_sec * 1000 * 1000 * 1000 + timespec.tv_nsec;  // 1000 ^ 3 converts second to nanosecond
  }

  switch (stage) {
    case MODEL_LOAD_START:
      load_begin_time_ = time;
      break;
    case MODEL_LOAD_END:
      load_end_time_ = time;
      break;
    case MODEL_PRE_PROC_START:
      time_info_.processBeginTime = time;
      break;
    case MODEL_PRE_PROC_END:
      time_info_.processEndTime = time;
      break;
    case MODEL_INFER_START:
      time_info_.inferenceBeginTime = time;
      break;
    case MODEL_INFER_END:
      time_info_.inferenceEndTime = time;
      break;
    case MODEL_AFTER_PROC_START:
      time_info_.dumpBeginTime = time;
      break;
    case MODEL_AFTER_PROC_END:
      time_info_.dumpEndTime = time;
      break;
    default:
      break;
  }
  return;
}

///
/// @ingroup ge
/// @brief send Output Op result to upper layer
/// @already malloced in ModelLoad, no need to malloc again
/// @param [in] data_id: the index of output_data
/// @param [in/out] output_data: real user output_data
/// @param [in] kind: the kind of rtMemcpy
/// @return Status result
/// @author
///
Status DavinciModel::CopyOutputData(uint32_t data_id, OutputData &output_data, rtMemcpyKind_t kind) {
  if (output_op_list_.empty()) {
    Status ret = SyncVarData();
    return ret;
  }

  output_data.index = data_id;
  output_data.model_id = model_id_;
  if (output_data.blobs.size() != new_output_data_info_.size()) {
    GELOGE(FAILED, "Output data buffer num=%zu not equal model data num=%zu", output_data.blobs.size(),
           new_output_data_info_.size());
    return FAILED;
  }

  std::vector<DataBuffer> &blobs = output_data.blobs;
  for (const auto &output : new_output_data_info_) {
    if (output.first >= blobs.size()) {
      GELOGE(FAILED, "Blobs not match: blobs=%zu, tensor=%zu, index=%u, size=%ld", blobs.size(),
             new_input_data_info_.size(), output.first, output.second.GetDataInfo().at(0).first);
      return FAILED;
    }

    if ((kind == RT_MEMCPY_DEVICE_TO_DEVICE) && (copy_only_addrs_.count(output.second.GetBasicAddr()) == 0)) {
      continue;  // Skip: Feed by zero copy.
    }

    DataBuffer &buffer = blobs[output.first];
    uint64_t mem_size = static_cast<uint64_t>(output.second.GetDataSize());
    if ((buffer.length == 0) || (mem_size == 0)) {
      GELOGI("Length of data is zero, No need copy. output tensor index=%u", output.first);
      continue;
    }
    if (is_dynamic_) {
      GELOGI("No need to check output data size.");
    } else if (buffer.length < mem_size) {
      GELOGE(FAILED, "Tensor data size=%lu, buffer size=%u", mem_size, buffer.length);
      return FAILED;
    } else if (buffer.length > mem_size) {
      GELOGW("Tensor data size=%lu, buffer size=%u", mem_size, buffer.length);
    }
    uint64_t data_size = output.second.GetDataSize();
    uint64_t buffer_length = buffer.length;
    void *buffer_addr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(buffer.data));

    GELOGI("[IMAS]CopyPlainData memcpy graph_%u type[F] output[%u] memaddr[%p] mem_size[%lu] datasize[%u]",
           runtime_param_.graph_id, output.first, output.second.GetBasicAddr(), data_size, buffer_length);
    GE_CHK_RT_RET(rtMemcpy(buffer_addr, buffer_length, output.second.GetBasicAddr(), data_size, kind));
  }
  return SUCCESS;
}

Status DavinciModel::GenOutputTensorInfo(const OpDescPtr &op_desc, uint32_t data_index, OutputData *output_data,
                                         std::vector<ge::OutputTensorInfo> &outputs) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(output_data);
  if (output_data->blobs.size() > data_index) {
    GELOGI("No need to generate output tensor info, model id:%u", model_id_);
    return SUCCESS;
  }
  std::vector<int64_t> out_buffer_size_vec;
  std::vector<std::vector<int64_t>> shape_info_vec;
  size_t input_num = op_desc->GetInputsSize();
  for (size_t i = 0; i < input_num; ++i) {
    int64_t size = 0;
    auto input_desc = op_desc->GetInputDescPtr(i);
    GE_CHECK_NOTNULL(input_desc);
    auto ret = TensorUtils::GetTensorSizeInBytes(*input_desc, size);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(ret, "Get size from TensorDesc failed, op:%s, input index:%zu", op_desc->GetName().c_str(), i);
      return ret;
    }
    out_buffer_size_vec.push_back(size);
    shape_info_vec.push_back(input_desc->GetShape().GetDims());
  }

  GELOGI("Output blobs size:%zu, data index:%u, model id:%u", out_buffer_size_vec.size(), data_index, model_id_);
  for (size_t i = 0; i < out_buffer_size_vec.size(); ++i) {
    std::unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[out_buffer_size_vec[i]]);
    if (data_buf == nullptr) {
      GELOGE(GE_GRAPH_MALLOC_FAILED, "Malloc buffer failed.");
      return GE_GRAPH_MALLOC_FAILED;
    }
    output_data->blobs.push_back({data_buf.get(), static_cast<uint64_t>(out_buffer_size_vec[i]), false});
    ge::OutputTensorInfo output;
    output.dims = shape_info_vec[i];
    output.data = std::move(data_buf);
    output.length = out_buffer_size_vec[i];
    outputs.emplace_back(std::move(output));
    GELOGI("Output index:%zu, data_length:%lu.", i, output.length);
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief send Output Op result to upper layer
/// @already malloced in ModelLoad, no need to malloc again
/// @param [in] data_id: the index of output_data
/// @param [in] rslt_flg: result flag
/// @param [in] seq_end_flag: sequence end flag
/// @param [out] output_data: real user output_data
/// @return Status result
/// @author
///
Status DavinciModel::ReturnResult(uint32_t data_id, const bool rslt_flg, const bool seq_end_flag,
                                  OutputData *output_data) {
  GE_CHK_BOOL_EXEC(listener_ != nullptr, return PARAM_INVALID, "listener_ is null.");
  std::vector<ge::OutputTensorInfo> outputs;

  // return result is not required
  if (!rslt_flg && !seq_end_flag) {
    GELOGW("Compute failed, model id: %u", model_id_);
    auto model_manager = ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    auto exception_infos = model_manager->GetExceptionInfos();
    if (exception_infos.size() > 0) {
      GE_CHK_STATUS_RET(data_dumper_.DumpExceptionInfo(exception_infos), "Dump exception info failed");
    } else {
      GELOGI("Exception info is null");
    }
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, INTERNAL_ERROR, outputs), "OnComputeDone failed.");
    return INTERNAL_ERROR;
  }

  if (output_op_list_.empty()) {
    GELOGW("Output tensor list is empty, model id: %u", model_id_);
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, INTERNAL_ERROR, outputs), "OnComputeDone failed.");
    return INTERNAL_ERROR;
  }

  GE_CHECK_NOTNULL(output_data);
  // index of data in output_data
  uint32_t data_index = 0;

  output_data->index = data_id;
  output_data->model_id = model_id_;

  // copy output data from op to designated position
  for (auto &op_desc : output_op_list_) {
    if (GenOutputTensorInfo(op_desc, data_index, output_data, outputs) != SUCCESS) {
      return INTERNAL_ERROR;
    }
    data_index += op_desc->GetInputsSize();
  }

  if (CopyOutputData(data_id, *output_data, RT_MEMCPY_DEVICE_TO_HOST) != SUCCESS) {
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, INTERNAL_ERROR, outputs), "OnComputeDone failed");
    return INTERNAL_ERROR;
  }

  if (seq_end_flag) {
    GELOGW("End of sequence, model id: %u", model_id_);
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, END_OF_SEQUENCE, outputs), "OnCompute Done failed.");
    return END_OF_SEQUENCE;
  }
  GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, SUCCESS, outputs), "OnComputeDone failed");
  return SUCCESS;
}
///
/// @ingroup ge
/// @brief return not output to upper layer for cloud case
/// @param [in] data_id
/// @return Status result
///
Status DavinciModel::ReturnNoOutput(uint32_t data_id) {
  GELOGI("ReturnNoOutput model id:%u", model_id_);
  for (auto op_desc : variable_op_list_) {
    Status ret = VarManager::Instance(session_id_)
                     ->SyncBroadCastData2Var(runtime_param_.graph_id, op_desc->GetName(), op_desc, mem_base_);
    GE_CHK_BOOL_EXEC(ret == SUCCESS, break, "sync var data ret failed, model id:%u, op name:%s.", model_id_,
                     op_desc->GetName().c_str());
  }

  GE_CHK_BOOL_EXEC(listener_ != nullptr, return PARAM_INVALID, "listener_ is null!");
  std::vector<ge::OutputTensorInfo> outputs;
  GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, SUCCESS, outputs), "OnComputeDone failed.");
  return SUCCESS;
}

void *DavinciModel::Run(DavinciModel *model) {
  GE_CHK_BOOL_EXEC(model != nullptr,
                   CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
                   return nullptr, "model_pointer is null!")
  bool seq_end_flag = false;
  uint32_t model_id = model->Id();
  uint32_t device_id = model->GetDeviceId();

  GELOGI("Model Run thread start, model_id:%u.", model_id);
  rtError_t rt_ret = rtSetDevice(static_cast<int32_t>(device_id));
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "Model run rtsetdevice failed.");
    return nullptr;
  }
  // DeviceReset before thread run finished!
  GE_MAKE_GUARD(not_used_var, [&] { GE_CHK_RT(rtDeviceReset(device_id)); });

  while (model->RunFlag()) {
    bool rslt_flg = true;
    if (model->GetDataInputer() == nullptr) {
      GELOGW("Data inputer is nullptr.");
      CsaInteract::GetInstance().StoreInternalErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
      break;
    }

    std::shared_ptr<InputDataWrapper> data_wrapper;
    Status ret = model->GetDataInputer()->Pop(data_wrapper);
    if (data_wrapper == nullptr || ret != SUCCESS) {
      GELOGI("data_wrapper is null!");
      continue;
    }
    GELOGI("Getting the input data, model_id:%u", model_id);

    GE_IF_BOOL_EXEC(!model->RunFlag(), break);

    InputData current_data = data_wrapper->GetInput();
    GELOGI("Model thread Run begin, model id:%u, data index:%u.", model_id, current_data.index);

    GE_TIMESTAMP_START(Model_SyncVarData);
    ret = model->SyncVarData();
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ret != SUCCESS, (void)model->ReturnResult(current_data.index, false, false, data_wrapper->GetOutput());
        CsaInteract::GetInstance().StoreInternalErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
        continue, "Copy input data to model failed.");  // [No need to check value]
    GE_IF_BOOL_EXEC(model->is_first_execute_, GE_TIMESTAMP_EVENT_END(Model_SyncVarData, "Model Run SyncVarData"));

    GELOGI("Copy input data, model id:%u", model_id);
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(),
                    model->SetProfileTime(MODEL_PRE_PROC_START));
    ret = model->CopyInputData(current_data, false);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ret != SUCCESS, (void)model->ReturnResult(current_data.index, false, false, data_wrapper->GetOutput());
        CsaInteract::GetInstance().StoreInternalErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
        continue, "Copy input data to model failed.");  // [No need to check value]
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), model->SetProfileTime(MODEL_PRE_PROC_END));
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), model->SetProfileTime(MODEL_INFER_START));
    if (ProfilingManager::Instance().ProfilingOpTraceOn()) {
      GELOGI("GetOpTraceIterNum:%d", ProfilingManager::Instance().GetOpTraceIterNum());
      for (int32_t i = 0; i < ProfilingManager::Instance().GetOpTraceIterNum(); i++) {
        if (!ProfilingManager::Instance().ProfilingLoadFlag()) {
          vector<int32_t> prof_device_id_vec = ProfilingManager::Instance().GetProfilingDeviceId();
          for (size_t j = 0; j < prof_device_id_vec.size(); ++j) {
            // just profiling, no need to check value
            (void)ProfilingManager::Instance().StartProfiling(i, prof_device_id_vec[j]);
          }
        }

        GELOGI("rtModelExecute start.");
        rt_ret = rtModelExecute(model->rt_model_handle_, model->rt_model_stream_, 0);
        GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, rslt_flg = false;
                        (void)model->ReturnResult(current_data.index, false, false, data_wrapper->GetOutput());
                        continue);  // [No need to check value]
        GELOGI("rtModelExecute end");

        GELOGI("rtStreamSynchronize start.");
        rt_ret = rtStreamSynchronize(model->rt_model_stream_);
        GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, rslt_flg = false;
                        (void)model->ReturnResult(current_data.index, false, seq_end_flag, data_wrapper->GetOutput());
                        continue);  // [No need to check value]
        GELOGI("rtStreamSynchronize end.");
        (void)ProfilingManager::Instance().StopProfiling();  // just profiling, no need to check value
      }
    } else {
      GE_TIMESTAMP_START(rtModelExecute);
      GELOGI("rtModelExecute start.");
      rt_ret = rtModelExecute(model->rt_model_handle_, model->rt_model_stream_, 0);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, rslt_flg = false;
                      (void)model->ReturnResult(current_data.index, false, false, data_wrapper->GetOutput());
                      CsaInteract::GetInstance().WriteErrorCode(rt_ret, ERROR_MODULE_RUNTIME, JOBSUBSTATE_GRAPH_EXEC);
                      continue);
      GELOGI("rtModelExecute end");
      GE_IF_BOOL_EXEC(model->is_first_execute_, GE_TIMESTAMP_EVENT_END(rtModelExecute, "GraphExcute::rtModelExecute"));

      GE_TIMESTAMP_START(rtStreamSynchronize);
      GELOGI("rtStreamSynchronize start.");
      rt_ret = rtStreamSynchronize(model->rt_model_stream_);
      if (rt_ret == RT_ERROR_END_OF_SEQUENCE) {
        seq_end_flag = true;
      }
      GE_IF_BOOL_EXEC(
          rt_ret != RT_ERROR_NONE, rslt_flg = false; GELOGI("seq_end_flg: %d", seq_end_flag);
          (void)model->ReturnResult(current_data.index, false, seq_end_flag,
                                    data_wrapper->GetOutput());  // [No need to check value]
          CsaInteract::GetInstance().StoreInternalErrorCode(rt_ret, ERROR_MODULE_RUNTIME, JOBSUBSTATE_GRAPH_EXEC);
          continue);
      GELOGI("rtStreamSynchronize end.");
      GE_IF_BOOL_EXEC(model->is_first_execute_,
                      GE_TIMESTAMP_EVENT_END(rtStreamSynchronize, "GraphExcute::Wait for rtStreamSynchronize"));
      GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), model->SetProfileTime(MODEL_INFER_END));
    }

    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(),
                    model->SetProfileTime(MODEL_AFTER_PROC_START));
    GE_TIMESTAMP_START(ReturnResult3);
    // copy output data from device to host
    GE_IF_BOOL_EXEC(!model->output_op_list_.empty(),
                    (void)model->ReturnResult(current_data.index, rslt_flg, false, data_wrapper->GetOutput()))
    // copy output data from device to host for variable graph
    GE_IF_BOOL_EXEC(model->output_op_list_.empty(), (void)model->ReturnNoOutput(current_data.index));
    GE_IF_BOOL_EXEC(model->is_first_execute_,
                    GE_TIMESTAMP_EVENT_END(ReturnResult3, "GraphExcute::CopyDataFromDeviceToHost"));
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(),
                    model->SetProfileTime(MODEL_AFTER_PROC_END));
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), (void)model->SinkTimeProfile(current_data));

    model->iterator_count_++;
    model->is_first_execute_ = false;
    GELOGI("run iterator count is %lu", model->iterator_count_);
  }

  CsaInteract::GetInstance().WriteInternalErrorCode();
  GELOGI("Model run end, model id:%u", model->model_id_);
  return nullptr;
}

///
/// @ingroup ge
/// @brief call API provided by data inputer to destroy thread
/// @param [in] no
/// @return Status Destroy result
/// @author
///
Status DavinciModel::DestroyThread() {
  GE_CHK_BOOL_RET_STATUS(data_inputer_ != nullptr, INTERNAL_ERROR, "data_inputer_ is nullptr.");

  run_flg_ = false;

  data_inputer_->Stop();

  if (thread_id_.joinable()) {
    thread_id_.join();
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief create model std::thread,
/// @brief start to execute Model
/// @param [in] no
/// @return Status create model thread and execute result
/// @author
///
Status DavinciModel::ModelRunStart() {
  GE_CHK_BOOL_RET_STATUS(data_inputer_ != nullptr, INTERNAL_ERROR, "data_inputer_ is nullptr.");

  LockRunFlg();
  GE_MAKE_GUARD(tmp_lock, [&] { UnlockRunFlg(); });

  GE_CHK_BOOL_RET_STATUS(!run_flg_, INTERNAL_ERROR, "Model already started.");

  run_flg_ = true;

  // create stream instance which rt_model_handel is running on
  GE_CHK_RT_RET(rtStreamCreate(&rt_model_stream_, priority_));
  is_inner_model_stream_ = true;

  string opt = "0";
  (void)ge::GetContext().GetOption(OPTION_GE_MAX_DUMP_OP_NUM, opt);  // option may not be set up, no need to check value
  int64_t maxDumpOpNum = std::strtol(opt.c_str(), nullptr, kDecimal);
  maxDumpOpNum_ = maxDumpOpNum;

  CREATE_STD_THREAD(thread_id_, DavinciModel::Run, this);
  GELOGI("model tread create success, model id:%u.", model_id_);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief call API provided by data inputer and destroy model Thread
/// @param [in] no
/// @return Status Destroy result
/// @author
///
Status DavinciModel::ModelRunStop() {
  LockRunFlg();
  GE_MAKE_GUARD(tmp_lock, [&] { UnlockRunFlg(); });

  GE_IF_BOOL_EXEC(!run_flg_, return SUCCESS);

  GE_CHK_STATUS_RET(DestroyThread(), "DestoyThead failed.");

  return SUCCESS;
}

void DavinciModel::UnbindTaskSinkStream() {
  // unbinding hcom stream
  UnbindHcomStream();
  if (is_stream_list_bind_) {
    for (size_t i = 0; i < stream_list_.size(); i++) {
      // unbind rt_model_handle and streams
      GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, stream_list_[i]) != RT_ERROR_NONE,
                 "Unbind stream from model failed! Index: %zu", i);
    }
  }

  if (is_inner_model_stream_) {
    if (!input_queue_ids_.empty() || !output_queue_ids_.empty()) {
      GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, rt_model_stream_) != RT_ERROR_NONE, "Unbind stream failed!");
    }
    // destroy stream that is bound with rt_model
    GE_LOGW_IF(rtStreamDestroy(rt_model_stream_) != RT_ERROR_NONE, "Destroy stream for rt_model failed.")
  }

  if (is_pure_head_stream_ && rt_head_stream_ != nullptr) {
    GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, rt_head_stream_) != RT_ERROR_NONE, "Unbind stream failed!");
    GE_LOGW_IF(rtStreamDestroy(rt_head_stream_) != RT_ERROR_NONE, "Destroy stream for rt_model failed.");
    rt_head_stream_ = nullptr;
  }

  if (rt_entry_stream_ != nullptr) {
    GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, rt_entry_stream_) != RT_ERROR_NONE, "Unbind stream failed!");
    GE_LOGW_IF(rtStreamDestroy(rt_entry_stream_) != RT_ERROR_NONE, "Destroy stream for rt_model failed.");
    rt_entry_stream_ = nullptr;
  }
}

Status DavinciModel::CreateKnownZeroCopyMap(const vector<void *> &inputs, const vector<void *> &outputs) {
  GELOGI("DavinciModel::CreateKnownZeroCopyMap in.");
  if (inputs.size() > data_op_list_.size()) {
    GELOGE(FAILED, "input data addr %u should less than input op number %u.", inputs.size(), data_op_list_.size());
    return FAILED;
  }
  // remove zero copy addr in last iteration
  knonw_input_data_info_.clear();
  knonw_output_data_info_.clear();
  for (size_t i = 0; i < inputs.size(); ++i) {
    const vector<void *> addr_list = ModelUtils::GetOutputDataAddrs(runtime_param_, data_op_list_[i]);
    knonw_input_data_info_[addr_list[kDataIndex]] = inputs[i];
    GELOGI("DavinciModel::CreateKnownZeroCopyMap input %d,v addr %p,p addr %p .", i, addr_list[kDataIndex], inputs[i]);
  }
  if (output_op_list_.size() < kOutputNum) {
    GELOGW("output op num in graph is %u.", output_op_list_.size());
    return SUCCESS;
  }
  const vector<void *> addr_list = ModelUtils::GetInputDataAddrs(runtime_param_, output_op_list_[kDataIndex]);
  for (size_t i = 0; i < addr_list.size() && i < outputs.size(); ++i) {
    knonw_output_data_info_[addr_list[i]] = outputs[i];
    GELOGI("DavinciModel::CreateKnownZeroCopyMap output %d,v addr %p,p addr %p .", i, addr_list[i], outputs[i]);
  }
  GELOGI("DavinciModel::CreateKnownZeroCopyMap success.");
  return SUCCESS;
}

Status DavinciModel::UpdateKnownZeroCopyAddr() {
  for (size_t i = 0; i < total_io_addrs_.size(); ++i) {
    auto it_in = knonw_input_data_info_.find(total_io_addrs_[i]);
    if (it_in != knonw_input_data_info_.end()) {
      GELOGI("DavinciModel::UpdateKnownZeroCopyAddr input %d,v addr %p,p addr %p .", i, total_io_addrs_[i],
             knonw_input_data_info_.at(total_io_addrs_[i]));
      total_io_addrs_[i] = knonw_input_data_info_.at(total_io_addrs_[i]);
    }
    auto it_out = knonw_output_data_info_.find(total_io_addrs_[i]);
    if (it_out != knonw_output_data_info_.end()) {
      GELOGI("DavinciModel::UpdateKnownZeroCopyAddr output %d,v addr %p,p addr %p .", i, total_io_addrs_[i],
             knonw_output_data_info_.at(total_io_addrs_[i]));
      total_io_addrs_[i] = knonw_output_data_info_.at(total_io_addrs_[i]);
    }
  }
  GELOGI("DavinciModel::UpdateKnownZeroCopyAddr success.");
  return SUCCESS;
}

Status DavinciModel::UpdateKnownNodeArgs(const vector<void *> &inputs, const vector<void *> &outputs) {
  GELOGI("DavinciModel::UpdateKnownNodeArgs in");
  GE_CHK_STATUS_RET(CreateKnownZeroCopyMap(inputs, outputs),
                    "DavinciModel::UpdateKnownNodeArgs create map for input/output zero copy.");
  if (!base_addr_not_changed_) {
    total_io_addrs_.clear();
    orig_total_io_addrs_.clear();
    for (size_t task_index = 0; task_index < task_list_.size(); ++task_index) {
      auto &task = task_list_[task_index];
      if (task != nullptr) {
        Status ret = task->UpdateArgs();
        if (ret != SUCCESS) {
          GELOGE(FAILED, "task %d created by davinci model is nullptr.", task_index);
          return FAILED;
        }
      }
    }
    // cache latest iterator io addr
    orig_total_io_addrs_ = total_io_addrs_;
  } else {
    total_io_addrs_ = orig_total_io_addrs_;
  }
  GE_CHK_STATUS_RET(UpdateKnownZeroCopyAddr(), "DavinciModel::UpdateKnownZeroCopyAddr failed.");

  if (total_args_size_ == 0) {
    GELOGW("DavinciModel::UpdateKnownNodeArgs device args %p, dst size %u, pass rtMemcpy.", args_, total_args_size_);
  } else {
    uint32_t total_addr_size = total_io_addrs_.size() * sizeof(uint64_t);
    GELOGI("DavinciModel::UpdateKnownNodeArgs device args %p, dst size %u, src size %u", args_, total_args_size_,
           total_addr_size);

    Status rt_ret =
        rtMemcpy(args_, total_args_size_, total_io_addrs_.data(), total_addr_size, RT_MEMCPY_HOST_TO_DEVICE);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(rt_ret, "rtMemcpy error, ret: Ox%X", rt_ret); return FAILED;)
  }

  GELOGI("DavinciModel::UpdateKnownNodeArgs success");
  return SUCCESS;
}

Status DavinciModel::InitTaskInfo(domi::ModelTaskDef &model_task_def) {
  GELOGI("InitTaskInfo in, task size %zu", model_task_def.task().size());
  task_list_.resize(model_task_def.task_size());
  for (int i = 0; i < model_task_def.task_size(); ++i) {
    // dynamic shape will create task_list_ before
    const domi::TaskDef &task = model_task_def.task(i);
    if (this->task_list_[i] == nullptr) {
      task_list_[i] = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task.type()));
    }
    GE_CHECK_NOTNULL(task_list_[i]);
    Status ret = task_list_[i]->Init(task, this);
    if (ret != SUCCESS) {
      GELOGE(ret, "Task index %d init failed.", i);
      return ret;
    }
  }
  GELOGI("InitTaskInfo out");
  return SUCCESS;
}

Status DavinciModel::MallocKnownArgs() {
  GELOGI("DavinciModel::MallocKnownArgs in");
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  if (model_task_def->task_size() == 0) {
    GELOGW("DavinciModel::MallocKnownArgs davincimodel has no task info.");
    return SUCCESS;
  }
  task_list_.resize(model_task_def->task_size());
  for (int32_t i = 0; i < model_task_def->task_size(); ++i) {
    const domi::TaskDef &taskdef = model_task_def->task(i);
    task_list_[i] = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(taskdef.type()));
    GE_CHECK_NOTNULL(task_list_[i]);
    Status ret = task_list_[i]->CalculateArgs(taskdef, this);
    if (ret != SUCCESS) {
      GELOGE(ret, "TaskInfo CalculateArgs failed.");
      return ret;
    }
  }
  // malloc args memory
  if (total_args_size_ == 0) {
    GELOGW("DavinciModel::MallocKnownArgs total_args_size_ equals to zero.");
    return SUCCESS;
  }

  rtError_t rt_ret = rtMalloc(&args_, total_args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtMalloc failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  // malloc fixed addr memory, eg: rts op
  if (total_fixed_addr_size_ != 0) {
    GELOGI("Begin to allocate fixed addr.");
    rt_ret = rtMalloc(&fixed_addrs_, total_fixed_addr_size_, RT_MEMORY_HBM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rtMalloc failed, ret: 0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }

  GELOGI("DavinciModel::MallocKnownArgs success, total args size %u. total fixed addr size %ld", total_args_size_,
         total_fixed_addr_size_);
  return SUCCESS;
}

Status DavinciModel::DistributeTask() {
  GELOGI("do Distribute.");
  for (auto &task : cpu_task_list_) {
    if (task == nullptr) {
      GELOGW("task is null");
      continue;
    }
    GE_CHK_STATUS_RET(task->Distribute());
  }

  task_desc_info_.clear();
  bool flag = GetL1FusionEnableOption();
  char *skt_enable_env = std::getenv("SKT_ENABLE");
  int64_t env_flag = (skt_enable_env != nullptr) ? std::strtol(skt_enable_env, nullptr, kDecimal) : 0;
  if (env_flag != 0) {
    flag = true;
  }

  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  GELOGI("there are %zu task need to save.", task_list_.size());
  for (size_t task_index = 0; task_index < task_list_.size(); ++task_index) {
    auto &task = task_list_.at(task_index);
    GE_CHK_STATUS_RET(task->Distribute(), "Task[%zu] distribute fail", task_index);
    // for data dump
    auto op_index = std::max(model_task_def->task(task_index).kernel().context().op_index(),
                             model_task_def->task(task_index).kernel_ex().op_index());
    OpDescPtr op = GetOpByIndex(op_index);
    GE_CHECK_NOTNULL(op);

    SaveDumpOpInfo(runtime_param_, op, task->GetTaskID(), task->GetStreamId());
    if (reinterpret_cast<void *>(task->GetDumpArgs()) != nullptr) {
      bool call_dump = GetDumpProperties().IsLayerNeedDump(name_, om_name_, op->GetName()) && task->CallSaveDumpInfo();
      if (call_dump || is_op_debug_reg_) {
        SaveDumpTask(task->GetTaskID(), task->GetStreamId(), op, task->GetDumpArgs());
      }
    }
    // Load task info for profiling
    TaskDescInfo task_desc_info;
    if (!om_name_.empty()) {
      task_desc_info.model_name = om_name_;
    } else {
      task_desc_info.model_name = name_;
    }
    task_desc_info.op_name = op->GetName();
    task_desc_info.block_dim = model_task_def->task(task_index).kernel().block_dim();
    task_desc_info.task_id = task->GetTaskID();
    task_desc_info.stream_id = task->GetStreamId();
    task_desc_info_.emplace_back(task_desc_info);
    if (flag) {
      if (task->GetSktTaskID() != 0xFFFFFFFF) {
        TaskDescInfo task_desc_info;
        string op_name = "super_kernel_" + to_string(task_index);
        task_desc_info.op_name = op_name;
        task_desc_info.task_id = task->GetSktTaskID();
        task_desc_info_.emplace_back(task_desc_info);
      }
    }
  }
  // launch dump kernel to aicpu
  GE_CHK_STATUS_RET(data_dumper_.LoadDumpInfo(), "Load dump info failed.");
  return SUCCESS;
}

void DavinciModel::SetEndGraphId(uint32_t task_id, uint32_t stream_id) {
  auto all_dump_model = GetDumpProperties().GetAllDumpModel();
  bool findByOmName = all_dump_model.find(om_name_) != all_dump_model.end();
  bool findByModelName = all_dump_model.find(name_) != all_dump_model.end();
  if (all_dump_model.find(ge::DUMP_ALL_MODEL) != all_dump_model.end() || findByOmName || findByModelName) {
    GELOGI("start save end_graph_info to dumper, task_id is %u, stream_id is %u", task_id, stream_id);
    data_dumper_.SaveEndGraphId(task_id, stream_id);
  }
}

///
/// @ingroup ge
/// @brief Set copy only for No task feed NetOutput address.
/// @return None.
///
void DavinciModel::SetCopyOnlyOutput() {
  for (const auto &output_outside_addrs : new_output_outside_addrs_) {
    ZeroCopyOffset output_outside = output_outside_addrs.second;
    for (uint32_t out_count = 0; out_count < output_outside.GetAddrCount(); ++out_count) {
      auto &addrs_mapping_list = output_outside.GetOutsideAddrs();
      std::map<const void *, std::vector<void *>> virtual_args_addrs = addrs_mapping_list[out_count];
      for (const auto &virtual_args_addr : virtual_args_addrs) {
        const auto &args_addrs = virtual_args_addr.second;
        if (args_addrs.empty()) {  // No task feed Output addr, Need copy directly.
          GELOGI("[ZCPY] just copy %p to netoutput.", virtual_args_addr.first);
          copy_only_addrs_.insert(virtual_args_addr.first);
        }
      }
    }
  }
}

///
/// @ingroup ge
/// @brief Set disabled input zero copy addr.
/// @param [in] const void *addr: address of task
/// @return None.
///
void DavinciModel::DisableZeroCopy(const void *addr) {
  if (real_virtual_addrs_.find(addr) == real_virtual_addrs_.end()) {
    return;
  }

  // Data link to RTS Op directly.
  std::lock_guard<std::mutex> lock(outside_addrs_mutex_);
  GELOGI("[ZCPY] disable zero copy of %p.", addr);
  copy_only_addrs_.insert(addr);
}

///
/// @ingroup ge
/// @brief Save outside address used info for ZeroCopy.
/// @param [in] const OpDescPtr &op_desc: current op desc
/// @param [in] const std::vector<void *> &outside_addrs: address of task
/// @param [in] const void *info: task args
/// @param [in] const char *args: task args
/// @param [in] size_t size: size of task args
/// @param [in] size_t offset: offset of task args
/// @return None.
///
void DavinciModel::SetZeroCopyAddr(const OpDescPtr &op_desc, const std::vector<void *> &outside_addrs, const void *info,
                                   void *args, size_t size, size_t offset) {
  // Internal call has ensured that op_desc is not nullptr
  GELOGD("[ZCPY] SetZeroCopyAddr for %s.", op_desc->GetName().c_str());
  size_t nums = outside_addrs.size();
  ZeroCopyTask zero_copy_task(op_desc->GetName(), static_cast<uint8_t *>(args), size);
  for (size_t i = 0; i < nums; ++i) {
    std::lock_guard<std::mutex> lock(outside_addrs_mutex_);

    for (auto &input_outside_addrs : new_input_outside_addrs_) {
      ZeroCopyOffset &input_outside = input_outside_addrs.second;
      bool ret = input_outside.SetOutsideAddrsValue(zero_copy_task, outside_addrs[i], args, offset + i * kAddrLen);
      if (ret) {
        void *args_val = static_cast<uint8_t *>(args) + offset + i * kAddrLen;
        SetBatchLabelAddr(op_desc, reinterpret_cast<uintptr_t>(args_val));
      }
    }

    for (auto &output_outside_addrs : new_output_outside_addrs_) {
      ZeroCopyOffset &output_outside = output_outside_addrs.second;
      bool ret = output_outside.SetOutsideAddrsValue(zero_copy_task, outside_addrs[i], args, offset + i * kAddrLen);
      if (ret) {
        void *args_val = static_cast<uint8_t *>(args) + offset + i * kAddrLen;
        SetBatchLabelAddr(op_desc, reinterpret_cast<uintptr_t>(args_val));
      }
    }
  }
  auto it = zero_copy_op_id_batch_label_.find(op_desc->GetId());
  if (it == zero_copy_op_id_batch_label_.end()) {
    zero_copy_task.SetBatchLabel(kDefaultBatchLable);
  } else {
    zero_copy_task.SetBatchLabel(it->second);
  }

  std::lock_guard<std::mutex> lock(outside_addrs_mutex_);
  if (zero_copy_task.IsTaskArgsSet()) {
    zero_copy_task.SetOriginalArgs(info, offset + nums * kAddrLen);
    zero_copy_tasks_.emplace_back(zero_copy_task);
  }
}

void DavinciModel::SetBatchLabelAddr(const OpDescPtr &op_desc, uintptr_t addr) {
  // Establish a mapping between batch label and zero copy address for multi-batch scenes
  auto it = zero_copy_op_id_batch_label_.find(op_desc->GetId());
  if (it == zero_copy_op_id_batch_label_.end()) {
    return;
  }

  const string &batch_label = it->second;
  auto iter = zero_copy_batch_label_addrs_.find(batch_label);
  if (iter != zero_copy_batch_label_addrs_.end()) {
    iter->second.insert(addr);
    GELOGD("[ZCPY] Set zero copy batch label and addrs success, batch label: %s, op name:%s.", batch_label.c_str(),
           op_desc->GetName().c_str());
  } else {
    set<uintptr_t> addrs = {addr};
    zero_copy_batch_label_addrs_.emplace(pair<string, set<uintptr_t>>(batch_label, addrs));
    GELOGD("[ZCPY] New added zero copy batch label and addrs success, batch label: %s, op name:%s.",
           batch_label.c_str(), op_desc->GetName().c_str());
  }
}

///
/// @ingroup ge
/// @brief Copy Check input size and model op size.
/// @param [in] const int64_t &input_size: input size.
/// @param [in] const int64_t &op_size: model op size.
/// @param [in] is_dynamic: dynamic batch input flag.
/// @return true if success
///
bool DavinciModel::CheckInputAndModelSize(const int64_t &input_size, const int64_t &op_size, bool is_dynamic) {
  if (is_dynamic) {  // dynamic is max size.
    GELOGI("No need to check input and model size.");
    return true;
  }

  if (input_size > op_size) {
    GELOGW(
        "Input size [%u] is bigger than om size need [%u], "
        "MAY cause inference result ERROR, please check model input",
        input_size, op_size);
  }
  bool is_dynamic_aipp = false;
  for (const auto &op_desc : data_op_list_) {
    if (op_desc->GetType() == AIPP_DATA_TYPE) {
      GELOGI("This is dynamic aipp model.");
      is_dynamic_aipp = true;
      break;
    }
  }
  if (is_dynamic_aipp) {
    GELOGI("This is dynamic aipp model, no need to judge smaller input size");
    return true;
  }
  // Judge overflow first
  if (input_size > (INT64_MAX - kDataMemAlignSizeCompare)) {
    GELOGI("The Input size [%ld] is smaller than model size [%ld] and is in the range of 64 bytes", input_size,
           op_size);
    return true;
  }
  // The input and model input size can not be exactly equal because user input is not definite.
  if ((input_size + kDataMemAlignSizeCompare) < op_size) {
    GELOGE(FAILED, "Input size [%ld] can not be smaller than op size [%ld] after 64-byte alignment", input_size,
           op_size);
    return false;
  }
  return true;
}

///
/// @ingroup ge
/// @brief Copy Inputs and Outputs addr to model for direct use.
/// @param [in] const InputData &input_data: model input data.
/// @param [in] OutputData &output_data: model output data.
/// @param [in] bool is_dynamic_input: whether is dynamic input, true: is dynamic input; false: not is dynamic input
/// @return SUCCESS handle successfully / PARAM_INVALID for failed
///
Status DavinciModel::CopyModelData(const InputData &input_data, OutputData &output_data, bool is_dynamic) {
  if (UpdateIoTaskArgs(new_input_data_info_, true, input_data.blobs, is_dynamic, input_data.batch_label) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[ZCPY] Update input data to model failed.");
    return PARAM_INVALID;
  }

  if (UpdateIoTaskArgs(new_output_data_info_, false, output_data.blobs, is_dynamic, input_data.batch_label) !=
      SUCCESS) {
    GELOGE(PARAM_INVALID, "[ZCPY] Update output data to model failed.");
    return PARAM_INVALID;
  }

  for (ZeroCopyTask &task : zero_copy_tasks_) {
    GE_CHK_STATUS_RET(task.DistributeParam(is_async_mode_, rt_model_stream_), "[ZCPY] Update args failed.");
  }

  output_data.index = input_data.index;
  output_data.model_id = model_id_;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Copy Data addr to model for direct use.
/// @param [in] data_info: model memory addr/size map { data_index, { tensor_size, tensor_addr } }.
/// @param [in] is_input: input data or output data
/// @param [in] blobs: user input/output data list.
/// @param [in] is_dynamic: whether is dynamic input, true: is dynamic input; false: not is dynamic input
/// @param [in] batch_label: batch label for multi-batch scenes
/// @return SUCCESS handle successfully / others handle failed
///
Status DavinciModel::UpdateIoTaskArgs(const std::map<uint32_t, ZeroCopyOffset> &data_info, bool is_input,
                                      const vector<DataBuffer> &blobs, bool is_dynamic, const string &batch_label) {
  string input_or_output = "input";
  is_input ? input_or_output = "input" : input_or_output = "output";
  if (blobs.size() != data_info.size()) {
    GELOGE(FAILED, "Verify %s data num failed: model requires %zu, but user actually feeds %zu",
           input_or_output.c_str(), data_info.size(), blobs.size());
    return FAILED;
  }

  for (const auto &data : data_info) {
    if (data.first >= blobs.size()) {  // check data index.
      GELOGE(FAILED, "Verify %s data num failed: can not find No.%zu data, because user only feeds %zu",
             input_or_output.c_str(), data.first, blobs.size());
      return FAILED;
    }

    const DataBuffer &buffer = blobs[data.first];  // index of data.
    if (buffer.data == nullptr) {
      GELOGE(FAILED, "data_buf.data is nullptr, index=%u", data.first);
      return FAILED;
    }

    if (!CheckInputAndModelSize(buffer.length, data.second.GetDataSize(), is_dynamic)) {
      GELOGE(FAILED, "Check input size and model size failed");
      return FAILED;
    }

    void *basic_addr = data.second.GetBasicAddr();
    uint64_t data_size = data.second.GetDataSize();
    if (copy_only_addrs_.count(basic_addr) > 0) {
      if (is_input) {
        GELOGI("[IMAS] Find addr %p need direct copy from user malloc input %p", basic_addr, buffer.data);
        if (rtMemcpy(basic_addr, data_size, buffer.data, buffer.length, RT_MEMCPY_DEVICE_TO_DEVICE) != RT_ERROR_NONE) {
          GELOGE(FAILED, "Non-zero copy data node copy failed");
          return FAILED;
        }
      }
      GELOGI("No need to exeucte zero copy task because this addr %p need direct copy.", basic_addr);
      continue;
    }

    for (size_t count = 0; count < data.second.GetDataCount(); ++count) {
      int64_t size = data.second.GetDataInfo().at(count).first;
      void *addr = data.second.GetDataInfo().at(count).second;
      void *buffer_addr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(buffer.data) +
                                                   data.second.GetRelativeOffset().at(count));
      GELOGI("[ZCPY] Copy %s blobs_index %u, virtual_addr: %p, size: %ld, user_data_addr: %p", input_or_output.c_str(),
             data.first, addr, size, buffer_addr);
      // For input data, just copy for rts task.
      for (ZeroCopyTask &task : zero_copy_tasks_) {
        if (task.GetBatchLabel() != kDefaultBatchLable && task.GetBatchLabel() != batch_label) {
          continue;
        }
        uintptr_t addr_val = reinterpret_cast<uintptr_t>(addr);
        if (task.UpdateTaskParam(addr_val, buffer_addr, zero_copy_batch_label_addrs_, batch_label) != SUCCESS) {
          return FAILED;
        }
      }
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief get unique identification for op when load two or more models
/// @param [in] const OpDescPtr: current op.
/// @param [in] string identification: unique identification for current op.
/// @return SUCCESS handle successfully / others handle failed
///
void DavinciModel::GetUniqueId(const OpDescPtr &op_desc, std::string &unique_identification) {
  std::string session_graph_id;
  GE_IF_BOOL_EXEC(AttrUtils::GetStr(*op_desc, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                  GELOGD("Get original type of session_graph_id."));
  if (session_graph_id.empty()) {
    return;
  } else if (session_graph_id.find("-1") != string::npos) {
    unique_identification = session_graph_id + "_" + to_string(model_id_);
  } else {
    unique_identification = session_graph_id;
  }
}

///
/// @ingroup ge
/// @brief For TVM Op, avoid Addr Reuse.
/// @return void*
///
const char *DavinciModel::GetRegisterStub(const string &binfile, const string &session_graph_id) {
  string binfile_key;
  if (session_graph_id.empty()) {
    binfile_key = binfile;
  } else {
    binfile_key = session_graph_id + "_" + binfile;
  }
  auto it = tvm_bin_kernel_.find(binfile_key);
  if (it != tvm_bin_kernel_.end()) {
    return it->c_str();
  } else {
    it = tvm_bin_kernel_.insert(tvm_bin_kernel_.end(), binfile_key);
    return it->c_str();
  }
}

///
/// @ingroup ge
/// @brief Constant Op Init.
/// @return Status
///
Status DavinciModel::InitConstant(const OpDescPtr &op_desc) {
  auto v_weights = ModelUtils::GetWeights(op_desc);
  auto v_output_size = ModelUtils::GetOutputSize(op_desc);
  auto v_output_addr = ModelUtils::GetOutputDataAddrs(runtime_param_, op_desc);
  GE_IF_BOOL_EXEC(v_weights.empty() || v_output_size.empty() || v_output_addr.empty(),
                  GELOGE(PARAM_INVALID, "const op:%s not set output", op_desc->GetName().c_str());
                  return PARAM_INVALID;);

  GeTensor *tensor = const_cast<GeTensor *>(v_weights[0].get());
  GE_IF_BOOL_EXEC(static_cast<size_t>(v_output_size[0]) < tensor->GetData().size(),
                  GELOGE(PARAM_INVALID, "output size:%u less than weight data size:%zu", v_output_size[0],
                         tensor->GetData().size());
                  return PARAM_INVALID;);

  GE_IF_BOOL_EXEC(tensor->GetData().size() == 0, GELOGW("const op:%s has no weight data.", op_desc->GetName().c_str());
                  return SUCCESS;);

  auto desc = tensor->GetTensorDesc();
  if (desc.GetDataType() == DT_STRING) {
    GeShape tensor_shape = desc.GetShape();
    /// if tensor is a scaler, it's shape size if zero, according ge_tensor.cc.
    /// the logic of GetShapeSize is wrong, the scaler tensor's GetShapeSize is zero
    /// and that of unknown shape is zero too.
    /// unknown shape will not appear here, so we can use zero judge a tensor is scaler or not
    int64_t elem_num = tensor_shape.GetShapeSize();
    if (elem_num == 0 && tensor_shape.GetDims().size() == 0) {
      elem_num = 1;
    }
    uint64_t *buff = reinterpret_cast<uint64_t *>(tensor->MutableData().data());
    GE_CHK_BOOL_RET_STATUS(ge::CheckInt64Uint32MulOverflow(elem_num, kBytes) == SUCCESS, FAILED,
                           "Shape size is invalid");
    uint64_t offset = static_cast<uint64_t>(elem_num * kBytes);

    uint64_t hbm_raw_data_base_addr =
        reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(v_output_addr[0])) + offset;
    for (int64_t i = elem_num - 1; i >= 0; --i) {
      buff[i] = hbm_raw_data_base_addr + (buff[i] - buff[0]);
    }
  }
  GELOGI("[IMAS]InitConstant memcpy graph_%u type[V] name[%s] output[%d] memaddr[%p] mem_size[%u] datasize[%zu]",
         runtime_param_.graph_id, op_desc->GetName().c_str(), 0, v_output_addr[0], v_output_size[0],
         tensor->GetData().size());
  GE_CHK_RT_RET(rtMemcpy(v_output_addr[0], v_output_size[0], tensor->GetData().data(), tensor->GetData().size(),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief TVM Op Init.
/// @return Status
///
Status DavinciModel::InitTbeHandle(const OpDescPtr &op_desc) {
  auto kernel = ge_model_->GetTBEKernelStore().FindKernel(op_desc->GetName());
  auto tbe_kernel = (kernel != nullptr) ? kernel : op_desc->TryGetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
  if (tbe_kernel == nullptr) {
    GELOGE(INTERNAL_ERROR, "TBE: %s can't find tvm bin file!", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  std::string session_graph_model_id;
  GetUniqueId(op_desc, session_graph_model_id);
  const char *bin_file_key = GetRegisterStub(op_desc->GetName(), session_graph_model_id);  // from set, always valid.
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();

  std::lock_guard<std::mutex> lock(tvm_bin_mutex_);
  if (rtQueryFunctionRegistered(bin_file_key) != RT_ERROR_NONE) {
    void *bin_handle = nullptr;
    if (!kernel_store.FindTBEHandle(bin_file_key, bin_handle)) {
      GELOGI("TBE: can't find the kernel_name[%s] in HandleMap", bin_file_key);

      rtDevBinary_t binary;
      std::string json_string;
      GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc, TVM_ATTR_NAME_MAGIC, json_string),
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
      GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc, TVM_ATTR_NAME_METADATA, meta_data),
                      GELOGI("Get original type of json_string"));
      GELOGI("TBE: meta data: %s", meta_data.empty() ? "null" : meta_data.c_str());
      GE_IF_BOOL_EXEC(!meta_data.empty(), GE_CHK_RT_RET(rtMetadataRegister(bin_handle, meta_data.c_str())));

      kernel_store.StoreTBEHandle(bin_file_key, bin_handle, tbe_kernel);
    } else {
      GELOGI("TBE: find the kernel_name[%s] in HandleMap", bin_file_key);
      kernel_store.ReferTBEHandle(bin_file_key);
    }

    std::string kernel_name;
    GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name),
                    GELOGI("Get original type of kernel_name"));
    GELOGI("TBE: binfile_key=%s, kernel_name=%s", bin_file_key, kernel_name.c_str());
    GE_CHK_RT_RET(rtFunctionRegister(bin_handle, bin_file_key, bin_file_key, kernel_name.c_str(), 0));
    used_tbe_handle_map_[bin_file_key] = 1;  // Init used num to 1.
    return SUCCESS;
  }

  // Kernel registed, Increase used num in store.
  StoreTbeHandle(bin_file_key);
  return SUCCESS;
}

void DavinciModel::StoreTbeHandle(const std::string &handle_key) {
  // Online mode FE may call rtFunctionRegister.
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();

  auto it = used_tbe_handle_map_.find(handle_key);
  if (it != used_tbe_handle_map_.end()) {
    // GE registered, increase reference.
    kernel_store.ReferTBEHandle(handle_key);
    it->second++;
    return;
  }

  void *bin_handle = nullptr;
  if (kernel_store.FindTBEHandle(handle_key, bin_handle)) {
    // GE registered, increase reference.
    used_tbe_handle_map_[handle_key] = 1;  // Init used num to 1.
    kernel_store.ReferTBEHandle(handle_key);
  }
}

void DavinciModel::CleanTbeHandle() {
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();

  kernel_store.EraseTBEHandle(used_tbe_handle_map_);
  used_tbe_handle_map_.clear();
  tvm_bin_kernel_.clear();
}

///
/// @ingroup ge
/// @brief insert active_stream_indication_
/// @return Status
///
Status DavinciModel::InitStreamActive(const OpDescPtr &op_desc) {
  if (op_desc->HasAttr(ATTR_NAME_SWITCH_BRANCH_NODE_LABEL)) {
    std::vector<uint32_t> active_stream_list;
    GE_CHK_BOOL_EXEC(AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list),
                     return INTERNAL_ERROR, "StreamActiveOp get attr ACTIVE_STREAM failed.");

    for (size_t j = 0; j < active_stream_list.size(); ++j) {
      active_stream_indication_.insert(active_stream_list[j]);
      GELOGI("flowctrl_op_index_map  node:%s, active_stream_id=%u.", op_desc->GetName().c_str(), active_stream_list[j]);
    }
  }

  return SUCCESS;
}

Status DavinciModel::InitStreamSwitch(const OpDescPtr &op_desc) {
  std::vector<uint32_t> active_stream_list;
  GE_LOGI_IF(!ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list),
             "GetInt ACTIVE_STREAM_LIST failed.");
  if (active_stream_list.size() != kTrueBranchStreamNum) {
    GELOGE(INTERNAL_ERROR, "Stream num of switch true branch must be %u.", kTrueBranchStreamNum);
    return INTERNAL_ERROR;
  }

  uint32_t true_stream_id = active_stream_list.front();
  active_stream_indication_.insert(true_stream_id);
  GELOGI("flowctrl_op_index_map  node:%s, true_stream_id=%u.", op_desc->GetName().c_str(), true_stream_id);

  return SUCCESS;
}

Status DavinciModel::InitStreamSwitchN(const OpDescPtr &op_desc) {
  std::vector<uint32_t> active_stream_list;
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list)) {
    GELOGE(INTERNAL_ERROR, "StreamSwitchNOp get attr ACTIVE_STREAM failed.");
    return INTERNAL_ERROR;
  }

  for (size_t j = 0; j < active_stream_list.size(); ++j) {
    active_stream_indication_.insert(active_stream_list[j]);
    GELOGI("StreamSwitchNOp node:%s, active_stream_id=%u.", op_desc->GetName().c_str(), active_stream_list[j]);
  }

  uint32_t batch_num = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGE(FAILED, "Failed to get attr ATTR_NAME_BATCH_NUM, StreamSwitchN: %s.", op_desc->GetName().c_str());
    return FAILED;
  }

  return SetDynamicBatchInfo(op_desc, batch_num);
}

Status DavinciModel::SetDynamicBatchInfo(const OpDescPtr &op_desc, uint32_t batch_num) {
  batch_info_.clear();
  combined_batch_info_.clear();

  (void)AttrUtils::GetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type_);
  (void)AttrUtils::GetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, user_designate_shape_order_);
  for (uint32_t i = 0; i < batch_num; ++i) {
    std::vector<int64_t> batch_shape;
    const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    if (!AttrUtils::GetListInt(op_desc, attr_name, batch_shape)) {
      GELOGE(FAILED, "Get attr ATTR_NAME_PRED_VALUE failed, Node: %s", op_desc->GetName().c_str());
      batch_info_.clear();
      return FAILED;
    }
    batch_info_.emplace_back(batch_shape);
    batch_shape.clear();
    const string attr_combined_batch = ATTR_NAME_COMBINED_BATCH + "_" + std::to_string(i);
    if (AttrUtils::GetListInt(op_desc, attr_combined_batch, batch_shape)) {
      combined_batch_info_.emplace_back(batch_shape);
    }
  }

  return SUCCESS;
}

Status DavinciModel::InitCase(const OpDescPtr &op_desc) {
  uint32_t batch_num = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGI("Not multi-batch Node: %s", op_desc->GetName().c_str());
    return SUCCESS;
  }

  return SetDynamicBatchInfo(op_desc, batch_num);
}

bool DavinciModel::IsBroadCastOpData(const ge::NodePtr &var_node) {
  for (auto out_anchor : var_node->GetAllOutDataAnchors()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_anchor);
    for (auto in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
      ge::NodePtr dst_node = in_anchor->GetOwnerNode();
      GE_RT_FALSE_CHECK_NOTNULL(dst_node);
      if (dst_node->GetType() == HCOMBROADCAST || dst_node->GetType() == HVDCALLBACKBROADCAST) {
        return true;
      }
    }
  }
  return false;
}

///
/// @ingroup ge
/// @brief Init model stream for NN model.
/// @param [in] stream   user input model stream.
/// @return Status
///
Status DavinciModel::InitModelStream(rtStream_t stream) {
  ExecuteMode curr_mode = is_async_mode_ ? ASYNCHRONIZATION : SYNCHRONIZATION;
  GE_CHK_BOOL_RET_STATUS((curr_mode == last_execute_mode_) || (last_execute_mode_ == INITIALIZATION), INTERNAL_ERROR,
                         "NnExecute not support mix execute.");
  last_execute_mode_ = curr_mode;

  // asynchronize mode, use user input stream.
  if (is_async_mode_) {
    rt_model_stream_ = stream;
    is_inner_model_stream_ = false;
    return SUCCESS;
  }

  // synchronize mode, use forbidden stream.
  if (stream != nullptr) {
    if ((rt_model_stream_ != nullptr) && is_inner_model_stream_) {
      GE_LOGW_IF(rtStreamDestroy(rt_model_stream_) != RT_ERROR_NONE, "Destroy rt_stream failed!");
    }

    rt_model_stream_ = stream;
    is_inner_model_stream_ = false;
    return SUCCESS;
  }

  if (rt_model_stream_ == nullptr) {
    GE_CHK_RT_RET(rtStreamCreateWithFlags(&rt_model_stream_, priority_, RT_STREAM_FORBIDDEN_DEFAULT));
    is_inner_model_stream_ = true;
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief ACL case, do not start  new thread, return execute result.
/// @param [in] stream   execute model stream.
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  model input data.
/// @param [out] output_data  model output data.
///
Status DavinciModel::NnExecute(rtStream_t stream, bool async_mode, const InputData &input_data,
                               OutputData &output_data) {
  is_async_mode_ = async_mode;
  GELOGI("Model Run begin, model id:%u, data index:%u, flag:%d.", model_id_, input_data.index, is_async_mode_);
  GE_CHK_STATUS_RET(InitModelStream(stream), "Init model stream failed.");
  is_dynamic_ = input_data.is_dynamic_batch;
  if (!is_dynamic_) {
    zero_copy_batch_label_addrs_.clear();
  }

  GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), SetProfileTime(MODEL_PRE_PROC_START));
  Status ret = CopyModelData(input_data, output_data, is_dynamic_);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ret, "Copy input data to model failed. model id: %u",
                                 model_id_);

  GELOGI("current_data.index=%u", input_data.index);
  GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), SetProfileTime(MODEL_PRE_PROC_END));

  if (!task_list_.empty()) {
    GELOGD("rtModelExecute do");
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), SetProfileTime(MODEL_INFER_START));
    rtError_t rt_ret = rtModelExecute(rt_model_handle_, rt_model_stream_, 0);
    GE_CHK_RT_EXEC(rt_ret, return RT_ERROR_TO_GE_STATUS(rt_ret));
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), SetProfileTime(MODEL_INFER_END));
    GELOGI("rtModelExecute end");
  }

  if (!is_async_mode_) {
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), SetProfileTime(MODEL_AFTER_PROC_START));
    ret = CopyOutputData(input_data.index, output_data, RT_MEMCPY_DEVICE_TO_DEVICE);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ret, "Copy Output data to user failed.");
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), SetProfileTime(MODEL_AFTER_PROC_END));
  }

  // report model time data
  GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), (void)SinkTimeProfile(input_data));
  GELOGI("Model run end, model id:%u", model_id_);
  return SUCCESS;
}

// Add active entry stream for special env.
Status DavinciModel::AddHeadStream() {
  if (active_stream_list_.empty()) {
    GELOGE(INTERNAL_ERROR, "Active stream is empty, stream list size: %zu, stream indication size: %zu.",
           stream_list_.size(), active_stream_indication_.size());
    return INTERNAL_ERROR;
  }

  if (active_stream_list_.size() == 1) {
    GELOGI("Just one active stream, take as head stream.");
    rt_head_stream_ = active_stream_list_[0];
    is_pure_head_stream_ = false;
  } else {
    // Create stream which rt_model_handel running on, this is S0, TS stream.
    GELOGI("Multiple active stream: %zu, create head stream.", active_stream_list_.size());
    GE_CHK_RT_RET(rtStreamCreateWithFlags(&rt_head_stream_, priority_, RT_STREAM_PERSISTENT));
    GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, rt_head_stream_, RT_INVALID_FLAG));  // Not active.
    is_pure_head_stream_ = true;

    for (auto s : active_stream_list_) {
      std::shared_ptr<CpuTaskActiveEntry> active_entry = MakeShared<CpuTaskActiveEntry>(rt_head_stream_);
      if (active_entry == nullptr) {
        GELOGE(MEMALLOC_FAILED, "Make CpuTaskActiveEntry task failed.");
        return MEMALLOC_FAILED;
      }

      Status status = active_entry->Init(s);
      if (status != SUCCESS) {
        return status;
      }

      cpu_task_list_.emplace_back(active_entry);
    }
  }

  // Create entry stream active head stream. AICPU stream.
  GE_CHK_RT_RET(rtStreamCreateWithFlags(&rt_entry_stream_, priority_, RT_STREAM_AICPU));
  GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, rt_entry_stream_, RT_HEAD_STREAM));
  return SUCCESS;
}

Status DavinciModel::InitEntryTask() {
  if (deploy_type_ == AICPU_DEPLOY_CROSS_THREAD) {
    GE_CHK_STATUS_RET(AddHeadStream(), "Add head stream failed.");
    return CpuActiveStream();
  } else {
    return LoadWithQueue();
  }
}

uint8_t *DavinciModel::MallocFeatureMapMem(size_t data_size) {
  uint8_t *mem_base = nullptr;
  const string purpose("feature map,used for op input and output.");
  if (std::getenv(kEnvGeuseStaticMemory) != nullptr) {
    data_size = static_cast<size_t>(VarManager::Instance(session_id_)->GetGraphMemoryMaxSize());
    string memory_key = std::to_string(0) + "_f";
    mem_base = MemManager::Instance(RT_MEMORY_HBM)->MallocMemory(purpose, memory_key, data_size, GetDeviceId());
  } else {
    mem_base = MemManager::Instance(RT_MEMORY_HBM)->MallocMemory(purpose, data_size, GetDeviceId());
  }

  if (mem_base != nullptr) {
    GE_CHK_RT(rtMemset(mem_base, data_size, 0U, data_size));
  }
  return mem_base;
}

uint8_t *DavinciModel::MallocP2PMem(size_t p2p_data_size) {
  uint8_t *p2p_mem_base = nullptr;
  const string purpose("p2p memory, used for some op related to hcom");
  if (std::getenv(kEnvGeuseStaticMemory) != nullptr) {
    string p2p_memory_key = std::to_string(0) + "_p";
    p2p_mem_base =
        MemManager::Instance(RT_MEMORY_P2P_DDR)->MallocMemory(purpose, p2p_memory_key, p2p_data_size, GetDeviceId());
  } else {
    p2p_mem_base = MemManager::Instance(RT_MEMORY_P2P_DDR)->MallocMemory(purpose, p2p_data_size, GetDeviceId());
  }
  return p2p_mem_base;
}

uint8_t *DavinciModel::MallocWeightsMem(size_t weights_size) {
  uint8_t *weights_mem_base = nullptr;
  const string purpose("weights memory in inference network.");
  if (std::getenv(kEnvGeuseStaticMemory) != nullptr) {
    string weight_memory_key = std::to_string(0) + "_w";
    weights_mem_base =
        MemManager::Instance(RT_MEMORY_HBM)->MallocMemory(purpose, weight_memory_key, weights_size, GetDeviceId());
  } else {
    weights_mem_base = MemManager::Instance(RT_MEMORY_HBM)->MallocMemory(purpose, weights_size, GetDeviceId());
  }
  return weights_mem_base;
}

void DavinciModel::FreeFeatureMapMem() {
  if (std::getenv(kEnvGeuseStaticMemory) != nullptr && is_inner_mem_base_) {
    string weight_memory_key = std::to_string(0) + "_f";
    if (MemManager::Instance(RT_MEMORY_HBM)->GetMemoryAddr(weight_memory_key) != nullptr) {
      GE_CHK_STATUS(MemManager::Instance(RT_MEMORY_HBM)->FreeMemory(weight_memory_key, GetDeviceId()),
                    "failed to free weight memory");
    }
    mem_base_ = nullptr;
  } else {
    GE_IF_BOOL_EXEC(mem_base_ != nullptr && is_inner_mem_base_,
                    GE_CHK_STATUS(MemManager::Instance(RT_MEMORY_HBM)->FreeMemory(mem_base_, GetDeviceId()),
                                  "failed to free feature_map memory");
                    mem_base_ = nullptr);
  }
}

void DavinciModel::FreeP2PMem() {
  if (std::getenv(kEnvGeuseStaticMemory) != nullptr) {
    std::string p2p_memory_key = std::to_string(0) + "_p";
    if (MemManager::Instance(RT_MEMORY_P2P_DDR)->GetMemoryAddr(p2p_memory_key) != nullptr) {
      GE_CHK_STATUS(MemManager::Instance(RT_MEMORY_P2P_DDR)->FreeMemory(p2p_memory_key, GetDeviceId()),
                    "failed to free p2p memory");
    }
    p2p_mem_base_ = nullptr;
  } else {
    GE_IF_BOOL_EXEC(p2p_mem_base_ != nullptr && is_inner_mem_base_,
                    GE_CHK_STATUS(MemManager::Instance(RT_MEMORY_P2P_DDR)->FreeMemory(p2p_mem_base_, GetDeviceId()),
                                  "failed to free p2p memory");
                    p2p_mem_base_ = nullptr);
  }
}

void DavinciModel::FreeWeightsMem() {
  if (std::getenv(kEnvGeuseStaticMemory) != nullptr) {
    string memory_key = std::to_string(0) + "_w";
    if (MemManager::Instance(RT_MEMORY_HBM)->GetMemoryAddr(memory_key) != nullptr) {
      GE_CHK_STATUS(MemManager::Instance(RT_MEMORY_HBM)->FreeMemory(memory_key, GetDeviceId()),
                    "failed to free feature_map memory");
    }
    weights_mem_base_ = nullptr;
  } else {
    GE_IF_BOOL_EXEC(weights_mem_base_ != nullptr && weights_mem_base_ != mem_base_ && is_inner_weight_base_,
                    GE_CHK_STATUS(MemManager::Instance(RT_MEMORY_HBM)->FreeMemory(weights_mem_base_, GetDeviceId()),
                                  "failed to free weight memory");
                    weights_mem_base_ = nullptr);
  }
}

Status DavinciModel::TransAllVarData(ComputeGraphPtr &graph, uint32_t graph_id) {
  GELOGI("TransAllVarData start: session_id:%lu, graph_id: %u.", session_id_, graph_id);
  rtContext_t ctx = nullptr;
  rtError_t rt_ret = rtCtxGetCurrent(&ctx);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Failed to get current context, error_code is: 0x%X.", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  std::vector<NodePtr> variable_node_list;
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() != VARIABLE) {
      continue;
    }
    variable_node_list.emplace_back(node);
  }

  GE_CHK_STATUS_RET_NOLOG(
      TransVarDataUtils::TransAllVarData(variable_node_list, session_id_, ctx, graph_id, kThreadNum));

  GELOGI("TransAllVarData success.");
  return SUCCESS;
}

void DavinciModel::SetDataDumperArgs(const ComputeGraphPtr &compute_graph) {
  GELOGI("set data dumper args, name: %s, id: %u.", name_.c_str(), model_id_);
  data_dumper_.SetModelName(name_);
  data_dumper_.SetModelId(model_id_);
  data_dumper_.SetOmName(om_name_);
  data_dumper_.SetComputeGraph(compute_graph);
  data_dumper_.SetRefInfo(saved_task_addrs_);
  data_dumper_.SetL1FusionAddr(l1_fusion_addr_);

  int32_t device_id = 0;
  rtError_t rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE || device_id < 0) {
    GELOGE(RT_FAILED, "Call rtGetDevice failed, ret = 0x%X, device_id = %d.", rt_ret, device_id);
    return;
  }
  data_dumper_.SetDeviceId(device_id);

  // set loop count addr
  auto get_var_addr = [](const OpDescPtr &op, const RuntimeParam &runtime_param) -> void * {
    if (op != nullptr) {
      auto v_output_size = ModelUtils::GetOutputSize(op);
      auto v_output_addr = ModelUtils::GetOutputDataAddrs(runtime_param, op);
      if (v_output_size.empty() || v_output_addr.empty()) {
        return nullptr;
      }
      return v_output_addr[0];
    }
    GELOGW("op is null.");
    return nullptr;
  };

  data_dumper_.SetLoopAddr(get_var_addr(GetVariableOp(NODE_NAME_GLOBAL_STEP), runtime_param_),
                           get_var_addr(GetVariableOp(NODE_NAME_FLOWCTRL_LOOP_PER_ITER), runtime_param_),
                           get_var_addr(GetVariableOp(NODE_NAME_FLOWCTRL_LOOP_COND), runtime_param_));

  GELOGI("SetDataDumperArgs end.");
}

uint32_t DavinciModel::GetFlowctrlIndex(uint32_t op_index) {
  std::lock_guard<std::mutex> lock(flowctrl_op_index_internal_map_mutex_);
  return (++flowctrl_op_index_internal_map_[op_index]) - 1;
}

void DavinciModel::PushHcclStream(rtStream_t value) {
  std::lock_guard<std::mutex> lock(all_hccl_stream_list_mutex_);
  all_hccl_stream_list_.push_back(value);
}

void DavinciModel::SaveHcclFollowStream(int64_t main_stream_id, rtStream_t stream) {
  std::lock_guard<std::mutex> lock(capacity_of_stream_mutex_);
  main_follow_stream_mapping_[main_stream_id].emplace_back(stream);
}

Status DavinciModel::GetComputeGraphInfo(vector<ComputeGraphDescInfo> &graph_desc_info) {
  GELOGI("GetComputeGraphInfo start.");
  auto &all_op_desc = data_dumper_.GetAllOpDescInfo();
  for (auto &op_desc : all_op_desc) {
    ComputeGraphDescInfo compute_graph_info;
    if (!om_name_.empty()) {
      compute_graph_info.model_name = om_name_;
    } else {
      compute_graph_info.model_name = name_;
    }
    compute_graph_info.op_name = op_desc.op_name;
    compute_graph_info.op_type = op_desc.op_type;
    compute_graph_info.input_format = op_desc.input_format;
    compute_graph_info.input_shape = op_desc.input_shape;
    compute_graph_info.input_data_type = op_desc.input_data_type;
    compute_graph_info.output_format = op_desc.output_format;
    compute_graph_info.output_shape = op_desc.output_shape;
    compute_graph_info.output_data_type = op_desc.output_data_type;

    graph_desc_info.emplace_back(compute_graph_info);
  }
  GELOGI("GetComputeGraphInfo end.");
  return SUCCESS;
}

void DavinciModel::SetTotalFixedAddrsSize(string tensor_name, int64_t fix_addr_size) {
  if (tensor_name_to_fixed_addr_size_.find(tensor_name) == tensor_name_to_fixed_addr_size_.end()) {
    tensor_name_to_fixed_addr_size_[tensor_name] = total_fixed_addr_size_;
    total_fixed_addr_size_ += fix_addr_size;
  }
}

Status DavinciModel::GetOrigInputInfo(uint32_t index, OriginInputInfo &orig_input_info) {
  GE_CHK_BOOL_RET_STATUS(index < data_op_list_.size(), PARAM_INVALID, "Index %u is invalid.", index);
  OpDescPtr data_op = data_op_list_[index];
  if (!data_op->HasAttr(ATTR_NAME_AIPP_INPUTS) || !data_op->HasAttr(ATTR_NAME_AIPP_OUTPUTS)) {
    GELOGE(GE_AIPP_NOT_EXIST, "GetOrigInputInfo: there is not AIPP related with index %u.", index);
    return GE_AIPP_NOT_EXIST;
  }

  vector<std::string> inputs;
  if (AttrUtils::GetListStr(data_op, ATTR_NAME_AIPP_INPUTS, inputs) && !inputs.empty()) {
    std::string input = inputs[kAippOriginInputIndex];
    GELOGI("GetOrigInputInfo: origin input str: %s", input.c_str());
    std::vector<std::string> infos = ge::StringUtils::Split(input, ':');
    if (infos.size() != kAippInfoNum) {
      GELOGW("origin input str is invalid.");
    }
    orig_input_info.format = TypeUtils::SerialStringToFormat(infos[kAippInfoFormat]);
    orig_input_info.data_type = TypeUtils::SerialStringToDataType(infos[kAippInfoDataType]);
    orig_input_info.dim_num = std::strtol(infos[kAippInfoDimNum].c_str(), nullptr, kDecimal);
  }

  return SUCCESS;
}

void DavinciModel::ParseAIPPInfo(std::string in_out_info, InputOutputDims &dims_info) {
  GELOGI("ParseAIPPInfo: origin str: %s", in_out_info.c_str());
  std::vector<std::string> infos = ge::StringUtils::Split(in_out_info, ':');
  if (infos.size() != kAippInfoNum) {
    GELOGW("origin input str is invalid.");
  }
  dims_info.name = infos[kAippInfoTensorName];
  dims_info.size = std::strtol(infos[kAippInfoTensorSize].c_str(), nullptr, kDecimal);
  dims_info.dim_num = std::strtol(infos[kAippInfoDimNum].c_str(), nullptr, kDecimal);

  std::vector<std::string> dims = ge::StringUtils::Split(infos[kAippInfoShape], ',');
  for (const auto &dim : dims) {
    if (dim.empty()) {
      continue;
    }
    dims_info.dims.emplace_back(std::strtol(dim.c_str(), nullptr, kDecimal));
  }
}

Status DavinciModel::GetAllAippInputOutputDims(uint32_t index, std::vector<InputOutputDims> &input_dims,
                                               std::vector<InputOutputDims> &output_dims) {
  GE_CHK_BOOL_RET_STATUS(index < data_op_list_.size(), PARAM_INVALID, "Index %u is invalid.", index);
  OpDescPtr data_op = data_op_list_[index];
  if (!data_op->HasAttr(ATTR_NAME_AIPP_INPUTS) || !data_op->HasAttr(ATTR_NAME_AIPP_OUTPUTS)) {
    GELOGE(GE_AIPP_NOT_EXIST, "GetAllAippInputOutputDims: there is not AIPP related with index %u.", index);
    return GE_AIPP_NOT_EXIST;
  }

  vector<std::string> inputs;
  if (AttrUtils::GetListStr(data_op, ATTR_NAME_AIPP_INPUTS, inputs) && !inputs.empty()) {
    GELOGI("GetAllAippInputOutputDims: Data: %s has %u related aippInfo.", data_op->GetName().c_str(), inputs.size());
    for (auto it : inputs) {
      InputOutputDims input_info;
      ParseAIPPInfo(it, input_info);
      input_dims.emplace_back(input_info);
      GELOGD("GetAllAippInputOutputDims Aipp origin input dims info: %s", it.c_str());

      ConstGeTensorDescPtr data_input_desc = data_op->GetInputDescPtr(kDataIndex);
      int64_t data_input_size;
      (void)TensorUtils::GetSize(*(data_op->GetInputDescPtr(kDataIndex)), data_input_size);
      GELOGD(
          "GetAllAippInputOutputDims related Data[%d]: tensor_name is %s, dim_num is %u, tensor_size: %zu, format: "
          "%s, data_type: %s, shape: %s .",
          index, data_op->GetName().c_str(), data_input_desc->GetShape().GetDimNum(), data_input_size,
          TypeUtils::FormatToSerialString(data_input_desc->GetFormat()).c_str(),
          TypeUtils::DataTypeToSerialString(data_input_desc->GetDataType()).c_str(),
          formats::JoinToString(data_input_desc->GetShape().GetDims()).c_str());
    }
  }

  vector<std::string> outputs;
  if (AttrUtils::GetListStr(data_op, ATTR_NAME_AIPP_OUTPUTS, outputs) && !outputs.empty()) {
    for (auto it : outputs) {
      InputOutputDims output_info;
      ParseAIPPInfo(it, output_info);
      output_dims.emplace_back(output_info);
      GELOGD("GetAllAippInputOutputDims Aipp output dims info: %s", it.c_str());
    }
  }

  return SUCCESS;
}

int64_t DavinciModel::GetFixedAddrsSize(string tensor_name) {
  if (tensor_name_to_fixed_addr_size_.find(tensor_name) != tensor_name_to_fixed_addr_size_.end()) {
    return tensor_name_to_fixed_addr_size_[tensor_name];
  } else {
    return total_fixed_addr_size_;
  }
}

}  // namespace ge
