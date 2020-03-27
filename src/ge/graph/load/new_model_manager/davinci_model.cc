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

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <securec.h>
#include <sys/prctl.h>

#include <algorithm>
#include <map>
#include <utility>

#include "cce/cce.h"
#include "cce/dnn.h"
#include "cce/optimizer/fusion_engine.h"
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
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/graph.h"
#include "graph/load/new_model_manager/tbe_handle_store.h"
#include "graph/load/output/output.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/util/debug.h"
#include "graph/model_serialize.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "mmpa/mmpa_api.h"
#include "model_output.h"
#include "omm/csa_interact.h"
#include "runtime/base.h"
#include "runtime/dev.h"
#include "runtime/event.h"
#include "runtime/mem.h"
#include "runtime/stream.h"

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
const uint32_t DEFAULT_DATA_INDEX = 0;
const uint32_t TRUE_BRANCH_STREAM_NUM = 1;
const uint32_t THREAD_NUM = 16;
const int kDecimal = 10;
const int kBytes = 8;

class RtContextSwitchGuard {
 public:
  RtContextSwitchGuard(rtCtxMode_t mode, uint32_t device_id) : last_(nullptr), current_(nullptr) {
    auto ret = rtCtxGetCurrent(&last_);
    if (ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Failed to get current context from rt, error-code %d", ret);
      return;
    }

    ret = rtCtxCreate(&current_, mode, static_cast<int32_t>(device_id));
    if (ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Failed to create new context for device %u, error-code %d", device_id, ret);
      return;
    }

    ret = rtCtxSetCurrent(current_);
    if (ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Failed to switch context to normal, device %u", device_id);
      return;
    }
  }

  ~RtContextSwitchGuard() {
    if (current_ != nullptr) {
      auto ret = rtCtxDestroy(current_);
      if (ret != RT_ERROR_NONE) {
        GELOGW("Failed to call rtCtxDestroy");
      }
    }
    if (last_ != nullptr) {
      auto ret = rtCtxSetCurrent(last_);
      if (ret != RT_ERROR_NONE) {
        GELOGW("Failed to call rtCtxSetCurrent");
      }
    }
  }

 private:
  rtContext_t last_;
  rtContext_t current_;
};

int CalcVarSizeInBytes(const GeTensorDesc &desc) {
  int var_size = GetSizeByDataType(desc.GetDataType());
  if (var_size <= 0) {
    GELOGE(PARAM_INVALID, "Failed to calc var data size from data type %s",
           TypeUtils::DataTypeToSerialString(desc.GetDataType()).c_str());
    return -1;
  }
  auto shape = desc.GetShape();
  auto dimNum = shape.GetDimNum();
  for (size_t dimIndex = 0; dimIndex < dimNum; ++dimIndex) {
    var_size *= static_cast<int>(shape.GetDim(dimIndex));
  }
  return var_size;
}

Status CopyVarFromDevice(uint64_t session_id, const NodePtr &var, std::unique_ptr<uint8_t[]> &var_data,
                         const GeTensorDesc &input_desc) {
  uint8_t *var_logic = nullptr;
  GE_CHECK_NOTNULL(var);
  auto ret = VarManager::Instance(session_id)->GetVarAddr(var->GetName(), input_desc, &var_logic);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR,
           "Failed to copy var %s from device, can not find it"
           " from var manager %u",
           var->GetName().c_str(), ret);
    return INTERNAL_ERROR;
  }

  uint8_t *var_addr = VarManager::Instance(session_id)->GetVarMemoryAddr(var_logic, RT_MEMORY_HBM);
  if (var_addr == nullptr) {
    GELOGE(INTERNAL_ERROR, "Failed to copy var %s from device, can not get var addr", var->GetName().c_str());
    return INTERNAL_ERROR;
  }

  int var_size_bytes = CalcVarSizeInBytes(input_desc);
  if (var_size_bytes <= 0) {
    return INTERNAL_ERROR;
  }

  std::unique_ptr<uint8_t[]> var_host(new (std::nothrow) uint8_t[var_size_bytes]);
  if (var_host == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed to malloc rt-host memory, size %d", var_size_bytes);
    return OUT_OF_MEMORY;
  }

  ret = rtMemcpy(reinterpret_cast<void *>(var_host.get()), var_size_bytes, reinterpret_cast<void *>(var_addr),
                 var_size_bytes, RT_MEMCPY_DEVICE_TO_HOST);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED,
           "Failed to copy var memory from device, var %s, size %d,"
           " rt-error-code %u",
           var->GetName().c_str(), var_size_bytes, ret);
    return RT_FAILED;
  }

  GELOGD("Copy var %s from device to host, size %d", var->GetName().c_str(), var_size_bytes);
  var_data.swap(var_host);

  return SUCCESS;
}

Status CopyVarToDevice(const NodePtr &var, const formats::TransResult &trans_result, void *var_addr) {
  GE_CHECK_NOTNULL(var);
  GELOGD("Copy var %s from host to device, size %zu", var->GetName().c_str(), trans_result.length);
  auto ret = rtMemcpy(var_addr, trans_result.length, reinterpret_cast<void *>(trans_result.data.get()),
                      trans_result.length, RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Failed to copy memory to device, size %zu", trans_result.length);
    return RT_FAILED;
  }
  return SUCCESS;
}

Status TransVarOnHost(uint8_t *var_data, const VarTransRoad &trans_road, formats::TransResult &result) {
  formats::TransResult resultLastTime{};
  bool use_init_data = true;
  for (const auto &trans_info : trans_road) {
    if (trans_info.node_type == RESHAPE || trans_info.node_type == REFORMAT) {
      GELOGD("Skip to trans variable data on the reshape/reformat node");
      continue;
    }
    uint8_t *src_data = nullptr;
    if (use_init_data) {
      src_data = var_data;
      use_init_data = false;
    } else {
      src_data = resultLastTime.data.get();
    }

    formats::TransResult tmp_result{};
    if (trans_info.node_type == TRANSDATA) {
      auto src_format = trans_info.input.GetFormat();
      auto src_shape = trans_info.input.GetShape().GetDims();
      auto dst_format = trans_info.output.GetFormat();
      auto dst_shape = trans_info.output.GetShape().GetDims();
      auto data_type = trans_info.input.GetDataType();
      GELOGD("Trans format from %s to %s, shape %s to %s, data-type %s",
             TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(dst_format).c_str(),
             formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(dst_shape).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str());
      auto ret = formats::TransFormat({src_data, src_format, dst_format, src_shape, dst_shape, data_type}, tmp_result);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR,
               "Failed to trans format from %s to %s, shape %s to %s, "
               "data type %s error code %u",
               TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(dst_format).c_str(),
               formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(dst_shape).c_str(),
               TypeUtils::DataTypeToSerialString(data_type).c_str(), ret);
        return ret;
      }
    } else if (trans_info.node_type == CAST) {
      auto input_shape = trans_info.input.GetShape();
      auto src_data_size = input_shape.GetShapeSize();
      auto src_data_type = trans_info.input.GetDataType();
      auto dst_data_type = trans_info.output.GetDataType();
      GELOGD("Trans data type from %s to %s, input shape %s, data size %ld",
             TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
             TypeUtils::DataTypeToSerialString(dst_data_type).c_str(), formats::ShapeToString(input_shape).c_str(),
             src_data_size);
      auto ret = formats::TransDataType({src_data, static_cast<size_t>(src_data_size), src_data_type, dst_data_type},
                                        tmp_result);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to trans data type from %s to %s, input shape %s, data size %ld, error code %u",
               TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
               TypeUtils::DataTypeToSerialString(dst_data_type).c_str(), formats::ShapeToString(input_shape).c_str(),
               src_data_size, ret);
        return ret;
      }
    } else {
      GELOGE(UNSUPPORTED, "Failed to trans var data, the trans type %s does not supported",
             trans_info.node_type.c_str());
      return UNSUPPORTED;
    }
    resultLastTime = tmp_result;
  }

  result = resultLastTime;
  return SUCCESS;
}

///
/// re-alloc var memory on device using var-manager
/// free origin var memory(var manager does not support now)
/// @param session_id
/// @param var
/// @param var_size_bytes
/// @param var_device
/// @return
///
Status ReAssignVarAddr(uint64_t session_id, const std::string &var_name, const GeTensorDesc &tensor_desc,
                       void **var_device) {
  uint8_t *var_logic = nullptr;
  Status ret = VarManager::Instance(session_id)->GetVarAddr(var_name, tensor_desc, &var_logic);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR,
           "Failed to get var %s device addr, can not find it"
           " from var manager %u",
           var_name.c_str(), ret);
    return INTERNAL_ERROR;
  }

  uint8_t *var_addr = VarManager::Instance(session_id)->GetVarMemoryAddr(var_logic, RT_MEMORY_HBM);
  if (var_addr == nullptr) {
    GELOGE(INTERNAL_ERROR, "Failed to convert var %s logic addr to real addr", var_name.c_str());
    return INTERNAL_ERROR;
  }
  *var_device = var_addr;

  return SUCCESS;
}

Status TransVarData(const NodePtr &var, const VarTransRoad &trans_road, uint64_t session_id, uint32_t device_id) {
  // do not need to do anything if only all reshape/reformat node on the trans_road
  GE_CHECK_NOTNULL(var);
  bool need_trans = false;
  if (std::any_of(trans_road.begin(), trans_road.end(), [](const ge::TransNodeInfo &road) {
        return road.node_type != RESHAPE && road.node_type != REFORMAT;
      })) {
    need_trans = true;
  }

  if (!need_trans) {
    return SUCCESS;
  }

  // Sync var data from device
  std::unique_ptr<uint8_t[]> var_data;
  if (trans_road.size() == 0) {
    GELOGE(INTERNAL_ERROR, "Failed to get trans_road, trans_road is empty.");
    return INTERNAL_ERROR;
  }
  const GeTensorDesc &input_desc = trans_road.begin()->input;
  auto ret = CopyVarFromDevice(session_id, var, var_data, input_desc);
  if (ret != SUCCESS) {
    return ret;
  }

  formats::TransResult trans_result{};
  ret = TransVarOnHost(var_data.get(), trans_road, trans_result);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to trans var data on host, error code %u", ret);
    return ret;
  }

  void *var_device = nullptr;
  ///
  /// It is a temporary solution to use the last GeTensorDesc to assign variable memory because the variable manager
  /// depends on TensorDesc and it is difficult to be modified. The correct solution is to assign memory based on the
  /// size of the converted variable. To complete the final solution, the dependency of the variable manager on
  /// TensorDesc needs to be removed. This change is large and needs to be performed step by step.
  ///
  ret = ReAssignVarAddr(session_id, var->GetName(), trans_road.rbegin()->output, &var_device);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to re-assign memory on device, size %zu", trans_result.length);
    return ret;
  }

  // sync new data to device
  ret = CopyVarToDevice(var, trans_result, var_device);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to send var data to device");
    return ret;
  }

  return SUCCESS;
}
}  // namespace

std::mutex DavinciModel::tvm_bin_mutex_;
std::set<std::string> DavinciModel::tvm_bin_kernel_;

DavinciModel::DavinciModel(int32_t priority, const std::shared_ptr<ModelListener> &listener)
    : weights_mem_base_(nullptr),
      var_mem_base_(nullptr),
      mem_base_(nullptr),
      is_inner_mem_base_(false),
      is_inner_weight_base_(false),
      data_inputer_(nullptr),
      dataInputTid(0),
      is_model_has_inited_(false),
      model_id_(0),
      version_(0),
      ge_model_(nullptr),
      thread_id_(),
      listener_(listener),
      run_flg_(false),
      priority_(priority),
      rt_model_handle_(nullptr),
      rt_model_stream_(nullptr),
      is_inner_model_stream_(false),
      support_mem_shared_flag_(false),
      session_id_(0),
      device_id_(0),
      is_train_mode_(false),
      model_task_def_(nullptr),
      maxDumpOpNum_(0) {
  op_list_.clear();
}

DavinciModel::~DavinciModel() {
  try {
    Status ret = data_dumper_.UnloadDumpInfo();
    if (ret != SUCCESS) {
      GELOGW("UnloadDumpInfo fail, ret: %u.", ret);
    }

    GE_CHK_STATUS(ModelRunStop());
    UnbindTaskSinkStream();

    op_list_.clear();
    data_op_list_.clear();
    output_op_list_.clear();

    GE_DELETE_NEW_SINGLE(data_inputer_);

    for (size_t i = 0; i < label_list_.size(); ++i) {
      GE_LOGW_IF(rtLabelDestroy(label_list_[i]) != RT_ERROR_NONE, "Destroy label failed! Index: %zu", i);
    }

    for (size_t i = 0; i < stream_list_.size(); ++i) {
      GE_LOGW_IF(rtStreamDestroy(stream_list_[i]) != RT_ERROR_NONE, "Destroy stream failed! Index: %zu", i);
    }

    for (size_t i = 0; i < event_list_.size(); ++i) {
      GE_LOGW_IF(rtEventDestroy(event_list_[i]) != RT_ERROR_NONE, "Destroy event failed. Index: %zu", i);
    }

    FreeWeightsMem();

    FreeFeatureMapMem();

    if (model_task_def_) {
      // release rtModel
      GELOGI("do ReleaseTask");
      GE_CHK_RT(rtModelDestroy(rt_model_handle_));
      ReleaseTask();
    }

    CleanTbeHandle();

    var_mem_base_ = nullptr;
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
  for (const auto &task : task_list_) {
    if (task != nullptr) {
      GE_CHK_STATUS(task->Release());
    }
  }
}

Status DavinciModel::Assign(const GeModelPtr &ge_model) {
  if (ge_model == nullptr) {
    GELOGI("can't assign null ge_model");
    return FAILED;
  }
  ge_model_ = ge_model;
  model_task_def_ = ge_model_->GetModelTaskDefPtr();
  return SUCCESS;
}

Status DavinciModel::InitModelMem(void *dev_ptr, size_t memsize, void *weight_ptr, size_t weight_size) {
  if (is_model_has_inited_) {
    GELOGI("call InitModelMem more than once .");
    return FAILED;
  }
  is_model_has_inited_ = true;
  std::size_t data_size = TotalMemSize();
  ge::Buffer weights = ge_model_->GetWeight();

  uint8_t *weights_addr = weights.GetData();
  std::size_t weights_size = weights.GetSize();

  GE_CHECK_LE(weights_size, ALLOC_MEMORY_MAX_SIZE);

  if ((dev_ptr != nullptr) && (memsize < TotalMemSize())) {
    GELOGE(FAILED, "Invalid mem param: memsize=%zu totalsize=%zu.", memsize, TotalMemSize());
    return FAILED;
  }

  if ((weight_ptr != nullptr) && (weight_size < weights_size)) {
    GELOGE(FAILED, "Invalid mem param: weight_size=%zu totalsize=%zu.", weight_size, weights_size);
    return FAILED;
  }

  mem_base_ = static_cast<uint8_t *>(dev_ptr);
  weights_mem_base_ = static_cast<uint8_t *>(dev_ptr);
  is_inner_mem_base_ = false;
  is_inner_weight_base_ = false;

  if (TotalMemSize() && mem_base_ == nullptr) {
    mem_base_ = MallocFeatureMapMem(data_size);
    if (mem_base_ == nullptr) {
      return FAILED;
    }

    weights_mem_base_ = mem_base_;

    is_inner_mem_base_ = true;
    is_inner_weight_base_ = true;
  }

  if (weights_size != 0) {
    weights_mem_base_ = static_cast<uint8_t *>(weight_ptr);
    is_inner_weight_base_ = false;
    if (weight_ptr == nullptr) {
      weights_mem_base_ = MallocWeightsMem(weights_size);
      if (weights_mem_base_ == nullptr) {
        return FAILED;
      }
      is_inner_weight_base_ = true;
    }
    GE_CHK_RT_RET(rtMemcpy(weights_mem_base_, weights_size, weights_addr, weights_size, RT_MEMCPY_HOST_TO_DEVICE))
    GELOGI("copy weights data to device");
  }

  var_mem_base_ = VarManager::Instance(session_id_)->GetVarMemoryBase(RT_MEMORY_HBM);
  if (TotalVarMemSize() && var_mem_base_ == nullptr) {
    Status ret = VarManager::Instance(session_id_)->MallocVarMemory(TotalVarMemSize());
    if (ret != SUCCESS) {
      GELOGE(ret, "Malloc Var Memory Fail.");
      return ret;
    }
    var_mem_base_ = VarManager::Instance(session_id_)->GetVarMemoryBase(RT_MEMORY_HBM);
  }

  runtime_param_.mem_base = mem_base_;
  runtime_param_.weight_base = weights_mem_base_;
  runtime_param_.var_base = var_mem_base_;
  return SUCCESS;
}

void DavinciModel::InitRuntimeParams() {
  int64_t value = 0;
  bool ret;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_MEMORY_SIZE, value);
  runtime_param_.mem_size = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_WEIGHT_SIZE, value);
  runtime_param_.weight_size = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_STREAM_NUM, value);
  runtime_param_.stream_num = ret ? (uint32_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_EVENT_NUM, value);
  runtime_param_.event_num = ret ? (uint64_t)value : 0;
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
  GELOGI("Init(),memory_size:%lu, weight_size:%lu, stream_num:%u, session_id:%lu, var_size:%lu.",
         runtime_param_.mem_size, runtime_param_.weight_size, runtime_param_.stream_num, runtime_param_.session_id,
         runtime_param_.var_size);

  GELOGI("Init(),event_num:%u, batch_num:%u", runtime_param_.event_num, runtime_param_.batch_num);
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
                     (op_desc->GetType() == HCOMRECEIVE) || (op_desc->GetType() == HCOMREDUCESCATTER)),
                    uint32_t stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
                    (void)hcom_streams_.emplace(stream_id); GELOGD("hcom stream: %u.", stream_id); continue);

    bool is_aicpu_stream = false;
    GE_IF_BOOL_EXEC(AttrUtils::GetBool(op_desc, "is_aicpu_stream", is_aicpu_stream) && is_aicpu_stream,
                    uint32_t stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
                    (void)aicpu_streams_.emplace(stream_id); GELOGD("aicpu stream: %u.", stream_id); continue);
  }
}

Status DavinciModel::DoTaskSink() {
  // task sink is supported as model_task_def is set
  if (model_task_def_) {
    GELOGI("do task_sink.");

    // create model_handle to load model
    GE_CHK_RT_RET(rtModelCreate(&rt_model_handle_, 0));

    for (size_t i = 0; i < stream_list_.size(); i++) {
      GE_IF_BOOL_EXEC(active_stream_indication_.count(i) > 0, GELOGI("rtModelBindStream[%zu]", i);
                      GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, stream_list_[i], RT_INVALID_FLAG)); continue;);
      // bind rt_model_handel to all streams that relates to op
      GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, stream_list_[i], 0));
    }

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(InitTaskInfo(*model_task_def_.get()) != SUCCESS, return FAILED,
                                   "InitTaskInfo failed.");

    GE_CHK_STATUS_RET(DistributeTask(), "Distribute failed.");

    GE_CHK_RT_RET(rtModelLoadComplete(rt_model_handle_));
  }
  return SUCCESS;
}

// initialize op sequence and call initialization function of each op respectively
Status DavinciModel::Init(void *dev_ptr, size_t memsize, void *weight_ptr, size_t weight_size) {
  // validating params
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(priority_ < 0 || priority_ > 7, return PARAM_INVALID,
                                 "Priority must between 0-7, now is %d", priority_);
  GE_CHK_BOOL_RET_STATUS(ge_model_ != nullptr, PARAM_INVALID, "GeModel is null.");
  // Initializing runtime_param_
  InitRuntimeParams();

  version_ = ge_model_->GetVersion();
  name_ = ge_model_->GetName();

  CheckHasHcomOp();

  for (uint32_t i = 0; i < StreamNum(); i++) {
    rtStream_t stream = nullptr;
    GE_MAKE_GUARD_RTSTREAM(stream);

    if (hcom_streams_.find(i) != hcom_streams_.end()) {
      GE_CHK_RT_RET(rtStreamCreateWithFlags(&stream, priority_, RT_STREAM_PERSISTENT | RT_STREAM_FORCE_COPY));
    } else if (aicpu_streams_.find(i) != aicpu_streams_.end()) {
      GE_CHK_RT_RET(rtStreamCreateWithFlags(&stream, priority_, RT_STREAM_PERSISTENT | RT_STREAM_AICPU));
    } else {
      GE_CHK_RT_RET(rtStreamCreateWithFlags(&stream, priority_, RT_STREAM_PERSISTENT));
    }

    GE_DISMISS_GUARD(stream);
    stream_list_.push_back(stream);
  }

  for (uint32_t i = 0; i < EventNum(); i++) {
    rtEvent_t rt_event;
    GE_CHK_RT_RET(rtEventCreate(&rt_event));
    event_list_.push_back(rt_event);
  }

  for (uint32_t i = 0; ((BatchNum() != 0) && (i <= BatchNum())); i++) {
    rtLabel_t rtLabel;
    GE_CHK_RT_RET(rtLabelCreate(&rtLabel));
    GE_CHK_BOOL_RET_STATUS(rtLabel != nullptr, FAILED, "rtLabel is nullptr!");
    label_list_.push_back(rtLabel);
  }

  Graph graph = ge_model_->GetGraph();
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  GE_CHK_BOOL_RET_STATUS(compute_graph != nullptr, INTERNAL_ERROR, "Get compute graph is nullptr!");

  runtime_param_.graph_id = GetGraphID(compute_graph->GetName());

  GE_TIMESTAMP_START(TransAllVarData);
  GE_CHK_STATUS_RET(TransAllVarData(compute_graph, runtime_param_.graph_id), "TransAllVarData failed.");
  GE_TIMESTAMP_END(TransAllVarData, "GraphLoader::TransAllVarData");
  GE_CHK_STATUS_RET(CopyVarData(compute_graph), "copy var data failed.");

  GE_TIMESTAMP_START(InitModelMem);
  GE_CHK_STATUS_RET_NOLOG(InitModelMem(dev_ptr, memsize, weight_ptr, weight_size));
  GE_TIMESTAMP_END(InitModelMem, "GraphLoader::InitModelMem");

  InitDataDumper();
  data_inputer_ = new (std::nothrow) DataInputer();
  GE_CHK_BOOL_RET_STATUS(data_inputer_ != nullptr, INTERNAL_ERROR, "data_inputer_ is nullptr!");

  for (const ge::NodePtr &node : compute_graph->GetDirectNode()) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    GE_IF_BOOL_EXEC(node->GetOpDesc()->GetType() != VARIABLE, continue);
    GE_IF_BOOL_EXEC(IsBroadCastOpData(node),
                    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), VAR_ATTR_VAR_IS_BROADCAST, "var_is_restore"););
  }
  // for profiling
  op_name_map_ = compute_graph->GetGraphOpName();

  GE_TIMESTAMP_CALLNUM_START(LoadTBEKernelBinToOpDesc);
  GE_TIMESTAMP_CALLNUM_START(InitTbeHandle);

  vector<string> op_name;
  GE_IF_BOOL_EXEC(ge::AttrUtils::GetListStr(ge_model_, ATTR_MODEL_TASK_INDEX_OP_NAME, op_name),
                  GELOGI("get str of task_index_op_name"));
  if (op_name_map_.empty()) {
    for (size_t idx = 0; idx < op_name.size(); idx++) {
      op_name_map_[idx] = op_name[idx];
    }
    GELOGI("infer profiling: op_name_size(%zu)", op_name.size());
  }

  auto nodes = compute_graph->GetAllNodes();
  tbekernel_store_ = ge_model_->GetTBEKernelStore();
  for (size_t i = 0; i < nodes.size(); i++) {
    auto node = nodes.at(i);
    GE_CHK_BOOL_RET_STATUS(node != nullptr, PARAM_INVALID, "CreateOp failed.");

    auto op_desc = node->GetOpDesc();
    GE_CHK_BOOL_RET_STATUS(op_desc != nullptr, PARAM_INVALID, "op_desc is null.");
    op_list_[i] = op_desc;

    GE_TIMESTAMP_RESTART(LoadTBEKernelBinToOpDesc);
    tbekernel_store_.LoadTBEKernelBinToOpDesc(op_desc);
    GE_TIMESTAMP_ADD(LoadTBEKernelBinToOpDesc);

    if (op_desc->GetType() == DATA_TYPE || op_desc->GetType() == AIPP_DATA_TYPE ||
        op_desc->GetType() == ANN_DATA_TYPE) {
      data_op_list_.push_back(op_desc);
      GE_IF_BOOL_EXEC(
          (op_desc->GetInputDescPtr(0) != nullptr && op_desc->GetInputDescPtr(0)->GetFormat() != FORMAT_FILTER_HWCK),
          data_op_input_tensor_desc_map_[op_desc->GetName()] = op_desc->GetInputDescPtr(0));
      GE_IF_BOOL_EXEC(
          (op_desc->GetOutputDescPtr(0) != nullptr && op_desc->GetOutputDescPtr(0)->GetFormat() != FORMAT_FRACTAL_Z),
          data_op_output_tensor_desc_map_[op_desc->GetName()] = op_desc->GetOutputDescPtr(0));
      SetOutsideAddr(ModelUtils::GetOutputDataAddrs(runtime_param_, op_desc));
      data_dumper_.SaveDumpInput(node);
    }

    GE_IF_BOOL_EXEC(op_desc->GetType() == VARIABLE, variable_op_list_.push_back(op_desc));

    GE_IF_BOOL_EXEC(op_desc->GetType() == NETOUTPUT, output_op_list_.push_back(op_desc);
                    GE_CHK_STATUS_RET(ModelUtils::GetOutputSize(op_desc, output_size_list_, output_memory_size_list_),
                                      "Get output size fail");
                    SetOutsideAddr(ModelUtils::GetInputDataAddrs(runtime_param_, op_desc)));

    // Initialize constant op, only applies to training, ignoring inference constant op
    GE_IF_BOOL_EXEC(op_desc->GetType() == CONSTANTOP,
                    GE_CHK_STATUS_RET(InitConstant(op_desc), "Constant init failed. %s", op_desc->GetName().c_str()););

    GE_TIMESTAMP_RESTART(InitTbeHandle);
    uint32_t run_mode = static_cast<uint32_t>(domi::ImplyType::INVALID);
    GE_IF_BOOL_EXEC((AttrUtils::GetInt(op_desc, ATTR_NAME_IMPLY_TYPE, run_mode) &&
                     run_mode == static_cast<uint32_t>(domi::ImplyType::TVM)),
                    GE_CHK_STATUS_RET(InitTbeHandle(op_desc), "TBE init failed. %s", op_desc->GetName().c_str()););
    GE_TIMESTAMP_ADD(InitTbeHandle);

    GE_CHK_STATUS_RET(MarkActiveStream(op_desc), "MarkActiveStream failed, node:%s, opIndex:%zu",
                      op_desc->GetName().c_str(), i);
  }
  GE_TIMESTAMP_CALLNUM_END(LoadTBEKernelBinToOpDesc, "GraphLoader::LoadTBEKernelBinToOpDesc");
  GE_TIMESTAMP_CALLNUM_END(InitTbeHandle, "GraphLoader::InitTbeHandle");

  GE_TIMESTAMP_START(DoTaskSink);
  auto ret = DoTaskSink();
  GE_TIMESTAMP_END(DoTaskSink, "GraphLoader::DoTaskSink");
  return ret;
}

///
/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [in] input_queue_ids: input queue ids from user, nums equal Data Op.
/// @param [in] output_queue_ids: input queue ids from user, nums equal NetOutput Op.
/// @return: 0 for success / others for fail
///
Status DavinciModel::SetQueIds(const std::vector<uint32_t> &input_queue_ids,
                               const std::vector<uint32_t> &output_queue_ids) {
  if (input_queue_ids.empty() && output_queue_ids.empty()) {
    GELOGE(PARAM_INVALID, "Para is empty");
    return PARAM_INVALID;
  }

  input_queue_ids_ = input_queue_ids;
  output_queue_ids_ = output_queue_ids;
  return SUCCESS;
}

///
/// @brief define static mode and mutex mode
///
SysMode DavinciModel::mode_ = INFERENCE;
std::mutex DavinciModel::mutex_mode_;

///
/// @ingroup domi_ome
/// @brief get sys mode
/// @return SysMode required system mode
/// @author
///
SysMode DavinciModel::GetSysMode() {
  std::unique_lock<std::mutex> lock(mutex_mode_);
  return mode_;
}

///
/// @ingroup domi_ome
/// @brief set sys mode
/// @param [in] mode to be set
/// @return Status mode set result
/// @author
///
Status DavinciModel::SetSysMode(SysMode mode) {
  GE_CHK_BOOL_RET_STATUS(mode < RESERVED, PARAM_INVALID, "DavinciModel::SetSysMode Para Error");

  std::unique_lock<std::mutex> lock(mutex_mode_);
  mode_ = mode;
  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc) {
  if ((data_op_list_.empty()) || (data_op_list_[0]->GetInputsSize()) != 1) {
    GELOGI("data_op_list_ is empty or input_desc size is not 1.");
  } else {
    std::vector<uint32_t> input_formats;
    GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats), "get input desc info failed");
  }

  std::vector<uint32_t> output_formats;
  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats), "get output desc info failed");

  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfoForZeroCopy(vector<InputOutputDescInfo> &input_desc,
                                                       vector<InputOutputDescInfo> &output_desc) {
  if ((data_op_list_.empty()) || (data_op_list_[0]->GetInputsSize()) != 1) {
    GELOGE(FAILED, "OP List Pointer is null or input_desc size is not 1!");
    return FAILED;
  }

  std::vector<uint32_t> input_formats;
  GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats), "get input desc info failed");
  std::vector<uint32_t> output_formats;
  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats), "get output desc info failed");

  GE_CHK_BOOL_RET_STATUS(output_desc.size() == output_size_list_.size(), INTERNAL_ERROR,
                         "output_desc size[%zu] not equal output_size_list_[%zu] size!", output_desc.size(),
                         output_size_list_.size());

  GE_CHECK_GE(output_memory_size_list_.size(), output_size_list_.size());
  /// For function zero copy,the memory should be aligned by 512 bytes.
  /// And, because of the cce op limit, size should be lager than the real shape size. The memory should be padded by 32
  /// bytes.
  /// *size equals to ((tensorDesc->dataSize + 2 * 32 - 1) / 32) * 32;
  for (size_t i = 0; i < output_size_list_.size(); i++) {
    output_desc[i].size = output_memory_size_list_[i];
  }

  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc,
                                            std::vector<uint32_t> &input_formats,
                                            std::vector<uint32_t> &output_formats) {
  if ((data_op_list_.empty()) || (data_op_list_[0]->GetInputsSize()) != 1) {
    GELOGE(FAILED, "OP List Pointer is null or input_desc size is not 1!");
    return FAILED;
  }

  GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats), "get input desc info failed");

  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats), "get ouput desc info failed");

  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfoForZeroCopy(vector<InputOutputDescInfo> &input_desc,
                                                       vector<InputOutputDescInfo> &output_desc,
                                                       std::vector<uint32_t> &input_formats,
                                                       std::vector<uint32_t> &output_formats) {
  if ((data_op_list_.empty()) || (1 != data_op_list_[0]->GetInputsSize())) {
    GELOGE(FAILED, "OP List Pointer is null or input_desc size is not 1!");
    return FAILED;
  }

  GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats), "get input desc info failed");

  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats), "get ouput desc info failed");

  GE_CHK_BOOL_RET_STATUS(output_desc.size() == output_size_list_.size(), INTERNAL_ERROR,
                         "output_desc size[%zu] not equal output_size_list_[%zu] size!", output_desc.size(),
                         output_size_list_.size());

  GE_CHECK_GE(output_memory_size_list_.size(), output_size_list_.size());
  /// For function zero copy,the momery should be aligned by 512 bytes.
  /// And, because of the cce op limit, size should be lager than the real shape size. The memory should be padded by 32
  /// bytes.
  /// *size equals to ((tensorDesc->dataSize + 2 * 32 - 1) / 32) * 32;
  for (size_t i = 0; i < output_size_list_.size(); i++) {
    output_desc[i].size = output_memory_size_list_[i];
  }

  return SUCCESS;
}

Status DavinciModel::GetInputDescInfo(vector<InputOutputDescInfo> &input_desc, std::vector<uint32_t> &formats) {
  for (std::size_t index = 0; index < data_op_list_.size(); ++index) {
    InputOutputDescInfo input;
    uint32_t n, c, h, w;
    GE_CHECK_NOTNULL(data_op_list_[index]);
    GE_CHECK_NOTNULL(data_op_list_[index]->GetInputDescPtr(0));
    Format format = data_op_list_[index]->GetOutputDescPtr(0)->GetFormat();
    n = format == FORMAT_NHWC ? NHWC_DIM_N : NCHW_DIM_N;
    c = format == FORMAT_NHWC ? NHWC_DIM_C : NCHW_DIM_C;
    h = format == FORMAT_NHWC ? NHWC_DIM_H : NCHW_DIM_H;
    w = format == FORMAT_NHWC ? NHWC_DIM_W : NCHW_DIM_W;

    if (data_op_list_[index]->GetInputDescPtr(0)->GetShape().GetDimNum() == static_cast<size_t>(NORMAL_TENSOR_SIZE)) {
      input.shape_info.num = data_op_list_[index]->GetInputDescPtr(0)->GetShape().GetDim(n);
      input.shape_info.height = data_op_list_[index]->GetInputDescPtr(0)->GetShape().GetDim(h);
      input.shape_info.width = data_op_list_[index]->GetInputDescPtr(0)->GetShape().GetDim(w);
      input.shape_info.channel = data_op_list_[index]->GetInputDescPtr(0)->GetShape().GetDim(c);
    }
    for (size_t k = 0; k < data_op_list_[index]->GetInputDescPtr(0)->GetShape().GetDimNum(); k++) {
      input.shape_info.dims.push_back(data_op_list_[index]->GetInputDescPtr(0)->GetShape().GetDim(k));
    }

    input.data_type = data_op_list_[index]->GetInputDescPtr(0)->GetDataType();
    input.name = data_op_list_[index]->GetName();
    uint32_t input_size = 0;
    GE_CHK_STATUS_RET(TensorUtils::GetSize(*data_op_list_[index]->GetInputDescPtr(0), input_size),
                      "get input size failed.");
    input.size = input_size;
    formats.push_back(format);
    input_desc.push_back(input);
  }
  return SUCCESS;
}

void DavinciModel::CreateOutput(uint32_t index, OpDescPtr &op_desc, InputOutputDescInfo &output,
                                uint32_t &format_result) {
  /// netoutput input tensor desc
  GE_IF_BOOL_EXEC(op_desc->GetInputDescPtr(index) == nullptr, GELOGE(FAILED, "OpDesc GetInputDescPtr is nullptr");
                  return);
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
  (void)TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);
  output.size = static_cast<uint32_t>(tensor_size);
  output.data_type = op_desc->GetInputDescPtr(index)->GetDataType();
}

Status DavinciModel::GetOutputDescInfo(vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &formats) {
  for (size_t i = 0; i < output_op_list_.size(); i++) {
    auto &op_desc = output_op_list_[i];
    uint32_t out_size = static_cast<uint32_t>(op_desc->GetOutputsSize());

    for (uint32_t index = 0; index < out_size; index++) {
      bool is_output = false;
      GE_IF_BOOL_EXEC(op_desc->GetOutputDescPtr(index) == nullptr,
                      GELOGE(INTERNAL_ERROR, "OpDesc GetOutputDescPtr is nullptr");
                      return INTERNAL_ERROR);
      GE_CHK_STATUS(TensorUtils::GetOutputTensor(*op_desc->GetOutputDescPtr(index), is_output),
                    "get output tensor failed.");
      if (!is_output) {
        continue;
      }

      string output_name;
      InputOutputDescInfo output;
      uint32_t format_result;
      CreateOutput(index, op_desc, output, format_result);

      std::vector<std::string> src_name = op_desc->GetSrcName();
      std::vector<int64_t> src_index = op_desc->GetSrcIndex();
      GE_CHK_BOOL_RET_STATUS(src_name.size() > index && src_index.size() > index, INTERNAL_ERROR,
                             "construct output_name failed.");
      output_name = std::string("output_") + std::to_string(index) + "_" + src_name[index] + "_" +
                    std::to_string(src_index[index]);
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

Status DavinciModel::CopyInputData(const InputData &current_data, bool device_data) {
  Status ret = SUCCESS;
  uint32_t data_op_index = 0;

  for (auto op_desc : data_op_list_) {
    ret = CopyInputDataToModel(current_data.blobs, data_op_index, device_data);

    GE_CHK_BOOL_EXEC(ret == SUCCESS, break, "Copy input data to model ret fail, index:%u, model id:%u",
                     current_data.index, current_data.model_id);
    data_op_index++;
  }
  return ret;
}

Status DavinciModel::SyncVarData() {
  GELOGI("SyncBroadCastData2Var model id:%u", model_id_);
  Status ret = SUCCESS;

  for (auto op_desc : variable_op_list_) {
    ret =
        VarManager::Instance(session_id_)->SyncVarData(runtime_param_.graph_id, op_desc->GetName(), op_desc, mem_base_);
    GE_CHK_BOOL_EXEC(ret == SUCCESS, break, "sync var data ret fail, model id:%u, op name:%s", model_id_,
                     op_desc->GetName().c_str());
  }
  return ret;
}

///
/// @ingroup domi_ome
/// @brief copy input data to Model's firat OP. Address already malloced when Load
/// @copy need datatype transfer: FLOAT to FP16, 4D to 5D;
/// @param [in] data data pointer to be copy
/// @return Status result
/// @author
///
Status DavinciModel::CopyInputDataToModel(const std::vector<DataBuffer> &data, uint32_t data_op_index,
                                          bool device_data) {
  GE_CHK_BOOL_RET_STATUS(!data_op_list_.empty(), PARAM_INVALID, "data_op_list_ is empty!");

  GE_CHK_BOOL_RET_STATUS(data_op_list_.size() == data.size(), PARAM_INVALID,
                         "The input data list size (%zu) does not match the model input list size (%zu)", data.size(),
                         data_op_list_.size());

  GE_CHK_BOOL_RET_STATUS(data_op_index < data_op_list_.size(), PARAM_INVALID,
                         "input data op index(%u) is invalid, exceeds input op size(%zu)", data_op_index,
                         data_op_list_.size());

  /// input datatype conversion, converting FLOAT to FP16, 4D to 5D at the same time.
  /// Choose respective mode in API parameters.
  auto op_def = data_op_list_[data_op_index];
  GE_CHK_BOOL_EXEC(op_def != nullptr, return PARAM_INVALID, "op_def is null!");

  auto data_index = data_op_index;
  if (AttrUtils::GetInt(op_def, "index", data_index)) {
    GELOGI("ge_train:get new index %u , old %u", data_index, data_op_index);
  }

  GE_CHK_BOOL_EXEC(data_index < data.size(), return PARAM_INVALID, "index:%u >= size:%zu", data_index, data.size());
  GE_CHK_BOOL_RET_STATUS(op_def->GetInputsSize() == 1 && op_def->GetOutputsSize() == 1, PARAM_INVALID,
                         "Data Op has invalid input_desc_size(%zu) or output_desc_size(%zu)", op_def->GetInputsSize(),
                         op_def->GetOutputsSize());

  uint32_t input_size = 0;
  GE_CHK_STATUS(TensorUtils::GetSize(*op_def->GetInputDescPtr(0), input_size), "get input size failed.");

  GE_CHK_BOOL_RET_STATUS(input_size >= data[data_index].length, PARAM_INVALID,
                         "input data size(%u) does not match model required size(%u), ret fail.",
                         data[data_index].length, input_size);

  // float to float16
  bool need_trans_flag = ModelUtils::IsInputTensorNeedTrans(data_op_list_[data_op_index], 0);

  uint32_t output_size = 0;
  GE_CHK_STATUS(TensorUtils::GetSize(*op_def->GetOutputDescPtr(0), output_size), "get output size failed.");

  vector<GeAttrValue::INT> outputs = op_def->GetOutputOffset();
  GE_CHECK_VECTOR_NOT_EMPTY(outputs);

  bool need_memset = false;
  (void)AttrUtils::GetBool(op_def, "_need_memset", need_memset);
  if (need_memset) {
    void *data_out_addr = mem_base_ + outputs[0];
    // data+allreduce output align 512
    uint32_t output_size_align = (output_size + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
    GE_CHK_RT_RET(rtMemset(data_out_addr, output_size_align + 1, 0U, output_size_align));
  }
  if (device_data) {
    return CopyPlainData(data, data_index, data_op_index, outputs, output_size, RT_MEMCPY_DEVICE_TO_DEVICE);
  } else if (need_trans_flag) {
    return CopyTransData(data, data_index, data_op_index, outputs, output_size);
  } else {
    return CopyPlainData(data, data_index, data_op_index, outputs, output_size, RT_MEMCPY_HOST_TO_DEVICE);
  }
}

Status DavinciModel::CopyTransData(const std::vector<DataBuffer> &data, uint32_t data_index, uint32_t data_op_index,
                                   const std::vector<GeAttrValue::INT> &outputs, uint32_t output_size) {
  GE_CHECK_VECTOR_NOT_EMPTY(outputs);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(outputs[0] == -1, return PARAM_INVALID, "output offset is -1");
  GE_CHK_BOOL_EXEC(data_index < data.size(), return PARAM_INVALID, "index:%u >= size:%zu", data_index, data.size());

  auto input_tensor_desc = data_op_input_tensor_desc_map_[data_op_list_[data_op_index]->GetName()];
  auto output_tensor_desc = data_op_output_tensor_desc_map_[data_op_list_[data_op_index]->GetName()];

  uint8_t *src_data = reinterpret_cast<uint8_t *>(data[data_index].data);

  formats::TransResult tmp_result{};
  auto input_shape = input_tensor_desc->GetShape();
  auto src_data_size = input_shape.GetShapeSize();
  auto src_data_type = input_tensor_desc->GetDataType();
  auto dst_data_type = output_tensor_desc->GetDataType();
  GELOGD("Trans data type from %s to %s, input shape %s, data size %zu",
         TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
         TypeUtils::DataTypeToSerialString(dst_data_type).c_str(), formats::ShapeToString(input_shape).c_str(),
         src_data_size);
  auto ret =
      formats::TransDataType({src_data, static_cast<size_t>(src_data_size), src_data_type, dst_data_type}, tmp_result);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to trans data type from %s to %s, input shape %s, data size %zu, error code %d",
           TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
           TypeUtils::DataTypeToSerialString(dst_data_type).c_str(), formats::ShapeToString(input_shape).c_str(),
           src_data_size, ret);
    return ret;
  }

  void *mem_addr = mem_base_ + outputs[0];
  auto rt_ret = rtMemcpy(mem_addr, runtime_param_.mem_size - outputs[0], tmp_result.data.get(), tmp_result.length,
                         RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Failed to copy memory to device, size %zu", tmp_result.length);
    return RT_FAILED;
  }
  GELOGI("[IMAS]CopyTransData memcpy graph_%u type[F] name[%s] output[%d] datasize[%zu]", runtime_param_.graph_id,
         data_op_list_[data_op_index]->GetName().c_str(), 0, tmp_result.length);
  return SUCCESS;
}

Status DavinciModel::CopyPlainData(const std::vector<DataBuffer> &data, uint32_t data_index, uint32_t data_op_index,
                                   const std::vector<GeAttrValue::INT> &outputs, uint32_t output_size,
                                   rtMemcpyKind_t kind) {
  GE_CHK_BOOL_EXEC(data_index < data.size(), return PARAM_INVALID, "index:%u >= size:%zu", data_index, data.size());
  bool flag = data[data_index].isDataSupportMemShare && support_mem_shared_flag_;
  // if data attr support zero cpy,then update addrs info to flowtable
  if (flag) {
    GELOGI("No need to copy input data, user's input data buffer can be shared.");
    return SUCCESS;
  }

  GE_CHECK_VECTOR_NOT_EMPTY(outputs);
  // P2P memory space parameters
  void *host_data_addr = data[data_index].data;
  uint32_t copy_size = data[data_index].length;
  GELOGD("data output tensor is aipp tensor,copy data only.");

  void *data_out_addr = nullptr;
  if (VarManager::Instance(session_id_)->IsVarAddr(outputs[0])) {
    data_out_addr = var_mem_base_ + outputs[0] - runtime_param_.logic_var_base;
  } else {
    data_out_addr = mem_base_ + outputs[0];
    GELOGI("output[0]=%ld, copy_size=%u, total_size=%zu", outputs[0], copy_size, TotalMemSize());

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(((uint64_t)outputs[0] + (uint64_t)copy_size) > TotalMemSize(), return INTERNAL_ERROR,
                                   "input offset add size is large than total memory.");
  }

  GE_CHK_RT_RET(rtMemcpy(data_out_addr, copy_size, host_data_addr, copy_size, kind));

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief send Output Op result to upper layer
/// @already malloced in ModelLoad, no need to malloc again
/// @param [in] sink_op Sink Op
/// @return Status result
/// @author
///
Status DavinciModel::CopyOutputData(uint32_t model_id, uint32_t data_id, OutputData &output_data) {
  Status ret = SUCCESS;
  if (output_op_list_.empty()) {
    ret = SyncVarData();
  } else {
    output_data.index = data_id;
    output_data.model_id = model_id;
    GE_CHK_BOOL_RET_STATUS(output_data.blobs.size() == output_size_list_.size(), INTERNAL_ERROR,
                           "output buffer size[%zu] not equal output_size_list[%zu] size!", output_data.blobs.size(),
                           output_size_list_.size());

    // index of data in output_data
    uint32_t output_data_index = 0;
    for (auto &op_desc : output_op_list_) {
      ret = CopyOutputDataToUser(op_desc, output_data.blobs, output_data_index);
      GE_CHK_BOOL_EXEC(ret == SUCCESS, break, "Copy input data to model ret fail, index:%u, model id:%u",
                       output_data.index, output_data.model_id);
    }
  }

  (void)DumpOpInputOutput(op_list_, model_id);  // dump, not care result.
  return ret;
}

Status DavinciModel::CopyOutputDataToUser(OpDescPtr &op_desc, std::vector<DataBuffer> &blobs, uint32_t &data_index) {
  Output model_output(op_desc, this);

  GE_CHK_BOOL_RET_STATUS(model_output.Init() == SUCCESS, PARAM_INVALID, "make shared model_output failed");

  vector<uint32_t> v_output_size;
  vector<void *> v_output_data_addr;
  model_output.GetOutputData(v_output_data_addr, v_output_size);

  // for all output tensor, copy output data from op to designated position
  for (size_t i = 0; i < v_output_size.size(); ++i) {
    GE_CHK_BOOL_RET_STATUS(data_index < blobs.size(), PARAM_INVALID,
                           "The blobs size:%zu, data_op size:%zu, curr output size:%zu", blobs.size(),
                           data_op_list_.size(), v_output_size.size());

    DataBuffer &data_buf = blobs[data_index];
    data_index++;

    uint32_t size = data_buf.length;
    GE_CHK_BOOL_RET_STATUS(size <= v_output_size[i], PARAM_INVALID,
                           "Model output data size(%u) does not match required size(%u).", v_output_size[i],
                           data_buf.length);
    GE_CHK_RT_RET(rtMemcpy(data_buf.data, size, v_output_data_addr[i], size, RT_MEMCPY_DEVICE_TO_DEVICE));
  }

  return SUCCESS;
}

Status DavinciModel::SyncDataAndDump() {
  Status ret = SUCCESS;
  if (output_op_list_.empty()) {
    ret = SyncVarData();
  }

  (void)DumpOpInputOutput(op_list_, model_id_);  // dump, not care result.
  return ret;
}

///
/// @ingroup domi_ome
/// @brief send Output Op result to upper layer
/// @already malloced in ModelLoad, no need to malloc again
/// @param [in] sink_op Sink Op
/// @return Status result
/// @author
///
Status DavinciModel::ReturnResult(uint32_t model_id, uint32_t data_id, const bool rslt_flg, const bool seq_end_flag,
                                  OutputData *output_data) {
  GE_CHK_BOOL_EXEC(listener_ != nullptr, return PARAM_INVALID, "listener_ is null!");
  if (seq_end_flag) {
    GELOGW("End of sequence, model id: %u", model_id);
    GE_CHK_STATUS(listener_->OnComputeDone(model_id, data_id, END_OF_SEQUENCE), "OnComputeDone failed");
    return END_OF_SEQUENCE;
  }

  // return result is not required
  if (!rslt_flg) {
    GELOGW("Compute failed, model id: %u", model_id);
    GE_CHK_STATUS(listener_->OnComputeDone(model_id, data_id, INTERNAL_ERROR), "OnComputeDone failed");
    return INTERNAL_ERROR;
  }

  if (output_op_list_.empty()) {
    GELOGW("Output tensor list is empty, model id: %u", model_id);
    GE_CHK_STATUS(listener_->OnComputeDone(model_id, data_id, INTERNAL_ERROR), "OnComputeDone failed");
    return INTERNAL_ERROR;
  }

  GE_CHECK_NOTNULL(output_data);
  // index of data in output_data
  uint32_t data_index = 0;

  output_data->index = data_id;
  output_data->model_id = model_id;

  // copy output data from op to designated position
  for (auto &op_desc : output_op_list_) {
    Status ret = ModelOutput::CopyResult(this, op_desc, *output_data, data_index, support_mem_shared_flag_);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "CopyResult failed, op name: %s", op_desc->GetName().c_str());
      GE_CHK_STATUS(listener_->OnComputeDone(model_id, data_id, INTERNAL_ERROR), "OnComputeDone failed");
      return INTERNAL_ERROR;
    }
  }

  GE_IF_BOOL_EXEC((DumpOpInputOutput(op_list_, model_id) != SUCCESS),
                  GELOGW("dump op failed, model_id: %u", model_id););

  GE_CHK_STATUS(listener_->OnComputeDone(model_id, data_id, SUCCESS), "OnComputeDone failed");
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief return not output to upper layer for cloud case
/// @return Status result
///
Status DavinciModel::ReturnNoOutput(uint32_t model_id, uint32_t data_id) {
  GELOGI("ReturnNoOutput model id:%u", model_id);
  for (const auto &op_desc : variable_op_list_) {
    Status ret = VarManager::Instance(session_id_)
                     ->SyncBroadCastData2Var(runtime_param_.graph_id, op_desc->GetName(), op_desc, mem_base_);
    GE_CHK_BOOL_EXEC(ret == SUCCESS, break, "sync var data ret fail, model id:%u, op name:%s", model_id,
                     op_desc->GetName().c_str());
  }

  GE_IF_BOOL_EXEC(DumpOpInputOutput(op_list_, model_id) != SUCCESS, GELOGW("dump op failed, model_id: %u", model_id););
  GE_CHK_BOOL_EXEC(listener_ != nullptr, return PARAM_INVALID, "listener_ is null!");
  GE_CHK_STATUS(listener_->OnComputeDone(model_id, data_id, SUCCESS), "OnComputeDone failed");
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief dump all op input and output information
/// @param [in] op_list model_id
/// @return Status result
///
Status DavinciModel::DumpOpInputOutput(map<uint32_t, OpDescPtr> &op_list, uint32_t model_id) {
  if (op_list.empty()) {
    GELOGW("op_list is empty.");
    return FAILED;
  }
#ifdef FMK_SUPPORT_DUMP
  char *ge_dump_env = getenv("DUMP_OP");
  int dump_op_switch = (ge_dump_env != nullptr) ? std::strtol(ge_dump_env, nullptr, kDecimal) : 0;
  // 10 for decimal number
  if (dump_op_switch != 0) {
    int64_t cnt = 1;
    for (auto it : op_list) {
      if (maxDumpOpNum_ != 0 && cnt > maxDumpOpNum_) {
        GELOGW("dump op cnt > maxDumpOpNum, maxDumpOpNum: %ld.", maxDumpOpNum_);
        return SUCCESS;
      }
      Status ret = DumpSingleOpInputOutput(it.second, model_id);
      cnt++;
      if (ret != SUCCESS) {
        GELOGE(FAILED, "dump single op failed, model_id: %u", model_id);
        return FAILED;
      }
    }
  }
#else
  GELOGW("need to define FMK_SUPPORT_DUMP for dump op input and output.");
#endif

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief dump single op input and output information
/// @param [in] dump_op model_id
/// @return Status result
///
Status DavinciModel::DumpSingleOpInputOutput(const OpDescPtr &op_def, uint32_t model_id) {
  GE_CHK_BOOL_EXEC(op_def != nullptr, return PARAM_INVALID, "op_def is null!");
  string op_name = StringUtils::ReplaceAll(op_def->GetName(), "/", "-");
  GELOGI("dump op name:%s, type:%s, model_id: %u", op_def->GetName().c_str(), op_def->GetType().c_str(), model_id);
  string model_path = "./dump" + to_string(model_id);
  if (mmAccess(model_path.c_str()) != EN_OK) {
    int32_t ret = mmMkdir(model_path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    if (ret != EN_OK) {
      GELOGE(FAILED, "make dir failed, model_id: %u", model_id);
      return FAILED;
    }
  }
  const vector<uint32_t> input_size_vec = ModelUtils::GetInputSize(op_def);
  const vector<void *> input_addr_vec = ModelUtils::GetInputDataAddrs(runtime_param_, op_def, false);
  for (size_t i = 0; i < input_addr_vec.size(); i++) {
    uint32_t input_size = input_size_vec.at(i);
    char input_file_name[PATH_MAX] = {0};
    if ((sprintf_s(input_file_name, PATH_MAX, "%s/dump_%u_%s_%s_input_%zu.bin", model_path.c_str(), model_id,
                   op_def->GetType().c_str(), op_name.c_str(), i)) == -1) {
      GELOGE(FAILED, "construct input dump file path failed.");
      return FAILED;
    }
    if ((Debug::DumpDevMem(input_file_name, input_addr_vec.at(i), input_size)) != SUCCESS) {
      GELOGE(FAILED, "dump to input_file failed");
      return FAILED;
    }
  }

  const vector<uint32_t> output_size_vec = ModelUtils::GetOutputSize(op_def);
  const vector<void *> output_addr_vec = ModelUtils::GetOutputDataAddrs(runtime_param_, op_def, false);
  if (!(op_def->GetType() == "Const")) {
    for (size_t i = 0; i < output_addr_vec.size(); i++) {
      uint32_t output_size = output_size_vec.at(i);
      char output_file_name[PATH_MAX] = {0};
      if ((sprintf_s(output_file_name, PATH_MAX, "%s/dump_%u_%s_%s_output_%zu.bin", model_path.c_str(), model_id,
                     op_def->GetType().c_str(), op_name.c_str(), i)) == -1) {
        GELOGE(FAILED, "construct output dump file path failed.");
        return FAILED;
      }
      if ((Debug::DumpDevMem(output_file_name, output_addr_vec.at(i), output_size)) != SUCCESS) {
        GELOGE(FAILED, "dump to output_file failed");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

void *DavinciModel::Run(DavinciModel *model) {
  GE_CHK_BOOL_EXEC(model != nullptr,
                   CsaInteract::GetInstance().WriteErrorCode(FAILED, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
                   return nullptr, "model_pointer is null!")
  bool seq_end_flag = false;
  uint32_t interator_count = 0;
  uint32_t model_id = model->Id();
  uint32_t device_id = model->GetDeviceId();

  GELOGI("Model Run thread start, model_id:%u", model_id);
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
    GELOGI("Model thread Run begin, model id:%u, data index:%d.", model_id, current_data.index);

    GE_TIMESTAMP_START(Model_SyncVarData);
    ret = model->SyncVarData();
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ret != SUCCESS,
        (void)model->ReturnResult(model->model_id_, current_data.index, false, false, data_wrapper->GetOutput());
        CsaInteract::GetInstance().StoreInternalErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
        continue, "Copy input data to model failed.");  // [No need to check value]
    GE_TIMESTAMP_END(Model_SyncVarData, "Model Run SyncVarData");

    GELOGI("Copy input data, model id:%u", model_id);
    ret = model->CopyInputData(current_data, false);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ret != SUCCESS,
        (void)model->ReturnResult(model->model_id_, current_data.index, false, false, data_wrapper->GetOutput());
        CsaInteract::GetInstance().StoreInternalErrorCode(ret, ERROR_MODULE_FMK, JOBSUBSTATE_GRAPH_EXEC);
        continue, "Copy input data to model failed.");  // [No need to check value]

    if (ProfilingManager::Instance().ProfilingOpTraceOn()) {
      GELOGI("GetOpTraceIterNum:%d", ProfilingManager::Instance().GetOpTraceIterNum());
      for (int32_t i = 0; i < ProfilingManager::Instance().GetOpTraceIterNum(); i++) {
        if (!ProfilingManager::Instance().ProfilingLoadFlag()) {
          (void)ProfilingManager::Instance().StartProfiling(i);  // just profiling, no need to check value
        }
        // collect profiling for ge
        ProfilingManager::Instance().ReportProfilingData(model->GetTaskIdOpName());
        GELOGI("rtModelExecute start.");
        rtError_t rt_ret_prof_on = rtModelExecute(model->rt_model_handle_, model->rt_model_stream_, 0);
        GE_IF_BOOL_EXEC(rt_ret_prof_on != RT_ERROR_NONE, rslt_flg = false; (void)model->ReturnResult(
            model->model_id_, current_data.index, false, false, data_wrapper->GetOutput());
                        continue);  // [No need to check value]
        GELOGI("rtModelExecute end");

        GELOGI("rtStreamSynchronize start.");
        rt_ret_prof_on = rtStreamSynchronize(model->rt_model_stream_);
        GE_IF_BOOL_EXEC(rt_ret_prof_on != RT_ERROR_NONE, rslt_flg = false; (void)model->ReturnResult(
            model->model_id_, current_data.index, false, seq_end_flag, data_wrapper->GetOutput());
                        continue);  // [No need to check value]
        GELOGI("rtStreamSynchronize end.");
        ProfilingManager::Instance().StopProfiling();  // just profiling, no need to check value
      }
    } else {
      GE_TIMESTAMP_START(rtModelExecute);
      GELOGI("rtModelExecute start.");
      rtError_t rt_ret_prof_off = rtModelExecute(model->rt_model_handle_, model->rt_model_stream_, 0);
      GE_IF_BOOL_EXEC(
          rt_ret_prof_off != RT_ERROR_NONE, rslt_flg = false;
          (void)model->ReturnResult(model->model_id_, current_data.index, false, false, data_wrapper->GetOutput());
          CsaInteract::GetInstance().WriteErrorCode(rt_ret_prof_off, ERROR_MODULE_RUNTIME, JOBSUBSTATE_GRAPH_EXEC);
          continue);
      GELOGI("rtModelExecute end");
      GE_TIMESTAMP_END(rtModelExecute, "GraphExcute::rtModelExecute");

      GE_TIMESTAMP_START(rtStreamSynchronize);
      GELOGI("rtStreamSynchronize start.");
      rt_ret_prof_off = rtStreamSynchronize(model->rt_model_stream_);
      if (rt_ret_prof_off == RT_ERROR_END_OF_SEQUENCE) {
        seq_end_flag = true;
      }
      GE_IF_BOOL_EXEC(rt_ret_prof_off != RT_ERROR_NONE, rslt_flg = false; GELOGI("seq_end_flg: %d", seq_end_flag);
                      (void)model->ReturnResult(model->model_id_, current_data.index, false, seq_end_flag,
                                                data_wrapper->GetOutput());  // [No need to check value]
                      CsaInteract::GetInstance().StoreInternalErrorCode(rt_ret_prof_off, ERROR_MODULE_RUNTIME,
                                                                        JOBSUBSTATE_GRAPH_EXEC);
                      continue);
      GELOGI("rtStreamSynchronize end.");
      GE_TIMESTAMP_END(rtStreamSynchronize, "GraphExcute::Wait for rtStreamSynchronize");

      // collect profiling for ge
      if (ProfilingManager::Instance().ProfilingOn()) {
        ProfilingManager::Instance().ReportProfilingData(model->GetTaskIdOpName());
      }
    }

    GE_TIMESTAMP_START(ReturnResult3);
    // copy output data from device to host
    GE_IF_BOOL_EXEC(
        !model->output_op_list_.empty(),
        (void)model->ReturnResult(model->model_id_, current_data.index, rslt_flg, false, data_wrapper->GetOutput()))
    // copy output data from device to host for variable graph
    GE_IF_BOOL_EXEC(model->output_op_list_.empty(), (void)model->ReturnNoOutput(model->model_id_, current_data.index));
    GE_TIMESTAMP_END(ReturnResult3, "GraphExcute::CopyDataFromDeviceToHost");

    interator_count++;
    GELOGI("interator_count=%u", interator_count);
  }

  CsaInteract::GetInstance().WriteInternalErrorCode();
  GELOGI("Model run end, model id:%u", model->model_id_);
  GEEVENT("Model Run thread end, model_id:%u", model->model_id_);
  return nullptr;
}

///
/// @ingroup domi_ome
/// @brief call API provided by data inputer to destroy thread
/// @param [in] no
/// @return Status Destroy result
/// @author
///
Status DavinciModel::DestroyThread() {
  GE_CHK_BOOL_RET_STATUS(data_inputer_ != nullptr, INTERNAL_ERROR, "data_inputer_ is nullptr!");

  run_flg_ = false;

  data_inputer_->Stop();

  if (thread_id_.joinable()) {
    thread_id_.join();
  }

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief create model std::thread,
/// @brief start to execute Model
/// @param [in] no
/// @return Status create model thread and execute result
/// @author
///
Status DavinciModel::ModelRunStart() {
  GE_CHK_BOOL_RET_STATUS((DavinciModel::GetSysMode() != RESET) && (DavinciModel::GetSysMode() != STOP), INTERNAL_ERROR,
                         "Model Start FAIL in wrong sys mode!");

  GE_CHK_BOOL_RET_STATUS(data_inputer_ != nullptr, INTERNAL_ERROR, "data_inputer_ is nullptr!");

  LockRunFlg();
  GE_MAKE_GUARD(tmp_lock, [&] { UnlockRunFlg(); });

  GE_CHK_BOOL_RET_STATUS(!run_flg_, INTERNAL_ERROR, "Model already started!");

  run_flg_ = true;

  // create stream instance which rt_model_handel is running on
  GE_CHK_RT_RET(rtStreamCreate(&rt_model_stream_, priority_));
  is_inner_model_stream_ = true;

  string opt = "0";
  (void)ge::GetContext().GetOption("ge.maxDumpOpNum", opt);  // option may not be set up, no need to check value
  int64_t maxDumpOpNum = std::strtol(opt.c_str(), nullptr, kDecimal);
  maxDumpOpNum_ = maxDumpOpNum;

  CREATE_STD_THREAD(thread_id_, DavinciModel::Run, this);
  GELOGI("model tread create success, model id:%u", model_id_);
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief call API provided by data inputer and destroy model Thread
/// @param [in] no
/// @return Status Destroy result
/// @author
///
Status DavinciModel::ModelRunStop() {
  GE_CHK_BOOL_RET_STATUS((DavinciModel::GetSysMode() != RESET) && (DavinciModel::GetSysMode() != STOP), INTERNAL_ERROR,
                         "Model stop FAIL in wrong sys mode!");

  LockRunFlg();
  GE_MAKE_GUARD(tmp_lock, [&] { UnlockRunFlg(); });

  GE_IF_BOOL_EXEC(!run_flg_, return SUCCESS);

  GE_CHK_STATUS_RET(DestroyThread(), "DestoyThead failed!");

  return SUCCESS;
}

void DavinciModel::UnbindTaskSinkStream() {
  // unbinding hcom stream
  UnbindHcomStream();
  for (size_t i = 0; i < stream_list_.size(); i++) {
    // unbind rt_model_handle and streams
    GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, stream_list_[i]) != RT_ERROR_NONE,
               "Unbind stream from model failed! Index: %zu", i);
  }

  if (is_inner_model_stream_) {
    // destroy stream that is bound with rt_model
    GE_LOGW_IF(rtStreamDestroy(rt_model_stream_) != RT_ERROR_NONE, "Destroy stream for rt_model failed!")
  }
  return;
}

Status DavinciModel::InitTaskInfo(ModelTaskDef &model_task_def) {
  GELOGI("InitTaskInfo in,task size %zu", model_task_def.task().size());
  task_list_.resize(model_task_def.task_size());
  std::vector<std::future<Status>> futures(model_task_def.task_size());
  constexpr uint32_t thread_num = THREAD_NUM;
  ThreadPool executor(thread_num);
  rtContext_t ctx = nullptr;
  rtError_t rt_ret = rtCtxGetCurrent(&ctx);
  if (rt_ret != RT_ERROR_NONE || ctx == nullptr) {
    GELOGE(RT_FAILED, "Failed to get current context from rt, error-code 0x%X.", rt_ret);
    return RT_FAILED;
  }

  for (int32_t i = 0; i < model_task_def.task_size(); ++i) {
    futures[i] = executor.commit(
        [](const domi::TaskDef &task, DavinciModel *model, rtContext_t ctx, int32_t idx) -> Status {
          rtError_t ctx_ret = rtCtxSetCurrent(ctx);
          if (ctx_ret != RT_ERROR_NONE) {
            GELOGE(RT_FAILED, "Failed to set context from rt, error-code 0x%X.", ctx_ret);
            return RT_FAILED;
          }

          model->task_list_[idx] = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task.type()));

          Status ret = FAILED;
          if (model->task_list_[idx] != nullptr) {
            ret = model->task_list_[idx]->Init(task, model);
          }
          return ret;
        },
        model_task_def.task(i), this, ctx, i);
  }

  Status ret;
  for (size_t i = 0; i < futures.size(); ++i) {
    ret = futures[i].get();
    if (ret != SUCCESS) {
      GELOGE(ret, "Task index %zu init fail.", i);
      return ret;
    }
  }

  GELOGI("InitTaskInfo out");
  return SUCCESS;
}

Status DavinciModel::DistributeTask() {
  GELOGI("do Distribute.");

  op_task_id_map_.clear();
  Status ret;
  for (size_t task_index = 0; task_index < task_list_.size(); ++task_index) {
    auto &task = task_list_.at(task_index);
    if (task == nullptr) {
      GELOGW("task is null");
      continue;
    }
    ret = task->Distribute();
    if (ret != SUCCESS) {
      GELOGE(ret, "Distribute Fail!");
      return ret;
    }

    // for data dump
    if (reinterpret_cast<void *>(task->GetDumpArgs()) != nullptr) {
      auto op_index = std::max(model_task_def_->task(task_index).kernel().context().op_index(),
                               model_task_def_->task(task_index).kernel_ex().op_index());
      OpDescPtr op = GetOpByIndex(op_index);
      if (op == nullptr) {
        GELOGE(PARAM_INVALID, "Op index %u is null, op list size %zu.", op_index, op_list_.size());
        return PARAM_INVALID;
      }

      if (PropertiesManager::Instance().IsLayerNeedDump(name_, op->GetName())) {
        data_dumper_.SaveDumpTask(task->GetTaskID(), op, task->GetDumpArgs());
      }
    }

    // get op_name by task_index
    if (task->GetCtx() != nullptr) {
      auto iter = op_name_map_.find(task_index);
      if (iter == op_name_map_.end()) {
        continue;
      }

      // else task index is found in op_name_map_
      string op_name = op_name_map_[task_index];
      op_task_id_map_[task->GetTaskID()] = op_name;
    }
  }

  // launch dump kernel to aicpu
  ret = data_dumper_.LoadDumpInfo();
  if (ret != SUCCESS) {
    GELOGE(ret, "Load dump info fail.");
    return ret;
  }

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief Save Data and NetOutput address info for ZeroCopy.
/// @param [in] const std::vector<void *> &outside_addrs
/// @return None.
///
void DavinciModel::SetOutsideAddr(const std::vector<void *> &outside_addrs) {
  for (auto addr : outside_addrs) {
    if (outside_addrs_.find(addr) != outside_addrs_.end()) {
      continue;
    }

    (void)outside_addrs_.emplace(std::pair<const void *, std::vector<void *>>(addr, {}));
    GELOGI("SetOutsideAddr success.");
  }
}

///
/// @ingroup domi_ome
/// @brief Save outside address used info for ZeroCopy.
/// @param [in] const std::vector<void *> &outside_addrs: address of task
/// @param [in] const char *args_offset: arguments address save the address.
/// @return None.
///
void DavinciModel::SetZeroCopyAddr(const std::vector<void *> &outside_addrs, void *args_offset) {
  size_t nums = outside_addrs.size();
  for (size_t i = 0; i < nums; ++i) {
    std::lock_guard<std::mutex> lock(outside_addrs_mutex_);
    auto it = outside_addrs_.find(outside_addrs[i]);
    if (it == outside_addrs_.end()) {
      continue;
    }

    it->second.push_back(static_cast<char *>(args_offset) + i * sizeof(void *));
    GELOGI("SetZeroCopyAddr of outside_addrs.");
  }
}

///
/// @ingroup domi_ome
/// @brief Copy Inputs and Outputs addr to model for direct use.
/// @param [in] const domi::InputData &input_data: model input data.
/// @param [in] domi::OutputData &output_data: model output data.
/// @return SUCCESS handle successfully / PARAM_INVALID for failed
///
Status DavinciModel::ModelZeroCopy(const InputData &input_data, OutputData &output_data) {
  if (ZeroCopyInput(input_data) != SUCCESS) {
    GELOGE(PARAM_INVALID, "ZeroCopyInput failed.");
    return PARAM_INVALID;
  }

  if (ZeroCopyOutput(output_data) != SUCCESS) {
    GELOGE(PARAM_INVALID, "ZeroCopyOutput failed.");
    return PARAM_INVALID;
  }

  output_data.index = input_data.index;
  output_data.model_id = model_id_;
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief Copy Data addr to model for direct use.
/// @param [in] const domi::InputData &input_data: model input data info.
/// @return SUCCESS handle successfully / others handle failed
///
Status DavinciModel::ZeroCopyInput(const InputData &input_data) {
  GE_CHK_BOOL_RET_STATUS(!data_op_list_.empty(), SUCCESS, "data_op_list_ is empty!");
  GE_CHK_BOOL_RET_STATUS(data_op_list_.size() == input_data.blobs.size(), PARAM_INVALID,
                         "The input data list size (%zu) does not match the model input list size (%zu)",
                         input_data.blobs.size(), data_op_list_.size());

  const std::vector<DataBuffer> &blobs = input_data.blobs;
  for (size_t data_op_index = 0; data_op_index < data_op_list_.size(); ++data_op_index) {
    auto op_desc = data_op_list_[data_op_index];
    GE_CHK_BOOL_EXEC(op_desc != nullptr, return PARAM_INVALID, "op_desc is null!");

    auto data_index = static_cast<uint32_t>(data_op_index);
    if (AttrUtils::GetInt(op_desc, "index", data_index)) {
      GELOGI("ge_train:get new index %u , old %zu", data_index, data_op_index);
    }
    GE_CHK_BOOL_EXEC(data_index < blobs.size(), return PARAM_INVALID, "index:%u >= size:%zu", data_index, blobs.size());
    GE_CHK_BOOL_RET_STATUS(op_desc->GetInputsSize() == 1 && op_desc->GetOutputsSize() == 1, PARAM_INVALID,
                           "Data Op has invalid input_desc_size(%zu) or output_desc_size(%zu)",
                           op_desc->GetInputsSize(), op_desc->GetOutputsSize());

    uint32_t input_size = 0;
    const DataBuffer &data_buf = blobs[data_index];
    GE_CHK_STATUS(TensorUtils::GetSize(*op_desc->GetInputDescPtr(0), input_size), "get input size failed.");
    GE_CHK_BOOL_RET_STATUS(input_size >= data_buf.length, PARAM_INVALID,
                           "input data size(%u) does not match model required size(%u), ret fail.", data_buf.length,
                           input_size);

    const vector<void *> &outputs = ModelUtils::GetOutputDataAddrs(runtime_param_, op_desc);
    if (data_buf.data == nullptr) {
      GELOGE(INTERNAL_ERROR, "data_buf.data is nullptr");
      return INTERNAL_ERROR;
    }
    if (!outputs.empty() && ZeroCopyImpl(outputs[0], data_buf) != SUCCESS) {
      return FAILED;
    }
  }

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief Copy NetOutput addr to model for direct use.
/// @param [in] const domi::OutputData &output_data: model output data info.
/// @return SUCCESS handle successfully / others handle failed
///
Status DavinciModel::ZeroCopyOutput(const OutputData &output_data) {
  GE_CHK_BOOL_RET_STATUS(output_data.blobs.size() == output_size_list_.size(), INTERNAL_ERROR,
                         "output buffer size[%zu] not equal output_size_list[%zu] size!", output_data.blobs.size(),
                         output_size_list_.size());

  // index of data in output_data
  uint32_t output_data_index = 0;
  const std::vector<DataBuffer> &blobs = output_data.blobs;
  for (auto &op_desc : output_op_list_) {
    Output model_output(op_desc, this);
    GE_CHK_BOOL_RET_STATUS(model_output.Init() == SUCCESS, PARAM_INVALID, "init model_output failed");
    vector<uint32_t> v_output_size = ModelUtils::GetInputSize(op_desc);
    vector<void *> v_output_data_addr = ModelUtils::GetInputDataAddrs(runtime_param_, op_desc);

    // for all output tensor, copy output data from op to designated position
    for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
      GE_CHK_BOOL_RET_STATUS(output_data_index < blobs.size(), PARAM_INVALID,
                             "The blobs size:%zu, data_op size:%zu, curr output size:%zu", blobs.size(),
                             data_op_list_.size(), op_desc->GetOutputsSize());
      const DataBuffer &data_buf = blobs[output_data_index];
      output_data_index++;
      uint32_t size = data_buf.length;
      GE_CHK_BOOL_RET_STATUS(size <= v_output_size[i], PARAM_INVALID,
                             "Model output data size(%u) does not match required size(%u).", v_output_size[i],
                             data_buf.length);

      GELOGI("ZeroCopyOutput memcpy graph_%u type[F] name[%s] output[%lu] memsize[%u] datasize[%u]",
             runtime_param_.graph_id, op_desc->GetName().c_str(), i, data_buf.length, v_output_size[i]);
      if (ZeroCopyImpl(v_output_data_addr[i], data_buf) != SUCCESS) {
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief Copy address to args_ space for direct use.
/// @param [in] const void *src_addr: source address of the Op.
/// @param [in] const void *dst_addr: destination address of user data.
/// @return SUCCESS handle successfully / others handle failed
///
Status DavinciModel::ZeroCopyImpl(const void *src_addr, const DataBuffer &data_buf) {
  auto it = outside_addrs_.find(src_addr);
  if (it == outside_addrs_.end()) {
    GELOGE(FAILED, "ZeroCopyImpl failed to find outside_addrs.");
    return FAILED;
  }

  auto dst_addr = static_cast<uint8_t *>(data_buf.data);
  auto dst_size = static_cast<uint64_t>(data_buf.length);
  Status ret = ModelUtils::ConvertVirtualAddressToPhysical(dst_addr, dst_size, dst_addr);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Convert virtual address to physical for dst_addr failed.");
    return FAILED;
  }

  for (auto &addr : it->second) {
    __builtin_prefetch(addr);
    rtError_t rt_err = rtMemcpy(addr, sizeof(void *), &dst_addr, sizeof(void *), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_err != RT_ERROR_NONE) {
      GELOGE(FAILED, "ZeroCopyImpl: rtMemcpy failed");
      return FAILED;
    }
  }

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief get unique identification for op when load two or more models
/// @param [in] const OpDescPtr: current op.
/// @param [in] string identification: unique identification for current op.
/// @return SUCCESS handle successfully / others handle failed
///
void DavinciModel::GetUniqueId(const OpDescPtr &op_desc, std::string &unique_identification) {
  std::string session_graph_id;
  GE_IF_BOOL_EXEC(AttrUtils::GetStr(*op_desc, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                  GELOGI("Get original type of session_graph_id."));
  if (session_graph_id.empty()) {
    return;
  } else if (session_graph_id.find("-1") != string::npos) {
    unique_identification = session_graph_id + "_" + to_string(model_id_);
  } else {
    unique_identification = session_graph_id;
  }
}

///
/// @ingroup domi_ome
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
  std::lock_guard<std::mutex> lock(tvm_bin_mutex_);
  auto it = tvm_bin_kernel_.find(binfile_key);
  if (it != tvm_bin_kernel_.end()) {
    return it->c_str();
  } else {
    it = tvm_bin_kernel_.insert(tvm_bin_kernel_.end(), binfile_key);
    return it->c_str();
  }
}

///
/// @ingroup domi_ome
/// @brief Constant Op Init.
/// @return Status
///
Status DavinciModel::InitConstant(const ConstOpDescPtr &op_desc) const {
  auto v_weights = ModelUtils::GetWeights(op_desc);
  auto v_output_size = ModelUtils::GetOutputSize(op_desc);
  auto v_output_addr = ModelUtils::GetOutputDataAddrs(runtime_param_, op_desc);
  GE_IF_BOOL_EXEC(v_weights.empty() || v_output_size.empty() || v_output_addr.empty(),
                  GELOGE(PARAM_INVALID, "const op:%s not set output", op_desc->GetName().c_str());
                  return PARAM_INVALID;);

  GeTensor *tensor = const_cast<GeTensor *>(v_weights[0].get());
  GE_IF_BOOL_EXEC(v_output_size[0] < tensor->GetData().size(),
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
    int64_t elem_num = tensor_shape.GetShapeSize() == 0 ? 1 : tensor_shape.GetShapeSize();
    uint64_t *buff = reinterpret_cast<uint64_t *>(tensor->MutableData().data());
    GE_CHK_BOOL_RET_STATUS(CheckInt64Uint32MulOverflow(elem_num, kBytes) == SUCCESS, FAILED, "Shape size is invalid");
    int64_t offset = elem_num * kBytes;

    uint64_t hbm_raw_data_base_addr = reinterpret_cast<uint64_t>(v_output_addr[0]) + offset;
    for (int64_t i = elem_num - 1; i >= 0; --i) {
      buff[i] = hbm_raw_data_base_addr + (buff[i] - buff[0]);
    }
  }
  GE_CHK_RT_RET(rtMemcpy(v_output_addr[0], v_output_size[0], tensor->GetData().data(), tensor->GetData().size(),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief TVM Op Init.
/// @return Status
///

Status DavinciModel::InitTbeHandle(const OpDescPtr &op_desc) {
  TBEKernelPtr tbe_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
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

  // Need protection of tvm_bin_mutex_.
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

  std::lock_guard<std::mutex> lock(tvm_bin_mutex_);
  kernel_store.EraseTBEHandle(used_tbe_handle_map_);
  used_tbe_handle_map_.clear();
}

///
/// @ingroup domi_ome
/// @brief insert active_stream_indication_
/// @return Status
///
Status DavinciModel::MarkActiveStream(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc);
  std::string type = op_desc->GetType();
  GE_IF_BOOL_EXEC(
      type == STREAMSWITCH, std::vector<uint32_t> active_stream_list;
      GE_LOGI_IF(!ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list),
                 "GetInt ACTIVE_STREAM_LIST fail.");
      if (active_stream_list.size() != TRUE_BRANCH_STREAM_NUM) {
        GELOGE(INTERNAL_ERROR, "Stream num of switch true branch must be %u.", TRUE_BRANCH_STREAM_NUM);
        return INTERNAL_ERROR;
      } uint32_t true_stream_id = active_stream_list.front();
      active_stream_indication_.insert(true_stream_id);
      GELOGI("flowctrl_op_index_map  node:%s, true_stream_id=%u.", op_desc->GetName().c_str(), true_stream_id););
  GE_IF_BOOL_EXEC(type == STREAMACTIVE, if (op_desc->HasAttr(ATTR_NAME_SWITCH_BRANCH_NODE_LABEL)) {
    std::vector<uint32_t> active_stream_list;
    GE_CHK_BOOL_EXEC(AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list),
                     return INTERNAL_ERROR, "StreamActiveOp get attr ACTIVE_STREAM fail.");

    for (size_t j = 0; j < active_stream_list.size(); ++j) {
      active_stream_indication_.insert(active_stream_list[j]);
      GELOGI("flowctrl_op_index_map  node:%s, active_stream_id=%u.", op_desc->GetName().c_str(), active_stream_list[j]);
    }
  });
  return SUCCESS;
}

bool DavinciModel::IsBroadCastOpData(const ge::NodePtr &var_node) {
  for (const auto &out_anchor : var_node->GetAllOutDataAnchors()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_anchor);
    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
      ge::NodePtr dst_node = in_anchor->GetOwnerNode();
      GE_RT_FALSE_CHECK_NOTNULL(dst_node);
      if (dst_node->GetType() == HCOMBROADCAST) {
        return true;
      }
    }
  }
  return false;
}

///
/// @ingroup domi_ome
/// @brief Init model stream for NN model.
/// @param [in] stream   user input model stream.
/// @param [in] async_mode  is asynchronize mode.
/// @return Status
///
Status DavinciModel::InitModelStream(rtStream_t stream, bool async_mode) {
  // asynchronize mode, use user input stream.
  if (async_mode) {
    rt_model_stream_ = stream;
    is_inner_model_stream_ = false;
    return SUCCESS;
  }

  // synchronize mode, use forbidden stream.
  if (stream != nullptr) {
    if ((rt_model_stream_ != nullptr) && is_inner_model_stream_) {
      if (rtStreamDestroy(rt_model_stream_) != RT_ERROR_NONE) {
        GELOGE(RT_FAILED, "Destroy rt_stream failed!");
        return FAILED;
      }
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
/// @ingroup domi_ome
/// @brief ACL case, do not start  new thread, return execute result.
/// @param [in] stream   execute model stream.
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  model input data.
/// @param [out] output_data  model output data.
///
Status DavinciModel::NnExecute(rtStream_t stream, bool async_mode, const InputData &input_data,
                               OutputData &output_data) {
  GELOGI("Model Run begin, model id:%u, data index:%d, flag:%d.", model_id_, input_data.index, async_mode);
  GE_CHK_STATUS(InitModelStream(stream, async_mode), "Init model stream fail.");

  GELOGI("do rtModelExecute task sink, model id:%u", input_data.model_id);
  Status ret = ModelZeroCopy(input_data, output_data);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return INTERNAL_ERROR, "Copy input data to model failed.");

  GELOGI("current_data.index=%u", input_data.index);

  GELOGD("rtModelExecute do");

  rtError_t rt_ret = rtModelExecute(rt_model_handle_, rt_model_stream_, 0);
  GE_CHK_RT_EXEC(rt_ret, return INTERNAL_ERROR);
  GELOGI("rtModelExecute end");

  if (async_mode) {
    rt_ret = rtStreamSynchronize(rt_model_stream_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, return INTERNAL_ERROR);
  }

  ret = SyncDataAndDump();
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return INTERNAL_ERROR, "Copy Output data to user failed.");

  // collect profiling for ge
  if (ProfilingManager::Instance().ProfilingOn()) {
    ProfilingManager::Instance().ReportProfilingData(op_task_id_map_);
    GELOGI("Acl Profiling Op name taskId report.");
  }

  GELOGI("Model run end, model id:%u", model_id_);
  GEEVENT("Model Run thread end, model_id:%u", model_id_);
  return SUCCESS;
}

uint8_t *DavinciModel::MallocFeatureMapMem(uint64_t data_size) {
  uint8_t *mem_base = nullptr;
  if (std::getenv(kEnvGeuseStaticMemory) != nullptr) {
    data_size = static_cast<uint64_t>(VarManager::Instance(0)->GetGraphMemoryMaxSize());
    string memory_key = std::to_string(0) + "_f";
    mem_base = MemManager::Instance(RT_MEMORY_HBM)->MallocMemory(memory_key, data_size, GetDeviceId());
  } else {
    mem_base = MemManager::Instance(RT_MEMORY_HBM)->MallocMemory(data_size, GetDeviceId());
  }

  if (mem_base != nullptr) {
    GE_CHK_RT(rtMemset(mem_base, data_size, 0U, data_size));
  }
  return mem_base;
}

uint8_t *DavinciModel::MallocWeightsMem(uint32_t weights_size) {
  uint8_t *weights_mem_base = nullptr;
  if (std::getenv(kEnvGeuseStaticMemory) != nullptr) {
    string weight_memory_key = std::to_string(0) + "_w";
    weights_mem_base =
        MemManager::Instance(RT_MEMORY_HBM)->MallocMemory(weight_memory_key, weights_size, GetDeviceId());
  } else {
    weights_mem_base = MemManager::Instance(RT_MEMORY_HBM)->MallocMemory(weights_size, GetDeviceId());
  }
  return weights_mem_base;
}

void DavinciModel::FreeFeatureMapMem() {
  if (std::getenv(kEnvGeuseStaticMemory) != nullptr) {
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

uint32_t DavinciModel::GetGraphID(const std::string &session_graph_id) {
  std::string session_id = "_";
  auto pos = session_graph_id.find(session_id);
  if (pos != std::string::npos) {
    size_t graph_id_length = session_graph_id.length() - pos - session_id.length();
    std::string graph_id = session_graph_id.substr(pos + session_id.length(), graph_id_length);
    return static_cast<uint32_t>(std::strtol(graph_id.c_str(), nullptr, kDecimal));
  }
  return 0;
}

Status DavinciModel::TransAllVarData(ComputeGraphPtr &graph, uint32_t graph_id) {
  GELOGI("TransAllVarData start: session_id:%lu, graph_id: %u.", session_id_, graph_id);

  ThreadPool executor(THREAD_NUM);
  std::vector<std::future<Status>> vector_future;

  rtContext_t ctx = nullptr;
  rtError_t rt_ret = rtCtxGetCurrent(&ctx);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Failed to get current context, error_code is: 0x%X.", rt_ret);
    return RT_FAILED;
  }

  for (ge::NodePtr &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() != VARIABLE) {
      continue;
    }
    vector_future.push_back(executor.commit(
        [](ge::NodePtr &node, DavinciModel *model, rtContext_t ctx, uint32_t graph_id) -> Status {
          if (model == nullptr) {
            GELOGE(FAILED, "DavinciModel is NULL!");
            return FAILED;
          }
          rtError_t rt_ret = rtCtxSetCurrent(ctx);
          if (rt_ret != RT_ERROR_NONE) {
            GELOGE(RT_FAILED, "Failed to set context, error_code is: 0x%X.", rt_ret);
            return RT_FAILED;
          }
          uint32_t allocated_graph_id = 0;
          Status ret =
              VarManager::Instance(model->session_id_)->GetAllocatedGraphId(node->GetName(), allocated_graph_id);
          if (ret != SUCCESS) {
            GELOGE(INTERNAL_ERROR, "var has not been allocated, node:%s, graph_id:%u.", node->GetName().c_str(),
                   graph_id);
            return INTERNAL_ERROR;
          }
          uint32_t changed_graph_id = 0;
          ret = VarManager::Instance(model->session_id_)->GetChangedGraphId(node->GetName(), changed_graph_id);
          bool call_trans_var =
              (ret == SUCCESS && changed_graph_id == graph_id && changed_graph_id != allocated_graph_id);
          if (call_trans_var) {
            GELOGI("VarManager::GetChangedGraphId() success, node:%s, graph_id:%u.", node->GetName().c_str(), graph_id);
            VarTransRoad *trans_road = VarManager::Instance(model->session_id_)->GetTransRoad(node->GetName());
            if (trans_road == nullptr) {
              GELOGI("The variable %s does not have any trans road", node->GetName().c_str());
              return SUCCESS;
            }
            ret = TransVarData(node, *trans_road, model->session_id_, model->device_id_);
            if (ret != SUCCESS) {
              GELOGE(INTERNAL_ERROR, "TransVarData failed, node:%s, graph_id:%u.", node->GetName().c_str(), graph_id);
              return INTERNAL_ERROR;
            }
            VarManager::Instance(model->session_id_)->RemoveChangedGraphId(node->GetName());
          }
          return SUCCESS;
        },
        node, this, ctx, graph_id));
  }

  Status ret_status;
  for (size_t i = 0; i < vector_future.size(); ++i) {
    ret_status = vector_future[i].get();
    if (ret_status != SUCCESS) {
      GELOGE(ret_status, "TransAllVarData:: trans %zu vardata failed", i);
      return ret_status;
    }
  }

  GELOGI("TransAllVarData success.");

  return SUCCESS;
}

void DavinciModel::InitDataDumper() {
  GELOGI("data dumper init, name: %s, id: %u.", name_.c_str(), model_id_);
  data_dumper_.SetModelName(name_);
  data_dumper_.SetModelId(model_id_);
  data_dumper_.SetMemory(runtime_param_);

  int32_t device_id = 0;
  rtError_t rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE || device_id < 0) {
    GELOGE(RT_FAILED, "Call rtGetDevice fail, ret = 0x%X, device_id = %d.", rt_ret, device_id);
    return;
  }
  data_dumper_.SetDeviceId(device_id);
  GELOGI("InitDataDumper end.");
}

uint32_t DavinciModel::GetFlowctrlIndex(uint32_t op_index) {
  std::lock_guard<std::mutex> lock(flowctrl_op_index_internal_map_mutex_);
  return (++flowctrl_op_index_internal_map_[op_index]) - 1;
}

void DavinciModel::PushHcclStream(rtStream_t value) {
  std::lock_guard<std::mutex> lock(all_hccl_stream_list_mutex_);
  all_hccl_stream_list_.push_back(value);
}

Status TransTensor(uint8_t *var_data, const NodePtr &var_src, const NodePtr &var_dst, formats::TransResult &result) {
  GE_CHECK_NOTNULL(var_src);
  GE_CHECK_NOTNULL(var_src->GetOpDesc());
  GE_CHECK_NOTNULL(var_dst);
  GE_CHECK_NOTNULL(var_dst->GetOpDesc());
  auto src_data_shape_size = var_src->GetOpDesc()->GetOutputDesc(0).GetShape().GetShapeSize();
  auto src_data_datatype = var_src->GetOpDesc()->GetOutputDesc(0).GetDataType();
  auto dst_data_datatype = var_dst->GetOpDesc()->GetOutputDesc(0).GetDataType();
  GE_IF_BOOL_EXEC(
      src_data_datatype != dst_data_datatype,
      auto ret = formats::TransDataType(
          {var_data, static_cast<size_t>(src_data_shape_size), src_data_datatype, dst_data_datatype}, result);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "trans var data on host failed");
        return ret;
      });
  return SUCCESS;
}

Status DavinciModel::CopyTensorFromSrcVarNode(const NodePtr &var_src, const NodePtr &var_dst) {
  /// after FE fusion pass, input num of applymomentum op was changed, 0th input is var_fp32, 6th input is
  /// var_fp16(new).
  /// unlink edges between var_fp32 and "dst_node" (need fp16) of var_fp32, add edge between var_fp16 and dst_node.
  /// need copy value from var_fp32 to var_fp16.
  /// [opdesc of var_src and var_dst are checked before passed in, no need to check if they are nullptr]
  GE_IF_BOOL_EXEC(var_src == nullptr || var_dst == nullptr, GELOGE(FAILED, "node var is nullptr"); return FAILED);
  // src_node output_desc (fp32)
  GeTensorDesc output_desc = var_src->GetOpDesc()->GetOutputDesc(0);
  auto src_data_type = output_desc.GetDataType();
  auto src_shape = output_desc.GetShape();
  auto src_format = output_desc.GetFormat();
  GELOGI("src_node %s, src_format %s, src_shape %s, src_type %s", var_src->GetName().c_str(),
         TypeUtils::FormatToSerialString(src_format).c_str(), formats::ShapeToString(src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(src_data_type).c_str());
  // dst_node output_desc (fp16)
  GeTensorDesc dst_tensor_desc = var_dst->GetOpDesc()->GetOutputDesc(0);
  auto data_type = dst_tensor_desc.GetDataType();
  auto data_shape = dst_tensor_desc.GetShape();
  auto data_format = dst_tensor_desc.GetFormat();
  GELOGI("dst_node %s, src_format %s, src_shape %s, src_type %s", var_dst->GetName().c_str(),
         TypeUtils::FormatToSerialString(data_format).c_str(), formats::ShapeToString(data_shape).c_str(),
         TypeUtils::DataTypeToSerialString(data_type).c_str());
  // Sync var data from device
  std::unique_ptr<uint8_t[]> var_src_data;
  RtContextSwitchGuard switch_context(RT_CTX_NORMAL_MODE, device_id_);
  // copy from src_node
  auto ret = CopyVarFromDevice(session_id_, var_src, var_src_data, output_desc);
  GE_IF_BOOL_EXEC(ret != SUCCESS, GELOGE(FAILED, "Copy Var From Device failed"); return ret);
  // trans dtype
  formats::TransResult trans_result;
  ret = TransTensor(var_src_data.get(), var_src, var_dst, trans_result);
  GE_IF_BOOL_EXEC(ret != SUCCESS, GELOGE(INTERNAL_ERROR, "trans var data on host failed"); return ret);
  // reset src value.
  void *var_device = nullptr;
  ret = ReAssignVarAddr(session_id_, var_dst->GetName(), dst_tensor_desc, &var_device);
  GE_IF_BOOL_EXEC(ret != SUCCESS, GELOGE(INTERNAL_ERROR, "assign mem failed"); return ret);
  // copy to device
  ret = CopyVarToDevice(var_dst, trans_result, var_device);
  GE_IF_BOOL_EXEC(ret != SUCCESS, GELOGE(ret, "Failed to send var data to device"); return ret);
  return SUCCESS;
}

Status DavinciModel::CopyVarData(ComputeGraphPtr &compute_graph) {
  GELOGI("CopyVarData start: session_id:%lu.", session_id_);
  if (compute_graph == nullptr) {
    GELOGE(FAILED, "compute_graph is nullptr");
    return FAILED;
  }

  string cp_from_node;
  bool copy_value = false;
  for (auto &node : compute_graph->GetAllNodes()) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr || node->GetOpDesc()->GetType() != VARIABLE, continue);
    GE_IF_BOOL_EXEC(ge::AttrUtils::GetStr(node->GetOpDesc(), "_copy_from_var_node", cp_from_node),
                    GELOGI("Get original type of cp_from_node"));
    if (cp_from_node.length() != 0) {
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_copy_value", copy_value);
      if (!copy_value) {
        auto src_node = compute_graph->FindNode(cp_from_node);
        GE_CHECK_NOTNULL(src_node);
        GELOGI("current_var_node__: [%s] copy_from_var_node__: [%s].", node->GetName().c_str(),
               src_node->GetName().c_str());
        auto ret = CopyTensorFromSrcVarNode(src_node, node);
        GE_IF_BOOL_EXEC(ret != SUCCESS, GELOGE(FAILED, "copy tensor failed!"); return FAILED);
        // only copy once
        (void)ge::AttrUtils::SetBool(node->GetOpDesc(), "_copy_value", true);
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
