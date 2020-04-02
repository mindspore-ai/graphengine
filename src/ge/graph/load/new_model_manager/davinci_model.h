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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/ge_types.h"
#include "common/types.h"
#include "graph/load/new_model_manager/data_inputer.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "proto/task.pb.h"
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "common/opskernel/ge_task_info.h"
#include "framework/common/util.h"
#include "graph/model.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/utils/tensor_utils.h"
#include "common/helper/model_helper.h"
#include "common/helper/om_file_helper.h"
#include "graph/load/new_model_manager/data_dumper.h"
#include "graph/node.h"
#include "graph/utils/attr_utils.h"
#include "task_info/task_info.h"

#define WEIGHTS_ADDR_TO_CCE(var)

namespace ge {
using std::vector;
const uint32_t MEM_ALIGN_SIZE = 512;

// comments
class DavinciModel {
 public:
  ///
  /// @ingroup domi_ome
  /// @brief DavinciModel constructor
  /// @author
  ///
  DavinciModel(int32_t priority, const std::shared_ptr<ModelListener> &listener);

  ///
  /// @ingroup domi_ome
  /// @brief DavinciModel desctructor, free Parse and Init resources
  /// @author
  ///
  ~DavinciModel();

  ///
  /// @ingroup domi_ome
  /// @brief apply model to model_def_
  ///
  Status Assign(const GeModelPtr &ge_model);

  ///
  /// @ingroup domi_ome
  /// @brief DavinciModel initialization, including Stream, ccHandle, Event, DataInputer, etc
  /// @return execute result
  /// @author
  ///
  Status Init(void *dev_ptr = nullptr, size_t memsize = 0, void *weight_ptr = nullptr, size_t weightsize = 0);

  ///
  /// @ingroup ge
  /// @brief ACL case, Load task list with queue.
  /// @param [in] input_que_ids: input queue ids from user, nums equal Data Op.
  /// @param [in] output_que_ids: input queue ids from user, nums equal NetOutput Op.
  /// @return: 0 for success / others for fail
  ///
  Status SetQueIds(const std::vector<uint32_t> &input_queue_ids, const std::vector<uint32_t> &output_queue_ids);

  ///
  /// @ingroup domi_ome
  /// @brief Get DataInputer
  /// @return model ID
  ///
  uint32_t Id() const { return model_id_; }

  ///
  /// @ingroup domi_ome
  /// @brief Get DataInputer
  /// @return model ID
  ///
  void SetId(uint32_t model_id) { model_id_ = model_id; }

  static void *Run(DavinciModel *model_pointer);

  ///
  /// @ingroup domi_ome
  /// @brief NnExecute
  /// @param [in] stream   execute stream
  /// @param [in] async_mode  is asynchronize mode.
  /// @param [in] input_data  model input data
  /// @param [out] output_data  model output data
  ///
  Status NnExecute(rtStream_t stream, bool async_mode, const InputData &input_data, OutputData &output_data);

  ///
  /// @ingroup domi_ome
  /// @brief get sys mode
  /// @return SysMode
  ///
  static SysMode GetSysMode();

  ///
  /// @ingroup domi_ome
  /// @brief set sys mode
  /// @return Status
  ///
  static Status SetSysMode(SysMode mode);

  ///
  /// @ingroup domi_ome
  /// @brief lock mutex run flag
  /// @author
  ///
  void LockRunFlg() { mux_run_flg_.lock(); }

  ///
  /// @ingroup domi_ome
  /// @brief unlock mutex run flag
  /// @author
  ///
  void UnlockRunFlg() { mux_run_flg_.unlock(); }

  ///
  /// @ingroup domi_ome
  /// @brief get DataInputer
  /// @return DataInputer pointer
  ///
  DataInputer *const GetDataInputer() const { return data_inputer_; }

  // get Stream number
  uint32_t StreamNum() const { return runtime_param_.stream_num; }

  // get Event number
  uint32_t EventNum() const { return runtime_param_.event_num; }

  // get batch number
  uint32_t BatchNum() const { return runtime_param_.batch_num; }

  // get session id
  uint64_t SessionId() const { return runtime_param_.session_id; }

  vector<ge::OpDescPtr> GetOpDesc() {
    vector<ge::OpDescPtr> opDescVector;
    GE_IF_BOOL_EXEC(ge::AttrUtils::GetListOpDesc(GetGeModel(), MODEL_ATTR_FUSION_MODEL_DEF, opDescVector),
                    GELOGI("get opDesc of opDescVector"));
    return opDescVector;
  }

  // get model priority
  int32_t Priority() const { return priority_; }

  // get total mem size
  size_t TotalMemSize() const { return runtime_param_.mem_size; }

  // model name
  string Name() { return name_; }

  // version
  uint32_t Version() const { return version_; }

  // get total weights mem size
  size_t TotalWeightsMemSize() const { return runtime_param_.weight_size; }

  size_t TotalVarMemSize() const { return runtime_param_.var_size; }

  // get base memory address
  uint8_t *MemBase() { return mem_base_; }

  // get weight base memory address
  uint8_t *WeightsMemBase() { return weights_mem_base_; }

  uint8_t *VarMemBase() { return var_mem_base_; }

  // get Event list
  const vector<rtEvent_t> &GetEventList() const { return event_list_; }

  const vector<rtStream_t> &GetStreamList() const { return stream_list_; }

  const vector<rtLabel_t> &GetLabelList() const { return label_list_; }

  Status DestroyThread();

  // get Op
  map<uint32_t, OpDescPtr> GetOpList() const { return op_list_; }

  OpDescPtr GetOpByIndex(uint32_t index) {
    if (op_list_.find(index) == op_list_.end()) {
      return nullptr;
    }
    return op_list_.at(index);
  }

  OpDescPtr GetVariableOp(const string &name) {
    for (auto op_desc : variable_op_list_) {
      if (op_desc != nullptr && op_desc->GetName() == name) {
        return op_desc;
      }
    }
    return nullptr;
  }
  // get taskid to op name
  const map<uint32_t, std::string> &GetTaskIdOpName() const { return op_task_id_map_; }

  // get updated task info list
  std::vector<TaskInfoPtr> GetTaskList() { return task_list_; }

  ///
  /// @ingroup domi_ome
  /// @brief get model input and output format
  /// @return ccTensorFormat_t current model input and output format
  ///
  ge::Format GetFormat();

  rtModel_t GetRtModelHandle() {
    rtModel_t res = rt_model_handle_;
    return res;
  }

  uint64_t GetRtBaseAddr() const { return runtime_param_.logic_mem_base; }

  uint64_t GetRtWeightAddr() const { return runtime_param_.logic_weight_base; }

  uint64_t GetRtVarAddr() const { return runtime_param_.logic_var_base; }

  uint32_t GetFlowctrlIndex(uint32_t op_index);

  void PushHcclStream(rtStream_t value);

  bool IsBroadCastOpData(const ge::NodePtr &var_node);

  ///
  /// @ingroup domi_ome
  /// @brief For TVM Op, avoid Addr Reuse.
  /// @return void*
  ///
  static const char *GetRegisterStub(const string &tvm_binfile_key, const string &session_graph_model_id = "");

  ///
  /// @ingroup domi_ome
  /// @brief get model input and output desc info
  /// @param [out] input_shape  model input size
  /// @param [out] output_shape model output size
  /// @return execute result
  ///
  Status GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc, vector<InputOutputDescInfo> &output_desc);

  Status GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc, vector<InputOutputDescInfo> &output_desc,
                                std::vector<uint32_t> &inputFormats, std::vector<uint32_t> &output_formats);

  ///
  /// @ingroup domi_ome
  /// @brief Get model_id.
  /// @return model_id
  ///
  uint32_t GetModelId() const { return model_id_; }

  ///
  /// @ingroup domi_ome
  /// @brief get unique identification for op when load two or more models
  /// @param [in] op_desc : current op.
  /// @param [in] string identification: unique identification for current op.
  /// @return None
  ///
  void GetUniqueId(const OpDescPtr &op_desc, std::string &unique_identification);

  ///
  /// @ingroup domi_ome
  /// @brief get model input and output desc for zero copy
  /// @param [out] input_shape  model input size
  /// @param [out] output_shape model output size
  /// @return execute result
  ///
  Status GetInputOutputDescInfoForZeroCopy(vector<InputOutputDescInfo> &input_desc,
                                           vector<InputOutputDescInfo> &output_desc);

  Status GetInputOutputDescInfoForZeroCopy(vector<InputOutputDescInfo> &input_desc,
                                           vector<InputOutputDescInfo> &output_desc,
                                           std::vector<uint32_t> &inputFormats, std::vector<uint32_t> &output_formats);

  ///
  /// @ingroup domi_ome
  /// @brief copy input data to model
  /// @return Status
  ///
  Status CopyInputDataToModel(const std::vector<DataBuffer> &data, uint32_t data_op_index, bool device_data);

  Status ReturnResult(uint32_t model_id, uint32_t data_id, const bool rslt_flg, const bool seq_end_flg,
                      OutputData *output_data);

  Status ReturnNoOutput(uint32_t model_id, uint32_t data_id);

  ///
  /// @ingroup domi_ome
  /// @brief dump all op input and output information
  /// @param [in] op_list model_id
  /// @return Status
  ///
  Status DumpOpInputOutput(map<uint32_t, OpDescPtr> &op_list, uint32_t model_id);

  ///
  /// @ingroup domi_ome
  /// @brief dump single op input and output information
  /// @param [in] dump_op model_id
  /// @return Status
  ///
  Status DumpSingleOpInputOutput(const OpDescPtr &dump_op, uint32_t model_id);

  Status ModelRunStart();

  ///
  /// @ingroup domi_ome
  /// @brief stop run model
  /// @return Status
  ///
  Status ModelRunStop();

  ///
  /// @ingroup domi_ome
  /// @brief model run flag
  /// @return Status
  ///
  bool RunFlag() const { return run_flg_; }

  Status GetOutputDescInfo(vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &formats);

  ///
  /// @ingroup domi_ome
  /// @brief Set Session Id
  /// @return void
  ///
  void SetSessionId(uint64_t session_id) { session_id_ = session_id; }

  ///
  /// @ingroup domi_ome
  /// @brief Get Session Id
  /// @return sessionID
  ///
  uint64_t GetSessionId() const { return session_id_; }

  ///
  /// @ingroup domi_ome
  /// @brief SetDeviceId
  /// @return void
  ///
  void SetDeviceId(uint32_t device_id) { device_id_ = device_id; }

  ///
  /// @ingroup domi_ome
  /// @brief Get device Id
  /// @return  device id
  ///
  uint32_t GetDeviceId() const { return device_id_; }

  ///
  /// @ingroup domi_ome
  /// @brief Set Train Mode
  /// @return void
  ///
  void SetTrainMode(bool mode) { is_train_mode_ = mode; }

  ///
  /// @ingroup domi_ome
  /// @brief Get Train Mode
  /// @return bool true
  ///
  bool GetTrainMode() { return is_train_mode_; }

  GeModelPtr GetGeModel() { return ge_model_; }

  const RuntimeParam &GetRuntimeParam() { return runtime_param_; }

  int32_t GetDataInputTid() const { return dataInputTid; }
  void SetDataInputTid(int32_t data_input_tid) { dataInputTid = data_input_tid; }

  ///
  /// @ingroup domi_ome
  /// @brief Save outside address of Data or NetOutput used info for ZeroCopy.
  /// @param [in] const std::vector<void *> &outside_addrs: address of task
  /// @param [in] const void *args_offset: arguments address save the address.
  /// @return None.
  ///
  void SetZeroCopyAddr(const std::vector<void *> &outside_addrs_, void *args_offset);

  DavinciModel &operator=(const DavinciModel &model) = delete;

  DavinciModel(const DavinciModel &model) = delete;

 private:
  // memory address of weights
  uint8_t *weights_mem_base_;
  uint8_t *var_mem_base_;
  // memory address of model
  uint8_t *mem_base_;
  bool is_inner_mem_base_;
  bool is_inner_weight_base_;
  // input data manager
  DataInputer *data_inputer_;

  int32_t dataInputTid;

  ///
  /// @ingroup domi_ome
  /// @brief Save Data and NetOutput address info for ZeroCopy.
  /// @param [in] const std::vector<void *> &outside_addrs
  /// @return None.
  ///
  void SetOutsideAddr(const std::vector<void *> &outside_addrs);
  Status ModelZeroCopy(const InputData &input_data, OutputData &output_data);
  Status ZeroCopyInput(const InputData &input_data);
  Status ZeroCopyOutput(const OutputData &output_data);
  Status ZeroCopyImpl(const void *src_addr, const DataBuffer &data_buf);

  Status CopyInputData(const InputData &current_data, bool device_data = false);

  Status CopyTransData(const std::vector<DataBuffer> &data, uint32_t data_index, uint32_t data_op_index,
                       const std::vector<GeAttrValue::INT> &outputs, uint32_t output_size);

  Status CopyPlainData(const std::vector<DataBuffer> &data, uint32_t data_index, uint32_t data_op_index,
                       const std::vector<GeAttrValue::INT> &outputs, uint32_t output_size, rtMemcpyKind_t kind);

  Status CopyOutputData(uint32_t model_id, uint32_t data_id, OutputData &output_data);

  Status CopyOutputDataToUser(OpDescPtr &op_desc, std::vector<DataBuffer> &blobs, uint32_t &data_index);

  Status SyncVarData();

  Status SyncDataAndDump();

  Status InitModelMem(void *dev_ptr, size_t memsize, void *weight_ptr, size_t weightsize);

  Status GetInputDescInfo(vector<InputOutputDescInfo> &input_desc, std::vector<uint32_t> &formats);

  Status InitTaskInfo(domi::ModelTaskDef &modelTaskInfo);

  void UnbindHcomStream();

  Status DistributeTask();

  uint8_t *MallocFeatureMapMem(uint64_t data_size);

  uint8_t *MallocWeightsMem(uint32_t weights_size);

  void FreeFeatureMapMem();

  void FreeWeightsMem();

  void ReleaseTask();

  void UnbindTaskSinkStream();

  ///
  /// @ingroup domi_ome
  /// @brief Constant Op Init.
  /// @return Status
  ///
  Status InitConstant(const ConstOpDescPtr &op_desc) const;

  ///
  /// @ingroup domi_ome
  /// @brief TVM Op Init.
  /// @return Status
  ///
  Status InitTbeHandle(const OpDescPtr &op_desc);

  void StoreTbeHandle(const std::string &handle_key);
  void CleanTbeHandle();

  ///
  /// @ingroup domi_ome
  /// @brief Init model stream for NN model.
  /// @return Status
  ///
  Status InitModelStream(rtStream_t stream, bool async_mode);

  ///
  /// @ingroup domi_ome
  /// @brief insert active_stream_indication_
  /// @return Status
  ///
  Status MarkActiveStream(const OpDescPtr &op_desc);

  void InitRuntimeParams();

  void CheckHasHcomOp();

  Status DoTaskSink();

  void CreateOutput(uint32_t index, OpDescPtr &op_desc, InputOutputDescInfo &output, uint32_t &format_result);

  uint32_t GetGraphID(const std::string &session_graph_id);

  Status TransAllVarData(ComputeGraphPtr &graph, uint32_t graph_id);
  Status CopyVarData(ComputeGraphPtr &graph);
  Status CopyTensorFromSrcVarNode(const NodePtr &var_src, const NodePtr &var_dst);

  void SetDataDumperArgs();

  bool is_model_has_inited_;
  uint32_t model_id_;
  string name_;
  uint32_t version_;
  GeModelPtr ge_model_;

  map<uint32_t, OpDescPtr> op_list_;

  // data op_desc
  vector<OpDescPtr> data_op_list_;

  vector<OpDescPtr> output_op_list_;

  vector<OpDescPtr> variable_op_list_;

  vector<uint32_t> output_size_list_;

  // output op: save cce op actual needed memory size
  vector<uint32_t> output_memory_size_list_;

  std::thread thread_id_;

  std::shared_ptr<ModelListener> listener_;

  bool run_flg_;

  std::mutex mux_run_flg_;

  static SysMode mode_;

  static std::mutex mutex_mode_;

  int32_t priority_;

  vector<rtStream_t> stream_list_;

  std::mutex all_hccl_stream_list_mutex_;
  vector<rtStream_t> all_hccl_stream_list_;

  vector<rtEvent_t> event_list_;

  vector<rtLabel_t> label_list_;

  std::mutex outside_addrs_mutex_;
  std::map<const void *, std::vector<void *>> outside_addrs_;

  std::vector<TaskInfoPtr> task_list_;
  // rt_moodel_handle
  rtModel_t rt_model_handle_;

  rtStream_t rt_model_stream_;

  bool is_inner_model_stream_;

  // ACL queue schedule, save queue ids for Init.
  std::vector<uint32_t> input_queue_ids_;
  std::vector<uint32_t> output_queue_ids_;

  // save input/output tensor descriptor in maps
  std::map<std::string, ConstGeTensorDescPtr> data_op_input_tensor_desc_map_;
  std::map<std::string, ConstGeTensorDescPtr> data_op_output_tensor_desc_map_;

  bool support_mem_shared_flag_;

  uint64_t session_id_;

  uint32_t device_id_;

  bool is_train_mode_;

  std::mutex flowctrl_op_index_internal_map_mutex_;
  std::map<uint32_t, uint32_t> flowctrl_op_index_internal_map_;
  std::set<uint32_t> active_stream_indication_;

  std::shared_ptr<domi::ModelTaskDef> model_task_def_;
  std::set<uint32_t> aicpu_streams_;
  std::set<uint32_t> hcom_streams_;
  RuntimeParam runtime_param_;
  TBEKernelStore tbekernel_store_;

  static std::mutex tvm_bin_mutex_;  // lock for tvm maps.
  static std::set<std::string> tvm_bin_kernel_;

  std::map<std::string, uint32_t> used_tbe_handle_map_;

  // for profiling
  std::map<uint32_t, std::string> op_name_map_;
  std::map<uint32_t, std::string> op_task_id_map_;

  int64_t maxDumpOpNum_;
  // for data dump
  DataDumper data_dumper_;

  uint64_t iterator_count_;
};

#define TIME_LOG_HEAD_FMT "       OP_ID   OP_NAME                OP_TYPE           ELAPSED TIME(ms)"
#define OP_TIME_LOG_FMT "%d_%-5d %-5d | %-20s | %-15s | %10f | %10d"
#define MODEL_TIME_LOG_FMT "******** Model %d ends, elapsed time: %f ms ********"
const size_t INPUT_OUTPUT_NAME_MAX_LEN = 256;
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_H_
