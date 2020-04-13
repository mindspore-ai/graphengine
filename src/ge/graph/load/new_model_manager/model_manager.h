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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_MANAGER_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_MANAGER_H_

#include <pthread.h>
#include <stdint.h>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include "cce/aicpu_engine_struct.h"
#include "common/types.h"
#include "common/ge_types.h"
#include "common/ge_inner_error_codes.h"
#include "common/helper/model_helper.h"
#include "common/helper/om_file_helper.h"
#include "graph/model.h"
#include "runtime/base.h"
#include "graph/ge_context.h"
#include "ge/ge_api_types.h"

namespace ge {
class DavinciModel;

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ModelManager {
 public:
  static std::shared_ptr<ModelManager> GetInstance();
  static void FinalizeForPtr(ModelManager *) {}

  ///
  /// @ingroup domi_ome
  /// @brief load and init model
  /// @param [in] model_id model id
  /// @param [in] model including model ptr and size
  /// @param [in] listener used to return result
  /// @param [in/out] info model task generate info
  /// @return Status run result
  /// @author
  ///
  ge::Status LoadModelOffline(uint32_t &model_id, const ModelData &model,
                              std::shared_ptr<ModelListener> listener = nullptr, void *dev_ptr = nullptr,
                              size_t mem_size = 0, void *weight_ptr = nullptr, size_t weight_size = 0);

  ///
  /// @ingroup domi_ome
  /// @brief load and init model
  /// @param [out] model_id model id
  /// @param [in] model modeldef datatype
  /// @param [in] listener used to return result
  /// @param [in] isTrainMode model type
  /// @return Status run result
  /// @author @
  ///
  ge::Status LoadModelOnline(uint32_t &model_id, std::shared_ptr<ge::Model> &model,
                             std::shared_ptr<ModelListener> listener);

  ///
  /// @ingroup ge
  /// @brief ACL case, Load task list with queue.
  /// @param [out] model_id: model id for manager.
  /// @param [in] model_data: Model data load from offline model file.
  /// @param [in] input_que_ids: input queue ids from user, num equals Data Op.
  /// @param [in] output_que_ids: input queue ids from user, num equals NetOutput Op.
  /// @return: 0 for success / others for fail
  ///
  ge::Status LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                            const std::vector<uint32_t> &input_queue_ids,
                            const std::vector<uint32_t> &output_queue_ids);

  ///
  /// @ingroup domi_ome
  /// @brief unload model and free resources
  /// @param [in] model_id model id
  /// @return Status run result
  /// @author
  ///
  ge::Status Unload(uint32_t model_id);

  ///
  /// @ingroup omm
  /// @brief unload model and free resources
  /// @param [in] model_id model id
  /// @return Status run result
  /// @author
  ///
  ge::Status UnloadModeldef(uint32_t model_id);

  ///
  /// @ingroup domi_ome
  /// @brief process input data asynchronously
  /// cannot be invoked by multiple thread
  /// if one fails, other continue
  /// @param [in] input_data   input data
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  /// @return MODEL_NOT_READY  model not ready
  /// @return PUSH_DATA_FAILED push data into model queue failed
  /// @author
  ///
  ge::Status DataInput(const InputData &input_data, OutputData &output_data);

  ge::Status DataInputTensor(uint32_t model_id, const std::vector<TensorInfo> &inputs,
                             std::vector<TensorInfo> &outputs);

  ///
  /// @ingroup domi_ome
  /// @brief model start to run
  ///
  ge::Status Start(uint32_t model_id);

  ///
  /// @ingroup domi_ome
  /// @brief  ACL case, do not start new thread, return result
  /// @param [in] model_id  model id
  /// @param [in] stream   model stream
  /// @param [in] async_mode  is asynchronize mode.
  /// @param [in] input_data  model input data
  /// @param [out] output_data  model output data
  ///
  ge::Status ExecuteModel(uint32_t model_id, rtStream_t stream, bool async_mode, const InputData &input_data,
                          OutputData &output_data);

  ///
  /// @ingroup domi_ome
  /// @brief model stop
  ///
  ge::Status Stop(uint32_t model_id);

  ///
  /// @ingroup domi_ome
  /// @brief comment handle function
  ///
  ge::Status HandleCommand(const Command &command);
  static ge::Status HandleAclProfilingCommand(const Command &command);
  static ge::Status HandleProfileCommand(const Command &command);
  static ge::Status HandleDumpCommand(const Command &command);
  ///
  /// @ingroup domi_ome
  /// @brief get model memory usage
  /// @param [in] model_id  model id
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  ///
  ge::Status GetMaxUsedMemory(const uint32_t model_id, uint64_t &max_size);

  ///
  /// @ingroup domi_ome
  /// @brief get model input and output size
  /// @param [in] model_id  model id
  /// @param [out] input_shape   input tensor
  /// @param [out] output_shape  output tensor
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  ///
  ge::Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                    std::vector<InputOutputDescInfo> &output_desc);

  ge::Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                    std::vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &inputFormats,
                                    std::vector<uint32_t> &outputFormats);

  ///
  /// @ingroup domi_ome
  /// @brief set model input and output size zero copy
  /// @param [in] model_id  model id
  /// @param [out] input_shape   input tensor
  /// @param [out] output_shape  output tensor
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  ///
  ge::Status GetInputOutputDescInfoForZeroCopy(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                               std::vector<InputOutputDescInfo> &output_desc);

  ge::Status GetInputOutputDescInfoForZeroCopy(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                               std::vector<InputOutputDescInfo> &output_desc,
                                               std::vector<uint32_t> &inputFormats,
                                               std::vector<uint32_t> &outputFormats);

  ge::Status SetDevice(int32_t deviceId) const;

  ///
  /// @ingroup domi_ome
  /// @brief Get model according to given id
  ///
  std::shared_ptr<DavinciModel> GetModel(uint32_t id);

  ge::Status KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType op_type, uint64_t session_id, uint32_t model_id);

  ge::Status CreateAicpuSession(uint64_t session_id);

  static ge::Status GetModelMemAndWeightSize(const ModelData &model, size_t &mem_size, size_t &weight_size);

  void DestroyAicpuSession(uint64_t session_id);

  ge::Status DestroyAicpuKernel(uint64_t session_id, uint32_t model_id);

  ge::Status CreateAicpuKernel(uint64_t session_id, uint32_t model_id, uint64_t kernel_id);

 private:
  ///
  /// @ingroup domi_ome
  /// @brief constructor
  ///
  ModelManager();

  ///
  /// @ingroup domi_ome
  /// @brief destructor
  ///
  ~ModelManager();

  ///
  /// @ingroup domi_ome
  /// @brief insert new model into model manager set
  ///
  void InsertModel(uint32_t id, std::shared_ptr<DavinciModel> &davinci_model);

  ///
  /// @ingroup domi_ome
  /// @brief delete model from model manager set
  ///
  ge::Status DeleteModel(uint32_t id);

  void GenModelId(uint32_t *id);

  std::map<uint32_t, std::shared_ptr<DavinciModel>> model_map_;
  std::map<std::string, std::vector<uint64_t>> model_aicpu_kernel_;
  std::vector<uint32_t> free_model_id_;
  uint32_t max_model_id_;
  std::mutex map_mutex_;
  std::mutex sess_ids_mutex_;
  std::set<uint64_t> sess_ids_;
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_MANAGER_H_
