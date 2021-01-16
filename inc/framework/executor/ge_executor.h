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

#ifndef INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_H_
#define INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "common/dynamic_aipp.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge_types.h"
#include "common/types.h"
#include "graph/tensor.h"
#include "graph/ge_tensor.h"
#include "runtime/base.h"

namespace ge {
class SingleOp;
class DynamicSingleOp;

struct RunModelData {
  uint32_t index;  // Data index
  uint32_t modelId;
  std::vector<DataBuffer> blobs;       // All input/output data buffer
  uint32_t timestamp;                  // Data creation time
  uint32_t timeout;                    // Processing timeout
  uint64_t request_id = 0;             // Request ID
  uint64_t dynamic_batch_size = 0;     // Dynamic batch size scene, set dynamic size, not supported by default:0
  uint64_t dynamic_image_height = 0;   // Dynamic image size scene, set image height, not supported by default:0
  uint64_t dynamic_image_width = 0;    // Dynamic image size scene, set image width, not supported by default:0
  std::vector<uint64_t> dynamic_dims;  // Dynamic dims scene, set dynamic dims, not supported by default:empty
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeExecutor {
 public:
  GeExecutor();
  ~GeExecutor() = default;
  ge::Status Initialize();
  ge::Status Finalize();

  ge::Status UnloadModel(uint32_t modelId);

  // Get input and output descriptor
  ge::Status GetModelDescInfo(uint32_t model_id, std::vector<ge::TensorDesc> &input_desc,
                              std::vector<ge::TensorDesc> &output_desc, bool new_model_desc = false);

  ///
  /// @ingroup ge
  /// @brief Set dynamic batch size
  /// @param [in] model_id: model id allocate from manager
  /// @param [in] dynamic_input_addr: dynamic input addr created by user
  /// @param [in] length: length of dynamic input addr
  /// @param [in] batch_size: batch size entered by user in dynamic multi-batch scenario
  /// @return execute result
  ///
  ge::Status SetDynamicBatchSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t batch_size);

  ///
  /// @ingroup ge
  /// @brief Set dynamic image info
  /// @param [in] model_id: model id allocate from manager
  /// @param [in] dynamic_input_addr: dynamic input addr created by user
  /// @param [in] length: length of dynamic input addr
  /// @param [in] image_height: image height entered by user in dynamic multi-resolution scenario
  /// @param [in] image_width: image width entered by user in dynamic multi-resolution scenario
  /// @return execute result
  ///
  ge::Status SetDynamicImageSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t image_height,
                                 uint64_t image_width);

  ///
  /// @ingroup ge
  /// @brief Set dynamic dims info
  /// @param [in] model_id: model id allocate from manager
  /// @param [in] dynamic_input_addr: dynamic input addr created by user
  /// @param [in] length: length of dynamic input addr
  /// @param [in] dynamic_dim_num: number of dynamic dimension
  /// @param [in] dynamic_dims: array of dynamic dimensions
  /// @return execute result
  ///
  ge::Status SetDynamicDims(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                            const std::vector<uint64_t> &dynamic_dims);

  ///
  /// @ingroup ge
  /// @brief Get current dynamic dims info by combined dims
  /// @param [in] model_id: model id allocate from manager
  /// @param [in] dynamic_dims: cur gear dynamic dims value
  /// @param [out] cur_dynamic_dims: current dynamic dims
  /// @return execute result
  ///
  ge::Status GetCurDynamicDims(uint32_t model_id, const std::vector<uint64_t> &dynamic_dims,
                               std::vector<uint64_t> &cur_dynamic_dims);

  ///
  /// @ingroup ge
  /// @brief Get dynamic batch_info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @param [out] dynamic_type
  /// @return execute result
  ///
  ge::Status GetDynamicBatchInfo(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                 int32_t &dynamic_type);

  ///
  /// @ingroup ge
  /// @brief Get combined dynamic dims info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @return execute result
  ///
  ge::Status GetCombinedDynamicDims(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info);

  ///
  /// @ingroup ge
  /// @brief Get user designeate shape order
  /// @param [in] model_id
  /// @param [out] user_designate_shape_order
  /// @return execute result
  ///
  ge::Status GetUserDesignateShapeOrder(uint32_t model_id, std::vector<std::string> &user_designate_shape_order);

  ge::Status GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type);

  ///
  /// @ingroup ge
  /// @brief Set dynamic image info
  /// @param [in] model_id: model id allocate from manager
  /// @param [in] dynamic_input_addr: dynamic input addr created by user
  /// @param [in] length: length of dynamic input addr
  /// @param [in] aippBatchPara: kAippDynamicBatchPara vector by user in dynamic aipp
  /// @param [in] aippParms: kAippDynamicPara by user in dynamic aipp
  /// @return execute result
  ///
  ge::Status SetDynamicAippData(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                const std::vector<kAippDynamicBatchPara> &aippBatchPara,
                                const kAippDynamicPara &aippParms);

  ge::Status GetAIPPInfo(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_info);
  ge::Status GetModelAttr(uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info);

  ge::Status GetAippType(uint32_t model_id, uint32_t index, InputAippType &type, size_t &aipp_index);

  ge::Status CommandHandle(const ge::Command &command);

  ge::Status SetDump(const DumpConfig &dump_config);

  ///
  /// @ingroup ge
  /// @brief Query model memory consuming interface
  /// @param [in] model_id  Offline model ID
  /// @param [out] max_size Memory size
  /// @return SUCCESS
  /// @return FAILED
  ///
  ge::Status GetMaxUsedMemory(uint32_t model_id, uint32_t &max_size);

  ///
  /// @ingroup ge
  /// @brief Load data from model file to memory
  /// @param [in] const std::string &path: Offline model file path
  /// @param [out] ModelData &model_data: Offline model memory data
  /// @return SUCCESS handle successfully / others handle failed
  ///
  ge::Status LoadDataFromFile(const std::string &path, ge::ModelData &model_data);

  ///
  /// @ingroup ge
  /// @brief Load model from offline model memory data
  /// @param [in] ModelData &model_data: Offline model data
  /// @param [in] void *dev_ptr: Input/Output memory address
  /// @param [in] size_t mem_size: Input/Output memory length
  /// @param [in] void *weight_ptr: Weight memory address
  /// @param [in] size_t weight_size: Weight memory length
  /// @param [out] uint32_t &model_id: Corresponding identification after model loading
  /// @return SUCCESS handle successfully / others handle failed
  ///
  ge::Status LoadModelFromData(uint32_t &model_id, const ge::ModelData &model_data, void *dev_ptr, size_t mem_size,
                               void *weight_ptr, size_t weight_size);

  ///
  /// @ingroup ge
  /// @brief Load task list from ModelData with queue.
  /// @param [out] model_id: model id allocate from manager.
  /// @param [in] model_data: Model data load from offline model.
  /// @param [in] input_queue_ids: input queue ids create from user.
  /// @param [in] output_queue_ids: input queue ids create from user.
  /// @return: 0 for success / others for fail
  ///
  ge::Status LoadModelWithQ(uint32_t &model_id, const ge::ModelData &model_data,
                            const std::vector<uint32_t> &input_queue_ids,
                            const std::vector<uint32_t> &output_queue_ids);

  ///
  /// @ingroup ge
  /// @brief Synchronous execution of offline model(Do not create thread)
  /// @param [in] uint32_t model_id: Model ID to execute
  /// @param [in] void* stream: stream to execute
  /// @param [in] bool async_mode: is asynchronize mode.
  /// @param [in] const domi::InputData *input_data: Model input data
  /// @param [out] domi::OutputData *output_data: Model output data
  /// @return SUCCESS handle successfully / others handle failed
  ///
  ge::Status ExecModel(uint32_t model_id, void *stream, const ge::RunModelData &input_data,
                       ge::RunModelData &output_data, bool async_mode = false);

  ///
  /// @ingroup ge
  /// @brief Synchronous execution of offline model(Do not create thread)
  /// @param [in] uint32_t model_id: Model ID to execute
  /// @param [in] void* stream: stream to execute
  /// @param [in] bool async_mode: is asynchronize mode.
  /// @param [in] const domi::InputData *input_data: Model input data
  /// @param [in] const std::vector<GeTensorDesc> &input_desc: description of model input data
  /// @param [out] domi::OutputData *output_data: Model output data
  /// @param [out] std::vector<GeTensorDesc> &output_desc: description of model output data
  /// @return SUCCESS handle successfully / others handle failed
  ///
  ge::Status ExecModel(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                       const std::vector<GeTensorDesc> &input_desc, ge::RunModelData &run_output_data,
                       std::vector<GeTensorDesc> &output_desc, bool async_mode = false);

  ///
  /// @ingroup ge
  /// @brief Get weight memory size from model file
  /// @param [in] const std::string &path: Offline model file path
  /// @param [out] size_t &mem_size Execution memory size
  /// @param [out] size_t &weight_size Weight memory space size
  /// @return SUCCESS handle successfully / others handle failed
  ///
  ge::Status GetMemAndWeightSize(const std::string &path, size_t &mem_size, size_t &weight_size);

  ///
  /// @ingroup ge
  /// @brief Get weight memory size from model file
  /// @param [in] const void *model_data Offline model buffer
  /// @param [in] size_t model_size Offline model buffer length
  /// @param [out] size_t &mem_size Execution memory size
  /// @param [out] size_t &weight_size Weight memory space size
  /// @return SUCCESS handle successfully / others handle failed
  ///
  ge::Status GetMemAndWeightSize(const void *model_data, size_t model_size, size_t &mem_size, size_t &weight_size);

  static ge::Status LoadSingleOp(const std::string &modelName, const ge::ModelData &modelData, void *stream,
                                 SingleOp **single_op);

  static ge::Status ExecuteAsync(SingleOp *executor, const std::vector<DataBuffer> &inputs,
                                 std::vector<DataBuffer> &outputs);

  static ge::Status LoadDynamicSingleOp(const std::string &model_name, const ge::ModelData &modelData, void *stream,
                                        DynamicSingleOp **single_op);

  static ge::Status ExecuteAsync(DynamicSingleOp *executor, const std::vector<GeTensorDesc> &input_desc,
                                 const std::vector<DataBuffer> &inputs, std::vector<GeTensorDesc> &output_desc,
                                 std::vector<DataBuffer> &outputs);

  static ge::Status ReleaseSingleOpResource(void *stream);

  static ge::Status GetDeviceIdByModelId(uint32_t model_id, uint32_t &device_id);

  ge::Status GetBatchInfoSize(uint32_t model_id, size_t &shape_count);
  ge::Status GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &orig_input_info);
  ge::Status GetAllAippInputOutputDims(uint32_t model_id, uint32_t index, std::vector<InputOutputDims> &input_dims,
                                       std::vector<InputOutputDims> &output_dims);
  ge::Status GetOpDescInfo(uint32_t device_id, uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info);

 private:
  static bool isInit_;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_H_
