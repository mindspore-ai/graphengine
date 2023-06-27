/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include "ge/ge_allocator.h"
#include "common/dynamic_aipp.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "graph/tensor.h"
#include "graph/ge_tensor.h"
#include "framework/common/ge_model_inout_types.h"

namespace ge {
class SingleOp;
class DynamicSingleOp;
class GeRootModel;

struct RunModelData {
  uint32_t index;  // Data index
  uint32_t modelId;
  std::vector<DataBuffer> blobs;       // All input/output data buffer
  uint32_t timestamp;                  // Data creation time
  uint32_t timeout;                    // Processing timeout
  uint64_t request_id = 0UL;             // Request ID
  uint64_t dynamic_batch_size = 0UL;     // Dynamic batch size scene, set dynamic size, not supported by default:0
  uint64_t dynamic_image_height = 0UL;   // Dynamic image size scene, set image height, not supported by default:0
  uint64_t dynamic_image_width = 0UL;    // Dynamic image size scene, set image width, not supported by default:0
  std::vector<uint64_t> dynamic_dims;  // Dynamic dims scene, set dynamic dims, not supported by default:empty
};

class GE_FUNC_VISIBILITY GeExecutor {
 public:
  GeExecutor();
  ~GeExecutor() = default;

  Status Initialize();
  Status Finalize();

  ///
  /// @ingroup ge
  /// @brief Initialize global execute environment.
  /// @param [in] options: environment variables.
  /// @return init result
  ///
  static Status Initialize(const std::map<std::string, std::string> &options);

  ///
  /// @ingroup ge
  /// @brief Finalize global execute environment.
  /// @return execute result
  ///
  static Status FinalizeEx();

  Status UnloadModel(const uint32_t model_id);

  // Get input and output descriptor
  Status GetModelDescInfo(const uint32_t model_id, std::vector<TensorDesc> &input_desc,
                          std::vector<TensorDesc> &output_desc, const bool new_model_desc = false);

  Status GetModelDescInfoFromMem(const ModelData &model_data, ModelInOutInfo &info) const;
  
  ///
  /// @ingroup ge
  /// @brief Set dynamic batch size
  /// @param [in] model_id: model id allocate from manager
  /// @param [in] dynamic_input_addr: dynamic input addr created by user
  /// @param [in] length: length of dynamic input addr
  /// @param [in] batch_size: batch size entered by user in dynamic multi-batch scenario
  /// @return execute result
  ///
  Status SetDynamicBatchSize(const uint32_t model_id, void *const dynamic_input_addr, const uint64_t length,
                             const uint64_t batch_size);

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
  Status SetDynamicImageSize(const uint32_t model_id, void *const dynamic_input_addr, const uint64_t length,
                             const uint64_t image_height, const uint64_t image_width);

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
  Status SetDynamicDims(const uint32_t model_id, void *const dynamic_input_addr, const uint64_t length,
                        const std::vector<uint64_t> &dynamic_dims);

  ///
  /// @ingroup ge
  /// @brief Get current dynamic dims info by combined dims
  /// @param [in] model_id: model id allocate from manager
  /// @param [in] dynamic_dims: cur gear dynamic dims value
  /// @param [out] cur_dynamic_dims: current dynamic dims
  /// @return execute result
  ///
  Status GetCurDynamicDims(const uint32_t model_id, const std::vector<uint64_t> &dynamic_dims,
                           std::vector<uint64_t> &cur_dynamic_dims);

  ///
  /// @ingroup ge
  /// @brief Get dynamic batch_info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @param [out] dynamic_type
  /// @return execute result
  ///
  Status GetDynamicBatchInfo(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                             int32_t &dynamic_type);

  ///
  /// @ingroup ge
  /// @brief Get combined dynamic dims info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @return execute result
  ///
  Status GetCombinedDynamicDims(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info);

  ///
  /// @ingroup ge
  /// @brief Get user designeate shape order
  /// @param [in] model_id
  /// @param [out] user_designate_shape_order
  /// @return execute result
  ///
  Status GetUserDesignateShapeOrder(const uint32_t model_id,
                                    std::vector<std::string> &user_designate_shape_order);

  Status GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type);

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
  Status SetDynamicAippData(const uint32_t model_id, void *const dynamic_input_addr, const uint64_t length,
                            const std::vector<kAippDynamicBatchPara> &aipp_batch_para,
                            const kAippDynamicPara &aipp_parms);

  Status GetAIPPInfo(const uint32_t model_id, const uint32_t index, AippConfigInfo &aipp_info);

  Status GetOpAttr(const uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                   std::string &attr_value);

  Status GetModelAttr(const uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info);

  Status GetAippType(const uint32_t model_id, const uint32_t index, InputAippType &type, size_t &aipp_index);

  Status CommandHandle(const Command &command) const;

  Status SetDump(const DumpConfig &dump_config);

  ///
  /// @ingroup ge
  /// @brief Query model memory consuming interface
  /// @param [in] model_id  Offline model ID
  /// @param [out] max_size Memory size
  /// @return SUCCESS
  /// @return FAILED
  ///
  Status GetMaxUsedMemory(const uint32_t model_id, uint32_t &max_size);

  ///
  /// @ingroup ge
  /// @brief Load data from model file to memory
  /// @param [in] const std::string &path: Offline model file path
  /// @param [out] ModelData &model_data: Offline model memory data
  /// @return SUCCESS handle successfully / others handle failed
  ///
  Status LoadDataFromFile(const std::string &path, ModelData &model_data);

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
  Status LoadModelFromData(uint32_t &model_id, const ModelData &model_data, void *const dev_ptr,
                           const size_t mem_size, void *const weight_ptr, const size_t weight_size);

  ///
  /// @ingroup ge
  /// @brief Load task list from ModelData with queue.
  /// @param [out] model_id: model id allocate from manager.
  /// @param [in] model_data: Model data load from offline model.
  /// @param [in] input_queue_ids: input queue ids create from user.
  /// @param [in] output_queue_ids: input queue ids create from user.
  /// @return: 0 for success / others for fail
  ///
  Status LoadModelWithQ(uint32_t &model_id, const ModelData &model_data, const std::vector<uint32_t> &input_queue_ids,
                        const std::vector<uint32_t> &output_queue_ids);

  ///
  /// @ingroup ge
  /// @brief Load task list from GeRootModel with queue and param.
  /// @param [out] model_id: model id allocate from manager.
  /// @param [in] root_model: Instance of GeRootModel.
  /// @param [in] model_queue_param: params and queue ids and create from user.
  /// @return: 0 for success / others for fail
  ///
  Status LoadModelWithQ(uint32_t &model_id,
                        const std::shared_ptr<GeRootModel> &root_model,
                        const ModelQueueParam &model_queue_param) const;

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
  Status ExecModel(const uint32_t model_id, void *const stream, const RunModelData &input_data,
                   RunModelData &output_data, const bool async_mode = false);

  ///
  /// @ingroup ge
  /// @brief Load task list from root_model without input queue or output queue.
  /// @param [out] model_id: model id allocate from manager.
  /// @param [in] root_model: Instance of GeRootModel.
  /// @return: 0 for success / others for fail
  ///
  Status LoadModelWithoutQ(uint32_t &model_id, const std::shared_ptr<GeRootModel> &root_model) const;

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
  Status ExecModel(const uint32_t model_id, void *const stream, const RunModelData &run_input_data,
                   const std::vector<GeTensorDesc> &input_desc, RunModelData &run_output_data,
                   std::vector<GeTensorDesc> &output_desc, const bool async_mode = false);

  ///
  /// @ingroup ge
  /// @brief Get weight memory size from model file
  /// @param [in] const std::string &path: Offline model file path
  /// @param [out] size_t &mem_size Execution memory size
  /// @param [out] size_t &weight_size Weight memory space size
  /// @return SUCCESS handle successfully / others handle failed
  ///
  Status GetMemAndWeightSize(const std::string &path, size_t &mem_size, size_t &weight_size);

  ///
  /// @ingroup ge
  /// @brief Get weight memory size from model file
  /// @param [in] const void *model_data Offline model buffer
  /// @param [in] size_t model_size Offline model buffer length
  /// @param [out] size_t &mem_size Execution memory size
  /// @param [out] size_t &weight_size Weight memory space size
  /// @return SUCCESS handle successfully / others handle failed
  ///
  Status GetMemAndWeightSize(const void *const model_data, const size_t model_size, size_t &mem_size,
                             size_t &weight_size);

  static Status LoadSingleOp(const std::string &model_name, const ModelData &model_data, void *const stream,
                             SingleOp **const single_op);

  static Status LoadSingleOpV2(const std::string &model_name, const ModelData &model_data, void *const stream,
                               SingleOp **const single_op, const uint64_t model_id);

  static Status ExecuteAsync(SingleOp *const executor, const std::vector<DataBuffer> &inputs,
                             std::vector<DataBuffer> &outputs);

  static Status LoadDynamicSingleOp(const std::string &model_name, const ModelData &model_data, void *const stream,
                                    DynamicSingleOp **const single_op);

  static Status LoadDynamicSingleOpV2(const std::string &model_name, const ModelData &model_data, void *const stream,
                                      DynamicSingleOp **const single_op, const uint64_t model_id);

  static Status UnloadSingleOp(const uint64_t op_id);

  static Status UnloadDynamicSingleOp(const uint64_t op_id);

  static Status ExecuteAsync(DynamicSingleOp *const executor, const std::vector<GeTensorDesc> &input_desc,
                             const std::vector<DataBuffer> &inputs, std::vector<GeTensorDesc> &output_desc,
                             std::vector<DataBuffer> &outputs);

  static Status ReleaseSingleOpResource(void *const stream);

  static Status ClearCustomAicpuSo(const uint32_t device_id);

  static Status GetDeviceIdByModelId(const uint32_t model_id, uint32_t &device_id);

  Status GetBatchInfoSize(const uint32_t model_id, size_t &shape_count);
  Status GetOrigInputInfo(const uint32_t model_id, const uint32_t index, OriginInputInfo &orig_input_info);
  Status GetAllAippInputOutputDims(const uint32_t model_id, const uint32_t index,
                                   std::vector<InputOutputDims> &input_dims, std::vector<InputOutputDims> &output_dims);
  Status GetOpDescInfo(const uint32_t device_id, const uint32_t stream_id, const uint32_t task_id,
                       OpDescInfo &op_desc_info);

 static Status SetAllocator(void *const stream, ge::Allocator *const external_allocator);

 static Status ReleaseResource(const uint32_t device_id);

 static Status ReleaseResource();
 private:
  static std::atomic_bool is_inited_;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_H_
