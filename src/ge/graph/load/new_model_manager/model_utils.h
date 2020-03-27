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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_UTILS_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_UTILS_H_

#include <vector>

#include "cce/dnn.h"
#include "cce/taskdown_api.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/load/new_model_manager/task_info/task_info.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_adapter.h"

using std::vector;

namespace ge {
class ModelUtils {
 public:
  ModelUtils() = default;
  ~ModelUtils() = default;

  ///
  /// @ingroup domi_ome
  /// @brief Check is Output Op.
  /// @return bool
  ///
  static bool IsOutput(ConstOpDescPtr op_desc);

  ///
  /// @ingroup domi_ome
  /// @brief Check is the Input need trans code.
  /// @return bool
  ///
  static bool IsInputTensorNeedTrans(ConstOpDescPtr op_desc, size_t tensor_index);

  ///
  /// @ingroup domi_ome
  /// @brief Get input size.
  /// @return vector<uint32_t>
  ///
  static vector<uint32_t> GetInputSize(ConstOpDescPtr op_desc);

  ///
  /// @ingroup domi_ome
  /// @brief Get output size.
  /// @return vector<uint32_t>
  ///
  static vector<uint32_t> GetOutputSize(ConstOpDescPtr op_desc);

  ///
  /// @ingroup domi_ome
  /// @brief Get workspace size.
  /// @return vector<uint32_t>
  ///
  static vector<uint32_t> GetWorkspaceSize(ConstOpDescPtr op_desc);

  ///
  /// @ingroup domi_ome
  /// @brief Get weight size.
  /// @return vector<uint32_t>
  ///
  static vector<uint32_t> GetWeightSize(ConstOpDescPtr op_desc);

  ///
  /// @ingroup domi_ome
  /// @brief Get weights.
  /// @return vector<ConstGeTensorPtr>
  ///
  static vector<ConstGeTensorPtr> GetWeights(ConstOpDescPtr op_desc);

  ///
  /// @ingroup domi_ome
  /// @brief Save Output tensor info to vector.
  /// @return Status
  ///
  static Status GetOutputSize(ConstOpDescPtr op_desc, vector<uint32_t> &output_size_list,
                              vector<uint32_t> &output_memory_size_list);

  ///
  /// @ingroup domi_ome
  /// @brief Get AiCpuOp Input descriptor.
  /// @return vector<::tagCcAICPUTensor>
  ///
  static vector<::tagCcAICPUTensor> GetInputDescs(ConstOpDescPtr op_desc);
  ///
  /// @ingroup domi_ome
  /// @brief Get AiCpuOp Output descriptor.
  /// @return vector<::tagCcAICPUTensor>
  ///
  static vector<::tagCcAICPUTensor> GetOutputDescs(ConstOpDescPtr op_desc);

  ///
  /// @ingroup domi_ome
  /// @brief Get input data address.
  /// @return vector<void*>
  ///
  static vector<void *> GetInputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc,
                                          bool need_convert = true);
  ///
  /// @ingroup domi_ome
  /// @brief Get output data address.
  /// @return vector<void*>
  ///
  static vector<void *> GetOutputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc,
                                           bool need_convert = true);

  ///
  /// @ingroup domi_ome
  /// @brief Get workspace data address.
  /// @return vector<void*>
  ///
  static vector<void *> GetWorkspaceDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc);

  static ge::Status ConvertVirtualAddressToPhysical(uint8_t *virtual_address, uint64_t size,
                                                    uint8_t *&physical_address);
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_UTILS_H_
