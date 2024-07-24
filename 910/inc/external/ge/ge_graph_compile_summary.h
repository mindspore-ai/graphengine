/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GE_GE_GRAPH_COMPILE_SUMMARY_H
#define INC_EXTERNAL_GE_GE_GRAPH_COMPILE_SUMMARY_H

#include <memory>
#include <vector>
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_feature_memory.h"

namespace ge {
class GE_FUNC_VISIBILITY CompiledGraphSummary {
 public:
  class Builder;
  class SummaryData;

  ~CompiledGraphSummary();
  CompiledGraphSummary &operator=(const CompiledGraphSummary &) & = delete;
  CompiledGraphSummary(const CompiledGraphSummary &) = delete;

  ///
  /// @brief get whether or not the graph is static compiled
  /// @return return true if static
  ///
  bool IsStatic() const;

  ///
  /// @brief get const memory size after compiled
  /// @param [out] size const memory size
  /// @return Status result of function
  ///
  Status GetConstMemorySize(size_t &size) const;

  ///
  /// @brief get fearturemap memory size after compiled, without input and output
  /// @param [out] size fearturemap memory size
  /// @return Status result of function
  ///
  Status GetFeatureMemorySize(size_t &size) const;

  ///
  /// @brief get fix feature memory size after compiled
  /// @param [out] size const memory size
  /// @return Status result of function
  ///
  Status GetFixedFeatureMemorySize(size_t &size) const;

  ///
  /// @brief get all type feature memory size after compiled
  /// @return vector of FeatureMemory pointer
  ///
  std::vector<FeatureMemoryPtr> GetAllFeatureMemoryTypeSize() const;

  ///
  /// @brief get refreshable fearturemap memory size after compiled, without input and output and fix memory
  /// @param [out] size fearturemap memory size
  /// @return Status result of function
  ///
  Status GetRefreshableFeatureMemorySize(size_t &size) const;

  ///
  /// @brief get whether or not the graph support featuremap memory base refreshable
  /// @param [out] v refreshable or not
  /// @return Status result of function
  ///
  Status GetFeatureMemoryBaseRefreshable (bool &v) const;

  ///
  /// @brief get the used stream number of the whole compiled graph
  /// @param [out] num used stream number
  /// @return Status result of function
  ///
  Status GetStreamNum(size_t &num) const;

  ///
  /// @brief get the used event number of the whole compiled graph
  /// @param [out] num used event number
  /// @return Status result of function
  ///
  Status GetEventNum(size_t &num) const;

  ///
  /// @brief get the output tensor shapes of the whole compiled graph
  /// @param [out] shapes vector of ge::Shape
  /// @return Status result of function
  ///
  Status GetOutputShapes(std::vector<ge::Shape> &shapes) const;

  Status GetOutputDtypes(std::vector<ge::DataType> &dtypes) const;

  Status GetIOIndexesWithSameAddr(std::vector<std::pair<uint32_t, uint32_t>> &io_indexes) const;

  Status GetInputShardMethod(std::map<std::string, std::map<int32_t, std::vector<std::pair<int64_t, int64_t>>>>
                             &device_id_to_tensor_deployment) const;

 private:
  CompiledGraphSummary() = default;
  std::shared_ptr<SummaryData> data_{nullptr};
};

using CompiledGraphSummaryPtr = std::shared_ptr<CompiledGraphSummary>;
}  // namespace ge
#endif  // INC_EXTERNAL_GE_GE_GRAPH_COMPILE_SUMMARY_H
