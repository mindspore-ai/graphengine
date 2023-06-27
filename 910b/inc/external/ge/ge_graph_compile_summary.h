/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef INC_EXTERNAL_GE_GE_GRAPH_COMPILE_SUMMARY_H
#define INC_EXTERNAL_GE_GE_GRAPH_COMPILE_SUMMARY_H

#include <memory>
#include <vector>
#include "ge_api_types.h"
#include "ge_error_codes.h"

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
 private:
  CompiledGraphSummary() = default;
  std::shared_ptr<SummaryData> data_{nullptr};
};

using CompiledGraphSummaryPtr = std::shared_ptr<CompiledGraphSummary>;
}  // namespace ge
#endif  // INC_EXTERNAL_GE_GE_GRAPH_COMPILE_SUMMARY_H
