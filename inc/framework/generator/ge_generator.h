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

#ifndef INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_
#define INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/ge_inner_error_codes.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/op_desc.h"

namespace ge {
class GeGenerator {
 public:
  GeGenerator() = default;

  ~GeGenerator() = default;

  GeGenerator(const GeGenerator &) = delete;

  GeGenerator &operator=(const GeGenerator &) = delete;

  Status Initialize(const std::map<std::string, std::string> &options);

  Status Finalize();

  Status GenerateOfflineModel(const Graph &graph, const std::string &file_name_prefix,
                              const std::vector<GeTensor> &inputs = std::vector<GeTensor>());

  ///
  /// @ingroup ge
  /// @brief: Build single OP in Model.
  /// @param [in] op_desc: the OP description.
  /// @param [in] inputs: input tensors.
  /// @param [in] outputs: output tensors.
  /// @param [in] model_file_name: name of model file.
  /// @return SUCCESS or FAILED
  ///
  Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs, const std::string &model_file_name);

 private:
  class Impl;

  std::shared_ptr<Impl> impl_;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_
