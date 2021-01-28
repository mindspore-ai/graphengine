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
#include "ge/ge_ir_build.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge_types.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/op_desc.h"
#include "graph/detail/attributes_holder.h"
#include "omg/omg_inner_types.h"

namespace ge {
class GE_FUNC_VISIBILITY GeGenerator {
 public:
  static GeGenerator &GetInstance() {
    static GeGenerator Instance;
    return Instance;
  }
  GeGenerator() = default;

  ~GeGenerator() { (void)Finalize(); }

  GeGenerator(const GeGenerator &) = delete;

  GeGenerator &operator=(const GeGenerator &) = delete;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Initialize(const std::map<std::string, std::string> &options, OmgContext &context);

  Status Finalize();

  Status GenerateOfflineModel(const Graph &graph, const std::string &file_name_prefix,
                              const std::vector<GeTensor> &inputs = std::vector<GeTensor>());

  Status GenerateOnlineModel(const Graph &graph, const vector<GeTensor> &inputs, ge::ModelBufferData &model);

  Status GenerateInfershapeGraph(const Graph &graph);

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
  ///
  /// @ingroup ge
  /// @brief: Build single Op into model buff.
  /// @param [in] op_desc: the OP description.
  /// @param [in] inputs: input tensors.
  /// @param [in] outputs: output tensors.
  /// @param [in] engine_type: engine type.
  /// @param [out] model_buff: model buff of op.
  /// @return SUCCESS or FAILED
  Status BuildSingleOpModel(OpDescPtr &op_desc, const vector<GeTensor> &inputs, const vector<GeTensor> &outputs,
                            OpEngineType engine_type, ModelBufferData &model_buff);
  ///
  /// @ingroup ge
  /// @brief: Build single Op into model buff.
  /// @param [in] op_desc: the OP description.
  /// @param [in] inputs: input tensors.
  /// @param [in] outputs: output tensors.
  /// @param [in] graph_name: graph name.
  /// @param [out] graph: graph of single op.
  /// @return SUCCESS or FAILED
  Status BuildSingleOpGraph(OpDescPtr &op_desc, const vector<GeTensor> &inputs, const vector<GeTensor> &outputs,
                            std::string graph_name, Graph &graph);

 private:
  Status GenerateModel(const Graph &graph, const string &file_name_prefix, const vector<GeTensor> &inputs,
                       ge::ModelBufferData &model, bool is_offline = true);
  Status BuildSingleOp(OpDescPtr &op_desc, const vector<GeTensor> &inputs, const vector<GeTensor> &outputs,
                       const string &model_file_name, OpEngineType engine_type, ModelBufferData &model_buff,
                       bool is_offline = true);
  Status CheckForSingleOp(OpDescPtr &op_desc, const vector<GeTensor> &inputs, const vector<GeTensor> &outputs);

  class Impl;

  std::shared_ptr<Impl> impl_;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_
