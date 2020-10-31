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

#ifndef INC_EXTERNAL_GRAPH_GRAPH_H_
#define INC_EXTERNAL_GRAPH_GRAPH_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "./operator.h"

namespace ge {
class GraphImpl;

using GraphImplPtr = std::shared_ptr<GraphImpl>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Graph {
  friend class GraphUtils;

 public:
  explicit Graph(const std::string &name);

  Graph() = default;

  ~Graph() = default;

  Graph &SetInputs(const std::vector<Operator> &inputs);

  Graph &SetOutputs(const std::vector<Operator> &outputs);

  Graph &SetOutputs(const std::vector<std::pair<Operator, std::vector<size_t>>> &output_indexs);

  Graph &SetOutputs(const std::vector<std::pair<ge::Operator, std::string>> &outputs);

  Graph &SetTargets(const std::vector<Operator> &targets);

  bool IsValid() const;

  graphStatus AddOp(const ge::Operator &op);

  graphStatus FindOpByName(const string &name, ge::Operator &op) const;

  graphStatus FindOpByType(const string &type, std::vector<ge::Operator> &ops) const;

  graphStatus GetAllOpName(std::vector<string> &op_name) const;

  graphStatus SaveToFile(const string &file_name) const;

  graphStatus LoadFromFile(const string &file_name);

  const std::string &GetName() const;

  ///
  /// Set is need train iteration.
  /// If set true, it means this graph need to be run iteration some
  /// times(according variant "npu_runconfig/iterations_per_loop").
  /// @param need_iteration need_iteration:whether to set iteration or not
  ///
  void SetNeedIteration(bool need_iteration);

 private:
  GraphImplPtr impl_{nullptr};
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_GRAPH_H_
