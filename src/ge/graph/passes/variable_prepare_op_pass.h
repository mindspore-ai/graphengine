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

#ifndef GE_GRAPH_PASSES_VARIABLE_PREPARE_OP_PASS_H_
#define GE_GRAPH_PASSES_VARIABLE_PREPARE_OP_PASS_H_

#include <map>
#include <string>

#include "framework/common/ge_inner_error_codes.h"
#include "inc/graph_pass.h"

namespace ge {
class VariablePrepareOpPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph);

 private:
  Status DealVariableNode(ge::NodePtr &node);
  Status DealWritableNode(ge::NodePtr &writable_node, ge::NodePtr &var_node, int out_index);
  NodePtr GetFinalWritableNode(ge::NodePtr &writable_node, int &out_index);
  Status AddVariableRef(ge::NodePtr &node, ge::NodePtr &var_node, int index);
  NodePtr CreatVariableRef(const std::string &variable_ref_name, ge::NodePtr &var_node);
  int GetWritableNodeOutIndex(const NodePtr &node, int input_index);
  Status UpdateAssignOpDesc(const ge::NodePtr &node);
  void GenerateRefTypeAndInputOutputMap(const NodePtr &node);
  int FindRefOutIndex(const std::string &node_type, int input_index,
                      const std::map<std::string, std::map<int, int>> &ref_map);

  std::map<std::string, std::map<int, int>> ref_input_output_map_;
  static std::map<std::string, std::map<int, int>> ref_node_without_prototype_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_VARIABLE_PREPARE_OP_PASS_H_
