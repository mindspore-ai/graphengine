/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef GE_REMOVE_NODES_PASS_H_
#define GE_REMOVE_NODES_PASS_H_
#include "graph/passes/base_pass.h"

namespace ge {
class RemoveNodesPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;
  RemoveNodesPass &AddNodeType(const std::string &node_type, std::initializer_list<int> arg = {0});
  RemoveNodesPass &AddAttrName(const std::string &attr_name, std::initializer_list<int> arg = {0});

 private:
  std::map<std::string, std::initializer_list<int>> remove_node_types_to_arg_;
  std::map<std::string, std::initializer_list<int>> remove_node_attr_names_to_arg_;
};
}  // namespace ge
#endif //GE_REMOVE_NODES_PASS_H_
