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

#include "remove_nodes_pass.h"
#include "debug/ge_log.h"
#include "inc/framework/common/util.h"
#include "inc/graph/utils/node_utils.h"

namespace ge {
Status RemoveNodesPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto node_type = NodeUtils::GetNodeType(*node);
  auto type_iter = remove_node_types_to_arg_.find(node_type);
  if (type_iter != remove_node_types_to_arg_.end()) {
    GELOGI("Remove node %s by type %s", node->GetName().c_str(), node_type.c_str());
    return IsolateAndDeleteNode(node, type_iter->second);
  }
  for (const auto &attr_name_to_arg : remove_node_attr_names_to_arg_) {
    if (AttrUtils::HasAttr(node->GetOpDesc(), attr_name_to_arg.first)) {
      GELOGI("Remove node %s by attr name %s", node->GetName().c_str(), attr_name_to_arg.first.c_str());
      return IsolateAndDeleteNode(node, attr_name_to_arg.second);
    }
  }

  return SUCCESS;
}
RemoveNodesPass &RemoveNodesPass::AddNodeType(const string &node_type, std::initializer_list<int> arg) {
  remove_node_types_to_arg_[node_type] = std::move(arg);
  return *this;
}
RemoveNodesPass &RemoveNodesPass::AddAttrName(const string &attr_name, std::initializer_list<int> arg) {
  remove_node_attr_names_to_arg_[attr_name] = std::move(arg);
  return *this;
}
}  // namespace ge