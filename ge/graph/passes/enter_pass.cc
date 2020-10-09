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

#include "graph/passes/enter_pass.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/utils/graph_utils.h"

namespace ge {
Status EnterPass::Run(NodePtr &node) {
  GELOGD("EnterPass running");
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "param [node] must not be null.");
    return PARAM_INVALID;
  }

  if ((node->GetType() != ENTER) && (node->GetType() != REFENTER)) {
    return SUCCESS;
  }

  // enter node has only one input
  if (node->GetInDataNodes().empty()) {
    GELOGE(PARAM_INVALID, "enter_node %s has no input", node->GetName().c_str());
    return PARAM_INVALID;
  }
  NodePtr in_node = node->GetInDataNodes().at(0);
  if (in_node == nullptr) {
    GELOGE(PARAM_INVALID, "param [in_node] must not be null");
    return PARAM_INVALID;
  }

  if ((in_node->GetType() != CONSTANT) && (in_node->GetType() != CONSTANTOP)) {
    return SUCCESS;
  }

  bool need_remove_flag = in_node->GetInControlNodes().empty() &&
                          node->GetInControlNodes().empty() &&
                          node->GetOutDataNodes().empty();
  if (need_remove_flag) {
    for (auto &out_ctrl_node : node->GetOutControlNodes()) {
      if (out_ctrl_node == nullptr) {
        continue;
      }
      if (GraphUtils::RemoveEdge(node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Remove Enter ctrl output fail, %s->%s",
               node->GetName().c_str(), out_ctrl_node->GetName().c_str());
        return FAILED;
      }
    }
  }

  GELOGD("EnterPass success");
  return SUCCESS;
}
}  // namespace ge
