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

#include "graph/passes/mark_agnostic_pass.h"

#include "utils/node_utils.h"

namespace ge {
Status MarkAgnosticPass::Run(ComputeGraphPtr graph) {
  for (const auto &node : graph->GetDirectNode()) {
    auto node_type = NodeUtils::GetNodeType(*node);
    if (node_type == SWITCH || node_type == REFSWITCH || node_type == SWITCHN) {
      GELOGD("Mark format agnostic for switch ndoe %s", node->GetName().c_str());
      AttrUtils::SetInt(node->GetOpDesc(), "_format_agnostic", 1);
      AttrUtils::SetListInt(node->GetOpDesc(), "_format_agnostic_except_input", std::vector<int64_t>({1}));
      continue;
    }
    if (node_type == MERGE || node_type == REFMERGE) {
      GELOGD("Mark format agnostic for merge node %s", node->GetName().c_str());
      AttrUtils::SetInt(node->GetOpDesc(), "_format_agnostic", 1);
      AttrUtils::SetListInt(node->GetOpDesc(), "_format_agnostic_except_output", std::vector<int64_t>({1}));
      continue;
    }
  }
  return SUCCESS;
}
}