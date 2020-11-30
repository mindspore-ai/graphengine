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
#include "graph/passes/mark_agnostic_pass.h"

#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
Status MarkAgnosticPass::Run(ComputeGraphPtr graph) {
  for (const auto &node : graph->GetDirectNode()) {
    auto node_type = NodeUtils::GetNodeType(*node);
    if (node_type == SWITCH || node_type == SWITCHN) {
      GELOGD("Mark format agnostic and continuous for switch node %s", node->GetName().c_str());
      const OpDescPtr op_desc = node->GetOpDesc();
      const GeTensorDescPtr op_tensor = op_desc->MutableInputDesc(0);
      if (op_tensor == nullptr) {
        GELOGD("Op: %s, Index:0,has no input", node->GetName().c_str());
        continue;
      }
      AttrUtils::SetInt(op_tensor, "_format_continuous", 1);
      AttrUtils::SetInt(node->GetOpDesc(), "_format_agnostic", 1);
      AttrUtils::SetListInt(node->GetOpDesc(), "_format_agnostic_except_input", std::vector<int64_t>({1}));
      continue;
    }
    if (node_type == IDENTITY) {
      GELOGD("Mark format agnostic for identity node %s", node->GetName().c_str());
      AttrUtils::SetInt(node->GetOpDesc(), "_format_agnostic", 1);
      continue;
    }
    if (node_type == REFMERGE || node_type == REFSWITCH) {
      GELOGD("Mark format agnostic for regmerge and refswitch node %s", node->GetName().c_str());
      AttrUtils::SetInt(node->GetOpDesc(), "_format_agnostic", 1);
      AttrUtils::SetListInt(node->GetOpDesc(), "_format_agnostic_except_output", std::vector<int64_t>({1}));
      continue;
    }
    if (node_type == MERGE) {
      GELOGD("Mark format agnostic and continuous for merge node %s", node->GetName().c_str());
      const auto &input_nodes = node->GetInAllNodes();
      /// Enter-----------+
      ///                 +-> Merge
      /// NextIteration---+
      if (input_nodes.size() == 2) {
        if (input_nodes.at(0)->GetType() == ENTER && input_nodes.at(1)->GetType() == NEXTITERATION) {
          continue;
        }
      }
      const OpDescPtr op_desc = node->GetOpDesc();
      const GeTensorDescPtr op_tensor = op_desc->MutableOutputDesc(0);
      if (op_tensor == nullptr) {
        GELOGD("Op: %s, Index:0,has no output", node->GetName().c_str());
        continue;
      }
      AttrUtils::SetInt(op_tensor, "_format_continuous", 1);

      // Merge----------->NetOutput only set format_cofntinuous attr
      const auto &output_nodes = node->GetOutAllNodes();
      if (output_nodes.size() > 0) {
        if (output_nodes.at(0)->GetType() == NETOUTPUT) {
          continue;
        }
      }
      AttrUtils::SetInt(node->GetOpDesc(), "_format_agnostic", 1);
      AttrUtils::SetListInt(node->GetOpDesc(), "_format_agnostic_except_output", std::vector<int64_t>({1}));
      continue;
    }
  }
  return SUCCESS;
}
}  // namespace ge