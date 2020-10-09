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

#include "graph/passes/data_pass.h"

#include <string>

#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "register/op_registry.h"

namespace ge {
Status DataPass::Run(ComputeGraphPtr compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);
  if (compute_graph->GetParentNode() == nullptr) {      // for subgraph post process.
    return SUCCESS;
  }

  for (const NodePtr &node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (node->GetType() == DATA) {
      uint32_t parent_index = 0;
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
        break;        // parent_index not set, Graph from IR.
      }

      return SUCCESS; // Graph from Parser.
    }
  }

  std::string subgraph_name;
  const auto &parent_node = compute_graph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node->GetOpDesc());
  auto func_desc = parent_node->GetOpDesc();
  GE_CHK_STATUS_RET(func_desc->GetSubgraphNameByInstanceName(compute_graph->GetName(), subgraph_name),
                    "Subgraph: %s get subgraph name failed.", compute_graph->GetName().c_str());

  GELOGI("Post process for subgraph %s, Subgraph name: %s, Parent name: %s, Parent type: %s.",
         compute_graph->GetName().c_str(), subgraph_name.c_str(), parent_node->GetName().c_str(),
         parent_node->GetType().c_str());

  const auto &parent_graph = compute_graph->GetParentGraph();
  GE_CHECK_NOTNULL(parent_graph);
  for (const NodePtr &node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if ((node->GetType() == VARIABLE) || (node->GetType() == VARIABLEV2) || (node->GetType() == NETOUTPUT)) {
      continue;
    }

    node->GetOpDesc()->SetName(parent_node->GetName() + "_" + compute_graph->GetName() + "/" + node->GetName());
  }

  auto post_func = domi::OpRegistry::Instance()->GetParseSubgraphPostFunc(parent_node->GetType());
  if (post_func == nullptr) {
    GELOGW("The subgraph post func for node %s type %s is null.",
           parent_node->GetName().c_str(), parent_node->GetType().c_str());
    return SUCCESS;
  }

  auto graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  auto ret = post_func(subgraph_name, graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Failed to post-process subgraph %s on node %s type %s",
           graph.GetName().c_str(), parent_node->GetName().c_str(), parent_node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}
}  // namespace ge
