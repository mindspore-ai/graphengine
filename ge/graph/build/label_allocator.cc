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

#include "label_allocator.h"

#include "framework/common/types.h"
#include "common/util.h"
#include "common/ge_inner_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/label/label_maker.h"

namespace ge {
LabelAllocator::LabelAllocator(const ComputeGraphPtr &graph) : compute_graph_(graph) {}

Status LabelAllocator::AssignFunctionalLabels(uint32_t &label_index) {
  if (compute_graph_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "ComputeGraph not set, Assign labels failed.");
    return INTERNAL_ERROR;
  }

  // Add label task for sub graph.
  GELOGI("AssignFunctionalLabels start: %s.", compute_graph_->GetName().c_str());
  std::set<NodePtr> functional_nodes;
  for (auto graph : compute_graph_->GetAllSubgraphs()) {
    if (!CollectFunctionalNode(graph, functional_nodes)) {
      return INTERNAL_ERROR;
    }
  }

  // Add label for functional op.
  label_index = 0;
  for (auto node : functional_nodes) {
    LabelMakerPtr maker = LabelMakerFactory::Instance().Create(node->GetType(), compute_graph_, node);
    if (maker == nullptr) {
      GELOGE(INTERNAL_ERROR, "Node: %s label maker not registed.", node->GetType().c_str());
      return INTERNAL_ERROR;
    }

    if (maker->Run(label_index) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Node: %s run label maker failed.", node->GetType().c_str());
      return INTERNAL_ERROR;
    }
  }

  GELOGI("AssignFunctionalLabels success.");
  return SUCCESS;
}

bool LabelAllocator::CollectFunctionalNode(ComputeGraphPtr &graph, std::set<NodePtr> &functional_nodes) {
  if (graph == nullptr) {
    GELOGE(INTERNAL_ERROR, "Sub ComputeGraph is null.");
    return false;
  }

  NodePtr parent = graph->GetParentNode();
  if (parent == nullptr) {
    GELOGE(INTERNAL_ERROR, "ComputeGraph owner not set: %s.", graph->GetName().c_str());
    return false;
  }

  (void)functional_nodes.insert(parent);  // unique functional node.
  return true;
}
}  // namespace ge
