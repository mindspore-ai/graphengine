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

#include "graph/passes/global_step_insert_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "common/ge/ge_util.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/passes/pass_utils.h"

namespace ge {
NodePtr GlobalStepInsertPass::InsertOp(ComputeGraphPtr &compute_graph,
                                       const string &node_type,
                                       const string &node_name,
                                       const std::vector<GeTensorDesc> &input_list,
                                       const std::vector<GeTensorDesc> &output_list) {
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, node_type);
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(FAILED,"Make OpDesc failed"); return nullptr);

  for (auto &input_desc : input_list) {
    graphStatus graph_status = op_desc->AddInputDesc(input_desc);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add node:%s intput desc failed, error=%u.", node_name.c_str(), graph_status);
      return nullptr;
    }
  }

  for (auto &output_desc : output_list) {
    graphStatus graph_status = op_desc->AddOutputDesc(output_desc);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add node:%s output desc failed, error=%u.", node_name.c_str(), graph_status);
      return nullptr;
    }
  }

  GE_IF_BOOL_EXEC(compute_graph == nullptr, GELOGE(FAILED,"compute_graph is nullptr"); return nullptr);
  NodePtr node = compute_graph->AddNode(op_desc);
  GE_IF_BOOL_EXEC(node == nullptr,
    GELOGE(FAILED, "add node failed, name:%s, type:%s.", node_name.c_str(), node_type.c_str());
    return nullptr);

  GELOGI("Insert op success, name:%s, type:%s.", node_name.c_str(), node_type.c_str());
  return node;
}

Status GlobalStepInsertPass::Run(ComputeGraphPtr compute_graph) {
  NodePtr output_node = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  if (output_node == nullptr) {
    GELOGD("Node type %s can't be found in graph %u", NETOUTPUT, compute_graph->GetGraphID());
    return SUCCESS;
  }

  if (compute_graph->GetParentGraph() != nullptr) {
    GELOGD("Subgraph %s no need global step variable.", compute_graph->GetName().c_str());
    return SUCCESS;
  }

  NodePtr exist_node = compute_graph->FindNode(NODE_NAME_GLOBAL_STEP);
  if (exist_node != nullptr) {
    GELOGD("Node %s already exist, no need add.", NODE_NAME_GLOBAL_STEP.c_str());
    return SUCCESS;
  }
  // set global step tensor desc
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_UINT64);
  std::vector<GeTensorDesc> input_desc_list = {};
  std::vector<GeTensorDesc> output_desc_list = {tensor_desc};
  NodePtr global_step = InsertOp(compute_graph, VARIABLE, NODE_NAME_GLOBAL_STEP,
                                 input_desc_list, output_desc_list);
  if (global_step == nullptr) {
    GELOGE(FAILED, "Add global_step node failed, global_step is null.");
    return FAILED;
  }

  // add ctrl edges
  graphStatus add_ret = GraphUtils::AddEdge(global_step->GetOutControlAnchor(), output_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add global_step to netoutput edge failed, add_ret=%u.", add_ret);
    return FAILED;
  }
  GELOGD("Add global_step to netoutput edge in graph %u success", compute_graph->GetGraphID());
  return SUCCESS;
}
}  // namespace ge