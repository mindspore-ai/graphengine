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

#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace {
Status MappingSubgraphInput(const ComputeGraphPtr &graph, const std::function<int(int data_index)> &input) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      continue;
    }

    int index = -1;
    if (!AttrUtils::GetInt(node->GetOpDesc(), "index", index)) {
      GELOGE(FAILED, "Failed to get index from data[%s]", node->GetName().c_str());
      return FAILED;
    }

    int parent_index = input(index);
    GELOGI("Generate subgraph input map for subgraph %s, data index %d, parent index %d",
           graph->GetName().c_str(), index, parent_index);
    if (!AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(FAILED, "Failed to set parent index for node %s", node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MappingSubgraphOutput(const ComputeGraphPtr &graph, const std::function<int(int retval_index)> &output) {
  const auto &output_node = graph->FindFirstNodeMatchType(NETOUTPUT);
  if (output_node == nullptr) {
    return SUCCESS;
  }

  const auto &op_desc = output_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (size_t index = 0; index < op_desc->GetInputsSize(); ++index) {
    int parent_index = output(index);
    GELOGI("Generate subgraph output map for subgraph %s, index %zu, parent index %d",
           graph->GetName().c_str(), index, parent_index);
    if (parent_index == -1) {
      continue;
    }

    GeTensorDescPtr tensor = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(tensor);
    if (!AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(FAILED, "Failed to set parent index for graph %s", graph->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MappingSubgraphIndex(const ComputeGraphPtr &graph,
                            const std::function<int(int data_index)> &input,
                            const std::function<int(int retval_index)> &output) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(input);
  GE_CHECK_NOTNULL(output);
  if (MappingSubgraphInput(graph, input) != SUCCESS) {
    GELOGE(FAILED, "Failed to mapping subgraph input for graph: %s", graph->GetName().c_str());
    return FAILED;
  }

  if (MappingSubgraphOutput(graph, output) != SUCCESS) {
    GELOGE(FAILED, "Failed to mapping subgraph output for graph: %s", graph->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status ParseSubgraphPostFnCase(const string &subgraph_name, const ComputeGraphPtr &graph) {
  return MappingSubgraphIndex(graph,
      [](int data_index) { return data_index + 1; },
      [](int retval_index) { return retval_index; });
}

Status ParseSubgraphPostFnIf(const string &subgraph_name, const ComputeGraphPtr &graph) {
  return MappingSubgraphIndex(graph,
      [](int data_index) { return data_index + 1; },
      [](int retval_index) { return retval_index; });
}

Status ParseSubgraphPostFnWhile(const string &subgraph_name, const ComputeGraphPtr &graph) {
  return MappingSubgraphIndex(graph,
      [](int data_index) { return data_index; },
      [&](int retval_index) { return (subgraph_name == "cond") ? -1 : retval_index; });
}

Status ParseSubgraphPostFnFor(const string &subgraph_name, const ComputeGraphPtr &graph) {
  return MappingSubgraphIndex(graph,
      [](int data_index) { return (data_index == 0) ? 0 : data_index + 2; },
      [](int retval_index) { return retval_index; });
}

Status ParseSubgraphPostFnPartitionedCall(const string &subgraph_name, const ComputeGraphPtr &graph) {
  return MappingSubgraphIndex(graph,
      [](int data_index) { return data_index; },
      [](int retval_index) { return retval_index; });
}
}

Status DataPass::PostParseSubgraph(const ComputeGraphPtr &graph, const string &ir_name, const NodePtr &parent_node) {
  using ParseSubgraphFunc = std::function<Status(const string &subgraph_name, const ComputeGraphPtr &graph)>;
  const static std::map<string, ParseSubgraphFunc> subgraph_handle = {
    {FOR, ParseSubgraphPostFnFor},
    {CASE, ParseSubgraphPostFnCase},
    {IF, ParseSubgraphPostFnIf},
    {_IF, ParseSubgraphPostFnIf},
    {STATELESSIF, ParseSubgraphPostFnIf},
    {WHILE, ParseSubgraphPostFnWhile},
    {_WHILE, ParseSubgraphPostFnWhile},
    {STATELESSWHILE, ParseSubgraphPostFnWhile},
    {PARTITIONEDCALL, ParseSubgraphPostFnPartitionedCall},
    {STATEFULPARTITIONEDCALL, ParseSubgraphPostFnPartitionedCall}
  };

  auto post_func_it = subgraph_handle.find(parent_node->GetType());
  if (post_func_it == subgraph_handle.end()) {
    GELOGE(FAILED, "The subgraph post func for node %s type %s is null.",
           parent_node->GetName().c_str(), parent_node->GetType().c_str());
    return FAILED;
  }

  if (post_func_it->second(ir_name, graph) != SUCCESS) {
    GELOGE(FAILED, "Failed to post process subgraph %s on node %s type %s",
           graph->GetName().c_str(), parent_node->GetName().c_str(), parent_node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

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

  return PostParseSubgraph(compute_graph, subgraph_name, parent_node);
}
}  // namespace ge
