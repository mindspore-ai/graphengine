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

#include "graph/passes/end_graph_pass.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/tensor_utils.h"
#include "init/gelib.h"
#include "common/ge/ge_util.h"
#include "graph/debug/ge_attr_define.h"

using domi::ENDGRAPH;
using domi::NODE_NAME_END_GRAPH;
using domi::NODE_NAME_NET_OUTPUT;

namespace ge {
Status EndGraphPass::Run(ge::ComputeGraphPtr graph) {
  GELOGI("EndGraphPass Run.");
  if (graph == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "Compute graph is null.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  auto gelib = GELib::GetInstance();
  bool head_stream = (gelib == nullptr) ? false : gelib->HeadStream();
  if (!head_stream) {
    GELOGI("Configured head stream: %d, No need EndGraph.", head_stream);
    return SUCCESS;
  }

  NodePtr net_output_node = graph->FindNode(NODE_NAME_NET_OUTPUT);
  if (net_output_node == nullptr) {
    GELOGI("No output node found.");
    return SUCCESS;
  }

  OpDescPtr op_desc = MakeShared<OpDesc>(NODE_NAME_END_GRAPH, ENDGRAPH);
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Create EndGraph op:%s.", op_desc->GetName().c_str());
  (void)AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::move(std::vector<std::string>()));
  NodePtr end_graph_node = graph->AddNode(op_desc);
  if (end_graph_node == nullptr) {
    GELOGI("Add EndGraph:%s node to Graph fail.", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (GraphUtils::AddEdge(net_output_node->GetOutControlAnchor(), end_graph_node->GetInControlAnchor()) != SUCCESS) {
    GELOGI("Add ctrl edge to EndGraph:%s fail.", end_graph_node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGI("EndGraphPass Leave.");
  return SUCCESS;
}
}  // namespace ge
