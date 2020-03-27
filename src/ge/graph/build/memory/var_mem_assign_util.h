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

#ifndef GE_GRAPH_BUILD_MEMORY_VAR_MEM_ASSIGN_UTIL_H_
#define GE_GRAPH_BUILD_MEMORY_VAR_MEM_ASSIGN_UTIL_H_

#include <string>

#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#include "graph/utils/node_utils.h"

namespace ge {
class VarMemAssignUtil {
 public:
  static Status AssignVarMemory(ge::ComputeGraphPtr &compute_graph);
  static Status AssignConstantOpMemory(ge::ComputeGraphPtr &compute_graph);
  static Status AssignStaticMemory2Node(ge::ComputeGraphPtr &compute_graph);
  static Status AssignVarAttr2Nodes(ge::ComputeGraphPtr &compute_graph);
  static Status AssignMemory2HasRefAttrNode(ge::ComputeGraphPtr &compute_graph);

 private:
  static Status AssignMemory2VariableNode(ge::ComputeGraphPtr &compute_graph);

  static Status SetOutVariableAttr(const ge::NodePtr &node, const ge::NodePtr &var_node, int index,
                                   uint64_t session_id);
  static Status DealExportVariableNode(const ge::NodePtr &node, const ge::NodePtr &var_node, uint64_t session_id);
  static Status DealVariableNode(uint32_t graph_id, const ge::NodePtr &node, uint64_t session_id);

  static Status DealBroadCastNode(uint32_t graph_id, const ge::NodePtr &node, const ge::InDataAnchorPtr &in_data_anchor,
                                  const ge::NodePtr &var_node, uint64_t session_id);
  static Status AssignData2Fp32Var(const ge::NodePtr &node, uint64_t session_id);

  static ge::NodePtr GetFinalTransNode(const ge::NodePtr &ref_node);

  static Status DealTransNode(const ge::NodePtr &final_trans_node);
  static Status DealExportTransNode(const ge::NodePtr &node, const ge::NodePtr &final_trans_node);
  static Status AssignData2VarRef(const ge::NodePtr &variable_ref, const std::string &src_var_name,
                                  uint64_t session_id);

  static Status SetOutTransNodeToAssign(const ge::NodePtr &node, const ge::NodePtr &final_trans_node, size_t index);
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_VAR_MEM_ASSIGN_UTIL_H_
