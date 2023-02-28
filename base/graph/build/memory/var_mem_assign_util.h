/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/node_utils.h"

namespace ge {
using GraphToNodeMap = std::map<ge::ComputeGraphPtr, std::map<std::string, ge::NodePtr>>;

class VarMemAssignUtil {
 public:
  static Status AssignVarMemory(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignConstantOpMemory(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignStaticMemory2Node(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignVarAttr2Nodes(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignMemory2HasRefAttrNode(const ge::ComputeGraphPtr &compute_graph);
  static Status AssignData2Fp32Var(const ge::NodePtr &node, const uint64_t session_id);

 private:
  static Status AssignMemory2VariableNode(const ge::ComputeGraphPtr &compute_graph);

  static Status SetOutVariableAttr(const ge::NodePtr &node, const ge::NodePtr &var_node, const size_t index,
                                   const uint64_t session_id);
  static Status DealExportVariableNode(const ge::NodePtr &node, const ge::NodePtr &var_node, const uint64_t session_id,
                                       const uint32_t depth = 0U);
  static Status DealVariableNode(const uint32_t graph_id, const ge::NodePtr &node, const uint64_t session_id);

  static Status DealBroadCastNode(const uint32_t graph_id, const ge::NodePtr &node,
                                  const ge::InDataAnchorPtr &in_data_anchor,
                                  const ge::NodePtr &var_node, const uint64_t session_id);

  static ge::NodePtr GetFinalTransNode(const ge::NodePtr &trans_node, const uint32_t depth = 0U);

  static Status DealTransNode(const ge::NodePtr &final_trans_node);
  static Status DealExportTransNode(const ge::NodePtr &node, const ge::NodePtr &final_trans_node,
                                    const uint32_t depth = 0U);
  static Status AssignData2VarRef(const ge::NodePtr &has_ref_attr_node, const std::string &src_var_name,
                                  const uint64_t session_id, const uint32_t out_index, GraphToNodeMap &graph_to_node);

  static Status SetOutTransNodeToAssign(const ge::NodePtr &node, const ge::NodePtr &final_trans_node,
                                        const size_t index);
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_VAR_MEM_ASSIGN_UTIL_H_
