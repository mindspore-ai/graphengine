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

#ifndef GE_GRAPH_PASSES_FLOW_CTRL_PASS_H_
#define GE_GRAPH_PASSES_FLOW_CTRL_PASS_H_

#include <string>
#include <vector>

#include "common/ge_inner_error_codes.h"
#include "inc/graph_pass.h"

namespace ge {
///
/// Add flow control to the computeGraph
///
class FlowCtrlPass : public GraphPass {
 public:
  ///
  /// Add flow control to the computeGraph.
  /// @param compute_graph graph
  /// @return SUCCESS: do success
  ///         NOT_CHANGED : do nothing
  ///         Other: failed
  ///
  Status Run(ComputeGraphPtr compute_graph) override;

 private:
  ///
  /// Universal insert node to graph.
  /// @param compute_graph graph
  /// @param node_type inserted node type
  /// @param node_name inserted node name
  /// @param input_list input desc list
  /// @param output_list output desc list
  /// @return the inserted node. if insert failed return nullptr.
  ///
  NodePtr InsertOp(ComputeGraphPtr &compute_graph, const string &node_type, const string &node_name,
                   const std::vector<GeTensorDesc> &input_list, const std::vector<GeTensorDesc> &output_list);

  ///
  /// used for insert assign and assign add node.
  /// include add input desc info.
  /// @param compute_graph graph
  /// @param node_type node type(assign/assignAdd)
  /// @param node_name node name
  /// @param ref_node assign input0
  /// @param value_node assign input1
  /// @return the inserted node. if insert failed return nullptr.
  ///
  NodePtr InsertAssignOp(ComputeGraphPtr &compute_graph, const string &node_type, const string &node_name,
                         const NodePtr &ref_node, const NodePtr &value_node);

  ///
  /// insert StreamSwitch to graph.
  /// @param compute_graph graph
  /// @param switch_name inserted StreamSwitch node name
  /// @param loop_cond loop condition
  /// @param iter_per_loop iter per loop
  /// @return the inserted node. if insert failed return nullptr.
  ///
  NodePtr InsertStreamSwitchOp(ComputeGraphPtr &compute_graph, const string &switch_name, const NodePtr &loop_cond,
                               const NodePtr &iter_per_loop);

  ///
  /// check and add variable node to graph.
  /// if the variable is exists, do nothing.
  /// @param compute_graph graph
  /// @param name inserted variable node name
  /// @return the variable node. if insert failed return nullptr.
  ///
  NodePtr AddVariableNode(ComputeGraphPtr &compute_graph, const string &name);

  ///
  /// insert GlobalStepAssignAdd to graph.
  /// just for ME, please remove when ME do itself.
  /// @param compute_graph graph
  /// @param pre_node pre node
  /// @param global_step global step node
  /// @param loop_increment_node loop increment node
  /// @return the GlobalStepAssignAdd node. if insert failed return nullptr.
  ///
  NodePtr InsertGlobalStepAssignAddOp(ComputeGraphPtr &compute_graph, NodePtr &pre_node, const NodePtr &global_step,
                                      const NodePtr &loop_increment_node);

  ///
  /// create switch true branch for big cycle.
  /// @param compute_graph graph
  /// @param loop_cond_node loop condition node
  /// @param loop_increment_node loop increment node
  /// @param switch_node switch node
  /// @return SUCCESS: do success
  ///         Other: failed
  ///
  Status CreateIterCtrlTrueBranch(ComputeGraphPtr &compute_graph, const NodePtr &loop_cond_node,
                                  const NodePtr &loop_increment_node, NodePtr &switch_node);

  ///
  /// create switch false branch for big cycle.
  /// @param compute_graph graph
  /// @param loop_cond_node loop condition node
  /// @param loop_reset_node loop reset node
  /// @param switch_node switch node
  /// @return SUCCESS: do success
  ///         Other: failed
  ///
  Status CreateIterCtrlFalseBranch(ComputeGraphPtr &compute_graph, const NodePtr &loop_cond_node,
                                   const NodePtr &loop_reset_node, NodePtr &switch_node);

  ///
  /// add Fp/Bp iterator ctrl nodes(big cycle).
  /// @param compute_graph graph
  /// @param pre_node pre node(netoutput node)
  /// @return SUCCESS: do success
  ///         Other: failed
  ///
  Status AddFpBpIteratorCtrl(ComputeGraphPtr &compute_graph, NodePtr &pre_node);

  ///
  /// add special iterator ctrl nodes(small cycle).
  /// @param compute_graph graph
  /// @param loop_after_node pre node(iterate node)
  /// @return SUCCESS: do success
  ///         Other: failed
  ///
  Status AddSpecialNodeIteratorCtrl(ComputeGraphPtr &compute_graph, NodePtr &loop_after_node);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FLOW_CTRL_PASS_H_
