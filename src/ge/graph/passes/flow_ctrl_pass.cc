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

#include "graph/passes/flow_ctrl_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common/omg_util.h"
#include "common/ge/ge_util.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/passes/pass_utils.h"

namespace ge {
// when namespace change to ge, please delete the using code.
Status FlowCtrlPass::Run(ComputeGraphPtr compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);

  if (!PassUtils::IsNeedTrainIteFlowCtrl(compute_graph)) {
    GELOGI("No need FlowCtrl for graph %u", compute_graph->GetGraphID());
    return NOT_CHANGED;
  }

  GELOGI("FlowCtrl pass begin");
  bool graph_change = false;
  // 1. Add FP/BP flow ctrl (big cycle)
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    uint32_t true_stream_id = 0;
    bool is_found = AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_TRUE_BRANCH_STREAM, true_stream_id);
    // FP/BP cycle flag is true_stream_id == 0
    if (is_found && (true_stream_id == TRUE_STREAM_ID)) {
      // Add big cycle
      Status ret = AddFpBpIteratorCtrl(compute_graph, node);
      if (ret != SUCCESS) {
        GELOGE(ret, "AddFpBpIteratorCtrl fail, node: %s.", node->GetName().c_str());
        return ret;
      }
      graph_change = true;
      // only one big cycle, so break.
      break;
    }
  }

  // 2. Add special node flow ctrl. eg, IteratorGetNext. (small cycle)
  //    NOTE: Small cycle share the variables with big cycle.
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    bool need_cycle_flag = false;
    bool is_found = AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_STREAM_CYCLE_EVENT_FLAG, need_cycle_flag);
    // small cycle flag is need_stream_cycle_event == true
    if (is_found && need_cycle_flag) {
      Status ret = AddSpecialNodeIteratorCtrl(compute_graph, node);
      if (ret != SUCCESS) {
        GELOGE(ret, "AddSpecialNodeIteratorCtrl fail, node: %s.", node->GetName().c_str());
        return ret;
      }
      graph_change = true;
    }
  }
  GELOGI("FlowCtrl pass end, graph is %s.", graph_change ? "changed" : "not changed");
  return graph_change ? SUCCESS : NOT_CHANGED;
}

bool FlowCtrlPass::CheckMultiDataSet(ComputeGraphPtr &compute_graph) {
  int data_set_num = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    string type;
    bool is_found = AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
    if (is_found && type == "IteratorV2") {
      data_set_num++;
    }
  }
  GELOGI("The ComputeGraph contain %d dataSet.", data_set_num);
  return (data_set_num > 1) ? true : false;
}

NodePtr FlowCtrlPass::InsertOp(ComputeGraphPtr &compute_graph, const string &node_type, const string &node_name,
                               const std::vector<GeTensorDesc> &input_list,
                               const std::vector<GeTensorDesc> &output_list) {
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, node_type);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Make OpDesc failed, name:%s, type:%s.", node_name.c_str(), node_type.c_str());
    return nullptr;
  }

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

  GE_IF_BOOL_EXEC(compute_graph == nullptr, DOMI_LOGE("compute_graph is nullptr"); return nullptr);
  NodePtr node = compute_graph->AddNode(op_desc);
  if (node == nullptr) {
    GELOGE(FAILED, "add node failed, name:%s, type:%s.", node_name.c_str(), node_type.c_str());
    return nullptr;
  }

  GELOGI("Insert op success, name:%s, type:%s.", node_name.c_str(), node_type.c_str());
  return node;
}

NodePtr FlowCtrlPass::InsertStreamSwitchOp(ComputeGraphPtr &compute_graph, const string &switch_name,
                                           const NodePtr &loop_cond, const NodePtr &iter_per_loop) {
  GE_IF_BOOL_EXEC(loop_cond == nullptr || loop_cond->GetOpDesc() == nullptr, GELOGE(FAILED, "loop_cond is null");
                  return nullptr);
  GE_IF_BOOL_EXEC(iter_per_loop == nullptr || iter_per_loop->GetOpDesc() == nullptr,
                  GELOGE(FAILED, "iter_per_loop is nullptr");
                  return nullptr);
  std::vector<GeTensorDesc> input_desc_list = {loop_cond->GetOpDesc()->GetOutputDesc(0),
                                               iter_per_loop->GetOpDesc()->GetOutputDesc(0)};
  std::vector<GeTensorDesc> output_desc_list;
  NodePtr stream_switch = InsertOp(compute_graph, STREAMSWITCH, switch_name, input_desc_list, output_desc_list);
  if (stream_switch == nullptr) {
    GELOGE(FAILED, "InsertStreamSwitchOp failed, name:%s.", switch_name.c_str());
    return nullptr;
  }

  // set input 0
  graphStatus add_ret = GraphUtils::AddEdge(loop_cond->GetOutDataAnchor(0), stream_switch->GetInDataAnchor(0));
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add loop_cond_node to switch_node:%s edge failed, ret = %u.", switch_name.c_str(), add_ret);
    return nullptr;
  }

  // set input 1
  add_ret = GraphUtils::AddEdge(iter_per_loop->GetOutDataAnchor(0), stream_switch->GetInDataAnchor(1));
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add iter_per_loop_node to switch_node:%s edge failed, ret = %u.", switch_name.c_str(), add_ret);
    return nullptr;
  }

  // stream switch op need switch cond by attr.
  GE_IF_BOOL_EXEC(
    !AttrUtils::SetInt(stream_switch->GetOpDesc(), ATTR_NAME_STREAM_SWITCH_COND, static_cast<int64_t>(RT_LESS)),
    DOMI_LOGE("set ATTR_NAME_STREAM_SWITCH_COND failed");
    return nullptr);

  return stream_switch;
}

NodePtr FlowCtrlPass::AddVariableNode(ComputeGraphPtr &compute_graph, const string &name) {
  GE_IF_BOOL_EXEC(compute_graph == nullptr, DOMI_LOGE("compute_graph is nullptr"); return nullptr);
  NodePtr exist_node = compute_graph->FindNode(name);
  if (exist_node != nullptr) {
    GELOGD("Node %s already exist, no need add.", name.c_str());
    return exist_node;
  }
  // fetch and set tensor desc
  GeTensorDesc tensor_desc;
  if (ge::VarManager::Instance(compute_graph->GetSessionID()) == nullptr) {
    return nullptr;
  }
  Status ret = ge::VarManager::Instance(compute_graph->GetSessionID())->GetCurVarDesc(name, tensor_desc);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Get var desc fail, name:%s", name.c_str());
    return nullptr;
  }
  std::vector<GeTensorDesc> input_desc_list;
  std::vector<GeTensorDesc> output_desc_list = {tensor_desc};
  // insert node
  return InsertOp(compute_graph, VARIABLE, name, input_desc_list, output_desc_list);
}

Status FlowCtrlPass::AddGlobalStepVariableNode(ComputeGraphPtr &compute_graph) {
  NodePtr output_node = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  if (output_node == nullptr) {
    GELOGD("Node type %s can't be found in graph %u", NETOUTPUT, compute_graph->GetGraphID());
    return SUCCESS;
  }
  // Global step just add to main graph's netoutput node.And the main graph must be known shape
  if ((compute_graph->GetParentGraph() != nullptr) ||
      ((compute_graph->GetParentGraph() == nullptr) && (GraphUtils::IsUnknownShapeGraph(compute_graph)))) {
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
  NodePtr global_step = InsertOp(compute_graph, VARIABLE, NODE_NAME_GLOBAL_STEP, input_desc_list, output_desc_list);
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

NodePtr FlowCtrlPass::InsertAssignOp(ge::ComputeGraphPtr &compute_graph, const string &node_type,
                                     const string &node_name, const NodePtr &ref_node, const NodePtr &value_node) {
  GE_IF_BOOL_EXEC(ref_node == nullptr || value_node == nullptr || ref_node->GetOpDesc() == nullptr ||
                    value_node->GetOpDesc() == nullptr,
                  GELOGE(FAILED, "ref node or value node is null");
                  return nullptr);
  GeTensorDesc ref_tensor_desc = ref_node->GetOpDesc()->GetOutputDesc(0);
  GeTensorDesc val_tensor_desc = value_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<GeTensorDesc> input_desc_list = {ref_tensor_desc, val_tensor_desc};
  std::vector<GeTensorDesc> output_desc_list = {ref_tensor_desc};
  NodePtr assign_node = InsertOp(compute_graph, node_type, node_name, input_desc_list, output_desc_list);
  if (assign_node == nullptr) {
    GELOGE(FAILED, "Insert node %s(%s) failed.", node_name.c_str(), node_type.c_str());
    return nullptr;
  }
  // assign node input 0 = ref_node
  graphStatus add_ret = GraphUtils::AddEdge(ref_node->GetOutDataAnchor(0), assign_node->GetInDataAnchor(0));
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add ref_node to %s edge failed, add_ret=%u.", node_name.c_str(), add_ret);
    return nullptr;
  }
  // assign input 1 = value_node
  add_ret = GraphUtils::AddEdge(value_node->GetOutDataAnchor(0), assign_node->GetInDataAnchor(1));
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add value_node to %s edge failed, add_ret=%u.", node_name.c_str(), add_ret);
    return nullptr;
  }
  (void)ge::AttrUtils::SetBool(assign_node->GetOpDesc(), ATTR_NEED_COMPILE, true);

  return assign_node;
}

Status FlowCtrlPass::CreateIterCtrlTrueBranch(ComputeGraphPtr &compute_graph, const NodePtr &loop_cond_node,
                                              const NodePtr &loop_inc_node, NodePtr &switch_node) {
  /*
   *           loopCond
   *                |
   *                v
   * switch --> AssignAdd --> active
   *                ^
   *                |
   *         loopIncrement
   */
  // Insert AssignAdd node
  NodePtr assign_add_node =
    InsertAssignOp(compute_graph, ASSIGNADD, NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD, loop_cond_node, loop_inc_node);
  if (assign_add_node == nullptr || switch_node == nullptr) {
    GELOGE(PARAM_INVALID, "assign add node or switch node is null");
    return FAILED;
  }

  string active_name = switch_node->GetName() + "_StreamActive";
  // add attr for stream assign model to break branch.
  GE_CHK_STATUS_RET(SetStreamLabel(assign_add_node, active_name), "set stream label failed");

  // used for stream assign to find true branch
  GE_CHK_STATUS_RET(SetActiveLabelList(switch_node, {active_name}), "set active label list failed");

  // 2. Insert active node
  NodePtr active_node = InsertOp(compute_graph, STREAMACTIVE, active_name, {}, {});
  if (active_node == nullptr) {
    GELOGE(FAILED, "Insert stream active node:%s for IterCtrlTrueStream failed.", active_name.c_str());
    return FAILED;
  }
  GE_CHK_STATUS_RET(SetStreamLabel(active_node, active_name), "set stream label failed");
  GE_IF_BOOL_EXEC(!AttrUtils::SetBool(active_node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE, true),
                  DOMI_LOGE("set ATTR_NAME_IS_LOOP_ACTIVE failed");
                  return FAILED);

  // add ctrl edges
  graphStatus add_ret = GraphUtils::AddEdge(switch_node->GetOutControlAnchor(), assign_add_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add switch_node to assign_add_node ctrl edge failed, add_ret=%u.", add_ret);
    return FAILED;
  }

  add_ret = GraphUtils::AddEdge(assign_add_node->GetOutControlAnchor(), active_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add assign_add_node to active_node ctrl edge failed, add_ret=%u.", add_ret);
    return FAILED;
  }

  GELOGI("CreateIterCtrlTrueBranch success. StreamActive op:%s.", active_node->GetName().c_str());
  return SUCCESS;
}

Status FlowCtrlPass::CreateIterCtrlFalseBranch(ComputeGraphPtr &compute_graph, const NodePtr &loop_cond_node,
                                               const NodePtr &loop_reset_node, NodePtr &switch_node) {
  /*
   *           loopCond
   *                |
   *                v
   *   switch --> Assign --> active --> ModelExit
   *                ^
   *                |
   *            loopReset
   */
  // Insert Assign node and ctrl edge
  NodePtr assign_node =
    InsertAssignOp(compute_graph, ASSIGN, NODE_NAME_FLOWCTRL_LOOP_ASSIGN, loop_cond_node, loop_reset_node);
  if (assign_node == nullptr || switch_node == nullptr) {
    GELOGE(PARAM_INVALID, "assign_node or switch node is null");
    return FAILED;
  }

  GE_CHK_STATUS_RET(SetStreamLabel(assign_node, switch_node->GetName()), "set stream label failed");

  graphStatus add_ret = GraphUtils::AddEdge(switch_node->GetOutControlAnchor(), assign_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add switch_node to assign_node ctrl edge failed, add_ret=%u.", add_ret);
    return FAILED;
  }

  if (CheckMultiDataSet(compute_graph)) {
    GELOGI("Multi dataSae exist, model_exit node is need.");
    // 2. Insert active node and add ctrl edge
    string active_name = switch_node->GetName() + "_StreamExitActive";
    NodePtr active_node = InsertOp(compute_graph, STREAMACTIVE, active_name, {}, {});
    if (active_node == nullptr) {
      GELOGE(FAILED, "Insert stream active node:%s for IterCtrlTrueStream failed.", active_name.c_str());
      return FAILED;
    }
    GE_CHK_STATUS_RET(SetStreamLabel(active_node, switch_node->GetName()), "set stream label failed");
    GE_CHK_STATUS_RET(SetSwitchBranchNodeLabel(active_node, switch_node->GetName()),
                      "set switch branch node label failed");

    string model_exit_name = switch_node->GetName() + "_ModelExit";
    GE_CHK_STATUS_RET(SetActiveLabelList(active_node, {model_exit_name}), "set active label list failed");

    add_ret = GraphUtils::AddEdge(assign_node->GetOutControlAnchor(), active_node->GetInControlAnchor());
    if (add_ret != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add assign_node to active_node ctrl edge failed, add_ret=%u.", add_ret);
      return FAILED;
    }

    // 3. Insert model exit node and add ctrl edge
    NodePtr model_exit_node = InsertOp(compute_graph, MODELEXIT, model_exit_name, {}, {});
    if (model_exit_node == nullptr) {
      GELOGE(FAILED, "Insert model_exit node:%s for IterCtrlTrueStream failed.", model_exit_name.c_str());
      return FAILED;
    }
    GE_CHK_STATUS_RET(SetStreamLabel(model_exit_node, model_exit_name), "set stream label failed");

    add_ret = GraphUtils::AddEdge(active_node->GetOutControlAnchor(), model_exit_node->GetInControlAnchor());
    if (add_ret != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add active_node to model_exit_node ctrl edge failed, add_ret=%u.", add_ret);
      return FAILED;
    }
  }

  GELOGI("CreateIterCtrlFalseBranch success.");
  return SUCCESS;
}

Status FlowCtrlPass::AddFpBpIteratorCtrl(ComputeGraphPtr &compute_graph, NodePtr &pre_node) {
  GE_IF_BOOL_EXEC(pre_node == nullptr, DOMI_LOGE("pre_node is nullptr"); return FAILED);
  string pre_node_name = pre_node->GetName();
  GELOGI("Add FpBp Iterator ctrl, pre node:%s.", pre_node_name.c_str());
  // 1. Get or add variables
  NodePtr loop_cond_node = AddVariableNode(compute_graph, NODE_NAME_FLOWCTRL_LOOP_COND);
  if (loop_cond_node == nullptr) {
    GELOGE(FAILED, "Add variable:%s failed.", NODE_NAME_FLOWCTRL_LOOP_COND.c_str());
    return FAILED;
  }
  NodePtr loop_inc_node = AddVariableNode(compute_graph, NODE_NAME_FLOWCTRL_LOOP_INCREMENT);
  if (loop_inc_node == nullptr) {
    GELOGE(FAILED, "Add variable:%s failed.", NODE_NAME_FLOWCTRL_LOOP_INCREMENT.c_str());
    return FAILED;
  }
  NodePtr loop_reset_node = AddVariableNode(compute_graph, NODE_NAME_FLOWCTRL_LOOP_RESETVALUE);
  if (loop_reset_node == nullptr) {
    GELOGE(FAILED, "Add variable:%s failed.", NODE_NAME_FLOWCTRL_LOOP_RESETVALUE.c_str());
    return FAILED;
  }
  NodePtr iter_per_loop_node = AddVariableNode(compute_graph, NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
  if (iter_per_loop_node == nullptr) {
    GELOGE(FAILED, "Add variable:%s failed.", NODE_NAME_FLOWCTRL_LOOP_PER_ITER.c_str());
    return FAILED;
  }

  // 2. Add StreamSwitch
  string switch_name = pre_node_name + "_" + NODE_NAME_STREAM_SWITCH;
  NodePtr switch_node = InsertStreamSwitchOp(compute_graph, switch_name, loop_cond_node, iter_per_loop_node);
  if (switch_node == nullptr) {
    GELOGE(FAILED, "InsertStreamSwitchOp:%s failed.", switch_name.c_str());
    return FAILED;
  }
  GE_CHK_STATUS_RET(SetStreamLabel(switch_node, switch_name), "set stream label failed");

  graphStatus add_ret = GraphUtils::AddEdge(pre_node->GetOutControlAnchor(), switch_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add pre node:%s to switch_node:%s ctrl edge failed, ret = %u.", pre_node_name.c_str(),
           switch_name.c_str(), add_ret);
    return FAILED;
  }

  // 3. Create switch false branch: return results and reset the loopCond
  Status ret = CreateIterCtrlFalseBranch(compute_graph, loop_cond_node, loop_reset_node, switch_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "CreateIterCtrlFalseBranch fail, pre node:%s.", pre_node_name.c_str());
    return ret;
  }

  // 4. Create switch true branch:
  // active train streams and increase the loopCond
  ret = CreateIterCtrlTrueBranch(compute_graph, loop_cond_node, loop_inc_node, switch_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "CreateIterCtrlTrueBranch fail, pre node:%s.", pre_node_name.c_str());
    return ret;
  }
  return SUCCESS;
}

Status FlowCtrlPass::AddSpecialNodeIteratorCtrl(ComputeGraphPtr &compute_graph, NodePtr &loop_after_node) {
  /*
   * before add:
   *    iterator
   *       |
   *       v
   *   MemcpyAsync
   *
   * after add:
   *    iterator ----------┐
   *       |               ┆c
   *       v        c      v      c
   *   MemcpyAsync-----> switch -----> active
   *                       ^
   *                     /   \
   *          itersPerLoop  loopCond
   */
  GE_IF_BOOL_EXEC(loop_after_node == nullptr || compute_graph == nullptr,
                  DOMI_LOGE("loop after node or compute graph is null");
                  return FAILED);
  InDataAnchorPtr in_anchor = loop_after_node->GetInDataAnchor(0);
  if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
    GELOGE(FAILED, "Find %s in data anchor failed.", loop_after_node->GetName().c_str());
    return FAILED;
  }
  NodePtr loop_pre_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();

  // 1. Get variables
  NodePtr loop_cond_node = compute_graph->FindNode(NODE_NAME_FLOWCTRL_LOOP_COND);
  if (loop_cond_node == nullptr) {
    GELOGE(FAILED, "Find node :%s failed.", NODE_NAME_FLOWCTRL_LOOP_COND.c_str());
    return FAILED;
  }
  NodePtr iter_per_loop_node = compute_graph->FindNode(NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
  if (iter_per_loop_node == nullptr) {
    GELOGE(FAILED, "Find node :%s failed.", NODE_NAME_FLOWCTRL_LOOP_PER_ITER.c_str());
    return FAILED;
  }

  // 2. Add StreamSwitch and edges to switch_node.
  GE_IF_BOOL_EXEC(loop_pre_node == nullptr, DOMI_LOGE("loop pre node is null"); return FAILED);
  string switch_name = loop_pre_node->GetName() + "_" + NODE_NAME_STREAM_SWITCH;
  NodePtr switch_node = InsertStreamSwitchOp(compute_graph, switch_name, loop_cond_node, iter_per_loop_node);
  if (switch_node == nullptr) {
    GELOGE(FAILED, "InsertStreamSwitchOp:%s failed.", switch_name.c_str());
    return FAILED;
  }

  GE_CHK_STATUS_RET(SetStreamLabel(switch_node, switch_name), "set stream label failed");

  graphStatus add_ret = GraphUtils::AddEdge(loop_pre_node->GetOutControlAnchor(), switch_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add loop_pre_node:%s to switch_node:%s ctrl edge failed, ret = %u.",
           loop_pre_node->GetName().c_str(), switch_name.c_str(), add_ret);
    return FAILED;
  }
  add_ret = GraphUtils::AddEdge(loop_after_node->GetOutControlAnchor(), switch_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add node:%s to switch_node:%s ctrl edge failed, ret = %u.", loop_after_node->GetName().c_str(),
           switch_name.c_str(), add_ret);
    return FAILED;
  }

  // 3. Create switch true branch: only active
  string active_name = switch_name + "_StreamActive";
  NodePtr active_node = InsertOp(compute_graph, STREAMACTIVE, active_name, {}, {});
  if (active_node == nullptr) {
    GELOGE(FAILED, "Insert stream active node:%s for SpecialNodeIteratorCtrl failed.", active_name.c_str());
    return FAILED;
  }

  GE_CHK_STATUS_RET(SetStreamLabel(active_node, active_name), "set stream label failed");

  GE_IF_BOOL_EXEC(!AttrUtils::SetBool(active_node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE, true),
                  DOMI_LOGE("set ATTR_NAME_IS_LOOP_ACTIVE failed");
                  return FAILED);

  add_ret = GraphUtils::AddEdge(switch_node->GetOutControlAnchor(), active_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add switch_node:%s to active_node:%s ctrl edge failed, ret = %u.", switch_name.c_str(),
           active_name.c_str(), add_ret);
    return FAILED;
  }

  // used for stream assign to find true branch
  GE_CHK_STATUS_RET(SetActiveLabelList(switch_node, {active_name}), "set active label list failed");
  // used for stream assign to find active stream
  GE_CHK_STATUS_RET(SetActiveLabelList(active_node, {loop_pre_node->GetName()}), "set active label list failed");
  return SUCCESS;
}
}  // namespace ge
