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

#include "graph/passes/switch_op_pass.h"
#include <algorithm>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"

using domi::ATTR_NAME_STREAM_LABEL;
using domi::ATTR_NAME_WEIGHTS;
using domi::CAST_ATTR_DSTT;
using domi::CAST_ATTR_SRCT;

using domi::CAST;
using domi::CONSTANT;
using domi::ENTER;
using domi::EXIT;
using domi::MEMCPYASYNC;
using domi::MERGE;
using domi::NETOUTPUT;
using domi::NEXTITERATION;
using domi::REFENTER;
using domi::REFEXIT;
using domi::REFMERGE;
using domi::REFNEXTITERATION;
using domi::REFSWITCH;
using domi::STREAMACTIVE;
using domi::STREAMMERGE;
using domi::STREAMSWITCH;
using domi::SWITCH;

namespace ge {
Status SwitchOpPass::Run(ComputeGraphPtr graph) {
  GELOGD("SwitchOpPass Enter");

  GraphUtils::DumpGEGraph(graph, "BeforeSwitchOpPass");
  GraphUtils::DumpGEGraphToOnnx(*graph, "BeforeSwitchOpPass");

  GE_CHK_STATUS_RET(CheckCycleDependence(graph), "CheckCycleDependence fail.");

  for (auto &switch_node : switch_nodes_) {
    GE_CHK_STATUS_RET(ReplaceSwitchNode(graph, switch_node), "Add StreamSwitch node fail.");
  }

  for (auto &merge_node : merge_nodes_) {
    GE_CHK_STATUS_RET(ReplaceMergeNode(graph, merge_node), "Add StreamMerge node fail.");
  }

  GE_CHK_STATUS_RET(CombineSwitchNode(graph), "Combine StreamSwitch nodes fail.");

  for (auto &node : bypass_nodes_) {
    GE_CHK_BOOL_EXEC(graph->RemoveNode(node) == GRAPH_SUCCESS, return FAILED, "Remove switch node fail.");
  }

  for (auto &node : stream_switch_nodes_) {
    for (auto &out_ctrl_node : node->GetOutControlNodes()) {
      GELOGD("branch_head_nodes_ insert %s", out_ctrl_node->GetName().c_str());
      (void)branch_head_nodes_.insert(out_ctrl_node);
    }
  }

  for (auto &node : need_label_nodes_) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (!op_desc->HasAttr(ATTR_NAME_STREAM_LABEL)) {
      GE_CHK_STATUS_RET(UpdateCondBranch(node), "Set cond branch fail, start node:%s", node->GetName().c_str());
    }
  }

  GE_CHK_STATUS_RET(UpdateEnterNode(), "UpdateEnterNode fail.");

  GraphUtils::DumpGEGraph(graph, "AfterSwitchOpPass");
  GraphUtils::DumpGEGraphToOnnx(*graph, "AfterSwitchOpPass");

  GELOGD("SwitchOpPass Leave");
  return SUCCESS;
}

///
/// @brief Replace Switch Op
/// @param [in] graph
/// @param [in] switch_node
/// @return Status
///
Status SwitchOpPass::ReplaceSwitchNode(ComputeGraphPtr &graph, NodePtr &switch_node) {
  std::string type;
  GE_CHK_STATUS_RET(GetOriginalType(switch_node, type), "Get node type fail.");
  GE_CHK_BOOL_EXEC((type == SWITCH) || (type == REFSWITCH), return FAILED, "Type of input node is not switch.");

  OutDataAnchorPtr peer_data_anchor = nullptr;
  OutDataAnchorPtr peer_cond_anchor = nullptr;
  GE_CHK_BOOL_EXEC(BypassSwitchNode(switch_node, peer_data_anchor, peer_cond_anchor) == SUCCESS, return FAILED,
                   "Bypass switch node fail.");
  GE_CHECK_NOTNULL(peer_data_anchor);
  GE_CHECK_NOTNULL(peer_cond_anchor);
  OpDescPtr cond_desc = peer_cond_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHECK_NOTNULL(cond_desc);
  DataType cond_data_type = cond_desc->GetOutputDesc(peer_cond_anchor->GetIdx()).GetDataType();
  GE_CHK_BOOL_EXEC(cond_data_type == DT_BOOL, return FAILED,
                   "SwitchNode not support datatype %s, datatype of cond_input should be bool",
                   TypeUtils::DataTypeToSerialString(cond_data_type).c_str());

  OpDescPtr switch_desc = switch_node->GetOpDesc();
  GE_CHECK_NOTNULL(switch_desc);
  bool cyclic_flag = switch_desc->HasAttr(ATTR_NAME_CYCLIC_DEPENDENCE_FLAG);

  std::set<std::string> out_node_list;
  for (OutDataAnchorPtr &out_data_anchor : switch_node->GetAllOutDataAnchors()) {
    bool true_branch_flag = (static_cast<uint32_t>(out_data_anchor->GetIdx()) == SWITCH_TRUE_OUTPUT);
    NodePtr stream_switch = nullptr;
    out_node_list.clear();
    for (auto &peer_in_anchor : out_data_anchor->GetPeerAnchors()) {
      GE_IF_BOOL_EXEC(stream_switch == nullptr, {
        std::string suffix = (true_branch_flag ? "_t" : "_f");
        stream_switch = CreateStreamSwitchNode(graph, switch_node, suffix, peer_cond_anchor);
        GE_CHK_BOOL_EXEC(stream_switch != nullptr, return FAILED, "Create stream_switch node fail.");
        if (SetSwitchTrueBranchFlag(stream_switch, true_branch_flag) != SUCCESS) {
          GELOGE(FAILED, "SetSwitchTrueBranchFlag for node %s fail.", stream_switch->GetName().c_str());
          return FAILED;
        }
        if (MarkBranchs(peer_cond_anchor, stream_switch, true_branch_flag) != SUCCESS) {
          GELOGE(FAILED, "MarkBranchs for stream_switch %s fail.", stream_switch->GetName().c_str());
          return FAILED;
        }

        if (!cyclic_flag) {
          GE_CHK_STATUS(GraphUtils::AddEdge(peer_data_anchor->GetOwnerNode()->GetOutControlAnchor(),
                                            stream_switch->GetInControlAnchor()),
                        "StreamSwitch node add ctl edge fail.");
        }
      });

      GE_CHK_STATUS(GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor), "Remove Switch data output fail.");

      NodePtr out_node = peer_in_anchor->GetOwnerNode();
      GE_CHK_STATUS_RET(GetOriginalType(out_node, type), "Get node type fail.");
      if ((type == MERGE) || (type == REFMERGE)) {
        NodePtr memcpy_node = CreateMemcpyAsyncNode(graph, peer_data_anchor);
        GE_CHK_BOOL_EXEC(memcpy_node != nullptr, return FAILED, "Create memcpy_async node fail.");
        GE_CHK_STATUS(GraphUtils::AddEdge(peer_data_anchor, memcpy_node->GetInDataAnchor(0)),
                      "MemcpyAsync node add edge fail.");
        GE_CHK_STATUS(GraphUtils::AddEdge(memcpy_node->GetOutDataAnchor(0), peer_in_anchor),
                      "MemcpyAsync node add edge fail.");
        GE_CHK_STATUS(GraphUtils::AddEdge(stream_switch->GetOutControlAnchor(), memcpy_node->GetInControlAnchor()),
                      "MemcpyAsync node add ctl edge fail.");
        out_node_list.insert(memcpy_node->GetName());
      } else {
        GE_CHK_STATUS(GraphUtils::AddEdge(peer_data_anchor, peer_in_anchor), "StreamSwitch node add edge fail.");
        GE_CHK_STATUS(GraphUtils::AddEdge(stream_switch->GetOutControlAnchor(), out_node->GetInControlAnchor()),
                      "StreamSwitch node add ctl edge fail.");
        out_node_list.insert(out_node->GetName());
      }
    }
    GE_IF_BOOL_EXEC(stream_switch != nullptr, {
      CopyControlEdges(switch_node, stream_switch, true);
      switch_node_map_[stream_switch] = out_node_list;
      if (SetOriginalNodeName(stream_switch, switch_node->GetName()) != SUCCESS) {
        GELOGE(FAILED, "SetOriginalNodeName for node %s fail.", stream_switch->GetName().c_str());
        return FAILED;
      }
    });
  }

  RemoveControlEdges(switch_node);
  (void)bypass_nodes_.insert(switch_node);

  return SUCCESS;
}

///
/// @brief Replace Merge Op
/// @param [in] graph
/// @param [in] merge_node
/// @return Status
///
Status SwitchOpPass::ReplaceMergeNode(ComputeGraphPtr &graph, NodePtr &merge_node) {
  std::string type;
  GE_CHK_STATUS_RET(GetOriginalType(merge_node, type), "Get node type fail.");
  GE_CHK_BOOL_EXEC((type == MERGE) || (type == REFMERGE), return FAILED, "Type of input node is not merge.");

  OpDescPtr merge_op_desc = merge_node->GetOpDesc();
  GE_CHECK_NOTNULL(merge_op_desc);

  const std::string node_name = merge_node->GetName();
  GELOGI("Create StreamMerge Op, name=%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, STREAMMERGE);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc fail, StreamMerge:%s.", node_name.c_str());
    return FAILED;
  }

  for (InDataAnchorPtr &in_anchor : merge_node->GetAllInDataAnchors()) {
    GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(merge_op_desc->GetInputDesc(in_anchor->GetIdx())) == GRAPH_SUCCESS,
                     return FAILED, "Create StreamMerge op: add input desc fail.");
  }

  for (OutDataAnchorPtr &out_anchor : merge_node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(merge_op_desc->GetOutputDesc(out_anchor->GetIdx())) == GRAPH_SUCCESS,
                     return FAILED, "Create StreamMerge op: add output desc fail.");
  }

  NodePtr stream_merge = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(stream_merge != nullptr, return FAILED, "Insert StreamMerge node fail.");

  for (InDataAnchorPtr &in_data_anchor : merge_node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

    GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor), "Remove Merge data input fail.");
    GE_CHK_STATUS(GraphUtils::AddEdge(peer_out_anchor, stream_merge->GetInDataAnchor(in_data_anchor->GetIdx())),
                  "StreamMerge node add edge fail.");
  }

  for (OutDataAnchorPtr &out_data_anchor : merge_node->GetAllOutDataAnchors()) {
    for (InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHK_STATUS(GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor), "Remove Merge data output fail.");
      GE_CHK_STATUS(GraphUtils::AddEdge(stream_merge->GetOutDataAnchor(out_data_anchor->GetIdx()), peer_in_anchor),
                    "StreamMerge node add edge fail.");
    }
  }

  ReplaceControlEdges(merge_node, stream_merge);

  if (merge_op_desc->HasAttr(ATTR_NAME_NEXT_ITERATION)) {
    std::string next_iteration_name;
    GE_IF_BOOL_EXEC(!AttrUtils::GetStr(merge_op_desc, ATTR_NAME_NEXT_ITERATION, next_iteration_name),
                    GELOGE(INTERNAL_ERROR, "get ATTR_NAME_NEXT_ITERATION failed");
                    return INTERNAL_ERROR);

    GE_CHK_STATUS_RET(SetNextIteration(stream_merge, next_iteration_name), "set next iteration failed");
  } else {
    need_label_nodes_.emplace_back(stream_merge);
  }

  if (merge_op_desc->HasAttr(ATTR_INSERT_BY_MBATCH)) {
    if (!ge::AttrUtils::SetBool(op_desc, ATTR_INSERT_BY_MBATCH, true)) {
      GELOGE(FAILED, "Set attr ATTR_INSERT_BY_MBATCH fail, StreamMerge:%s.", node_name.c_str());
      return FAILED;
    }
  }

  (void)bypass_nodes_.insert(merge_node);

  GE_CHK_STATUS_RET(AddMemcpyAsyncNodes(graph, stream_merge), "StreamMerge add memcpy node fail.");

  return SUCCESS;
}

///
/// @brief Create StreamSwitch Node
/// @param [in] graph
/// @param [in] switch_node
/// @param [in] suffix
/// @param [in] peer_cond_anchor
/// @return ge::NodePtr
///
NodePtr SwitchOpPass::CreateStreamSwitchNode(ComputeGraphPtr &graph, const NodePtr &switch_node,
                                             const std::string &suffix, OutDataAnchorPtr &peer_cond_anchor) {
  GE_CHK_BOOL_EXEC(switch_node != nullptr, return nullptr, "Param of merge node is null.");
  OpDescPtr switch_op_desc = switch_node->GetOpDesc();
  GE_CHK_BOOL_EXEC(switch_op_desc != nullptr, return nullptr, "OpDesc of Switch node is invalid.");
  GE_IF_BOOL_EXEC(switch_op_desc->GetInputsSize() != SWITCH_INPUT_NUM, {
    GELOGE(FAILED, "Switch input param invalid, input_size=%lu, should be %u", switch_op_desc->GetInputsSize(),
           SWITCH_INPUT_NUM);
    return nullptr;
  });

  const std::string node_name = switch_node->GetName() + "_" + STREAMSWITCH + suffix;
  GELOGI("Create StreamSwitch, name=%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, STREAMSWITCH);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc fail, StreamSwitch:%s.", node_name.c_str());
    return nullptr;
  }

  if (!AttrUtils::SetInt(op_desc, ATTR_NAME_SWITCH_DATA_TYPE, RT_SWITCH_INT32) ||
      !AttrUtils::SetInt(op_desc, ATTR_NAME_STREAM_SWITCH_COND, (int64_t)RT_EQUAL)) {
    GELOGE(INTERNAL_ERROR, "set int failed");
    return nullptr;
  }

  // Already checked, first input is Variable will passed, second is condition will checked.
  GeTensorDesc cond_input_desc = switch_op_desc->GetInputDesc(SWITCH_PRED_INPUT);
  GeTensorDesc input_desc(GeShape(cond_input_desc.GetShape().GetDims()), cond_input_desc.GetFormat(), DT_INT32);
  GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(input_desc) == GRAPH_SUCCESS, return nullptr,
                   "Create StreamSwitch node: add input desc fail.");
  GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(input_desc) == GRAPH_SUCCESS, return nullptr,
                   "Create StreamSwitch node: add input desc fail.");

  NodePtr stream_switch = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(stream_switch != nullptr, return nullptr, "Insert StreamSwitch node fail.");

  GE_CHK_STATUS(GraphUtils::AddEdge(peer_cond_anchor, stream_switch->GetInDataAnchor(0)),
                "StreamSwitch node add cond edge fail.");

  return stream_switch;
}

///
/// @brief Add MemcpyAsync Node
/// @param [in] graph
/// @param [in] in_node
/// @return ge::NodePtr
///
NodePtr SwitchOpPass::CreateMemcpyAsyncNode(ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor) {
  GE_CHK_BOOL_EXEC(out_data_anchor != nullptr, return nullptr, "Param of input node is null.");
  OpDescPtr pre_op_desc = out_data_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHK_BOOL_EXEC(pre_op_desc != nullptr, return nullptr, "OpDesc of pre node is invalid.");

  std::string node_name = pre_op_desc->GetName() + "_" + MEMCPYASYNC;
  node_name = CheckDuplicateName(node_name);
  GELOGI("Create MemcpyAsync op:%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, MEMCPYASYNC);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc fail, MemcpyAsync:%s.", node_name.c_str());
    return nullptr;
  }

  GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx())) == GRAPH_SUCCESS,
                   return nullptr, "Create MemcpyAsync op: add input desc fail.");
  GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx())) == GRAPH_SUCCESS,
                   return nullptr, "Create MemcpyAsync op: add output desc fail.");

  NodePtr memcpy_node = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(memcpy_node != nullptr, return nullptr, "Insert MemcpyAsync node fail.");

  return memcpy_node;
}

///
/// @brief Combine switch nodes link to same cond
/// @param [in] graph
/// @return Status
///
Status SwitchOpPass::CombineSwitchNode(ComputeGraphPtr &graph) {
  for (auto iter = cond_node_map_.begin(); iter != cond_node_map_.end(); ++iter) {
    OutDataAnchorPtr peer_cond_anchor = iter->first;
    GE_CHECK_NOTNULL(peer_cond_anchor);
    std::list<NodePtr> false_switch_list = iter->second[SWITCH_FALSE_OUTPUT];
    std::list<NodePtr> true_switch_list = iter->second[SWITCH_TRUE_OUTPUT];
    std::set<NodePtr> same_cond_switch;
    same_cond_switch.insert(false_switch_list.begin(), false_switch_list.end());
    same_cond_switch.insert(true_switch_list.begin(), true_switch_list.end());

    NodePtr cond_node = peer_cond_anchor->GetOwnerNode();
    GELOGI("CombineSwitchNode: cond_node=%s", cond_node->GetName().c_str());

    NodePtr cast_node = CreateCastOp(graph, peer_cond_anchor);
    GE_CHK_BOOL_EXEC(cast_node != nullptr, return FAILED, "Create cast_node fail.");

    NodePtr active_node = CreateActiveNode(graph, cond_node);
    GE_CHK_BOOL_EXEC(active_node != nullptr, return FAILED, "Create StreamActive node fail.");
    GE_CHK_STATUS(GraphUtils::AddEdge(cast_node->GetOutControlAnchor(), active_node->GetInControlAnchor()),
                  "StreamActive add ctl edge fail.");
    if (SetActiveLabelList(active_node, {cast_node->GetName()}) != SUCCESS) {
      GELOGE(FAILED, "SetActiveLabelList for node %s fail.", active_node->GetName().c_str());
      return FAILED;
    }

    const std::string cond_group = cond_node->GetName();
    for (uint32_t i = 0; i < SWITCH_OUTPUT_NUM; ++i) {
      bool true_branch_flag = (i == SWITCH_TRUE_OUTPUT);
      std::list<NodePtr> &switch_list = (true_branch_flag ? true_switch_list : false_switch_list);
      GE_IF_BOOL_EXEC(switch_list.empty(), continue);

      // select first stream_switch
      NodePtr stream_switch = switch_list.front();
      OpDescPtr switch_desc = stream_switch->GetOpDesc();
      GE_CHECK_NOTNULL(switch_desc);
      switch_desc->SetName(cond_group + "/" + STREAMSWITCH + (true_branch_flag ? "_t" : "_f"));
      stream_switch_nodes_.emplace_back(stream_switch);
      need_label_nodes_.emplace_back(stream_switch);

      // 0_input: original pred input, 1_input: constant node
      GE_CHK_STATUS_RET(AddConstNode(graph, stream_switch), "Add const node fail");
      GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_cond_anchor, stream_switch->GetInDataAnchor(0)),
                    "StreamSwitch remove data edge fail.");
      GE_CHK_STATUS(GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), stream_switch->GetInDataAnchor(0)),
                    "Cast add data edge fail.");

      for (NodePtr &node : switch_list) {
        GE_CHECK_NOTNULL(node);
        GE_IF_BOOL_EXEC(node != stream_switch, {
          GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_cond_anchor, node->GetInDataAnchor(0)),
                        "StreamSwitch remove data edge fail.");
        });
        GE_CHK_STATUS(ModifySwitchInCtlEdges(node, cast_node, same_cond_switch), "ModifySwitchInCtlEdges fail");
        GE_CHK_STATUS(ModifySwitchOutCtlEdges(node, stream_switch, active_node), "ModifySwitchOutCtlEdges fail");
      }

      GE_CHK_STATUS(GraphUtils::AddEdge(active_node->GetOutControlAnchor(), stream_switch->GetInControlAnchor()),
                    "StreamActive add ctl edge fail.");
    }
  }
  return SUCCESS;
}

///
/// @brief Create Active Op
/// @param [in] graph
/// @param [in] cond_node
/// @return ge::NodePtr
///
NodePtr SwitchOpPass::CreateActiveNode(ComputeGraphPtr &graph, NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, return nullptr, "Param of pre cond_node is null.");
  std::string node_name = node->GetName() + "_" + STREAMACTIVE;
  node_name = CheckDuplicateName(node_name);
  GELOGI("Create StreamActive op:%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, STREAMACTIVE);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc fail, StreamActive:%s.", node_name.c_str());
    return nullptr;
  }

  NodePtr active_node = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(active_node != nullptr, return nullptr, "Create StreamActive node fail.");

  GE_IF_BOOL_EXEC(GraphUtils::AddEdge(node->GetOutControlAnchor(), active_node->GetInControlAnchor()) != SUCCESS,
                  GELOGE(INTERNAL_ERROR, "add edge failed");
                  return nullptr);

  GE_IF_BOOL_EXEC(SetSwitchBranchNodeLabel(active_node, node_name) != SUCCESS,
                  GELOGE(INTERNAL_ERROR, "set switch branch node label failed");
                  return nullptr);

  return active_node;
}

///
/// @brief Add MemcpyAsync Op as StreamMerge in_node
/// @param [in] graph
/// @param [in] node
/// @return Status
///
Status SwitchOpPass::AddMemcpyAsyncNodes(ComputeGraphPtr &graph, NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, return FAILED, "Param of pre node is null.");
  for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    NodePtr in_node = peer_out_anchor->GetOwnerNode();

    const std::string type = in_node->GetType();
    // For WhileLoop no need memcpy & active for merge.
    GE_IF_BOOL_EXEC((type == ENTER) || (type == REFENTER) || (type == NEXTITERATION) || (type == REFNEXTITERATION),
                    continue);

    GE_IF_BOOL_EXEC(type != MEMCPYASYNC, {
      in_node = CreateMemcpyAsyncNode(graph, peer_out_anchor);
      GE_CHK_BOOL_EXEC(in_node != nullptr, return FAILED, "Create MemcpyAsync node fail.");
      GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor), "MemcpyAsync node remove edge fail.");
      GE_CHK_STATUS(GraphUtils::AddEdge(peer_out_anchor, in_node->GetInDataAnchor(0)),
                    "MemcpyAsync node add edge fail.");
      GE_CHK_STATUS(GraphUtils::AddEdge(in_node->GetOutDataAnchor(0), in_data_anchor),
                    "MemcpyAsync node add edge fail.");
    });

    NodePtr active_node = CreateActiveNode(graph, in_node);
    GE_CHK_BOOL_EXEC(active_node != nullptr, return FAILED, "Create StreamActive node fail.");
    GE_CHK_STATUS(GraphUtils::AddEdge(active_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                  "StreamActive add ctl edge fail.");
    if (SetActiveLabelList(active_node, {node->GetName()}) != SUCCESS) {
      GELOGE(FAILED, "SetActiveLabelList for node %s fail.", active_node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

///
/// @brief Bypass Switch Node
/// @param [in] switch_node
/// @param [out] peer_data_anchor
/// @param [out] peer_cond_anchor
/// @return Status
///
Status SwitchOpPass::BypassSwitchNode(NodePtr &switch_node, OutDataAnchorPtr &peer_data_anchor,
                                      OutDataAnchorPtr &peer_cond_anchor) {
  GE_CHK_BOOL_EXEC(switch_node != nullptr, return FAILED, "Switch_node is null.");
  for (uint32_t idx = 0; idx < SWITCH_INPUT_NUM; ++idx) {
    InDataAnchorPtr in_data_anchor = switch_node->GetInDataAnchor(idx);
    GE_CHK_BOOL_EXEC(in_data_anchor != nullptr, return FAILED, "Check Switch input anchor fail.");
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHK_BOOL_EXEC(peer_out_anchor != nullptr, return FAILED, "Check Pre node output anchor fail.");
    // Remove Switch data input.
    GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor), "remove edge failed");

    if (idx == SWITCH_DATA_INPUT) {
      peer_data_anchor = peer_out_anchor;
    } else {
      if (FindSwitchCondInput(false, peer_out_anchor) != SUCCESS) {
        GELOGE(FAILED, "FindSwitchCondInput fail, switch=%s", switch_node->GetName().c_str());
        return FAILED;
      }
      peer_cond_anchor = peer_out_anchor;
    }
  }

  return SUCCESS;
}

///
/// @brief Find Switch cond input
/// @param [in] pass_switch_flag
/// @param [out] peer_cond_anchor
/// @return Status
///
Status SwitchOpPass::FindSwitchCondInput(bool pass_switch_flag, OutDataAnchorPtr &peer_cond_anchor) {
  NodePtr tmp_node = nullptr;
  string type;
  bool need_pass_type = true;
  while (need_pass_type) {
    if (tmp_node == nullptr) {
      GE_CHECK_NOTNULL(peer_cond_anchor);
      tmp_node = peer_cond_anchor->GetOwnerNode();
    } else {
      InDataAnchorPtr in_data_anchor = tmp_node->GetInDataAnchor(SWITCH_DATA_INPUT);
      GE_CHECK_NOTNULL(in_data_anchor);
      peer_cond_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_cond_anchor);
      tmp_node = peer_cond_anchor->GetOwnerNode();
    }

    GE_CHK_STATUS_RET(GetOriginalType(tmp_node, type), "Get node type fail");
    need_pass_type = (pass_switch_flag && ((type == SWITCH) || (type == REFSWITCH)));
  }

  return SUCCESS;
}

///
/// @brief Mark Switch Branch
/// @param [in] peer_cond_anchor
/// @param [in] stream_switch
/// @param [in] true_branch_flag
/// @return Status
///
Status SwitchOpPass::MarkBranchs(OutDataAnchorPtr &peer_cond_anchor, NodePtr &stream_switch, bool true_branch_flag) {
  uint32_t index = true_branch_flag ? SWITCH_TRUE_OUTPUT : SWITCH_FALSE_OUTPUT;
  GE_CHECK_NOTNULL(stream_switch);
  auto it = cond_node_map_.find(peer_cond_anchor);
  if (it != cond_node_map_.end()) {
    GE_IF_BOOL_EXEC(it->second.size() != SWITCH_OUTPUT_NUM, {
      GELOGE(INTERNAL_ERROR, "cond_node_map_ check size fail, node: %s", stream_switch->GetName().c_str());
      return FAILED;
    });
    it->second[index].emplace_back(stream_switch);
  } else {
    std::list<NodePtr> false_node_list;
    std::list<NodePtr> true_node_list;
    std::list<NodePtr> &node_list = true_branch_flag ? true_node_list : false_node_list;
    node_list.emplace_back(stream_switch);
    std::vector<std::list<NodePtr>> switch_list;
    switch_list.emplace_back(false_node_list);
    switch_list.emplace_back(true_node_list);
    auto result = cond_node_map_.insert(
      std::pair<OutDataAnchorPtr, std::vector<std::list<NodePtr>>>(peer_cond_anchor, switch_list));
    GE_IF_BOOL_EXEC(!result.second, {
      GELOGE(INTERNAL_ERROR, "cond_node_map_ insert fail, node: %s", stream_switch->GetName().c_str());
      return FAILED;
    });
  }
  return SUCCESS;
}

///
/// @brief Create cast node
/// @param [in] graph
/// @param [in] peer_cond_anchor
/// @return NodePtr
///
NodePtr SwitchOpPass::CreateCastOp(ComputeGraphPtr &graph, OutDataAnchorPtr &peer_cond_anchor) {
  GE_CHK_BOOL_EXEC(peer_cond_anchor != nullptr, return nullptr, "Param of pre cond_node is null.");
  OpDescPtr cond_desc = peer_cond_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHK_BOOL_EXEC(cond_desc != nullptr, return nullptr, "Get cond_desc fail.");

  const std::string cast_name = cond_desc->GetName() + "_" + CAST;
  GELOGI("Create cast_node: %s, input datatype:DT_BOOL, out datatype:DT_INT32", cast_name.c_str());
  OpDescPtr cast_desc = MakeShared<OpDesc>(cast_name, CAST);
  if (cast_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc fail, Cast:%s.", cast_name.c_str());
    return nullptr;
  }
  if (!(AttrUtils::SetInt(cast_desc, CAST_ATTR_SRCT, (int64_t)DT_BOOL) &&
        AttrUtils::SetInt(cast_desc, CAST_ATTR_DSTT, (int64_t)DT_INT32) &&
        AttrUtils::SetInt(cast_desc, CAST_ATTR_DST_TYPE, (int64_t)DT_INT32) &&
        AttrUtils::SetBool(cast_desc, CAST_ATTR_TRUNCATE, false))) {
    GELOGE(FAILED, "Set CAST_ATTR_SRCT or CAST_ATTR_DSTT or CAST_ATTR_DST_TYPE or CAST_ATTR_TRUNCATE fail, node: %s.",
           cast_name.c_str());
    return nullptr;
  }
  GeTensorDesc tensor_desc = cond_desc->GetOutputDesc(peer_cond_anchor->GetIdx());
  tensor_desc.SetDataType(DT_BOOL);
  GE_CHK_BOOL_EXEC(cast_desc->AddInputDesc(tensor_desc) == SUCCESS, return nullptr, "Cast_node add input desc fail.");
  tensor_desc.SetDataType(DT_INT32);
  GE_CHK_BOOL_EXEC(cast_desc->AddOutputDesc(tensor_desc) == SUCCESS, return nullptr, "Cast_node add output desc fail.");

  NodePtr cast_node = graph->AddNode(cast_desc);
  GE_CHK_BOOL_EXEC(cast_node != nullptr, return nullptr, "Create cast_node fail.");

  GE_CHK_STATUS(GraphUtils::AddEdge(peer_cond_anchor, cast_node->GetInDataAnchor(0)), "Cast add data edge fail.");

  return cast_node;
}

///
/// @brief Add const node as switch input1
/// @param [in] graph
/// @param [in] stream_switch
/// @return Status
///
Status SwitchOpPass::AddConstNode(ComputeGraphPtr &graph, NodePtr &stream_switch) {
  GE_CHK_BOOL_EXEC(stream_switch != nullptr, return FAILED, "stream_switch is null.");
  OpDescPtr op_desc = stream_switch->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  bool value = false;
  GE_CHK_BOOL_EXEC(AttrUtils::GetBool(op_desc, ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, value), return FAILED,
                   "StreamSwitch get attr TRUE_BRANCH_STREAM fail.");

  const std::string const_node_name = op_desc->GetName() + "_Constant_" + (value ? "t" : "f");
  GELOGI("Create const op: %s", const_node_name.c_str());
  OpDescPtr const_op_desc = MakeShared<OpDesc>(const_node_name, CONSTANT);
  if (const_op_desc == nullptr) {
    GELOGE(FAILED, "Create op_desc fail, Constant:%s.", const_node_name.c_str());
    return FAILED;
  }

  auto resize_value = (int32_t)value;
  GeTensorDesc data_desc = op_desc->GetInputDesc(1);
  GeTensorPtr const_value =
    MakeShared<GeTensor>(data_desc, reinterpret_cast<uint8_t *>(&resize_value), sizeof(int32_t));
  if (const_value == nullptr) {
    GELOGE(FAILED, "Create tensor fail.");
    return FAILED;
  }
  GE_CHK_BOOL_EXEC(AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, const_value), return FAILED);
  GE_CHK_BOOL_EXEC(const_op_desc->AddOutputDesc(data_desc) == GRAPH_SUCCESS, return FAILED,
                   "Create Const op: add output desc fail.");

  NodePtr const_node = graph->AddNode(const_op_desc);
  GE_CHK_BOOL_EXEC(const_node != nullptr, return FAILED, "Insert Const node fail.");
  GE_CHK_STATUS(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), stream_switch->GetInDataAnchor(1)),
                "StreamSwitch node add ctl edge fail.");

  return SUCCESS;
}

///
/// @brief update cond branch
/// @param [in] node
/// @return Status
///
Status SwitchOpPass::UpdateCondBranch(NodePtr &node) {
  std::string stream_label;
  std::unordered_set<NodePtr> branch_nodes;
  std::unordered_set<NodePtr> handled_set;
  std::stack<NodePtr> nodes;
  nodes.push(node);

  static const std::set<std::string> end_type_set = {STREAMSWITCH, STREAMMERGE};
  bool merge_flag = false;
  bool exit_flag = false;
  bool net_output_flag = false;

  while (!nodes.empty()) {
    NodePtr cur_node = nodes.top();
    nodes.pop();
    if (handled_set.count(cur_node) > 0) {
      continue;
    }
    GE_CHECK_NOTNULL(cur_node);
    if (UpdateAttachFlag(cur_node, stream_label, merge_flag, exit_flag, net_output_flag) != SUCCESS) {
      GELOGE(FAILED, "UpdateAttachFlag fail, cur_node: %s.", cur_node->GetName().c_str());
      return FAILED;
    }

    const std::string type = cur_node->GetType();
    for (auto &out_node : cur_node->GetOutAllNodes()) {
      const std::string out_type = out_node->GetType();
      bool stop_flag = (end_type_set.count(out_type) > 0) ||
                       ((type != STREAMSWITCH) && (branch_head_nodes_.count(out_node) > 0)) ||
                       (((type == ENTER) || (type == REFENTER)) && (out_type != STREAMACTIVE));
      if (!stop_flag) {
        nodes.push(out_node);
        GELOGD("branch_nodes insert %s", out_node->GetName().c_str());
        branch_nodes.insert(out_node);
      }
    }
    handled_set.insert(cur_node);
  }

  if (node->GetType() == STREAMSWITCH) {
    GE_CHK_STATUS_RET(SetActiveLabelList(node, {stream_label}), "set active_label_list failed");
  }

  bool attach_flag = (merge_flag || exit_flag) && net_output_flag;
  if (attach_flag) {
    GELOGI("No need to keep on attaching label.");
    return SUCCESS;
  }

  for (NodePtr tmp_node : branch_nodes) {
    GELOGD("Attach label %s to node: %s", stream_label.c_str(), tmp_node->GetName().c_str());
    GE_CHK_STATUS_RET(SetStreamLabel(tmp_node, stream_label), "set stream label failed");
  }

  return SUCCESS;
}

///
/// @brief update attach flag
/// @param [in] node
/// @param [out] stream_label
/// @param [out] merge_flag
/// @param [out] exit_flag
/// @param [out] net_output_flag
/// @return Status
///
Status SwitchOpPass::UpdateAttachFlag(const NodePtr &node, std::string &stream_label, bool &merge_flag, bool &exit_flag,
                                      bool &net_output_flag) {
  const std::string type = node->GetType();
  if (type == STREAMSWITCH) {
    if (node->GetInDataNodes().empty()) {
      GELOGE(INTERNAL_ERROR, "cur_node %s has no input_data_node", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    stream_label = node->GetInDataNodes().at(0)->GetName();
    GE_CHK_STATUS_RET(SetStreamLabel(node, stream_label), "set stream label failed");
    bool value = false;
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHK_BOOL_EXEC(AttrUtils::GetBool(op_desc, ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, value), return FAILED,
                     "StreamSwitch get attr TRUE_BRANCH_STREAM fail.");
    stream_label += (value ? "_t" : "_f");
  } else if (type == STREAMMERGE) {
    stream_label = node->GetName();
    GE_CHK_STATUS_RET(SetStreamLabel(node, stream_label), "set stream label failed");
    merge_flag = true;
  } else if ((type == EXIT) || (type == REFEXIT)) {
    GE_CHK_STATUS_RET(SetStreamLabel(node, stream_label), "set stream label failed");
    exit_flag = true;
  } else if (type == NETOUTPUT) {
    net_output_flag = true;
  }

  return SUCCESS;
}

///
/// @brief update loop branch
/// @param [in] enter_nodes
/// @param [in] stream_label
/// @return Status
///
Status SwitchOpPass::UpdateLoopBranch(const std::stack<NodePtr> &enter_nodes, const std::string &stream_label) {
  std::stack<NodePtr> nodes(enter_nodes);
  NodePtr cur_node = nullptr;
  while (!nodes.empty()) {
    cur_node = nodes.top();
    nodes.pop();
    for (NodePtr &out_node : cur_node->GetOutAllNodes()) {
      OpDescPtr out_desc = out_node->GetOpDesc();
      GE_CHECK_NOTNULL(out_desc);
      if (out_desc->HasAttr(ATTR_NAME_STREAM_LABEL)) {
        continue;
      }
      GELOGD("Attach label %s to node: %s", stream_label.c_str(), out_node->GetName().c_str());
      GE_CHK_STATUS_RET(SetStreamLabel(out_node, stream_label), "set stream label failed");
      nodes.push(out_node);
    }
  }

  return SUCCESS;
}

///
/// @brief update enter nodes
/// @return Status
///
Status SwitchOpPass::UpdateEnterNode() {
  std::unordered_map<NodePtr, std::vector<NodePtr>> enter_active_map;
  for (auto &enter_node : enter_nodes_) {
    for (auto &out_ctrl_node : enter_node->GetOutControlNodes()) {
      if (out_ctrl_node->GetType() != STREAMACTIVE) {
        continue;
      }
      auto iter = enter_active_map.find(out_ctrl_node);
      if (iter == enter_active_map.end()) {
        enter_active_map[out_ctrl_node] = {enter_node};
      } else {
        iter->second.emplace_back(enter_node);
      }
    }
  }

  for (auto &pair : enter_active_map) {
    std::string stream_label;
    NodePtr active_node = pair.first;
    GE_CHECK_NOTNULL(active_node);
    OpDescPtr active_desc = active_node->GetOpDesc();
    GE_CHECK_NOTNULL(active_desc);
    (void)AttrUtils::GetStr(active_desc, ATTR_NAME_STREAM_LABEL, stream_label);
    if (stream_label.empty()) {
      stream_label = active_desc->GetName();
      GE_CHK_STATUS_RET(SetStreamLabel(active_node, stream_label), "set stream label failed");
    }
    std::stack<NodePtr> enter_nodes;
    for (auto &enter_node : pair.second) {
      GE_CHK_STATUS_RET(SetStreamLabel(enter_node, stream_label), "set stream label failed");
      enter_nodes.emplace(enter_node);
    }

    std::vector<std::string> active_label_list;
    if (!AttrUtils::GetListStr(active_desc, ATTR_NAME_ACTIVE_LABEL_LIST, active_label_list) ||
        (active_label_list.size() != 1) || active_label_list[0].empty()) {
      GELOGE(INTERNAL_ERROR, "Get attr ATTR_NAME_ACTIVE_LABEL_LIST fail, node: %s", active_desc->GetName().c_str());
      return INTERNAL_ERROR;
    }
    if (UpdateLoopBranch(enter_nodes, active_label_list[0]) != SUCCESS) {
      GELOGE(FAILED, "UpdateLoopBranch fail.");
      return FAILED;
    }
  }

  return SUCCESS;
}

///
/// @brief Check duplicate node_name
/// @param [in] node_name
/// @return std::string
///
std::string SwitchOpPass::CheckDuplicateName(const std::string &node_name) {
  std::string tmp_name = node_name;
  auto iter = node_num_map_.find(tmp_name);
  if (iter != node_num_map_.end()) {
    tmp_name = tmp_name + "_" + std::to_string(iter->second);
    (iter->second)++;
  } else {
    node_num_map_[tmp_name] = 1;
  }
  return tmp_name;
}

///
/// @brief Check cyclic dependence
/// @param [in] graph
/// @return Status
///
Status SwitchOpPass::CheckCycleDependence(ComputeGraphPtr &graph) {
  std::string type;
  std::unordered_map<NodePtr, std::vector<NodePtr>> cond_switch_map;
  for (NodePtr &node : graph->GetDirectNode()) {
    GE_CHK_STATUS_RET(GetOriginalType(node, type), "Get node type fail");
    if ((type == SWITCH) || (type == REFSWITCH)) {
      InDataAnchorPtr in_cond_anchor = node->GetInDataAnchor(SWITCH_PRED_INPUT);
      GE_CHK_BOOL_EXEC(in_cond_anchor != nullptr, return INTERNAL_ERROR, "Check Switch in_cond_anchor fail.");
      OutDataAnchorPtr peer_out_anchor = in_cond_anchor->GetPeerOutAnchor();
      GE_CHK_BOOL_EXEC(peer_out_anchor != nullptr, return INTERNAL_ERROR, "Check Switch peer_out_anchor fail.");
      if (FindSwitchCondInput(true, peer_out_anchor) != SUCCESS) {
        GELOGE(FAILED, "FindSwitchCondInput fail, switch=%s", node->GetName().c_str());
        return FAILED;
      }

      NodePtr cond_node = peer_out_anchor->GetOwnerNode();
      auto iter = cond_switch_map.find(cond_node);
      if (iter == cond_switch_map.end()) {
        cond_switch_map[cond_node] = {node};
      } else {
        iter->second.emplace_back(node);
      }

      switch_nodes_.emplace_back(node);
    } else if ((type == MERGE) || (type == REFMERGE)) {
      merge_nodes_.emplace_back(node);
    } else if ((type == ENTER) || (type == REFENTER)) {
      enter_nodes_.emplace_back(node);
    }
  }

  MarkCycleDependence(cond_switch_map);

  return SUCCESS;
}

///
/// @brief Mark cyclic dependence
/// @param [in] graph
/// @param [in] cond_switch_map
/// @return void
///
void SwitchOpPass::MarkCycleDependence(const std::unordered_map<NodePtr, std::vector<NodePtr>> &cond_switch_map) {
  std::stack<NodePtr> out_nodes;
  NodePtr tmp_node = nullptr;
  std::unordered_set<NodePtr> handled_set;
  for (auto &iter : cond_switch_map) {
    std::set<NodePtr> switch_nodes(iter.second.begin(), iter.second.end());
    for (auto &switch_node : switch_nodes) {
      GE_CHECK_NOTNULL_JUST_RETURN(switch_node);
      GELOGD("CheckCycleDependence: cond_node=%s, switch=%s", iter.first->GetName().c_str(),
             switch_node->GetName().c_str());
      for (const NodePtr &node : switch_node->GetOutAllNodes()) {
        out_nodes.push(node);
      }
    }
    handled_set.clear();
    while (!out_nodes.empty()) {
      tmp_node = out_nodes.top();
      GE_CHECK_NOTNULL_JUST_RETURN(tmp_node);
      out_nodes.pop();
      if (handled_set.count(tmp_node) > 0) {
        continue;
      }
      GELOGD("CheckCycleDependence: tmp_node=%s", tmp_node->GetName().c_str());
      for (NodePtr &out_node : tmp_node->GetOutAllNodes()) {
        if (switch_nodes.find(out_node) == switch_nodes.end()) {
          out_nodes.push(out_node);
          continue;
        }
        GE_IF_BOOL_EXEC(SetCyclicDependenceFlag(out_node) != SUCCESS, GELOGW("set cyclic dependence failed"); return );
        auto map_iter = switch_cyclic_map_.find(out_node);
        if (map_iter == switch_cyclic_map_.end()) {
          switch_cyclic_map_[out_node] = {tmp_node->GetName()};
        } else {
          map_iter->second.insert(tmp_node->GetName());
        }
      }
      handled_set.insert(tmp_node);
    }
  }

  return;
}

///
/// @brief Modify in ctl edge for switch_node
/// @param [in] switch_node
/// @param [in] cast_node
/// @param [in] same_cond_switch
/// @return Status
///
Status SwitchOpPass::ModifySwitchInCtlEdges(NodePtr &switch_node, NodePtr &cast_node,
                                            const std::set<NodePtr> &same_cond_switch) {
  GE_CHECK_NOTNULL(switch_node);
  GE_CHECK_NOTNULL(cast_node);
  GELOGI("ModifySwitchInCtlEdges: switch_node=%s, active_node=%s", switch_node->GetName().c_str(),
         cast_node->GetName().c_str());

  std::string orig_switch_name = switch_node->GetName();
  OpDescPtr switch_desc = switch_node->GetOpDesc();
  GE_CHECK_NOTNULL(switch_desc);
  if (!AttrUtils::GetStr(switch_desc, ATTR_NAME_ORIG_NODE_NAME, orig_switch_name) || orig_switch_name.empty()) {
    GELOGE(INTERNAL_ERROR, "Get attr ATTR_NAME_ORIG_NODE_NAME fail, node: %s", switch_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  for (NodePtr &in_ctl_node : switch_node->GetInControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(in_ctl_node->GetOutControlAnchor(), switch_node->GetInControlAnchor()),
                  "Remove ctl edge fail.");
    GE_IF_BOOL_EXEC(!in_ctl_node->GetOutControlAnchor()->IsLinkedWith(cast_node->GetInControlAnchor()), {
      GE_CHK_STATUS(GraphUtils::AddEdge(in_ctl_node->GetOutControlAnchor(), cast_node->GetInControlAnchor()),
                    "Add ctl edge fail.");
    });

    GE_IF_BOOL_EXEC(in_ctl_node->GetType() != STREAMSWITCH, continue);
    if (same_cond_switch.count(in_ctl_node) > 0) {
      GE_CHK_STATUS(GraphUtils::RemoveEdge(in_ctl_node->GetOutControlAnchor(), cast_node->GetInControlAnchor()),
                    "Remove ctl edge fail.");
      continue;
    }
    auto find_res1 = switch_node_map_.find(in_ctl_node);
    GE_IF_BOOL_EXEC(find_res1 == switch_node_map_.end(), {
      GELOGE(INTERNAL_ERROR, "StreamSwitch node %s not found in switch_node_map_.", in_ctl_node->GetName().c_str());
      return INTERNAL_ERROR;
    });
    auto find_res2 = find_res1->second.find(orig_switch_name);
    auto find_res3 = find_res1->second.find(cast_node->GetName());
    GE_IF_BOOL_EXEC((find_res2 != find_res1->second.end()) && (find_res3 == find_res1->second.end()), {
      find_res1->second.erase(find_res2);
      find_res1->second.insert(cast_node->GetName());
      continue;
    });
  }

  return SUCCESS;
}

///
/// @brief Modify out ctl edge for switch_node
/// @param [in] switch_node
/// @param [in] stream_switch
/// @param [in] active_node
/// @return Status
///
Status SwitchOpPass::ModifySwitchOutCtlEdges(NodePtr &switch_node, NodePtr &stream_switch, NodePtr &active_node) {
  GE_CHECK_NOTNULL(switch_node);
  GE_CHECK_NOTNULL(stream_switch);
  GE_CHECK_NOTNULL(active_node);
  GELOGI("ModifySwitchOutCtlEdges: switch_node=%s, stream_switch=%s, active_node=%s", switch_node->GetName().c_str(),
         stream_switch->GetName().c_str(), active_node->GetName().c_str());
  auto find_res = switch_node_map_.find(switch_node);
  GE_IF_BOOL_EXEC(find_res == switch_node_map_.end(), {
    GELOGE(INTERNAL_ERROR, "StreamSwitch node %s not found in switch_node_map_.", switch_node->GetName().c_str());
    return INTERNAL_ERROR;
  });
  GE_IF_BOOL_EXEC(find_res->second.empty(), {
    GELOGE(INTERNAL_ERROR, "true_nodes of StreamSwitch node %s is empty.", switch_node->GetName().c_str());
    return INTERNAL_ERROR;
  });

  for (NodePtr &node : switch_node->GetOutControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(switch_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                  "Remove ctl edge fail.");
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string orig_name = op_desc->GetName();
    GE_IF_BOOL_EXEC(op_desc->HasAttr(ATTR_NAME_ORIG_NODE_NAME), {
      if (!AttrUtils::GetStr(op_desc, ATTR_NAME_ORIG_NODE_NAME, orig_name) || orig_name.empty()) {
        GELOGE(INTERNAL_ERROR, "Get attr ATTR_NAME_ORIG_NODE_NAME fail, node: %s.", op_desc->GetName().c_str());
        return INTERNAL_ERROR;
      }
    });
    if (find_res->second.find(orig_name) == find_res->second.end()) {
      auto active_out_control_anchor = active_node->GetOutControlAnchor();
      GE_CHECK_NOTNULL(active_out_control_anchor);
      GE_IF_BOOL_EXEC(!active_out_control_anchor->IsLinkedWith(node->GetInControlAnchor()), {
        GE_CHK_STATUS(GraphUtils::AddEdge(active_out_control_anchor, node->GetInControlAnchor()), "Add ctl edge fail.");
      });
    } else {
      auto stream_switch_out_control_anchor = stream_switch->GetOutControlAnchor();
      GE_CHECK_NOTNULL(stream_switch_out_control_anchor);
      GE_IF_BOOL_EXEC(!stream_switch_out_control_anchor->IsLinkedWith(node->GetInControlAnchor()), {
        GE_CHK_STATUS(GraphUtils::AddEdge(stream_switch_out_control_anchor, node->GetInControlAnchor()),
                      "Add ctl edge fail.");
      });
    }
  }

  GE_IF_BOOL_EXEC(switch_node != stream_switch, (void)bypass_nodes_.insert(switch_node));

  return SUCCESS;
}

///
/// @brief Copy Control Edges
/// @param [in] old_node
/// @param [in] new_node
/// @param [in] input_check_flag
/// @return void
///
void SwitchOpPass::CopyControlEdges(NodePtr &old_node, NodePtr &new_node, bool input_check_flag) {
  GE_CHECK_NOTNULL_JUST_RETURN(old_node);
  GE_CHECK_NOTNULL_JUST_RETURN(new_node);
  GE_IF_BOOL_EXEC(old_node == new_node, return );
  auto iter = switch_cyclic_map_.find(old_node);
  bool check_flag = input_check_flag && (iter != switch_cyclic_map_.end());
  for (NodePtr &node : old_node->GetInControlNodes()) {
    if (check_flag && (iter->second.count(node->GetName()) > 0)) {
      for (auto &out_node : old_node->GetOutAllNodes()) {
        auto out_control_anchor = node->GetOutControlAnchor();
        GE_CHECK_NOTNULL_JUST_RETURN(out_control_anchor);
        GE_IF_BOOL_EXEC(!out_control_anchor->IsLinkedWith(out_node->GetInControlAnchor()), {
          GE_CHK_STATUS(GraphUtils::AddEdge(out_control_anchor, out_node->GetInControlAnchor()), "Add ctl edge fail.");
        });
      }
    } else {
      auto out_control_anchor = node->GetOutControlAnchor();
      GE_CHECK_NOTNULL_JUST_RETURN(out_control_anchor);
      GE_IF_BOOL_EXEC(!out_control_anchor->IsLinkedWith(new_node->GetInControlAnchor()), {
        GE_CHK_STATUS(GraphUtils::AddEdge(out_control_anchor, new_node->GetInControlAnchor()), "Add in ctl edge fail.");
      });
    }
  }

  for (NodePtr &node : old_node->GetOutControlNodes()) {
    GE_IF_BOOL_EXEC(!new_node->GetOutControlAnchor()->IsLinkedWith(node->GetInControlAnchor()), {
      GE_CHK_STATUS(GraphUtils::AddEdge(new_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                    "Add out ctl edge fail.");
    });
  }
}

///
/// @brief Remove Control Edges
/// @param [in] node
/// @return void
///
void SwitchOpPass::RemoveControlEdges(NodePtr &node) {
  GE_CHECK_NOTNULL_JUST_RETURN(node);
  for (NodePtr &in_node : node->GetInControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(in_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                  "Remove in ctl edge fail.");
  }

  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    for (auto &in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
      GE_CHK_STATUS(GraphUtils::RemoveEdge(out_data_anchor, in_ctrl_anchor), "Remove in ctl edge fail.");
    }
  }

  auto out_control_anchor = node->GetOutControlAnchor();
  GE_CHECK_NOTNULL_JUST_RETURN(out_control_anchor);
  for (auto &peer_anchor : out_control_anchor->GetPeerAnchors()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(out_control_anchor, peer_anchor), "Remove out ctl edge fail.");
  }
}

///
/// @brief Replace Control Edges
/// @param [in] old_node
/// @param [in] new_node
/// @return void
///
void SwitchOpPass::ReplaceControlEdges(NodePtr &old_node, NodePtr &new_node) {
  GE_IF_BOOL_EXEC(old_node == new_node, return );
  CopyControlEdges(old_node, new_node);
  RemoveControlEdges(old_node);
}
}  // namespace ge
