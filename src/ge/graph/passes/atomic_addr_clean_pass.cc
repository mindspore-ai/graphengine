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

#include "graph/passes/atomic_addr_clean_pass.h"

#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "init/gelib.h"

namespace {
bool is_loop_graph = false;
}
namespace ge {
Status AtomicAddrCleanPass::Run(ComputeGraphPtr graph) {
  GE_TIMESTAMP_START(AtomicAddrCleanPass);
  if (graph == nullptr) {
    GELOGE(PARAM_INVALID, "param [graph] must not be null.");
    return PARAM_INVALID;
  }
  GELOGD("AtomicAddrCleanPass begin.");
  // 1.Recoginze atomic and loop mark
  vector<NodePtr> atomic_node_vec;
  for (NodePtr &node : graph->GetDirectNode()) {
    if (IsAtomicOp(node)) {
      atomic_node_vec.push_back(node);
    }
    if (!is_loop_graph && node->GetType() == LOOPCOND) {
      // there is loop in this graph
      GELOGD("There is no loop node. It will insert clean node follow atomic node.");
      is_loop_graph = true;
    }
  }
  if (atomic_node_vec.empty()) {
    GELOGI("There is no atomic node. Ignore atomicAddrClean pass.");
    return SUCCESS;
  }
  // 2.Insert clean node and link to atomic node
  Status ret;
  if (is_loop_graph) {
    ret = HandleLoopGraph(graph, atomic_node_vec);
    if (ret != SUCCESS) {
      return ret;
    }
  } else {
    ret = HandleNormalGraph(graph, atomic_node_vec);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  GELOGD("AtomicAddrCleanPass end.");
  GE_TIMESTAMP_END(AtomicAddrCleanPass, "GraphManager::AtomicAddrCleanPass");
  return SUCCESS;
}

Status AtomicAddrCleanPass::HandleLoopGraph(ComputeGraphPtr &graph, const vector<NodePtr> &atomic_node_vec) {
  // Loop graph , insert clean node follow atomic node
  int index = 0;
  for (const auto &node : atomic_node_vec) {
    // Insert atomic clean op
    NodePtr clean_addr_node = InsertAtomicAddrCleanNode(graph);
    if (clean_addr_node == nullptr) {
      GELOGE(FAILED, "Insert AtomicAddrClean node failed. Ignore atomicAddrClean pass.");
      return FAILED;
    }

    GE_CHECK_NOTNULL(clean_addr_node->GetOpDesc());
    string node_name = clean_addr_node->GetOpDesc()->GetName();
    std::ostringstream oss;
    oss << node_name << index;
    node_name = oss.str();
    clean_addr_node->GetOpDesc()->SetName(node_name);  // [Cascade Pointer]
    GELOGD("Inserted atomic clean node name is %s", node_name.c_str());

    auto ret = LinkToAtomicNode(node, clean_addr_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "Link control anchor failed from atomic node to atomic_addr_clean node.");
      return ret;
    }
    index++;
  }
  return SUCCESS;
}

Status AtomicAddrCleanPass::HandleNormalGraph(ComputeGraphPtr &graph, const vector<NodePtr> &atomic_node_vec) {
  GELOGD("Not loop graph. It will insert only 1 clean node.");
  // not loop graph , insert only one clean node in graph
  NodePtr clean_addr_node = InsertAtomicAddrCleanNode(graph);
  if (clean_addr_node == nullptr) {
    GELOGE(FAILED, "Insert AtomicAddrClean node failed. Ignore atomicAddrClean pass.");
    return FAILED;
  }
  for (const auto &node : atomic_node_vec) {
    auto ret = LinkToAtomicNode(node, clean_addr_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "Link control anchor failed from atomic node to atomic_addr_clean node.");
      return ret;
    }
  }

  // for HCOM atomic node, add one more control link to peer-in node
  for (auto &node : hcom_node_vec_) {
    for (auto &in_anchor : node->GetAllInDataAnchors()) {
      GE_CHECK_NOTNULL(in_anchor->GetPeerOutAnchor());
      NodePtr peer_in_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
      Status ret = LinkToAtomicNode(peer_in_node, clean_addr_node);
      if (ret != SUCCESS) {
        GELOGE(ret, "Link failed, %s : %s", peer_in_node->GetName().c_str(), clean_addr_node->GetName().c_str());
        return ret;
      }
    }
  }
  return SUCCESS;
}

NodePtr AtomicAddrCleanPass::InsertAtomicAddrCleanNode(ComputeGraphPtr &graph) {
  OpDescPtr op_desc = MakeShared<OpDesc>(NODE_NAME_ATOMIC_ADDR_CLEAN, ATOMICADDRCLEAN);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Make shared atomic addr clean op failed.");
    return nullptr;
  }
  string session_graph_id;
  if (!AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Get graph session_graph_id attr failed.");
  }
  if (!session_graph_id.empty()) {
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  }

  string name = op_desc->GetName() + session_graph_id;
  op_desc->SetName(name);
  GELOGI("Create cleanAddr op:%s.", op_desc->GetName().c_str());
  // To avoid same name between graphs, set session graph id to this node
  NodePtr clean_addr_node = graph->AddNodeFront(op_desc);
  return clean_addr_node;
}

Status AtomicAddrCleanPass::LinkToAtomicNode(const NodePtr &atomic_node, NodePtr &atomic_clean_node) {
  GE_IF_BOOL_EXEC(atomic_node == nullptr || atomic_clean_node == nullptr,
                  DOMI_LOGE("param [atomic_node][atomic_clean_node] must not be null.");
                  return PARAM_INVALID);
  InControlAnchorPtr in_ctrl_anchor = atomic_node->GetInControlAnchor();
  OutControlAnchorPtr out_ctrl_anchor = atomic_clean_node->GetOutControlAnchor();
  if (in_ctrl_anchor == nullptr || out_ctrl_anchor == nullptr) {
    GELOGE(INTERNAL_ERROR, "Get control anchor faild, dst node: %s.", atomic_node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  graphStatus status = GraphUtils::AddEdge(out_ctrl_anchor, in_ctrl_anchor);
  if (status != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Graph add cleanAddrNode op out ctrl edge fail, dst node: %s.",
           atomic_node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  GELOGD("Graph add cleanAddrNode op out ctrl edge, dst node: %s.", atomic_node->GetName().c_str());
  std::string stream_label;
  if (is_loop_graph && AttrUtils::GetStr(atomic_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
    if (!AttrUtils::SetStr(atomic_clean_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
      GELOGW("LinkToAtomicNode: SetStr failed");
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

bool AtomicAddrCleanPass::IsAtomicOp(const NodePtr &node) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(FAILED, "node is null."); return false);
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  // 1.Check if isAtomic attrs exist for HCOM
  std::shared_ptr<GELib> instance_ptr = GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGW("GELib not initialized");
    return false;
  }

  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  vector<OpInfo> op_info_vec = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  for (const auto &op_info : op_info_vec) {
    if (op_info.isAtomic) {
      GELOGI("Recognized atomic op %s from HCCL engine.", op_desc->GetName().c_str());
      hcom_node_vec_.push_back(node);
      return true;
    }
  }
  // 2.Check atomic attr in node
  std::map<string, std::map<int, int>> node_workspace_offset;
  bool has_atomic_input = op_desc->HasAttr(ATOMIC_ATTR_INPUT_INDEX);
  bool has_atomic_output = op_desc->HasAttr(ATOMIC_ATTR_OUTPUT_INDEX);
  node_workspace_offset = op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_OFFSET, node_workspace_offset);
  if (!has_atomic_input && !has_atomic_output && node_workspace_offset.empty()) {
    return false;
  }

  graphStatus ret = op_desc->SetAttr(ATOMIC_ATTR_IS_ATOMIC_NODE, GeAttrValue::CreateFrom<GeAttrValue::BOOL>(true));
  if (ret != GRAPH_SUCCESS) {
    GELOGW("set attr ATOMIC_ATTR_IS_ATOMIC_NODE fail.");
  }
  GELOGD("Recognized atomic op %s from FE engine.", op_desc->GetName().c_str());
  return true;
}
}  // namespace ge
