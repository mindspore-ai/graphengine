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

#include "graph/passes/folding_pass.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "inc/kernel.h"
#include "inc/kernel_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_local_engine/engine/host_cpu_engine.h"

namespace ge {
namespace folding_pass {
shared_ptr<Kernel> GetKernelByType(const NodePtr &node) {
  if (node == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return nullptr;
  }
  KernelFactory &factory = KernelFactory::Instance();
  string type = node->GetType();
  if (type == FRAMEWORKOP) {
    if (!ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type)) {
      return nullptr;
    }
  }

  return factory.Create(type);
}
bool IsNoNeedConstantFolding(const NodePtr &node) {
  auto node_desc = node->GetOpDesc();
  return node_desc == nullptr || node_desc->HasAttr(ATTR_NO_NEED_CONSTANT_FOLDING);
}
}  // namespace folding_pass

namespace {
IndexsToAnchors GetIndexAndPeerInDataAnchors(NodePtr &node) {
  IndexsToAnchors indexes_to_anchors;
  for (auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    for (auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor == nullptr) {
        continue;
      }
      const auto &peer_node = peer_in_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      indexes_to_anchors[out_anchor->GetIdx()].push_back(peer_in_anchor);
    }
  }

  return indexes_to_anchors;
}

NodePtr AddConstNodeToGraph(GeTensorPtr &tensor, ComputeGraphPtr &graph) {
  auto const_desc = OpDescUtils::CreateConstOp(tensor);
  if (const_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed to get const desc from tensor");
    return nullptr;
  }

  GE_IF_BOOL_EXEC(graph == nullptr, GELOGW("input param graph is null"); return nullptr);
  (void)AttrUtils::SetListStr(const_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::move(std::vector<std::string>()));
  return graph->AddNodeFront(const_desc);
}

NodePtr AddIdentityNodeToGraph(const std::string &name, const GeTensorDesc &tensor, ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    GELOGE(INTERNAL_ERROR, "Compute graph ptr is null in creating identity node.");
    return nullptr;
  }

  OpDescPtr desc = MakeShared<OpDesc>("", "");
  if (desc == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to create op desc.");
    return nullptr;
  }

  desc->SetName(name);
  desc->SetType(IDENTITY);
  auto ret = desc->AddInputDesc(tensor);
  auto ret2 = desc->AddOutputDesc(tensor);
  if ((ret != GRAPH_SUCCESS) || (ret2 != GRAPH_SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "Failed to add input/output desc in creating Identity.");
    return nullptr;
  }

  return graph->AddNodeFront(desc);
}
}  // namespace

Status FoldingPass::RunOpKernel(NodePtr &node, const vector<ConstGeTensorPtr> &inputs,
                                std::vector<GeTensorPtr> &outputs) {
  return HostCpuEngine::GetInstance().Run(node, inputs, outputs);
}

Status FoldingPass::Folding(NodePtr &node, vector<GeTensorPtr> &outputs) {
  GE_CHECK_NOTNULL(node);
  GELOGD("begin folding node:%s", node->GetName().c_str());
  // Before processing nodes, collect the relations between the out anchor and the peer out data nodes
  // to prepare for const reconnection
  auto indexes_to_anchors = GetIndexAndPeerInDataAnchors(node);

  auto ret = DealWithInNodes(node);
  if (ret != SUCCESS) {
    return ret;
  }
  if (AddConstNode(node, indexes_to_anchors, outputs) != SUCCESS) {
    return INTERNAL_ERROR;
  }

  auto in_data_nodes = node->GetInDataNodes();
  std::unordered_set<NodePtr> in_data_nodes_set(in_data_nodes.begin(), in_data_nodes.end());
  if (IsolateAndDeleteNode(node, {}) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to isolate and delete node %s, type %s.", node->GetName().c_str(),
           node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  for (auto iter = in_data_nodes_set.begin(); iter != in_data_nodes_set.end(); ++iter) {
    auto pre_node = *iter;
    if (pre_node->GetOutDataNodesSize() == 0) {
      if (pre_node->GetType() == DATA) {
        GELOGI("No need to remove data, node name:%s.", pre_node->GetName().c_str());
        continue;
      }
      if (IsolateAndDeleteNode(pre_node, {}) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to isolate and delete in data node %s, type %s.", pre_node->GetName().c_str(),
               pre_node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status FoldingPass::DealWithInNodes(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  auto graph = node->GetOwnerComputeGraph();
  auto in_data_anchors = node->GetAllInDataAnchors();
  for (auto &in_data_anchor : in_data_anchors) {
    if (in_data_anchor == nullptr) {
      continue;
    }
    auto in_node_anchor = in_data_anchor->GetPeerOutAnchor();
    if (in_node_anchor == nullptr) {
      continue;
    }
    auto in_node = in_node_anchor->GetOwnerNode();
    if (in_node == nullptr) {
      continue;
    }
    if ((in_node->GetType() == SWITCH) || (in_node->GetType() == REFSWITCH)) {
      GELOGI("The in_node name is %s, and node type is %s.", in_node->GetName().c_str(), in_node->GetType().c_str());
      auto ret = in_node_anchor->Unlink(in_data_anchor);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to unlink anchor between const node %s to constant-folding-node %s, type %s.",
               in_node->GetName().c_str(), node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
      GELOGI("Unlink anchor between in_node %s and node %s success.", in_node->GetName().c_str(),
             node->GetName().c_str());
      auto identity_name = node->GetName() + "_ctrl_identity_" + std::to_string(in_data_anchor->GetIdx());
      auto identity =
        AddIdentityNodeToGraph(identity_name, node->GetOpDesc()->GetInputDesc(in_data_anchor->GetIdx()), graph);
      if (identity == nullptr) {
        GELOGE(INTERNAL_ERROR, "Failed to add identity node to graph.");
        return INTERNAL_ERROR;
      }
      ret = GraphUtils::AddEdge(in_node_anchor, identity->GetInDataAnchor(0));
      if (ret != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to add edge, from node %s to node %s.", in_node->GetName().c_str(),
               identity->GetName().c_str());
        return INTERNAL_ERROR;
      }
      GELOGI("Create new identity node success.");
      ret = GraphUtils::AddEdge(identity->GetOutControlAnchor(), node->GetInControlAnchor());
      if (ret != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to add edge, from node %s to node %s.", in_node->GetName().c_str(),
               node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status FoldingPass::AddConstNode(NodePtr &node, IndexsToAnchors indexes_to_anchors,
                                 std::vector<GeTensorPtr> &v_weight) {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "node is null");
    return FAILED;
  }
  auto graph = node->GetOwnerComputeGraph();
  for (auto &index_to_anchors : indexes_to_anchors) {
    auto index = static_cast<size_t>(index_to_anchors.first);
    if (index >= v_weight.size()) {
      GELOGE(INTERNAL_ERROR,
             "Failed to constant fold on node %s type %s, "
             "the out nodes num %lu calculated is less than the node out anchor index %zu",
             node->GetName().c_str(), node->GetType().c_str(), v_weight.size(), index);
      return INTERNAL_ERROR;
    }
    GeTensorPtr weight = v_weight[index];
    if (weight == nullptr) {
      GELOGE(INTERNAL_ERROR, "Failed to constant fold on node %s type %s, the %lust node calculated is null",
             node->GetName().c_str(), node->GetType().c_str(), index);
      return INTERNAL_ERROR;
    }

    auto const_node = AddConstNodeToGraph(weight, graph);
    if (const_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "Failed to add dynamic const node, node name:%s, index:%zu.", node->GetName().c_str(),
             index);
      return INTERNAL_ERROR;
    }
    GELOGI("add const_node:%s, replace node %s, type %s, index %zu.", const_node->GetName().c_str(),
           node->GetName().c_str(), node->GetType().c_str(), index);
    // add new const to re-pass node
    for (auto &in_anchor : index_to_anchors.second) {
      if (in_anchor == nullptr) {
        GELOGE(INTERNAL_ERROR, "In anchor is nullptr.");
        return INTERNAL_ERROR;
      }
      auto ret = ConnectNodeToInAnchor(in_anchor, const_node, 0);
      if (ret != SUCCESS) {
        return ret;
      }
      NodeUtils::UpdateIsInputConst(*(in_anchor->GetOwnerNode()));
    }
    Status ret = GraphUtils::AddEdge(node->GetOutControlAnchor(), const_node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to add control edge, from node %s to const node %s.", node->GetName().c_str(),
             const_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string stream_label;
    if (AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
      GE_CHECK_NOTNULL(const_node->GetOpDesc());
      if (!AttrUtils::SetStr(const_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
        GELOGE(INTERNAL_ERROR, "Failed to set stream label on dynamic const node %s, with stream label:%s.",
               const_node->GetName().c_str(), stream_label.c_str());
        return INTERNAL_ERROR;
      }
    }
    GELOGD("Add control edge when insert dynamic const, from node %s to const node %s, with stream label:%s.",
           node->GetName().c_str(), const_node->GetName().c_str(), stream_label.c_str());
  }

  return SUCCESS;
}

Status FoldingPass::RemoveNodeKeepingCtrlEdges(NodePtr &node) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(PARAM_INVALID, "node is null"); return PARAM_INVALID);
  auto ret = GraphUtils::IsolateNode(node, {});
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to isolate the folding-node %s type %s", node->GetName().c_str(),
           node->GetType().c_str());
    return INTERNAL_ERROR;
  }

  auto graph = node->GetOwnerComputeGraph();
  ret = GraphUtils::RemoveNodeWithoutRelink(graph, node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to remove node %s from graph", node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  AddNodeDeleted(node);
  return SUCCESS;
}

Status FoldingPass::ConnectNodeToInAnchor(InDataAnchorPtr &in_anchor, NodePtr &node, int node_index) {
  // the origin edge must be removed before add
  if (in_anchor == nullptr || node == nullptr) {
    GELOGE(PARAM_INVALID, "in anchor or node is null");
    return PARAM_INVALID;
  }
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor != nullptr) {
    if (ge::GraphUtils::RemoveEdge(peer_out_anchor, in_anchor) != GRAPH_SUCCESS) {
      GELOGW("RemoveEdge failed.");
    }
  }

  auto new_out_anchor = node->GetOutDataAnchor(node_index);
  if (new_out_anchor == nullptr) {
    GELOGE(INTERNAL_ERROR,
           "Failed to add node to in anchor,"
           " the index %d for node %s, type %s is invalid",
           node_index, node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (GraphUtils::AddEdge(new_out_anchor, in_anchor) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR,
           "Failed to add edge between anchors,"
           " new node %s, type %s",
           node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  AddRePassNodesWithInOut(node);
  return SUCCESS;
}
}  // namespace ge
