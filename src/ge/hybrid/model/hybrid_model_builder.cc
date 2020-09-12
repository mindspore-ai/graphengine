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

#include "hybrid/model/hybrid_model_builder.h"
#include "common/math/math_util.h"
#include "graph/ge_context.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
namespace {
const uint32_t kSubgraphIndex = 0U;
const uint32_t kVarOutputIndex = 0U;
const uint32_t kAlignment = 32;
const int kBytes = 8;

int64_t CalcVarSizeInBytes(const GeTensorDesc &desc) {
  int64_t var_size = 0;
  auto data_type = desc.GetDataType();
  if (data_type == DT_STRING) {
    (void)TensorUtils::GetSize(desc, var_size);
  } else {
    var_size = GetSizeByDataType(data_type);
    if (var_size <= 0) {
      GELOGW("Failed to calc var data size from data type %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
      return -1;
    }
    auto shape = desc.GetShape();
    auto dim_num = shape.GetDimNum();
    for (size_t dim_index = 0; dim_index < dim_num; ++dim_index) {
      var_size *= shape.GetDim(dim_index);
    }
    // padding up to multiple of kAlignment, and add extra kAlignment
    var_size = (var_size + kAlignment * 2 - 1) / kAlignment * kAlignment;
  }
  return var_size;
}
}  // namespace
HybridModelBuilder::HybridModelBuilder(HybridModel &hybrid_model)
    : hybrid_model_(hybrid_model), runtime_param_(hybrid_model.root_runtime_param_) {
  ge_root_model_ = hybrid_model_.ge_root_model_;
}

Status HybridModelBuilder::Build() {
  GE_CHK_STATUS_RET(ValidateParams(), "Failed to validate GeRootModel");
  hybrid_model_.model_name_ = ge_root_model_->GetRootGraph()->GetName();
  GELOGI("[%s] Start to build hybrid model.", GetGraphName());
  GE_CHK_STATUS_RET(InitRuntimeParams(), "[%s] Failed to InitRuntimeParams", GetGraphName());
  GE_CHK_STATUS_RET(IndexSpecialNodes(), "[%s] Failed to index nodes", GetGraphName());
  GE_CHK_STATUS_RET(IndexTaskDefs(), "[%s] Failed to index task defs", GetGraphName());
  GE_CHK_STATUS_RET(LoadGraph(), "[%s] Failed to load graph", GetGraphName());
  GE_CHK_STATUS_RET(AssignUninitializedConstantOps(), "[%s] Failed to assign uninitialized constants", GetGraphName());
  GE_CHK_STATUS_RET(TransAllVarData(), "[%s] Failed to trans all var data", GetGraphName());
  GE_CHK_STATUS_RET(CopyVarData(), "[%s] Failed to copy var data", GetGraphName());
  GE_CHK_STATUS_RET(InitModelMem(), "[%s] Failed to init memory", GetGraphName());
  GE_CHK_STATUS_RET(InitWeights(), "[%s] Failed to init weights", GetGraphName());
  GE_CHK_STATUS_RET(InitConstantOps(), "[%s] Failed to init constant op", GetGraphName());
  GE_CHK_STATUS_RET(InitVariableTensors(), "[%s] Failed to init variables", GetGraphName());
  GE_CHK_STATUS_RET(LoadTasks(), "[%s] Failed to load tasks", GetGraphName());
  GELOGI("[%s] Done building hybrid model successfully.", GetGraphName());
  return SUCCESS;
}

Status HybridModelBuilder::ValidateParams() {
  GE_CHECK_NOTNULL(ge_root_model_);
  GE_CHECK_NOTNULL(ge_root_model_->GetRootGraph());
  return SUCCESS;
}

Status HybridModelBuilder::BuildNodeItem(const NodePtr &node, NodeItem &node_item) {
  auto op_desc = node->GetOpDesc();
  vector<string> dependencies = node->GetOpDesc()->GetOpInferDepends();
  GE_CHK_STATUS_RET(ParseDependentInputNodes(node_item, dependencies), "[%s] Failed to parse node dependencies.",
                    node_item.NodeName().c_str());

  node_item.outputs.resize(node_item.num_outputs);
  for (int i = 0; i < node_item.num_outputs; ++i) {
    auto out_data_anchor = node->GetOutDataAnchor(i);
    if (out_data_anchor == nullptr) {
      GELOGE(INTERNAL_ERROR, "out anchor[%d] of node %s is nullptr", i, node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    for (auto &dst_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      auto dst_node = dst_in_anchor->GetOwnerNode();
      if (dst_node == nullptr) {
        GELOGW("dst node is nullptr. out anchor = %d", out_data_anchor->GetIdx());
        continue;
      }

      NodeItem *dst_node_item = nullptr;
      GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, &dst_node_item), "[%s] Failed to get or create node item.",
                        dst_node->GetName().c_str());
      node_item.outputs[i].emplace_back(dst_in_anchor->GetIdx(), dst_node_item);
    }
  }

  GE_CHK_STATUS_RET_NOLOG(ResolveRefIo(node_item));
  return SUCCESS;
}

Status HybridModelBuilder::ResolveRefIo(NodeItem &node_item) {
  bool is_ref = false;
  auto &op_desc = *node_item.op_desc;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (!is_ref) {
    return SUCCESS;
  }

  auto inputs = op_desc.GetAllInputName();
  auto outputs = op_desc.GetAllOutputName();
  for (auto &output : outputs) {
    for (auto &input : inputs) {
      if (input.first == output.first) {
        auto input_idx = static_cast<int>(input.second);
        auto output_idx = static_cast<int>(output.second);
        node_item.reuse_inputs[output_idx] = input_idx;
        GELOGD("[%s] Output[%d] reuse input[%d]", node_item.NodeName().c_str(), output_idx, input_idx);
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::GetOrCreateNodeItem(const NodePtr &node, NodeItem **node_item) {
  auto &node_items = hybrid_model_.node_items_;
  auto it = node_items.find(node);
  if (it != node_items.end()) {
    *node_item = it->second.get();
    return SUCCESS;
  }

  auto new_node = std::unique_ptr<NodeItem>(new (std::nothrow) NodeItem(node));
  GE_CHECK_NOTNULL(new_node);
  GE_CHECK_NOTNULL(new_node->op_desc);
  GE_CHK_STATUS_RET(new_node->Init(), "Failed to init NodeItem [%s] .", node->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(NodeExecutorManager::GetInstance().GetExecutor(*node, &new_node->node_executor));

  // we do not need L2 Buffer
  const char *const kIsFirstNode = "is_first_node";
  const char *const kIsLastNode = "is_last_node";
  (void)AttrUtils::SetBool(new_node->op_desc, kIsFirstNode, false);
  (void)AttrUtils::SetBool(new_node->op_desc, kIsLastNode, false);

  if (new_node->is_dynamic && (new_node->IsControlOp() || new_node->NodeType() == PARTITIONEDCALL)) {
    new_node->shape_inference_type = DEPEND_COMPUTE;
  }

  new_node->node_id = node_index;
  new_node->op_desc->SetId(node_index);
  node_index += 1;

  *node_item = new_node.get();
  node_items[node] = std::move(new_node);
  return SUCCESS;
}

Status HybridModelBuilder::ParseDependentInputNodes(NodeItem &node_item, const std::vector<string> &dependencies) {
  std::set<NodePtr> dependent_input_nodes;
  auto &ge_node = node_item.node;

  // The input tensors become valid after computation is done for parent nodes of type DEPEND_COMPUTE.
  // Wait for these parent nodes before execution.
  for (const auto &in_anchor : ge_node->GetAllInDataAnchors()) {
    const auto &peer_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      GELOGD("[%s] Input[%d] do not have peer anchor", node_item.NodeName().c_str(), in_anchor->GetIdx());
      continue;
    }
    auto src_node = peer_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    auto src_node_item = MutableNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);

    if (src_node_item->shape_inference_type == DEPEND_COMPUTE) {
      GELOGD("[%s] Add input data dependent node [%s] due to inference type = DEPEND_COMPUTE",
             node_item.NodeName().c_str(), src_node_item->NodeName().c_str());

      src_node_item->has_observer = true;
      node_item.dependents_for_execution.emplace_back(src_node);
    }

    if (src_node_item->shape_inference_type == DEPEND_SHAPE_RANGE) {
      GELOGD("[%s] Add input shape dependent node [%s] due to inference type = DEPEND_SHAPE_RANGE",
             node_item.NodeName().c_str(), src_node_item->NodeName().c_str());
      src_node_item->has_observer = true;
      dependent_input_nodes.emplace(src_node);
    }
  }

  // cond or branch need to be prepared before the execution of IF or CASE
  if (node_item.node_type == IF || node_item.node_type == CASE) {
    const auto &in_anchor = ge_node->GetInDataAnchor(0);
    GE_CHECK_NOTNULL(in_anchor);
    const auto &peer_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_anchor);
    auto src_node = peer_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    auto src_node_item = MutableNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    src_node_item->has_observer = true;
    node_item.dependents_for_execution.emplace_back(src_node);
    GELOGD("[%s] Dependent added from %s for control op's cond/branch", node_item.NodeName().c_str(),
           src_node_item->NodeName().c_str());
  }

  for (const auto &input_name : dependencies) {
    int input_index = node_item.op_desc->GetInputIndexByName(input_name);
    if (input_index < 0) {
      GELOGE(INTERNAL_ERROR, "[%s] Failed to get input index by name: %s", node_item.NodeName().c_str(),
             input_name.c_str());
      return INTERNAL_ERROR;
    }

    const auto &in_anchor = ge_node->GetInDataAnchor(input_index);
    GE_CHECK_NOTNULL(in_anchor);
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    auto src_node_item = MutableNodeItem(src_node);
    src_node_item->to_const_output_id_list.emplace(peer_out_anchor->GetIdx());
    src_node_item->has_observer = true;

    dependent_input_nodes.emplace(src_node);
    GELOGD("[%s] Dependent added from output of [%s:%d]", node_item.NodeName().c_str(),
           src_node_item->NodeName().c_str(), peer_out_anchor->GetIdx());
  }

  for (const auto &dep_node : dependent_input_nodes) {
    node_item.dependents_for_shape_inference.emplace_back(dep_node);
  }

  return SUCCESS;
}

Status HybridModelBuilder::UpdateAnchorStatus(const NodePtr &node) {
  if (NodeUtils::SetAllAnchorStatus(node) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[%s] NodeUtils::SetAllAnchorStatus failed.", node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  for (auto &anchor : node->GetAllInDataAnchors()) {
    auto peer_anchor = anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_SUSPEND) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[%s] AnchorUtils::SetStatus failed.", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    } else if (peer_anchor->GetOwnerNode()->GetType() == CONSTANT) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_CONST) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[%s] AnchorUtils::SetStatus failed.", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    } else {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_DATA) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[%s] AnchorUtils::SetStatus failed.", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::DoUnlinkDataAnchors(const OutDataAnchorPtr &out_data_anchor,
                                               const InDataAnchorPtr &in_data_anchor) {
  GE_CHK_GRAPH_STATUS_RET(out_data_anchor->Unlink(in_data_anchor), "Failed to unlink %s:%d from %s:%d",
                          out_data_anchor->GetOwnerNode()->GetName().c_str(), out_data_anchor->GetIdx(),
                          in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetIdx());

  GELOGD("Succeeded in unlinking %s:%d from %s:%d", out_data_anchor->GetOwnerNode()->GetName().c_str(),
         out_data_anchor->GetIdx(), in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetIdx());
  return SUCCESS;
}

Status HybridModelBuilder::DoLinkDataAnchors(OutDataAnchorPtr &out_data_anchor, InDataAnchorPtr &in_data_anchor) {
  GE_CHK_GRAPH_STATUS_RET(out_data_anchor->LinkTo(in_data_anchor), "Failed to link %s:%d to %s:%d",
                          out_data_anchor->GetOwnerNode()->GetName().c_str(), out_data_anchor->GetIdx(),
                          in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetIdx());

  GELOGD("Succeeded in linking %s:%d to %s:%d", out_data_anchor->GetOwnerNode()->GetName().c_str(),
         out_data_anchor->GetIdx(), in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetIdx());
  return SUCCESS;
}

Status HybridModelBuilder::MergeInputNodes(ComputeGraph &graph) {
  const auto &wrapped_node = graph.GetParentNode();
  std::set<NodePtr> root_nodes;
  for (const auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != DATA_TYPE) {
      if (node->GetInDataNodes().empty()) {
        root_nodes.emplace(node);
      }

      continue;
    }

    auto data_op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(data_op_desc);

    uint32_t parent_index = 0;
    if (!AttrUtils::GetInt(data_op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(FAILED, "[%s] Failed to get attr [%s]", data_op_desc->GetName().c_str(),
             ATTR_NAME_PARENT_NODE_INDEX.c_str());
      return FAILED;
    }

    auto wrapped_node_in_anchor = wrapped_node->GetInDataAnchor(parent_index);
    GE_CHECK_NOTNULL(wrapped_node_in_anchor);
    auto src_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr || src_out_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    auto src_node = wrapped_node_in_anchor->GetPeerOutAnchor()->GetOwnerNode();
    wrapped_node_in_anchor->UnlinkAll();

    // link src to outputs of DataNode
    for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(out_data_anchor);
      for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        auto dst_node = peer_in_data_anchor->GetOwnerNode();
        root_nodes.emplace(dst_node);
        GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(out_data_anchor, peer_in_data_anchor));
        GE_CHK_STATUS_RET_NOLOG(DoLinkDataAnchors(src_out_anchor, peer_in_data_anchor));
      }
    }
  }

  // transfer in control edges to all root nodes
  for (auto &root_node : root_nodes) {
    auto in_nodes = root_node->GetInAllNodes();
    std::set<NodePtr> in_node_set(in_nodes.begin(), in_nodes.end());
    for (auto &in_control_node : wrapped_node->GetInControlNodes()) {
      if (in_node_set.count(in_control_node) == 0) {
        GELOGD("[%s] Restore control edge to [%s]", in_control_node->GetName().c_str(), root_node->GetName().c_str());
        GE_CHECK_NOTNULL(in_control_node->GetOutControlAnchor());
        (void)in_control_node->GetOutControlAnchor()->LinkTo(root_node->GetInControlAnchor());
      }
    }
  }

  wrapped_node->GetInControlAnchor()->UnlinkAll();
  return SUCCESS;
}

Status HybridModelBuilder::MergeNetOutputNode(ComputeGraph &graph) {
  const auto &parent_node = graph.GetParentNode();
  const NodePtr &net_output_node = graph.FindFirstNodeMatchType(NETOUTPUT);
  GE_CHECK_NOTNULL(net_output_node);
  const auto &net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  auto all_in_nodes = net_output_node->GetInAllNodes();
  auto all_out_nodes = parent_node->GetOutAllNodes();
  net_output_node->GetInControlAnchor()->UnlinkAll();
  parent_node->GetOutControlAnchor()->UnlinkAll();

  for (const auto &in_data_anchor : net_output_node->GetAllInDataAnchors()) {
    auto src_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(src_out_anchor);
    GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(src_out_anchor, in_data_anchor));

    auto index = in_data_anchor->GetIdx();
    auto input_desc = net_output_desc->MutableInputDesc(index);
    if (input_desc == nullptr) {
      GELOGE(INTERNAL_ERROR, "[%s] Failed to get input desc[%d]", net_output_desc->GetName().c_str(), index);
      return INTERNAL_ERROR;
    }

    uint32_t parent_index = 0;
    if (!AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGW("SubGraph: %s NetOutput input tensor %d, attr %s not found.", graph.GetName().c_str(), index,
             ATTR_NAME_PARENT_NODE_INDEX.c_str());
      continue;
    }

    const OutDataAnchorPtr &parent_out_anchor = parent_node->GetOutDataAnchor(parent_index);
    GE_CHECK_NOTNULL(parent_out_anchor);
    for (InDataAnchorPtr &dst_in_anchor : parent_out_anchor->GetPeerInDataAnchors()) {
      if (dst_in_anchor == nullptr) {
        continue;
      }

      GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(parent_out_anchor, dst_in_anchor));
      GE_CHK_STATUS_RET_NOLOG(DoLinkDataAnchors(src_out_anchor, dst_in_anchor));
    }
  }

  // transfer out control edges
  std::set<NodePtr> in_node_set(all_in_nodes.begin(), all_in_nodes.end());
  std::set<NodePtr> out_node_set(all_out_nodes.begin(), all_out_nodes.end());
  for (auto &src_node : in_node_set) {
    GELOGD("[%s] process in node.", src_node->GetName().c_str());
    auto out_nodes = src_node->GetOutAllNodes();
    std::set<NodePtr> node_set(out_nodes.begin(), out_nodes.end());
    for (auto &dst_node : out_node_set) {
      if (node_set.count(dst_node) == 0) {
        src_node->GetOutControlAnchor()->LinkTo(dst_node->GetInControlAnchor());
        GELOGD("[%s] Restore control edge to [%s]", src_node->GetName().c_str(), dst_node->GetName().c_str());
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::UnfoldSubgraphs(ComputeGraph &root_graph, ComputeGraphPtr &merged_graph) {
  merged_graph = MakeShared<ComputeGraph>("MergedGraph");
  for (const auto &node : root_graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    const auto &op_type = node->GetType();
    if (op_type != PARTITIONEDCALL) {
      merged_graph->AddNode(node);
      GELOGD("[%s] Node added to merged graph.", op_desc->GetName().c_str());
      continue;
    }

    bool is_unknown_shape = false;
    GE_CHK_GRAPH_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown_shape),
                            "Failed to invoke GetNodeUnknownShapeStatus.");
    if (!is_unknown_shape) {
      merged_graph->AddNode(node);
      GELOGD("[%s] Known shape partitioned call added to merged graph.", op_desc->GetName().c_str());
      continue;
    }

    auto subgraph = NodeUtils::GetSubgraph(*node, kSubgraphIndex);
    GE_CHECK_NOTNULL(subgraph);
    GE_CHK_GRAPH_STATUS_RET(UnfoldSubgraph(root_graph, *merged_graph, *subgraph), "[%s] Failed to merge subgraph.",
                            subgraph->GetName().c_str());
  }

  // invoke before adding subgraphs. in case modify node id in known-shaped subgraphs.
  GE_CHK_GRAPH_STATUS_RET(merged_graph->TopologicalSorting(), "Failed to invoke TopologicalSorting on merged graph.");

  for (auto &remained_subgraph : root_graph.GetAllSubgraphs()) {
    GELOGD("Adding subgraph [%s] to merged-graph.", remained_subgraph->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(merged_graph->AddSubgraph(remained_subgraph), "Failed to add subgraph [%s]",
                            remained_subgraph->GetName().c_str());
  }

  return SUCCESS;
}

Status HybridModelBuilder::UnfoldSubgraph(ComputeGraph &root_graph, ComputeGraph &parent_graph,
                                          ComputeGraph &sub_graph) {
  auto parent_node = sub_graph.GetParentNode();
  GE_CHECK_NOTNULL(parent_node);

  GE_CHK_STATUS_RET(MergeInputNodes(sub_graph), "[%s] Failed to merge data nodes for subgraph",
                    sub_graph.GetName().c_str());
  GE_CHK_STATUS_RET(MergeNetOutputNode(sub_graph), "[%s] Failed to merge net output nodes for subgraph",
                    sub_graph.GetName().c_str());
  GELOGD("[%s] Done merging subgraph inputs and outputs successfully.", sub_graph.GetName().c_str());

  for (auto &sub_node : sub_graph.GetDirectNode()) {
    auto sub_op_type = sub_node->GetType();
    if (sub_op_type == DATA_TYPE || sub_op_type == NETOUTPUT) {
      continue;
    }

    if (sub_op_type == CONSTANT || sub_op_type == VARIABLE) {
      GELOGE(INTERNAL_ERROR, "Unexpected node in unknown subgraph. type = %s, node = %s::%s", sub_op_type.c_str(),
             sub_graph.GetName().c_str(), sub_node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    if (sub_op_type == PARTITIONEDCALL) {
      bool is_unknown_shape = false;
      GE_CHK_GRAPH_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*sub_node, is_unknown_shape),
                              "[%s] Failed to invoke GetNodeUnknownShapeStatus.", sub_node->GetName().c_str());
      if (is_unknown_shape) {
        auto sub_sub_graph = NodeUtils::GetSubgraph(*sub_node, kSubgraphIndex);
        GE_CHECK_NOTNULL(sub_sub_graph);
        GE_CHK_STATUS_RET(UnfoldSubgraph(root_graph, parent_graph, *sub_sub_graph), "[%s] Failed to merge subgraph",
                          sub_sub_graph->GetName().c_str());
        continue;
      }
    }

    parent_graph.AddNode(sub_node);
    GELOGD("[%s::%s] added to parent graph: [%s].", sub_graph.GetName().c_str(), sub_node->GetName().c_str(),
           parent_graph.GetName().c_str());
  }

  GELOGD("[%s] Done merging subgraph. remove it from root graph.", sub_graph.GetName().c_str());
  root_graph.RemoveSubgraph(sub_graph.GetName());
  return SUCCESS;
}

Status HybridModelBuilder::BuildOutputMapping(GraphItem &graph_item, const NodeItem &node_item, bool is_root_graph) {
  auto output_size = node_item.op_desc->GetAllInputsSize();
  GE_CHECK_LE(output_size, UINT32_MAX);
  graph_item.output_edges_.resize(output_size);

  for (auto &in_data_anchor : node_item.node->GetAllInDataAnchors()) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    auto src_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);

    auto src_node_item = GetNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    auto output_offset = src_node_item->output_start + peer_out_anchor->GetIdx();
    GELOGI("Output[%d], node = %s, output_index = %d, output_offset = %d ", in_data_anchor->GetIdx(),
           src_node_item->NodeName().c_str(), peer_out_anchor->GetIdx(), output_offset);

    graph_item.output_edges_[in_data_anchor->GetIdx()] = {src_node_item, peer_out_anchor->GetIdx()};
  }

  if (!is_root_graph) {
    for (uint32_t i = 0; i < static_cast<uint32_t>(output_size); ++i) {
      uint32_t p_index = i;
      // Net output of Subgraph of while do not have parent index
      if (AttrUtils::GetInt(node_item.op_desc->GetInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, p_index)) {
        GELOGD("[%s] Parent index not set for input[%u].", node_item.NodeName().c_str(), i);
      }

      graph_item.output_index_mapping_.emplace_back(p_index);
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::LoadGraph() {
  auto root_graph = ge_root_model_->GetRootGraph();
  if (!GetContext().GetHostExecFlag()) {
    std::shared_ptr<ComputeGraph> merged_graph;
    GELOGI("Before merging subgraphs DirectNodesSize = %zu, GetAllNodesSize = %zu", root_graph->GetDirectNodesSize(),
           root_graph->GetAllNodesSize());
    GE_CHK_GRAPH_STATUS_RET(UnfoldSubgraphs(*root_graph, merged_graph), "Failed to unfold subgraphs.");
    root_graph = std::move(merged_graph);
    GELOGI("After merging subgraphs DirectNodesSize = %zu, GetAllNodesSize = %zu", root_graph->GetDirectNodesSize(),
           root_graph->GetAllNodesSize());
    GE_DUMP(root_graph, "hybrid_merged_graph");
  }

  GE_CHK_STATUS_RET(LoadDynamicSubgraph(*root_graph, true), "Failed to load root graph.");
  GELOGD("Done loading root graph successfully.");

  for (auto &sub_graph : root_graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(sub_graph);
    GELOGD("Start to load subgraph [%s]", sub_graph->GetName().c_str());
    auto parent_node = sub_graph->GetParentNode();
    GE_CHECK_NOTNULL(parent_node);
    auto parent_node_item = MutableNodeItem(parent_node);
    // parent node is in another known subgraph
    if (parent_node_item == nullptr) {
      GELOGD("[%s] Subgraph is in another known shaped subgraph, skip it.", sub_graph->GetName().c_str());
      continue;
    }

    if (sub_graph->GetGraphUnknownFlag()) {
      GE_CHK_STATUS_RET(LoadDynamicSubgraph(*sub_graph, false), "Failed to load subgraph: [%s]",
                        sub_graph->GetName().c_str());
    } else {
      GE_CHK_STATUS_RET(IdentifyVariableOutputs(*parent_node_item), "[%s] Failed to identify ref outputs.",
                        parent_node_item->NodeName().c_str());

      // if parent is function control op. need add a virtual partitioned call
      if (parent_node_item->IsControlOp()) {
        GE_CHK_STATUS_RET(LoadKnownShapedSubgraph(*sub_graph, parent_node_item),
                          "Failed to load function control op subgraph [%s]", sub_graph->GetName().c_str());
      }
    }
  }

  GELOGI("Done loading all subgraphs successfully.");
  return SUCCESS;
}

const NodeItem *HybridModelBuilder::GetNodeItem(const NodePtr &node) const { return hybrid_model_.GetNodeItem(node); }

NodeItem *HybridModelBuilder::MutableNodeItem(const NodePtr &node) { return hybrid_model_.MutableNodeItem(node); }

Status HybridModelBuilder::VarNodeToTensor(const NodePtr &var_node, std::unique_ptr<TensorValue> &tensor) {
  string var_name = var_node->GetName();
  auto tensor_desc = var_node->GetOpDesc()->MutableOutputDesc(0);
  uint8_t *var_logic = nullptr;
  GE_CHK_STATUS_RET(var_manager_->GetVarAddr(var_name, *tensor_desc, &var_logic),
                    "Failed to get var addr. var_name = %s, session_id = %ld", var_name.c_str(),
                    hybrid_model_.GetSessionId());

  uint8_t *dev_mem = var_manager_->GetVarMemoryAddr(var_logic, RT_MEMORY_HBM);
  if (dev_mem == nullptr) {
    GELOGE(INTERNAL_ERROR,
           "Failed to copy var %s from device, cant not get "
           "var addr from logic addr %p",
           var_node->GetName().c_str(), var_logic);
    return INTERNAL_ERROR;
  }

  int64_t var_size = CalcVarSizeInBytes(*tensor_desc);
  tensor.reset(new (std::nothrow) TensorValue(dev_mem, var_size));
  GE_CHECK_NOTNULL(tensor);
  return SUCCESS;
}

Status HybridModelBuilder::HandleDtString(const GeTensor &tensor, void *var_addr) {
  auto desc = tensor.GetTensorDesc();
  if (desc.GetDataType() == DT_STRING) {
    GeShape tensor_shape = desc.GetShape();
    /// if tensor is a scaler, it's shape size if zero, according ge_tensor.cc.
    /// the logic of GetShapeSize is wrong, the scaler tensor's GetShapeSize is zero
    /// and that of unknown shape is zero too.
    /// unknown shape will not appear here, so we can use zero judge a tensor is scalar or not
    int64_t elem_num = tensor_shape.GetShapeSize();
    if (elem_num == 0 && tensor_shape.GetDims().empty()) {
      elem_num = 1;
    }

    auto &mutable_tensor = const_cast<GeTensor &>(tensor);
    uint64_t *buff = reinterpret_cast<uint64_t *>(mutable_tensor.MutableData().data());
    GE_CHK_BOOL_RET_STATUS(ge::CheckInt64Uint32MulOverflow(elem_num, kBytes) == SUCCESS, FAILED,
                           "Shape size is invalid");
    auto offset = static_cast<uint64_t>(elem_num * kBytes);
    auto hbm_raw_data_base_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(var_addr) + offset);
    for (int64_t i = elem_num - 1; i >= 0; --i) {
      buff[i] = hbm_raw_data_base_addr + (buff[i] - buff[0]);
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::AssignUninitializedConstantOps() {
  if (GetContext().GetHostExecFlag()) {
    GELOGI("no need to assign when exec on host.");
    return SUCCESS;
  }
  for (auto &it : hybrid_model_.constant_op_nodes_) {
    const string &var_name = it.first;
    const NodePtr &var_node = it.second;
    auto tensor_desc = var_node->GetOpDesc()->MutableOutputDesc(0);
    if (!var_manager_->IsVarExist(var_name, *tensor_desc)) {
      // allocate constant
      GELOGD("[%s] Constant not allocated during graph building. now allocate it.", var_name.c_str());
      GE_CHK_STATUS_RET(var_manager_->AssignVarMem(var_name, *tensor_desc, RT_MEMORY_HBM));
      GE_CHK_STATUS_RET(var_manager_->SetAllocatedGraphId(var_name, runtime_param_.graph_id));
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::InitConstantOps() {
  for (auto &it : hybrid_model_.constant_op_nodes_) {
    const string &var_name = it.first;
    const NodePtr &var_node = it.second;
    std::unique_ptr<TensorValue> var_tensor;

    GE_CHK_STATUS_RET_NOLOG(VarNodeToTensor(var_node, var_tensor));
    GELOGD("Init const op tensor. name = %s, size = %ld", var_name.c_str(), var_tensor->GetSize());
    var_tensor->SetName("ConstOp_" + var_name);

    auto op_desc = var_node->GetOpDesc();
    auto v_weights = ModelUtils::GetWeights(op_desc);
    auto v_output_size = var_tensor->GetSize();
    auto v_output_addr = var_tensor->MutableData();

    auto *ge_tensor = const_cast<GeTensor *>(v_weights[0].get());
    if (ge_tensor->GetData().size() > 0) {
      GE_CHK_STATUS_RET_NOLOG(HandleDtString(*ge_tensor, v_output_addr));

      GELOGI("[IMAS]InitConstant memcpy graph_%u type[V] name[%s] output[%d] memaddr[%p] mem_size[%zu] datasize[%zu]",
             runtime_param_.graph_id, op_desc->GetName().c_str(), 0, v_output_addr, v_output_size,
             ge_tensor->GetData().size());
      GE_CHK_RT_RET(rtMemcpy(v_output_addr, v_output_size, ge_tensor->GetData().data(), ge_tensor->GetData().size(),
                             RT_MEMCPY_HOST_TO_DEVICE));
    } else {
      GELOGI("[%s] Const op has no weight data.", op_desc->GetName().c_str());
    }

    hybrid_model_.variable_tensors_.emplace(var_name, std::move(var_tensor));
  }

  return SUCCESS;
}

Status HybridModelBuilder::InitVariableTensors() {
  for (auto &it : hybrid_model_.variable_nodes_) {
    string var_name = it.first;
    NodePtr &var_node = it.second;
    std::unique_ptr<TensorValue> tensor;
    GE_CHK_STATUS_RET_NOLOG(VarNodeToTensor(var_node, tensor));
    GELOGD("Init variable tensor. name = %s, size = %ld, addr = %p", var_name.c_str(), tensor->GetSize(),
           tensor->GetData());
    tensor->SetName("Var_" + var_name);
    hybrid_model_.variable_tensors_.emplace(var_name, std::move(tensor));
  }

  return SUCCESS;
}

Status HybridModelBuilder::InitWeights() {
  // Train do not have weight. (only got ConstOp)
  return SUCCESS;
}

Status HybridModelBuilder::LoadTasks() {
  for (auto &it : hybrid_model_.node_items_) {
    auto &node_item = it.second;
    auto &node_ptr = node_item->node;
    if (node_item->node_type == NETOUTPUT) {
      continue;
    }

    GELOGD("[%s] Start to build kernel task", node_ptr->GetName().c_str());
    auto load_ret = node_item->node_executor->LoadTask(hybrid_model_, node_ptr, node_item->kernel_task);
    if (load_ret != UNSUPPORTED && load_ret != SUCCESS) {
      GELOGE(load_ret, "[%s] Failed to load task", node_ptr->GetName().c_str());
      return load_ret;
    }

    GELOGD("[%s] Done loading task successfully.", node_ptr->GetName().c_str());
  }

  return SUCCESS;
}

Status HybridModelBuilder::LoadGeModel(ComputeGraph &sub_graph, const GeModelPtr &ge_model) {
  auto parent_node = sub_graph.GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  auto op_type = parent_node->GetType();
  if (op_type == IF || op_type == CASE || op_type == WHILE) {
    GELOGD("Set ge_model for control op subgraph: [%s], task_size = %d", sub_graph.GetName().c_str(),
           ge_model->GetModelTaskDefPtr()->task_size());
    subgraph_models_.emplace(sub_graph.GetName(), ge_model);
  } else {
    GELOGD("Set ge_model for subgraph: [%s], task_size = %d", sub_graph.GetName().c_str(),
           ge_model->GetModelTaskDefPtr()->task_size());
    hybrid_model_.known_shape_sub_models_.emplace(sub_graph.GetParentNode(), ge_model);
  }

  return SUCCESS;
}

Status HybridModelBuilder::IndexTaskDefs() {
  const auto &root_graph = ge_root_model_->GetRootGraph();
  for (auto &it : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    auto &name = it.first;
    auto &ge_model = it.second;
    GE_CHECK_NOTNULL(ge_model);

    const auto &sub_graph = root_graph->GetSubgraph(name);
    if (sub_graph == nullptr) {
      continue;
    }

    bool is_unknown_shape = sub_graph->GetGraphUnknownFlag();
    if (!is_unknown_shape) {
      GE_CHK_STATUS_RET_NOLOG(LoadGeModel(*sub_graph, ge_model));
      continue;
    }

    // index task defs
    GELOGD("To index tasks for subgraph: %s", name.c_str());
    unordered_map<int64_t, NodePtr> node_map;
    for (const auto &node : sub_graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node);
      GE_CHECK_NOTNULL(node->GetOpDesc());
      auto node_id = node->GetOpDesc()->GetId();
      GELOGD("op_index = %ld, node_name = %s", node_id, node->GetName().c_str());
      node_map.emplace(node_id, node);
    }

    auto tasks = ge_model->GetModelTaskDefPtr()->task();
    for (int i = 0; i < tasks.size(); ++i) {
      const domi::TaskDef &task_def = tasks[i];
      GELOGI("Task id = %d, task type = %d", i, task_def.type());
      auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
      uint32_t op_index = -1;
      if (task_type == RT_MODEL_TASK_KERNEL) {
        op_index = task_def.kernel().context().op_index();
      } else if (task_type == RT_MODEL_TASK_KERNEL_EX) {
        op_index = task_def.kernel_ex().op_index();
      } else if (task_type == RT_MODEL_TASK_HCCL) {
        op_index = task_def.kernel_hccl().op_index();
      } else {
        GELOGD("Skip task type: %d", static_cast<int>(task_type));
        continue;
      }

      auto iter = node_map.find(op_index);
      if (iter == node_map.end()) {
        GELOGE(INTERNAL_ERROR, "Failed to get node by index = %u", op_index);
        return INTERNAL_ERROR;
      }

      auto &node = iter->second;
      if (task_type == RT_MODEL_TASK_KERNEL) {
        ge_model->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(node->GetOpDesc());
      }

      GELOGD("Task loaded for node: %s, task type = %d, op_index = %u", node->GetName().c_str(), task_type, op_index);
      hybrid_model_.task_defs_[node].emplace_back(task_def);
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::IndexSpecialNodes() {
  GELOGD("Start to index special nodes");
  const auto &root_graph = ge_root_model_->GetRootGraph();
  for (auto &node : root_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    auto op_type = node->GetType();
    GELOGD("node name = %s, node type = %s", node->GetName().c_str(), node->GetType().c_str());
    if (op_type == VARIABLE) {
      hybrid_model_.variable_nodes_.emplace(node->GetName(), node);
    } else if (op_type == CONSTANTOP) {
      hybrid_model_.constant_op_nodes_.emplace(node->GetName(), node);
    } else if (op_type == DATA && node->GetOwnerComputeGraph() != root_graph) {
      NodePtr src_node;
      int peer_out_index = -1;
      GE_CHK_STATUS_RET_NOLOG(GetPeerNodeAcrossSubGraphs(node, src_node, peer_out_index));
      GELOGD("Got peer node for data node %s, peer node = %s(%s)", node->GetName().c_str(), src_node->GetName().c_str(),
             src_node->GetType().c_str());

      auto src_op_type = src_node->GetType();
      if (src_op_type == CONSTANTOP || src_op_type == VARIABLE) {
        for (auto &dst_node_and_in_anchor : node->GetOutDataNodesAndAnchors()) {
          auto &dst_node = dst_node_and_in_anchor.first;
          auto &in_anchor = dst_node_and_in_anchor.second;
          node_ref_inputs_[dst_node].emplace_back(std::make_pair(in_anchor->GetIdx(), src_node));
        }
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::GetPeerNodeAcrossSubGraphs(const NodePtr &data_node, NodePtr &peer_node,
                                                      int &peer_out_index) {
  auto sub_graph = data_node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(sub_graph);
  GELOGD("To get peer node of %s::%s", sub_graph->GetName().c_str(), data_node->GetName().c_str());
  auto wrapped_node = data_node->GetOwnerComputeGraph()->GetParentNode();
  if (wrapped_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "[%s] Node is in root graph.", data_node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  auto data_op_desc = data_node->GetOpDesc();
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(data_op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(INTERNAL_ERROR, "[%s] Failed to get attr [%s]", data_op_desc->GetName().c_str(),
           ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return INTERNAL_ERROR;
  }

  auto wrapped_node_in_anchor = wrapped_node->GetInDataAnchor(parent_index);
  GE_CHECK_NOTNULL(wrapped_node_in_anchor);
  auto src_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
  if (src_out_anchor == nullptr || src_out_anchor->GetOwnerNode() == nullptr) {
    GELOGE(INTERNAL_ERROR, "[%s] Parent node do not have peer anchor.", data_node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  auto src_wrapped_node_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(src_wrapped_node_out_anchor);
  auto src_wrapped_node = src_wrapped_node_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(src_wrapped_node);

  // connected to root-graph's DATA
  auto src_node_type = src_wrapped_node->GetType();
  if (src_node_type != PARTITIONEDCALL) {
    peer_node = src_wrapped_node;
    peer_out_index = kVarOutputIndex;
    GELOGD("[%s] Node is connected to root graph's node: %s", data_node->GetName().c_str(),
           peer_node->GetName().c_str());
    return SUCCESS;
  }

  auto src_graph = NodeUtils::GetSubgraph(*src_wrapped_node, kSubgraphIndex);
  GE_CHECK_NOTNULL(src_graph);
  auto src_net_output_node = src_graph->FindFirstNodeMatchType(NETOUTPUT);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(src_net_output_node == nullptr, return INTERNAL_ERROR,
                                 "Failed to find NetOutput in subgraph: %s", src_graph->GetName().c_str());
  auto net_output_desc = src_net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  auto out_index = static_cast<uint32_t>(src_wrapped_node_out_anchor->GetIdx());
  GELOGD("src graph = %s, src parent output index = %d", src_graph->GetName().c_str(), out_index);

  // link src to outputs of DataNode
  auto input_size = net_output_desc->GetAllInputsSize();
  GE_CHECK_LE(input_size, UINT32_MAX);
  for (uint32_t i = 0; i < static_cast<uint32_t>(input_size); ++i) {
    uint32_t p_index = 0;
    if (!AttrUtils::GetInt(net_output_desc->GetInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, p_index)) {
      GELOGW("SubGraph: %s input tensor %u attr %s not found.", src_graph->GetName().c_str(), i,
             ATTR_NAME_PARENT_NODE_INDEX.c_str());
      continue;
    }

    GELOGD("NetOutput's input[%u], parent_node_index = %u", i, p_index);
    if (p_index == out_index) {
      auto in_anchor = src_net_output_node->GetInDataAnchor(i);
      GE_CHECK_NOTNULL(in_anchor);
      auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      peer_node = peer_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);
      peer_out_index = peer_out_anchor->GetIdx();
      GELOGD("Found peer node of Data node: %s::%s is %s::%s", sub_graph->GetName().c_str(),
             data_node->GetName().c_str(), src_graph->GetName().c_str(), peer_node->GetName().c_str());
      return SUCCESS;
    }
  }

  GELOGE(FAILED, "Failed to find peer node for %s::%s", sub_graph->GetName().c_str(), data_node->GetName().c_str());
  return FAILED;
}
Status HybridModelBuilder::InitRuntimeParams() {
  int64_t value = 0;
  bool ret = false;
  if (ge_root_model_->GetSubgraphInstanceNameToModel().empty()) {
    GELOGE(INTERNAL_ERROR, "Root model has no sub model");
    return INTERNAL_ERROR;
  }

  // session id and var size is same for every model
  auto first_model = ge_root_model_->GetSubgraphInstanceNameToModel().begin()->second;
  ret = ge::AttrUtils::GetInt(first_model, ge::MODEL_ATTR_SESSION_ID, value);
  runtime_param_.session_id = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(first_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, value);
  runtime_param_.logic_var_base = ret ? static_cast<uint64_t>(value) : 0;
  runtime_param_.graph_id = ge_root_model_->GetRootGraph()->GetGraphID();
  value = 0;
  for (auto &it : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    (void)ge::AttrUtils::GetInt(it.second, ATTR_MODEL_VAR_SIZE, value);
    if (value > 0) {
      runtime_param_.var_size = static_cast<uint64_t>(value);
      break;
    }
  }

  GELOGI("InitRuntimeParams(), session_id:%lu, var_size:%lu. graph_id = %u", runtime_param_.session_id,
         runtime_param_.var_size, runtime_param_.graph_id);

  var_manager_ = VarManager::Instance(runtime_param_.session_id);
  GE_CHECK_NOTNULL(var_manager_);
  return SUCCESS;
}

Status HybridModelBuilder::IdentifyVariableOutputs(NodeItem &node_item) {
  GELOGD("Start to parse outputs of node: %s", node_item.NodeName().c_str());
  auto subgraph = NodeUtils::GetSubgraph(*node_item.node, kSubgraphIndex);
  GE_CHECK_NOTNULL(subgraph);
  auto net_output_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  if (net_output_node == nullptr) {
    GELOGD("[%s] Subgraph do not got net output", subgraph->GetName().c_str());
    return SUCCESS;
  }
  auto net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  // constant/variable connected to net output
  for (const auto &in_data_anchor : net_output_node->GetAllInDataAnchors()) {
    auto src_node = GetPeerNode(in_data_anchor);
    GE_CHECK_NOTNULL(src_node);
    auto src_op_type = src_node->GetType();
    GELOGD("Node %s, output %d, src node = %s, src node type = %s", node_item.NodeName().c_str(),
           in_data_anchor->GetIdx(), src_node->GetName().c_str(), src_op_type.c_str());

    if (src_op_type != CONSTANTOP && src_op_type != VARIABLE) {
      continue;
    }

    uint32_t parent_index = 0;
    GE_CHK_STATUS_RET_NOLOG(GetParentNodeOutputIndex(*net_output_desc, in_data_anchor->GetIdx(), parent_index));
    GELOGD("Got parent output index = %u", parent_index);
    node_item.ref_outputs.emplace(parent_index, src_node);
  }

  // Data nodes marked with REF_VAR_SRC_VAR_NAME
  // Using variable tensor as data's output
  for (auto &node : subgraph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      continue;
    }

    string ref_var_name;
    (void)AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_name);
    if (ref_var_name.empty()) {
      continue;
    }

    GELOGD("Data node ref to variable: %s", ref_var_name.c_str());
    NodePtr src_node;
    auto var_node = hybrid_model_.GetVariableNode(ref_var_name);
    GE_CHECK_NOTNULL(var_node);
    GELOGD("Found var node [%s] by ref_var_name [%s]", var_node->GetName().c_str(), ref_var_name.c_str());
    int peer_output_index = -1;
    GE_CHK_STATUS_RET_NOLOG(GetPeerNodeAcrossSubGraphs(node, src_node, peer_output_index));
    auto src_node_item = MutableNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    src_node_item->ref_outputs.emplace(peer_output_index, var_node);
  }

  return SUCCESS;
}

NodePtr HybridModelBuilder::GetPeerNode(const InDataAnchorPtr &in_data_anchor) {
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  if (peer_out_anchor != nullptr) {
    return peer_out_anchor->GetOwnerNode();
  }

  return nullptr;
}

Status HybridModelBuilder::GetParentNodeOutputIndex(const OpDesc &op_desc, int index, uint32_t &out_index) {
  auto input_desc = op_desc.MutableInputDesc(index);
  GE_CHECK_NOTNULL(input_desc);
  if (!AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, out_index)) {
    GELOGE(INTERNAL_ERROR, "NetOutput input tensor %d, attr %s not found.", index, ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status HybridModelBuilder::InitModelMem() {
  hybrid_model_.var_mem_base_ = var_manager_->GetVarMemoryBase(RT_MEMORY_HBM);
  auto total_var_size = hybrid_model_.TotalVarMemSize();
  if (total_var_size == 0 && !hybrid_model_.constant_op_nodes_.empty()) {
    total_var_size = var_manager_->GetVarMemSize(RT_MEMORY_HBM) > 0 ? var_manager_->GetVarMemMaxSize() : 0;
    GELOGD("Model var size = 0. but got uninitialized constant. set var size to %zu.", total_var_size);
  }

  if (total_var_size > 0 && hybrid_model_.var_mem_base_ == nullptr) {
    GE_CHK_STATUS_RET(var_manager_->MallocVarMemory(total_var_size), "Malloc Var Memory Fail.");
    hybrid_model_.var_mem_base_ = var_manager_->GetVarMemoryBase(RT_MEMORY_HBM);
  }

  runtime_param_.var_base = hybrid_model_.var_mem_base_;
  return SUCCESS;
}

Status HybridModelBuilder::TransAllVarData() {
  GELOGI("TransAllVarData start: session_id:%lu, graph_id: %u.", runtime_param_.session_id, runtime_param_.graph_id);
  rtContext_t ctx = nullptr;
  rtError_t rt_ret = rtCtxGetCurrent(&ctx);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Failed to get current context, error_code is: 0x%X.", rt_ret);
    return RT_FAILED;
  }

  std::vector<NodePtr> variable_node_list;
  for (auto &it : hybrid_model_.variable_nodes_) {
    variable_node_list.emplace_back(it.second);
    GELOGD("[%s] added for trans var data", it.first.c_str());
  }

  GE_CHK_STATUS_RET(
    TransVarDataUtils::TransAllVarData(variable_node_list, runtime_param_.session_id, ctx, runtime_param_.graph_id),
    "TransAllVarData failed.");

  GELOGI("TransAllVarData success.");
  return SUCCESS;
}

Status HybridModelBuilder::CopyVarData() {
  GE_CHK_STATUS_RET(
    TransVarDataUtils::CopyVarData(ge_root_model_->GetRootGraph(), runtime_param_.session_id, hybrid_model_.device_id_),
    "CopyVarData failed.");
  GELOGI("CopyVarData success.");
  return SUCCESS;
}

Status HybridModelBuilder::LoadKnownShapedSubgraph(ComputeGraph &graph, NodeItem *parent_node_item) {
  GELOGD("Start to load known shaped subgraph [%s]", graph.GetName().c_str());
  auto graph_item = std::unique_ptr<GraphItem>(new (std::nothrow) GraphItem());
  GE_CHECK_NOTNULL(graph_item);
  graph_item->is_dynamic_ = false;
  auto subgraph_name = graph.GetName();
  auto wrapper_op_desc = MakeShared<OpDesc>(subgraph_name + "_partitioned_call", PARTITIONEDCALL);
  GE_CHECK_NOTNULL(wrapper_op_desc);

  for (auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto &op_type = node->GetType();

    if (op_type == DATA) {
      int32_t data_index = 0;
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, data_index)) {
        GELOGE(FAILED, "[%s] Failed to get attr [%s]", node->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        return FAILED;
      }

      (void)wrapper_op_desc->AddInputDesc(op_desc->GetInputDesc(0));
      graph_item->input_index_mapping_.emplace_back(data_index);
    } else if (op_type == NETOUTPUT) {
      int output_index = 0;
      for (const auto &output_desc : op_desc->GetAllInputsDescPtr()) {
        int32_t data_index = output_index++;
        if (!AttrUtils::GetInt(output_desc, ATTR_NAME_PARENT_NODE_INDEX, data_index)) {
          GELOGI("[%s] Failed to get attr [%s]", node->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        }

        GE_CHK_GRAPH_STATUS_RET(wrapper_op_desc->AddOutputDesc(*output_desc),
                                "[%s] Failed to add output desc. output index = %d", graph.GetName().c_str(),
                                output_index);

        graph_item->output_index_mapping_.emplace_back(data_index);
      }
    }
  }

  auto temp_graph = MakeShared<ComputeGraph>("temp");
  GE_CHECK_NOTNULL(temp_graph);
  auto wrapper_node = temp_graph->AddNode(wrapper_op_desc);
  GeModelPtr ge_model = subgraph_models_[subgraph_name];
  GE_CHECK_NOTNULL(ge_model);
  hybrid_model_.known_shape_sub_models_.emplace(wrapper_node, ge_model);

  NodeItem *node_item = nullptr;
  GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(wrapper_node, &node_item));
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->outputs.resize(node_item->num_outputs);
  graph_item->node_items_.emplace_back(node_item);
  graph_item->output_node_ = node_item;
  graph_item->total_inputs_ = node_item->num_inputs;
  graph_item->total_outputs_ = node_item->num_outputs;

  GELOGD("NodeItem create for known shape subgraph [%s], NodeItem = %s", graph.GetName().c_str(),
         node_item->DebugString().c_str());

  GELOGD("Done parse known shape subgraph successfully. graph = [%s]", graph.GetName().c_str());
  graph_item->SetName(graph.GetName());
  GELOGD("Done loading known shape subgraph: [%s]", graph_item->GetName().c_str());
  hybrid_model_.subgraph_items_.emplace(graph.GetName(), std::move(graph_item));
  return SUCCESS;
}

Status HybridModelBuilder::LoadDynamicSubgraph(ComputeGraph &graph, bool is_root_graph) {
  GELOGD("Start to load subgraph [%s]", graph.GetName().c_str());
  // for known partitioned call, load all nodes
  auto graph_item = std::unique_ptr<GraphItem>(new (std::nothrow) GraphItem());
  GE_CHECK_NOTNULL(graph_item);

  graph_item->is_dynamic_ = true;
  graph_item->node_items_.reserve(graph.GetDirectNodesSize());
  int input_start = 0;
  int output_start = 0;
  std::vector<NodeItem *> data_nodes;
  for (auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    const auto &op_type = node->GetType();

    NodeItem *node_item = nullptr;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(node, &node_item));
    GE_CHK_STATUS_RET_NOLOG(BuildNodeItem(node, *node_item));
    GE_CHK_STATUS_RET_NOLOG(UpdateAnchorStatus(node));  // needed by FE generate task

    node_item->input_start = input_start;
    node_item->output_start = output_start;
    input_start += node_item->num_inputs;
    output_start += node_item->num_outputs;

    if (op_type == DATA_TYPE || op_type == AIPP_DATA_TYPE) {
      data_nodes.emplace_back(node_item);
    } else if (op_type == NETOUTPUT) {
      graph_item->output_node_ = node_item;
      GE_CHK_STATUS_RET_NOLOG(BuildOutputMapping(*graph_item, *node_item, is_root_graph));
    }

    graph_item->node_items_.emplace_back(node_item);
    // parse var outputs
    GE_CHK_STATUS_RET_NOLOG(ParseVarOutputs(*node_item));
    GELOGD("NodeItem created: %s", node_item->DebugString().c_str());
  }

  graph_item->total_inputs_ = input_start;
  graph_item->total_outputs_ = output_start;
  GE_CHK_STATUS_RET_NOLOG(BuildInputMapping(*graph_item, data_nodes, is_root_graph));
  if (is_root_graph) {
    graph_item->SetName("Root-Graph");
    GELOGD("Done loading dynamic subgraph: [%s]", graph_item->GetName().c_str());
    hybrid_model_.root_graph_item_ = std::move(graph_item);
  } else {
    graph_item->SetName(graph.GetName());
    GELOGD("Done loading dynamic subgraph: [%s]", graph_item->GetName().c_str());
    hybrid_model_.subgraph_items_.emplace(graph.GetName(), std::move(graph_item));
  }

  return SUCCESS;
}

Status HybridModelBuilder::ParseVarOutputs(NodeItem &node_item) {
  for (int i = 0; i < node_item.num_outputs; ++i) {
    auto output_tensor_desc = node_item.op_desc->GetOutputDesc(i);
    std::string var_name;
    (void)AttrUtils::GetStr(output_tensor_desc, ASSIGN_VAR_NAME, var_name);
    if (!var_name.empty()) {
      auto var_node = hybrid_model_.GetVariableNode(var_name);
      GE_CHECK_NOTNULL(var_node);
      node_item.ref_outputs.emplace(i, var_node);
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::BuildInputMapping(GraphItem &graph_item, vector<NodeItem *> &data_nodes,
                                             bool is_root_graph) {
  uint32_t data_op_index = 0;
  for (auto &node_item : data_nodes) {
    auto node = node_item->node;
    int data_index = data_op_index;
    if (is_root_graph) {
      if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, data_index)) {
        GELOGI("ge_train: get new index %u, old %u", data_index, data_op_index);
      }
      data_op_index++;
    } else {
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, data_index)) {
        GELOGE(FAILED, "[%s] Failed to get attr [%s]", node->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        return FAILED;
      }
    }

    if (graph_item.input_nodes_.size() <= static_cast<size_t>(data_index)) {
      graph_item.input_nodes_.resize(data_index + 1);
    }

    graph_item.input_nodes_[data_index] = node_item;
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
