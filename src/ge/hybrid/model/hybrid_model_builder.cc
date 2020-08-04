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
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "framework/common/debug/log.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
namespace {
const uint32_t kSubgraphIndex = 0U;
const uint32_t kVarOutputIndex = 0U;
const int kBytes = 8;

int64_t CalcVarSizeInBytes(const GeTensorDesc &desc) {
  int64_t var_size = GetSizeByDataType(desc.GetDataType());
  if (var_size <= 0) {
    GELOGE(PARAM_INVALID, "Failed to calc var data size from data type %s",
           TypeUtils::DataTypeToSerialString(desc.GetDataType()).c_str());
    return -1;
  }
  auto shape = desc.GetShape();
  auto dim_num = shape.GetDimNum();
  for (size_t dim_index = 0; dim_index < dim_num; ++dim_index) {
    var_size *= shape.GetDim(dim_index);
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
  graph_name_ = ge_root_model_->GetRootGraph()->GetName();
  GELOGI("[%s] Start to build hybrid model.", GetGraphName());
  GE_CHK_STATUS_RET(InitRuntimeParams(), "[%s] Failed to InitRuntimeParams", GetGraphName());
  GE_CHK_STATUS_RET(NodeExecutorManager::GetInstance().EnsureInitialized(), "Failed to initialize executors");
  GE_CHK_STATUS_RET(IndexSpecialNodes(), "[%s] Failed to index nodes", GetGraphName());
  GE_CHK_STATUS_RET(IndexTaskDefs(), "[%s] Failed to index task defs", GetGraphName());
  GE_CHK_STATUS_RET(LoadGraph(), "[%s] Failed to load graph", GetGraphName());
  GE_CHK_STATUS_RET(TransAllVarData(), "[%s] Failed to trans all var data", GetGraphName());
  GE_CHK_STATUS_RET(CopyVarData(), "[%s] Failed to copy var data", GetGraphName());
  GE_CHK_STATUS_RET(InitModelMem(), "[%s] Failed to init memory", GetGraphName());
  GE_CHK_STATUS_RET(InitWeights(), "[%s] Failed to init weights", GetGraphName());
  GE_CHK_STATUS_RET(InitConstantOps(), "[%s] Failed to init constant op", GetGraphName());
  GE_CHK_STATUS_RET(InitVariableTensors(), "[%s] Failed to init variables", GetGraphName());
  GE_CHK_STATUS_RET(ResolveRootNodes(), "[%s] Failed to resolve root nodes", GetGraphName());
  GE_CHK_STATUS_RET(LoadTasks(), "[%s] Failed to load tasks", GetGraphName());
  GELOGI("[%s] Done building hybrid model successfully.", GetGraphName());
  return SUCCESS;
}

Status HybridModelBuilder::ValidateParams() {
  GE_CHECK_NOTNULL(ge_root_model_);
  GE_CHECK_NOTNULL(ge_root_model_->GetRootGraph());
  return SUCCESS;
}

Status HybridModelBuilder::ResolveRootNodes() {
  for (auto &node : hybrid_model_.node_items_) {
    if (node->node->GetInDataNodes().empty()) {
      hybrid_model_.root_nodes_.emplace_back(node.get());
      GELOGI("[%s] Root node added. node name = %s", GetGraphName(), node->NodeName().c_str());
    }
  }

  if (hybrid_model_.root_nodes_.empty()) {
    GELOGE(PARAM_INVALID, "[%s] Root nodes is empty.", GetGraphName());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status HybridModelBuilder::BuildNoteItem(const NodePtr &node, NodeItem &node_item) {
  GE_CHK_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node, node_item.is_dynamic),
                    "[%s] Failed to get shape status.", node->GetName().c_str());

  auto op_desc = node->GetOpDesc();
  vector<string> dependencies = node->GetOpDesc()->GetOpInferDepends();
  GE_CHK_STATUS_RET(ParseDependentInputNodes(node_item, dependencies), "[%s] Failed to parse node dependencies.",
                    node_item.NodeName().c_str());

  auto it = node_ref_inputs_.find(node);
  if (it != node_ref_inputs_.end()) {
    for (auto &idx_and_node : it->second) {
      // var and constant only have one output
      node_item.const_input_shapes[idx_and_node.first] =
        idx_and_node.second->GetOpDesc()->MutableOutputDesc(kVarOutputIndex);
    }
  }

  node_item.outputs.resize(node_item.num_outputs);
  for (int i = 0; i < node_item.num_outputs; ++i) {
    auto out_data_anchor = node->GetOutDataAnchor(i);
    if (out_data_anchor == nullptr) {
      GELOGE(INTERNAL_ERROR, "out anchor[%zu] of node %s is nullptr", i, node->GetName().c_str());
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

  return SUCCESS;
}

Status HybridModelBuilder::GetOrCreateNodeItem(const NodePtr &node, NodeItem **node_item) {
  auto &node_items = hybrid_model_.node_items_;
  auto node_id = node->GetOpDesc()->GetId();
  if (node_id < 0 || static_cast<size_t>(node_id) > node_items.size()) {
    GELOGE(INTERNAL_ERROR, "[%s] Index out of range. node_id = %ld, num_nodes = %zu", node->GetName().c_str(), node_id,
           node_items.size());
    return INTERNAL_ERROR;
  }

  auto &node_ptr = node_items[node_id];
  if (node_ptr != nullptr) {
    *node_item = node_ptr.get();
    return SUCCESS;
  }

  auto new_node = std::unique_ptr<NodeItem>(new (std::nothrow) NodeItem(node));
  GE_CHECK_NOTNULL(new_node);
  GE_CHECK_NOTNULL(new_node->op_desc);
  GE_CHK_STATUS_RET_NOLOG(NodeExecutorManager::GetInstance().GetExecutor(*node, &new_node->node_executor));

  // we do not need L2 Buffer
  const char *const kIsFirstNode = "is_first_node";
  const char *const kIsLastNode = "is_last_node";
  (void)AttrUtils::SetBool(new_node->op_desc, kIsFirstNode, false);
  (void)AttrUtils::SetBool(new_node->op_desc, kIsLastNode, false);

  int32_t unknown_shape_type_val = 0;
  (void)AttrUtils::GetInt(new_node->op_desc, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  new_node->shape_inference_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
  if (new_node->shape_inference_type == DEPEND_SHAPE_RANGE || new_node->shape_inference_type == DEPEND_COMPUTE) {
    new_node->has_observer = true;
  }

  *node_item = new_node.get();
  node_items[node_id] = std::move(new_node);
  return SUCCESS;
}

Status HybridModelBuilder::ParseDependentInputNodes(NodeItem &node_item, const std::vector<string> &dependencies) {
  std::set<NodePtr> dependent_input_nodes;
  auto &ge_node = node_item.node;
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
    node_item.dependent_node_list.emplace_back(dep_node);
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
  for (const auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != DATA_TYPE) {
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
        GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(out_data_anchor, peer_in_data_anchor));
        GE_CHK_STATUS_RET_NOLOG(DoLinkDataAnchors(src_out_anchor, peer_in_data_anchor));
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::MergeNetOutputNode(ComputeGraph &graph) {
  const auto &parent_node = graph.GetParentNode();
  const NodePtr &net_output_node = graph.FindFirstNodeMatchType(NETOUTPUT);
  GE_CHECK_NOTNULL(net_output_node);
  const auto &net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

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

  return SUCCESS;
}

Status HybridModelBuilder::MergeSubgraphs(ComputeGraph &root_graph, ComputeGraphPtr &merged_graph) {
  merged_graph = MakeShared<ComputeGraph>("MergedGraph");
  for (const auto &node : root_graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    const auto &op_type = node->GetType();
    if (op_type == DATA || op_type == AIPP_DATA_TYPE || op_type == NETOUTPUT) {
      merged_graph->AddNode(node);
      GELOGD("[%s] Node added to merged graph.", op_desc->GetName().c_str());
      continue;
    }

    if (op_type != PARTITIONEDCALL) {
      GELOGE(INTERNAL_ERROR, "[%s] Unexpected node in root graph. type = %s", op_desc->GetName().c_str(),
             op_type.c_str());
      return INTERNAL_ERROR;
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
    GE_CHK_STATUS_RET(MergeInputNodes(*subgraph), "Failed to merge data nodes for subgraph: %s",
                      subgraph->GetName().c_str());
    GE_CHK_STATUS_RET(MergeNetOutputNode(*subgraph), "Failed to merge net output nodes for subgraph: %s",
                      subgraph->GetName().c_str());
    GELOGD("Merging subgraph %s successfully.", subgraph->GetName().c_str());
    for (auto &sub_node : subgraph->GetAllNodes()) {
      auto sub_op_type = sub_node->GetType();
      if (sub_op_type == DATA_TYPE || sub_op_type == NETOUTPUT) {
        continue;
      }

      if (sub_op_type == CONSTANT || sub_op_type == CONSTANTOP || sub_op_type == VARIABLE) {
        GELOGE(INTERNAL_ERROR, "Unexpected node in unknown subgraph. type = %s, node = %s::%s", sub_op_type.c_str(),
               subgraph->GetName().c_str(), sub_node->GetName().c_str());
        return INTERNAL_ERROR;
      }

      merged_graph->AddNode(sub_node);
      GELOGD("%s::%s added to merged graph.", subgraph->GetName().c_str(), sub_node->GetName().c_str());
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::ParseNetOutput(const NodeItem &node_item) {
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
    hybrid_model_.output_offsets_.emplace_back(output_offset);
  }

  for (int i = 0; i < node_item.num_inputs; ++i) {
    hybrid_model_.net_output_input_offsets_.emplace_back(node_item.input_start + i);
  }

  return SUCCESS;
}

Status HybridModelBuilder::LoadGraph() {
  auto root_graph = ge_root_model_->GetRootGraph();
  GELOGI("Before merge subgraphs DirectNodesSize = %zu, GetAllNodesSize = %zu", root_graph->GetDirectNodesSize(),
         root_graph->GetAllNodesSize());
  ComputeGraphPtr merged_graph;
  GE_CHK_STATUS_RET_NOLOG(MergeSubgraphs(*root_graph, merged_graph));
  GELOGI("After merge subgraphs DirectNodesSize = %zu, GetAllNodesSize = %zu", merged_graph->GetDirectNodesSize(),
         merged_graph->GetAllNodesSize());

  merged_graph->SetGraphID(runtime_param_.graph_id);
  GE_DUMP(merged_graph, "hybrid_merged_graph");
  int input_start = 0;
  int output_start = 0;
  uint32_t data_op_index = 0;
  hybrid_model_.node_items_.resize(merged_graph->GetDirectNodesSize());

  int64_t node_index = 0;
  for (auto &node : merged_graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    op_desc->SetId(node_index++);
  }

  for (const auto &node : merged_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    const auto &op_type = node->GetType();

    NodeItem *node_item = nullptr;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(node, &node_item));
    GE_CHK_STATUS_RET_NOLOG(BuildNoteItem(node, *node_item));
    GE_CHK_STATUS_RET_NOLOG(UpdateAnchorStatus(node));  // needed by FE generate task

    node_item->input_start = input_start;
    node_item->output_start = output_start;
    input_start += node_item->num_inputs;
    output_start += node_item->num_outputs;

    if (op_type == DATA_TYPE || op_type == AIPP_DATA_TYPE) {
      auto data_index = data_op_index;
      if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, data_index)) {
        GELOGI("ge_train: get new index %u, old %u", data_index, data_op_index);
      }
      hybrid_model_.input_nodes_.emplace(data_index, node_item);
      data_op_index++;
    } else if (op_type == NETOUTPUT) {
      hybrid_model_.net_output_node_ = node_item;
      GE_CHK_STATUS_RET_NOLOG(ParseNetOutput(*node_item));
    } else if (op_type == PARTITIONEDCALL) {  // known graph
      GE_CHK_STATUS_RET_NOLOG(ParsePartitionedCall(*node_item));
    }

    GELOGI("NodeItem created: %s", node_item->DebugString().c_str());
  }

  for (auto &it : hybrid_model_.input_nodes_) {
    auto input_index = it.first;
    auto input_node = it.second;

    if (input_node->outputs.empty()) {
      GELOGE(INTERNAL_ERROR, "data output anchor is empty");
      return INTERNAL_ERROR;
    }

    for (auto &out : input_node->outputs) {
      std::vector<int> offsets;
      for (auto &dst_anchor_and_node : out) {
        auto dst_node_item = dst_anchor_and_node.second;
        offsets.emplace_back(dst_node_item->input_start + dst_anchor_and_node.first);
      }

      hybrid_model_.input_offsets_.emplace(input_index, std::move(offsets));
    }
  }

  hybrid_model_.total_inputs_ = input_start;
  hybrid_model_.total_outputs_ = output_start;
  GELOGI("HybridGraph::LoadGraph OUT");
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
  if (var_size < 0) {
    GELOGE(INTERNAL_ERROR, "[%s] Invalid var size: %ld", var_name.c_str(), var_size);
    return INTERNAL_ERROR;
  }

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

Status HybridModelBuilder::InitConstantOps() {
  for (auto &it : hybrid_model_.constant_op_nodes_) {
    string var_name = it.first;
    NodePtr &var_node = it.second;
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

      GELOGI("[IMAS]InitConstant memcpy graph_%u type[V] name[%s] output[%d] memaddr[%p] mem_size[%u] datasize[%zu]",
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
  for (auto &node_item : hybrid_model_.node_items_) {
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

    bool is_unknown_shape = false;
    GE_CHK_GRAPH_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*sub_graph->GetParentNode(), is_unknown_shape),
                            "Failed to invoke GetNodeUnknownShapeStatus.");
    if (!is_unknown_shape) {
      GELOGD("Set ge_model for subgraph: %s", sub_graph->GetName().c_str());
      hybrid_model_.known_shape_sub_graphs_.emplace(sub_graph->GetParentNode(), ge_model);
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
      GELOGW("SubGraph: %s input tensor %zu attr %s not found.", src_graph->GetName().c_str(), i,
             ATTR_NAME_PARENT_NODE_INDEX.c_str());
      continue;
    }

    GELOGD("NetOutput's input[%zu], parent_node_index = %u", i, p_index);
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
  runtime_param_.session_id = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(first_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, value);
  runtime_param_.logic_var_base = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(first_model, ATTR_MODEL_VAR_SIZE, value);
  runtime_param_.var_size = ret ? (uint64_t)value : 0;
  runtime_param_.graph_id = ge_root_model_->GetRootGraph()->GetGraphID();
  GELOGI("InitRuntimeParams(), session_id:%u, var_size:%lu. graph_id = %u", runtime_param_.session_id,
         runtime_param_.var_size, runtime_param_.graph_id);

  var_manager_ = VarManager::Instance(runtime_param_.session_id);
  GE_CHECK_NOTNULL(var_manager_);
  return SUCCESS;
}

Status HybridModelBuilder::ParsePartitionedCall(NodeItem &node_item) {
  GELOGD("Start to parse outputs of node: %s", node_item.NodeName().c_str());
  auto subgraph = NodeUtils::GetSubgraph(*node_item.node, kSubgraphIndex);
  GE_CHECK_NOTNULL(subgraph);
  auto net_output_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  GE_CHECK_NOTNULL(net_output_node);
  auto net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

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
}  // namespace hybrid
}  // namespace ge
