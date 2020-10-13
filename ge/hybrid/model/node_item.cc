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

#include "node_item.h"
#include <sstream>
#include "common/debug/log.h"
#include "graph/common/omg_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
namespace {
const char * const kAttrNameOriginalFusionGraph = "_original_fusion_graph";
const char * const kNodeTypeRetVal = "_RetVal";

Status ParseInputMapping(Node &node, OpDesc &op_desc, FusedSubgraph &fused_subgraph) {
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(FAILED,
           "[%s] Failed to get attr [%s]",
           op_desc.GetName().c_str(),
           ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return FAILED;
  }

  for (auto &node_and_anchor : node.GetOutDataNodesAndAnchors()) {
    auto dst_op_desc = node_and_anchor.first->GetOpDesc();
    GE_CHECK_NOTNULL(dst_op_desc);
    auto in_idx = node_and_anchor.second->GetIdx();
    auto tensor_desc = dst_op_desc->MutableInputDesc(in_idx);
    fused_subgraph.input_mapping[parent_index].emplace_back(tensor_desc);
    GELOGD("Input[%u] mapped to [%s:%u]", parent_index, dst_op_desc->GetName().c_str(), in_idx);
  }

  return SUCCESS;
}

Status ParseOutputMapping(OpDescPtr op_desc, FusedSubgraph &fused_subgraph) {
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(FAILED,
           "[%s] Failed to get attr [%s]",
           op_desc->GetName().c_str(),
           ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return FAILED;
  }

  fused_subgraph.output_mapping.emplace(parent_index, op_desc);
  return SUCCESS;
}

Status ParseFusedSubgraph(NodeItem &node_item) {
  if (!node_item.op_desc->HasAttr(kAttrNameOriginalFusionGraph)) {
    return SUCCESS;
  }

  GELOGI("[%s] Start to parse fused subgraph.", node_item.node_name.c_str());
  auto fused_subgraph = std::unique_ptr<FusedSubgraph>(new (std::nothrow)FusedSubgraph());
  GE_CHECK_NOTNULL(fused_subgraph);

  ComputeGraphPtr fused_graph;
  (void) AttrUtils::GetGraph(*node_item.op_desc, kAttrNameOriginalFusionGraph, fused_graph);
  GE_CHECK_NOTNULL(fused_graph);

  fused_graph->SetGraphUnknownFlag(true);
  fused_subgraph->graph = fused_graph;
  GE_CHK_GRAPH_STATUS_RET(fused_graph->TopologicalSorting());

  for (auto &node : fused_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string node_type;
    GE_CHK_STATUS_RET(GetOriginalType(node, node_type));
    if (node_type == DATA) {
      GE_CHK_GRAPH_STATUS_RET(ParseInputMapping(*node, *op_desc, *fused_subgraph));
    } else if (node_type == kNodeTypeRetVal) {
      GE_CHK_GRAPH_STATUS_RET(ParseOutputMapping(op_desc, *fused_subgraph));
    } else {
      fused_subgraph->nodes.emplace_back(node);
    }
  }

  node_item.fused_subgraph = std::move(fused_subgraph);
  GELOGI("[%s] Done parsing fused subgraph successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}
}  // namespace
NodeItem::NodeItem(NodePtr node): node(std::move(node)) {
  this->op_desc = this->node->GetOpDesc().get();
  this->node_id = this->op_desc->GetId();
  this->num_inputs = this->op_desc->GetInputsSize();
  this->num_outputs = this->op_desc->GetOutputsSize();
  this->node_name = this->node->GetName();
  this->node_type = this->node->GetType();
}

Status NodeItem::Init() {
  int32_t unknown_shape_type_val = 0;
  (void) AttrUtils::GetInt(op_desc, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  shape_inference_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);

  (void) AttrUtils::GetBool(op_desc, ATTR_NAME_FORCE_UNKNOWN_SHAPE, is_dynamic);
  GELOGD("node name = %s, is_dynamic = %d.", this->node_name.c_str(), is_dynamic);
  if (!is_dynamic) {
    GE_CHK_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node, is_dynamic),
                      "[%s] Failed to get shape status.",
                      node->GetName().c_str());
  }

  GE_CHK_STATUS_RET(ParseFusedSubgraph(*this), "[%s] Failed to parse fused subgraph", node_name.c_str());
  if (is_dynamic) {
    for (int i = 0; i < num_inputs; ++i) {
      const auto &input_desc = op_desc->MutableInputDesc(i);
      GE_CHECK_NOTNULL(input_desc);
      if (input_desc->MutableShape().IsUnknownShape()) {
        is_input_shape_static.push_back(false);
      } else {
        num_static_input_shapes++;
        is_input_shape_static.push_back(true);
        GELOGD("[%s] The shape of input[%d] is static. shape = [%s]",
               NodeName().c_str(), i, input_desc->MutableShape().ToString().c_str());
      }
    }

    for (int i = 0; i < num_outputs; ++i) {
      const auto &output_desc = op_desc->MutableOutputDesc(i);
      GE_CHECK_NOTNULL(output_desc);
      if (output_desc->MutableShape().IsUnknownShape()) {
        is_output_shape_static = false;
        break;
      }
    }
  }

  return SUCCESS;
}

bool NodeItem::IsControlOp() const {
  auto op_type = op_desc->GetType();
  return op_type == IF || op_type == CASE || op_type == WHILE || op_type == FOR;
}

std::string NodeItem::DebugString() const {
  std::stringstream ss;
  ss << "Node: ";
  ss << "id = " << node_id;
  ss << ", name = [" << node->GetName();
  ss << "], type = " << node->GetType();
  ss << ", is_dynamic = " << (is_dynamic ? "True" : "False");
  ss << ", is_output_static = " << (is_output_shape_static ? "True" : "False");
  ss << ", unknown_shape_op_type = " << shape_inference_type;
  ss << ", input_start = " << input_start;
  ss << ", num_inputs = " << num_inputs;
  ss << ", output_start = " << output_start;
  ss << ", num_outputs = " << num_outputs;
  ss << ", dependent_nodes = [";
  for (const auto &dep_node : dependents_for_shape_inference) {
    ss << dep_node->GetName() << ", ";
  }
  ss << "]";
  int index = 0;
  for (auto &items : outputs) {
    ss << ", output[" << index++ << "]: ";
    for (auto &item : items) {
      ss << "(" << item.second->NodeName() << ":" <<item.first << "), ";
    }
  }

  return ss.str();
}

void NodeItem::SetToDynamic() {
  num_static_input_shapes = 0;
  is_dynamic = true;
  for (size_t i = 0; i < is_input_shape_static.size(); ++i) {
    is_input_shape_static[i] = false;
  }
  if (kernel_task != nullptr && !kernel_task->IsSupportDynamicShape()) {
    GELOGD("[%s] Dynamic shape is not supported, clear node task.", node_name.c_str());
    kernel_task = nullptr;
  }
}
}  // namespace hybrid
}  // namespace ge
