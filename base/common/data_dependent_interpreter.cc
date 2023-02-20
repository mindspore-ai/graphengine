/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "common/data_dependent_interpreter.h"

#include "common/checker.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/node.h"
#include "graph/debug/ge_attr_define.h"

#include "common/omg_util.h"
#include "graph/utils/graph_utils.h"

namespace gert {
namespace {
const ge::char_t *const kUbGraph = "_original_fusion_graph";
bool IsUbFusedNode(const ge::NodePtr &node) {
  return ge::AttrUtils::HasAttr(node->GetOpDesc(), kUbGraph);
}
ge::graphStatus IsDataDependentByAttr(const ge::NodePtr &node, int32_t input_index, bool &is_data_dependent) {
  auto data_dependent_inputs = node->GetOpDesc()->GetOpInferDepends();
  if (data_dependent_inputs.empty()) {
    is_data_dependent = false;
    return ge::GRAPH_SUCCESS;
  }
  auto input_name = node->GetOpDesc()->GetInputNameByIndex(static_cast<uint32_t>(input_index));
  is_data_dependent = std::find(data_dependent_inputs.cbegin(), data_dependent_inputs.cend(), input_name) !=
                      data_dependent_inputs.cend();
  return ge::GRAPH_SUCCESS;
}
ge::NodePtr FindSubgraphDataNode(const ge::ComputeGraphPtr &graph, int32_t parent_node_index) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != ge::DATA) {
      continue;
    }
    int32_t parent_index = 0;
    if (!ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(ge::INTERNAL_ERROR, "[Get][Attr] failed, node:[%s(%s)]  attr:[%s]", node->GetName().c_str(),
             node->GetType().c_str(), ge::ATTR_NAME_PARENT_NODE_INDEX.c_str());
      REPORT_CALL_ERROR("E19999", "invoke GetInt failed, node:[%s(%s)]  attr:[%s]", node->GetName().c_str(),
                        node->GetType().c_str(), ge::ATTR_NAME_PARENT_NODE_INDEX.c_str());
      return nullptr;
    }
    if (parent_index == parent_node_index) {
      return node;
    }
  }
  return nullptr;
}
}  // namespace

DataDependentInterpreter::DataDependentInterpreter(const ge::NodePtr &node,
                                                   const gert::OpImplSpaceRegistryPtr &space_registry) :
  node_(node), space_registry_(space_registry) {}

ge::graphStatus DataDependentInterpreter::IsDataDependentByImplOp(const ge::NodePtr &node,
                                                                  int32_t input_index, bool &is_data_dependent) const {
  std::string type;
  GE_ASSERT_SUCCESS(ge::GetOriginalType(node, type), "Failed to get original type from %s(%s).",
                    node->GetName().c_str(), node->GetType().c_str());
  if (space_registry_ == nullptr) {
    GELOGW("Attention: default registry is not existed. Tiling will be executed failed");
    is_data_dependent = false;
    return ge::GRAPH_SUCCESS;
  }
  auto op_impl = space_registry_->GetOpImpl(type);
  if (op_impl == nullptr) {
    GELOGW("The node %s type %s does not registered by `IMPL_OP`", node->GetName().c_str(), type.c_str());
    is_data_dependent = false;
    // 这里产生了变更，原有实现中，如果impl找不到，并且1.0标记了任意一个输入为数据依赖，那么整个节点所有输入都会被认为是数据依赖。
    // 变更后，如果impl找不到，那么仅会返回1.0标记的输入为数据依赖。这个变更影响应该不大，验证过后，本注释可以被删除
    return ge::GRAPH_SUCCESS;
  }
  if (!op_impl->HasDataDependency()) {
    is_data_dependent = false;
    return ge::GRAPH_SUCCESS;
  }
  size_t ir_index;
  auto ret = ge::OpDescUtils::GetInputIrIndexByInstanceIndex(node->GetOpDesc(), input_index, ir_index);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Failed to get ir index by input_index[%d] for node %s(%s).", input_index,
           node->GetName().c_str(), node->GetType().c_str());
    return ge::FAILED;
  }
  is_data_dependent = op_impl->IsInputDataDependency(ir_index);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus DataDependentInterpreter::IsDataDependent(int32_t index, bool &is_data_dependent) const {
  bool by_ir;
  GE_ASSERT_SUCCESS(IsDataDependentByIr(index, by_ir));

  if (!IsUbFusedNode(node_)) {
    is_data_dependent = by_ir;
    return ge::GRAPH_SUCCESS;
  }

  bool by_ub;
  GE_ASSERT_SUCCESS(IsDataDependentByUbGraph(index, by_ub));
  is_data_dependent = GetByIrAndUb(by_ir, by_ub, index);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus DataDependentInterpreter::IsDataDependentByIr(int32_t index, bool &is_data_dependent) const {
  bool by_1_0 = false;
  bool by_2_0 = false;
  GE_ASSERT_SUCCESS(IsDataDependentByImplOp(node_, index, by_2_0));
  GE_ASSERT_SUCCESS(IsDataDependentByAttr(node_, index, by_1_0));

  is_data_dependent = GetByIr(by_1_0, by_2_0, index);
  return ge::GRAPH_SUCCESS;
}
bool DataDependentInterpreter::GetByIr(bool by_1_0, bool by_2_0, int32_t index_for_log) const {
  if (by_1_0 == by_2_0) {
    return by_2_0;
  }
  if (by_1_0) {  // by_2_0 is false
    GELOGW(
        "The node %s type %s input index %d is interpreted data-dependent, because there is data dependent attr on the "
        "node. But the IMPL_OP does not registered as data-dependent",
        node_->GetName().c_str(), node_->GetType().c_str(), index_for_log);
  }
  return true;
}
ge::graphStatus DataDependentInterpreter::IsDataDependentByUbGraph(int32_t index, bool &is_data_dependent) const {
  auto ub_graph = GetUbGraph();
  GE_ASSERT_NOTNULL(ub_graph);

  auto data_node = FindSubgraphDataNode(ub_graph, index);
  GE_ASSERT_NOTNULL(data_node, "Failed to find the data node from ub graph by index %d from node %s type %s.",
                    index, node_->GetName().c_str(), node_->GetType().c_str());

  is_data_dependent = false;
  for (const auto &node_and_anchor : data_node->GetOutDataNodesAndAnchors()) {
    bool node_data_dependent;
    GE_ASSERT_SUCCESS(DataDependentInterpreter(node_and_anchor.first, space_registry_)
                          .IsDataDependentByIr(node_and_anchor.second->GetIdx(), node_data_dependent));
    if (node_data_dependent) {
      is_data_dependent = true;
      break;
    }
  }

  return ge::GRAPH_SUCCESS;
}
bool DataDependentInterpreter::GetByIrAndUb(bool by_ir, bool by_ub, int32_t index_for_log) const {
  if (by_ir == by_ub) {
    return by_ir;
  }

  if (by_ir) {  // by_ub is false
    GELOGW(
        "The UB-fused node %s type %s input index %d is interpreted data-dependent. The data-dependent flag is marked "
        "by IR, but not the UB graph",
        node_->GetName().c_str(), node_->GetType().c_str(), index_for_log);
  }
  return true;
}
ge::ComputeGraphPtr DataDependentInterpreter::GetUbGraph() const {
  if (ub_graph_cache_ == nullptr) {
    GE_ASSERT_TRUE(ge::AttrUtils::GetGraph(node_->GetOpDesc(), kUbGraph, ub_graph_cache_));
  }
  return ub_graph_cache_;
}
}  // namespace gert