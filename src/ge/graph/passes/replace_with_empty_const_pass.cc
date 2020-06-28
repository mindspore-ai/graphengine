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

#include "graph/passes/replace_with_empty_const_pass.h"
#include <sstream>
#include <string>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/graph_utils.h"

namespace ge {
Status ReplaceWithEmptyConstPass::Run(NodePtr &node) {
  GELOGD("ReplaceWithEmptyConstPass in.");
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter is null.");
    return PARAM_INVALID;
  }
  if (node->GetOpDesc() == nullptr) {
    GELOGE(PARAM_INVALID, "Param [opDesc] must not be null.");
    return PARAM_INVALID;
  }
  if (node->GetType() == CONSTANT || node->GetType() == CONSTANTOP) {
    GELOGI("Node %s is const. Ignore current pass.", node->GetName().c_str());
    return SUCCESS;
  }
  // Node like no op, it has no output
  if (node->GetOpDesc()->GetAllOutputsDescPtr().empty()) {
    GELOGI("Node %s has no output desc. Ignore current pass.", node->GetName().c_str());
    return SUCCESS;
  }
  // If outputs of current node are all empty, replace it with empty const
  bool is_all_output_empty = true;
  for (const auto &output_desc_ptr : node->GetOpDesc()->GetAllOutputsDescPtr()) {
    if (output_desc_ptr == nullptr) {
      GELOGI("Node %s Got empty output_desc_ptr, ignore current pass.", node->GetName().c_str());
      return SUCCESS;
    }
    if (!IsEmptyTenor(output_desc_ptr->GetShape())) {
      is_all_output_empty = false;
      break;
    }
  }
  if (is_all_output_empty) {
    GELOGI("Node %s has empty tensor output. It will be replaced by empty const.", node->GetName().c_str());
    // Replace op which all output is empty with empty const
    Status ret = ReplaceWithEmptyConst(node);
    if (ret != SUCCESS) {
      // If replace failed, it should not break whole process, so still return success
      GELOGW("Failed to repalce node %s with empty const.", node->GetName().c_str());
    }
  }
  GELOGD("ReplaceWithEmptyConstPass end.");
  return SUCCESS;
}

Status ReplaceWithEmptyConstPass::ReplaceWithEmptyConst(NodePtr &node_to_replace) {
  std::map<string, vector<int>> shape_out_idx_map;
  auto op_desc = node_to_replace->GetOpDesc();
  // Collect out_idx follow different out shape
  for (const auto &out_anchor : node_to_replace->GetAllOutDataAnchors()) {
    auto out_desc = op_desc->GetOutputDesc(out_anchor->GetIdx());
    shape_out_idx_map[GetDimStr(out_desc.GetShape())].emplace_back(out_anchor->GetIdx());
  }

  for (const auto &shape_2_out_idx : shape_out_idx_map) {
    // Create empty const
    // The out_desc in one group should be same shape, so here only get first out_desc. its valid index.
    auto out_desc = op_desc->GetOutputDesc(shape_2_out_idx.second[0]);
    NodePtr const_node;
    auto graph = node_to_replace->GetOwnerComputeGraph();
    Status ret = InsertEmptyConst(out_desc, const_node, graph);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Failed insert const node.");
      return FAILED;
    }

    // Repalce data anchors
    if (GraphUtils::ReplaceNodeDataAnchors(const_node, node_to_replace, {}, shape_2_out_idx.second) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "[%s] ReplaceNodeAnchors failed.", node_to_replace->GetName().c_str());
      return FAILED;
    }
    // Copy in control edge
    if (GraphUtils::CopyInCtrlEdges(node_to_replace, const_node) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "CopyInCtrlEdges from %s to %s failed.", node_to_replace->GetName().c_str(),
             const_node->GetName().c_str());
      return FAILED;
    }
    // Copy out control edge
    if (GraphUtils::CopyOutCtrlEdges(node_to_replace, const_node) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "CopyOutCtrlEdges from %s to %s failed.", node_to_replace->GetName().c_str(),
             const_node->GetName().c_str());
      return FAILED;
    }
    AddRePassNodesWithInOut(const_node);
    GELOGI("Node %s has been replaced by empty const %s.", node_to_replace->GetName().c_str(),
           const_node->GetName().c_str());
  }
  IsolateAndDeleteNode(node_to_replace, {});
  return SUCCESS;
}
Status ReplaceWithEmptyConstPass::InsertEmptyConst(const GeTensorDesc &out_desc, NodePtr &const_node,
                                                   ComputeGraphPtr &graph) {
  GeTensorPtr empty_tensor = MakeShared<ge::GeTensor>(out_desc);
  if (empty_tensor == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed create empty tensor.");
    return OUT_OF_MEMORY;
  }
  auto const_desc = OpDescUtils::CreateConstOp(empty_tensor);
  if (const_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed to get const desc from tensor");
    return OUT_OF_MEMORY;
  }

  const_node = graph->AddNode(const_desc);
  if (const_node == nullptr) {
    GELOGE(FAILED, "Failed insert const node.");
    return FAILED;
  }
  return SUCCESS;
}

bool ReplaceWithEmptyConstPass::IsEmptyTenor(const GeShape &shape) const {
  for (auto dim : shape.GetDims()) {
    if (dim == 0) {
      return true;
    }
  }
  return false;
}

string ReplaceWithEmptyConstPass::GetDimStr(const GeShape &shape) {
  std::stringstream dim_str;
  for (auto dim : shape.GetDims()) {
    dim_str << dim << '-';
  }
  return dim_str.str();
}
}  // namespace ge
