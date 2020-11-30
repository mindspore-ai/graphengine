/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
    vector<GeTensorPtr> outputs;
    Status ret = GetOutputsOfCurrNode(node, outputs);
    if (ret != SUCCESS) {
      // If replace failed, it should not break whole process, so still return success
      GELOGW("Failed to get outputs of node %s.", node->GetName().c_str());
    }
    else {
      ret = Folding(node, outputs);
      if (ret != SUCCESS) {
        // If replace failed, it should not break whole process, so still return success
        GELOGW("Failed to repalce node %s with empty const.", node->GetName().c_str());
      }
    }
  }
  GELOGD("ReplaceWithEmptyConstPass end.");
  return SUCCESS;
}
Status GetOutputsOfCurrNode(const NodePtr &node_to_repalce, vector<GeTensorPtr> &outputs) {
  for (const auto &out_anchor : node_to_replace->GetAllOutDataAnchors()) {
    auto out_desc = op_desc->GetOutputDesc(out_anchor->GetIdx());
    GeTensorPtr empty_tensor = MakeShared<ge::GeTensor>(out_desc);
    GE_CHECK_NOTNULL(empty_tensor);
    outputs.emplace_back(empty_tensor);
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
}  // namespace ge
