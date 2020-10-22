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

#include "graph/passes/variable_format_pass.h"
#include <map>
#include <set>
#include <string>
#include "framework/common/debug/ge_log.h"

namespace ge {
Status VariableFormatPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);

  for (auto &node : graph->GetDirectNode()) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    GE_IF_BOOL_EXEC(node->GetOpDesc()->GetType() != VARIABLE, continue);

    ge::NodePtr use_node = nullptr;
    if (GetApplyMomentumOpByVariableInput(node, use_node)) {
      GE_CHK_STATUS_RET(UpdateVariableOutFormat(node, use_node), "update variable out format failed");
      GE_CHK_STATUS_RET(UpdateApplyMomentumInputFormat(use_node), "update apply momentum input format failed");
    }
  }

  return domi::SUCCESS;
}

bool VariableFormatPass::GetApplyMomentumOpByVariableInput(const ge::NodePtr &var_node, ge::NodePtr &use_node) {
  GE_IF_BOOL_EXEC(var_node == nullptr, return false);

  std::map<std::string, std::set<int>> confirm_ops = {{"ApplyMomentum", {1}}};
  for (auto &out_anchor : var_node->GetAllOutDataAnchors()) {
    for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(ConfirmUseOpAndIndexByAnchor(in_anchor, confirm_ops, use_node), return true);
    }
  }

  return false;
}

bool VariableFormatPass::ConfirmUseOpAndIndexByAnchor(const ge::InDataAnchorPtr &in_anchor,
                                                      const map<string, std::set<int>> &confirm_ops,
                                                      ge::NodePtr &use_node) {
  GE_IF_BOOL_EXEC(in_anchor == nullptr, return false);
  ge::NodePtr dst_node = in_anchor->GetOwnerNode();
  ge::OpDescPtr dst_op_desc = dst_node->GetOpDesc();
  GE_IF_BOOL_EXEC(dst_op_desc == nullptr, return false);
  const string &dst_type = dst_op_desc->GetType();
  int input_index = in_anchor->GetIdx();

  GELOGD("ConfirmUseOpAndIndex, var name %s, dst_type = %s, input index %d", dst_node->GetName().c_str(),
         dst_type.c_str(), input_index);

  GE_IF_BOOL_EXEC(confirm_ops.count(dst_type) > 0,
                  GE_IF_BOOL_EXEC(confirm_ops.at(dst_type).count(input_index) > 0, use_node = dst_node; return true););
  return false;
}

Status VariableFormatPass::UpdateVariableOutFormat(const ge::NodePtr &var_node, ge::NodePtr &use_node) {
  GE_CHECK_NOTNULL(var_node);
  GE_CHECK_NOTNULL(use_node);
  ge::OpDescPtr op_desc_ptr = use_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc_ptr);
  GE_CHECK_NOTNULL(use_node->GetInDataAnchor(0));
  GE_CHECK_NOTNULL(use_node->GetInDataAnchor(0)->GetPeerOutAnchor());
  NodePtr in_node = use_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  if (in_node != nullptr) {
    string in_op_type = in_node->GetType();
    if ((in_op_type == VARIABLE) && (in_node->GetOpDesc() != nullptr) &&
        (in_node->GetOpDesc()->MutableOutputDesc(0) != nullptr)) {
      ge::Format format = in_node->GetOpDesc()->MutableOutputDesc(0)->GetFormat();
      ge::OpDescPtr cur_op_desc_ptr = var_node->GetOpDesc();
      if (cur_op_desc_ptr != nullptr) {
        cur_op_desc_ptr->MutableOutputDesc(0)->SetFormat(format);
        cur_op_desc_ptr->MutableOutputDesc(0)->SetOriginFormat(format);
      }
    }
  }
  return domi::SUCCESS;
}

Status VariableFormatPass::UpdateApplyMomentumInputFormat(const ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc_ptr);
  GE_CHECK_NOTNULL(node->GetInDataAnchor(0));
  GE_CHECK_NOTNULL(node->GetInDataAnchor(0)->GetPeerOutAnchor());
  GE_CHECK_NOTNULL(op_desc_ptr->MutableInputDesc(0));
  GE_CHECK_NOTNULL(op_desc_ptr->MutableInputDesc(1));
  GE_CHECK_NOTNULL(op_desc_ptr->MutableOutputDesc(0));
  NodePtr in_node = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  if (in_node != nullptr) {
    string in_op_type = in_node->GetType();
    if ((in_op_type == VARIABLE) && (in_node->GetOpDesc() != nullptr)) {
      ge::Format format = in_node->GetOpDesc()->MutableOutputDesc(0)->GetFormat();
      op_desc_ptr->MutableInputDesc(0)->SetFormat(format);
      op_desc_ptr->MutableInputDesc(0)->SetOriginFormat(format);
      op_desc_ptr->MutableInputDesc(1)->SetFormat(format);
      op_desc_ptr->MutableInputDesc(1)->SetOriginFormat(format);
      op_desc_ptr->MutableOutputDesc(0)->SetFormat(format);
      op_desc_ptr->MutableOutputDesc(0)->SetOriginFormat(format);
    }
  }
  return domi::SUCCESS;
}
}  // namespace ge
