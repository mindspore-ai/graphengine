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

#include "graph/common/omg_util.h"

#include <algorithm>

#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace ge {
///
/// @brief get the Original Type of FrameworkOp
/// @param [in] node
/// @param [out] type
/// @return Status
///
Status GetOriginalType(const ge::NodePtr &node, string &type) {
  GE_CHECK_NOTNULL(node);
  type = node->GetType();
  GE_IF_BOOL_EXEC(type != FRAMEWORKOP, return SUCCESS);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  bool ret = ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
  if (!ret) {
    GELOGE(INTERNAL_ERROR, "Get FrameWorkOp original type [%s]", type.c_str());
    return INTERNAL_ERROR;
  }
  GELOGD("Get FrameWorkOp original type [%s]", type.c_str());
  return SUCCESS;
}

///
/// @brief set op stream_label
/// @param [in] node
/// @param [in] label
/// @return Status
///
Status SetStreamLabel(const ge::NodePtr &node, const std::string &label) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);

  if (!AttrUtils::SetStr(tmp_desc, ge::ATTR_NAME_STREAM_LABEL, label)) {
    GELOGE(FAILED, "Op: %s set ATTR_NAME_STREAM_LABEL failed", node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op cycle_event flag
/// @param [in] node
/// @return Status
///
Status SetCycleEvent(const ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetBool(tmp_desc, ge::ATTR_NAME_STREAM_CYCLE_EVENT_FLAG, true)) {
    GELOGE(FAILED, "Op: %s set ATTR_NAME_STREAM_CYCLE_EVENT_FLAG failed", node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op active_label_list
/// @param [in] node
/// @param [in] active_label_list
/// @return Status
///
Status SetActiveLabelList(const ge::NodePtr &node, const std::vector<std::string> &active_label_list) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetListStr(tmp_desc, ge::ATTR_NAME_ACTIVE_LABEL_LIST, active_label_list)) {
    GELOGE(FAILED, "Op: %s set ATTR_NAME_ACTIVE_LABEL_LIST failed", node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op branch_label
/// @param [in] node
/// @param [in] branch_label
/// @return Status
///
Status SetSwitchBranchNodeLabel(const ge::NodePtr &node, const std::string &branch_label) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetStr(tmp_desc, ge::ATTR_NAME_SWITCH_BRANCH_NODE_LABEL, branch_label)) {
    GELOGE(FAILED, "Op: %s set ATTR_NAME_SWITCH_BRANCH_NODE_LABEL failed", node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op true_branch flag
/// @param [in] node
/// @param [in] value
/// @return Status
///
Status SetSwitchTrueBranchFlag(const ge::NodePtr &node, bool value) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetBool(tmp_desc, ge::ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, value)) {
    GELOGE(FAILED, "Op: %s set ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG failed", node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op original name
/// @param [in] node
/// @param [in] orig_name
/// @return Status
///
Status SetOriginalNodeName(const ge::NodePtr &node, const std::string &orig_name) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetStr(tmp_desc, ge::ATTR_NAME_ORIG_NODE_NAME, orig_name)) {
    GELOGE(FAILED, "Op: %s set ATTR_NAME_ORIG_NODE_NAME failed", node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op cyclic_dependence flag
/// @param [in] node
/// @return Status
///
Status SetCyclicDependenceFlag(const ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  if (!AttrUtils::SetBool(tmp_desc, ge::ATTR_NAME_CYCLIC_DEPENDENCE_FLAG, true)) {
    GELOGE(FAILED, "Op: %s set ATTR_NAME_CYCLIC_DEPENDENCE_FLAG failed", node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief set op next_iteration name
/// @param [in] node
/// @param [in] next
/// @return Status
///
Status SetNextIteration(const ge::NodePtr &node, const std::string &next) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);

  if (!AttrUtils::SetStr(tmp_desc, ge::ATTR_NAME_NEXT_ITERATION, next)) {
    GELOGE(FAILED, "Op: %s set ATTR_NAME_NEXT_ITERATION failed", node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}
}  // namespace ge
