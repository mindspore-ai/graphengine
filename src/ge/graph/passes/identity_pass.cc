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

#include "graph/passes/identity_pass.h"

#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/common/omg_util.h"

namespace ge {
namespace {
///
/// A `Identity` node may after a `Switch` node and has control-dependency-out nodes.
/// Or a `Identity` node may before a `Merge` node and has control-dependency-in nodes.
/// The identity nodes are used to represent control dependencies in condition branch, and can not be deleted.
///
Status CheckIdentityUsable(const NodePtr &node, bool &usable) {
  std::string node_type;
  for (auto &in_node : node->GetInDataNodes()) {
    auto ret = GetOriginalType(in_node, node_type);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to get node type from node %s", node->GetName().c_str());
      return ret;
    }
    if ((node_type != SWITCH) && (node_type != REFSWITCH)) {
      GELOGD("skip identity %s connected to switch", node->GetName().c_str());
      break;
    }
    GE_CHECK_NOTNULL(node->GetOutControlAnchor());
    if (!node->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
      usable = true;
      return SUCCESS;
    }
  }
  for (auto &out_node : node->GetOutDataNodes()) {
    auto ret = GetOriginalType(out_node, node_type);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to get node type from node %s", node->GetName().c_str());
      return ret;
    }
    if ((node_type != MERGE) && (node_type != REFMERGE)) {
      GELOGD("skip identity %s connected to merge", node->GetName().c_str());
      break;
    }
    GE_CHECK_NOTNULL(node->GetInControlAnchor());
    if (!node->GetInControlAnchor()->GetPeerOutControlAnchors().empty()) {
      usable = true;
      return SUCCESS;
    }
  }
  usable = false;
  return SUCCESS;
}
}  // namespace

Status IdentityPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    GELOGE(status_ret, "Identity pass get original type fail.");
    return status_ret;
  }
  if ((type != IDENTITY) && (type != IDENTITYN)) {
    return SUCCESS;
  }

  if (!force_) {
    bool usable = false;
    auto ret = CheckIdentityUsable(node, usable);
    if (ret != SUCCESS) {
      return ret;
    }
    if (usable) {
      return SUCCESS;
    }
  }
  size_t n = node->GetOpDesc()->GetOutputsSize();
  if (node->GetOpDesc()->GetInputsSize() != n) {
    GELOGE(PARAM_INVALID, "Identity input / output size must be equal. in size:%lu, out size:%lu",
           node->GetOpDesc()->GetInputsSize(), n);
    return PARAM_INVALID;
  }
  std::vector<int> io_map;
  for (size_t i = 0; i < n; i++) {
    io_map.push_back(i);
  }
  return IsolateAndDeleteNode(node, io_map);
}
}  // namespace ge
