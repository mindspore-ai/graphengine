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

#include "graph/passes/const_pass.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"

namespace ge {
Status ConstPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);

  if ((node->GetType() != CONSTANT) && (node->GetType() != CONSTANTOP)) {
    return SUCCESS;
  }
  GELOGD("ConstPass running, node: %s.", node->GetName().c_str());

  // const has no control input
  if (node->GetInControlNodes().empty()) {
  	auto out_ctrl_anchor = node->GetOutControlAnchor();
  	if (out_ctrl_anchor != nullptr) {
  	  GELOGD("Node: %s unlink all out control edge.", node->GetName().c_str());
  	  out_ctrl_anchor->UnlinkAll();
  	}

  	if (node->GetOutAllNodes().empty()) {
  	  // it is an isolated const, just remove it.
  	  GELOGD("Delete isolated const: %s.", node->GetName().c_str());
  	  auto graph = node->GetOwnerComputeGraph();
  	  if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != GRAPH_SUCCESS) {
  	  	GELOGE(FAILED, "Remove const %s failed.", node->GetName().c_str());
  	  	return FAILED;
  	  }
  	  AddNodeDeleted(node);
  	}
  }

  return SUCCESS;
}
}  // namespace ge