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

#include "graph/passes/isolated_op_remove_pass.h"

#include "common/debug/log.h"
#include "common/types.h"
#include "common/util.h"


namespace ge {
Status IsolatedOpRemovePass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (NodePtr &node_ptr : graph->GetDirectNode()) {
    GE_IF_BOOL_EXEC(node_ptr->GetOpDesc() == nullptr, continue);
    if (node_ptr->GetInDataNodes().size() == 0 && node_ptr->GetOutAllNodes().size() == 0 &&
        !(node_ptr->GetOpDesc()->HasAttr(TO_BE_OUTPUT))) {
      GE_RETURN_WITH_LOG_IF_ERROR(graph->RemoveNode(node_ptr), "remove graph node [%s] fail",
                                  node_ptr->GetOpDesc()->GetName().c_str());
    }
  }

  return SUCCESS;
}
}  // namespace ge
