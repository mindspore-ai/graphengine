/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "graph/passes/no_data_out_const_elimination_pass.h"

namespace ge {
Status NoDataOutConstEliminationPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GELOGD("RemoveConstWithoutDataPass running of %s.", node->GetName().c_str());
  if (node->GetType() == CONSTANT || node->GetType() == CONSTANTOP) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    // delete const which has no input and no output of data
    if (node->GetOpDesc()->GetInputsSize() == 0 && node->GetOutDataNodes().size() == 0) {
      GELOGI("Remove const %s.", node->GetName().c_str());
      if (IsolateAndDeleteNode(node, {}) != SUCCESS) {
        GELOGE(FAILED, "IsolateAndDeleteNode %s failed.", node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
