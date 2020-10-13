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

#include "graph/passes/addn_pass.h"

#include <vector>

namespace ge {
namespace {
const size_t kInputSizeSingle = 1;
}  // namespace

Status AddNPass::Run(NodePtr &node) {
  GELOGD("AddNPass running");
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "param [node] must not be null.");
    return PARAM_INVALID;
  }

  if (node->GetType() == ADDN) {
    if (node->GetOpDesc() == nullptr) {
      GELOGE(PARAM_INVALID, "Param [node] op desc is null.");
      return PARAM_INVALID;
    }
    // AddN with single input can be optimized
    if (node->GetOpDesc()->GetInputsSize() == kInputSizeSingle) {
      std::vector<int> io_map = {PassUtils::GetUniqueInDataAnchorIndex(node)};
      return IsolateAndDeleteNode(node, io_map);
    }
  }
  return SUCCESS;
}
}  // namespace ge
