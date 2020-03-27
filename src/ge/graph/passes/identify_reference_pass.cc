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

#include "graph/passes/identify_reference_pass.h"

#include <string>
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
Status IdentifyReferencePass::Run(NodePtr &node) {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "param [node] must not be null.");
    return PARAM_INVALID;
  }
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(PARAM_INVALID, "OpDesc of param [node] must not be null.");
    return PARAM_INVALID;
  }

  auto input_names = op_desc->GetAllInputNames();
  auto outputs = op_desc->GetAllOutputName();
  for (auto &output : outputs) {
    for (auto &input_name : input_names) {
      if (input_name == output.first) {
        bool is_ref = true;
        if (AttrUtils::SetBool(op_desc, ATTR_NAME_REFERENCE, is_ref)) {
          GELOGI("param [node] %s is reference node, set attribute %s to be true.",
                 node->GetName().c_str(), ATTR_NAME_REFERENCE.c_str());
          return SUCCESS;
        }
      }
    }
  }

  return SUCCESS;
}
}  // namespace ge
