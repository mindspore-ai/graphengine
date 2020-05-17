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

#include "graph/passes/guarantee_const_pass.h"

#include <string>

#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/common/omg_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"

using domi::GUARANTEECONST;

namespace ge {
namespace {
const uint32_t kGuaranteeConstInputsSize = 1;
}
Status GuaranteeConstPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    GELOGE(status_ret, "GuaranteeConstPass get original type fail.");
    return status_ret;
  }
  if (type != GUARANTEECONST) {
    return SUCCESS;
  }
  if (node->GetOpDesc()->GetAllInputsDesc().size() != kGuaranteeConstInputsSize) {
    GELOGE(PARAM_INVALID, "input size error. Input size:%zu", node->GetOpDesc()->GetAllInputsDesc().size());
    return PARAM_INVALID;
  }
  // [Cascade pointer]
  const auto &in_desc = node->GetOpDesc()->MutableInputDesc(0);
  GE_CHECK_NOTNULL(in_desc);
  // Input tensor cannot be a resource variable handle.
  const DataType &input_dtype = in_desc->GetDataType();
  if (input_dtype == DT_RESOURCE) {
    GELOGE(FAILED, "Input tensor cannot be a resource variable handle in [%s].", node->GetName().c_str());
    return FAILED;
  }

  return IsolateAndDeleteNode(node, {0});
}
}  // namespace ge
