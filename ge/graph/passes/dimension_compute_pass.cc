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


#include "graph/passes/dimension_compute_pass.h"

#include <memory>
#include <vector>

#include "common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/attr_utils.h"
#include "inc/kernel.h"

namespace ge {
Status DimensionComputePass::Run(ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto op_kernel = folding_pass::GetKernelByType(node);
  if (op_kernel == nullptr || folding_pass::IsNoNeedConstantFolding(node)) {
    return SUCCESS;
  }
  std::vector<GeTensorPtr> outputs;
  auto ret = op_kernel->Compute(node, outputs);
  if (ret != SUCCESS) {
    if (ret == NOT_CHANGED) {
      return SUCCESS;
    } else {
      GELOGE(ret, "DimensionComputePass Compute failed");
      return ret;
    }
  }

  if (outputs.empty()) {
    GELOGE(INTERNAL_ERROR,
           "Failed to compute dims for node %s,"
           " no output weight",
           node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  return Folding(node, outputs);
}
}  // namespace ge
