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

#include "graph/passes/no_use_reshape_remove_pass.h"

#include <string>
#include <vector>

#include "external/graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "common/op/ge_op_utils.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/passes/pass_utils.h"

namespace ge {
namespace {
const int kReshapeDataIndex = 0;
const int kReshapeShapeIndex = 1;
}  // namespace
Status NoUseReshapeRemovePass::Run(ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr op_desc_ptr = node->GetOpDesc();
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "NoUseReshapeRemovePass enter. OpDesc is null.");
    return PARAM_INVALID;
  }
  if (op_desc_ptr->GetType() != domi::RESHAPE) {
    return SUCCESS;
  }
  GELOGI("NoUseReshapeRemovePass enter.");

  bool to_be_deleted = true;
  // compare input and output dims
  if (op_desc_ptr->GetAllInputsDesc().empty() || op_desc_ptr->GetAllOutputsDesc().empty()) {
    GELOGE(INTERNAL_ERROR, "Input or output num is zero. node name:%s, input size:%zu, output size:%zu",
           op_desc_ptr->GetName().c_str(), op_desc_ptr->GetAllInputsDesc().size(),
           op_desc_ptr->GetAllOutputsDesc().size());
    return INTERNAL_ERROR;
  }
  const auto &input_desc = op_desc_ptr->MutableInputDesc(0);
  const auto &output_desc = op_desc_ptr->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  GE_CHECK_NOTNULL(output_desc);
  std::vector<int64_t> input_4dims = input_desc->GetShape().GetDims();
  std::vector<int64_t> output_4dims = output_desc->GetShape().GetDims();

  if (input_4dims.size() != output_4dims.size()) {
    GELOGI("Input and output dim size is not equal.Keep this reshape op.");
    return SUCCESS;
  }

  size_t vec_size = input_4dims.size();
  for (size_t i = 0; i < vec_size; i++) {
    if (input_4dims[i] < 0) {
      GELOGI("Input shape is unknown.Keep this reshape op.");
      return SUCCESS;
    }
    if (input_4dims[i] != output_4dims[i]) {
      to_be_deleted = false;
      break;
    }
  }
  if (to_be_deleted) {
    GELOGI("NoUseReshapeRemovePass remove useless node:%s", node->GetName().c_str());
    auto ret = PassUtils::UnlinkNodeWithControlCopy(node, kReshapeShapeIndex);
    if (ret != SUCCESS) {
      GELOGE(ret, "DimensionAdjustPass unlink node with control copy fail.");
      return ret;
    }
    return IsolateAndDeleteNode(node, {kReshapeDataIndex});
  }
  return SUCCESS;
}
}  // namespace ge
