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

#include "graph/passes/dimension_adjust_pass.h"

#include <memory>
#include <string>
#include <vector>
#include "graph/utils/node_utils.h"

namespace ge {
namespace {
const int kDataInputIndex = 0;
const int kRemoveInputIndex = 1;
}  // namespace

Status DimensionAdjustPass::Run(ge::NodePtr &node) {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "node is nullptr");
    return PARAM_INVALID;
  }

  OpDescPtr op_desc_ptr = node->GetOpDesc();
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "GetOpDesc return nullptr");
    return PARAM_INVALID;
  }

  string type;
  Status ret = GetOriginalType(node, type);
  if (ret != SUCCESS) {
    GELOGE(ret, "DimensionAdjustPass get originnal type fail.");
    return ret;
  }

  KernelFactory &factory = KernelFactory::Instance();
  shared_ptr<Kernel> op_kernel = factory.Create(type);
  if (op_kernel == nullptr) {
    return SUCCESS;
  }
  bool is_unknown = false;
  auto ret_status = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
  if (ret_status != GRAPH_SUCCESS) {
    GELOGW("Get node unknown status failed, node name:%s, type:%s.", node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (is_unknown) {
    GELOGI("Current node %s, type %s is unknown shape which should be skip.",
           node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  // call compute function
  ret = op_kernel->Compute(node);
  if (ret != SUCCESS) {
    if (ret == NOT_CHANGED) {
      return SUCCESS;
    }
    GELOGE(ret, "DimensionAdjustPass compute failed");
    return ret;
  }
  if (node->GetAllInDataAnchors().size() > static_cast<size_t>(kRemoveInputIndex)) {
    ret = PassUtils::UnlinkNodeWithControlCopy(node, kRemoveInputIndex);
    if (ret != SUCCESS) {
      GELOGE(ret, "DimensionAdjustPass unlink node with control copy fail.");
      return ret;
    }
  }

  std::vector<int> data_relink_io_map = {kDataInputIndex};
  return IsolateAndDeleteNode(node, data_relink_io_map);
}
}  // namespace ge
