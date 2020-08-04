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

#include "graph/passes/infershape_pass.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/shape_refiner.h"

namespace ge {
Status InferShapePass::Run(NodePtr &node) {
  auto ret = ShapeRefiner::InferShapeAndType(node, !OptionExists(kOptimizeAfterSubGraph));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GE_GRAPH_INFERSHAPE_FAILED, "infershape failed. node: %s", node->GetName().c_str());
    return GE_GRAPH_INFERSHAPE_FAILED;
  }
  return SUCCESS;
}
}  // namespace ge
