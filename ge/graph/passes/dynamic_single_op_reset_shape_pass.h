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

#ifndef GE_GRAPH_PASSES_DYNAMIC_SINGLE_OP_RESET_SHAPE_PASS_H_
#define GE_GRAPH_PASSES_DYNAMIC_SINGLE_OP_RESET_SHAPE_PASS_H_
#include "graph/graph.h"
#include "inc/graph_pass.h"
#include "init/gelib.h"

namespace ge {
class DynamicSingleOpResetShapePass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  Status ResetOpShape(OpDescPtr &op_desc);
  Status ResetInputTensorShape(OpDescPtr &op_desc, const GeShape &dynamic_shape, bool &reset_shape_flag);
  Status ResetOutputTensorShape(OpDescPtr &op_desc, const GeShape &dynamic_shape);
  Status CheckAllAicpuNodes(const ComputeGraphPtr &graph, bool &is_not_aicpu);
  bool CheckIfConstInput(const GeTensorDescPtr &input_tensor_desc);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_DYNAMIC_SINGLE_OP_RESET_SHAPE_PASS_H_
