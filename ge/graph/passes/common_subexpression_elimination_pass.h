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
#ifndef GE_COMMON_SUBEXPRESSION_ELIMINATION_H_
#define GE_COMMON_SUBEXPRESSION_ELIMINATION_H_

#include "external/graph/types.h"
#include "inc/graph_pass.h"

namespace ge {
class CommonSubexpressionEliminationPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph) override ;
};
}  // namespace ge
#endif //GE_COMMON_SUBEXPRESSION_ELIMINATION_H_
