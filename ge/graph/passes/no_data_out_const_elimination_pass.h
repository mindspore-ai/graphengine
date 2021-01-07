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

#ifndef GE_GRAPH_PASSES_REMOVE_CONST_WITHOUT_DATA_PASS_H_
#define GE_GRAPH_PASSES_REMOVE_CONST_WITHOUT_DATA_PASS_H_

#include "graph/passes/base_pass.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"

namespace ge {
class NoDataOutConstEliminationPass : public BaseNodePass {
 public:
  Status Run(ge::NodePtr &node) override;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_REMOVE_CONST_WITHOUT_DATA_PASS_H_
