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

#ifndef GE_GRAPH_PASSES_UNUSED_OP_REMOVE_PASS_H_
#define GE_GRAPH_PASSES_UNUSED_OP_REMOVE_PASS_H_

#include <string>
#include <vector>

#include "framework/common/ge_types.h"
#include "inc/graph_pass.h"

namespace ge {
class UnusedOpRemovePass : public GraphPass {
 public:
  explicit UnusedOpRemovePass(FrameworkType type) : fmktype_(type) {}
  ~UnusedOpRemovePass() {}
  Status Run(ge::ComputeGraphPtr graph) override;
  bool IsExceptions(const ge::NodePtr &node);

 private:
  Status CollectParentNode(const ge::ComputeGraphPtr &graph, const ge::NodePtr &node,
                           std::vector<ge::NodePtr> &node_vec);
  std::vector<std::string> v_remove_ops;
  FrameworkType fmktype_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_UNUSED_OP_REMOVE_PASS_H_
