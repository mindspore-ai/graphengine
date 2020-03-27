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

#ifndef GE_GRAPH_PASSES_CONTROL_OP_ATTR_PASS_H_
#define GE_GRAPH_PASSES_CONTROL_OP_ATTR_PASS_H_

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine_manager/dnnengine_manager.h"
#include "inc/graph_pass.h"

namespace ge {
class ControlOpAttrPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  Status AcquireEngineInfo();
  Status HandleStreamLabel(const ComputeGraphPtr &graph);
  Status HandleSwitchNodes(ComputeGraphPtr &graph);
  bool CheckNeedActiveNode(const std::string &stream_label);

  std::unordered_map<std::string, uint32_t> stream_label_num_;
  // map<label, <has_not_independent_engine_node, has_independent_engine_node>>
  std::unordered_map<std::string, std::pair<bool, bool>> label_flag_;
  std::vector<NodePtr> switch_nodes_;
  std::map<string, EngineConfPtr> engine_confs_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_CONTROL_OP_ATTR_PASS_H_
