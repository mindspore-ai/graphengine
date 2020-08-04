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

#include "hybrid_execution_context.h"

namespace ge {
namespace hybrid {
NodeStatePtr GraphExecutionContext::GetOrCreateNodeState(const NodePtr &node) {
  auto &node_state = node_states[node];
  if (node_state == nullptr) {
    const NodeItem *node_item = model->GetNodeItem(node);
    if (node_item == nullptr) {
      return nullptr;
    }
    node_state.reset(new (std::nothrow) NodeState(*node_item));
  }

  return node_state;
}

void GraphExecutionContext::OnError(Status error_code) {
  GELOGE(error_code, "Error occurred while executing model");
  {
    std::lock_guard<std::mutex> lk(mu_);
    this->status = error_code;
  }

  compile_queue.Stop();
  execution_queue.Stop();
}

Status GraphExecutionContext::GetStatus() {
  std::lock_guard<std::mutex> lk(mu_);
  return status;
}
}  // namespace hybrid
}  // namespace ge