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

#include "subgraph_context.h"

#include "common/debug/log.h"

namespace ge {
namespace hybrid {
SubgraphContext::SubgraphContext(const GraphItem *graph_item) : graph_item_(graph_item) {

}

Status SubgraphContext::Init() {
  GE_CHECK_NOTNULL(graph_item_);
  GELOGD("[%s] Start to init subgraph context. total inputs = %d, total outputs = %d",
         graph_item_->GetName().c_str(),
         graph_item_->TotalInputs(),
         graph_item_->TotalOutputs());
  all_inputs_.resize(static_cast<unsigned long>(graph_item_->TotalInputs()));
  all_outputs_.resize(static_cast<unsigned long>(graph_item_->TotalOutputs()));

  return SUCCESS;
}

NodeStatePtr SubgraphContext::GetOrCreateNodeState(const NodeItem *node_item) {
  std::lock_guard<std::mutex> lk(mu_);
  auto &node_state = node_states_[node_item];
  if (node_state == nullptr) {
    node_state.reset(new(std::nothrow)NodeState(*node_item, this));
  }

  return node_state;
}

Status SubgraphContext::SetInput(int index, const TensorValue &tensor) {
  if (static_cast<size_t>(index) >= all_inputs_.size()) {
    GELOGE(INTERNAL_ERROR,
           "output index output range. all input num = %zu, input index = %d",
           all_inputs_.size(),
           index);
    return INTERNAL_ERROR;
  }
  all_inputs_[index] = tensor;
  return SUCCESS;
}

Status SubgraphContext::SetInput(const NodeItem &node_item, int input_index, const TensorValue &tensor) {
  auto index = node_item.input_start + input_index;
  return SetInput(index, tensor);
}

Status SubgraphContext::SetOutput(const NodeItem &node_item, int output_index, const TensorValue &tensor) {
  auto index = node_item.output_start + output_index;
  if ((output_index >= node_item.num_outputs) || (static_cast<size_t>(index) >= all_outputs_.size())) {
    GELOGE(INTERNAL_ERROR,
           "output index output range. all output num = %zu, node_item = %s, output index = %d",
           all_outputs_.size(),
           node_item.DebugString().c_str(),
           output_index);
    return INTERNAL_ERROR;
  }

  all_outputs_[index] = tensor;
  return SUCCESS;
}

Status SubgraphContext::GetInput(int index, TensorValue &tensor) {
  GE_CHECK_GE(all_inputs_.size(), index + 1U);
  tensor = all_inputs_[index];
  return SUCCESS;
}

Status SubgraphContext::GetOutputs(std::vector<TensorValue> &outputs) {
  if (graph_item_->IsDynamic()) {
    GELOGD("[%s] graph is dynamic, get outputs from net output input tensors", graph_item_->GetName().c_str());
    // get from net output inputs
    auto output_node = graph_item_->GetOutputNode();
    if (output_node != nullptr) {
      for (int i = 0; i < output_node->num_inputs; ++i) {
        TensorValue tensor;
        GE_CHK_STATUS_RET_NOLOG(GetInput(output_node->input_start + i, tensor));
        GELOGD("[%s] Adding output tensor by input index [%d], tensor = %s",
               graph_item_->GetName().c_str(),
               output_node->input_start + i,
               tensor.DebugString().c_str());
        outputs.emplace_back(std::move(tensor));
      }
    }
  } else {
    GELOGD("[%s] graph is non-dynamic, get outputs from subgraph outputs", graph_item_->GetName().c_str());
    for (auto &tensor : all_outputs_) {
      GELOGD("[%s] Adding output tensor: %s", graph_item_->GetName().c_str(), tensor.DebugString().c_str());
      outputs.emplace_back(tensor);
    }
  }

  return SUCCESS;
}

bool SubgraphContext::Await(const NodePtr &node) {
  return node_done_manager_.Await(node);
}

void SubgraphContext::OnError(Status error) {
  GELOGE(error, "[%s] Error occurred while executing graph.", graph_item_->GetName().c_str());
  node_done_manager_.Destroy();
}

void SubgraphContext::NodeDone(const NodePtr &node) {
  node_done_manager_.NodeDone(node);
}
}  // namespace hybrid
}  // namespace ge
