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

#include "hybrid_model.h"
#include <vector>
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
HybridModel::HybridModel(GeRootModelPtr ge_model) : ge_root_model_(std::move(ge_model)) {}

Status HybridModel::Init() {
  GELOGD("Start to init hybrid model.");
  GE_CHK_STATUS_RET(HybridModelBuilder(*this).Build(), "Failed to build hybrid model.");
  GELOGD("HybridModel initialized successfully.");
  return SUCCESS;
}

void HybridModel::Print() const {
  for (const auto &node : node_items_) {
    GELOGD("%s", node->DebugString().c_str());
  }
}

TensorValue *HybridModel::GetWeight(const NodeItem *const_node) const {
  auto it = weights_.find(const_node->node_id);
  if (it == weights_.end() || it->second == nullptr) {
    GELOGE(INTERNAL_ERROR, "[%s] Failed to get weight", const_node->NodeName().c_str());
    return nullptr;
  }

  return it->second.get();
}

TensorValue *HybridModel::GetVariable(const string &name) const {
  auto it = variable_tensors_.find(name);
  if (it == variable_tensors_.end()) {
    GELOGI("Failed to get variable tensor. var name = [%s]", name.c_str());
    return nullptr;
  }

  GELOGD("Got variable tensor. var name = [%s], tensor = %s", name.c_str(), it->second->DebugString().c_str());
  return it->second.get();
}

NodePtr HybridModel::GetVariableNode(const string &name) const {
  auto it = variable_nodes_.find(name);
  if (it == variable_nodes_.end()) {
    GELOGI("Failed to get variable node by name = [%s]", name.c_str());
    return nullptr;
  }

  return it->second;
}

const std::vector<domi::TaskDef> *HybridModel::GetTaskDefs(const NodePtr &node) const {
  auto it = task_defs_.find(node);
  if (it == task_defs_.end()) {
    return nullptr;
  }

  return &it->second;
}

NodeItem *HybridModel::MutableNodeItem(const NodePtr &node) {
  auto node_id = node->GetOpDesc()->GetId();
  if (node_id < 0 || static_cast<size_t>(node_id) > node_items_.size()) {
    GELOGE(INTERNAL_ERROR, "index out of range. node_id = %ld, num_nodes = %zu", node_id, node_items_.size());
    return nullptr;
  }
  return node_items_[node_id].get();
}

const NodeItem *HybridModel::GetNodeItem(const NodePtr &node) const {
  auto node_id = node->GetOpDesc()->GetId();
  if (node_id < 0 || static_cast<size_t>(node_id) > node_items_.size()) {
    GELOGE(INTERNAL_ERROR, "Index out of range. node_id = %ld, num_nodes = %zu.", node_id, node_items_.size());
    return nullptr;
  }
  return node_items_[node_id].get();
}

GeModelPtr HybridModel::GetGeModel(const NodePtr &node) const {
  auto it = known_shape_sub_graphs_.find(node);
  if (it == known_shape_sub_graphs_.end()) {
    GELOGE(INTERNAL_ERROR, "[%s] Failed to get GeModel for subgraph node.", node->GetName().c_str());
    return nullptr;
  }

  return it->second;
}

const vector<int> &HybridModel::GetNetOutputInputOffsets() const { return net_output_input_offsets_; }

void HybridModel::SetDeviceId(uint32_t device_id) { device_id_ = device_id; }
}  // namespace hybrid
}  // namespace ge
