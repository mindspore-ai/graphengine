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
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
HybridModel::HybridModel(GeRootModelPtr ge_model) : ge_root_model_(std::move(ge_model)) {
}

HybridModel::~HybridModel() {
  GELOGD("[%s] HybridModel destroyed.", model_name_.c_str());
}

Status HybridModel::Init() {
  GELOGD("Start to init hybrid model.");
  GE_CHK_STATUS_RET(HybridModelBuilder(*this).Build(), "Failed to build hybrid model.");
  GELOGD("HybridModel initialized successfully.");
  return SUCCESS;
}

TensorValue* HybridModel::GetVariable(const string &name) const {
  auto it = variable_tensors_.find(name);
  if (it == variable_tensors_.end()) {
    GELOGI("Failed to get variable tensor. var name = [%s]", name.c_str());
    return nullptr;
  }

  GELOGD("Got variable tensor. var name = [%s], tensor = %s", name.c_str(), it->second->DebugString().c_str());
  return it->second.get();
}

NodePtr HybridModel::GetVariableNode(const string &name) const {
  auto it = device_variable_nodes_.find(name);
  if (it != device_variable_nodes_.end()) {
    return it->second;
  }
  auto host_find = host_variable_nodes_.find(name);
  if (host_find != host_variable_nodes_.end()) {
    return host_find->second;
  }
  GELOGI("Failed to get variable node by name = [%s]", name.c_str());
  return nullptr;
}

const std::vector<domi::TaskDef> *HybridModel::GetTaskDefs(const NodePtr &node) const {
  auto it = task_defs_.find(node);
  if (it == task_defs_.end()) {
    return nullptr;
  }

  return &it->second;
}

NodeItem *HybridModel::MutableNodeItem(const NodePtr &node) {
  auto it = node_items_.find(node);
  if (it == node_items_.end()) {
    return nullptr;
  }

  return it->second.get();
}

const NodeItem *HybridModel::GetNodeItem(const NodePtr &node) const {
  auto it = node_items_.find(node);
  if (it == node_items_.end()) {
    return nullptr;
  }

  return it->second.get();
}

GeModelPtr HybridModel::GetGeModel(const NodePtr &node) const {
  auto it = known_shape_sub_models_.find(node);
  if (it == known_shape_sub_models_.end()) {
    GELOGE(INTERNAL_ERROR, "[%s] Failed to get GeModel for subgraph node.", node->GetName().c_str());
    return nullptr;
  }

  return it->second;
}

const GraphItem* HybridModel::GetRootGraphItem() const {
  return root_graph_item_.get();
}

const GraphItem *HybridModel::GetSubgraphItem(const std::string &graph_name) const {
  GELOGD("To find subgraph item by name = %s", graph_name.c_str());
  auto it = subgraph_items_.find(graph_name);
  if (it == subgraph_items_.end()) {
    GELOGD("Subgraph item not found by node = %s", graph_name.c_str());
    return nullptr;
  }

  return it->second.get();
}

const GraphItem *HybridModel::GetSubgraphItem(const ComputeGraphPtr &subgraph) const {
  if (subgraph == nullptr) {
    GELOGE(PARAM_INVALID, "subgraph is nullptr");
    return nullptr;
  }

  auto subgraph_name = subgraph->GetName();
  return GetSubgraphItem(subgraph_name);
}

const string &HybridModel::GetModelName() const {
    return model_name_;
}
}  // namespace hybrid
}  // namespace ge
