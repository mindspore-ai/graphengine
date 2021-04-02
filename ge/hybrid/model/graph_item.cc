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

#include "framework/common/util.h"
#include "graph_item.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int kInvalidIndex = -1;
}  // namespace
GraphItem::~GraphItem() {
  GELOGD("[%s] GraphItem destroyed.", name_.c_str());
}

const vector<NodeItem *> &hybrid::GraphItem::GetAllNodes() const {
  return node_items_;
}

const vector<NodeItem *> &GraphItem::GetAllNodes(int group) const {
  if (group == -1) {
    return GetAllNodes();
  }

  if (group >= static_cast<int>(grouped_node_items_.size())) {
    static vector<NodeItem *> empty_nodes;
    return empty_nodes;
  }

  return grouped_node_items_[group];
}

const vector<const NodeItem *> &GraphItem::GetInputNodes() const {
  return input_nodes_;
}

Status GraphItem::GetOutputDescList(vector<ConstGeTensorDescPtr> &output_desc_list) const {
  if (output_node_ == nullptr) {
    return SUCCESS;
  }

  if (is_dynamic_) {
    for (auto &tensor_desc : output_node_->GetOpDesc()->GetAllInputsDescPtr()) {
      output_desc_list.emplace_back(tensor_desc);
    }
  } else {
    for (auto &tensor_desc : output_node_->GetOpDesc()->GetAllOutputsDescPtr()) {
      output_desc_list.emplace_back(tensor_desc);
    }
  }

  return SUCCESS;
}

bool GraphItem::IsDynamic() const {
  return is_dynamic_;
}

const vector<int> &GraphItem::GetInputIndexMapping() const {
  return input_index_mapping_;
}

int GraphItem::GetParentOutputIndex(size_t index) const {
  if (index >= output_index_mapping_.size()) {
    return kInvalidIndex;
  }

  return output_index_mapping_[index];
}

const NodeItem *GraphItem::GetOutputNode() const {
  return output_node_;
}
const vector<std::pair<const NodeItem *, int>> &GraphItem::GetOutputEdges() const {
  return output_edges_;
}

Status GraphItem::GroupNodes() {
  int last_group = INT32_MIN;
  std::set<int> seen_groups;
  for (auto node : node_items_) {
    int group = node->group;
    if (group != last_group) {
      if (seen_groups.find(group) != seen_groups.end()) {
        GELOGE(INTERNAL_ERROR,
            "[Find][Group]Unordered node group found. node = %s, group = %d", node->NodeName().c_str(), group);
        return INTERNAL_ERROR;
      } else {
        last_group = group;
        seen_groups.insert(group);
        grouped_node_items_.emplace_back(std::vector<NodeItem *>());
      }
    }

    GELOGD("Adding node [%s] to group %d", node->NodeName().c_str(), group);
    grouped_node_items_.back().emplace_back(node);
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
