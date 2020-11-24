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
GraphItem::~GraphItem() { GELOGD("[%s] GraphItem destroyed.", name_.c_str()); }

const vector<NodeItem *> &hybrid::GraphItem::GetAllNodes() const { return node_items_; }

const vector<const NodeItem *> &GraphItem::GetInputNodes() const { return input_nodes_; }

Status GraphItem::GetOutputDescList(vector<ConstGeTensorDescPtr> &output_desc_list) const {
  if (output_node_ == nullptr) {
    return SUCCESS;
  }

  if (is_dynamic_) {
    for (auto &tensor_desc : output_node_->op_desc->GetAllInputsDescPtr()) {
      output_desc_list.emplace_back(tensor_desc);
    }
  } else {
    for (auto &tensor_desc : output_node_->op_desc->GetAllOutputsDescPtr()) {
      output_desc_list.emplace_back(tensor_desc);
    }
  }

  return SUCCESS;
}

bool GraphItem::IsDynamic() const { return is_dynamic_; }

const vector<int> &GraphItem::GetInputIndexMapping() const { return input_index_mapping_; }

int GraphItem::GetParentOutputIndex(size_t index) const {
  if (index >= output_index_mapping_.size()) {
    return kInvalidIndex;
  }

  return output_index_mapping_[index];
}

const NodeItem *GraphItem::GetOutputNode() const { return output_node_; }
}  // namespace hybrid
}  // namespace ge
