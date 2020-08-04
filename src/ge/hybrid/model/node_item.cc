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

#include "node_item.h"
#include <sstream>

namespace ge {
namespace hybrid {
NodeItem::NodeItem(NodePtr node) : node(std::move(node)) {
  this->op_desc = this->node->GetOpDesc().get();
  this->node_id = this->op_desc->GetId();
  this->num_inputs = this->op_desc->GetInputsSize();
  this->num_outputs = this->op_desc->GetOutputsSize();
  this->node_name = this->node->GetName();
  this->node_type = this->node->GetType();
}

std::string NodeItem::DebugString() const {
  std::stringstream ss;
  ss << "Node: ";
  ss << "id = " << node_id;
  ss << ", name = " << node->GetName();
  ss << ", type = " << node->GetType();
  ss << ", is_dynamic = " << (is_dynamic ? "True" : "False");
  ss << ", unknown_shape_op_type = " << shape_inference_type;
  ss << ", input_start = " << input_start;
  ss << ", num_inputs = " << num_inputs;
  ss << ", output_start = " << output_start;
  ss << ", num_outputs = " << num_outputs;
  ss << ", dependent_nodes = [";
  for (const auto &dep_node : dependent_node_list) {
    ss << dep_node->GetName() << ", ";
  }
  ss << "]";
  int index = 0;
  for (auto &items : outputs) {
    ss << ", output[" << index++ << "]: ";
    for (auto &item : items) {
      ss << "(" << item.second->NodeName() << ":" << item.first << "), ";
    }
  }

  return ss.str();
}
}  // namespace hybrid
}  // namespace ge