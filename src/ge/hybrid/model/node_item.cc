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
#include "common/debug/log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "hybrid/node_executor/node_executor.h"

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

Status NodeItem::Init() {
  int32_t unknown_shape_type_val = 0;
  (void)AttrUtils::GetInt(op_desc, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  shape_inference_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);

  GE_CHK_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node, is_dynamic), "[%s] Failed to get shape status.",
                    node->GetName().c_str());

  if (is_dynamic) {
    for (int i = 0; i < num_inputs; ++i) {
      const auto &input_desc = op_desc->MutableInputDesc(i);
      GE_CHECK_NOTNULL(input_desc);
      if (input_desc->MutableShape().IsUnknownShape()) {
        is_input_shape_static.push_back(false);
      } else {
        num_static_input_shapes++;
        is_input_shape_static.push_back(true);
        GELOGD("[%s] The shape of input[%d] is static. shape = [%s]", NodeName().c_str(), i,
               input_desc->MutableShape().ToString().c_str());
      }
    }

    for (int i = 0; i < num_outputs; ++i) {
      const auto &output_desc = op_desc->MutableOutputDesc(i);
      GE_CHECK_NOTNULL(output_desc);
      if (output_desc->MutableShape().IsUnknownShape()) {
        is_output_shape_static = false;
        break;
      }
    }
  }

  return SUCCESS;
}

bool NodeItem::IsControlOp() const {
  auto op_type = op_desc->GetType();
  return op_type == IF || op_type == CASE || op_type == WHILE || op_type == FOR;
}

std::string NodeItem::DebugString() const {
  std::stringstream ss;
  ss << "Node: ";
  ss << "id = " << node_id;
  ss << ", name = [" << node->GetName();
  ss << "], type = " << node->GetType();
  ss << ", is_dynamic = " << (is_dynamic ? "True" : "False");
  ss << ", is_output_static = " << (is_output_shape_static ? "True" : "False");
  ss << ", unknown_shape_op_type = " << shape_inference_type;
  ss << ", input_start = " << input_start;
  ss << ", num_inputs = " << num_inputs;
  ss << ", output_start = " << output_start;
  ss << ", num_outputs = " << num_outputs;
  ss << ", dependent_nodes = [";
  for (const auto &dep_node : dependents_for_shape_inference) {
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

void NodeItem::SetToDynamic() {
  num_static_input_shapes = 0;
  is_dynamic = true;
  for (size_t i = 0; i < is_input_shape_static.size(); ++i) {
    is_input_shape_static[i] = false;
  }
  if (kernel_task != nullptr && !kernel_task->IsSupportDynamicShape()) {
    GELOGD("[%s] Dynamic shape is not supported, clear node task.", node_name.c_str());
    kernel_task = nullptr;
  }
}
}  // namespace hybrid
}  // namespace ge
