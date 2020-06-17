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

#ifndef GE_HYBRID_MODEL_NODE_ITEM_H_
#define GE_HYBRID_MODEL_NODE_ITEM_H_

#include <vector>
#include "graph/node.h"
#include "graph/op_desc.h"
#include "framework/common/types.h"
#include "hybrid/common/tensor_value.h"

namespace ge {
namespace hybrid {
class NodeTask;
class NodeExecutor;

// for caching static information across execution
struct NodeItem {
  explicit NodeItem(NodePtr node);
  ~NodeItem() = default;

  const std::string &NodeName() const { return node_name; }

  const std::string &NodeType() const { return node_type; }

  std::string DebugString() const;

  NodePtr node;
  OpDesc *op_desc;
  int node_id;
  int num_inputs;
  int num_outputs;

  int input_start = -1;
  int output_start = -1;
  bool is_dynamic = false;
  bool has_observer = false;
  UnknowShapeOpType shape_inference_type = DEPEND_IN_SHAPE;
  std::string node_name;
  std::string node_type;
  std::vector<ge::NodePtr> dependent_node_list;
  std::set<int> to_const_output_id_list;

  // src_output_id, dst_anchor_id, dst_node
  vector<NodeItem *> inputs;
  vector<vector<pair<uint32_t, NodeItem *>>> outputs;

  std::shared_ptr<NodeTask> kernel_task;
  const NodeExecutor *node_executor = nullptr;
  std::map<int, ge::GeTensorDescPtr> const_input_shapes;
  std::map<int, ge::NodePtr> ref_outputs;
};
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_MODEL_NODE_ITEM_H_
