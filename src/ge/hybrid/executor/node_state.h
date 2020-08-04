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

#ifndef GE_HYBRID_EXECUTOR_NODE_STATE_H_
#define GE_HYBRID_EXECUTOR_NODE_STATE_H_

#include "hybrid/model/node_item.h"

namespace ge {
namespace hybrid {

class NodeTask;

// 存放一些会变化的信息...
class NodeState {
 public:
  NodeState() = default;
  explicit NodeState(const NodeItem &node_item);
  ~NodeState() = default;

  inline int NodeId() const { return node_item->node_id; }

  inline Node *GetNode() const { return node_item->node.get(); }

  OpDesc *GetOpDesc() const { return op_desc.get(); }

  inline const NodeItem *GetNodeItem() const { return node_item; }

  inline const string &GetName() const { return node_item->NodeName(); }

  inline const string &GetType() const { return node_item->NodeType(); }

  // private:
  const NodeItem *node_item = nullptr;
  std::shared_ptr<NodeTask> kernel_task = nullptr;

  bool is_compiled = false;
  OpDescPtr op_desc;
};

using NodeStatePtr = std::shared_ptr<NodeState>;
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_EXECUTOR_NODE_STATE_H_
