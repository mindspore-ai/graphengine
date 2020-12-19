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

#ifndef COMMON_GRAPH_REF_RELATION_H_
#define COMMON_GRAPH_REF_RELATION_H_

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/compute_graph.h"
#include "graph/types.h"
#include "graph/ge_error_codes.h"
#include "node.h"

namespace ge {
enum InOutFlag {
  NODE_IN   = 0,  // input flag
  NODE_OUT  = 1,  // output flag
};

struct RefCell {
  std::string node_name;
  ge::NodePtr node = nullptr;
  InOutFlag in_out = NODE_IN;
  int in_out_idx   = 0;

  bool operator == (const RefCell &c) const {
    return node_name == c.node_name && node == c.node && in_out == c.in_out && in_out_idx == c.in_out_idx;
  }

  RefCell() = default;
  RefCell(std::string name, ge::NodePtr node_ptr, InOutFlag in_out_flag, int idx) {
    node_name = name;
    node = node_ptr;
    in_out = in_out_flag;
    in_out_idx = idx;
  };
  ~RefCell() = default;
};

struct RefCellHash{
    size_t operator () (const RefCell &c) const {
      unsigned long number = static_cast<unsigned long>(reinterpret_cast<uintptr_t>(c.node.get()));
      string tmp = c.node_name + std::to_string(c.in_out) + std::to_string(c.in_out_idx)
                  + std::to_string(number);
      return std::hash<string>()(tmp);
    }
};

class RefRelations {
public:
  graphStatus LookUpRefRelations(const RefCell &key, std::unordered_set<RefCell, RefCellHash> &result);
  graphStatus BuildRefRelations(ge::ComputeGraph &root_graph);
  graphStatus Clear();

  RefRelations();
  ~RefRelations() = default;
public:
  class Impl;
  std::shared_ptr<Impl> impl_ = nullptr;
};

}  // namespace ge
#endif  // COMMON_GRAPH_REF_RELATION_H_
