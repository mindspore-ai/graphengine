/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef GRAPHENGINE_LLT_ST_GRAPH_BUILDER_H
#define GRAPHENGINE_LLT_ST_GRAPH_BUILDER_H

#include <string>
#include <vector>

#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/graph.h"
#include "graph/node.h"

namespace ge {
namespace st {
class ComputeGraphBuilder {
 public:
  explicit ComputeGraphBuilder(const std::string &name) {
    graph_ = std::make_shared<ComputeGraph>(name);
  }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  void AddDataEdge(NodePtr &src_node, int src_idx, NodePtr &dst_node, int dst_idx);
  void AddControlEdge(NodePtr &src_node, NodePtr &dst_node);
  ComputeGraphPtr GetComputeGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }
  Graph GetGraph() {
    graph_->TopologicalSorting();
    return GraphUtils::CreateGraphFromComputeGraph(graph_);
  }

 private:
  ComputeGraphPtr graph_;
};
}  // namespace st
}  // namespace ge

#endif  // GRAPHENGINE_LLT_ST_GRAPH_BUILDER_H
