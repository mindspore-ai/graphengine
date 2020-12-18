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

#ifndef COMMON_GRAPH_FORMAT_REFINER_H_
#define COMMON_GRAPH_FORMAT_REFINER_H_

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>
#include "./compute_graph.h"
#include "./external/graph/types.h"
#include "./ge_error_codes.h"

namespace ge {
// ShapeRefiner performs shape inference for compute graphs
class FormatRefiner {
 public:
  static graphStatus InferOrigineFormat(const ge::ComputeGraphPtr &graph);

 private:
  static graphStatus RefreshConstantOutProcess(const ComputeGraphPtr &graph, const OpDescPtr &op_desc);
  static graphStatus GetAnchorPoints(const ge::ComputeGraphPtr &graph, std::vector<ge::NodePtr> &anchor_points,
                                     std::vector<ge::NodePtr> &data_nodes,
                                     std::unordered_map<ge::NodePtr, bool> &node_status);
  static graphStatus AnchorProcess(const ge::NodePtr &anchor_node, std::unordered_map<ge::NodePtr, bool> &node_status);
  static void RefreshOriginFormatOfAnchor(std::vector<ge::NodePtr> &anchor_points);
  static graphStatus BackInferProcess(std::deque<ge::NodePtr> &nodes, ge::NodePtr &node,
                                      std::unordered_map<ge::NodePtr, bool> &node_status);
  static graphStatus ForwardInferProcess(std::deque<ge::NodePtr> &nodes, ge::NodePtr &node,
                                         std::unordered_map<ge::NodePtr, bool> &node_status);
  static graphStatus DataNodeFormatProcess(const ComputeGraphPtr &graph, std::vector<ge::NodePtr> &data_nodes,
                                           ge::Format data_format, std::unordered_map<ge::NodePtr, bool> &node_status);
  static bool IsGraphInferred(const ComputeGraphPtr &graph);
};
}  // namespace ge
#endif  // COMMON_GRAPH_FORMAT_REFINER_H_
