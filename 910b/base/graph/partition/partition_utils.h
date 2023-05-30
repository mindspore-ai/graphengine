/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef D_BASE_GRAPH_PARTITION_PARTITION_UTILS_H
#define D_BASE_GRAPH_PARTITION_PARTITION_UTILS_H

#include "graph/node.h"
namespace ge {
constexpr size_t kUniqueStageNameNum = 1UL;
class PartitionUtils {
 public:
  static bool IsDataLike(const NodePtr &node);
  static graphStatus CheckWritableVarNode(const ComputeGraphPtr &root_graph);
  static graphStatus CheckDtResourceNodes(const ComputeGraphPtr &root_graph);
  static graphStatus GetStageNames(const NodePtr &node, std::vector<std::string> &stage_names);
  static bool CheckSameStageName(const std::vector<std::string> &name1, const std::vector<std::string> &name2);
  static graphStatus SetSubgraphGraphId(const ComputeGraphPtr &root_graph,
                                        const ComputeGraphPtr &subgraph);
  static bool IsOutNode(const NodePtr &node);
  static graphStatus SetSubgraphLogicDeviceId(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &subgraph);
  static graphStatus GetParentDataAndOutIdx(const NodePtr &sub_data, int32_t &in_idx,
                                            NodePtr &root_data, int32_t &out_idx);
  static graphStatus GetParentNetoutputAndInIdx(const NodePtr &sub_netoutput, const uint32_t in_tensor_idx,
                                                NodePtr &root_netoutput, int32_t &in_idx);
};
} // namespace ge
#endif  // D_BASE_GRAPH_PARTITION_PARTITION_UTILS_H
