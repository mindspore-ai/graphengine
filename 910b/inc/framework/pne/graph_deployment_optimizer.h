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

#ifndef AIR_INC_FRAMEWORK_PNE_GRAPH_DEPLOYMENT_OPTIMIZER_H_
#define AIR_INC_FRAMEWORK_PNE_GRAPH_DEPLOYMENT_OPTIMIZER_H_

#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/parallelism/tensor_parallel_attrs.h"

namespace ge {
class GraphDeploymentOptimizer {
 public:
  static Status OptimizeForAutoDeploy(const ComputeGraphPtr &compute_graph, ComputeGraphPtr &optimized_graph);
  static Status OptimizeByRecomputation(const ComputeGraphPtr &graph);
  static Status OptimizeByGraphSlicing(const std::vector<std::pair<NodePtr, tp::NodeSliceStrategy>> &nodes_sliced_infos,
                                       const ComputeGraphPtr &compute_graph);
};
}  // namespace ge

#endif  // AIR_INC_FRAMEWORK_PNE_GRAPH_DEPLOYMENT_OPTIMIZER_H_
