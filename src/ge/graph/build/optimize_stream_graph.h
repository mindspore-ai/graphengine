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

#ifndef GE_GRAPH_BUILD_OPTIMIZE_STREAM_GRAPH_H_
#define GE_GRAPH_BUILD_OPTIMIZE_STREAM_GRAPH_H_

#include <vector>

#include "common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"

namespace ge {
class OptimizeStreamGraph {
 public:
  OptimizeStreamGraph() = default;

  OptimizeStreamGraph(const OptimizeStreamGraph &) = delete;

  OptimizeStreamGraph &operator=(const OptimizeStreamGraph &) = delete;

  virtual ~OptimizeStreamGraph();

  Status OptimizeStreamedSubGraph(const ComputeGraphPtr &comp_graph, std::vector<SubGraphInfoPtr> &subgraph_ptr_list,
                                  struct RunContext &run_context);

 private:
  void RefreshNodeId(const ComputeGraphPtr &comp_graph, std::vector<SubGraphInfoPtr> &subgraph_ptr_list);

  bool IsSameStreamId(const ComputeGraphPtr &comp_graph);
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_OPTIMIZE_STREAM_GRAPH_H_
