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

#ifndef COMMON_GRAPH_DEBUG_GRAPH_DEBUG_H_
#define COMMON_GRAPH_DEBUG_GRAPH_DEBUG_H_
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "external/graph/graph.h"
#include "./ge_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_log.h"
#include "graph/node.h"
#include "utils/graph_utils.h"

namespace ge {
enum DotFileFlag {
  // Show nodes, edges, size, type and format
  DOT_FLAG_DEFAULT = 0,
  DOT_NOT_SHOW_EDGE_LABEL = 1,
};
class GraphDebugPrinter {
 public:
  static graphStatus DumpGraphDotFile(const Graph &graph, const std::string &output_dot_file_name,
                                      uint32_t flag = DOT_FLAG_DEFAULT);
  static graphStatus DumpGraphDotFile(const ComputeGraphPtr graph, const std::string &output_dot_file_name,
                                      uint32_t flag = DOT_FLAG_DEFAULT);
  static void DumpNodeToDot(const NodePtr node, std::ostringstream &out_);
  static void DumpEdgeToDot(const NodePtr node, std::ostringstream &out_, uint32_t flag = DOT_FLAG_DEFAULT);
};
}  // namespace ge

#endif  // COMMON_GRAPH_DEBUG_GRAPH_DEBUG_H_
