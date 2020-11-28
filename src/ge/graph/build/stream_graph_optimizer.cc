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

#include "stream_graph_optimizer.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "init/gelib.h"

using std::vector;

namespace {
const int64_t kInvalidStream = -1;
}  // namespace
namespace ge {
StreamGraphOptimizer::~StreamGraphOptimizer() {}

void StreamGraphOptimizer::RefreshNodeId(const ComputeGraphPtr &comp_graph, Graph2SubGraphInfoList &subgraph_map) {
  size_t node_size = comp_graph->GetAllNodesSize();
  GELOGD("Refresh placeholder and end nodeId start from node num: %zu", node_size);
  for (const auto &subgraph_pair : subgraph_map) {
    for (const auto &subgraph_info : subgraph_pair.second) {
      ComputeGraphPtr subgraph = subgraph_info->GetSubGraph();
      if (subgraph == nullptr) {
        continue;
      }
      for (ge::NodePtr &node : subgraph->GetDirectNode()) {
        GE_CHECK_NOTNULL_EXEC(node->GetOpDesc(), return );
        if ((node->GetType() == END) || (node->GetType() == PLACEHOLDER)) {
          node->GetOpDesc()->SetId(static_cast<int64_t>(node_size));
          node_size++;
        }
      }
    }
  }
}

bool StreamGraphOptimizer::IsSameStreamId(const ComputeGraphPtr &comp_graph) {
  if (comp_graph == nullptr) {
    return false;
  }
  std::set<int64_t> stream_set;
  for (const ge::NodePtr &cur_node : comp_graph->GetDirectNode()) {
    GE_IF_BOOL_EXEC(cur_node->GetOpDesc() == nullptr, continue);
    int64_t stream_id = cur_node->GetOpDesc()->GetStreamId();
    if (stream_id == kInvalidStream) {
      continue;
    }
    GELOGD("Node %s in subgraph %s stream id is: %ld, node num: %zu", cur_node->GetName().c_str(),
           comp_graph->GetName().c_str(), stream_id, comp_graph->GetDirectNodesSize());
    stream_set.insert(stream_id);
  }
  if (stream_set.size() > 1) {
    GELOGI("Nodes of graph: %s have different stream id, node num: %zu, different stream num: %zu.",
           comp_graph->GetName().c_str(), comp_graph->GetDirectNodesSize(), stream_set.size());
    return false;
  }
  return true;
}

Status StreamGraphOptimizer::OptimizeStreamedSubGraph(const ComputeGraphPtr &comp_graph,
                                                      Graph2SubGraphInfoList &subgraph_map,
                                                      struct RunContext &run_context) {
  RefreshNodeId(comp_graph, subgraph_map);

  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  GE_CHECK_NOTNULL(instance);

  for (const auto &subgraph_pair : subgraph_map) {
    for (const auto &subgraph_info : subgraph_pair.second) {
      ComputeGraphPtr subgraph = subgraph_info->GetSubGraph();
      GE_CHECK_NOTNULL(subgraph);

      GELOGD("Optimize subgraph %s", subgraph->GetName().c_str());

      std::string engine_name = subgraph_info->GetEngineName();

      vector<GraphOptimizerPtr> graph_optimizers;
      if (instance->DNNEngineManagerObj().IsEngineRegistered(engine_name)) {
        instance->OpsKernelManagerObj().GetGraphOptimizerByEngine(engine_name, graph_optimizers);
        GELOGI("Subgraph: %s start optimize streamed graph. engineName: %s, graph Optimizer num: %zu.",
               subgraph->GetName().c_str(), engine_name.c_str(), graph_optimizers.size());

        auto nodes = subgraph->GetDirectNode();
        if (nodes.empty()) {
          continue;
        }

        if (!IsSameStreamId(subgraph)) {
          GELOGI("There are more than one stream in subgraph %s", subgraph->GetName().c_str());
          continue;
        }
        OpDescPtr op_desc = nodes.at(0)->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        int64_t stream_id = op_desc->GetStreamId();
        if (static_cast<size_t>(stream_id) >= run_context.graphStreamList.size()) {
          GELOGE(FAILED, "stream_id %ld is bigger than run_context.graphStreamList.size() %zu", stream_id,
                 run_context.graphStreamList.size());
          return FAILED;
        }
        run_context.stream = run_context.graphStreamList[stream_id];
        GELOGD("Subgraph has same stream id, subgraph: %s, engine_name: %s, stream_id: %ld, rtstream: %lu.",
               subgraph->GetName().c_str(), engine_name.c_str(), stream_id,
               static_cast<uint64_t>(reinterpret_cast<uintptr_t>(run_context.stream)));
        for (auto iter = graph_optimizers.begin(); iter != graph_optimizers.end(); ++iter) {
          GE_CHECK_NOTNULL(*iter);
          Status ret = (*iter)->OptimizeStreamGraph(*subgraph, run_context);
          if (ret != SUCCESS) {
            GELOGE(
              ret,
              "[optimizeStreamedSubGraph]: optimize streamed subgraph failed, subgraph: %s, engine_name: %s, graph "
              "Optimizer num: %zu, ret: %u",
              subgraph->GetName().c_str(), engine_name.c_str(), graph_optimizers.size(), ret);
            return ret;
          }
          GELOGD(
            "[optimizeStreamedSubGraph]: optimize streamed subgraph success, subgraph: %s, engine_name: %s, graph "
            "Optimizer num: %zu!",
            subgraph->GetName().c_str(), engine_name.c_str(), graph_optimizers.size());
        }
      }
    }
  }

  GELOGD("Optimize streamed subgraph success.");
  return SUCCESS;
}
}  // namespace ge
