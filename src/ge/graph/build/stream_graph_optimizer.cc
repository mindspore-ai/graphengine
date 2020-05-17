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
static const int64_t kInvalidStream = -1;
}  // namespace
namespace ge {
StreamGraphOptimizer::~StreamGraphOptimizer() {}

void StreamGraphOptimizer::RefreshNodeId(const ComputeGraphPtr &comp_graph, vector<SubGraphInfoPtr> &subgraph_infos) {
  size_t node_size = comp_graph->GetDirectNodesSize();
  GELOGI("Refresh placeholder and end nodeId start from node num: %zu", node_size);
  for (const auto &sub_graph_info : subgraph_infos) {
    ComputeGraphPtr sub_graph = sub_graph_info->GetSubGraph();
    if (sub_graph == nullptr) {
      continue;
    }
    for (ge::NodePtr &node : sub_graph->GetDirectNode()) {
      GE_CHECK_NOTNULL_EXEC(node->GetOpDesc(), return );
      if ((node->GetType() == domi::END) || (node->GetType() == domi::PLACEHOLDER)) {
        node->GetOpDesc()->SetId(static_cast<int64_t>(node_size));
        node_size++;
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
                                                      vector<SubGraphInfoPtr> &subgraph_infos,
                                                      struct RunContext &run_context) {
  Status ret = SUCCESS;
  GELOGI("Begin to Get optimize streamed subgraph.");

  RefreshNodeId(comp_graph, subgraph_infos);

  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  GE_CHECK_NOTNULL(instance);

  for (auto &sub_graph_info : subgraph_infos) {
    ComputeGraphPtr sub_graph = sub_graph_info->GetSubGraph();
    if (sub_graph == nullptr) {
      continue;
    }

    std::string engine_name = sub_graph_info->GetEngineName();

    vector<GraphOptimizerPtr> graph_optimizers;
    if (instance->DNNEngineManagerObj().IsEngineRegistered(engine_name)) {
      instance->OpsKernelManagerObj().GetGraphOptimizerByEngine(engine_name, graph_optimizers);
      GELOGI("Subgraph: %s start optimize streamed graph. engineName: %s, subgraph num: %zu, graph Optimizer num: %zu.",
             sub_graph->GetName().c_str(), engine_name.c_str(), subgraph_infos.size(), graph_optimizers.size());

      auto nodes = sub_graph->GetDirectNode();
      if (nodes.empty()) {
        continue;
      }
      if (!IsSameStreamId(sub_graph)) {
        GELOGI("There are more than one stream in subgraph %s", sub_graph->GetName().c_str());
        continue;
      }
      OpDescPtr op_desc = nodes.at(0)->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      int64_t stream_id = op_desc->GetStreamId();
      if (static_cast<size_t>(stream_id) >= run_context.graphStreamList.size()) {
        GELOGE(FAILED, "stream_id is bigger than run_context.graphStreamList.size()");
        return FAILED;
      }
      run_context.stream = run_context.graphStreamList[stream_id];
      GELOGD("Subgraph has same stream id, subgraph: %s, engine_name: %s, stream_id: %ld, rtstream: %lu.",
             sub_graph->GetName().c_str(), engine_name.c_str(), stream_id,
             static_cast<uint64_t>(reinterpret_cast<uintptr_t>(run_context.stream)));
      for (auto iter = graph_optimizers.begin(); iter != graph_optimizers.end(); ++iter) {
        GE_CHECK_NOTNULL(*iter);
        ret = (*iter)->OptimizeStreamGraph(*sub_graph, run_context);
        if (ret != SUCCESS) {
          GELOGE(ret,
                 "[optimizeStreamedSubGraph]: optimize streamed subgraph failed, subgraph: %s, engine_name: %s, graph "
                 "Optimizer num: %zu, ret: %u",
                 sub_graph->GetName().c_str(), engine_name.c_str(), graph_optimizers.size(), ret);
          return ret;
        }
        GELOGI(
          "[optimizeStreamedSubGraph]: optimize streamed subgraph success, subgraph: %s, engine_name: %s, graph "
          "Optimizer num: %zu!",
          sub_graph->GetName().c_str(), engine_name.c_str(), graph_optimizers.size());
      }
    }
  }

  return ret;
}
}  // namespace ge
