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

#include "graph/build/logical_stream_allocator.h"
#include <queue>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/common/ge_call_wrapper.h"

using std::map;
using std::set;
using std::string;
using std::vector;
using std::queue;

namespace ge {
LogicalStreamPass::LogicalStreamPass(const string &name) : name_(name) {}

const string &LogicalStreamPass::GetName() const {
  return name_;
}

bool LogicalStreamPass::IsEngineSkip(const Subgraph &subgraph) const {
  return subgraph.engine_conf.skip_assign_stream;
}

bool LogicalStreamPass::IsEngineAttach(const Subgraph &subgraph) const {
  return subgraph.engine_conf.attach;
}

bool LogicalStreamPass::IsEngineIndependent(const Subgraph &subgraph) const {
  return subgraph.engine_conf.independent;
}

bool LogicalStreamPass::HasStreamLabel(const Subgraph &subgraph) const {
  return !subgraph.subgraph_info.GetStreamLabel().empty();
}

bool LogicalStreamPass::HasAssignedStream(const Subgraph &subgraph) const {
  return subgraph.stream_id != kInvalidStream;
}

Status AssignByLabelPass::Run(ComputeGraphPtr graph, const vector<SubgraphPtr> &subgraphs, Context &context) {
  bool changed = false;
  int64_t &next_stream = context.next_stream;
  map<string, int64_t> label_streams;

  for (const SubgraphPtr &subgraph : subgraphs) {
    const string &stream_label = subgraph->subgraph_info.GetStreamLabel();
    if (!stream_label.empty()) {
      // Subgraphs of the same stream_label are assigned to the same stream,
      // and different stream_labels are assigned new streams.
      auto iter = label_streams.find(stream_label);
      if (iter == label_streams.end()) {
        subgraph->stream_id = next_stream;
        GELOGI("Assign new stream %ld for label %s.", next_stream, stream_label.c_str());

        label_streams.emplace(stream_label, next_stream);
        next_stream++;
      } else {
        subgraph->stream_id = iter->second;
      }
      changed = true;
    }
  }

  return changed ? SUCCESS : NOT_CHANGED;
}

Status IndependentStreamPass::Run(ComputeGraphPtr graph, const vector<SubgraphPtr> &subgraphs, Context &context) {
  bool changed = false;
  int64_t &next_stream = context.next_stream;

  // <engine, <label, stream>>
  map<string, map<string, int64_t>> engine_streams;

  for (const SubgraphPtr &subgraph : subgraphs) {
    if (!IsEngineIndependent(*subgraph)) {
      continue;
    }

    const string &engine = subgraph->engine_conf.id;
    const string &stream_label = subgraph->subgraph_info.GetStreamLabel();
    auto &label_streams = engine_streams[engine];
    auto iter = label_streams.find(stream_label);
    if (iter == label_streams.end()) {
      subgraph->stream_id = next_stream;
      GELOGI("Assign new independent stream %ld for engine %s (label: %s).", next_stream, engine.c_str(),
             stream_label.c_str());

      label_streams.emplace(stream_label, next_stream);
      next_stream++;
    } else {
      subgraph->stream_id = iter->second;
    }
    changed = true;
  }

  return changed ? SUCCESS : NOT_CHANGED;
}

Status AssignByDependencyPass::Run(ComputeGraphPtr graph, const vector<SubgraphPtr> &subgraphs, Context &context) {
  bool changed = false;
  map<NodePtr, SubgraphPtr> end_subgraph_map;
  map<NodePtr, SubgraphPtr> pld_subgraph_map;
  InitEndSubgraphMap(subgraphs, end_subgraph_map);
  InitPldSubgraphMap(subgraphs, pld_subgraph_map);

  for (const SubgraphPtr &subgraph : subgraphs) {
    if (HasAssignedStream(*subgraph)) {
      continue;
    }

    SubgraphPtr reusable_subgraph = GetReusableSubgraph(subgraph, end_subgraph_map, pld_subgraph_map);
    if (reusable_subgraph == nullptr) {
      (void)AssignNewStream(subgraph);
    } else {
      if (HasAssignedStream(*reusable_subgraph)) {
        subgraph->stream_id = reusable_subgraph->stream_id;
      } else {
        int64_t stream_id = AssignNewStream(reusable_subgraph);
        subgraph->stream_id = stream_id;
        GELOGI("Reusable subgraph %s has not been assigned a stream, now assign new stream %ld.",
               reusable_subgraph->name.c_str(), stream_id);
      }

      if (reusable_subgraph->reused_subgraph != nullptr) {
        reusable_subgraph = reusable_subgraph->reused_subgraph;
      }

      subgraph->reused_subgraph = reusable_subgraph;
      reused_subgraphs_.emplace_back(subgraph, reusable_subgraph);
      GELOGI("Subgraph %s of engine %s reuses stream of subgraph %s of engine %s.", subgraph->name.c_str(),
             subgraph->engine_conf.id.c_str(), reusable_subgraph->name.c_str(),
             reusable_subgraph->engine_conf.id.c_str());
    }
    changed = true;
  }

  UpdateAssignedSubgraphs(context);
  UpdateReusedSubgraphs();

  return changed ? SUCCESS : NOT_CHANGED;
}

void AssignByDependencyPass::InitEndSubgraphMap(const vector<SubgraphPtr> &subgraphs,
                                                map<NodePtr, SubgraphPtr> &end_subgraph_map) {
  for (const auto &subgraph : subgraphs) {
    const SubGraphInfo &subgraph_info = subgraph->subgraph_info;
    for (const auto &item : subgraph_info.GetEnd2PldMap()) {
      end_subgraph_map.emplace(item.first, subgraph);
    }
  }
}

void AssignByDependencyPass::InitPldSubgraphMap(const vector<SubgraphPtr> &subgraphs,
                                                map<NodePtr, SubgraphPtr> &pld_subgraph_map) {
  for (const auto &subgraph : subgraphs) {
    const SubGraphInfo &subgraph_info = subgraph->subgraph_info;
    for (const auto &item : subgraph_info.GetPld2EndMap()) {
      pld_subgraph_map.emplace(item.first, subgraph);
    }
  }
}

bool AssignByDependencyPass::CouldReuse(const SubgraphPtr &subgraph, const SubgraphPtr &pred_subgraph,
                                        const map<NodePtr, SubgraphPtr> &pld_subgraph_map) {
  if ((subgraph == nullptr) || (pred_subgraph == nullptr)) {
    return false;
  }

  if (subgraph->engine_conf.scheduler_id != pred_subgraph->engine_conf.scheduler_id) {
    return false;
  }

  if (IsEngineIndependent(*pred_subgraph) || HasStreamLabel(*pred_subgraph)) {
    return false;
  }

  // If the engine of the predecessor subgraph is the same as the other successor subgraphs, the stream is not reused.
  for (const auto &end_pld_pair : pred_subgraph->subgraph_info.GetEnd2PldMap()) {
    auto iter = pld_subgraph_map.find(end_pld_pair.second);
    if (iter != pld_subgraph_map.end()) {
      const SubgraphPtr &pred_subgraph_succ = iter->second;
      if ((pred_subgraph_succ != subgraph) &&
          (pred_subgraph_succ->engine_conf.id == pred_subgraph->engine_conf.id)) {
        return false;
      }
    }
  }

  if ((subgraph->engine_conf.id == pred_subgraph->engine_conf.id) ||
      IsEngineAttach(*subgraph)) {
    return true;
  }

  if ((pred_subgraph->reused_subgraph != nullptr) &&
      (pred_subgraph->reused_subgraph->engine_conf.id == subgraph->engine_conf.id)) {
    return true;
  }

  return false;
}

LogicalStreamPass::SubgraphPtr AssignByDependencyPass::GetReusableSubgraph(
    const SubgraphPtr &subgraph, const map<NodePtr, SubgraphPtr> &end_subgraph_map,
    const map<NodePtr, SubgraphPtr> &pld_subgraph_map) {
  const SubGraphInfo &subgraph_info = subgraph->subgraph_info;
  for (const auto &pld_2_end : subgraph_info.GetPld2EndMap()) {
    const NodePtr &peer_end = pld_2_end.second;
    auto iter = end_subgraph_map.find(peer_end);
    if (iter != end_subgraph_map.end()) {
      const SubgraphPtr &pred_subgraph = iter->second;
      if (CouldReuse(subgraph, pred_subgraph, pld_subgraph_map)) {
        return pred_subgraph;
      }
    }
  }

  return nullptr;
}

int64_t AssignByDependencyPass::AssignNewStream(SubgraphPtr subgraph) {
  const string &engine_name = subgraph->engine_conf.id;
  int64_t max_parallel_num = subgraph->max_parallel_num;

  int64_t stream_id = 0;
  auto next_iter = engine_next_streams_.find(engine_name);
  if (next_iter != engine_next_streams_.end()) {
    stream_id = next_iter->second;
  }

  if (stream_id >= max_parallel_num) {
    stream_id = 0;
  }

  subgraph->stream_id = stream_id;
  engine_next_streams_[engine_name] = stream_id + 1;
  assigned_subgraphs_.emplace_back(subgraph);

  if ((stream_id + 1) > engine_stream_num_[engine_name]) {
    engine_stream_num_[engine_name] = stream_id + 1;
  }

  GELOGI("Subgraph %s assigns new temp stream %ld (engine: %s).", subgraph->name.c_str(), stream_id,
         engine_name.c_str());

  return stream_id;
}

void AssignByDependencyPass::UpdateAssignedSubgraphs(Context &context) {
  // If the default stream is valid, the first assigned stream will reuse the default stream id
  // and other streams use new id. To ensure that the id of the new stream is continuous,
  // we first subtract one from next_stream.
  int64_t to_be_updated_stream = kInvalidStream;
  if (context.default_stream != kInvalidStream) {
    context.next_stream--;
    to_be_updated_stream = context.next_stream;
  }

  // Update the starting stream id for each engine.
  int64_t &next_stream = context.next_stream;
  map<string, int64_t> engine_start_streams;
  for (const auto &item : engine_stream_num_) {
    int64_t stream_count = item.second;
    engine_start_streams[item.first] = next_stream;
    next_stream += stream_count;
  }

  // Update the subgraph streams assigned by engine.
  for (auto &subgraph : assigned_subgraphs_) {
    subgraph->stream_id += engine_start_streams[subgraph->engine_conf.id];
    if (subgraph->stream_id == to_be_updated_stream) {
      subgraph->stream_id = context.default_stream;
      GELOGI("Subgraph %s of engine %s reuses default stream %ld.", subgraph->name.c_str(),
             subgraph->engine_conf.id.c_str(), context.default_stream);
    } else {
      GELOGI("Stream of subgraph %s has been updated to %ld.", subgraph->name.c_str(), subgraph->stream_id);
    }
  }
}

void AssignByDependencyPass::UpdateReusedSubgraphs() {
  // Update streams for the subgraphs of reusing stream.
  for (const auto &item : reused_subgraphs_) {
    auto &cur_subgraph = item.first;
    auto &reused_graph = item.second;
    cur_subgraph->stream_id = reused_graph->stream_id;
    GELOGI("Stream of subgraph %s has been updated to %ld.", cur_subgraph->name.c_str(), cur_subgraph->stream_id);
  }
}

Status SingleStreamPass::Run(ComputeGraphPtr graph, const vector<SubgraphPtr> &subgraphs, Context &context) {
  // context.default_stream can be kInvalidStream only when graph is the root graph.
  int64_t new_stream = context.default_stream;
  if (new_stream == kInvalidStream) {
    new_stream = context.next_stream;
    ++context.next_stream;
  }

  for (const SubgraphPtr &subgraph : subgraphs) {
    if (!HasAssignedStream(*subgraph)) {
      const string &stream_label = subgraph->subgraph_info.GetStreamLabel();
      if (!stream_label.empty()) {
        GELOGE(INTERNAL_ERROR, "Stream labels are not supported (subgraph: %s, stream label: %s).",
               subgraph->name.c_str(), stream_label.c_str());
        return INTERNAL_ERROR;
      }
      subgraph->stream_id = new_stream;
    }
  }

  return SUCCESS;
}

Status NodeStreamUpdatePass::Run(ComputeGraphPtr graph, const vector<SubgraphPtr> &subgraphs, Context &context) {
  // Check if all subgraphs have been assigned a stream.
  for (const SubgraphPtr &subgraph : subgraphs) {
    const string &engine_name = subgraph->engine_conf.id;

    if (!IsEngineSkip(*subgraph) && !HasAssignedStream(*subgraph)) {
      GELOGE(INTERNAL_ERROR, "Subgraph %s has not yet been assigned a stream (engine: %s).", subgraph->name.c_str(),
             engine_name.c_str());
      return INTERNAL_ERROR;
    } else {
      GELOGI("Subgraph %s is assigned stream %ld (engine: %s).", subgraph->name.c_str(), subgraph->stream_id,
             engine_name.c_str());
    }
  }

  // Init the stream id of node.
  for (NodePtr &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    node->GetOpDesc()->SetStreamId(kInvalidStream);
  }

  // Set the stream id of the subgraph to the node.
  for (const SubgraphPtr &subgraph : subgraphs) {
    int64_t stream_id = subgraph->stream_id;
    const string &engine_name = subgraph->engine_conf.id;
    auto compute_graph = subgraph->subgraph_info.GetSubGraph();
    for (NodePtr &node : compute_graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node->GetOpDesc());
      if (node->GetOpDesc()->HasAttr(ATTR_NAME_RTS_LABEL_NODE)) {
        node->GetOpDesc()->SetStreamId(context.default_stream);
        GELOGD("Node %s of type %s in subgraph %s is assigned parent stream %ld (engine: %s).", node->GetName().c_str(),
               node->GetType().c_str(), subgraph->name.c_str(), context.default_stream, engine_name.c_str());
      } else if (IsEngineSkip(*subgraph) && node->GetInNodes().empty()) {
        GELOGD("Node %s of type %s in subgraph %s doesn't need to assign a stream (engine: %s).",
               node->GetName().c_str(), node->GetType().c_str(), subgraph->name.c_str(), engine_name.c_str());
      } else {
        node->GetOpDesc()->SetStreamId(stream_id);
        GELOGD("Node %s of type %s in subgraph %s is assigned stream %ld (engine: %s).", node->GetName().c_str(),
               node->GetType().c_str(), subgraph->name.c_str(), stream_id, engine_name.c_str());
      }
    }
  }

  return SUCCESS;
}

int64_t UpdateForSkippedEnginePass::GetSingleInoutStream(const NodePtr &node) const {
  set<int64_t> stream_ids;

  for (const auto &in_node : node->GetInAllNodes()) {
    GE_CHECK_NOTNULL_EXEC(in_node->GetOpDesc(), return kInvalidStream);
    int64_t stream_id = in_node->GetOpDesc()->GetStreamId();
    if (stream_id != kInvalidStream) {
      stream_ids.insert(stream_id);
    }
  }

  for (const auto &out_node : node->GetOutAllNodes()) {
    GE_CHECK_NOTNULL_EXEC(out_node->GetOpDesc(), return kInvalidStream);
    int64_t stream_id = out_node->GetOpDesc()->GetStreamId();
    if (stream_id != kInvalidStream) {
      stream_ids.insert(stream_id);
    }
  }

  if (stream_ids.size() == 1) {
    int64_t stream_id = *(stream_ids.begin());
    GELOGI("The stream of all input and output nodes of node %s (type: %s) is %ld.", node->GetName().c_str(),
           node->GetType().c_str(), stream_id);
    return stream_id;
  }

  return kInvalidStream;
}

Status UpdateForSkippedEnginePass::Run(ComputeGraphPtr graph, const vector<SubgraphPtr> &subgraphs, Context &context) {
  set<OpDescPtr> ops_without_label;

  // Check if subgraph is engine skipped and without stream label or not
  for (const SubgraphPtr &subgraph : subgraphs) {
    if (IsEngineSkip(*subgraph)) {
      auto compute_graph = subgraph->subgraph_info.GetSubGraph();
      for (NodePtr &node : compute_graph->GetDirectNode()) {
        auto op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        auto stream_id = op_desc->GetStreamId();
        if ((stream_id != kInvalidStream) && !HasStreamLabel(*subgraph)) {
          ops_without_label.emplace(op_desc);
        }
      }
    }
  }

  // Try reassign the stream id
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    int64_t stream_id = op_desc->GetStreamId();
    if (ops_without_label.find(op_desc) != ops_without_label.end()) {
      if (AreAllPredStreamsInvalid(node) && op_desc->GetSubgraphInstanceNames().empty()) {
        op_desc->SetStreamId(kInvalidStream);
        GELOGI("Node %s of type %s reassign to stream %ld from stream %ld.", node->GetName().c_str(),
               node->GetType().c_str(), kInvalidStream, stream_id);
      } else if (!node->GetOutAllNodes().empty()) {
        int64_t inout_stream = GetSingleInoutStream(node);
        if (inout_stream != kInvalidStream) {
          op_desc->SetStreamId(inout_stream);
          GELOGI("Node %s of type %s reassign to stream %ld from stream %ld.", node->GetName().c_str(),
                 node->GetType().c_str(), inout_stream, stream_id);
        }
      }
    }
  }

  return SUCCESS;
}

bool UpdateForSkippedEnginePass::AreAllPredStreamsInvalid(const NodePtr &node) const {
  for (const auto &pre_node : node->GetInAllNodes()) {
    auto pre_node_desc = pre_node->GetOpDesc();
    if (pre_node_desc != nullptr) {
      int64_t stream_id = pre_node_desc->GetStreamId();
      if (stream_id != kInvalidStream) {
        return false;
      }
    }
  }
  return true;
}

Status AllReduceParallelPass::Run(ComputeGraphPtr graph, const vector<SubgraphPtr> &subgraphs, Context &context) {
  if (!context.enable_hcom_parallel) {
    return NOT_CHANGED;
  }

  GELOGI("AllReduceParallelPass is enabled.");
  GE_DUMP(graph, "BeforeAllReduceParallel");

  // All successors of HcomAllReduce.
  set<NodePtr> all_reduce_succs;

  for (const NodePtr &node : graph->GetDirectNode()) {
    if (!IsHcomNode(node->GetType()) ||
        (node->GetInDataNodes().size() <= 1)) {
      continue;
    }

    string reduce_stream_label;
    GE_CHECK_NOTNULL(node->GetOpDesc());
    (void)AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, reduce_stream_label);

    set<NodePtr> cur_nodes = {node};
    while (!cur_nodes.empty()) {
      set<NodePtr> all_out_data_nodes;
      for (auto &curr_node : cur_nodes) {
        for (const NodePtr &out_node : curr_node->GetOutDataNodes()) {
          string out_stream_label;
          GE_CHECK_NOTNULL(out_node->GetOpDesc());
          (void)AttrUtils::GetStr(out_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, out_stream_label);
          // normally, Allreduce do not have streamLabel. when in horovod scenario Allreduce will have streamLabel
          bool isSuccessorParallel =
              (out_stream_label == reduce_stream_label) || (!reduce_stream_label.empty() && out_stream_label.empty());
          if (isSuccessorParallel) {
            all_reduce_succs.emplace(out_node);
            all_out_data_nodes.emplace(out_node);
          }
        }
      }
      cur_nodes = all_out_data_nodes;
    }
  }

  map<int64_t, int64_t> old_stream_to_new;
  for (const NodePtr &node : all_reduce_succs) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    auto old_stream = node->GetOpDesc()->GetStreamId();
    if (old_stream != kInvalidStream) {
      int64_t new_stream = kInvalidStream;
      auto iter = old_stream_to_new.find(old_stream);
      if (iter != old_stream_to_new.end()) {
        new_stream = iter->second;
      } else {
        new_stream = context.next_stream;
        context.next_stream++;
        old_stream_to_new.emplace(old_stream, new_stream);
      }

      if (!IsHcomNode(node->GetType())) {
        GELOGI("Stream of node %s has been updated from %ld to %ld.", node->GetName().c_str(), old_stream, new_stream);
        node->GetOpDesc()->SetStreamId(new_stream);
      }
    }
  }

  return !all_reduce_succs.empty() ? SUCCESS : NOT_CHANGED;
}

bool AllReduceParallelPass::IsHcomNode(const std::string& node_type) {
  return (node_type == HCOMALLREDUCE || node_type == HVDCALLBACKALLREDUCE);
}


LogicalStreamAllocator::LogicalStreamAllocator(const map<string, SchedulerConf> &scheduler_confs,
                                               const map<string, int> &max_parallel_num)
    : scheduler_confs_(scheduler_confs), max_parallel_num_(max_parallel_num) {}

void LogicalStreamAllocator::EnableSingleStream(bool enable) { context_.enable_single_stream = enable; }

void LogicalStreamAllocator::EnableHcomParallel(bool enable) { context_.enable_hcom_parallel = enable; }

Status LogicalStreamAllocator::Assign(const ComputeGraphPtr &root_graph, const Graph2SubGraphInfoList &subgraph_map,
                                      int64_t &stream_num) {
  GE_CHECK_NOTNULL(root_graph);

  map<string, EngineConfPtr> engine_confs;
  GE_TIMESTAMP_START(InitEngineConfs);
  for (const auto &item : scheduler_confs_) {
    const SchedulerConf &scheduler = item.second;
    for (const auto &engine_pair : scheduler.cal_engines) {
      EngineConfPtr engine_conf = engine_pair.second;
      if (engine_conf != nullptr) {
        engine_confs[engine_pair.first] = engine_conf;
      }
    }
  }
  GE_TIMESTAMP_END(InitEngineConfs, "GraphBuilder::AssignStreamInitEngineConfs");

  Status status = DoAssign(root_graph, subgraph_map, engine_confs);
  if (status != SUCCESS) {
    GELOGE(status, "Assign streams failed.");
    return status;
  }

  vector<ComputeGraphPtr> subgraphs = root_graph->GetAllSubgraphs();
  for (const ComputeGraphPtr &subgraph : subgraphs) {
    Status status = DoAssign(subgraph, subgraph_map, engine_confs);
    if (status != SUCCESS) {
      GELOGE(status, "Assign streams failed.");
      return status;
    }
  }

  RefreshContinuousStreams(root_graph);

  stream_num = context_.next_stream;
  GELOGI("Assigned logical stream num: %ld.", stream_num);

  return SUCCESS;
}

Status LogicalStreamAllocator::DoAssign(const ComputeGraphPtr &graph, const Graph2SubGraphInfoList &subgraph_map,
                                        const map<string, EngineConfPtr> &engine_confs) {
  GE_CHECK_NOTNULL(graph);

  NodePtr parent_node = graph->GetParentNode();
  if ((parent_node == nullptr) || (parent_node->GetOpDesc() == nullptr)) {
    context_.default_stream = kInvalidStream;
  } else {
    context_.default_stream = parent_node->GetOpDesc()->GetStreamId();
  }

  auto iter = subgraph_map.find(graph);
  if (iter == subgraph_map.end()) {
    GELOGE(FAILED, "Graph %s not found.", graph->GetName().c_str());
    return FAILED;
  }

  const vector<SubGraphInfoPtr> &subgraph_info_list = iter->second;
  vector<SubgraphPtr> subgraphs;
  GE_TIMESTAMP_START(ConvertSubgraphs);
  Status status = ConvertSubgraphs(subgraph_info_list, engine_confs, subgraphs);
  GE_TIMESTAMP_END(ConvertSubgraphs, "GraphBuilder::AssignStreamConvertSubgraphs");
  if (status != SUCCESS) {
    GELOGE(status, "Create subgraphs failed.");
    return status;
  }

  GELOGD("Subgraphs of graph %s", graph->GetName().c_str());
  for (const auto &subgraph : subgraphs) {
    if (subgraph != nullptr) {
      GELOGD("subgraph: %s", subgraph->name.c_str());
    }
  }

  return RunPasses(graph, subgraphs);
}

Status LogicalStreamAllocator::ConvertSubgraphs(const vector<SubGraphInfoPtr> &subgraph_infos,
                                                const map<string, EngineConfPtr> &engine_confs,
                                                vector<SubgraphPtr> &subgraphs) {
  for (auto &subgraph_info : subgraph_infos) {
    GE_CHECK_NOTNULL(subgraph_info);

    string subgraph_name;
    ComputeGraphPtr computer_graph = subgraph_info->GetSubGraph();
    if (computer_graph != nullptr) {
      subgraph_name = computer_graph->GetName();
    }

    const string &engine_name = subgraph_info->GetEngineName();
    auto engine_conf_iter = engine_confs.find(engine_name);
    if ((engine_conf_iter == engine_confs.end()) || (engine_conf_iter->second == nullptr)) {
      GELOGE(INTERNAL_ERROR, "Engine conf of subgraph %s not found (engine name: %s).", subgraph_name.c_str(),
             engine_name.c_str());

      return INTERNAL_ERROR;
    }

    SubgraphPtr subgraph = MakeShared<Subgraph>(*subgraph_info, *engine_conf_iter->second);
    GE_CHECK_NOTNULL(subgraph);
    subgraph->name = subgraph_name;

    auto parallel_iter = max_parallel_num_.find(engine_name);
    if (parallel_iter != max_parallel_num_.end()) {
      subgraph->max_parallel_num = parallel_iter->second;
    }

    subgraphs.emplace_back(subgraph);
  }

  return SUCCESS;
}

Status LogicalStreamAllocator::RunPasses(const ComputeGraphPtr &graph, const vector<SubgraphPtr> &subgraphs) {
  vector<LogicalStreamPassPtr> passes;

  if (context_.enable_single_stream) {
    passes.emplace_back(MakeShared<SingleStreamPass>());
    passes.emplace_back(MakeShared<NodeStreamUpdatePass>());
    passes.emplace_back(MakeShared<UpdateForSkippedEnginePass>());
  } else {
    passes.emplace_back(MakeShared<AssignByLabelPass>());
    passes.emplace_back(MakeShared<IndependentStreamPass>());
    passes.emplace_back(MakeShared<AssignByDependencyPass>());
    passes.emplace_back(MakeShared<NodeStreamUpdatePass>());
    passes.emplace_back(MakeShared<AllReduceParallelPass>());
    passes.emplace_back(MakeShared<UpdateForSkippedEnginePass>());
  }

  for (auto &pass : passes) {
    GE_CHECK_NOTNULL(pass);

    Status status = pass->Run(graph, subgraphs, context_);
    if (status == SUCCESS) {
      GELOGD("Stream pass %s return SUCCESS.", pass->GetName().c_str());
    } else if (status == NOT_CHANGED) {
      GELOGD("Stream pass %s return NOT_CHANGED.", pass->GetName().c_str());
    } else {
      GELOGE(status, "Stream pass %s failed.", pass->GetName().c_str());
      return status;
    }
  }

  return SUCCESS;
}

void LogicalStreamAllocator::RefreshContinuousStreams(const ComputeGraphPtr &graph) {
  int64_t stream_num = context_.next_stream;
  vector<bool> stream_has_node(stream_num);


  for (const NodePtr &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    if (node != nullptr) {
      auto op_desc = node->GetOpDesc();
      if (op_desc != nullptr) {
        int64_t stream_id = op_desc->GetStreamId();
        if ((stream_id != kInvalidStream) && (stream_id < stream_num)) {
          stream_has_node[stream_id] = true;
        }
      }
    }
  }

  context_.next_stream = 0;
  vector<int64_t> old_to_new_streams(stream_num, kInvalidStream);
  for (size_t old_stream = 0; old_stream < stream_has_node.size(); old_stream++) {
    if (stream_has_node[old_stream]) {
      old_to_new_streams[old_stream] = context_.next_stream;
      context_.next_stream++;
    }
  }

  for (const NodePtr &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    auto op_desc = node->GetOpDesc();
    if (op_desc != nullptr) {
      int64_t stream_id = op_desc->GetStreamId();
      if ((stream_id != kInvalidStream) && (stream_id < stream_num)) {
        op_desc->SetStreamId(old_to_new_streams[stream_id]);
      }
    }
  }
}
}  // namespace ge
