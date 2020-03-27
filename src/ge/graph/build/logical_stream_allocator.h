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

#ifndef GE_GRAPH_BUILD_LOGICAL_STREAM_ALLOCATOR_H_
#define GE_GRAPH_BUILD_LOGICAL_STREAM_ALLOCATOR_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "engine_manager/dnnengine_manager.h"
#include "graph/manager/graph_manager_utils.h"

namespace ge {
// Define default fuctions for stream passes.
#define STREAM_PASS_DEFAULT_FUNC(CLASS)  \
  CLASS() : LogicalStreamPass(#CLASS) {} \
  ~CLASS() override = default;           \
  CLASS(const CLASS &) = delete;         \
  CLASS &operator=(const CLASS &) = delete

static const int64_t kInvalidStream = -1;

// Base stream class.
class LogicalStreamPass {
 public:
  static const int64_t kDefaultMaxParalleNum = 1;

  struct Subgraph;
  using SubgraphPtr = std::shared_ptr<Subgraph>;

  struct Subgraph {
    string name;
    int64_t stream_id = kInvalidStream;

    const SubGraphInfo &subgraph_info;
    const EngineConf &engine_conf;
    int64_t max_parallel_num = kDefaultMaxParalleNum;

    SubgraphPtr reused_subgraph = nullptr;

    Subgraph(const SubGraphInfo &subgraph_info, const EngineConf &engine_conf)
        : subgraph_info(subgraph_info), engine_conf(engine_conf) {}
  };

  struct Context {
    // Next stream id.
    int64_t next_stream = 0;
    bool hcom_parallel = false;
  };

  explicit LogicalStreamPass(const std::string &name);
  LogicalStreamPass(const LogicalStreamPass &) = delete;
  LogicalStreamPass &operator=(const LogicalStreamPass &) = delete;
  virtual ~LogicalStreamPass() = default;

  const std::string &GetName() const;
  virtual Status Run(ComputeGraphPtr whole_graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) = 0;

 protected:
  bool IsEngineSkip(const Subgraph &subgraph) const;
  bool IsEngineAttach(const Subgraph &subgraph) const;
  bool IsEngineIndependent(const Subgraph &subgraph) const;
  bool HasStreamLabel(const Subgraph &subgraph) const;
  bool HasAssignedStream(const Subgraph &subgraph) const;

 private:
  std::string name_;
};

using LogicalStreamPassPtr = std::shared_ptr<LogicalStreamPass>;

// Allocate streams by label.
class AssignByLabelPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(AssignByLabelPass);
  Status Run(ComputeGraphPtr whole_graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
};

// Engines such as hccl require independent Stream.
class IndependentStreamPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(IndependentStreamPass);
  Status Run(ComputeGraphPtr whole_graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
};

// Reuse streams or assign new streams based on dependencies.
class AssignByDependencyPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(AssignByDependencyPass);
  Status Run(ComputeGraphPtr whole_graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;

 private:
  void InitEndSubgraphMap(const std::vector<SubgraphPtr> &subgraphs, std::map<NodePtr, SubgraphPtr> &end_subgraph_map);
  void InitPldSubgraphMap(const std::vector<SubgraphPtr> &subgraphs, std::map<NodePtr, SubgraphPtr> &pld_subgraph_map);

  SubgraphPtr GetReusableSubgraph(const SubgraphPtr &subgraph, const std::map<NodePtr, SubgraphPtr> &end_subgraph_map,
                                  const std::map<NodePtr, SubgraphPtr> &pld_subgraph_map);

  int64_t AssignNewStream(SubgraphPtr subgraph);

  void UpdateAssignedSubgraphs(Context &context);
  void UpdateReusedSubgraphs();

  bool CouldReuse(const SubgraphPtr &subgraph, const SubgraphPtr &pred_subgraph,
                  const std::map<NodePtr, SubgraphPtr> &pld_subgraph_map);

  // <engine name, next stream id>
  std::map<std::string, int64_t> engine_next_streams_;

  // <engine name, stream num>
  std::map<std::string, int64_t> engine_stream_num_;

  // Subgraphs of assign stream by engine
  std::set<SubgraphPtr> assigned_subgraphs_;

  // <current subgraph, reused subgraph>
  std::vector<std::pair<SubgraphPtr, SubgraphPtr>> reused_subgraphs_;
};

// Update the stream of subgraphs to nodes.
class NodeStreamUpdatePass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(NodeStreamUpdatePass);
  Status Run(ComputeGraphPtr whole_graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;

 private:
  /// Optimize for case like:
  ///  NodeA(stream1) -> Const(stream2) -> NodeB(stream1)
  /// To case:
  ///  NodeA(stream1) -> Const(stream1) -> NodeB(stream1)
  /// Which could reduce event number (Const could be other type which belong to skipped engine subgraph)
  Status UpdateForSkippedEngine(const ComputeGraphPtr &whole_graph, const std::vector<SubgraphPtr> &subgraphs);

  int64_t GetSingleInoutStream(const NodePtr &node) const;
  // Judge if all predecessors' streams of node are INVALID_STREAM
  bool AreAllPredStreamsInvalid(const NodePtr &node) const;
};

// AllReduce and backward operators execute in parallel.
class AllReduceParallelPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(AllReduceParallelPass);
  Status Run(ComputeGraphPtr whole_graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
};

// Assign logical streams which is not limited by the number of tasks.
class LogicalStreamAllocator {
  using Subgraph = LogicalStreamPass::Subgraph;
  using SubgraphPtr = LogicalStreamPass::SubgraphPtr;
  using Context = LogicalStreamPass::Context;

 public:
  LogicalStreamAllocator(const std::map<std::string, SchedulerConf> &scheduler_confs,
                         const std::map<std::string, int> &max_parallel_num, bool hcom_parallel = false);
  LogicalStreamAllocator(const LogicalStreamAllocator &) = delete;
  LogicalStreamAllocator &operator=(const LogicalStreamAllocator &) = delete;
  ~LogicalStreamAllocator() = default;

  Status Assign(const ComputeGraphPtr &whole_graph, const std::vector<SubGraphInfoPtr> &subgraphs, int64_t &stream_num);

 private:
  Status ConvertSubgraphs(const std::vector<SubGraphInfoPtr> &subgraph_infos,
                          const std::map<std::string, EngineConfPtr> &engine_confs,
                          std::vector<SubgraphPtr> &subgraphs);
  Status RunPasses(const ComputeGraphPtr &whole_graph, const std::vector<SubgraphPtr> &subgraphs, int64_t &stream_num);

  const std::map<std::string, SchedulerConf> &scheduler_confs_;
  const std::map<std::string, int> &max_parallel_num_;
  Context context_;
};
}  // namespace ge

#endif  // GE_GRAPH_BUILD_LOGICAL_STREAM_ALLOCATOR_H_
