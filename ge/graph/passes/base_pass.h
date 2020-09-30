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

#ifndef GE_GRAPH_PASSES_BASE_PASS_H_
#define GE_GRAPH_PASSES_BASE_PASS_H_

#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
enum NodePassOption {
  // if there is a sub graph on the node, the pass on the node will do:
  // Pass(node) -> pass all sub graphs on the node -> Pass(node)
  // when pass the node for the second time, the kOptimizeAfterSubGraph will be set as a flag key
  kOptimizeAfterSubGraph,

  // add new options before kOptionEnd
  kOptionEnd
};

class BaseNodePass {
 public:
  ///
  /// Optimize on one node. the function can add nodes to the graph, change
  /// connections between nodes while optimizing or remove nodes from the graph.
  /// @param node
  /// @return
  ///
  virtual Status Run(NodePtr &node) = 0;

  virtual ~BaseNodePass() = default;

  std::unordered_set<NodePtr> GetNodesNeedRePass() { return nodes_need_re_pass_; }

  std::unordered_set<NodePtr> GetNodesDeleted() { return nodes_deleted_; }

  void SetOption(NodePassOption option, const std::string &value) { options_[option] = value; }

  void ClearOptions() { options_.clear(); }

  void init() {
    nodes_need_re_pass_.clear();
    nodes_deleted_.clear();
  }

 protected:
  Status IsolateAndDeleteNode(NodePtr &node, const std::vector<int> &io_map);

  Status IsolateAndDeleteNode(NodePtr &node, const std::initializer_list<int> &io_map) {
    return IsolateAndDeleteNode(node, std::vector<int>(io_map));
  }

  ///
  /// Add a node to be optimized again. If you add a new node to the graph, or
  /// change a node connections, and you want to make sure the node will be
  /// optimized by other passes, call this function.
  /// @param node
  ///
  void AddRePassNode(NodePtr &node) { nodes_need_re_pass_.insert(node); }

  ///
  /// Add a node and it's input/output data nodes to be optimized again.
  /// @param node
  ///
  void AddRePassNodesWithInOut(NodePtr &node) {
    AddRePassNode(node);
    auto out_nodes = node->GetOutNodes();
    for (auto &out_node : out_nodes) {
      AddRePassNode(out_node);
    }
    auto in_nodes = node->GetInNodes();
    for (auto &in_node : in_nodes) {
      AddRePassNode(in_node);
    }
  }

  ///
  /// If you deleted a node from the graph, especially current node. The remain
  /// iterate passes will continue process on the deleted node(if it can be
  /// reached by edge connections) till the last one. Obviously it is a waste of
  /// time. You can add the deleted nodes by calling this function, to stop the
  /// next iterations.
  /// @param node
  ///
  void AddNodeDeleted(const NodePtr &node) { nodes_deleted_.insert(node); }

  bool OptionExists(NodePassOption option) { return options_.count(option) > 0; }

 private:
  std::unordered_set<NodePtr> nodes_need_re_pass_;
  std::unordered_set<NodePtr> nodes_deleted_;
  std::map<NodePassOption, std::string> options_;
};

using NamesToPass = std::vector<std::pair<std::string, BaseNodePass *>>;

class GEPass {
 public:
  explicit GEPass(ComputeGraphPtr &graph) : graph_(graph), root_graph_(graph), depth_(1) {}
  virtual ~GEPass() = default;
  Status Run(const NamesToPass &names_to_passes);

 private:
  GEPass(ComputeGraphPtr &graph, ComputeGraphPtr &root_graph, int depth)
      : graph_(graph), root_graph_(root_graph), depth_(depth) {}
  Status RunPassesOneGraph(const NamesToPass &names_to_passes);
  Status RunPassesOnSubGraph(const NodePtr &node, const NamesToPass &names_to_passes, bool &has_sub_graph);
  ComputeGraphPtr graph_;
  ComputeGraphPtr root_graph_;
  int depth_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_BASE_PASS_H_
