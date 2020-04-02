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

#ifndef GE_GRAPH_PREPROCESS_MULTI_BATCH_COPY_GRAPH_H_
#define GE_GRAPH_PREPROCESS_MULTI_BATCH_COPY_GRAPH_H_
#include <vector>
#include <map>
#include <queue>

#include "external/ge/ge_api_error_codes.h"

#include "graph/compute_graph.h"

namespace ge {
namespace multibatch {
Status ProcessMultiBatch(ComputeGraphPtr &graph);

enum NodeStatus {
  kNodeInBatchBranch,
  kNodeOutBatchBranch,
  kNodeStartNode,
};

class MultiBatchGraphCopyer {
 public:
  explicit MultiBatchGraphCopyer(ComputeGraphPtr &graph) : graph_(graph) {}
  ~MultiBatchGraphCopyer() = default;

  void AddShape(const std::vector<int64_t> &shape) { shapes_.emplace_back(shape); }

  Status CopyGraph();

 private:
  Status Init();
  Status CheckArguments();

  // add nodes functions
  Status CreateNewNodes();

  NodePtr InsertShapeDataNode();
  Status InsertSwitchNForData(const NodePtr &data);
  Status UpdateMaxShapeToData(const NodePtr &data);

  Status InsertMergeForEdgeNode(const NodePtr &node);
  /**
   * Insert a merge node for src node `node` on output index `index`. The merge node will be used to merge all nodes
   * in batch-branch to one output to the node out of the batch-branch.
   * Cond 1: If the `index` is -1, then the src node link a data edge(at output 0) to the merge node,
   * Cond 2: In condition 1, if the src node does not have any data output, we create a const node after it,
   * the result like this:
   * src_node ---------> const_for_src_node --------> merge
   *           control                        data
   * Cond 3: If the src node is a data-like node, the SwitchN after it will be link to the merge node.
   * @param node
   * @param index
   * @return
   */
  NodePtr InsertMergeNode(const NodePtr &node, int index);
  Status CopyNodeInBatchBranch(const NodePtr &node);

  // link edges functions
  Status LinkEdges();
  Status LinkDataToSwitchN(const NodePtr &data);
  Status LinkToMerge(const NodePtr &node);
  Status LinkToNodeInBranch(const NodePtr &node);
  Status LinkToNodeOutBranch(const NodePtr &node);
  Status LinkDataToMerge(const NodePtr &data, const NodePtr &merge);
  Status LinkNodeToMerge(const NodePtr &node, int out_index, const NodePtr &merge);
  Status CopyInDataEdges(const NodePtr &origin_node, int batch_num, const NodePtr &copyed_node);
  Status CopyInControlEdges(const NodePtr &node, int batch_num, const NodePtr &copyed_node);

  bool IsInBatchBranch(const NodePtr &node);
  NodeStatus GetNodeStatus(const NodePtr &node);
  Status CheckCopyResult(const std::vector<NodePtr> &start_nodes);

  // arguments
  ComputeGraphPtr graph_;
  std::vector<std::vector<int64_t>> shapes_;

  // the shape data node created
  NodePtr shape_data_;

  // all nodes in the origin graph
  std::vector<NodePtr> origin_all_nodes_;

  // all data nodes in the origin graph
  std::vector<NodePtr> origin_data_nodes_;

  // the nodes in-batch-branch, and the nodes copyed by shapes
  std::map<Node *, std::vector<NodePtr>> nodes_to_batch_nodes_;

  // the data nodes, and the SwitchN nodes inserted after it
  std::map<Node *, NodePtr> data_nodes_to_switchn_;

  // the nodes on the in/out-batch-branch edge, and the merge nodes inserted after it
  std::map<Node *, std::vector<NodePtr>> nodes_to_merge_nodes_;
};
}  // namespace multibatch
}  // namespace ge
#endif  // GE_GRAPH_PREPROCESS_MULTI_BATCH_COPY_GRAPH_H_
