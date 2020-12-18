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

#ifndef INC_GRAPH_COMPUTE_GRAPH_H_
#define INC_GRAPH_COMPUTE_GRAPH_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <deque>
#include "detail/attributes_holder.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/range_vistor.h"

namespace ge {
class Node;
using NodePtr = std::shared_ptr<Node>;
class Edge;
using EdgePtr = std::shared_ptr<Edge>;

class InDataAnchor;
using InDataAnchorPtr = std::shared_ptr<InDataAnchor>;

class OutDataAnchor;
using OutDataAnchorPtr = std::shared_ptr<OutDataAnchor>;

class ControlAnchor;
using ControlAnchorPtr = std::shared_ptr<ControlAnchor>;
class InControlAnchor;
using InControlAnchorPtr = std::shared_ptr<InControlAnchor>;
class OutControlAnchor;
using OutControlAnchorPtr = std::shared_ptr<OutControlAnchor>;
class GeAttrValue;
using AttrValuePtr = std::shared_ptr<GeAttrValue>;
using ConstComputeGraph = const ComputeGraph;

class OperatorImpl;
using OperatorImplPtr = std::shared_ptr<OperatorImpl>;

class ComputeGraph : public std::enable_shared_from_this<ComputeGraph>, public AttrHolder {
  friend class GraphUtils;

 public:
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstComputeGraph>>;

  explicit ComputeGraph(const std::string &name);
  ~ComputeGraph() override;

  std::string GetName() const;
  void SetName(const std::string &name);

  using AttrHolder::DelAttr;
  using AttrHolder::GetAttr;
  using AttrHolder::HasAttr;
  using AttrHolder::SetAttr;

  size_t GetAllNodesSize() const;
  Vistor<NodePtr> GetAllNodes() const;
  // is_unknown_shape: false, same with GetAllNodes func
  // is_unknown_shape: true, same with GetDirectNodes func
  Vistor<NodePtr> GetNodes(bool is_unknown_shape) const;
  size_t GetDirectNodesSize() const;
  Vistor<NodePtr> GetDirectNode() const;
  Vistor<NodePtr> GetInputNodes() const;
  Vistor<NodePtr> GetOutputNodes() const;

  NodePtr FindNode(const std::string &name) const;
  NodePtr FindFirstNodeMatchType(const std::string &name) const;
  /*lint -e504*/
  // AddNode with NodePtr
  NodePtr AddNode(NodePtr node);
  NodePtr AddNode(OpDescPtr op);
  NodePtr AddNode(OpDescPtr op, int64_t id);    // for unserialize
  NodePtr AddNodeFront(NodePtr node);
  NodePtr AddNodeFront(const OpDescPtr &op);
  NodePtr AddInputNode(NodePtr node);
  NodePtr AddOutputNode(NodePtr node);
  NodePtr AddOutputNodeByIndex(NodePtr node, int32_t index);

  graphStatus RemoveNode(const NodePtr &node);
  graphStatus RemoveInputNode(const NodePtr &node);
  graphStatus RemoveOutputNode(const NodePtr &node);
  graphStatus RemoveConstInput(const NodePtr &node);

  /// Add a subgraph to this graph. The subgraph must has a parent graph and parent node,
  /// which means the member functions `SetParentGraph` and `SetParentNode` of the subgraph
  /// must be called before add it to the root graph. and subgraph->GetParentNode()->GetOwnerGraph()
  /// must equal to subgraph->GetOwnerGraph().
  /// The subgraphs can only be added to a *root graph*. A root graph is a graph without any parent graph.
  /// The subgraph's name SHOULD(not must) be the same as the parameter `name`
  graphStatus AddSubgraph(const std::string &name, const std::shared_ptr<ComputeGraph> &subgraph);
  graphStatus AddSubgraph(const std::shared_ptr<ComputeGraph> &subgraph);

  void RemoveSubgraph(const std::string &name);
  void RemoveSubgraph(const std::shared_ptr<ComputeGraph> &subgraph);

  std::shared_ptr<ComputeGraph> GetSubgraph(const std::string &name) const;
  std::vector<std::shared_ptr<ComputeGraph>> GetAllSubgraphs() const;

  // obsolete
  std::shared_ptr<ComputeGraph> AddSubGraph(std::shared_ptr<ComputeGraph> sub_graph);
  // obsolete
  graphStatus RemoveSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph);

  ///
  /// @brief Update input-mapping
  /// @param [in] input_mapping : index_of_cur_graph_node_input -> index_of_new_graph_node_input
  /// @return graphStatus
  ///
  graphStatus UpdateInputMapping(const std::map<uint32_t, uint32_t> &input_mapping);

  ///
  /// @brief Update output-mapping
  /// @param [in] output_mapping : index_of_cur_graph_node_output -> index_of_new_graph_node_output
  /// @return graphStatus
  ///
  graphStatus UpdateOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping);

  graphStatus TopologicalSorting();
  bool IsValid() const;
  void InValid() { is_valid_flag_ = false; }
  void Dump() const;

  void Swap(ComputeGraph &graph);

  graphStatus IsolateNode(const NodePtr &node);
  graphStatus Verify();
  graphStatus InferShape();
  graphStatus InferOriginFormat();
  graphStatus InferShapeInNeed();
  graphStatus InsertEventNodes();
  bool operator==(const ComputeGraph &r_compute_graph) const;

  /*lint +e504*/
  const std::map<std::vector<std::string>, std::vector<std::string>> &GetShareParamLayer() const {
    return params_share_map_;
  }

  void SetShareParamLayer(const std::map<std::vector<std::string>, std::vector<std::string>> params_share_map) {
    params_share_map_ = params_share_map;
  }

  void SetInputsOrder(const std::vector<std::string> &inputs_order) { inputs_order_ = inputs_order; }

  void SetGraphOutNodes(std::map<std::string, std::vector<int32_t>> out_nodes_map) { out_nodes_map_ = out_nodes_map; }

  void AppendGraphOutNodes(std::map<std::string, std::vector<int32_t>> out_nodes_map) {
    for (auto &item : out_nodes_map) {
      (void)out_nodes_map_.emplace(item.first, item.second);
    }
  }

  shared_ptr<ComputeGraph> GetParentGraph();
  void SetParentGraph(const shared_ptr<ComputeGraph> &parent);
  shared_ptr<Node> GetParentNode();
  void SetParentNode(const shared_ptr<Node> &parent);

  const std::map<std::string, std::vector<int32_t>> &GetGraphOutNodes() const { return out_nodes_map_; }

  void SetOrigGraph(ComputeGraphPtr orig_graph) { origGraph_ = orig_graph; }

  ComputeGraphPtr GetOrigGraph(void) { return origGraph_; }
  void SetOutputSize(uint32_t size) { output_size_ = size; }
  uint32_t GetOutputSize() const { return output_size_; }
  void SetInputSize(uint32_t size) { input_size_ = size; }
  uint32_t GetInputSize() const { return input_size_; }

  // false: known shape  true: unknow shape
  bool GetGraphUnknownFlag() const { return is_unknown_shape_graph_; }
  void SetGraphUnknownFlag(bool flag) { is_unknown_shape_graph_ = flag; }

  ///
  /// Set is need train iteration.
  /// If set true, it means this graph need to be run iteration some
  /// times(according variant "npu_runconfig/iterations_per_loop").
  /// @param need_iteration is need iteration
  ///
  void SetNeedIteration(bool need_iteration) { need_iteration_ = need_iteration; }

  void SetUserDefOutput(const std::string &output_name);

  const std::string GetOutput();

  ///
  /// Get is need train iteration.
  /// @return is need iteration
  ///
  bool GetNeedIteration() const { return need_iteration_; }

  void SetGraphOpName(const std::map<uint32_t, std::string> &op_name_map) { op_name_map_ = op_name_map; }
  const std::map<uint32_t, std::string> &GetGraphOpName() const { return op_name_map_; }

  const std::map<OperatorImplPtr, NodePtr> &GetAllNodesInfo() const;

  void SetAllNodesInfo(const std::map<OperatorImplPtr, NodePtr> &nodes) { all_nodes_infos_ = nodes; }

  void SetGraphOutNodesInfo(std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info) {
    output_nodes_info_ = out_nodes_info;
  }

  void AppendGraphOutNodesInfo(std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info) {
    output_nodes_info_.insert(output_nodes_info_.end(), out_nodes_info.begin(), out_nodes_info.end());
  }

  const std::vector<std::pair<NodePtr, int32_t>> &GetGraphOutNodesInfo() const { return output_nodes_info_; }

  void SetGraphTargetNodesInfo(const std::vector<NodePtr> &target_nodes_info) {
    target_nodes_info_ = target_nodes_info;
  }
  const std::vector<NodePtr> &GetGraphTargetNodesInfo() const { return target_nodes_info_; }

  void SetSessionID(uint64_t session_id) { session_id_ = session_id; }
  uint64_t GetSessionID() const { return session_id_; }

  void SetGraphID(uint32_t graph_id) { graph_id_ = graph_id; }
  uint32_t GetGraphID() const { return graph_id_; }

  void SaveDataFormat(ge::Format data_format) { data_format_ = data_format; }
  ge::Format GetDataFormat() const { return data_format_; }
  bool IsSummaryGraph() const { return is_summary_graph_; }
  void SetSummaryFlag(bool is_summary_graph) { is_summary_graph_ = is_summary_graph; }
  // Graph Before BFE
  ComputeGraphPtr origGraph_;

 protected:
  ProtoAttrMapHelper MutableAttrMap() override;
  ConstProtoAttrMapHelper GetAttrMap() const override;

 private:
  graphStatus DFSTopologicalSorting(std::vector<NodePtr> &node_vec, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::vector<NodePtr> &stack, bool reverse);
  graphStatus BFSTopologicalSorting(std::vector<NodePtr> &node_vec, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::deque<NodePtr> &stack);
  graphStatus CollectBreadthOutNode(const NodePtr &node, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::map<string, NodePtr> &breadth_node_map);
  /// nodes like : (a) <--- (c) ---> (b)
  /// node a and b have only one parent node c, and a is connected to c firstly
  /// topo order of DFS is `c, b, a` with `dfs_reverse=false` as default
  /// in same case, user could get `c, a, b` with `dfs_reverse=true`
  graphStatus TopologicalSortingGraph(bool dfs_reverse = false);
  graphStatus SortNodes(std::vector<NodePtr> &stack, std::map<NodePtr, uint32_t> &mapInEdgeNum);
  Vistor<NodePtr> AllGraphNodes(std::vector<std::shared_ptr<ComputeGraph>> &subgraphs) const;
  size_t GetInEdgeSize(const NodePtr &node);
  size_t GetOutEdgeSize(const NodePtr &node);
  graphStatus RemoveExtraOutEdge(const NodePtr &node);
  bool GraphMembersAreEqual(const ComputeGraph &r_graph) const;
  bool GraphAttrsAreEqual(const ComputeGraph &r_graph) const;
  bool VectorInputNodePtrIsEqual(const std::vector<NodePtr> &r_node_ptr_vector,
                                 const std::vector<NodePtr> &l_node_ptr_vector) const;

  void SetNodesOwner();

  friend class ModelSerializeImp;
  friend class GraphDebugImp;
  friend class OnnxUtils;
  friend class TuningUtils;

  std::string name_;
  uint32_t graph_id_ = 0;
  ProtoAttrMapHelper attrs_;
  std::vector<NodePtr> nodes_;
  std::map<OperatorImplPtr, NodePtr> all_nodes_infos_;
  std::vector<NodePtr> target_nodes_info_;

  std::vector<NodePtr> input_nodes_;
  std::vector<std::string> inputs_order_;
  uint32_t input_size_ = 1;
  std::map<std::string, std::vector<int32_t>> out_nodes_map_;
  uint32_t output_size_ = 1;
  std::vector<std::pair<NodePtr, int32_t>> output_nodes_info_;

  std::vector<std::shared_ptr<ComputeGraph>> sub_graph_;
  std::map<std::string, std::shared_ptr<ComputeGraph>> names_to_subgraph_;
  std::weak_ptr<ComputeGraph> parent_graph_;
  std::weak_ptr<Node> parent_node_;

  // the members followed should not in the ComputeGraph class
  bool is_valid_flag_;
  bool is_summary_graph_ = false;
  // Indicates whether it is need iteration
  bool need_iteration_ = false;
  std::map<std::vector<std::string>, std::vector<std::string>> params_share_map_;
  // TaskIdx -> op_name Map
  std::map<uint32_t, std::string> op_name_map_;
  uint64_t session_id_ = 0;
  ge::Format data_format_ = ge::FORMAT_ND;
  // unknown graph indicator, default is false, mean known shape
  bool is_unknown_shape_graph_ = false;
};
}  // namespace ge
#endif  // INC_GRAPH_COMPUTE_GRAPH_H_
