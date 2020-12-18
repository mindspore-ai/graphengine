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

#ifndef INC_GRAPH_UTILS_GRAPH_UTILS_H_
#define INC_GRAPH_UTILS_GRAPH_UTILS_H_

#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/utils/anchor_utils.h"

#define GE_DUMP(compute_graph, name)                                                                               \
  do {                                                                                                             \
    GraphUtils::DumpGEGraph(compute_graph, name);                                                                  \
    GraphUtils::DumpGEGraphToOnnx(*compute_graph, name);                                                           \
    uint64_t i = 0;                                                                                                \
    for (const auto &sub_graph_func : compute_graph->GetAllSubgraphs()) {                                          \
      auto sub_graph_func_name = std::string(name) + std::string("_sub_graph_") + std::to_string(i++);             \
      GraphUtils::DumpGEGraph(sub_graph_func, sub_graph_func_name);                                                \
      GraphUtils::DumpGEGraphToOnnx(*sub_graph_func, sub_graph_func_name);                                         \
    }                                                                                                              \
  } while (0)

#define REFER_ATTR_VALUE(VT_ENUM, DataType, attr, ret) \
  do {                                                 \
    DataType ret;                                      \
    attr.GetValue<DataType>(ret);                      \
  } while (0)

#define PRINT_ATTR_VALUE_IF(value_type, VT_ENUM, DataType, attr, stream) \
  do {                                                                   \
    if (value_type == VT_ENUM) {                                         \
      REFER_ATTR_VALUE(VT_ENUM, DataType, attr, ret)                     \
      stream << ret;                                                     \
    }                                                                    \
  } while (0)

#define PRINT_LIST_ATTR_VALUE_IF(value_type, VT_ENUM, DataType, attr, stream) \
  do {                                                                        \
    if (value_type == VT_ENUM) {                                              \
      REFER_ATTR_VALUE(VT_ENUM, DataType, attr, ret)                          \
      stream << "[";                                                          \
      for (int i = 0; i < ret.size(); i++) {                                  \
        stream << ret[i];                                                     \
        if (i + 1 != ret.size()) stream << ", ";                              \
      }                                                                       \
      stream << "]";                                                          \
    }                                                                         \
  } while (0)

#define PRINT_ATTR_VALUE_ELIF(value_type, VT_ENUM, DataType, attr, stream) \
  else PRINT_ATTR_VALUE_IF(value_type, VT_ENUM, DataType, attr, stream)

#define PRINT_LIST_ATTR_VALUE_ELIF(value_type, VT_ENUM, DataType, attr, stream) \
  else PRINT_LIST_ATTR_VALUE_IF(value_type, VT_ENUM, DataType, attr, stream)

#define PRINT_SHAPE(i_o, n, idx, stream)                                               \
  do {                                                                                 \
    auto op = n->GetOpDesc();                                                          \
    GeTensorDesc td = i_o == "input" ? op->GetInputDesc(idx) : op->GetOutputDesc(idx); \
    auto shape = td.GetShape().GetDims();                                              \
    stream << "[";                                                                     \
    for (int i = 0; i < shape.size(); i++) {                                           \
      stream << shape[i];                                                              \
      if (i + 1 < shape.size()) stream << ", ";                                        \
    }                                                                                  \
    stream << "]";                                                                     \
  } while (0)

#define PRINT_ATTR_FUNC(stream)                                                                                    \
  [&](GeAttrValue attr) {                                                                                          \
    auto type = attr.GetValueType();                                                                               \
    PRINT_ATTR_VALUE_IF(type, GeAttrValue::ValueType::VT_STRING, GeAttrValue::STR, attr, stream)                   \
    PRINT_ATTR_VALUE_ELIF(type, GeAttrValue::ValueType::VT_FLOAT, GeAttrValue::FLOAT, attr, stream)                \
    PRINT_ATTR_VALUE_ELIF(type, GeAttrValue::ValueType::VT_BOOL, GeAttrValue::BOOL, attr, stream)                  \
    PRINT_ATTR_VALUE_ELIF(type, GeAttrValue::ValueType::VT_INT, GeAttrValue::INT, attr, stream)                    \
    PRINT_LIST_ATTR_VALUE_ELIF(type, GeAttrValue::ValueType::VT_LIST_STRING, GeAttrValue::LIST_STR, attr, stream)  \
    PRINT_LIST_ATTR_VALUE_ELIF(type, GeAttrValue::ValueType::VT_LIST_FLOAT, GeAttrValue::LIST_FLOAT, attr, stream) \
    PRINT_LIST_ATTR_VALUE_ELIF(type, GeAttrValue::ValueType::VT_LIST_BOOL, GeAttrValue::LIST_BOOL, attr, stream)   \
    PRINT_LIST_ATTR_VALUE_ELIF(type, GeAttrValue::ValueType::VT_LIST_INT, GeAttrValue::LIST_INT, attr, stream)     \
    else if (type == GeAttrValue::ValueType::VT_TENSOR_DESC) stream << "TENSOR_DESC";                              \
    else if (type == GeAttrValue::ValueType::VT_TENSOR) stream << "TENSOR";                                        \
    else if (type == GeAttrValue::ValueType::VT_BYTES) stream << "BYTES";                                          \
    else if (type == GeAttrValue::ValueType::VT_LIST_TENSOR_DESC) stream << "LIST_TENSOR_DESC";                    \
    else if (type == GeAttrValue::ValueType::VT_LIST_TENSOR) stream << "LIST_TENSOR";                              \
    else if (type == GeAttrValue::ValueType::VT_LIST_BYTES) stream << "LIST_BYTES";                                \
  };

namespace ge {
enum IOType { kIn, kOut };

struct NodeIndexIO {
  NodeIndexIO(ge::NodePtr node, uint32_t index, IOType io_type)
      : node_(std::move(node)), index_(index), io_type_(io_type) {
    if (node_ != nullptr) {
      value_ = node_->GetName() + (io_type_ == kOut ? "_out_" : "_in_") + std::to_string(index_);
    }
  }
  NodeIndexIO(ge::NodePtr node, int index, IOType io_type)
      : node_(std::move(node)), index_(static_cast<uint32_t>(index)), io_type_(io_type) {
    if (node_ != nullptr) {
      value_ = node_->GetName() + (io_type_ == kOut ? "_out_" : "_in_") + std::to_string(index_);
    }
  }
  ~NodeIndexIO() {}

  NodePtr node_ = nullptr;
  uint32_t index_ = 0;
  IOType io_type_ = kOut;
  std::string value_;

  const std::string &ToString() const { return value_; }
};

class GraphUtils {
 public:
  static ComputeGraphPtr GetComputeGraph(const Graph &graph);

  static Graph CreateGraphFromComputeGraph(const ComputeGraphPtr compute_graph);

  static GraphPtr CreateGraphPtrFromComputeGraph(const ComputeGraphPtr compute_graph);

  static graphStatus RecoverGraphOperators(const Graph &graph);

  static ComputeGraphPtr CreateGraphFromOperator(const string &name, const std::vector<Operator> &inputs);

  static graphStatus AddEdge(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst);

  static graphStatus AddEdge(const OutDataAnchorPtr &src, const Format &src_format, const InDataAnchorPtr &dst,
                             const Format &dst_format);

  static graphStatus AddEdge(const AnchorPtr &src, const AnchorPtr &dst);

  static graphStatus AddEdge(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst);

  static graphStatus AddEdge(const OutDataAnchorPtr &src, const InControlAnchorPtr &dst);

  // check whether src is link to dst and then remove
  static graphStatus RemoveEdge(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst);

  static graphStatus RemoveEdge(const AnchorPtr &src, const AnchorPtr &dst);

  static graphStatus RemoveEdge(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst);

  static graphStatus RemoveEdge(const OutDataAnchorPtr &src, const InControlAnchorPtr &dst);

  static graphStatus ReplaceEdgeDst(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                    const InDataAnchorPtr &new_dst);

  static graphStatus ReplaceEdgeDst(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                                    const InControlAnchorPtr &new_dst);

  static graphStatus InsertNodeBetweenDataAnchors(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                                  const NodePtr &new_node);

  static graphStatus RemoveSubgraphRecursively(const ComputeGraphPtr &compute_graph, const NodePtr &remove_node);

  static graphStatus RemoveNodeWithoutRelink(const ComputeGraphPtr &compute_graph, const NodePtr &node);

  static graphStatus InsertTransNode(ComputeGraphPtr compute_graph, const InDataAnchorPtr &in_data_anchor,
                                     const std::vector<OpDescPtr> &vec_op_desc);

  ///
  /// @brief Insert node: src->insert_node:input_index, insert_node:output_index->dst
  /// @param [in] src
  /// @param [in] dsts
  /// @param [in] insert_node
  /// @param [in] input_index
  /// @param [in] output_index
  /// @return graphStatus
  ///
  static graphStatus InsertNodeAfter(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                                     const NodePtr &insert_node, uint32_t input_index = 0, uint32_t output_index = 0);

  static graphStatus RemoveJustNode(ComputeGraphPtr compute_graph, const NodePtr &node);

  static graphStatus RemoveJustNode(ComputeGraph &compute_graph, const NodePtr &node);

  static void RecordOriginalNames(std::vector<ge::NodePtr> original_nodes, const ge::NodePtr &node);

  static void RecordOriginalNames(std::vector<std::string> names_tmp, const ge::NodePtr &node);

  static bool MatchDumpStr(const std::string &suffix);

  static void DumpGEGraph(const ge::ComputeGraphPtr &graph,
                          const std::string &suffix,
                          bool is_always_dump = false,
                          const std::string &user_graph_name = "");

  static void DumpGEGrph(const ge::ComputeGraphPtr &graph,
                                  const std::string &path,
                                  const std::string &suffix);

  static bool LoadGEGraph(const char *file, ge::ComputeGraph &compute_graph);

  static bool LoadGEGraph(const char *file, ge::ComputeGraphPtr &compute_graph);

  static void BreakConnect(const std::map<OperatorImplPtr, NodePtr> &all_nodes_infos);

  static void DumpGEGraphToOnnx(const ge::ComputeGraph &compute_graph, const std::string &suffix);

  static void DumpGrphToOnnx(const ge::ComputeGraph &compute_graph,
                             const std::string &path, const std::string &suffix);

  static bool LoadGEGraphFromOnnx(const char *file, ge::ComputeGraph &compute_graph);

  static bool ReadProtoFromTextFile(const char *file, google::protobuf::Message *message);

  static void WriteProtoToTextFile(const google::protobuf::Message &proto, const char *real_path);

  static graphStatus AppendInputNode(const ComputeGraphPtr &graph, const NodePtr &node);

  ///
  /// Isolating `node`, relinking data links from the in-anchor peer nodes to
  /// the out-anchor peer nodes according to `io_map`, relinking control links
  /// to ensure that input nodes of `node` are before out nodes
  ///
  /// Link the `io_map[i]` input anchor peer node to `i` output anchor peer
  /// nodes, then unlink all links connecting with `node`. If `io_map[i]` < 0,
  /// unlink all links from `i` output anchor without any relinking.
  ///
  /// @param node
  /// @param io_map
  /// @return
  ///
  static graphStatus IsolateNode(const NodePtr &node, const std::initializer_list<int> &io_map);
  static graphStatus IsolateNode(const NodePtr &node, const std::vector<int> &io_map);

  ///
  /// Isolate `node` which must be one input one output, equivalent to
  /// `IsolateNode(node, {0})`
  /// @param node
  /// @return
  ///
  static graphStatus IsolateNodeOneIO(const NodePtr &node);

  ///
  /// The data anchors replacing behavior is the same with
  /// `ReplaceNodeDataAnchors`. In addition, replace all `old_node` control
  /// anchors with `new_node`'s.
  /// @param new_node
  /// @param old_node
  /// @param inputs_map
  /// @param outputs_map
  /// @return
  ///
  static graphStatus ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                        std::initializer_list<int> inputs_map, std::initializer_list<int> outputs_map);

  static graphStatus ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                        const std::vector<int> &inputs_map, const std::vector<int> &outputs_map);

  ///
  /// Replace `old_node` data anchors with `new_node`'s according to `inputs_map` and `outputs_map`.
  /// Replace the `i` in/out data anchor on `old_node` with
  /// `inputs_map[i]`/`outputs_map[i]` data anchor on `new_node`.
  /// If `inputs_map[i]`/`outputs_map[i]` < 0 or the index not contained in
  /// `inputs_map[i]`/`outputs_map[i]`, the `i` data anchor will remain
  /// on `old_node`.
  /// @param new_node
  /// @param old_node
  /// @param inputs_map
  /// @param outputs_map
  /// @return
  ///
  static graphStatus ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                            std::initializer_list<int> inputs_map,
                                            std::initializer_list<int> outputs_map);

  static graphStatus ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                            const std::vector<int> &inputs_map, const std::vector<int> &outputs_map);

  ///
  /// Copy all in-control edges from `src_node` to `dst_node`
  /// @param src_node
  /// @param dst_node
  /// @return
  ///
  static graphStatus CopyInCtrlEdges(const NodePtr &src_node, NodePtr &dst_node);

  static graphStatus MoveInCtrlEdges(const NodePtr &src_node, NodePtr &dst_node);

  ///
  /// Copy all out-control edges from `src_node` to `dst_node`
  /// @param src_node
  /// @param dst_node
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus CopyOutCtrlEdges(const NodePtr &src_node, NodePtr &dst_node);

  ///
  /// Move all out-control edges from `src_node` to `dst_node`
  /// @param src_node
  /// @param dst_node
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus MoveOutCtrlEdges(NodePtr &src_node, NodePtr &dst_node);

  ///
  /// Copy all in-data edges from `src_node` to `dst_node`
  /// @param src_node
  /// @param dst_node
  /// @return
  ///
  static graphStatus CopyInDataEdges(const NodePtr &src_node, NodePtr &dst_node);

  static ComputeGraphPtr FindRootGraph(ComputeGraphPtr graph);

  ///
  /// Make a copy of ComputeGraph.
  /// @param graph: original graph.
  /// @param prefix: node name prefix of new graph.
  /// @return ComputeGraphPtr
  ///
  static ComputeGraphPtr CloneGraph(const ComputeGraphPtr &graph, const string &prefix,
                                    std::vector<NodePtr> &input_nodes, std::vector<NodePtr> &output_nodes);

  ///
  /// Copy tensor attribute to new node.
  /// @param [in] dst_desc: cloned node.
  /// @param [in] src_node: original node.
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus CopyTensorAttrs(const OpDescPtr &dst_desc, const NodePtr &src_node);

  static graphStatus TopologicalSortingByName(const ge::ComputeGraphPtr &compute_graph, vector<NodePtr> &node_vec);

  ///
  /// Get reference-mapping of all data_anchors in graph
  /// @param [in] graph
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus GetRefMapping(const ComputeGraphPtr &graph,
                                   std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                   std::map<std::string, std::string> &anchor_to_symbol);

  ///
  /// Determine if the graph is a UNKNOWN_SHAPE graph based on whether the graph and all subgraphs
  /// of the graph have UNKNOWN_SHAPE operators or not.
  /// Note: This function will only look 'down' from the graph, not 'up'. For example, the following
  /// scenario (K for known shape, U for unknown shape), ROOT graph is UNKNOWN_SHAPE while SUB graph is KNOWN_SHAPE
  /// ROOT graph:      A -----> B -----> C
  ///                  K    subgraph     U
  ///                           |
  ///                           V
  /// SUB graph:          D --> E --> F
  ///                     K     K     K
  /// @param [in] graph
  /// @return bool
  ///
  static bool IsUnknownShapeGraph(const ComputeGraphPtr &graph);

  static NodePtr FindNodeFromAllNodes(ComputeGraphPtr &graph, const std::string &name);

 private:
  ///
  /// Get reference-mapping for in_data_anchors of node
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus HandleInAnchorMapping(const NodePtr &node,
                                           std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                           std::map<std::string, std::string> &anchor_to_symbol);

  ///
  /// Get reference-mapping for out_data_anchors of node
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus HandleOutAnchorMapping(const NodePtr &node,
                                            std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                            std::map<std::string, std::string> &anchor_to_symbol);

  ///
  /// Handle input of subgraph
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus HandleSubgraphInput(const NodePtr &node,
                                         std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                         std::map<std::string, std::string> &anchor_to_symbol);

  ///
  /// Handle input of Merge op
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus HandleMergeInput(const NodePtr &node,
                                      std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                      std::map<std::string, std::string> &anchor_to_symbol);

  ///
  /// Handle output of subgraph
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus HandleSubgraphOutput(const NodePtr &node,
                                          std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                          std::map<std::string, std::string> &anchor_to_symbol);

  ///
  /// Relink all edges for cloned ComputeGraph.
  /// @param [in] node: original node.
  /// @param [in] prefix: node name prefix of new node.
  /// @param [in] all_nodes: all nodes in new graph.
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus RelinkGraphEdges(const NodePtr &node, const string &prefix,
                                      const std::unordered_map<string, NodePtr> &all_nodes);

  ///
  /// Union ref-mapping
  /// @param [in] exist_node_info1
  /// @param [in] exist_node_info2
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @param [out] symbol
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus UnionSymbolMapping(const NodeIndexIO &exist_node_info1, const NodeIndexIO &exist_node_info2,
                                        std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                        std::map<std::string, std::string> &anchor_to_symbol, std::string &symbol);

  ///
  /// Update symbol mapping with a new reference pair
  /// @param [in] cur_node_info
  /// @param [in] exist_node_info
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  ///
  static graphStatus UpdateRefMapping(const NodeIndexIO &cur_node_info, const NodeIndexIO &exist_node_info,
                                      std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                      std::map<std::string, std::string> &anchor_to_symbol);

  ///
  /// Check if out_data_anchor is reference of input
  /// @param [in] out_data_anchor
  /// @param [out] reuse_in_index
  /// @return bool
  ///
  static bool IsRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index);
};

class ComputeGraphBuilder {
 public:
  ComputeGraphBuilder() : owner_graph_(nullptr) {}
  ComputeGraphBuilder(const ComputeGraphBuilder &) = delete;
  ComputeGraphBuilder &operator=(const ComputeGraphBuilder &) = delete;
  ComputeGraphBuilder(const ComputeGraphBuilder &&) = delete;
  ComputeGraphBuilder &operator=(const ComputeGraphBuilder &&) = delete;
  ~ComputeGraphBuilder() = default;

  ///
  /// @brief Add node to graph
  /// @param [in] op_desc
  /// @return ComputeGraphBuilder
  ///
  virtual ComputeGraphBuilder &AddNode(const OpDescPtr &op_desc);

  ///
  /// @brief Add data-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] out_anchor_ind
  /// @param [in] dst_name
  /// @param [in] in_anchor_ind
  /// @return ComputeGraphBuilder
  ///
  virtual ComputeGraphBuilder &AddDataLink(const std::string &src_name, uint32_t out_anchor_ind,
                                           const std::string &dst_name, uint32_t in_anchor_ind);

  ///
  /// @brief Add ctrl-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] dst_name
  /// @return ComputeGraphBuilder
  ///
  virtual ComputeGraphBuilder &AddControlLink(const std::string &src_name, const std::string &dst_name);

  ///
  /// @brief Build graph
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return ComputeGraphPtr
  ///
  virtual ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) = 0;

  /// @brief Get node with name
  /// @param [in] name
  /// @return NodePtr
  ///
  NodePtr GetNode(const std::string &name);

  /// @brief Get all nodes
  /// @return std::vector<NodePtr>
  ///
  std::vector<NodePtr> GetAllNodes();

 protected:
  ///
  /// @brief Build nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void BuildNodes(graphStatus &error_code, std::string &error_msg);

  ///
  /// @brief Build data-links
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void BuildDataLinks(graphStatus &error_code, std::string &error_msg);

  ///
  /// @brief Build ctrl-links
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void BuildCtrlLinks(graphStatus &error_code, std::string &error_msg);

  ComputeGraphPtr owner_graph_;

  // node_name -> node
  std::map<std::string, NodePtr> node_names_;
  std::vector<OpDescPtr> nodes_;

  // <src_node_name, out_anchor_ind> -> <dst_node_name, in_anchor_ind>
  std::vector<std::pair<std::pair<std::string, uint32_t>, std::pair<std::string, uint32_t>>> data_links_;
  // src_node_name -> dst_node_name
  std::vector<std::pair<std::string, std::string>> ctrl_links_;
};

class CompleteGraphBuilder : public ComputeGraphBuilder {
 public:
  explicit CompleteGraphBuilder(std::string name, bool retval_flag = true)
      : name_(std::move(name)), parent_node_(nullptr), retval_flag_(retval_flag) {}
  CompleteGraphBuilder(const CompleteGraphBuilder &) = delete;
  CompleteGraphBuilder &operator=(const CompleteGraphBuilder &) = delete;
  CompleteGraphBuilder(const CompleteGraphBuilder &&) = delete;
  CompleteGraphBuilder &operator=(const CompleteGraphBuilder &&) = delete;
  ~CompleteGraphBuilder() = default;

  ///
  /// @brief Add node to graph
  /// @param [in] op_desc
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &AddNode(const OpDescPtr &op_desc) override;

  ///
  /// @brief Add data-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] out_anchor_ind
  /// @param [in] dst_name
  /// @param [in] in_anchor_ind
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &AddDataLink(const std::string &src_name, uint32_t out_anchor_ind,
                                    const std::string &dst_name, uint32_t in_anchor_ind) override;

  ///
  /// @brief Add ctrl-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] dst_name
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &AddControlLink(const std::string &src_name, const std::string &dst_name) override;

  ///
  /// @brief Set index_th input anchor for graph
  /// @param [in] index
  /// @param [in] node_names
  /// @param [in] anchor_inds
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &SetInput(uint32_t index, const std::vector<std::string> &node_names,
                                 const std::vector<uint32_t> &anchor_inds);

  ///
  /// @brief Set index_th input of graph as useless
  /// @param [in] index
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &SetUselessInput(uint32_t index);

  ///
  /// @brief Add output anchor for graph
  /// @param [in] owner_node_name
  /// @param [in] anchor_ind
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &AddOutput(const std::string &owner_node_name, uint32_t anchor_ind);

  ///
  /// @brief Add target for graph
  /// @param [in] target_name
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &AddTarget(const std::string &target_name);

  ///
  /// @brief Set parent-node of graph
  /// @param [in] parent_node
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &SetParentNode(const NodePtr &parent_node);

  ///
  /// @brief Set mapping-relation of parent-node in_anchor_ind & Data-node
  /// @param [in] input_mapping: index_of_graph_input -> in_anchor_index_of_parent_node
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &SetInputMapping(const std::map<uint32_t, uint32_t> &input_mapping);

  ///
  /// @brief Set mapping-relation of parent-node out_anchor_ind & NetOutput-node out_anchor_ind
  /// @param [in] output_mapping: index_of_graph_output -> out_anchor_index_of_parent_node
  /// @return CompleteGraphBuilder
  ///
  CompleteGraphBuilder &SetOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping);

  ///
  /// @brief Build graph
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return ComputeGraphPtr
  ///
  ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) override;

 private:
  ///
  /// @brief Add data nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void AddDataNodes(graphStatus &error_code, std::string &error_msg);

  ///
  /// @brief Add data node
  /// @param [in] index
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  NodePtr AddDataNode(uint32_t index, graphStatus &error_code, std::string &error_msg);

  ///
  /// @brief Add RetVal nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void AddRetValNodes(graphStatus &error_code, std::string &error_msg);

  ///
  /// @brief Build target-nodes for graph
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void BuildGraphTargets(graphStatus &error_code, std::string &error_msg);

  ///
  /// @brief Add NetOutput node
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void AddNetOutputNode(graphStatus &error_code, std::string &error_msg);

  ///
  /// @brief Build NetOutput nodes with data & ctrl edges
  /// @param [in] net_output_desc
  /// @param [in] peer_out_anchors
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void BuildNetOutputNodeWithLink(const OpDescPtr &net_output_desc,
                                  const std::vector<OutDataAnchorPtr> &peer_out_anchors,
                                  graphStatus &error_code, std::string &error_msg);

  ///
  /// @brief process after build
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void PostProcess(graphStatus &error_code, std::string &error_msg);

  std::string name_;
  NodePtr parent_node_;
  bool retval_flag_;
  std::map<uint32_t, std::pair<std::vector<std::string>, std::vector<uint32_t>>> graph_inputs_;
  std::vector<std::pair<std::string, uint32_t>> graph_outputs_;
  std::vector<std::string> graph_targets_;

  // index_of_graph_input -> in_anchor_index_of_parent_node
  std::map<uint32_t, uint32_t> input_mapping_;
  // index_of_graph_output -> out_anchor_index_of_parent_node
  std::map<uint32_t, uint32_t> output_mapping_;
};

class PartialGraphBuilder : public ComputeGraphBuilder {
 public:
  PartialGraphBuilder() = default;
  PartialGraphBuilder(const PartialGraphBuilder &) = delete;
  PartialGraphBuilder &operator=(const PartialGraphBuilder &) = delete;
  PartialGraphBuilder(const PartialGraphBuilder &&) = delete;
  PartialGraphBuilder &operator=(const PartialGraphBuilder &&) = delete;
  ~PartialGraphBuilder() = default;

  ///
  /// @brief Add node to graph
  /// @param [in] op_desc
  /// @return PartialGraphBuilder
  ///
  PartialGraphBuilder &AddNode(const OpDescPtr &op_desc) override;

  ///
  /// @brief Add data-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] out_anchor_ind
  /// @param [in] dst_name
  /// @param [in] in_anchor_ind
  /// @return PartialGraphBuilder
  ///
  PartialGraphBuilder &AddDataLink(const std::string &src_name, uint32_t out_anchor_ind,
                                   const std::string &dst_name, uint32_t in_anchor_ind) override;

  ///
  /// @brief Add ctrl-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] dst_name
  /// @return PartialGraphBuilder
  ///
  PartialGraphBuilder &AddControlLink(const std::string &src_name, const std::string &dst_name) override;

  ///
  /// @brief Set owner graph
  /// @param [in] graph
  /// @return PartialGraphBuilder
  ///
  PartialGraphBuilder &SetOwnerGraph(const ComputeGraphPtr &graph);

  ///
  /// @brief Add exist node
  /// @param [in] node
  /// @return PartialGraphBuilder
  ///
  PartialGraphBuilder &AddExistNode(const NodePtr &node);

  ///
  /// @brief Build multi nodes with links
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return ComputeGraphPtr
  ///
  ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) override;

 private:
  ///
  /// @brief Build exist nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  ///
  void BuildExistNodes(graphStatus &error_code, std::string &error_msg);

  std::vector<NodePtr> exist_nodes_;
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_GRAPH_UTILS_H_
