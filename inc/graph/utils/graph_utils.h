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

#ifndef INC_GRAPH_UTILS_GRAPH_UTILS_H_
#define INC_GRAPH_UTILS_GRAPH_UTILS_H_

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph/compute_graph.h"
#include "graph/utils/anchor_utils.h"
#include "graph/graph.h"
#include "graph/model.h"

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
class GraphUtils {
 public:
  static ComputeGraphPtr GetComputeGraph(const Graph &graph);

  static Graph CreateGraphFromComputeGraph(const ComputeGraphPtr compute_graph);

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

  static graphStatus RemoveNodeWithoutRelink(const ComputeGraphPtr &compute_graph, const NodePtr &node);

  static graphStatus InsertTransNode(ComputeGraphPtr compute_graph, const InDataAnchorPtr &in_data_anchor,
                                     const std::vector<OpDescPtr> &vec_op_desc);

  static graphStatus RemoveJustNode(ComputeGraphPtr compute_graph, const NodePtr &node);

  static graphStatus RemoveJustNode(ComputeGraph &compute_graph, const NodePtr &node);

  static void RecordOriginalNames(std::vector<ge::NodePtr> original_nodes, const ge::NodePtr &node);

  static void RecordOriginalNames(std::vector<std::string> names_tmp, const ge::NodePtr &node);

  static bool CheckIsTrainGraph(const ge::ComputeGraphPtr &compute_graph);

  static bool MatchDumpStr(const std::string &suffix);

  static void DumpGEGraph(const ge::ComputeGraphPtr &graph, const std::string &suffix, bool is_always_dump = false);

  static bool LoadGEGraph(const char *file, ge::ComputeGraph &compute_graph);

  static bool CheckGlobalStepNode(const ge::NodePtr &node);

  static void BreakConnect(const std::map<OperatorImplPtr, NodePtr> &all_nodes_infos);

  static void DumpGEGraphToOnnx(const ge::ComputeGraph &compute_graph, const std::string &suffix);

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
};
}  // namespace ge

#endif  // INC_GRAPH_UTILS_GRAPH_UTILS_H_
