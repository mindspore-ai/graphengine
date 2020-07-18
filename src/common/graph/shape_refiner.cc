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

#include "graph/shape_refiner.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "external/graph/operator.h"
#include "external/graph/operator_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/compute_graph.h"
#include "utils/node_utils.h"
#include "utils/op_desc_utils.h"
#include "utils/tensor_utils.h"
#include "utils/type_utils.h"

namespace ge {
namespace {
const uint32_t kWhileBodySubGraphIdx = 1;

graphStatus ReverseBrushWhileBodySubGraph(const ConstNodePtr &node) {
  GELOGD("Enter reverse brush while body subgraph process!");

  auto sub_graph_body = NodeUtils::GetSubgraph(*node, kWhileBodySubGraphIdx);
  if (sub_graph_body == nullptr) {
    GELOGE(GRAPH_FAILED, "Get while body graph failed!");
    return GRAPH_FAILED;
  }

  for (const auto &node_sub : sub_graph_body->GetAllNodes()) {
    if (node_sub->GetInDataNodes().size() == 0) {
      continue;
    }

    for (size_t i = 0; i < node_sub->GetAllInDataAnchorsSize(); i++) {
      auto input_desc = node_sub->GetOpDesc()->MutableInputDesc(i);
      (void)input_desc->SetUnknownDimNumShape();
    }
    for (size_t i = 0; i < node_sub->GetAllOutDataAnchorsSize(); i++) {
      auto output_desc = node_sub->GetOpDesc()->MutableOutputDesc(i);
      (void)output_desc->SetUnknownDimNumShape();
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus UpdateParentNodeForBranch(const ConstNodePtr &node,
                                      std::vector<std::vector<GeTensorDesc>> &ref_out_tensors) {
  GELOGD("Enter update parent node shape for class branch op process");
  // check sub_graph shape.If not same ,do unknown shape process
  for (size_t i = 0; i < ref_out_tensors.size(); i++) {
    if (ref_out_tensors[i].empty()) {
      continue;
    }
    auto ref_out_tensor = ref_out_tensors[i].at(0);
    ge::GeShape &ref_out_tensor_shape = ref_out_tensor.MutableShape();
    for (auto &tensor : ref_out_tensors[i]) {
      if (ref_out_tensor.GetDataType() != tensor.GetDataType()) {
        GELOGE(GRAPH_FAILED, "node[%s] does not support diff dtype output", node->GetName().c_str());
        return GRAPH_FAILED;
      }
      auto shape = tensor.MutableShape();
      if (shape.GetDims().size() != ref_out_tensor_shape.GetDims().size()) {
        GELOGD("node is %s, i : %d, shape size: %lu, ref_out_tensor_shape size: %lu", node->GetName().c_str(), i,
               shape.GetShapeSize(), ref_out_tensor_shape.GetShapeSize());
        ref_out_tensor_shape = GeShape(UNKNOWN_RANK);
        break;
      }
      for (size_t j = 0; j < ref_out_tensor_shape.GetDims().size(); j++) {
        if (ref_out_tensor_shape.GetDim(j) == shape.GetDim(j)) {
          continue;
        }
        GELOGD("node is %s, i : %d, j: %d ,shape size: %lu, ref_out_tensor_shape size: %lu", node->GetName().c_str(), i,
               j, shape.GetShapeSize(), ref_out_tensor_shape.GetShapeSize());
        (void)ref_out_tensor_shape.SetDim(j, UNKNOWN_DIM);
      }
    }
    (void)node->GetOpDesc()->UpdateOutputDesc(i, ref_out_tensor);
  }
  return GRAPH_SUCCESS;
}

graphStatus UpdateParentNodeForWhile(const ConstNodePtr &node, std::vector<std::vector<GeTensorDesc>> &ref_data_tensors,
                                     std::vector<std::vector<GeTensorDesc>> &ref_out_tensors) {
  GELOGD("Enter update parent node shape for class while op process");
  if (ref_data_tensors.size() != ref_out_tensors.size()) {
    GELOGE(GRAPH_FAILED, "while op [%s] input number[%zu] and output number[%zu] is not same!", node->GetName().c_str(),
           ref_data_tensors.size(), ref_out_tensors.size());
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < ref_data_tensors.size(); i++) {
    if (ref_out_tensors[i].size() != 1) {
      GELOGE(GRAPH_FAILED, "while op, every output should only find one output tensor in all graph!");
      return GRAPH_FAILED;
    }
  }
  bool is_need_reverse_brush = false;
  // check input and output
  for (size_t i = 0; i < ref_out_tensors.size(); i++) {
    if (ref_out_tensors[i].empty()) {
      continue;
    }
    auto ref_out_tensor = ref_out_tensors[i].at(0);
    auto tmp_shape = ref_out_tensor.MutableShape();
    // ref_i's data and output tensor shape should be same
    for (auto &tensor : ref_data_tensors[i]) {
      if (ref_out_tensor.GetDataType() != tensor.GetDataType()) {
        GELOGE(GRAPH_FAILED, "node[%s] does not support diff dtype or format output.", node->GetName().c_str());
        return GRAPH_FAILED;
      }
      auto shape = tensor.MutableShape();
      if (shape.GetDims() != tmp_shape.GetDims()) {
        ref_out_tensor.SetUnknownDimNumShape();
        is_need_reverse_brush = true;
        break;
      }
    }
    (void)node->GetOpDesc()->UpdateOutputDesc(i, ref_out_tensor);
  }
  // reverse refresh while body shape
  if (is_need_reverse_brush) {
    return ReverseBrushWhileBodySubGraph(node);
  }
  return GRAPH_SUCCESS;
}

graphStatus UpdateSubGraphDataNodes(const ConstNodePtr &node) {
  auto op_desc = node->GetOpDesc();
  auto sub_graph_names = op_desc->GetSubgraphInstanceNames();
  if (sub_graph_names.empty()) {
    return GRAPH_SUCCESS;
  }

  auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  for (const auto &name : sub_graph_names) {
    if (name.empty()) {
      GELOGW("The node %s contains empty subgraph instance name", node->GetName().c_str());
      continue;
    }
    auto sub_graph = root_graph->GetSubgraph(name);
    if (sub_graph == nullptr) {
      GE_LOGE("Can node find the subgrpah %s for node %s", name.c_str(), node->GetName().c_str());
      return GRAPH_FAILED;
    }
    for (const auto &node_sub : sub_graph->GetDirectNode()) {
      if (node_sub->GetType() != DATA) {
        continue;
      }
      int ref_i;
      auto data_opdesc = node_sub->GetOpDesc();
      if (data_opdesc == nullptr) {
        GE_LOGE("Invalid data node on the sub graph %s parent node %s, no OpDesc", name.c_str(),
                node->GetName().c_str());
        return GRAPH_FAILED;
      }
      if (!AttrUtils::GetInt(node_sub->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, ref_i)) {
        GE_LOGE("Invalid data node on the sub graph %s parent node %s, no ref-index attribute", name.c_str(),
                node->GetName().c_str());
        return GRAPH_FAILED;
      }
      auto input_desc = op_desc->MutableInputDesc(ref_i);
      if (input_desc == nullptr) {
        GE_LOGE(
          "The ref index(%d) on the data %s on the sub graph %s "
          "parent node %s are incompatible, inputs num %u",
          ref_i, node_sub->GetName().c_str(), name.c_str(), node->GetName().c_str(), node->GetAllOutDataAnchorsSize());
        return GRAPH_FAILED;
      }
      GELOGI("Ref index is %d, input_desc dtype is %d, node name is %s", ref_i, input_desc->GetDataType(),
             node->GetName().c_str());
      auto ret = data_opdesc->UpdateInputDesc(0, *input_desc);

      if (ret != GRAPH_SUCCESS) {
        GE_LOGE("Failed to update input desc of data %s on the sub graph %s parent node %s",
                node_sub->GetName().c_str(), name.c_str(), node->GetName().c_str());
        return ret;
      }
      ret = data_opdesc->UpdateOutputDesc(0, *input_desc);
      if (ret != GRAPH_SUCCESS) {
        GE_LOGE("Failed to update output desc of data %s on the sub graph %s parent node %s",
                node_sub->GetName().c_str(), name.c_str(), node->GetName().c_str());
        return ret;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus FindSubgraphDataAndNetoutput(std::shared_ptr<ComputeGraph> &sub_graph, NodePtr &netoutput,
                                         const ConstNodePtr &node,
                                         std::vector<std::vector<GeTensorDesc>> &ref_data_tensors) {
  auto sub_nodes = sub_graph->GetDirectNode();
  for (size_t i = sub_nodes.size(); i > 0; --i) {
    auto sub_node = sub_nodes.at(i - 1);
    if (sub_node->GetType() == NETOUTPUT) {
      netoutput = sub_node;
    }
    if (sub_node->GetType() == DATA) {
      if (sub_node->GetOpDesc() == nullptr) {
        return GRAPH_FAILED;
      }

      int ref_i;
      if (!AttrUtils::GetInt(sub_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, ref_i)) {
        GELOGE(GRAPH_FAILED, "subgraph data node[%s] has no parent node!", sub_node->GetName().c_str());
        return GRAPH_FAILED;
      }
      if (ref_i < 0 || static_cast<uint32_t>(ref_i) >= node->GetAllInDataAnchorsSize()) {
        GELOGE(GRAPH_FAILED, "data node[%s]'s ref index[%d] is not in range [0, %zu)!", sub_node->GetName().c_str(),
               ref_i, node->GetAllInDataAnchorsSize());
        return GRAPH_FAILED;
      }
      ref_data_tensors[ref_i].emplace_back(sub_node->GetOpDesc()->GetOutputDesc(0));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus UpdateParentNodeOutTensor(const ConstNodePtr &node) {
  auto op_desc = node->GetOpDesc();
  auto sub_graph_names = op_desc->GetSubgraphInstanceNames();
  if (sub_graph_names.empty()) {
    return GRAPH_SUCCESS;
  }

  std::vector<std::vector<GeTensorDesc>> ref_data_tensors(node->GetAllInDataAnchorsSize());
  std::vector<std::vector<GeTensorDesc>> ref_out_tensors(node->GetAllOutDataAnchorsSize());
  auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());

  for (const auto &name : sub_graph_names) {
    if (name.empty()) {
      GELOGW("The node %s contains empty subgraph instance name", node->GetName().c_str());
      continue;
    }
    auto sub_graph = root_graph->GetSubgraph(name);
    if (sub_graph == nullptr) {
      GE_LOGE("Can node find the subgrpah %s for node %s", name.c_str(), node->GetName().c_str());
      return GRAPH_FAILED;
    }
    NodePtr netoutput = nullptr;
    auto ret = FindSubgraphDataAndNetoutput(sub_graph, netoutput, node, ref_data_tensors);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
    if (netoutput == nullptr) {
      GE_LOGE("No NetOutput node on sub graph %s, parent node %s", name.c_str(), node->GetName().c_str());
      return GRAPH_FAILED;
    }
    auto netoutput_opdesc = netoutput->GetOpDesc();
    if (netoutput_opdesc == nullptr) {
      GE_LOGE("Invalid NetOutput node on sub graph %s, parent node %s, no OpDesc on it", name.c_str(),
              node->GetName().c_str());
      return GRAPH_FAILED;
    }
    for (auto &edge_anchor : netoutput->GetAllInDataAnchors()) {
      auto edge_desc = netoutput_opdesc->MutableInputDesc(edge_anchor->GetIdx());
      if (edge_desc == nullptr) {
        GE_LOGE("Invalid NetOutput node on sub graph %s, parent node %s, can not find input tensor %d", name.c_str(),
                node->GetName().c_str(), edge_anchor->GetIdx());
        return GRAPH_FAILED;
      }
      GELOGI("Netoutput in anchor index is %zu, input tensor dim is %zu", edge_anchor->GetIdx(),
             edge_desc->GetShape().GetDimNum());
      int ref_i;
      if (!AttrUtils::GetInt(edge_desc, ATTR_NAME_PARENT_NODE_INDEX, ref_i)) {
        // if there is no ref index on the TensorDesc, it means the output data will be ignored outer.
        continue;
      }
      GELOGI("Parent node index of edge desc is %d", ref_i);
      if (ref_i < 0 || static_cast<uint32_t>(ref_i) >= node->GetAllOutDataAnchorsSize()) {
        return GRAPH_FAILED;
      }
      ref_out_tensors[ref_i].emplace_back(*edge_desc);
    }
  }

  if (node->GetType() == WHILE) {
    return UpdateParentNodeForWhile(node, ref_data_tensors, ref_out_tensors);
  }
  return UpdateParentNodeForBranch(node, ref_out_tensors);
}
}  // namespace
void ShapeRefiner::PrintInOutTensorShape(const ge::NodePtr &node, const std::string &phase) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "node is null");
    return;
  }
  if (!IsLogEnable(GE, DLOG_DEBUG)) {
    return;
  }
  ge::OpDescPtr op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(GRAPH_FAILED, "op_desc is null."); return );
  std::string str;
  if (op_desc->GetInputsSize() != 0) {
    std::string input_desc_str = "input shape: ";
    for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
      input_desc_str += "[";
      for (int64_t dim : input_desc->GetShape().GetDims()) {
        input_desc_str += std::to_string(dim) + " ";
      }
      input_desc_str += "]";
      input_desc_str += ":" + TypeUtils::DataTypeToSerialString(input_desc->GetDataType()) + ":" +
                        TypeUtils::FormatToSerialString(input_desc->GetFormat()) + " ";
    }
    str += input_desc_str;
  }

  if (op_desc->GetAllOutputsDescSize() != 0) {
    std::string output_desc_str = "output shape: ";
    for (const auto &output_desc : op_desc->GetAllOutputsDescPtr()) {
      if (output_desc == nullptr) {
        continue;
      }
      output_desc_str += "[";
      for (int64_t dim : output_desc->GetShape().GetDims()) {
        output_desc_str += std::to_string(dim) + " ";
      }
      output_desc_str += "]";
      output_desc_str += ":" + TypeUtils::DataTypeToSerialString(output_desc->GetDataType()) + ":" +
                         TypeUtils::FormatToSerialString(output_desc->GetFormat()) + " ";
    }
    str += output_desc_str;
  }
  GELOGD("Shape dump [%s], Node name: [%s]. %s", phase.c_str(), node->GetName().c_str(), str.c_str());
}

graphStatus ShapeRefiner::InferShapeAndType(const ConstNodePtr &node, Operator &op) {
  return InferShapeAndType(node, op, true);
}
graphStatus ShapeRefiner::InferShapeAndType(const ConstNodePtr &node, Operator &op, bool before_subgraph) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(GRAPH_FAILED, "node is null."); return GRAPH_FAILED);
  auto op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(GRAPH_FAILED, "op_desc is null."); return GRAPH_FAILED);
  const auto &op_type = op_desc->GetType();

  graphStatus ret;
  if (before_subgraph) {
    ret = UpdateSubGraphDataNodes(node);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }

  // Get infer func and execute
  ret = op_desc->CallInferFunc(op);
  if (ret == GRAPH_PARAM_INVALID) {
    // Op ir no infer func, try to get infer func from operator factory
    auto node_op = ge::OperatorFactory::CreateOperator("node_op", op_desc->GetType());
    if (node_op.IsEmpty()) {
      GELOGW("get op from OperatorFactory fail. opType: %s", op_type.c_str());
      return ret;
    }

    GELOGD("get op from OperatorFactory success. opType: %s", op_type.c_str());
    auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
    node_op.BreakConnect();
    if (temp_op_desc == nullptr) {
      GELOGE(GRAPH_FAILED, "temp op desc is null");
      return GRAPH_FAILED;
    }
    if (!op_desc->UpdateInputName(temp_op_desc->GetAllInputName())) {
      GELOGW("InferShapeAndType UpdateInputName failed");
      for (const auto &out_desc : op_desc->GetAllOutputsDescPtr()) {
        if (out_desc != nullptr && out_desc->GetShape().GetDims().empty()) {
          break;
        }
        return GRAPH_SUCCESS;
      }
    }
    if (!op_desc->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
      GELOGW("InferShapeAndType UpdateOutputName failed");
    }
    op_desc->AddInferFunc(temp_op_desc->GetInferFunc());
    ret = op_desc->CallInferFunc(op);
    GELOGI("op CallInferFunc second. ret: %u", ret);
  }
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  if (!before_subgraph) {
    return UpdateParentNodeOutTensor(node);
  }
  return GRAPH_SUCCESS;
}

InferenceContextPtr CreateInferenceContext(const std::unordered_map<NodePtr, InferenceContextPtr> &context_map,
                                           const NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "node is null");
    return nullptr;
  }
  InferenceContextPtr inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create());
  if (inference_context == nullptr) {
    GELOGE(GRAPH_FAILED, "Failed to alloc InferenceContext");
    return nullptr;
  }

  auto all_in_data_anchors = node->GetAllInDataAnchors();
  std::vector<std::vector<ShapeAndType>> input_shapes_and_types(all_in_data_anchors.size());
  std::vector<std::string> marks;

  bool has_input_shapes_and_types = false;
  for (const auto &in_anchor : all_in_data_anchors) {
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }

    auto input_node = out_anchor->GetOwnerNode();
    if (input_node == nullptr) {
      continue;
    }

    auto iter = context_map.find(input_node);
    if (iter != context_map.end()) {
      const auto &src_context = iter->second;
      GE_IF_BOOL_EXEC(src_context == nullptr, GELOGE(GRAPH_FAILED, "src_context is null."); return nullptr);
      GELOGD("node:%s get %ld marks from node:%s", node->GetName().c_str(), src_context->GetMarks().size(),
             input_node->GetName().c_str());
      for (auto mark : src_context->GetMarks()) {
        marks.push_back(mark);
      }
      auto output_idx = out_anchor->GetIdx();
      auto input_idx = in_anchor->GetIdx();
      auto output_shape_and_type = src_context->GetOutputHandleShapesAndTypes();
      if (output_idx < static_cast<int>(output_shape_and_type.size())) {
        GELOGI("Add shape and type from %s:%d to %s:%d", input_node->GetName().c_str(), output_idx,
               node->GetName().c_str(), input_idx);
        input_shapes_and_types[input_idx] = output_shape_and_type[output_idx];
        has_input_shapes_and_types = true;
      } else {
        GELOGI("[%s] Output out of range. index = %d, size = %zu", node->GetName().c_str(), output_idx,
               output_shape_and_type.size());
      }
    }
  }

  if (has_input_shapes_and_types) {
    inference_context->SetInputHandleShapesAndTypes(std::move(input_shapes_and_types));
  }
  inference_context->SetMarks(marks);

  return inference_context;
}

namespace {
std::unordered_map<NodePtr, InferenceContextPtr> context_map;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ShapeRefiner::InferShapeAndType(const NodePtr &node) {
  return InferShapeAndType(node, true);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ShapeRefiner::InferShapeAndType(const NodePtr &node,
                                                                                           bool before_subgraph) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(GRAPH_FAILED, "node is null."); return GRAPH_FAILED);
  if (node->Verify() != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Verifying %s failed.", node->GetName().c_str());
    return GRAPH_FAILED;
  }

  auto inference_context = CreateInferenceContext(context_map, node);
  if (inference_context == nullptr) {
    GELOGE(GRAPH_FAILED, "inference context is null");
    return GRAPH_FAILED;
  }

  GELOGD("create context for node:%s, marks %zu", node->GetName().c_str(), inference_context->GetMarks().size());

  PrintInOutTensorShape(node, "before_infershape");

  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  op.SetInferenceContext(inference_context);
  graphStatus status = InferShapeAndType(node, op, before_subgraph);
  if (status == GRAPH_PARAM_INVALID || status == GRAPH_SUCCESS) {
    (void)ge::NodeUtils::UpdatePeerNodeInputDesc(node);
  } else {
    GELOGE(GRAPH_FAILED, "%s call infer function failed.", node->GetName().c_str());
    return GRAPH_FAILED;
  }

  auto ctx_after_infer = op.GetInferenceContext();
  if (ctx_after_infer != nullptr) {
    GELOGD("[%s] after infershape. mark:%zu", node->GetName().c_str(), ctx_after_infer->GetMarks().size());
    if (!ctx_after_infer->GetOutputHandleShapesAndTypes().empty() || !ctx_after_infer->GetMarks().empty()) {
      GELOGD("[%s] set inference context after. mark:%zu", node->GetName().c_str(), ctx_after_infer->GetMarks().size());
      (void)context_map.emplace(node, ctx_after_infer);
    }
  }

  PrintInOutTensorShape(node, "after_infershape");

  return GRAPH_SUCCESS;
}
}  // namespace ge
