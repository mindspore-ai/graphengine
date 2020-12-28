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

#include "graph/passes/multi_batch_clone_pass.h"

#include "common/formats/utils/formats_trans_utils.h"
#include "common/ge/ge_util.h"
#include "graph/common/local_context.h"
#include "graph/preprocess/multi_batch_options.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_registry.h"

namespace ge {
namespace {
constexpr uint8_t kDataInIndex = 0;
constexpr uint8_t kDataOutIndex = 0;
constexpr uint8_t kCaseArgIndex = 1;

const std::string kMultiBatchCaseNode = "ascend_mbatch_shape_case";
const std::string kMultiBatchDataNode = "ascend_mbatch_shape_data";
const std::string kMultiBatchConstNode = "ascend_mbatch_shape_const";
const std::string kMultiBatchMapIndexNode = "ascend_mbatch_shape_mapindex";
const std::string kMultiBatchNodePostfix = "_ascend_mbatch_batch_";
}  // namespace

Status MultiBatchClonePass::Run(ComputeGraphPtr graph) {
  if (graph->GetParentGraph() != nullptr) {
    GELOGD("Subgraph %s skip the MultiBatchClonePass", graph->GetName().c_str());
    return SUCCESS;
  }

  if (!multibatch::InitDynamicParams(batch_shapes_)) {
    GELOGD("There is no multi-batch options, no need clone multi-batch graph");
    return SUCCESS;
  }

  GELOGD("Begin to run Multi-batch clone on graph: %s", graph->GetName().c_str());
  GE_CHK_STATUS_RET(multibatch::CheckDynamicParams(batch_shapes_), "Invalid multi-batch param");
  if (CollectIoNodes(graph) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Collect input output nodes failed");
    return INTERNAL_ERROR;
  }

  // parser data dynamic info from atc parameter --input_shape
  if (multibatch::ParserDataToDynmaicInfo(batch_shapes_, GetLocalOmgContext().user_input_dims,
                                          data_to_dynamic_info_) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Parse each data's own dynamic info failed");
    return PARAM_INVALID;
  }

  (void)AttrUtils::GetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id_);
  ComputeGraphPtr branch = MakeShared<ComputeGraph>(graph->GetName());
  if (branch == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch graph failed");
    return OUT_OF_MEMORY;
  }
  (void)AttrUtils::SetStr(branch, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id_);

  graph->InValid();  // Will modify, need topological again.
  graph->Swap(*branch);
  if (CreateRootGraph(graph) != SUCCESS) {
    return FAILED;
  }

  if (CreateSubgraphs(graph, branch) != SUCCESS) {
    return FAILED;
  }

  GE_CHK_STATUS_RET(PruneDirectOutput(graph), "Prune direct output failed");
  GELOGD("MultiBatchClonePass Leave");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Collect input output node from original graph.
/// @param [in] const ComputeGraphPtr &graph: original graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CollectIoNodes(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == DATA) {
      all_data_nodes_.emplace_back(node);
    } else if (node->GetType() == CONSTANT) {
      all_const_nodes_.emplace_back(node);
    } else if (node->GetType() == NETOUTPUT) {
      all_output_nodes_.emplace_back(node);
    }

    // If the node save as input/output node, delete record.
    (void)graph->RemoveInputNode(node);
    (void)graph->RemoveOutputNode(node);
  }

  if (all_data_nodes_.empty() || all_output_nodes_.size() != 1) {
    GELOGE(FAILED, "data nodes: %zu, output nodes: %zu", all_data_nodes_.size(), all_output_nodes_.size());
    return FAILED;
  }

  int64_t data_index = 0;
  for (size_t i = 0; i < all_data_nodes_.size(); ++i) {
    const auto &op_desc = all_data_nodes_[i]->GetOpDesc();
    if (!AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, data_index)) {
      (void)AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, i);
    }
  }

  const auto &output = all_output_nodes_[0];
  for (size_t i = 0; i < output->GetAllInDataAnchorsSize(); ++i) {
    const auto in_anchor = output->GetInDataAnchor(i);
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    const auto data_node = out_anchor->GetOwnerNode();
    if (data_node->GetType() == DATA) {
      direct_output_[i] = data_node->GetName();
      GE_CHK_GRAPH_STATUS_RET(
          GraphUtils::RemoveEdge(data_node->GetOutDataAnchor(kDataOutIndex), output->GetInDataAnchor(i)),
          "Remove edge failed");
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Create nodes for root graph.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CreateRootGraph(const ComputeGraphPtr &graph) {
  uint32_t input_num = all_data_nodes_.size() + all_const_nodes_.size();
  uint32_t output_num = all_output_nodes_[0]->GetAllInDataAnchorsSize();

  OpDescBuilder op_builder(kMultiBatchCaseNode, CASE);
  op_builder.AddInput("branch_index").AddDynamicInput("input", input_num).AddDynamicOutput("output", output_num);
  const OpDescPtr op_desc = op_builder.Build();
  if (op_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch case desc failed");
    return OUT_OF_MEMORY;
  }

  op_desc->RegisterSubgraphIrName("branches", kDynamic);
  case_node_ = graph->AddNode(op_desc);
  if (case_node_ == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch case node failed");
    return OUT_OF_MEMORY;
  }

  uint32_t batch_num = static_cast<uint32_t>(batch_shapes_.size());
  if (!AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGE(FAILED, "Set attr ATTR_NAME_BATCH_NUM failed, Case: %s.", op_desc->GetName().c_str());
    return FAILED;
  }

  for (uint32_t i = 0; i < batch_num; i++) {
    const std::string &attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    if (!AttrUtils::SetListInt(op_desc, attr_name, batch_shapes_[i])) {
      GELOGE(FAILED, "Set attr ATTR_NAME_PRED_VALUE failed, Case: %s.", op_desc->GetName().c_str());
      return FAILED;
    }
  }

  std::vector<std::string> data_name_order;
  for (auto &item : GetLocalOmgContext().user_input_dims) {
    data_name_order.push_back(item.first);
  }
  if (!AttrUtils::SetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, data_name_order)) {
    GELOGE(FAILED, "Failed to add user designate shape order attr on case node %s",
           op_desc->GetName().c_str());
    return FAILED;
  }
  GE_CHK_STATUS_RET(multibatch::StampDynamicType(op_desc), "Set dynamic type failed");

  GE_CHK_STATUS_RET(CreateIndexNode(graph), "Create index node failed");
  GE_CHK_STATUS_RET(CreateInputNode(graph), "Create input node failed");
  GE_CHK_STATUS_RET(CreateConstNode(graph), "Create const node failed");
  GE_CHK_STATUS_RET(CreateOutputNode(graph), "Create output node failed");

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Create index data node for root graph.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @param [in] NodePtr node: index data node.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CreateIndexDataNode(const ComputeGraphPtr &graph, NodePtr &node) {
  const OpDescPtr data_desc = MakeShared<OpDesc>(kMultiBatchDataNode, DATA);
  if (data_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch data node failed");
    return FAILED;
  }

  GeTensorDesc data_tensor(GeShape({static_cast<int64_t>(batch_shapes_[0].size())}), FORMAT_ND, DT_INT32);
  if (data_desc->AddInputDesc(data_tensor) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add input desc failed");
    return FAILED;
  }
  if (data_desc->AddOutputDesc(data_tensor) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add output desc failed");
    return FAILED;
  }

  size_t data_index = all_data_nodes_.size();
  (void)AttrUtils::SetInt(data_desc, ATTR_NAME_INDEX, data_index);
  (void)AttrUtils::SetBool(data_desc, ATTR_INSERT_BY_MBATCH, true);

  node = graph->AddNode(data_desc);
  if (node == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch data node failed");
    return OUT_OF_MEMORY;
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Create index const node for root graph.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @param [in] NodePtr node: index const node.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CreateIndexConstNode(const ComputeGraphPtr &graph, NodePtr &node) {
  const OpDescPtr const_desc = MakeShared<OpDesc>(kMultiBatchConstNode, CONSTANT);
  if (const_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch const node failed");
    return FAILED;
  }

  int64_t count = batch_shapes_.size() * batch_shapes_[0].size();
  std::unique_ptr<int32_t[]> addr(new (std::nothrow) int32_t[count]);
  GE_CHECK_NOTNULL(addr);

  size_t i = 0;
  for (auto &batch_shape : batch_shapes_) {
    for (int64_t dim : batch_shape) {
      addr[i++] = static_cast<int32_t>(dim);
    }
  }

  GeTensorDesc const_tensor(GeShape({count}), FORMAT_ND, DT_INT32);
  GeTensor tensor(const_tensor);
  (void)tensor.SetData(reinterpret_cast<uint8_t *>(addr.get()), count * sizeof(int32_t));
  if (!AttrUtils::SetTensor(const_desc, ATTR_NAME_WEIGHTS, tensor)) {
    GELOGE(OUT_OF_MEMORY, "Failed to init tensor value for const %s", const_desc->GetName().c_str());
    return FAILED;
  }

  if (const_desc->AddOutputDesc(const_tensor) != GRAPH_SUCCESS) {
    GELOGE(OUT_OF_MEMORY, "Failed to add output desc for const node %s", const_desc->GetName().c_str());
    return FAILED;
  }

  node = graph->AddNode(const_desc);
  if (node == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch const node failed");
    return OUT_OF_MEMORY;
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Create index node for root graph.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CreateIndexNode(const ComputeGraphPtr &graph) {
  // Data --> MapIndex --> Case
  NodePtr data_node;
  GE_CHK_STATUS_RET(CreateIndexDataNode(graph, data_node), "Create data node failed");

  NodePtr const_node;
  GE_CHK_STATUS_RET(CreateIndexConstNode(graph, const_node), "Create const node failed");

  OpDescBuilder op_builder(kMultiBatchMapIndexNode, "MapIndex");
  op_builder.AddInput("x", data_node->GetOpDesc()->GetOutputDesc(0))
      .AddInput("data_seq", const_node->GetOpDesc()->GetOutputDesc(0))
      .AddOutput("y", GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));

  const OpDescPtr op_desc = op_builder.Build();
  if (op_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch index desc failed");
    return FAILED;
  }
  NodePtr index_node = graph->AddNode(op_desc);
  if (index_node == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch index node failed");
    return OUT_OF_MEMORY;
  }

  if (GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), index_node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to add edge between node:%s to MapIndex:%s", data_node->GetName().c_str(),
           index_node->GetName().c_str());
    return FAILED;
  }
  if (GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), index_node->GetInDataAnchor(1)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to add edge between node:%s to MapIndex:%s", const_node->GetName().c_str(),
           index_node->GetName().c_str());
    return FAILED;
  }
  if (GraphUtils::AddEdge(index_node->GetOutDataAnchor(0), case_node_->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to add edge between MapIndex:%s to Case:%s", index_node->GetName().c_str(),
           case_node_->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Create input node for root graph.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CreateInputNode(const ComputeGraphPtr &graph) {
  // Data --> Case
  std::vector<NodePtr> all_data_nodes;
  const size_t arg_index = kCaseArgIndex;
  for (size_t i = 0; i < all_data_nodes_.size(); ++i) {
    const auto &node = all_data_nodes_[i];
    const OpDescPtr op_desc = AttrUtils::CopyOpDesc(node->GetOpDesc());
    if (op_desc == nullptr) {
      GELOGE(OUT_OF_MEMORY, "Create multi-batch Data node failed, name: %s", node->GetName().c_str());
      return FAILED;
    }

    if (GraphUtils::CopyTensorAttrs(op_desc, node) != GRAPH_SUCCESS) {
      return FAILED;
    }

    op_desc->SetName(node->GetName());
    const NodePtr &data = graph->AddNode(op_desc);
    GE_CHK_BOOL_EXEC(data != nullptr, return FAILED, "Add node[%s] to graph failed", op_desc->GetName().c_str());
    if (GraphUtils::AddEdge(data->GetOutDataAnchor(0), case_node_->GetInDataAnchor(arg_index + i)) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Failed to add edge between Data:%s to Case:%s",
             data->GetName().c_str(), case_node_->GetName().c_str());
      return FAILED;
    }

    if (SetMaxShapeToData(data) != SUCCESS) {
      return FAILED;
    }
    all_data_nodes.emplace_back(data);
  }

  all_data_nodes_.swap(all_data_nodes);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Create Const node for root graph.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CreateConstNode(const ComputeGraphPtr &graph) {
  // Const --> Case
  std::vector<NodePtr> all_const_nodes;
  const size_t arg_index = kCaseArgIndex + all_data_nodes_.size();
  for (size_t i = 0; i < all_const_nodes_.size(); ++i) {
    const auto &node = all_const_nodes_[i];
    const OpDescPtr op_desc = AttrUtils::CopyOpDesc(node->GetOpDesc());
    if (op_desc == nullptr) {
      GELOGE(OUT_OF_MEMORY, "Create multi-batch Const node failed, name: %s", node->GetName().c_str());
      return FAILED;
    }

    op_desc->SetName(node->GetName());
    if (GraphUtils::CopyTensorAttrs(op_desc, node) != GRAPH_SUCCESS) {
      return FAILED;
    }

    const NodePtr &data = graph->AddNode(op_desc);
    GE_CHK_BOOL_EXEC(data != nullptr, return FAILED, "Add node[%s] to graph failed", op_desc->GetName().c_str());
    if (GraphUtils::AddEdge(data->GetOutDataAnchor(0), case_node_->GetInDataAnchor(arg_index + i)) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Failed to add edge between Const:%s to Case:%s",
             data->GetName().c_str(), case_node_->GetName().c_str());
      return FAILED;
    }
    all_const_nodes.emplace_back(data);
  }

  size_t data_index = all_data_nodes_.size();
  for (size_t i = 0; i < all_const_nodes_.size(); ++i, ++data_index) {  // Trans subgraph Const to Data.
    const OpDescPtr &op_desc = all_const_nodes_[i]->GetOpDesc();
    op_desc->SetType(DATA);
    (void)op_desc->DelAttr(ATTR_NAME_WEIGHTS);  // Delete weight.

    // Const no InputDesc, Data need InputDesc.
    (void)op_desc->AddInputDesc(op_desc->GetOutputDesc(kDataOutIndex));
    (void)AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, data_index);
    (void)NodeUtils::AppendInputAnchor(all_const_nodes_[i], 1);
  }

  all_const_nodes_.swap(all_const_nodes);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Create output node for root graph.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CreateOutputNode(const ComputeGraphPtr &graph) {
  const auto &output = all_output_nodes_[0];
  const OpDescPtr op_desc = AttrUtils::CopyOpDesc(output->GetOpDesc());
  if (op_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch output node failed");
    return FAILED;
  }

  if (GraphUtils::CopyTensorAttrs(op_desc, output) != GRAPH_SUCCESS) {
    return FAILED;
  }

  op_desc->SetName(output->GetName());
  const NodePtr &node = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(node != nullptr, return FAILED, "Add node[%s] to graph failed", op_desc->GetName().c_str());

  for (size_t i = 0; i < case_node_->GetAllOutDataAnchorsSize(); ++i) {
    const auto it = direct_output_.find(i);
    if (it == direct_output_.end()) {
      if (GraphUtils::AddEdge(case_node_->GetOutDataAnchor(i), node->GetInDataAnchor(i)) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Failed to add edge between Case:%s to NetOutput:%s",
               case_node_->GetName().c_str(), node->GetName().c_str());
        return FAILED;
      }
    } else {
      const auto data_node = graph->FindNode(it->second);
      if (data_node == nullptr) {
        GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "Data node:%s not found", it->second.c_str());
        return GE_GRAPH_GRAPH_NODE_NULL;
      }
      if (GraphUtils::AddEdge(data_node->GetOutDataAnchor(kDataOutIndex), node->GetInDataAnchor(i)) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Failed to add edge between Data:%s to NetOutput:%s",
               data_node->GetName().c_str(), node->GetName().c_str());
        return FAILED;
      }
    }
  }

  all_output_nodes_.clear();
  all_output_nodes_.emplace_back(node);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Set max shape to Data node in root graph.
/// @param [in] const NodePtr &data: data in Root/Case graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::SetMaxShapeToData(const NodePtr &data) {
  auto data_shape = NodeUtils::GetOutputDesc(*data, kDataOutIndex).GetShape();
  auto data_name = data->GetName();
  const auto &dims = data_shape.GetDims();
  if (std::all_of(dims.begin(), dims.end(), [](int64_t val) { return val >= 0; })) {
    return SUCCESS;
  }
  (void)AttrUtils::SetListInt(data->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, data_shape.GetDims());

  GeTensorDesc tensor(NodeUtils::GetOutputDesc(*data, kDataOutIndex));
  std::vector<std::string> input_dims_str;
  for (size_t i = 0; i < batch_shapes_.size(); ++i) {
    auto shape = data_shape;
    auto ret = CalcShape(data_to_dynamic_info_.at(data_name).at(i), shape);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to calculate the batched shape for data node %s, the shapes may not match",
			 data->GetName().c_str());
      return ret;
    }
    tensor.SetShape(shape);
    string input_str;
    int64_t tensor_size = 0;
	(void)TensorUtils::GetTensorSizeInBytes(tensor, tensor_size);
    input_str = TypeUtils::FormatToSerialString(tensor.GetFormat()) + ":" +
				TypeUtils::DataTypeToSerialString(tensor.GetDataType()) + ":" + data->GetName() + ":" +
			    std::to_string(tensor_size) + ":" + std::to_string(tensor.GetShape().GetDimNum()) + ":" +
			    formats::JoinToString(tensor.GetShape().GetDims());
    input_dims_str.emplace_back(input_str);
  }
  (void)AttrUtils::SetListStr(data->GetOpDesc(), "_all_origin_gears_inputs", input_dims_str);

  size_t max_shape_index = 0;
  int64_t max_size = 0;
  for (size_t i = 0; i < batch_shapes_.size(); ++i) {
    int64_t size = 1;
    for (auto dim : data_to_dynamic_info_.at(data_name).at(i)) {
      if (INT64_MAX / dim < size) {
        GELOGE(PARAM_INVALID, "The shape %s size overflow",
               formats::ShapeToString(data_to_dynamic_info_.at(data_name).at(i)).c_str());
        return PARAM_INVALID;
      }
      size *= dim;
    }
    if (size > max_size) {
      max_size = size;
      max_shape_index = i;
    }
  }

  return SetShapeToData(data_to_dynamic_info_.at(data_name).at(max_shape_index), data, data_shape);
}

///
/// @ingroup ge
/// @brief Update Data node in Subgraph.
/// @param [in] const NodePtr &data: data in Subgraph.
/// @param [in] size_t index: The batch index.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::UpdateSubgraphData(const NodePtr &data, size_t index) {
  int node_index = -1;
  if (!AttrUtils::GetInt(data->GetOpDesc(), ATTR_NAME_INDEX, node_index)) {
    GELOGE(FAILED, "Failed to get index from data[%s]", data->GetName().c_str());
    return FAILED;
  }

  int parent_index = node_index + 1;
  if (!AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(FAILED, "Failed to set parent index for node %s", data->GetName().c_str());
    return FAILED;
  }

  auto data_shape = NodeUtils::GetOutputDesc(*data, kDataOutIndex).GetShape();
  const auto &dims = data_shape.GetDims();
  if (std::all_of(dims.begin(), dims.end(), [](int64_t val) { return val >= 0; })) {
    return SUCCESS;
  }

  (void)AttrUtils::SetListInt(data->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, data_shape.GetDims());
  auto data_name = data->GetName();
  size_t pos = data_name.find(kMultiBatchNodePostfix);
  if (pos == string::npos) {
    GELOGE(FAILED, "Cannot find key string [%s] of multi-batch in name of virtual input node, node name: %s.",
           kMultiBatchNodePostfix.c_str(), data_name.c_str());
    return FAILED;
  }

  auto parent_name = data_name.substr(0, pos);
  return SetShapeToData(data_to_dynamic_info_.at(parent_name).at(index), data, data_shape);
}

///
/// @ingroup ge
/// @brief Set max shape to Data node in root graph.
/// @param [in] const std::vector<int64_t> &shapes: dims of shape.
/// @param [in] const NodePtr &data: data in Root/Case graph.
/// @param [in] GeShape &data_shape: dims of data node.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::SetShapeToData(const vector<int64_t> &shapes, const NodePtr &data, GeShape &data_shape) {
  // must not be error, the calc result has been checked in function InsertSwitchNForData
  if (multibatch::CalcShape(shapes, data_shape) != SUCCESS) {
    return INTERNAL_ERROR;
  }

  if (NodeUtils::UpdateInputShape(*data, kDataInIndex, data_shape) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to update input shape for data %s", data->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (NodeUtils::UpdateOutputShape(*data, kDataOutIndex, data_shape) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to update output shape for data %s", data->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGI("Update %s input/output shape to %s", data->GetName().c_str(), formats::ShapeToString(data_shape).c_str());
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Create nodes for root graph.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @param [in] const ComputeGraphPtr &branch: original graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CreateSubgraphs(const ComputeGraphPtr &graph, const ComputeGraphPtr &branch) {
  const auto &op_desc = case_node_->GetOpDesc();
  for (size_t i = 0; i < batch_shapes_.size(); ++i) {
    std::vector<NodePtr> input_nodes;
    std::vector<NodePtr> output_nodes;
    const std::string postfix = kMultiBatchNodePostfix + std::to_string(i);
    ComputeGraphPtr subgraph = (i == 0) ? branch : GraphUtils::CloneGraph(branch, postfix, input_nodes, output_nodes);
    if (subgraph == nullptr) {
      GELOGE(FAILED, "Create multi-batch case node failed");
      return FAILED;
    }

    subgraph->SetName("Batch_" + std::to_string(i));
    subgraph->SetParentNode(case_node_);
    subgraph->SetParentGraph(graph);
    graph->AddSubgraph(subgraph->GetName(), subgraph);
    all_branch_output_[subgraph] = subgraph->FindFirstNodeMatchType(NETOUTPUT);
    GE_CHK_STATUS_RET(UpdateSubgraphOutput(all_branch_output_[subgraph]),
		      "Update %s failed", all_branch_output_[subgraph]->GetName().c_str());

    const string key_name = "branches" + std::to_string(i);
    op_desc->AddSubgraphName(key_name);
    op_desc->SetSubgraphInstanceName(i, subgraph->GetName());

    for (const auto &data : input_nodes) {
      GE_CHK_STATUS_RET(UpdateSubgraphData(data, i), "Update %s failed", subgraph->GetName().c_str());
    }
  }

  // Origninal graph take as first subgraph, update node name.
  for (const auto &n : branch->GetDirectNode()) {
    const auto &op_desc = n->GetOpDesc();
    op_desc->SetName(n->GetName() + kMultiBatchNodePostfix + "0");
    if (n->GetType() == DATA) {
      GE_CHK_STATUS_RET(UpdateSubgraphData(n, 0), "Update %s failed", branch->GetName().c_str());
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Update output_node in Subgraph.
/// @param [in] const NodePtr &output_node: output_node in Subgraph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::UpdateSubgraphOutput(const NodePtr &output_node) {
  const auto &op_desc = output_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (size_t index = 0; index < op_desc->GetInputsSize(); ++index) {
    GeTensorDescPtr tensor = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(tensor);
    if (!AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, index)) {
      GELOGE(FAILED, "Failed to set parent index for node %s", output_node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Remove subgraph suspend output anchor.
/// @param [in] ComputeGraphPtr &graph: Parent compute graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::PruneDirectOutput(const ComputeGraphPtr &graph) {
  const auto &func_desc = case_node_->GetOpDesc();
  uint32_t unused_num = 0;
  uint32_t output_num = func_desc->GetOutputsSize();
  for (size_t i = 0; i < output_num; ++i) {
    bool is_unused_tensor = true;
    for (const auto &item : all_branch_output_) {
      const auto &netoutput = item.second;
      GE_CHECK_NOTNULL(netoutput);
      const auto in_anchor = netoutput->GetInDataAnchor(i);
      if (in_anchor->GetPeerOutAnchor() != nullptr) {
        is_unused_tensor = false;
        break;
      }
    }

    if (is_unused_tensor) {
      unused_num++;
      continue;
    }

    GE_CHK_STATUS_RET(UpdateOutputTensor(i, unused_num), "Graph:%s Update output failed", graph->GetName().c_str());
  }

  if (unused_num == 0) {
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(NodeUtils::RemoveOutputAnchor(case_node_, output_num - unused_num), "Remove output failed");
  for (const auto &item : all_branch_output_) {
    GE_CHK_STATUS_RET(NodeUtils::RemoveInputAnchor(item.second, output_num - unused_num), "Remove input failed");
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Update subgraph suspend output tensor.
/// @param [in] parent_index: parent index for check.
/// @param [in] unused_num: total unused tensor.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::UpdateOutputTensor(uint32_t parent_index, uint32_t unused_num) {
  if (unused_num == 0) {
    return SUCCESS;
  }

  uint32_t update_index = parent_index - unused_num;
  for (const auto &item : all_branch_output_) {
    const auto &node = item.second;
    const auto &new_anchor = node->GetInDataAnchor(update_index);
    const auto &old_anchor = node->GetInDataAnchor(parent_index);
    const auto &out_anchor = old_anchor->GetPeerOutAnchor();
    const auto &out_node = out_anchor->GetOwnerNode();

    const auto &op_desc = node->GetOpDesc();
    (void)op_desc->UpdateInputDesc(update_index, op_desc->GetInputDesc(parent_index));

    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(out_anchor, new_anchor), "Add edge failed");
    GELOGI("Add edge success, func node: %s, node: %s, parent index: %u, update index: %u",
           case_node_->GetName().c_str(), out_node->GetName().c_str(), parent_index, update_index);

    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(out_anchor, old_anchor), "Remove edge failed");
    GELOGI("Remove edge success, func node: %s, node: %s", case_node_->GetName().c_str(), out_node->GetName().c_str());
  }

  const auto &new_anchor = case_node_->GetOutDataAnchor(update_index);
  const auto &old_anchor = case_node_->GetOutDataAnchor(parent_index);
  for (const auto in_anchor : old_anchor->GetPeerInDataAnchors()) {
    const auto &in_node = in_anchor->GetOwnerNode();
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(old_anchor, in_anchor), "Remove edge failed");
    GELOGI("Remove edge success, func node: %s, node: %s", case_node_->GetName().c_str(), in_node->GetName().c_str());

    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(new_anchor, in_anchor), "Add edge failed");
    GELOGI("Add edge success, func node: %s, node: %s, parent index: %u, update index: %u",
           case_node_->GetName().c_str(), in_node->GetName().c_str(), parent_index, update_index);
  }

  return SUCCESS;
}
}  // namespace ge
