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
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_registry.h"
#include "graph/common/omg_util.h"

namespace ge {
namespace {
constexpr uint8_t kDataInIndex = 0;
constexpr uint8_t kDataOutIndex = 0;
constexpr uint8_t kCaseArgIndex = 1;
const int kDivisionConst = 2;
const size_t kNumOfGetnextNode = 1;

const std::string kMultiBatchCaseNode = "ascend_mbatch_shape_case";
const std::string kMultiBatchDataNode = "ascend_mbatch_shape_data";
const std::string kMultiBatchGetDynamicDimsNode = "ascend_mbatch_get_dynamic_dims_node";
const std::string kMultiBatchConstNode = "ascend_mbatch_shape_const";
const std::string kMultiBatchMapIndexNode = "ascend_mbatch_shape_mapindex";
const std::string kMultiBatchNodePostfix = "_ascend_mbatch_batch_";
const char *const kGetNextName = "IteratorV2";
}  // namespace

inline bool IsGetNextType(const NodePtr &node) {
  std::string original_type;
  GE_IF_BOOL_EXEC(GetOriginalType(node, original_type) != SUCCESS,
                  GELOGW("Get original type failed."); return false);
  return (original_type == kGetNextName);
}

Status MultiBatchClonePass::Run(ComputeGraphPtr graph) {
  GE_IF_BOOL_EXEC(graph == nullptr, GELOGE(FAILED, "Original graph is nullptr"); return FAILED);
  if (graph->GetParentGraph() != nullptr) {
    GELOGD("Subgraph %s skip the MultiBatchClonePass", graph->GetName().c_str());
    return SUCCESS;
  }
  if (!GetLocalOmgContext().need_multi_batch) {
    GELOGI("No need to process_multi for no_train graph.");
    return SUCCESS;
  }
  std::vector<NodePtr> data_nodes;
  std::vector<NodePtr> getnext_nosink_nodes;
  std::vector<NodePtr> getnext_sink_nodes;
  if (multibatch::CheckSequenceOfOptions(graph, data_nodes, getnext_nosink_nodes, getnext_sink_nodes) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Train_Dynamic] CheckSequenceOfOptions failed.");
    return PARAM_INVALID;
  }
  if (multibatch::UpdateNameOfInputShape(graph, data_nodes, getnext_nosink_nodes, getnext_sink_nodes) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Train_Dynamic] UpdateNameForInputShapeOfOption failed.");
    return PARAM_INVALID;
  }
  if (multibatch::DeleteIdentityInsertByAdapter(graph) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Train_Dynamic] DeleteIdentityInsertByAdapter failed.");
    return PARAM_INVALID;
  }
  if (!multibatch::InitDynamicParams(batch_shapes_)) {
    GELOGD("There is no multi-batch options, no need clone multi-batch graph");
    return SUCCESS;
  }
  if (multibatch::CheckNegativeCountOfOptions(batch_shapes_) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Train_Dynamic] Input_shape and dynamic_dims should set correct params.");
    return PARAM_INVALID;
  }
  GELOGD("Begin to run Multi-batch clone on graph: %s", graph->GetName().c_str());
  GE_CHK_STATUS_RET(multibatch::CheckDynamicParams(batch_shapes_), "Invalid multi-batch param");
  if (CollectIoNodes(graph) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Collect input output nodes failed");
    return INTERNAL_ERROR;
  }

  // parser data dynamic info from atc parameter --input_shape
  if (CheckAndParseDynamicData() != SUCCESS) {
    GELOGE(PARAM_INVALID, "Parse each data's own dynamic info failed");
    return PARAM_INVALID;
  }

  (void)AttrUtils::GetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id_);
  ComputeGraphPtr branch = MakeShared<ComputeGraph>(graph->GetName());
  GE_IF_BOOL_EXEC(branch == nullptr, GELOGE(OUT_OF_MEMORY, "Create multi batch graph failed"); return OUT_OF_MEMORY);
  (void)AttrUtils::SetStr(branch, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id_);

  graph->InValid();  // Will modify, need topological again.
  graph->Swap(*branch);
  GE_CHK_STATUS_RET(CreateRootGraph(graph), "Construct root graph failed.");
  GE_CHK_STATUS_RET(CreateOriGraph(branch), "Construct original graph failed.")
  GE_CHK_STATUS_RET(CreateSubgraphs(graph, branch), "Construct subgraph failed.");

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
    if (!GetLocalOmgContext().dynamic_node_type.empty() && IsGetNextType(node)) {
      all_data_nodes_.emplace_back(node);
      GE_CHK_STATUS_RET(InitParamsOfGetNext(node), "Init params of %s failed.", node->GetName().c_str());
    }
    if (node->GetType() == DATA) {
      all_data_nodes_.emplace_back(node);
    } else if (node->GetType() == CONSTANT || node->GetType() == CONSTANTOP) {
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
  size_t getnext_node_count = 0;
  for (size_t i = 0; i < all_data_nodes_.size(); ++i) {
    if (IsGetNextType(all_data_nodes_[i])) {
      // just one getnext node in graph
      getnext_node_count++;
      continue;
    }
    const auto &op_desc = all_data_nodes_[i]->GetOpDesc();
    if (!AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, data_index)) {
      (void)AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, i - getnext_node_count);
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
  GELOGD("Data count is %zu, const count is %zu, getnext count is %zu, output count is %zu, direct out count is %zu.",
         all_data_nodes_.size(), all_const_nodes_.size(), getnext_node_count, all_output_nodes_.size(),
         direct_output_.size());

  return SUCCESS;
}

Status MultiBatchClonePass::CheckAndParseDynamicData(){
  size_t unknown_shape_count = 0;
  auto data_name_and_shape = GetLocalOmgContext().user_input_dims;
  std::vector<std::string> data_name_order;
  for (auto &item : data_name_and_shape) {
    data_name_order.push_back(item.first);
  }
  if (!getnext_sink_dynamic_dims_) {
    for (const auto &node : all_data_nodes_) {
      auto data_desc = NodeUtils::GetOutputDesc(*node, kDataOutIndex);
      auto data_shape = data_desc.GetShape();
      auto data_format = data_desc.GetFormat() == Format::FORMAT_NCHW ? "NCHW" :
                         data_desc.GetFormat() == Format::FORMAT_NHWC ? "NHWC" : "Others";
      auto data_name = node->GetName();

      std::vector<int64_t> data_shape_dims = data_shape.GetDims();
      if (std::all_of(data_shape_dims.begin(), data_shape_dims.end(), [](int64_t val) { return val >= 0; })) {
        continue;
      }
      ++unknown_shape_count;
      auto iter = find(data_name_order.begin(), data_name_order.end(), data_name);
      if (iter == data_name_order.end()) {
        if (!GetLocalOmgContext().dynamic_batch_size.empty()) {
          auto ret = multibatch::CheckDynamicBatchShape(data_shape_dims, data_name);
          GE_IF_BOOL_EXEC(ret == false, GELOGE(PARAM_INVALID, "Failed to check dynamic batch shape of %s.",
                                               data_name.c_str()); return PARAM_INVALID);
        } else if (!GetLocalOmgContext().dynamic_image_size.empty()) {
          auto ret = multibatch::CheckDynamicImageSizeShape(data_shape_dims, data_name, data_format);
          GE_IF_BOOL_EXEC(ret == false, GELOGE(PARAM_INVALID, "Failed to check dynamic image size shape of %s.",
                                               data_name.c_str()); return PARAM_INVALID);
        } else if (!GetLocalOmgContext().dynamic_dims.empty()) {
          ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "reason"},
            {"--input_shape", "all dynamic data must be set in --input_shape"});
          GELOGE(INTERNAL_ERROR, "data: %s shape:%s must be set int --input_shape",
                 node->GetName().c_str(), data_shape.ToString().c_str());
          return INTERNAL_ERROR;
        }
        data_name_and_shape.emplace_back(data_name, data_shape_dims);
      }
    }
  }
  auto ret = multibatch::ParserDataToDynmaicInfo(batch_shapes_, data_name_and_shape, data_to_dynamic_info_);
  GE_CHK_STATUS_RET(ret, "Failed to parse data to dynamic info.");
  if (!getnext_sink_dynamic_dims_ && unknown_shape_count == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10040");
    GELOGE(PARAM_INVALID,
           "Need unknow shape data when user set --dynamic_batch_size, --dynamic_image_size or --dynamic_dims");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status MultiBatchClonePass::InitParamsOfGetNext(const NodePtr &node) {
  data_count_from_getnext_ = 0;
  getnext_sink_dynamic_dims_ = false;
  GE_CHECK_NOTNULL(node->GetOpDesc());
  data_count_from_getnext_ = node->GetOpDesc()->GetOutputsSize();
  if (GetLocalOmgContext().dynamic_node_type == GETNEXT) {
    data_count_from_getnext_ = data_count_from_getnext_ / kDivisionConst;
    for (size_t i = 0; i < data_count_from_getnext_; ++i) {
      GeTensorDesc output_desc = node->GetOpDesc()->GetOutputDesc(i);
      GELOGD("The %zu data shape from getnext sink is %s.", i,
             formats::JoinToString(output_desc.GetShape().GetDims()).c_str());
      const auto &dims = output_desc.GetShape().GetDims();
      if (std::all_of(dims.begin(), dims.end(), [](int64_t val) {return val >= 0; })) {
        GELOGD("The %zu data from %s is static.", i, node->GetName().c_str());
      } else {
        getnext_sink_dynamic_dims_ = true;
        GELOGD("Dynamic dims in the pattern of getnext sink.");
      }
    }
  }
  if (node->GetOutControlAnchor() != nullptr) {
    for (const auto &peer_in_control_anchor : node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      NodePtr next_node = peer_in_control_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(next_node);
      if (next_node->GetType() == CONSTANTOP) {
        out_control_nodes_.insert(next_node);
        GELOGD("Control edge: %s connect with %s.", node->GetName().c_str(), next_node->GetName().c_str());
      }
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
  GELOGD("Start create root graph of %s.", graph->GetName().c_str());
  uint32_t input_num = all_data_nodes_.size() + all_const_nodes_.size();
  if (data_count_from_getnext_ != 0) {
    input_num = input_num + data_count_from_getnext_ - kNumOfGetnextNode;
  }
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
  if (!AttrUtils::SetBool(op_desc, ATTR_INSERT_BY_MBATCH, true)) {
    GELOGE(INTERNAL_ERROR, "Failed to add insert attr on case node %s", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
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
Status MultiBatchClonePass::CreateIndexDataNode(const ComputeGraphPtr &graph, NodePtr &shape_node) {
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
  data_index = data_count_from_getnext_ != 0 ? data_index - kNumOfGetnextNode : data_index;
  (void)AttrUtils::SetInt(data_desc, ATTR_NAME_INDEX, data_index);
  (void)AttrUtils::SetBool(data_desc, ATTR_INSERT_BY_MBATCH, true);

  shape_node = graph->AddNode(data_desc);
  if (shape_node == nullptr) {
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
  // Data/GetDynamicDims --> MapIndex --> Case
  if (!getnext_sink_dynamic_dims_) {
    GE_CHK_STATUS_RET(CreateIndexDataNode(graph, shape_node_), "Create data node failed");
  } else {
    GE_CHK_STATUS_RET(CreateGetDynamicDimsNode(graph, shape_node_), "Create get dynamic dims node failed");
  }

  NodePtr const_node;
  GE_CHK_STATUS_RET(CreateIndexConstNode(graph, const_node), "Create const node failed");
  GELOGD("Shape node name is %s, type is %s, const node name is %s.", shape_node_->GetName().c_str(),
         shape_node_->GetType().c_str(), const_node->GetName().c_str());
  OpDescBuilder op_builder(kMultiBatchMapIndexNode, "MapIndex");
  op_builder.AddInput("x", shape_node_->GetOpDesc()->GetOutputDesc(0))
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

  GE_CHK_STATUS_RET(AddAttrForGetDynamicDims(shape_node_), "Failed to add attr for %s.",
                    shape_node_->GetName().c_str());
  if (GraphUtils::AddEdge(shape_node_->GetOutDataAnchor(0), index_node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to add edge between node:%s to MapIndex:%s", shape_node_->GetName().c_str(),
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

Status MultiBatchClonePass::CreateGetDynamicDimsNode(const ComputeGraphPtr &graph, NodePtr &shape_node) {
  const OpDescPtr data_desc = MakeShared<OpDesc>(kMultiBatchGetDynamicDimsNode, GETDYNAMICDIMS);
  if (data_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch get dynamic dims node failed");
    return OUT_OF_MEMORY;
  }

  // input of GetDynamicDims is shape_of_each_data, output is gear_info
  for (size_t i = 0; i < GetLocalOmgContext().user_input_dims.size(); ++i) {
    size_t input_shape_dims = GetLocalOmgContext().user_input_dims.at(i).second.size();
    // add input desc without GeShape for const input, value of input_shape is 1 transferred by adapter
    if (input_shape_dims == 1 && GetLocalOmgContext().user_input_dims.at(i).second.at(0) == 0) {
      GeTensorDesc tensor_desc;
      tensor_desc.SetFormat(FORMAT_ND);
      tensor_desc.SetDataType(DT_INT32);
      auto ret = data_desc->AddInputDesc(tensor_desc);
      GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to add input desc for created data");
          return FAILED);
      continue;
    }
    GeTensorDesc tensor_desc(GeShape({static_cast<int32_t>(input_shape_dims)}), FORMAT_ND, DT_INT32);
    auto ret = data_desc->AddInputDesc(tensor_desc);
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to add input desc for created data");
        return FAILED);
  }
  GeTensorDesc tensor_desc(GeShape({static_cast<int32_t>(batch_shapes_.at(0).size())}), FORMAT_ND, DT_INT32);
  auto ret = data_desc->AddOutputDesc(tensor_desc);
  GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to add output desc for created data");
      return FAILED);

  (void)AttrUtils::SetBool(data_desc, ATTR_INSERT_BY_MBATCH, true);

  shape_node = graph->AddNode(data_desc);
  if (shape_node == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create multi-batch dynamic dims node failed");
    return OUT_OF_MEMORY;
  }
  return SUCCESS;
}

Status MultiBatchClonePass::AddAttrForGetDynamicDims(const NodePtr &shape_node) {
  if (!getnext_sink_dynamic_dims_) {
    GELOGD("No need to add attr when not insert get dynamic dims node.");
    return SUCCESS;
  }
  GELOGD("Add attr for :%s, type is %s:", shape_node->GetName().c_str(), shape_node->GetType().c_str());
  if (!AttrUtils::SetInt(shape_node->GetOpDesc(), ATTR_GETNEXT_SINK_DATA_COUNT, data_count_from_getnext_)) {
    GELOGE(INTERNAL_ERROR, "set ATTR_GETNEXT_SINK_DATA_COUNT failed");
    return INTERNAL_ERROR;
  }
  vector<int64_t> shape_info;
  for (size_t i = 0; i < GetLocalOmgContext().user_input_dims.size(); ++i) {
    if (GetLocalOmgContext().user_input_dims.at(i).second.size() == 1 &&
        GetLocalOmgContext().user_input_dims.at(i).second.at(0) == 0) {
      shape_info.emplace_back(0);
      continue;
    }
    shape_info.emplace_back(GetLocalOmgContext().user_input_dims.at(i).second.size());
    for (size_t j = 0; j < GetLocalOmgContext().user_input_dims.at(i).second.size(); ++j) {
      shape_info.emplace_back(GetLocalOmgContext().user_input_dims.at(i).second.at(j));
    }
  }
  if (!AttrUtils::SetListInt(shape_node->GetOpDesc(), ATTR_GETNEXT_SINK_SHAPE_INFO, shape_info)) {
    GELOGE(INTERNAL_ERROR, "set ATTR_GETNEXT_SINK_SHAPE_INFO failed");
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status MultiBatchClonePass::LinkGetNextToGetDynamicDims(const NodePtr &getnext_node, const NodePtr &shape_node) {
  GELOGD("Start relink shape anchor of %s to %s.", getnext_node->GetName().c_str(), shape_node->GetName().c_str());
  size_t input_index = 0;
  size_t data_count = getnext_node->GetAllOutDataAnchors().size() / kDivisionConst;
  for (size_t out_index = data_count; out_index < getnext_node->GetAllOutDataAnchors().size(); ++out_index,
      ++input_index) {
    GELOGD("Start add %s of %zu out_anchor to %s of %zu in_anchor.", getnext_node->GetName().c_str(), out_index,
           shape_node->GetName().c_str(), input_index);
    auto out_data_anchor =  getnext_node->GetOutDataAnchor(out_index);
    auto ret = GraphUtils::AddEdge(out_data_anchor, shape_node->GetInDataAnchor(input_index));
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to link getnext %s to getdynamicdims %s",
                                                 getnext_node->GetName().c_str(), shape_node->GetName().c_str());
        return INTERNAL_ERROR);
  }
  return SUCCESS;
}

Status MultiBatchClonePass::LinkGetDynamicDimsToNetOutput(const NodePtr &output_node) {
  if (!GetLocalOmgContext().dynamic_node_type.empty()) {
    if (!AttrUtils::SetStr(output_node->GetOpDesc(), ATTR_ALL_GEARS_INFO, GetLocalOmgContext().dynamic_dims)) {
      GELOGE(INTERNAL_ERROR, "Failed to set all gears info attr on netoutput %s.", output_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }
  if (getnext_sink_dynamic_dims_) {
    GELOGD("Start link %s to %s.", shape_node_->GetName().c_str(), output_node->GetName().c_str());
    size_t input_index = output_node->GetAllInDataAnchors().size();
    if (NodeUtils::AppendInputAnchor(output_node, input_index + 1) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Append input anchor of %s of %zu failed.", output_node->GetName().c_str(), input_index);
      return INTERNAL_ERROR;
    }
    auto ret = GraphUtils::AddEdge(shape_node_->GetOutDataAnchor(kDataOutIndex),
                                   output_node->GetInDataAnchor(input_index));
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to link netoutput %s to getdynamicdims %s",
                                                 output_node->GetName().c_str(), shape_node_->GetName().c_str());
        return INTERNAL_ERROR);
    if (!AttrUtils::SetBool(output_node->GetOpDesc(), ATTR_GETNEXT_SINK_DYNMAIC, true)) {
      GELOGE(INTERNAL_ERROR, "Failed to set getnext sink dynamic attr on netoutput %s.",
             output_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
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
  size_t case_input_index = kCaseArgIndex;
  NodePtr getnext_node = nullptr;
  size_t input_index_of_getnext = 0;
  for (size_t i = 0; i < all_data_nodes_.size(); ++i, ++case_input_index) {
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
    if (IsGetNextType(node)) {
      getnext_node = data;
      input_index_of_getnext = case_input_index;
      case_input_index = case_input_index + data_count_from_getnext_;
      continue;
    } else {
      if (GraphUtils::AddEdge(data->GetOutDataAnchor(0), case_node_->GetInDataAnchor(case_input_index)) !=
          GRAPH_SUCCESS) {
        GELOGE(FAILED, "Failed to add edge between Data:%s to Case:%s", data->GetName().c_str(),
               case_node_->GetName().c_str());
        return FAILED;
      }
    }

    if (SetMaxShape(data) != SUCCESS) {
      GELOGE(FAILED, "Set max shape of %s failed.", data->GetName().c_str());
      return FAILED;
    }
    all_data_nodes.emplace_back(data);
  }
  if (getnext_node != nullptr) {
    if (LinkEdgeForGetNext(getnext_node, input_index_of_getnext) != SUCCESS) {
      GELOGE(FAILED, "Failed to link edge for %s.", getnext_node->GetName().c_str());
      return FAILED;
    }
    if (SetMaxShape(getnext_node) != SUCCESS) {
      GELOGE(FAILED, "Set max shape of %s failed.", getnext_node->GetName().c_str());
      return FAILED;
    }
    all_data_nodes.emplace_back(getnext_node);
  }

  all_data_nodes_.swap(all_data_nodes);
  return SUCCESS;
}

Status MultiBatchClonePass::LinkEdgeForGetNext(const NodePtr &getnext_node, size_t &case_input_index) {
  GELOGD("Start link edge for %s, which is the %zu input of %s.", getnext_node->GetName().c_str(),
         case_input_index, case_node_->GetName().c_str());
  for (size_t out_index = 0; out_index < data_count_from_getnext_; ++out_index, ++case_input_index) {
    if (GraphUtils::AddEdge(getnext_node->GetOutDataAnchor(out_index),
                            case_node_->GetInDataAnchor(case_input_index)) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Failed to add data edge between %zu Data:%s to %zu Case:%s", out_index,
             getnext_node->GetName().c_str(), case_input_index, case_node_->GetName().c_str());
      return FAILED;
    }
  }
  if (getnext_sink_dynamic_dims_) {
    GE_CHK_STATUS_RET(LinkGetNextToGetDynamicDims(getnext_node, shape_node_), "Failed to add link for %s.",
                      shape_node_->GetName().c_str());
  }
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
  size_t arg_index = kCaseArgIndex + all_data_nodes_.size();
  if (data_count_from_getnext_ != 0) {
    arg_index = arg_index + data_count_from_getnext_ - kNumOfGetnextNode;
  }

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
      GELOGE(FAILED, "Failed to add edge between Const:%s to Case:%s", data->GetName().c_str(),
             case_node_->GetName().c_str());
      return FAILED;
    }
    all_const_nodes.emplace_back(data);
  }
  ChangeConstToData();
  all_const_nodes_.swap(all_const_nodes);
  return SUCCESS;
}

void MultiBatchClonePass::ChangeConstToData() {
  size_t data_index = all_data_nodes_.size();
  if (data_count_from_getnext_ != 0) {
    data_index = data_index + data_count_from_getnext_ - kNumOfGetnextNode;
  }
  for (size_t i = 0; i < all_const_nodes_.size(); ++i, ++data_index) {  // Trans subgraph Const to Data.
    auto &const_node = all_const_nodes_[i];
    bool need_change_type = true;
    if (out_control_nodes_.find(const_node) != out_control_nodes_.end()) {
      GELOGD("No need to change %s to data type.", const_node->GetName().c_str());
      need_change_type = false;
      break;
    }
    if (!need_change_type) {
      continue;
    }
    const OpDescPtr &op_desc = all_const_nodes_[i]->GetOpDesc();
    op_desc->SetType(DATA);
    (void)op_desc->DelAttr(ATTR_NAME_WEIGHTS);  // Delete weight.

    // Const no InputDesc, Data need InputDesc.
    (void)op_desc->AddInputDesc(op_desc->GetOutputDesc(kDataOutIndex));
    (void)AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, data_index);
    (void)NodeUtils::AppendInputAnchor(all_const_nodes_[i], 1);
  }
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
  GE_CHK_STATUS_RET(LinkGetDynamicDimsToNetOutput(node), "Failed to add edge between %s to netoutput: %s.",
                    shape_node_->GetName().c_str(), output->GetName().c_str());
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
Status MultiBatchClonePass::SetMaxShape(const NodePtr &data) {
  GELOGD("Start set max shape for %s.", data->GetName().c_str());
  if (!IsGetNextType(data)) {
    if (SetMaxShapeToData(data, kDataOutIndex) != SUCCESS) {
      GELOGE(PARAM_INVALID, "Failed to update max shape of %s.", data->GetName().c_str());
      return PARAM_INVALID;
    }
  } else {
    for (size_t out_anchor_index = 0; out_anchor_index < data_count_from_getnext_; ++out_anchor_index) {
      if (SetMaxShapeToData(data, out_anchor_index) != SUCCESS) {
        GELOGE(PARAM_INVALID, "Failed to update max shape of %s.", data->GetName().c_str());
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

Status MultiBatchClonePass::SetMaxShapeToData(const NodePtr &node, size_t out_anchor_index) {
  GELOGD("Start update max shape of %s, %zu output.", node->GetName().c_str(), out_anchor_index);
  auto data_shape = NodeUtils::GetOutputDesc(*node, out_anchor_index).GetShape();
  string data_name = node->GetName();
  if (IsGetNextType(node)) {
    data_name.append("_").append(std::to_string(out_anchor_index));
  }
  GELOGD("Update max shape of %s, shape dims is %s.", data_name.c_str(),
         formats::JoinToString(data_shape.GetDims()).c_str());
  const auto &dims = data_shape.GetDims();
  if (!IsGetNextType(node)) {
    if (std::all_of(dims.begin(), dims.end(), [](int64_t val) { return val >= 0; })) {
      GELOGD("No need to do anything for static data.");
      return SUCCESS;
    }
  } else {
    if (std::all_of(dims.begin(), dims.end(), [](int64_t val) { return val >= 0; })) {
      if (getnext_sink_dynamic_dims_) {
        // need to update shape of Shape_node when getnext node has dynamic data
        GE_CHK_STATUS_RET(UpdateShapeOfShapeNode(node, out_anchor_index), "Failed to update shape of shape node");
      }
      return SUCCESS;
    }
  }
  (void)AttrUtils::SetListInt(node->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, data_shape.GetDims());

  GeTensorDesc tensor(NodeUtils::GetOutputDesc(*node, kDataOutIndex));
  std::vector<std::string> input_dims_str;
  for (size_t i = 0; i < batch_shapes_.size(); ++i) {
    auto shape = data_shape;
    auto ret = multibatch::CalcShape(data_to_dynamic_info_.at(data_name).at(i), shape);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to calculate the shape for data node %s, the shape may not match", node->GetName().c_str());
      return ret;
    }
    tensor.SetShape(shape);
    int64_t tensor_size = 0;
    (void)TensorUtils::GetTensorSizeInBytes(tensor, tensor_size);
    string input_str = TypeUtils::FormatToSerialString(tensor.GetFormat()) + ":" +
	               TypeUtils::DataTypeToSerialString(tensor.GetDataType()) + ":" + node->GetName() + ":" +
	               std::to_string(tensor_size) + ":" + std::to_string(tensor.GetShape().GetDimNum()) + ":" +
                       formats::JoinToString(tensor.GetShape().GetDims());
    input_dims_str.emplace_back(input_str);
  }
  (void)AttrUtils::SetListStr(node->GetOpDesc(), "_all_origin_gears_inputs", input_dims_str);

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
  return SetShapeToData(data_to_dynamic_info_.at(data_name).at(max_shape_index), node, data_shape, out_anchor_index);
}

///
/// @ingroup ge
/// @brief Set max shape to Data/GetNext node in root graph.
/// @param [in] const std::vector<int64_t> &shapes: dims of shape.
/// @param [in] const NodePtr &data: data in Root/Case graph.
/// @param [in] GeShape &data_shape: dims of data node.
/// @param [in] size_t out_anchor_index: out anchor index of data node.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::SetShapeToData(const std::vector<int64_t> &shapes, const NodePtr &data, GeShape &data_shape,
                                           size_t out_anchor_index) {
  GELOGD("Start set shape to %zu out of %s.", out_anchor_index, data->GetName().c_str());
  if (multibatch::CalcShape(shapes, data_shape) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to calculate the batched shape for data node %s, the shapes may not match",
           data->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (NodeUtils::UpdateOutputShape(*data, out_anchor_index, data_shape) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to update output shape for data %s", data->GetName().c_str());
    return INTERNAL_ERROR;
  }
  if (!IsGetNextType(data)) {
    if (NodeUtils::UpdateInputShape(*data, kDataInIndex, data_shape) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to update input shape for data %s", data->GetName().c_str());
      return INTERNAL_ERROR;
    }
  } else {
    if (getnext_sink_dynamic_dims_) {
      // need to update shape of Shape_node when getnext_sink_dynamic
      GE_CHK_STATUS_RET(UpdateShapeOfShapeNode(data, out_anchor_index), "Failed to update shape of shape node");
    }
  }

  GELOGI("Update the data %s input/output shape to the max %s", data->GetName().c_str(),
         formats::ShapeToString(data_shape).c_str());
  return SUCCESS;
}

Status MultiBatchClonePass::UpdateShapeOfShapeNode(const NodePtr &node, size_t out_anchor_index) {
  GELOGD("Start update output shape of shape node insert by adapter, which is the %zu out of %s.", out_anchor_index,
         node->GetName().c_str());
  auto data_shape = NodeUtils::GetOutputDesc(*node, out_anchor_index).GetShape();
  size_t shape_index = out_anchor_index + (node->GetAllOutDataAnchors().size() / kDivisionConst);
  GeTensorDesc output_desc = node->GetOpDesc()->GetOutputDesc(shape_index);
  std::vector<int64_t> output_dims = {static_cast<int64_t>(data_shape.GetDims().size())};
  GeShape output_shape(output_dims);
  output_desc.SetShape(output_shape);
  if (node->GetOpDesc()->UpdateOutputDesc(shape_index, output_desc) != SUCCESS) {
    GELOGE(FAILED, "Update output desc fail.");
    return FAILED;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Update Data node in Subgraph.
/// @param [in] const NodePtr &data: data in Subgraph.
/// @param [in] size_t batch_index: The batch index.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::UpdateSubgraphData(const NodePtr &data, size_t batch_index) {
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
  GELOGD("Start update shape of %s , batch index is %zu, dims is %s.", data->GetName().c_str(), batch_index,
         formats::JoinToString(dims).c_str());
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
  return SetShapeToData(data_to_dynamic_info_.at(parent_name).at(batch_index), data, data_shape, kDataOutIndex);
}

Status MultiBatchClonePass::CreateOriGraph(const ComputeGraphPtr &graph) {
  if (data_count_from_getnext_ == 0) {
    GELOGD("No need to change original graph without getnext node.");
    return SUCCESS;
  }
  GELOGD("Start change original graph: %s when exit getnext node.", graph->GetName().c_str());
  size_t data_index = all_data_nodes_.size() - kNumOfGetnextNode;
  for (const auto &node : graph->GetDirectNode()) {
    if (IsGetNextType(node)) {
      for (size_t out_index = 0; out_index < data_count_from_getnext_; ++out_index, ++data_index) {
        auto out_data_anchor =  node->GetOutDataAnchor(out_index);
        GE_IF_BOOL_EXEC(out_data_anchor == nullptr, continue);
        NodePtr data_node = CreateDataNode(graph, out_data_anchor, data_index);
        GE_IF_BOOL_EXEC(data_node == nullptr, GELOGE(INTERNAL_ERROR, "Create %zu data node failed.",
                                                     out_data_anchor->GetIdx()); return INTERNAL_ERROR);
        for (auto &in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
          GE_IF_BOOL_EXEC(in_anchor == nullptr, continue);
          NodePtr dst_node = in_anchor->GetOwnerNode();
          if (GraphUtils::RemoveEdge(out_data_anchor, in_anchor) != GRAPH_SUCCESS) {
            GELOGE(INTERNAL_ERROR, "Failed to remove edge between %s to %s", node->GetName().c_str(),
                   dst_node->GetName().c_str());
            return INTERNAL_ERROR;
          }
          if (GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(in_anchor->GetIdx())) !=
              GRAPH_SUCCESS) {
            GELOGE(INTERNAL_ERROR, "Failed to add edge between %s to %s", data_node->GetName().c_str(),
                   dst_node->GetName().c_str());
            return INTERNAL_ERROR;
          }
        }
      }
      if (graph->RemoveNode(node) != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "Remove node %s failed!", node->GetName().c_str());
        return GRAPH_FAILED;
      }
      break;
    }
  }
  return SUCCESS;
}

NodePtr MultiBatchClonePass::CreateDataNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor,
                                            size_t data_index) {
  size_t out_anchor_index = out_data_anchor->GetIdx();
  std::string node_name = out_data_anchor->GetOwnerNode()->GetName() + "_" +  std::to_string(out_anchor_index);
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, DATA);
  if (op_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Create data node failed.");
    return nullptr;
  }
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, data_index);

  OpDescPtr getnext_op_desc = out_data_anchor->GetOwnerNode()->GetOpDesc();
  if (getnext_op_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Op desc of %s is nullptr.", out_data_anchor->GetOwnerNode()->GetName().c_str());
    return nullptr;
  }
  if (op_desc->AddInputDesc(getnext_op_desc->GetOutputDesc(out_anchor_index)) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Add %s input desc failed.", op_desc->GetName().c_str());
    return nullptr;
  }
  if (op_desc->AddOutputDesc(getnext_op_desc->GetOutputDesc(out_anchor_index)) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Add %s output desc failed.", op_desc->GetName().c_str());
    return nullptr;
  }
  NodePtr data_node = graph->AddNode(op_desc);
  GELOGD("Success create %s node.", data_node->GetName().c_str());
  return data_node;
}

///
/// @ingroup ge
/// @brief Create nodes for root graph.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @param [in] const ComputeGraphPtr &branch: original graph.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchClonePass::CreateSubgraphs(const ComputeGraphPtr &graph, const ComputeGraphPtr &branch) {
  GELOGD("Start create subgraphs for %s.", graph->GetName().c_str());
  const auto &op_desc = case_node_->GetOpDesc();
  for (size_t i = 0; i < batch_shapes_.size(); ++i) {
    std::vector<NodePtr> input_nodes;
    std::vector<NodePtr> output_nodes;
    const std::string postfix = kMultiBatchNodePostfix + std::to_string(i);
    ComputeGraphPtr subgraph = (i == 0) ? branch : GraphUtils::CloneGraph(branch, postfix, input_nodes, output_nodes);
    GE_IF_BOOL_EXEC(subgraph == nullptr, GELOGE(FAILED, "Create multi-batch case node failed"); return FAILED);
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

    GELOGD("The %s has %zu input, %zu output.", subgraph->GetName().c_str(), input_nodes.size(), output_nodes.size());
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
  GELOGD("Start prune direct output.");
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
    GELOGD("No need to update output tensor.");
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
