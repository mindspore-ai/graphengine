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
#include "graph/preprocess/multi_batch_copy_graph.h"

#include <queue>
#include <set>
#include <string>

#include "common/formats/utils/formats_trans_utils.h"
#include "common/ge/ge_util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/passes/multi_batch_clone_pass.h"
#include "graph/passes/prune_pass.h"
#include "graph/preprocess/multi_batch_options.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/pass_manager.h"
#include "graph/common/local_context.h"

using std::set;
using std::string;
using std::vector;
using std::map;

namespace ge {
namespace multibatch {
namespace {
const char *const kMbatchSwitchnName = "mbatch-switch-name";
const int kSwitchNDataIndex = 0;
const int kSwitchNPredIndex = 1;
const int kDataOutIndex = 0;
const int kDataInIndex = 0;
const int kMergeDataOutIndex = 0;
const int kStaticOutput = -1;


inline bool IsDataLikeType(const std::string &node_type) { return (node_type == DATA) || (node_type == AIPP); }

NodePtr InsertMergeNodeToGraph(const std::string &name, size_t input_num, const ComputeGraphPtr &graph) {
  OpDescPtr desc = MakeShared<OpDesc>();
  if (desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed to insert merge node, name %s", name.c_str());
    return nullptr;
  }
  desc->SetName(name);
  desc->SetType(MERGE);
  GeTensorDesc tensor_desc;
  for (size_t i = 0; i < input_num; ++i) {
    auto ret = desc->AddInputDesc("x" + std::to_string(i), tensor_desc);
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                    GELOGE(INTERNAL_ERROR, "Failed to create merge node %s, failed to add input %zu, error-code %u",
                           name.c_str(), i, ret);
                    return nullptr);
  }
  auto ret = desc->AddOutputDesc("y", tensor_desc);
  GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                  GELOGE(INTERNAL_ERROR, "Failed to create merge node %s, failed to add output 'y', error-code %u",
                         name.c_str(), ret);
                  return nullptr);
  tensor_desc.SetDataType(DT_INT32);
  ret = desc->AddOutputDesc("value_index", tensor_desc);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to create merge node %s, failed to add output 'value_index', error-code %u",
           name.c_str(), ret);
    return nullptr;
  }

  if (!AttrUtils::SetBool(desc, ATTR_INSERT_BY_MBATCH, true)) {
    GELOGE(INTERNAL_ERROR, "Failed to create merge node %s, failed to add attr", name.c_str());
    return nullptr;
  }
  return graph->AddNode(desc);
}

NodePtr InsertCopyNode(const NodePtr &node, size_t n) {
  const std::string &name = node->GetName() + "_ascend_mbatch_batch_" + std::to_string(n);
  auto src_op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(src_op_desc == nullptr, GELOGE(INTERNAL_ERROR, "Failed to copy node %s to %s, the OpDesc is null",
                                                 node->GetName().c_str(), name.c_str());
                  return nullptr);

  auto desc = AttrUtils::CopyOpDesc(src_op_desc);
  GE_IF_BOOL_EXEC(desc == nullptr, GELOGE(OUT_OF_MEMORY, "Failed to create op desc for copy node for node %s name %s",
                                          node->GetName().c_str(), name.c_str());
                  return nullptr);

  desc->SetName(name);
  desc->CopyAttrsFrom(*src_op_desc);
  for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
    auto input_desc = desc->MutableInputDesc(i);
    GE_IF_BOOL_EXEC(input_desc == nullptr,
                    GELOGW("Get null input desc by index %u from node %s when copy from %s", i,
                           desc->GetName().c_str(), node->GetName().c_str());
                    continue);

    input_desc->CopyAttrsFrom(src_op_desc->GetInputDesc(i));
  }
  for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
    auto output_desc = desc->MutableOutputDesc(i);
    GE_IF_BOOL_EXEC(output_desc == nullptr,
                    GELOGE(INTERNAL_ERROR, "Failed to get output desc by index %u from node %s when copy from %s", i,
                           desc->GetName().c_str(), node->GetName().c_str());
                    return nullptr);

    output_desc->CopyAttrsFrom(src_op_desc->GetOutputDesc(i));
  }
  const std::string &batch_label = "Batch_" + std::to_string(n);
  if (!AttrUtils::SetStr(desc, ATTR_NAME_BATCH_LABEL, batch_label)) {
    GELOGE(FAILED, "set attr ATTR_NAME_BATCH_LABEL failed, node:%s.", name.c_str());
    return nullptr;
  }

  (void)AttrUtils::SetListStr(desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {node->GetName()});

  auto graph = node->GetOwnerComputeGraph();
  return graph->AddNode(desc);
}

bool IsAllDimsPositive(const std::vector<int64_t> &dims) {
  for (auto dim : dims) {
    if (dim < 0) {
      return false;
    }
  }
  return true;
}

NodePtr InsertConst(const std::string &name, const ComputeGraphPtr &graph) {
  auto desc = MakeShared<OpDesc>();
  if (desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed to create const op %s, out of memory", name.c_str());
    return nullptr;
  }
  desc->SetName(name);
  desc->SetType(CONSTANT);
  GeTensor tensor;
  tensor.SetData(std::vector<uint8_t>({0}));
  if (!AttrUtils::SetTensor(desc, ATTR_NAME_WEIGHTS, tensor)) {
    GELOGE(OUT_OF_MEMORY, "Failed to init tensor value for const %s", name.c_str());
    return nullptr;
  }
  if (!AttrUtils::SetBool(desc, ATTR_INSERT_BY_MBATCH, true)) {
    GELOGE(OUT_OF_MEMORY, "Failed to set insert flag for const node %s", name.c_str());
    return nullptr;
  }
  if (desc->AddOutputDesc(GeTensorDesc()) != GRAPH_SUCCESS) {
    GELOGE(OUT_OF_MEMORY, "Failed to add output desc for const node %s", name.c_str());
    return nullptr;
  }
  return graph->AddNode(desc);
}

bool IsOnlyOutputToAipp(const NodePtr &node) {
  for (const auto &out_node : node->GetOutDataNodes()) {
    if (out_node->GetType() != AIPP) {
      return false;
    }
  }
  return true;
}

Status CheckDataShape(const std::vector<NodePtr> &nodes) {
  size_t unknown_shape_count = 0;
  for (const auto &node : nodes) {
    if (node->GetType() != DATA) {
      continue;
    }
    for (auto dim : NodeUtils::GetOutputDesc(*node, kDataOutIndex).GetShape().GetDims()) {
      if (dim < 0) {
        unknown_shape_count++;
        break;
      }
    }
  }
  if (unknown_shape_count == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10040");
    GELOGE(PARAM_INVALID,
           "Need unknow shape data when user set --dynamic_batch_size, --dynamic_image_size or --dynamic_dims");
    return PARAM_INVALID;
  }

  return SUCCESS;
}
}  // namespace

Status MultiBatchGraphCopyer::CopyGraph() {
  auto ret = Init();
  if (ret != SUCCESS) {
    return ret;
  }

  if (LabelStatus() != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to label status for all nodes.");
    return INTERNAL_ERROR;
  }

  ret = CheckAndParseDynamicData();
  if (ret != SUCCESS) {
    return ret;
  }

  ret = CreateNewNodes();
  if (ret != SUCCESS) {
    return ret;
  }

  ret = LinkEdges();
  if (ret != SUCCESS) {
    return ret;
  }

  ret = InsertIdentityAfterSwitchN();
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to insert identity nodes after switchn node.");
    return INTERNAL_ERROR;
  }

  GELOGI("Begin to remove useless nodes by prune pass after copy process");
  PrunePass prune_pass;
  ret = prune_pass.Run(graph_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to prune");
    return ret;
  }
  return CheckCopyResult(origin_data_nodes_);
}

Status MultiBatchGraphCopyer::Init() {
  auto ret = CheckArguments();
  if (ret != SUCCESS) {
    return ret;
  }

  for (auto &node : graph_->GetAllNodes()) {
    origin_all_nodes_.emplace_back(node);
    if (IsDataLikeType(node->GetType())) {
      origin_data_nodes_.emplace_back(node);
    }
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::LabelStatus() {
  map<string, vector<NodePtr>> frame_enters;
  InitStatus(frame_enters);

  bool changed = true;
  // If anyone of in node is kNodeInBatchBranch, it is also kNodeInBatchBranch
  while (changed) {
    changed = false;
    for (const auto &node : origin_all_nodes_) {
      for (auto &in_node : node->GetInAllNodes()) {
        bool is_in_batch = origin_nodes_status_.find(in_node.get()) != origin_nodes_status_.end() &&
                           origin_nodes_status_[in_node.get()] == kNodeInBatchBranch;
        if (is_in_batch) {
          if (origin_nodes_status_.find(node.get()) == origin_nodes_status_.end() ||
              origin_nodes_status_[node.get()] != kNodeInBatchBranch) {
            origin_nodes_status_[node.get()] = kNodeInBatchBranch;
            ResetEnterStatus(frame_enters, node);
            changed = true;
          }
          break;
        }
      }
    }
  }

  for (const auto &node : origin_all_nodes_) {
    if (!(node->GetOpDesc()->GetSubgraphInstanceNames().empty())) {
      origin_nodes_status_[node.get()] = kNodeNotSupportNode;
      continue;
    }
    if (node->GetType() == NETOUTPUT) {
      origin_nodes_status_[node.get()] = kNodeOutBatchBranch;
      continue;
    }
    if (IsDataLikeType(node->GetType())) {
      if (IsOnlyOutputToAipp(node)) {
        origin_nodes_status_[node.get()] = kNodeOutBatchBranch;
      } else {
        origin_nodes_status_[node.get()] = kNodeStartNode;
      }
      continue;
    }
    if (origin_nodes_status_.find(node.get()) == origin_nodes_status_.end()) {
      origin_nodes_status_[node.get()] = kNodeOutBatchBranch;
    }
  }
  return SUCCESS;
}

void MultiBatchGraphCopyer::InitStatus(map<string, vector<NodePtr>> &frame_enters) {
  for (const auto &node : origin_all_nodes_) {
    if (node->GetType() != ENTER && node->GetType() != REFENTER) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    string frame_name;
    if (AttrUtils::GetStr(op_desc, ENTER_ATTR_FRAME_NAME, frame_name)) {
      frame_enters[frame_name].emplace_back(node);
    }
  }

  for (const auto &data : origin_data_nodes_) {
    auto data_shape = NodeUtils::GetOutputDesc(*data, kDataOutIndex).GetShape();
    if (!IsAllDimsPositive(data_shape.GetDims())) {
      origin_nodes_status_[data.get()] = kNodeInBatchBranch;
    }
  }
}

void MultiBatchGraphCopyer::ResetEnterStatus(map<string, vector<NodePtr>> &frame_enters, const NodePtr &node) {
  if (node->GetType() != ENTER && node->GetType() != REFENTER) {
    return;
  }

  for (const auto &frame_enter : frame_enters) {
    auto &enters = frame_enter.second;
    if (std::find(enters.begin(), enters.end(), node) != enters.end()) {
      for (const auto &enter : enters) {
        origin_nodes_status_[enter.get()] = kNodeInBatchBranch;
      }
      break;
    }
  }
}

Status MultiBatchGraphCopyer::CheckAndParseDynamicData(){
  size_t unknown_shape_count = 0;
  auto data_name_and_shape = GetLocalOmgContext().user_input_dims;
  GELOGD("raw data_name_and_shape size: %zu", data_name_and_shape.size());
  for (const auto &node : origin_all_nodes_) {
    auto data_desc = NodeUtils::GetOutputDesc(*node, kDataOutIndex);
    auto data_shape = data_desc.GetShape();
    auto data_format = data_desc.GetFormat() == Format::FORMAT_NCHW ? "NCHW" :
                       data_desc.GetFormat() == Format::FORMAT_NHWC ? "NHWC" : "Others";

    auto data_name = node->GetName();
    auto branch_status = GetNodeStatus(node);
    if (branch_status != kNodeStartNode) {
      continue;
    }
    if (IsAllDimsPositive(data_shape.GetDims())) {
      continue;
    }
    ++unknown_shape_count;
    auto iter = find(data_name_order_.begin(), data_name_order_.end(), data_name);
    if (iter == data_name_order_.end()) {
      if (dynamic_type_ == DynamicType::kDynamicBatch) {
        auto ret = CheckDynamicBatchShape(data_shape.GetDims(), data_name);
        if (!ret) {
          return PARAM_INVALID;
        }
      } else if (dynamic_type_ == DynamicType::kDynamicImageSize) {
        auto ret = CheckDynamicImageSizeShape(data_shape.GetDims(), data_name, data_format);
        if (!ret) {
          return PARAM_INVALID;
        }
      } else if (dynamic_type_ == DynamicType::kDynamicDims) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10001",
                                                        {"parameter", "reason"},
                                                        {"--input_shape",
                                                         "all dynamic data must be set in --input_shape"});
        GELOGE(INTERNAL_ERROR, "data: %s shape:%s must be set int --input_shape",
               node->GetName().c_str(), data_shape.ToString().c_str());
        return INTERNAL_ERROR;
      }
      data_name_and_shape.emplace_back(data_name, data_shape.GetDims());
    }
  }
  auto ret = ParserDataToDynmaicInfo(shapes_, data_name_and_shape, data_to_dynamic_info_);
  if (ret != SUCCESS){
    return ret;
  }
  if (unknown_shape_count == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10040");
    GELOGE(PARAM_INVALID,
           "Need unknow shape data when user set --dynamic_batch_size, --dynamic_image_size or --dynamic_dims");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::CreateNewNodes() {
  shape_data_ = InsertShapeDataNode();
  if (shape_data_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "Failed to create the shape data node for muti-batch");
    return INTERNAL_ERROR;
  }

  for (const auto &node : origin_all_nodes_) {
    auto node_type = node->GetType();
    Status ret = INTERNAL_ERROR;
    auto branch_status = GetNodeStatus(node);
    GELOGD("Process node %s, status %d", node->GetName().c_str(), static_cast<int>(branch_status));
    switch (branch_status) {
      case kNodeStartNode:
        GELOGD("Name: %s, type: %s, status: kNodeStartNode.", node->GetName().c_str(), node->GetType().c_str());
        ret = InsertSwitchNForData(node);
        if (ret == SUCCESS) {
          ret = UpdateMaxShapeToData(node);
        }
        break;
      case kNodeInBatchBranch:
        GELOGD("Name: %s, type: %s, status: kNodeInBatchBranch.", node->GetName().c_str(), node->GetType().c_str());
        ret = CopyNodeInBatchBranch(node);
        break;
      case kNodeOutBatchBranch:
        GELOGD("Name: %s, type: %s, status: kNodeOutBatchBranch.", node->GetName().c_str(), node->GetType().c_str());
        ret = InsertMergeForEdgeNode(node);
        break;
      case kNodeNotSupportNode:
        GELOGD("Name: %s, type: %s, status: kNodeNotSupportNode.", node->GetName().c_str(), node->GetType().c_str());
        break;
      default:
        GELOGE(INTERNAL_ERROR, "Unexpected status %d on node %s", static_cast<int>(branch_status),
               node->GetName().c_str());
        break;
    }
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to deal with node %s in multi-batch process", node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

NodePtr MultiBatchGraphCopyer::InsertMergeNode(const NodePtr &node, int index) {
  if (index < 0) {
    // the merge node must has data inputs, if origin connection is a control
    // edge, we use data edge instead
    index = 0;
  }

  auto &merge_nodes = nodes_to_merge_nodes_[node.get()];
  if (merge_nodes.empty()) {
    auto count = node->GetAllOutDataAnchorsSize();
    if (count == 0) {
      count = 1;
    }
    merge_nodes.resize(count, nullptr);
  }

  if (merge_nodes.at(index) != nullptr) {
    return merge_nodes[index];
  }

  auto merge_node_name = node->GetName() + "_ascend_mbatch_merge_" + std::to_string(index);
  auto merge_node = InsertMergeNodeToGraph(merge_node_name, shapes_.size(), node->GetOwnerComputeGraph());
  GE_IF_BOOL_EXEC(merge_node == nullptr, GELOGE(INTERNAL_ERROR, "Failed to create merge node for node %s, out index %d",
                                                node->GetName().c_str(), index);
                  return nullptr);
  merge_nodes[index] = merge_node;
  GELOGI("Create merge node %s for node %s index %d", merge_node_name.c_str(), node->GetName().c_str(), index);
  return merge_node;
}
Status MultiBatchGraphCopyer::CopyInDataEdges(const NodePtr &origin_node, int batch_num, const NodePtr &copyed_node) {
  for (auto &in_anchor : origin_node->GetAllInDataAnchors()) {
    auto origin_src_anchor = in_anchor->GetPeerOutAnchor();
    if (origin_src_anchor == nullptr) {
      GELOGD("The node %s does not have input on index %d", origin_node->GetName().c_str(), in_anchor->GetIdx());
      continue;
    }
    auto origin_src_node = origin_src_anchor->GetOwnerNode();
    auto dst_anchor = copyed_node->GetInDataAnchor(in_anchor->GetIdx());
    GE_CHECK_NOTNULL(dst_anchor);
    auto switchn_iter = data_nodes_to_switchn_.find(origin_src_node.get());
    if (switchn_iter != data_nodes_to_switchn_.end()) {
      auto ret = GraphUtils::AddEdge(switchn_iter->second->GetOutDataAnchor(batch_num), dst_anchor);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to add data edge between %s(%d) to %s(%d), error-code %u",
               switchn_iter->second->GetName().c_str(), batch_num, copyed_node->GetName().c_str(), in_anchor->GetIdx(),
               ret);
        return INTERNAL_ERROR;
      }
      GELOGD("Add data edge from %s(%d) to %s(%d)", switchn_iter->second->GetName().c_str(), batch_num,
             copyed_node->GetName().c_str(), in_anchor->GetIdx());
      continue;
    }

    auto batch_branch_iter = nodes_to_batch_nodes_.find(origin_src_node.get());
    if (batch_branch_iter != nodes_to_batch_nodes_.end()) {
      auto src_batch_node = batch_branch_iter->second.at(batch_num);
      auto ret = GraphUtils::AddEdge(src_batch_node->GetOutDataAnchor(origin_src_anchor->GetIdx()), dst_anchor);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to add data edge between %s(%d) to %s(%d), error-code %u",
               src_batch_node->GetName().c_str(), batch_num, copyed_node->GetName().c_str(), in_anchor->GetIdx(), ret);
        return INTERNAL_ERROR;
      }
      GELOGD("Add data edge from %s(%d) to %s(%d)", src_batch_node->GetName().c_str(), batch_num,
             copyed_node->GetName().c_str(), in_anchor->GetIdx());
      continue;
    }

    auto ret = GraphUtils::AddEdge(origin_src_anchor, dst_anchor);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to add data edge between origin node %s(%d) to copyed %s(%d)",
             origin_src_node->GetName().c_str(), origin_src_anchor->GetIdx(), copyed_node->GetName().c_str(),
             dst_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    GELOGD("Add data edge between branch-out %s(%d) to branch-in %s(%d)", origin_src_node->GetName().c_str(),
           origin_src_anchor->GetIdx(), copyed_node->GetName().c_str(), dst_anchor->GetIdx());
  }
  return SUCCESS;
}
Status MultiBatchGraphCopyer::CopyInControlEdges(const NodePtr &node, int batch_num, const NodePtr &copyed_node) {
  for (auto &origin_src_node : node->GetInControlNodes()) {
    auto switchn_iter = data_nodes_to_switchn_.find(origin_src_node.get());
    if (switchn_iter != data_nodes_to_switchn_.end()) {
      // reconnect data node
      auto ret = GraphUtils::AddEdge(switchn_iter->second->GetOutControlAnchor(), copyed_node->GetInControlAnchor());
      if (ret != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to add control edge between %s to %s, error-code %u",
               switchn_iter->second->GetName().c_str(), copyed_node->GetName().c_str(), ret);
        return INTERNAL_ERROR;
      }
      GELOGD("Add control edge from %s to %s", switchn_iter->second->GetName().c_str(), copyed_node->GetName().c_str());
      continue;
    }

    auto batch_branch_iter = nodes_to_batch_nodes_.find(origin_src_node.get());
    if (batch_branch_iter != nodes_to_batch_nodes_.end()) {
      // reconnect node in batch branch
      auto src_batch_node = batch_branch_iter->second.at(batch_num);
      auto ret = GraphUtils::AddEdge(src_batch_node->GetOutControlAnchor(), copyed_node->GetInControlAnchor());
      if (ret != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to add data edge between %s to %s, error-code %u",
               src_batch_node->GetName().c_str(), copyed_node->GetName().c_str(), ret);
        return INTERNAL_ERROR;
      }
      GELOGD("Add control edge from %s to %s", src_batch_node->GetName().c_str(), copyed_node->GetName().c_str());
      continue;
    }

    auto ret = GraphUtils::AddEdge(origin_src_node->GetOutControlAnchor(), copyed_node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to add control edge from origin %s to copyed %s",
             origin_src_node->GetName().c_str(), copyed_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GELOGD("Add control edge between branch-out %s to branch-in %s", origin_src_node->GetName().c_str(),
           copyed_node->GetName().c_str());
  }
  return SUCCESS;
}
NodePtr MultiBatchGraphCopyer::InsertShapeDataNode() {
  auto desc = MakeShared<OpDesc>();
  if (desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed to create shape data node, out of memory");
    return nullptr;
  }
  string node_name = "ascend_mbatch_shape_data";
  // Only flush subgraph name
  if (graph_->GetParentGraph() != nullptr) {
    node_name = graph_->GetName() + "_" + node_name;
  }
  desc->SetName(node_name);
  desc->SetType(DATA);

  GeTensorDesc tensor_desc;
  tensor_desc.SetFormat(FORMAT_ND);
  tensor_desc.SetShape(GeShape({static_cast<int64_t>(shapes_.at(0).size())}));
  tensor_desc.SetDataType(DT_INT64);
  auto ret = desc->AddInputDesc(tensor_desc);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to add input desc for created data");
    return nullptr;
  }
  ret = desc->AddOutputDesc(tensor_desc);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to add output desc for created data");
    return nullptr;
  }

  if (!AttrUtils::SetBool(desc, ATTR_INSERT_BY_MBATCH, true)) {
    GELOGE(INTERNAL_ERROR, "Failed to add attr for created data");
    return nullptr;
  }

  auto data_node = graph_->AddNode(desc);
  if (data_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "Failed to add shape data node to graph");
    return nullptr;
  }
  ret = GraphUtils::AppendInputNode(graph_, data_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to append data node %s as input to graph", data_node->GetName().c_str());
    return nullptr;
  }

  return data_node;
}
Status MultiBatchGraphCopyer::CheckArguments() {
  if (graph_ == nullptr) {
    GELOGE(PARAM_INVALID, "Failed to copy graph, the graph is null");
    return PARAM_INVALID;
  }

  return CheckDynamicParams(shapes_);
}
Status MultiBatchGraphCopyer::CheckCopyResult(const std::vector<NodePtr> &start_nodes) {
  for (auto &node : start_nodes) {
    if (IsOnlyOutputToAipp(node)) {
      continue;
    }
    auto dims = NodeUtils::GetOutputDesc(*node, kDataOutIndex).GetShape().GetDims();
    if (!IsAllDimsPositive(dims)) {
      GELOGE(INTERNAL_ERROR, "Failed to copy multi batch graph, the node %s still has unknown shape %s",
             node->GetName().c_str(), formats::ShapeToString(dims).c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}
bool MultiBatchGraphCopyer::IsInBatchBranch(const NodePtr &node) {
  return (nodes_to_batch_nodes_.count(node.get()) > 0) || (data_nodes_to_switchn_.count(node.get()) > 0);
}
Status MultiBatchGraphCopyer::LinkDataToMerge(const NodePtr &data, const NodePtr &merge) {
  // The caller should make sure that the there is a SwitchN node in the map
  auto &switchn = data_nodes_to_switchn_[data.get()];
  GELOGI("Link edge between data %s to merge %s throw switchn %s", data->GetName().c_str(), merge->GetName().c_str(),
         switchn->GetName().c_str());
  for (size_t i = 0; i < shapes_.size(); ++i) {
    auto ret = GraphUtils::AddEdge(switchn->GetOutDataAnchor(i), merge->GetInDataAnchor(i));
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                    GELOGE(INTERNAL_ERROR, "Failed to add edge between switchn %s(%zu) to merge %s(%zu), error-code %u",
                           switchn->GetName().c_str(), i, merge->GetName().c_str(), i, ret);
                    return INTERNAL_ERROR);
  }
  return SUCCESS;
}
Status MultiBatchGraphCopyer::LinkNodeToMerge(const NodePtr &node, int out_index, const NodePtr &merge) {
  auto &copyed_nodes = nodes_to_batch_nodes_[node.get()];
  if (copyed_nodes.size() != shapes_.size()) {
    GELOGE(INTERNAL_ERROR,
           "Failed to create merge node for node %s, the copyed nodes for it count %zu different with shape %zu",
           node->GetName().c_str(), copyed_nodes.size(), shapes_.size());
    return INTERNAL_ERROR;
  }
  for (size_t i = 0; i < copyed_nodes.size(); ++i) {
    auto src_node = copyed_nodes[i];
    if (src_node->GetAllOutDataAnchorsSize() == 0) {
      // if the node does not has any data output, we should create an const for it, like this:
      //       c          d
      // node ---> const ---> merge
      auto const_name = src_node->GetName() + "_merge_const";
      GELOGI("The node %s on the batch branch edge does not have any data output, create a const %s for it",
             src_node->GetName().c_str(), const_name.c_str());
      auto const_node = InsertConst(const_name, graph_);
      GE_IF_BOOL_EXEC(const_node == nullptr,
                      GELOGE(OUT_OF_MEMORY, "Failed to create const for node %s to connect to a merge node",
                             src_node->GetName().c_str());
                      return OUT_OF_MEMORY);

      auto ret = GraphUtils::AddEdge(src_node->GetOutControlAnchor(), const_node->GetInControlAnchor());
      GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to add control edge from %s to %s",
                                                   src_node->GetName().c_str(), const_node->GetName().c_str());
                      return INTERNAL_ERROR);

      src_node = const_node;
    }
    auto ret = GraphUtils::AddEdge(src_node->GetOutDataAnchor(out_index), merge->GetInDataAnchor(i));
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR,
             "Failed to add edge between copyed node %s(%d) to inserted merge node %s(%zu), error-code %u",
             copyed_nodes[i]->GetName().c_str(), out_index, merge->GetName().c_str(), i, ret);
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}
Status MultiBatchGraphCopyer::UpdateMaxShapeToData(const NodePtr &data) {
  auto data_shape = NodeUtils::GetOutputDesc(*data, kDataOutIndex).GetShape();
  auto data_name = data->GetName();
  if (IsAllDimsPositive(data_shape.GetDims())) {
    return SUCCESS;
  }
  size_t max_shape_index = 0;
  int64_t max_size = 0;
  for (size_t i = 0; i < shapes_.size(); ++i) {
    int64_t size = 1;
    for (auto dim : data_to_dynamic_info_.at(data_name).at(i)) {
      if (INT64_MAX / dim < size) {
        GELOGE(PARAM_INVALID, "The shape %s size overflow",
               formats::ShapeToString(data_to_dynamic_info_[data_name].at(i)).c_str());
        return PARAM_INVALID;
      }
      size *= dim;
    }
    if (size > max_size) {
      max_size = size;
      max_shape_index = i;
    }
  }
  // must not be error, the calc result has been checked in function InsertSwitchNForData
  (void)CalcShape(data_to_dynamic_info_.at(data_name).at(max_shape_index), data_shape);
  auto ret = NodeUtils::UpdateOutputShape(*data, kDataOutIndex, data_shape);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to update output shape for data %s", data->GetName().c_str());
    return INTERNAL_ERROR;
  }
  ret = NodeUtils::UpdateInputShape(*data, kDataInIndex, data_shape);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to update input shape for data %s", data->GetName().c_str());
    return INTERNAL_ERROR;
  }
  GELOGI("Update the data %s input/output shape to the max %s", data->GetName().c_str(),
         formats::ShapeToString(data_shape).c_str());
  return SUCCESS;
}
Status MultiBatchGraphCopyer::InsertSwitchNForData(const NodePtr &data) {
  auto data_shape = NodeUtils::GetOutputDesc(*data, kDataOutIndex).GetShape();
  auto data_name = data->GetName();
  (void)AttrUtils::SetListInt(data->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, data_shape.GetDims());
  if (IsAllDimsPositive(data_shape.GetDims())) {
    GELOGI("The shape of data %s are positive(%s), skip the multi batch process", data->GetName().c_str(),
           data_shape.ToString().c_str());
    return SUCCESS;
  }

  auto switchn_desc = MakeShared<OpDesc>();
  if (switchn_desc == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed to create switchn for data %s", data->GetName().c_str());
    return OUT_OF_MEMORY;
  }
  switchn_desc->SetName(data->GetName() + "_ascend_mbatch_switchn");
  switchn_desc->SetType(SWITCHN);

  GeTensorDesc tensor(NodeUtils::GetOutputDesc(*data, kDataOutIndex));
  if (switchn_desc->AddInputDesc("data", tensor) != GRAPH_SUCCESS) {  // data
    return OUT_OF_MEMORY;
  }
  GeTensorDesc pred_tensor;
  if (switchn_desc->AddInputDesc("pred_value", pred_tensor) != GRAPH_SUCCESS) {  // pred
    return OUT_OF_MEMORY;
  }
  std::vector<std::string> input_dims_str;
  for (size_t i = 0; i < shapes_.size(); ++i) {
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
    if (!AttrUtils::SetListInt(tensor, ATTR_NAME_SWITCHN_PRED_VALUE, shapes_.at(i))) {
      GELOGE(INTERNAL_ERROR, "Failed to add attr value on output %zu tensor", i);
      return INTERNAL_ERROR;
    }
    (void) AttrUtils::SetListInt(tensor, ATTR_NAME_COMBINED_DYNAMIC_DIMS, shape.GetDims());
    if (switchn_desc->AddOutputDesc("output" + std::to_string(i), tensor) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Opdesc AddOutputDesc failed");
      return GRAPH_FAILED;
    }
    GELOGD("The SwitchN %s output index %zu, shape %s", switchn_desc->GetName().c_str(), i, shape.ToString().c_str());
  }
  (void)AttrUtils::SetListStr(data->GetOpDesc(), "_all_origin_gears_inputs", input_dims_str);
  if (!AttrUtils::SetListStr(switchn_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, data_name_order_)) {
    GELOGE(INTERNAL_ERROR, "Failed to add user designate shape order attr on switchn node %s",
           switchn_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }
  if (!AttrUtils::SetBool(switchn_desc, ATTR_INSERT_BY_MBATCH, true)) {
    GELOGE(INTERNAL_ERROR, "Failed to add insert attr on switchn node %s", switchn_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }
  if (!AttrUtils::SetStr(data->GetOpDesc(), kMbatchSwitchnName, switchn_desc->GetName())) {
    GELOGE(INTERNAL_ERROR, "Failed to add switchn attr on data node %s", data->GetName().c_str());
    return INTERNAL_ERROR;
  }
  if (StampDynamicType(switchn_desc) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to add dynamic type attr on switchn node %s", switchn_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  auto switchn = graph_->AddNode(switchn_desc);
  if (switchn == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Failed to create switchn %s from desc", switchn_desc->GetName().c_str());
    return OUT_OF_MEMORY;
  }
  data_nodes_to_switchn_[data.get()] = switchn;
  return SUCCESS;
}
Status MultiBatchGraphCopyer::InsertMergeForEdgeNode(const NodePtr &node) {
  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto src_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr) {
      GELOGD("The node %s does not has input at index %d", node->GetName().c_str(), in_data_anchor->GetIdx());
      continue;
    }
    auto in_node = src_out_anchor->GetOwnerNode();
    if (!IsInBatchBranch(in_node)) {
      continue;
    }
    auto merge_node = InsertMergeNode(in_node, src_out_anchor->GetIdx());
    if (merge_node == nullptr) {
      return INTERNAL_ERROR;
    }
  }

  for (auto &in_node : node->GetInControlNodes()) {
    if (!IsInBatchBranch(in_node)) {
      continue;
    }
    auto merge_node = InsertMergeNode(in_node, -1);
    if (merge_node == nullptr) {
      return INTERNAL_ERROR;
    }
  }

  return SUCCESS;
}
Status MultiBatchGraphCopyer::CopyNodeInBatchBranch(const NodePtr &node) {
  auto &copyed_nodes = nodes_to_batch_nodes_[node.get()];
  for (size_t i = 0; i < shapes_.size(); ++i) {
    auto copyed_node = InsertCopyNode(node, i);
    if (copyed_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "Failed to add node to graph when copy node %s", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    copyed_nodes.emplace_back(copyed_node);
    GELOGI("Copy node %s type %s for shape %s, new node name %s", node->GetName().c_str(), node->GetType().c_str(),
           formats::JoinToString(shapes_.at(i)).c_str(), copyed_node->GetName().c_str());
  }
  return SUCCESS;
}
Status MultiBatchGraphCopyer::LinkEdges() {
  Status ret;
  for (const auto &node : origin_all_nodes_) {
    if (data_nodes_to_switchn_.count(node.get()) > 0) {
      ret = LinkDataToSwitchN(node);
      if (ret != SUCCESS) {
        return ret;
      }
    }
    if (nodes_to_merge_nodes_.count(node.get()) > 0) {
      ret = LinkToMerge(node);
      if (ret != SUCCESS) {
        return ret;
      }
    }
    if (nodes_to_batch_nodes_.count(node.get()) > 0) {
      ret = LinkToNodeInBranch(node);
    } else {
      ret = LinkToNodeOutBranch(node);
    }
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}
Status MultiBatchGraphCopyer::LinkDataToSwitchN(const NodePtr &data) {
  auto switchn = data_nodes_to_switchn_[data.get()];
  auto ret =
      GraphUtils::AddEdge(shape_data_->GetOutDataAnchor(kDataOutIndex), switchn->GetInDataAnchor(kSwitchNPredIndex));
  GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to link shape data %s to switchn %s",
                                               shape_data_->GetName().c_str(), switchn->GetName().c_str());
                  return INTERNAL_ERROR);

  ret = GraphUtils::AddEdge(data->GetOutDataAnchor(kDataOutIndex), switchn->GetInDataAnchor(kSwitchNDataIndex));
  GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to link data %s to switchn %s",
                                               data->GetName().c_str(), switchn->GetName().c_str());
                  return INTERNAL_ERROR);
  return SUCCESS;
}
Status MultiBatchGraphCopyer::LinkToMerge(const NodePtr &node) {
  auto &merge_nodes = nodes_to_merge_nodes_[node.get()];
  for (size_t i = 0; i < merge_nodes.size(); ++i) {
    auto merge_node = merge_nodes[i];
    if (merge_node == nullptr) {
      continue;
    }
    if (nodes_to_batch_nodes_.count(node.get()) > 0) {
      auto ret = LinkNodeToMerge(node, i, merge_node);
      if (ret != SUCCESS) {
        return ret;
      }
      continue;
    }
    if (data_nodes_to_switchn_.count(node.get()) > 0) {
      auto ret = LinkDataToMerge(node, merge_node);
      if (ret != SUCCESS) {
        return ret;
      }
      continue;
    }
    GELOGE(INTERNAL_ERROR, "The merge node %s is created, index %zu, but can not find the src node",
           merge_node->GetName().c_str(), i);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}
Status MultiBatchGraphCopyer::LinkToNodeInBranch(const NodePtr &node) {
  auto &branch_nodes = nodes_to_batch_nodes_[node.get()];
  for (size_t i = 0; i < branch_nodes.size(); ++i) {
    auto ret = CopyInDataEdges(node, i, branch_nodes[i]);
    if (ret != SUCCESS) {
      return ret;
    }
    ret = CopyInControlEdges(node, i, branch_nodes[i]);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}
Status MultiBatchGraphCopyer::LinkToNodeOutBranch(const NodePtr &node) {
  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto src_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr) {
      GELOGD("The node %s does not has input at index %d", node->GetName().c_str(), in_data_anchor->GetIdx());
      continue;
    }
    auto in_node = src_out_anchor->GetOwnerNode();
    if (!IsInBatchBranch(in_node)) {
      continue;
    }
    auto iter = nodes_to_merge_nodes_.find(in_node.get());
    if (iter == nodes_to_merge_nodes_.end()) {
      GELOGE(INTERNAL_ERROR, "Failed to link IO data edge from %s(%d) to %s(%d), no merge node found",
             in_node->GetName().c_str(), src_out_anchor->GetIdx(), node->GetName().c_str(), in_data_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    auto merge_node = iter->second[src_out_anchor->GetIdx()];
    if (merge_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "Failed to link IO data edge from %s(%d) to %s(%d), no merge node found",
             in_node->GetName().c_str(), src_out_anchor->GetIdx(), node->GetName().c_str(), in_data_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    auto ret = src_out_anchor->Unlink(in_data_anchor);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to unlink the control edge from %s(%d) to %s(%d)", in_node->GetName().c_str(),
             src_out_anchor->GetIdx(), node->GetName().c_str(), in_data_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    ret = GraphUtils::AddEdge(merge_node->GetOutDataAnchor(kMergeDataOutIndex), in_data_anchor);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Failed to add data edge from %s(%d) to %s(%d)", merge_node->GetName().c_str(),
             src_out_anchor->GetIdx(), node->GetName().c_str(), in_data_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    GELOGI("Link data edge from merge %s(from %s(%d)) to %s(%d)", merge_node->GetName().c_str(),
           in_node->GetName().c_str(), src_out_anchor->GetIdx(), node->GetName().c_str(), in_data_anchor->GetIdx());
  }

  for (auto &in_node : node->GetInControlNodes()) {
    if (!IsInBatchBranch(in_node)) {
      continue;
    }
    auto iter = nodes_to_merge_nodes_.find(in_node.get());
    if (iter == nodes_to_merge_nodes_.end()) {
      GELOGE(INTERNAL_ERROR, "Failed to link IO control edge from %s to %s, no merge node found",
             in_node->GetName().c_str(), node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    auto merge_node = iter->second[0];
    if (merge_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "Failed to link IO control edge from %s to %s, no merge node found",
             in_node->GetName().c_str(), node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    GE_IF_BOOL_EXEC(in_node->GetOutControlAnchor() == nullptr,
                    GELOGE(INTERNAL_ERROR, "Innode outputControlAnchor is null");
                    return INTERNAL_ERROR);
    auto ret = in_node->GetOutControlAnchor()->Unlink(node->GetInControlAnchor());
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to unlink the control edge from %s to %s",
                                                 in_node->GetName().c_str(), node->GetName().c_str());
                    return INTERNAL_ERROR);
    ret = GraphUtils::AddEdge(merge_node->GetOutControlAnchor(), node->GetInControlAnchor());
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(INTERNAL_ERROR, "Failed to add control edge from %s to %s",
                                                 merge_node->GetName().c_str(), node->GetName().c_str());
                    return INTERNAL_ERROR);
    GELOGI("Link control edge from merge %s(from %s) to %s", merge_node->GetName().c_str(), in_node->GetName().c_str(),
           node->GetName().c_str());
  }

  return SUCCESS;
}

Status MultiBatchGraphCopyer::InsertIdentityAfterSwitchN() {
  for (auto &node : graph_->GetAllNodes()) {
    if (node->GetType() != SWITCHN) {
      continue;
    }
    auto switchn_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(switchn_desc);
    size_t i = 0;
    for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        auto out_node = in_data_anchor->GetOwnerNode();
        auto op_desc = out_node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        if ((out_node->GetType() == MERGE) && (op_desc->HasAttr(ATTR_INSERT_BY_MBATCH))) {
          GELOGD("No need to insert identity between %s and %s.", node->GetName().c_str(), out_node->GetName().c_str());
          continue;
        }

        auto identity_desc = MakeShared<OpDesc>(node->GetName() + "_identity_" + std::to_string(i), IDENTITY);
        GE_CHECK_NOTNULL(identity_desc);

        string batch_label;
        if (AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label)) {
          if (!AttrUtils::SetStr(identity_desc, ATTR_NAME_BATCH_LABEL, batch_label)) {
            GELOGE(FAILED, "Set attr ATTR_NAME_BATCH_LABEL failed, node:%s.", identity_desc->GetName().c_str());
            return FAILED;
          }
        }

        auto data_desc = switchn_desc->GetOutputDesc(i);
        i++;
        GE_CHK_STATUS_RET(identity_desc->AddInputDesc("x", data_desc));
        GE_CHK_STATUS_RET(identity_desc->AddOutputDesc("y", data_desc));

        auto identity_node = graph_->AddNode(identity_desc);
        GE_CHECK_NOTNULL(identity_node);
        GE_CHK_STATUS_RET(out_data_anchor->LinkTo(identity_node->GetInDataAnchor(0)));
        GE_CHECK_NOTNULL(identity_node->GetOutControlAnchor());
        GE_CHK_STATUS_RET(identity_node->GetOutControlAnchor()->LinkTo(out_node->GetInControlAnchor()));
      }
    }
  }

  return SUCCESS;
}

Status ProcessMultiBatch(ComputeGraphPtr &graph) {
  std::vector<std::vector<int64_t>> shapes;
  if (!InitDynamicParams(shapes)) {
    GELOGD("There is no multi-batch options, no need to process multi-batch copy");
    return SUCCESS;
  }
  DynamicType dynamic_type = DynamicType::kDynamicUnknown;
  if (!GetLocalOmgContext().dynamic_batch_size.empty()) {
    dynamic_type = DynamicType::kDynamicBatch;
  } else if (!GetLocalOmgContext().dynamic_image_size.empty()) {
    dynamic_type = DynamicType::kDynamicImageSize;;
  } else if (!GetLocalOmgContext().dynamic_dims.empty()) {
    dynamic_type = DynamicType::kDynamicDims;
  }
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_designate_shape;
  user_designate_shape = GetLocalOmgContext().user_input_dims;

  GELOGI("Begin to copy graph for multi-batch");
  multibatch::MultiBatchGraphCopyer copyer(graph);
  for (auto &shape : shapes) {
    copyer.AddShape(shape);
  }
  copyer.SetDynamicType(dynamic_type);
  copyer.SetUserDesignateShape(user_designate_shape);
  return copyer.CopyGraph();
}

//              +-----------+
//              |   Data    |                      +-----------+       +-----------+       +-----------+
//              +-----------+                      |    Data   | ----> | SoftmaxV2 | ----> | NetOutput |
//                       \                      /. +-----------+       +-----------+       +-----------+
//                        \                    /.
// +-----------+       +-----------+          /.   +-----------+       +-----------+       +-----------+
// |   Data    | ----> |    Case   |         S---  |    Data   | ----> | SoftmaxV2 | ----> | NetOutput |
// +-----------+       +-----------+          \.   +-----------+       +-----------+       +-----------+
//                               \             \.
//                                \             \. +-----------+       +-----------+       +-----------+
//                           +-----------+         |    Data   | ----> | SoftmaxV2 | ----> | NetOutput |
//                           | NetOutput |         +-----------+       +-----------+       +-----------+
//                           +-----------+
// +-----------+                  /
// |   Data    | --------------->/
// +-----------+
void GetDynamicShapeByGraph(const ComputeGraphPtr &graph, const NodePtr &node,
                            set<size_t> &dynamic_output_index, vector<string> &dynamic_output_dims) {
  GELOGD("Try get dynamic shape info, Graph: %s, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
  const auto &func_desc = node->GetOpDesc();
  if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
    GELOGD("Graph: %s Not multi-batch, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
    return;
  }

  const auto &dynamic_branch_names = func_desc->GetSubgraphInstanceNames();
  for (size_t i = 0; i < func_desc->GetOutputsSize(); ++i) {
    for (size_t j = 0; j < dynamic_branch_names.size(); ++j) {
      const auto &subgraph = graph->GetSubgraph(dynamic_branch_names[j]);
      if (subgraph == nullptr) {
        GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "Subgraph not found, name: %s", dynamic_branch_names[j].c_str());
        dynamic_output_dims.clear();
        return;
      }

      const auto &out_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
      if (out_node == nullptr) {
        GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "NetOutput not found, name: %s", dynamic_branch_names[j].c_str());
        dynamic_output_dims.clear();
        return;
      }

      GELOGI("Find the subgraph Output node %s and the index is %zu", out_node->GetName().c_str(), i);
      const auto &out_desc = out_node->GetOpDesc();
      if (out_desc == nullptr || out_desc->GetInputsSize() <= i) {
        GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "Get Input desc failed, name: %s, index: %zu", out_node->GetName().c_str(), i);
        dynamic_output_dims.clear();
        return;
      }

      const auto &input_tensor = out_desc->GetInputDesc(i);
      const auto &shape_msg = input_tensor.GetShape().ToString();
      string output_shape = std::to_string(j) + "," + std::to_string(i) + "," + shape_msg;
      GELOGI("The shape msg in dynamic batch is %s", output_shape.c_str());
      dynamic_output_dims.emplace_back(output_shape);

      uint32_t parent_index = 0;
      (void)AttrUtils::GetInt(input_tensor, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
      dynamic_output_index.insert(parent_index);
    }
  }
}

//                                         +-----------+       +-----------+ i = 0
//                                  +----> | SoftmaxV2 | ----> |MemcpyAsync| ----> \.
//                                 /       +-----------+       +-----------+        \.
//                                /                                                  \.
// +-----------+       +-----------+       +-----------+       +-----------+ i = 1 +-----------+
// |   Data    | ----> |  SwitchN  | ----> | SoftmaxV2 | ----> |MemcpyAsync| ----> |   Merge   |
// +-----------+       +-----------+       +-----------+       +-----------+       +-----------+
//                                \                                                  /       \.  j = 0
//                                 \       +-----------+       +-----------+ i = 2  /         \.
//                                  +----> | SoftmaxV2 | ----> |MemcpyAsync| ----> /       +-----------+
//                                         +-----------+       +-----------+               | NetOutput |
//                                                                                         +-----------+
// +-----------+                                                                              /.
// |   Data    | --------------------------------------------------------------------------->/.  j = 1
// +-----------+
void GetDynamicShapeByMerge(const ComputeGraphPtr &graph, const NodePtr &node,
                            set<size_t> &dynamic_output_index, vector<string> &dynamic_output_dims) {
  GELOGD("Try get dynamic shape info, Graph: %s, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
  const auto &netoutput_desc = node->GetOpDesc();
  const auto &inputnode_to_netoutput = node->GetInAllNodes();
  for (size_t i = 0; i < inputnode_to_netoutput.size(); ++i) {
    bool insert_by_mbatch = false;
    (void)AttrUtils::GetBool(inputnode_to_netoutput.at(i)->GetOpDesc(), ATTR_INSERT_BY_MBATCH, insert_by_mbatch);
    if (inputnode_to_netoutput.at(i)->GetType() == MERGE && insert_by_mbatch) {
      GELOGI("Find the merge node %s with mbatch attr and the index is %zu",
             inputnode_to_netoutput.at(i)->GetName().c_str(), i);
      dynamic_output_index.insert(i);
      for (size_t j = 0; j < inputnode_to_netoutput.at(i)->GetInNodes().size(); ++j) {
        auto input_desc = inputnode_to_netoutput.at(i)->GetOpDesc();
        auto input_tensor_desc = input_desc->GetInputDesc(j);
        auto shape_msg = input_tensor_desc.GetShape().ToString();
        string output_shape = std::to_string(j) + "," + std::to_string(i) + "," + shape_msg;
        GELOGI("The shape msg in dynamic batch is %s", output_shape.c_str());
        dynamic_output_dims.emplace_back(output_shape);
      }
    }
  }
}

// Connect NetOutput directly: DTS2020070612498
void GetDirectOutputShape(const ComputeGraphPtr &graph, const NodePtr &node,
                          const set<size_t> &dynamic_output_index, vector<string> &dynamic_output_dims) {
  GELOGD("Try get directly shape info, Graph: %s, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
  const auto &netoutput_desc = node->GetOpDesc();
  const auto &inputnode_to_netoutput = node->GetInAllNodes();
  for (size_t i = 0; i < inputnode_to_netoutput.size(); ++i) {
    if (dynamic_output_index.count(i) > 0) {
      continue;
    }

    auto tensor_desc = netoutput_desc->GetInputDesc(i);
    auto shape = tensor_desc.GetShape().ToString();
    string static_output_shape = std::to_string(kStaticOutput) + "," + std::to_string(i) + "," + shape;
    GELOGI("The static output shape msg is %s", static_output_shape.c_str());
    dynamic_output_dims.emplace_back(static_output_shape);
  }
}

Status GetDynamicOutputShape(ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  GELOGI("Start to get output dynamic batch shape message");

  NodePtr net_output;
  set<size_t> dynamic_output_index;
  vector<string> dynamic_output_dims;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == NETOUTPUT) {
      net_output = node;
      GetDynamicShapeByMerge(graph, node, dynamic_output_index, dynamic_output_dims);
    } else if (node->GetType() == CASE) {
      GetDynamicShapeByGraph(graph, node, dynamic_output_index, dynamic_output_dims);
    }
  }

  if ((net_output != nullptr) && !dynamic_output_dims.empty()) {
    GetDirectOutputShape(graph, net_output, dynamic_output_index, dynamic_output_dims);
    if (!AttrUtils::SetListStr(net_output->GetOpDesc(), ATTR_NAME_DYNAMIC_OUTPUT_DIMS, dynamic_output_dims)) {
      GELOGE(FAILED, "Set dynamic output dims attr failed");
      return FAILED;
    }
  }

  return SUCCESS;
}
}  // namespace multibatch
}  // namespace ge
