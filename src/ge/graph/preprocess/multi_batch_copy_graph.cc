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

#include "graph/preprocess/multi_batch_copy_graph.h"

#include <string>
#include <queue>
#include <set>

#include "common/formats/utils/formats_trans_utils.h"
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/passes/prune_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {
namespace multibatch {
namespace {
const char *const kMbatchSwitchnName = "mbatch-switch-name";
const int kSwitchNDataIndex = 0;
const int kSwitchNPredIndex = 1;
const int kDataOutIndex = 0;
const int kDataInIndex = 0;
const int kMergeDataOutIndex = 0;
const size_t kMaxShapesCount = 100;
const size_t kMinShapesCount = 2;

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

NodePtr InsertCopyNode(const NodePtr &node, const std::string &name) {
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
                    GELOGE(INTERNAL_ERROR, "Failed to get input desc by index %u from node %s when copy from %s", i,
                           desc->GetName().c_str(), node->GetName().c_str());
                    return nullptr);

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
  auto graph = node->GetOwnerComputeGraph();
  return graph->AddNode(desc);
}

Status CalcShape(const std::vector<int64_t> &batch_shape, GeShape &data_shape) {
  size_t batch_shape_index = 0;
  for (size_t i = 0; i < data_shape.GetDimNum(); ++i) {
    if (data_shape.GetDim(i) < 0) {
      if (batch_shape_index >= batch_shape.size()) {
        GELOGE(PARAM_INVALID,
               "Failed to calc tensor shape, the batch shape count %zu, doees not match the data shape %s",
               batch_shape.size(), data_shape.ToString().c_str());
        return PARAM_INVALID;
      }
      data_shape.SetDim(i, batch_shape[batch_shape_index++]);
    }
  }
  if (batch_shape_index != batch_shape.size()) {
    GELOGE(PARAM_INVALID, "Failed to calc tensor shape, the batch shape count %zu, does not match the data shape %s",
           batch_shape.size(), data_shape.ToString().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

bool IsAllDimsPositive(const std::vector<int64_t> &dims) {
  for (auto dim : dims) {
    if (dim <= 0) {
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
    GELOGE(PARAM_INVALID, "There are no unknown shape data, the dynamic batch/imagesize options will be ignored");
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

  ret = CheckDataShape(origin_data_nodes_);
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
        ret = InsertSwitchNForData(node);
        if (ret == SUCCESS) {
          ret = UpdateMaxShapeToData(node);
        }
        break;
      case kNodeInBatchBranch:
        ret = CopyNodeInBatchBranch(node);
        break;
      case kNodeOutBatchBranch:
        ret = InsertMergeForEdgeNode(node);
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
NodeStatus MultiBatchGraphCopyer::GetNodeStatus(const NodePtr &node) {
  if (node->GetType() == NETOUTPUT) {
    return kNodeOutBatchBranch;
  }
  if (IsDataLikeType(node->GetType()) && !IsOnlyOutputToAipp(node)) {
    return kNodeStartNode;
  }
  for (auto &in_node : node->GetInDataNodes()) {
    if (IsInBatchBranch(in_node)) {
      return kNodeInBatchBranch;
    }
  }
  return kNodeOutBatchBranch;
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
  desc->SetName("ascend_mbatch_shape_data");
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
  if (shapes_.size() < kMinShapesCount) {
    GELOGE(PARAM_INVALID, "The minimum batch-shapes count is %zu", kMinShapesCount);
    return PARAM_INVALID;
  }
  if (shapes_.size() > kMaxShapesCount) {
    GELOGE(PARAM_INVALID, "The max batch-shapes count is %zu", kMaxShapesCount);
    return PARAM_INVALID;
  }
  std::set<std::vector<int64_t>> shapes_set;
  size_t shape_size = shapes_.at(0).size();
  for (auto &shape : shapes_) {
    if (shape_size != shape.size()) {
      GELOGE(PARAM_INVALID, "All batch shapes size must be the same, first group's size is %zu and another's is %zu.",
             shape_size, shape.size());
      return PARAM_INVALID;
    }
    for (auto dim : shape) {
      if (dim <= 0) {
        GELOGE(PARAM_INVALID, "Invalid dim %ld, all dims must more than 0", dim);
        return PARAM_INVALID;
      }
    }
    shapes_set.insert(shape);
  }
  if (shapes_set.size() != shapes_.size()) {
    GELOGE(PARAM_INVALID, "There are duplicated batch-shapes, please check");
    return PARAM_INVALID;
  }
  return SUCCESS;
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
  GELOGI("Link edge bwetween data %s to merge %s throw switchn %s", data->GetName().c_str(), merge->GetName().c_str(),
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
  if (IsAllDimsPositive(data_shape.GetDims())) {
    return SUCCESS;
  }

  size_t max_shape_index = 0;
  int64_t max_size = 0;
  for (size_t i = 0; i < shapes_.size(); ++i) {
    int64_t size = 1;
    for (auto dim : shapes_[i]) {
      if (INT64_MAX / dim < size) {
        GELOGE(PARAM_INVALID, "The shape %s size overflow", formats::ShapeToString(shapes_[i]).c_str());
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
  (void)CalcShape(shapes_[max_shape_index], data_shape);

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
  if (switchn_desc->AddInputDesc(tensor) != GRAPH_SUCCESS) {  // data
    return OUT_OF_MEMORY;
  }
  GeTensorDesc pred_tensor;
  if (switchn_desc->AddInputDesc(pred_tensor) != GRAPH_SUCCESS) {  // pred
    return OUT_OF_MEMORY;
  }
  for (size_t i = 0; i < shapes_.size(); ++i) {
    auto shape = data_shape;
    auto ret = CalcShape(shapes_.at(i), shape);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to calculate the batched shape for data node %s, the shapes may not match",
             data->GetName().c_str());
      return ret;
    }
    tensor.SetShape(shape);
    if (!AttrUtils::SetListInt(tensor, ATTR_NAME_SWITCHN_PRED_VALUE, shapes_.at(i))) {
      GELOGE(INTERNAL_ERROR, "Failed to add attr value on output %zu tensor", i);
      return INTERNAL_ERROR;
    }
    if (switchn_desc->AddOutputDesc(tensor) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Opdesc AddOutputDesc failed");
      return GRAPH_FAILED;
    }
    GELOGD("The SwitchN %s output index %zu, shape %s", switchn_desc->GetName().c_str(), i, shape.ToString().c_str());
  }

  if (!AttrUtils::SetBool(switchn_desc, ATTR_INSERT_BY_MBATCH, true)) {
    GELOGE(INTERNAL_ERROR, "Failed to add insert attr on switchn node %s", switchn_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }
  if (!AttrUtils::SetStr(data->GetOpDesc(), kMbatchSwitchnName, switchn_desc->GetName())) {
    GELOGE(INTERNAL_ERROR, "Failed to add switchn attr on data node %s", data->GetName().c_str());
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
    auto copyed_node = InsertCopyNode(node, node->GetName() + "_ascend_mbatch_batch_" + std::to_string(i));
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

Status ProcessMultiBatch(ComputeGraphPtr &graph) {
  const int kDecimal = 10;
  std::vector<std::vector<int64_t>> shapes;
  if (!domi::GetContext().dynamic_batch_size.empty()) {
    GELOGD("Found dynamic batch option, value %s", domi::GetContext().dynamic_batch_size.c_str());
    std::vector<std::string> dims = ge::StringUtils::Split(domi::GetContext().dynamic_batch_size, ',');
    for (const auto &dim : dims) {
      if (dim.empty()) {
        continue;
      }
      shapes.emplace_back(std::vector<int64_t>({std::strtol(dim.c_str(), nullptr, kDecimal)}));
      GELOGI("Found dynamic batch, shape %s", formats::JoinToString(*shapes.rbegin()).c_str());
    }
  }
  if (!domi::GetContext().dynamic_image_size.empty()) {
    GELOGD("Found dynamic image size option, value %s", domi::GetContext().dynamic_image_size.c_str());
    std::vector<std::string> shape_strs = ge::StringUtils::Split(domi::GetContext().dynamic_image_size, ';');
    for (const auto &shape_str : shape_strs) {
      if (shape_str.empty()) {
        continue;
      }
      std::vector<int64_t> shape;
      std::vector<std::string> dims = ge::StringUtils::Split(shape_str, ',');
      for (const auto &dim : dims) {
        if (dim.empty()) {
          continue;
        }
        shape.emplace_back(std::strtol(dim.c_str(), nullptr, kDecimal));
      }
      shapes.emplace_back(shape);
      GELOGI("Found dynamic image size, shape %s", formats::JoinToString(shape).c_str());
    }
  }
  if (shapes.empty()) {
    GELOGD("There is no multi-batch options, no need to process multi-batch copy");
    return SUCCESS;
  }

  GELOGI("Begin to copy graph for multi-batch");
  multibatch::MultiBatchGraphCopyer copyer(graph);
  for (auto &shape : shapes) {
    copyer.AddShape(shape);
  }
  return copyer.CopyGraph();
}
}  // namespace multibatch
}  // namespace ge
