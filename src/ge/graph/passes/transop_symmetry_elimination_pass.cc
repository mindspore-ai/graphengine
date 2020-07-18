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

#include "transop_symmetry_elimination_pass.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/common/transop_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"

namespace {
const int kTransOpOutIndex = 0;
static std::map<ge::DataType, ge::DataType> precision_loss_transfer_map = {{ge::DT_FLOAT, ge::DT_BOOL}};
}  // namespace
namespace ge {
Status TransOpSymmetryEliminationPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  if (!TransOpUtil::IsTransOp(node)) {
    return SUCCESS;
  }
  GELOGD("Symmetry Elimination Pass in.");
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor);
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNode());
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNode()->GetOpDesc());
      if (!CheckCanBeEliminated(node, peer_in_anchor)) {
        break;
      }

      auto dst_node = peer_in_anchor->GetOwnerNode();
      Status ret = EliminateTransOp(node, out_anchor, dst_node, peer_in_anchor);
      if (ret != SUCCESS) {
        // if eliminate failed ,it should't break precess, so give a warning here
        GELOGW("Eliminate %s and %s failed, ignore current pass.", node->GetName().c_str(),
               dst_node->GetName().c_str());
        return ret;
      }
    }
  }
  GELOGD("Symmetry Elimination Pass end.");
  return SUCCESS;
}

bool TransOpSymmetryEliminationPass::CheckCanBeEliminated(const ge::NodePtr &src_node,
                                                          const InDataAnchorPtr &dst_in_anchor) {
  auto dst_node = dst_in_anchor->GetOwnerNode();
  if (src_node->GetType() != dst_node->GetType()) {
    GELOGD("Pre node %s type %s is not equal with node %s type %s. Ignore pass.", src_node->GetName().c_str(),
           src_node->GetType().c_str(), dst_node->GetName().c_str(), dst_node->GetType().c_str());
    return false;
  }
  if (dst_in_anchor->GetIdx() != TransOpUtil::GetTransOpDataIndex(src_node)) {
    GELOGD("Next node %s type %s input %d is not for transform. Ignore pass.", dst_node->GetName().c_str(),
           dst_node->GetType().c_str(), dst_in_anchor->GetIdx());
    return false;
  }
  if (!DescAreSymmetry(src_node, dst_node) || !CheckPrecisionLoss(src_node)) {
    GELOGD("Not satisfied symmetry or has precision loss, ignore pass.");
    return false;
  }
  return true;
}
bool TransOpSymmetryEliminationPass::DescAreSymmetry(const NodePtr &src_node, const NodePtr &dst_node) {
  const auto &src_input_desc = src_node->GetOpDesc()->MutableInputDesc(0);
  const auto &dst_output_desc = dst_node->GetOpDesc()->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(src_input_desc);
  GE_CHECK_NOTNULL(dst_output_desc);
  const auto &src_input_dtype = src_input_desc->GetDataType();
  const auto &src_input_format = src_input_desc->GetFormat();
  const auto &src_input_shape = src_input_desc->GetShape().GetDims();
  const auto &dst_output_dtype = dst_output_desc->GetDataType();
  const auto &dst_output_format = dst_output_desc->GetFormat();
  const auto &dst_output_shape = dst_output_desc->GetShape().GetDims();

  if (src_node->GetType() == CAST && dst_node->GetType() == CAST) {
    bool is_format_symmetry =
      (src_input_format == dst_output_format) || (dst_output_format == FORMAT_ND) || (src_input_format == FORMAT_ND);
    return (src_input_dtype == dst_output_dtype) && is_format_symmetry;
  } else {
    return (src_input_dtype == dst_output_dtype) && (src_input_shape == dst_output_shape) &&
           (src_input_format == dst_output_format);
  }
}
bool TransOpSymmetryEliminationPass::CheckPrecisionLoss(const ge::NodePtr &src_node) {
  auto idx = TransOpUtil::GetTransOpDataIndex(src_node);
  auto input_desc = src_node->GetOpDesc()->GetInputDesc(idx);
  auto output_desc = src_node->GetOpDesc()->GetOutputDesc(kTransOpOutIndex);
  auto src_dtype = input_desc.GetDataType();
  auto dst_dtype = output_desc.GetDataType();
  auto iter = precision_loss_transfer_map.find(src_dtype);
  if (iter != precision_loss_transfer_map.end() && iter->second == dst_dtype) {
    GELOGW("Node %s transfer data type from %s to %s ,it will cause precision loss.", src_node->GetName().c_str(),
           TypeUtils::DataTypeToSerialString(src_dtype).c_str(), TypeUtils::DataTypeToSerialString(dst_dtype).c_str());
    return false;
  }
  return true;
}

Status TransOpSymmetryEliminationPass::EliminateTransOp(NodePtr &src_node, const OutDataAnchorPtr &src_out_anchor,
                                                        NodePtr &dst_node, const InDataAnchorPtr &dst_in_anchor) {
  // Two transform nodes can be offset like A->T1->T2->B
  // 1.Unlink T1->T2
  auto ret = src_out_anchor->Unlink(dst_in_anchor);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Unlink data anchor from %s to %s.", src_node->GetName().c_str(), dst_node->GetName().c_str());
    return ret;
  }
  // 2.Link A->T2
  auto data_idx = TransOpUtil::GetTransOpDataIndex(src_node);
  auto in_anchor = src_node->GetInDataAnchor(data_idx);
  GE_CHECK_NOTNULL(in_anchor);
  GE_CHECK_NOTNULL(in_anchor->GetPeerOutAnchor());
  auto pre_normal_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
  ret = GraphUtils::AddEdge(in_anchor->GetPeerOutAnchor(), dst_in_anchor);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Add data edge from %s to %s failed.", pre_normal_node->GetName().c_str(),
           dst_node->GetName().c_str());
    return ret;
  }
  // 3.Copy in-control/data-in-control from T1->T2
  ret = GraphUtils::CopyInCtrlEdges(src_node, dst_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Copy control edge from %s to %s failed.", src_node->GetName().c_str(), dst_node->GetName().c_str());
    return ret;
  }
  // 4.IsolateAndDelete T2, A will link to B automatically, and all control edge will also relink.
  ret = IsolateAndDeleteNode(dst_node, {0});
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Isolate removed node: %s, type: %s failed", dst_node->GetName().c_str(),
           dst_node->GetType().c_str());
    return ret;
  }
  GELOGI("Trans op symmetry eliminate successfully. Node %s has been removed.", dst_node->GetName().c_str());
  // 5.If T1 has no data out, isolate and deleted it.
  if (src_node->GetOutDataNodesSize() == 0) {
    // 5.1 Copy out control to pre normal node
    ret = GraphUtils::CopyOutCtrlEdges(src_node, pre_normal_node);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Copy control edge from %s to %s failed.", src_node->GetName().c_str(),
             dst_node->GetName().c_str());
      return ret;
    }
    // 5.2 Isolate and delete T1
    ret = IsolateAndDeleteNode(src_node, {});
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Isolate removed node: %s, type: %s failed", src_node->GetName().c_str(),
             src_node->GetType().c_str());
      return ret;
    }
    GELOGI("Trans op symmetry eliminate successfully. Node %s has been removed.", src_node->GetName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge
