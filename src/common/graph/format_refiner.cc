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

#include "graph/format_refiner.h"

#include <deque>
#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "./compute_graph.h"
#include "./ge_error_codes.h"
#include "./graph/ge_tensor.h"
#include "./operator.h"
#include "./operator_factory.h"
#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "utils/node_utils.h"
#include "utils/op_desc_utils.h"
#include "utils/tensor_utils.h"
#include "utils/type_utils.h"

namespace ge {
namespace {
static const std::unordered_set<string> kChangeDimNodes = {RESHAPE, PERMUTE, EXPANDDIMS, SQUEEZE};
static bool net_format_is_nd = true;
static Format g_user_set_format = FORMAT_ND;
static bool is_first_infer = true;
}  // namespace

graphStatus FormatRefiner::RefreshConstantOutProcess(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetType() == CONSTANTOP && is_first_infer == true) {
    ConstGeTensorPtr tensor_value;
    if (!AttrUtils::GetTensor(op_desc, "value", tensor_value)) {
      GELOGE(GRAPH_FAILED, "Get value failed, node name:%s.", op_desc->GetName().c_str());
      return GRAPH_FAILED;
    }
    GE_CHECK_NOTNULL(tensor_value);
    (void)op_desc->UpdateOutputDesc(0, tensor_value->GetTensorDesc());
  }
  return GRAPH_SUCCESS;
}
graphStatus FormatRefiner::GetAnchorPoints(const ge::ComputeGraphPtr &graph, std::vector<ge::NodePtr> &anchor_points,
                                           std::vector<ge::NodePtr> &data_nodes,
                                           std::unordered_map<ge::NodePtr, bool> &node_status) {
  if (graph == nullptr) {
    GELOGE(GRAPH_FAILED, "input graph is null");
    return GRAPH_FAILED;
  }
  anchor_points.clear();
  // Get all anchor point nodes and switch nodes
  for (const auto &node_ptr : graph->GetAllNodes()) {
    std::vector<bool> is_node_set_format;
    if (node_ptr == nullptr) {
      return GRAPH_FAILED;
    }
    auto op_desc = node_ptr->GetOpDesc();
    if (op_desc == nullptr) {
      return GRAPH_FAILED;
    }
    graphStatus status = RefreshConstantOutProcess(op_desc);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "refresh constant out process failed!");
      return GRAPH_FAILED;
    }
    // consider special node save process
    // get all input desc format
    bool node_is_all_nd = false;
    for (uint32_t i = 0; i < static_cast<uint32_t>(op_desc->GetInputsSize()); i++) {
      auto input_desc = op_desc->GetInputDesc(i);
      // Operator pre-set format but not origin format
      auto input_format = input_desc.GetFormat();
      // Pre-save data node and default infer fail
      if (node_ptr->GetType() == DATA) {
        data_nodes.push_back(node_ptr);
      }
      if (input_format != FORMAT_ND && input_format != FORMAT_RESERVED) {
        node_is_all_nd = true;
      }
    }
    // Get all output desc format
    for (uint32_t i = 0; i < static_cast<uint32_t>(op_desc->GetOutputsSize()); i++) {
      GeTensorDesc output_desc = op_desc->GetOutputDesc(i);
      auto output_format = output_desc.GetFormat();
      if (output_format != FORMAT_ND && output_format != FORMAT_RESERVED) {
        node_is_all_nd = true;
      }
    }
    // check anchor point valid
    if (!node_is_all_nd) {
      continue;
    }
    GELOGD("Node[%s] is anchor point!", node_ptr->GetName().c_str());
    anchor_points.push_back(node_ptr);
  }
  GELOGI("anchor_points number is %zu", anchor_points.size());
  return GRAPH_SUCCESS;
}
graphStatus FormatRefiner::AnchorProcess(const ge::NodePtr &anchor_node,
                                         std::unordered_map<ge::NodePtr, bool> &node_status) {
  if (anchor_node == nullptr) {
    GELOGE(GRAPH_FAILED, "anchor node is null!");
    return GRAPH_FAILED;
  }
  std::deque<ge::NodePtr> nodes;
  nodes.push_back(anchor_node);
  while (!nodes.empty()) {
    ge::NodePtr node = nodes.front();
    nodes.pop_front();
    graphStatus status = BackInferProcess(nodes, node, node_status);
    if (status != GRAPH_SUCCESS && node != nullptr) {
      GELOGE(status, "BackInferProcess failed!node name [%s]", node->GetName().c_str());
      return status;
    }
    status = ForwardInferProcess(nodes, node, node_status);
    if (status != GRAPH_SUCCESS && node != nullptr) {
      GELOGE(status, "ForwardInferProcess failed!node name [%s]", node->GetName().c_str());
      return status;
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus FormatRefiner::BackInferProcess(std::deque<ge::NodePtr> &nodes, ge::NodePtr &node,
                                            std::unordered_map<ge::NodePtr, bool> &node_status) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());

  GELOGD("Enter back infer process!Node is [%s]", (node->GetName()).c_str());
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    GELOGD("Node is [%s] [B]", (node->GetName()).c_str());
    auto in_data_anchor_idx = in_anchor->GetIdx();
    auto to_be_set_format = (node->GetOpDesc()->GetInputDesc(in_data_anchor_idx)).GetOriginFormat();
    if (to_be_set_format == FORMAT_ND) {
      GELOGD("Node [%s] [B], format is ND", (node->GetName()).c_str());
      continue;
    }
    auto peer_out_data_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_data_anchor == nullptr) {
      GELOGW("Node[%s] %dth in data anchor's peer_out_anchor is null", (node->GetName()).c_str(), in_data_anchor_idx);
      continue;
    }
    auto peer_out_data_node = peer_out_data_anchor->GetOwnerNode();
    if (peer_out_data_node == nullptr || peer_out_data_node->GetOpDesc() == nullptr) {
      GELOGW("Node[%s]\'s peer_out_data_node or peer_out_data_node desc is null", (node->GetName()).c_str());
      continue;
    }
    // Check format whether have been set
    int idx = peer_out_data_anchor->GetIdx();
    auto ge_tensor_desc = peer_out_data_node->GetOpDesc()->GetOutputDesc(idx);
    if (ge_tensor_desc.GetOriginFormat() == FORMAT_ND) {
      auto dim_num = ge_tensor_desc.GetShape().GetDimNum();
      if (dim_num == 0) {
        GELOGD("node name:%s idx:%d out is scalar. stop back infer!", peer_out_data_node->GetName().c_str(), idx);
        continue;
      }
      /// Check whether node to change dims ()
      /// Because some node will calculate with 5D, C dim maybe multi meaning
      auto peer_out_data_node_type = peer_out_data_node->GetType();
      auto iter1 = kChangeDimNodes.find(peer_out_data_node_type);
      // 4 means dims num
      if ((iter1 != kChangeDimNodes.end()) && (dim_num < 4)) {
        GELOGD("Node[%s] is change dim node and shape is smaller than 4. do not modify format",
               (peer_out_data_node->GetName()).c_str());
        continue;
      }

      ge_tensor_desc.SetOriginFormat(to_be_set_format);
      ge_tensor_desc.SetFormat(to_be_set_format);
      (void)peer_out_data_node->GetOpDesc()->UpdateOutputDesc(idx, ge_tensor_desc);

      // Call operator infer format api (forward) to get out format
      GELOGD("call infer format func[Back]!Node is [%s] ", (peer_out_data_node->GetName()).c_str());
      graphStatus status = peer_out_data_node->InferOriginFormat();
      if (status != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "Node[%s] infer format failed", (peer_out_data_node->GetName()).c_str());
        return GRAPH_FAILED;
      }
      nodes.push_back(peer_out_data_node);
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus FormatRefiner::ForwardInferProcess(std::deque<ge::NodePtr> &nodes, ge::NodePtr &node,
                                               std::unordered_map<ge::NodePtr, bool> &node_status) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  GELOGD("Enter forward infer process!Node is [%s]", (node->GetName()).c_str());
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GELOGD("Node is [%s] [F]", (node->GetName()).c_str());
    GE_IF_BOOL_EXEC(out_data_anchor == nullptr, continue);
    auto out_data_anchor_idx = out_data_anchor->GetIdx();
    auto to_be_set_format = (node->GetOpDesc()->GetOutputDesc(out_data_anchor_idx)).GetOriginFormat();
    if (to_be_set_format == FORMAT_ND) {
      GELOGD("Node [%s] format is ND.[F]", (node->GetName()).c_str());
      continue;
    }
    for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      if (peer_in_data_anchor == nullptr) {
        GELOGW("Node[%s] some peer_in_anchor is null", (node->GetName()).c_str());
        continue;
      }
      auto peer_in_data_node = peer_in_data_anchor->GetOwnerNode();
      if (peer_in_data_node == nullptr || peer_in_data_node->GetOpDesc() == nullptr) {
        GELOGW("Node[%s] peer_in_data_node or peer_in_data_node desc is null", node->GetName().c_str());
        continue;
      }
      // Check format whether have been set
      int idx = peer_in_data_anchor->GetIdx();
      auto ge_tensor_desc = peer_in_data_node->GetOpDesc()->GetInputDesc(idx);
      if (ge_tensor_desc.GetOriginFormat() == FORMAT_ND) {
        auto dim_num = ge_tensor_desc.GetShape().GetDimNum();
        if (dim_num == 0) {
          GELOGI("node name:%s idx:%d in is scalar. stop forward infer!", peer_in_data_node->GetName().c_str(), idx);
          continue;
        }
        /// Check whether node to change dims ()
        /// Because some node will calculate with 5D, C dim maybe multi meaning
        auto peer_in_data_node_type = peer_in_data_node->GetType();
        auto iter1 = kChangeDimNodes.find(peer_in_data_node_type);
        // 4 means dims num
        if ((iter1 != kChangeDimNodes.end()) && (dim_num < 4)) {
          GELOGD("Node[%s] is change dim node. do not infer origin format", (peer_in_data_node->GetName()).c_str());
          continue;
        }
        ge_tensor_desc.SetOriginFormat(to_be_set_format);
        ge_tensor_desc.SetFormat(to_be_set_format);
        (void)peer_in_data_node->GetOpDesc()->UpdateInputDesc(idx, ge_tensor_desc);

        /// Because netoutput node added before infer format ,so netoutput is end condition
        /// must set netoutput format , because saved result depend on format
        if (peer_in_data_node_type == NETOUTPUT) {
          continue;
        }

        // Call operator infer format api (forward) to get out format
        GELOGD("call infer format func[Forward]!Node is [%s] ", (peer_in_data_node->GetName()).c_str());
        graphStatus status = peer_in_data_node->InferOriginFormat();
        if (status != GRAPH_SUCCESS) {
          GELOGE(GRAPH_FAILED, "Node[%s] infer format failed", (peer_in_data_node->GetName()).c_str());
          return GRAPH_FAILED;
        }
        nodes.push_back(peer_in_data_node);
      }
    }
  }
  return GRAPH_SUCCESS;
}

void FormatRefiner::RefreshOriginFormatOfAnchor(std::vector<ge::NodePtr> &anchor_points) {
  for (const auto &node : anchor_points) {
    if (node == nullptr || node->GetOpDesc() == nullptr) {
      continue;
    }
    for (const auto &input_desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
      if (input_desc != nullptr) {
        input_desc->SetOriginFormat(input_desc->GetFormat());
      }
    }
    for (const auto &output_desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
      if (output_desc != nullptr) {
        output_desc->SetOriginFormat(output_desc->GetFormat());
      }
    }
  }
}

void FormatRefiner::SetInferOrigineFormatFlag(bool is_first) { is_first_infer = is_first; }

graphStatus FormatRefiner::DataNodeFormatProcess(std::vector<ge::NodePtr> &data_nodes, ge::Format data_format,
                                                 std::unordered_map<ge::NodePtr, bool> &node_status) {
  bool is_internal_format = TypeUtils::IsInternalFormat(data_format);
  bool need_process = ((!is_first_infer) && (is_internal_format == false) && (data_format != FORMAT_ND));
  if (!need_process) {
    GELOGI("no necessary to do DataNodeFormatProcess.IsFirstInfer: %d, data_format:%s", is_first_infer,
           TypeUtils::FormatToSerialString(data_format).c_str());
    return GRAPH_SUCCESS;
  }
  GELOGD("Enter DataNodeFormatProcess");
  std::vector<ge::NodePtr> uninfered_data_nodes;
  // Check and renew data nodes format
  for (const auto &data_node : data_nodes) {
    GE_CHECK_NOTNULL(data_node);
    auto op_desc = data_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHECK_NOTNULL(op_desc->GetOutputDescPtr(0));
    auto curr_format = op_desc->GetOutputDescPtr(0)->GetOriginFormat();
    if (curr_format != FORMAT_ND) {
      // Data format has been infered , continue
      continue;
    }
    // Set format for un-infered data node
    auto input_descs = op_desc->GetAllInputsDescPtr();
    auto output_descs = op_desc->GetAllOutputsDescPtr();

    for (const auto &input_desc : input_descs) {
      if (input_desc != nullptr) {
        input_desc->SetOriginFormat(data_format);
        input_desc->SetFormat(data_format);
      }
    }
    for (const auto &output_desc : output_descs) {
      if (output_desc != nullptr) {
        output_desc->SetOriginFormat(data_format);
        output_desc->SetFormat(data_format);
      }
    }
    uninfered_data_nodes.push_back(data_node);
  }
  // Reinfer format from uninfered data nodes
  for (const auto &node : uninfered_data_nodes) {
    if (node == nullptr) {
      continue;
    }
    GELOGD("data node [%s] start infer format process", node->GetName().c_str());
    auto status = AnchorProcess(node, node_status);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "data node [%s] infer format process failed!", node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  GELOGD("DataNodeFormatProcess success");
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::InferOrigineFormat(const ge::ComputeGraphPtr &graph) {
  GELOGI("Enter InferOrigineFormat process!");

  // True: infered false:no-infered
  std::unordered_map<ge::NodePtr, bool> node_status;
  std::vector<ge::NodePtr> anchor_points;
  std::vector<ge::NodePtr> data_nodes;
  // global net format
  net_format_is_nd = true;
  g_user_set_format = FORMAT_ND;

  if (graph == nullptr) {
    GELOGE(GRAPH_FAILED, "input graph is null");
    return GRAPH_FAILED;
  }
  // User set global net format
  graphStatus status = GetAnchorPoints(graph, anchor_points, data_nodes, node_status);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "GetAnchorPoints Process Faild!");
    return GRAPH_FAILED;
  }
  // Refresh origin format of anchor point
  RefreshOriginFormatOfAnchor(anchor_points);
  // Infer format process
  for (const auto &anchor_node : anchor_points) {
    if (anchor_node == nullptr) {
      continue;
    }
    status = AnchorProcess(anchor_node, node_status);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Anchor node [%s] process failed!", anchor_node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  /// According to discuss with sys-enginer, data node default format is ND.Its format
  /// should be set by infered.But if some data-node can not be got by infer, set context's
  /// format for these data nodes.
  /// Notice: ignore 5D formats
  auto data_format = graph->GetDataFormat();
  status = DataNodeFormatProcess(data_nodes, data_format, node_status);

  // Set infer flag to false
  SetInferOrigineFormatFlag(false);
  return status;
}
}  // namespace ge
