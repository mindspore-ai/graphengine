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

#include "format_refiner.h"

#include <deque>
#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "graph/ref_relation.h"
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

using namespace ge;
using namespace std;
namespace ge {
namespace {
const size_t kDimSize4d = 4;
const std::unordered_set<string> kChangeDimNodes = {PERMUTE, EXPANDDIMS, SQUEEZE};
const string kIsGraphInferred = "_is_graph_inferred";
thread_local RefRelations reflection_builder;
}  // namespace

graphStatus ReflectionProcess(const std::unordered_set<RefCell, RefCellHash> &reflection,
                              std::deque<ge::NodePtr> &nodes,
                              ge::Format to_be_set_format) {
  for (const auto &cell : reflection) {
    auto node = cell.node;
    auto in_out_idx = cell.in_out_idx;
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (cell.in_out == ge::NODE_IN) {
      auto desc = node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(in_out_idx));
      desc.SetOriginFormat(to_be_set_format);
      desc.SetFormat(to_be_set_format);
      (void)node->GetOpDesc()->UpdateInputDesc(static_cast<uint32_t>(in_out_idx), desc);
    } else {
      auto desc = node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(in_out_idx));
      desc.SetOriginFormat(to_be_set_format);
      desc.SetFormat(to_be_set_format);
      (void)node->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(in_out_idx), desc);
    }
    nodes.push_back(cell.node);
  }

  return GRAPH_SUCCESS;
}

graphStatus BiasAddFormatFixProcess(ge::NodePtr &node_ptr) {
  // 5 meas dim num
  if (node_ptr->GetType() != "BiasAdd") {
    return GRAPH_SUCCESS;
  }
  std::unordered_map<string, Format> kTfFormatFix = {
    {"NHWC", FORMAT_NDHWC},
    {"NCHW", FORMAT_NCDHW}
  };
  for (size_t i = 0; i < node_ptr->GetOpDesc()->GetInputsSize(); i++) {
    auto in_desc = node_ptr->GetOpDesc()->MutableInputDesc(i);
    GE_CHECK_NOTNULL(in_desc);
    if (in_desc->MutableShape().GetDimNum() != 5) { // 5 means dim num
      continue;
    }
    auto format = in_desc->GetOriginFormat();
    auto key = TypeUtils::FormatToSerialString(format);
    auto fixed_format = (kTfFormatFix.count(key) == 0) ? format : kTfFormatFix[key];
    in_desc->SetOriginFormat(fixed_format);
    in_desc->SetFormat(fixed_format);
    GELOGD("fix the %zu'th input of node[%s]. Origin format is %s , after fixed it is %s",
           i, node_ptr->GetName().c_str(), TypeUtils::FormatToSerialString(format).c_str(),
           TypeUtils::FormatToSerialString(fixed_format).c_str());
  }
  for (size_t i = 0; i < node_ptr->GetOpDesc()->GetOutputsSize(); i++) {
    auto out_desc = node_ptr->GetOpDesc()->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(out_desc);
    if (out_desc->MutableShape().GetDimNum() != 5) { // 5 means dim num
      continue;
    }
    auto format = out_desc->GetOriginFormat();
    auto key = TypeUtils::FormatToSerialString(format);
    auto fixed_format = (kTfFormatFix.count(key) == 0) ? format : kTfFormatFix[key];
    out_desc->SetOriginFormat(fixed_format);
    out_desc->SetFormat(fixed_format);
    GELOGD("fix the %zu'th output of node[%s]. Origin format is %s , after fixed it is %s",
           i, node_ptr->GetName().c_str(), TypeUtils::FormatToSerialString(format).c_str(),
           TypeUtils::FormatToSerialString(fixed_format).c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::RefreshConstantOutProcess(const ComputeGraphPtr &graph, const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetType() == CONSTANTOP && !IsGraphInferred(graph)) {
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
  for (auto &node_ptr : graph->GetAllNodes()) {
    if (node_ptr == nullptr) {
      return GRAPH_FAILED;
    }
    auto op_desc = node_ptr->GetOpDesc();
    if (op_desc == nullptr) {
      return GRAPH_FAILED;
    }
    graphStatus status = RefreshConstantOutProcess(graph, op_desc);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "refresh constant out process failed!");
      return GRAPH_FAILED;
    }
    // consider special node save process
    // get all input desc format
    bool node_is_all_nd = false;
    auto input_size = static_cast<uint32_t>(op_desc->GetAllInputsSize());
    for (uint32_t i = 0; i < input_size; i++) {
      // Operator pre-set format but not origin format
      GE_IF_BOOL_EXEC(op_desc->MutableInputDesc(i) == nullptr, continue);
      auto input_format = op_desc->MutableInputDesc(i)->GetFormat();
      // Pre-save data node (only main graph data) and default infer fail
      if (node_ptr->GetType() == DATA) {
        data_nodes.push_back(node_ptr);
      }
      if (input_format != FORMAT_ND && input_format != FORMAT_RESERVED) {
        node_is_all_nd = true;
      }
    }
    // Get all output desc format
    auto output_size = static_cast<uint32_t>(op_desc->GetOutputsSize());
    for (uint32_t i = 0; i < output_size; i++) {
      GE_IF_BOOL_EXEC(op_desc->MutableOutputDesc(i) == nullptr, continue);
      auto output_format = op_desc->MutableOutputDesc(i)->GetFormat();
      if (output_format != FORMAT_ND && output_format != FORMAT_RESERVED) {
        node_is_all_nd = true;
      }
    }
    // check anchor point valid
    if (!node_is_all_nd) {
      continue;
    }
    // special process for biasAdd op
    // In tensorflow, biasAdd's format is alwayse NHWC even though set the arg
    // "data_format" to NDHWC or NCDHW.It will destroy our format-infer mechanism
    // so here do special process
    status = BiasAddFormatFixProcess(node_ptr);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "fix biasAdd process failed!");
      return GRAPH_FAILED;
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
    auto input_desc = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(in_data_anchor_idx));
    GE_IF_BOOL_EXEC(input_desc == nullptr, continue);
    auto to_be_set_format = input_desc->GetOriginFormat();
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
    // do peer_out_node name and index as key to lookup reflections
    ge::RefCell key(peer_out_data_node->GetName(), peer_out_data_node, ge::NODE_OUT, idx);
    std::unordered_set<RefCell, RefCellHash> reflection;
    auto status = reflection_builder.LookUpRefRelations(key, reflection);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "LookUpRefRelations failed!Node is [%s],the %d out edge",
             (peer_out_data_node->GetName()).c_str(), idx);
      return GRAPH_FAILED;
    }

    auto ge_tensor_desc = peer_out_data_node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(idx));
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

      if (reflection.empty()) {
        ge_tensor_desc.SetOriginFormat(to_be_set_format);
        ge_tensor_desc.SetFormat(to_be_set_format);
        (void)peer_out_data_node->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(idx), ge_tensor_desc);

        // Call operator infer format api (forward) to get out format
        GELOGD("call infer format func[Back]!Node is [%s] ", (peer_out_data_node->GetName()).c_str());
        status = peer_out_data_node->InferOriginFormat();
        if (status != GRAPH_SUCCESS) {
          GELOGE(GRAPH_FAILED, "Node[%s] infer format failed", (peer_out_data_node->GetName()).c_str());
          return GRAPH_FAILED;
        }
        nodes.push_back(peer_out_data_node);
      } else {
        auto status = ReflectionProcess(reflection, nodes, to_be_set_format);
        if (status != GRAPH_SUCCESS) {
          GELOGE(GRAPH_FAILED, "reflection process failed!");
          return GRAPH_FAILED;
        }
      }
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
    auto to_be_set_format =
      node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(out_data_anchor_idx))->GetOriginFormat();
    if (to_be_set_format == FORMAT_ND) {
      GELOGD("Node [%s] format is ND.[F]", (node->GetName()).c_str());
      continue;
    }
    for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(peer_in_data_anchor == nullptr, continue);

      auto peer_in_data_node = peer_in_data_anchor->GetOwnerNode();
      GE_IF_BOOL_EXEC(peer_in_data_node == nullptr, continue);
      GE_IF_BOOL_EXEC(peer_in_data_node->GetOpDesc() == nullptr, continue);

      // Check format whether have been set
      int idx = peer_in_data_anchor->GetIdx();
      // do peer_out_node name and index as key to lookup reflections
      ge::RefCell key(peer_in_data_node->GetName(), peer_in_data_node, ge::NODE_IN, idx);
      std::unordered_set<RefCell, RefCellHash> reflection;
      auto status = reflection_builder.LookUpRefRelations(key, reflection);
      if (status != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "LookUpRefRelations failed!Node is [%s],the %d input edge",
               (peer_in_data_node->GetName()).c_str(), idx);
        return GRAPH_FAILED;
      }
      auto ge_tensor_desc = peer_in_data_node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(idx));
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

        if (reflection.empty()) {
          ge_tensor_desc.SetOriginFormat(to_be_set_format);
          ge_tensor_desc.SetFormat(to_be_set_format);
          (void)peer_in_data_node->GetOpDesc()->UpdateInputDesc(static_cast<uint32_t>(idx), ge_tensor_desc);

          /// Because netoutput node added before infer format ,so netoutput is end condition
          /// must set netoutput format , because saved result depend on format
          if (peer_in_data_node_type == NETOUTPUT) {
            continue;
          }

          // Call operator infer format api (forward) to get out format
          GELOGD("call infer format func[Back]!Node is [%s] ", (peer_in_data_node->GetName()).c_str());
          status = peer_in_data_node->InferOriginFormat();
          if (status != GRAPH_SUCCESS) {
            GELOGE(GRAPH_FAILED, "Node[%s] infer format failed", (peer_in_data_node->GetName()).c_str());
            return GRAPH_FAILED;
          }
          nodes.push_back(peer_in_data_node);
        } else {
          auto status = ReflectionProcess(reflection, nodes, to_be_set_format);
          if (status != GRAPH_SUCCESS) {
            GELOGE(GRAPH_FAILED, "reflection process failed!");
            return GRAPH_FAILED;
          }
        }
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
      // single op support private format set, its origin format should not be override
      auto ori_format = input_desc->GetOriginFormat();
      if (input_desc != nullptr && (ori_format == FORMAT_ND || ori_format == FORMAT_RESERVED)) {
        input_desc->SetOriginFormat(input_desc->GetFormat());
      }
    }
    for (const auto &output_desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
      auto ori_format = output_desc->GetOriginFormat();
      if (output_desc != nullptr && (ori_format == FORMAT_ND || ori_format == FORMAT_RESERVED)) {
        output_desc->SetOriginFormat(output_desc->GetFormat());
      }
    }
  }
}

graphStatus FormatRefiner::DataNodeFormatProcess(const ComputeGraphPtr &graph, std::vector<ge::NodePtr> &data_nodes,
                                                 ge::Format data_format,
                                                 std::unordered_map<ge::NodePtr, bool> &node_status) {
  if (!(IsGraphInferred(graph) && (!TypeUtils::IsInternalFormat(data_format)) && (data_format != FORMAT_ND))) {
    GELOGI("no necessary to do DataNodeFormatProcess. is_graph_inferred:%d, data_format:%s", IsGraphInferred(graph),
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

    auto input_desc = op_desc->MutableInputDesc(0);
    auto output_desc = op_desc->MutableOutputDesc(0);
    GE_CHECK_NOTNULL(input_desc);
    GE_CHECK_NOTNULL(output_desc);

    auto curr_format = output_desc->GetOriginFormat();
    if (curr_format != FORMAT_ND) {
      // Data format has been infered , continue
      continue;
    }
    // keep data format be ND because lacking of defination when input shape num is smaller than 4
    if (input_desc->MutableShape().GetDimNum() < kDimSize4d) {
      continue;
    }
    // Set format for un-infered data node
    input_desc->SetOriginFormat(data_format);
    input_desc->SetFormat(data_format);
    output_desc->SetOriginFormat(data_format);
    output_desc->SetFormat(data_format);
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

  if (graph == nullptr) {
    GELOGE(GRAPH_FAILED, "input graph is null");
    return GRAPH_FAILED;
  }
  // build reflection relations of boundary
  (void)reflection_builder.Clear();
  auto status = reflection_builder.BuildRefRelations(*graph);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "build reflection relations failed for main and subgraph!");
    return GRAPH_FAILED;
  }
  // User set global net format
  status = GetAnchorPoints(graph, anchor_points, data_nodes, node_status);
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
  status = DataNodeFormatProcess(graph, data_nodes, data_format, node_status);

  (void)AttrUtils::SetBool(graph, kIsGraphInferred, true);

  return status;
}

bool FormatRefiner::IsGraphInferred(const ComputeGraphPtr &graph) {
  bool is_graph_inferred = false;
  return (AttrUtils::GetBool(graph, kIsGraphInferred, is_graph_inferred) && is_graph_inferred);
}
}  // namespace ge
