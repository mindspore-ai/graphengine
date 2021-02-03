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

#include "graph/passes/hccl_continuous_memcpy_pass.h"

#include <string>

#include "common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge/ge_util.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"

namespace {
const int kAnchorNum = 0;
const int32_t kAnchorAssignRefIndex = 0;
const int32_t kAnchorAssignValueIndex = 1;
}  // namespace
namespace ge {
Status HcclContinuousMemcpyPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGE(INTERNAL_ERROR, "node has no op_desc, node_name : %s.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    Status ret = ContinuousInputProcess(graph, node);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "failed ProcessBroadcastMemcpy, node_name:%s.", node->GetName().c_str());
      return ret;
    }

    ret = P2pmemInputProcess(graph, node);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "failed P2pmemInputProcess, node_name:%s.", node->GetName().c_str());
      return ret;
    }

  }
  return SUCCESS;
}

// If broadcast input size is bigger than 1, and input from variable,
// cause by broadcast input memory should be continuous,
// another featuremap mem will be allocated for broadcast input.
// In this condition, move data from variable mem to broadcast input featuremap mem will be executed each step.
// In order to avoid move action out of model, use memcpy node instead of move action code.
Status HcclContinuousMemcpyPass::ContinuousInputProcess(const ComputeGraphPtr &graph, const NodePtr node) {
  auto op_desc = node->GetOpDesc();

  bool is_input_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);

  if (is_input_continuous && op_desc->GetInputsSize() > 1) {
    GELOGI("continuous input op is:%s.", op_desc->GetName().c_str());
    // if input size bigger than one, insert memcpy between var data for support continous mem alloc
    for (auto &hccl_in_anchor : node->GetAllInDataAnchors()) {
      if (hccl_in_anchor == nullptr) {
        continue;
      }
      auto src_out_anchor = hccl_in_anchor->GetPeerOutAnchor();
      if (src_out_anchor == nullptr) {
        GELOGE(INTERNAL_ERROR, "hcom op input has no peer anchor, node_name:%s", node->GetName().c_str());
        return INTERNAL_ERROR;
      }

      if (IsDataNode(src_out_anchor->GetOwnerNode()->GetType())) {
        Status ret = ModifyEdgeConnection(graph, src_out_anchor, hccl_in_anchor);
        if (ret != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "Failed to modify the connection.");
          return ret;
        }
      }
    }
  }
  return SUCCESS;
}

// if input is var type, and node input need p2p mem, then memcpy should be insert between the two
Status HcclContinuousMemcpyPass::P2pmemInputProcess(const ComputeGraphPtr &graph, const NodePtr node) {
  auto op_desc = node->GetOpDesc();

  vector<int64_t> input_memory_types;
  (void) ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, input_memory_types);

  if (input_memory_types.empty()) {
    return SUCCESS;
  }

  for (uint32_t index = 0; index < input_memory_types.size() && index < op_desc->GetInputsSize(); index++) {
    if (input_memory_types[index] != RT_MEMORY_P2P_DDR) {
      continue;
    }

    GELOGD("p2p input op is:%s.", op_desc->GetName().c_str());
    auto hccl_in_anchor = node->GetInDataAnchor(index);
    if (hccl_in_anchor == nullptr) {
      continue;
    }
    auto src_out_anchor = hccl_in_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr) {
      GELOGE(INTERNAL_ERROR, "hcom op input has no peer anchor, node_name:%s", node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    if (IsDataNode(src_out_anchor->GetOwnerNode()->GetType())) {
      Status ret = ModifyEdgeConnection(graph, src_out_anchor, hccl_in_anchor);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to modify the connection.");
        return ret;
      }
    }
  }
  return SUCCESS;
}

bool HcclContinuousMemcpyPass::IsDataNode(const std::string& node_type) {
  return (node_type == CONSTANTOP) || (node_type == VARIABLE) || (node_type == DATA) || (node_type == CONSTANT);
}

///
/// @brief Add Identity Node
/// @param [in] ge::ComputeGraphPtr graph
/// @param [in] ge::OutDataAnchorPtr in_node
/// @return ge::NodePtr
///
NodePtr HcclContinuousMemcpyPass::CreateIdentityNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor) {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);
  NodePtr pre_node = out_data_anchor->GetOwnerNode();
  OpDescPtr pre_op_desc = pre_node->GetOpDesc();
  if (pre_op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "OpDesc of pre node is invalid.");
    return nullptr;
  }

  std::string node_name = pre_node->GetName() + "_" + IDENTITY;
  node_name = CheckDuplicateName(node_name);
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name.c_str(), IDENTITY);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Create Identity op: MakeShared op_desc fail.");
    return nullptr;
  }
  GELOGI("Create Identity op:%s.", op_desc->GetName().c_str());

  graphStatus ret = op_desc->AddInputDesc("x", pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Create Identity op: add input desc fail.");
    return nullptr;
  }

  ret = op_desc->AddOutputDesc("y", pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Create Identity op: add output desc fail.");
    return nullptr;
  }
  // because history reason ,this pass can not do work after constant fold so mark it
  (void)AttrUtils::SetBool(op_desc, ATTR_NO_NEED_CONSTANT_FOLDING, false);

  NodePtr memcpy_node = graph->AddNode(op_desc);
  if (memcpy_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "Insert Identity node fail.");
    return nullptr;
  }

  return memcpy_node;
}

///
/// @brief Check duplicate node_name
/// @param [in] std::string& node_name
/// @return std::string
///
std::string HcclContinuousMemcpyPass::CheckDuplicateName(const std::string &node_name) {
  std::string tmp_name = node_name;
  auto iter = node_num_map_.find(tmp_name);
  if (iter != node_num_map_.end()) {
    tmp_name = tmp_name + "_" + std::to_string(iter->second);
    (iter->second)++;
  } else {
    node_num_map_[tmp_name] = 1;
  }
  return tmp_name;
}

///
/// @brief Modify edge connection
/// @param [in] ComputeGraphPtr graph
/// @param [in] OutDataAnchorPtr src_out_anchor
/// @param [in] InDataAnchorPtr hccl_in_anchor
/// @return status
///
Status HcclContinuousMemcpyPass::ModifyEdgeConnection(const ComputeGraphPtr &graph, const OutDataAnchorPtr &src_out_anchor,
                                            const InDataAnchorPtr &hccl_in_anchor) {
  GE_CHECK_NOTNULL(src_out_anchor->GetOwnerNode());
  GE_CHECK_NOTNULL(hccl_in_anchor->GetOwnerNode());

  Status ret = InsertIdentityBeforeHccl(graph, src_out_anchor, hccl_in_anchor);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "add identity failed, var_node:%s, hccl_node:%s.",
           src_out_anchor->GetOwnerNode()->GetName().c_str(),
           hccl_in_anchor->GetOwnerNode()->GetName().c_str());
    return ret;
  }

  ret = InsertAssignAfterBroadcastIfNeed(graph, src_out_anchor, hccl_in_anchor);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "add assign failed, var_node:%s, hccl_node:%s.",
           src_out_anchor->GetOwnerNode()->GetName().c_str(),
           hccl_in_anchor->GetOwnerNode()->GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

///
/// @brief Insert Identity node Between Hccl node and variable
/// @param [in] ComputeGraphPtr graph
/// @param [in] OutDataAnchorPtr src_out_anchor
/// @param [in] InDataAnchorPtr hccl_in_anchor
/// @return status
///
Status HcclContinuousMemcpyPass::InsertIdentityBeforeHccl(const ComputeGraphPtr &graph, const OutDataAnchorPtr &src_out_anchor,
                                                const InDataAnchorPtr &hccl_in_anchor) {
  GELOGI("Between op %s and op %s need insert memcpy async op.", src_out_anchor->GetOwnerNode()->GetName().c_str(),
         hccl_in_anchor->GetOwnerNode()->GetName().c_str());
  NodePtr memcpy_node = CreateIdentityNode(graph, src_out_anchor);
  GE_CHECK_NOTNULL(memcpy_node);

  Status ret1 = src_out_anchor->Unlink(hccl_in_anchor);
  if (ret1 != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "The op %s Unlink anchor %s fail.", src_out_anchor->GetOwnerNode()->GetName().c_str(),
           hccl_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }
  auto out_data_anchor_0 = memcpy_node->GetOutDataAnchor(kAnchorNum);
  GE_CHECK_NOTNULL(out_data_anchor_0);
  ret1 = out_data_anchor_0->LinkTo(hccl_in_anchor);
  if (ret1 != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "The op %s link anchor %s  fail.", memcpy_node->GetName().c_str(),
           hccl_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }

  Status ret = src_out_anchor->LinkTo(memcpy_node->GetInDataAnchor(kAnchorNum));
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "The op %s link anchor %s fail.", src_out_anchor->GetOwnerNode()->GetName().c_str(),
           memcpy_node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

///
/// @brief Insert assign node after broadcast node and variable to refresh variable data
/// @param [in] ComputeGraphPtr graph
/// @param [in] OutDataAnchorPtr var_out_anchor
/// @param [in] InDataAnchorPtr hccl_in_anchor
/// @return status
///
Status HcclContinuousMemcpyPass::InsertAssignAfterBroadcastIfNeed(const ComputeGraphPtr &graph,
                                                        const OutDataAnchorPtr &var_out_anchor,
                                                        const InDataAnchorPtr &hccl_in_anchor) {
  if (hccl_in_anchor->GetOwnerNode()->GetType() != HCOMBROADCAST) {
    GELOGD("%s not broadcast, no need to insert assign node", hccl_in_anchor->GetOwnerNode()->GetName().c_str());
    return SUCCESS;
  }

  if (var_out_anchor->GetOwnerNode()->GetType() != VARIABLE) {
    GELOGD("%s not variable, no need to insert assign node", var_out_anchor->GetOwnerNode()->GetName().c_str());
    return SUCCESS;
  }

  GELOGI("after op %s and op %s need insert assign op.", var_out_anchor->GetOwnerNode()->GetName().c_str(),
         hccl_in_anchor->GetOwnerNode()->GetName().c_str());

  for (auto peer_in_anchor : var_out_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor->GetOwnerNode()->GetType() == ASSIGN) {
      GELOGD("variable %s out assign node is exist.", var_out_anchor->GetOwnerNode()->GetName().c_str());
      return SUCCESS;
    }
  }

  NodePtr assign_node = CreateAssignNode(graph, var_out_anchor);
  GE_CHECK_NOTNULL(assign_node);

  OutDataAnchorPtr hccl_out_anchor = hccl_in_anchor->GetOwnerNode()->GetOutDataAnchor(hccl_in_anchor->GetIdx());
  GE_CHECK_NOTNULL(hccl_out_anchor);

  Status ret = hccl_out_anchor->LinkTo(assign_node->GetInDataAnchor(kAnchorAssignValueIndex));
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "The op %s link anchor %s fail.", hccl_out_anchor->GetOwnerNode()->GetName().c_str(),
           assign_node->GetName().c_str());
    return FAILED;
  }

  ret = var_out_anchor->LinkTo(assign_node->GetInDataAnchor(kAnchorAssignRefIndex));
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "The op %s link anchor %s fail.", var_out_anchor->GetOwnerNode()->GetName().c_str(),
           assign_node->GetName().c_str());
    return FAILED;
  }

  // add control edge between assign node and node after broadcast node
  OutControlAnchorPtr assign_out_control_anchor = assign_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(assign_out_control_anchor);

  for (auto in_data_anchor : hccl_out_anchor->GetPeerInDataAnchors()) {
    if (in_data_anchor->GetOwnerNode()->GetName() == assign_node->GetName()) {
      continue;
    }
    ret = assign_out_control_anchor->LinkTo(in_data_anchor->GetOwnerNode()->GetInControlAnchor());
      if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "The op %s link control anchor %s fail.",
             assign_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             in_data_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
  }

  for (auto in_control_anchor : hccl_out_anchor->GetOwnerNode()->GetOutControlAnchor()->GetPeerInControlAnchors()) {
    if (in_control_anchor->GetOwnerNode()->GetName() == assign_node->GetName()) {
      continue;
    }
    ret = assign_out_control_anchor->LinkTo(in_control_anchor);
      if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "The op %s link control anchor %s fail.",
             assign_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             in_control_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

///
/// @brief create assign Node, add to graph
/// @param [in] ge::ComputeGraphPtr graph
/// @param [in] ge::OutDataAnchorPtr variable node out anchor
/// @return ge::NodePtr
///
NodePtr HcclContinuousMemcpyPass::CreateAssignNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor) {
  GE_CHECK_NOTNULL_EXEC(graph , return nullptr);
  NodePtr pre_node = out_data_anchor->GetOwnerNode();
  OpDescPtr pre_op_desc = pre_node->GetOpDesc();
  if (pre_op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "OpDesc of pre node is invalid.");
    return nullptr;
  }

  std::string node_name = pre_node->GetName() + "_" + ASSIGN;
  node_name = CheckDuplicateName(node_name);
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name.c_str(), ASSIGN);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Create Assign op: MakeShared op_desc fail.");
    return nullptr;
  }
  GELOGI("Create Assign op:%s.", op_desc->GetName().c_str());

  graphStatus ret = op_desc->AddInputDesc("ref", pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Create Assign op: add ref input desc fail.");
    return nullptr;
  }

  ret = op_desc->AddInputDesc("value", pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Create Assign op: add value input desc fail.");
    return nullptr;
  }

  ret = op_desc->AddOutputDesc("ref", pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Create Assign op: add output desc fail.");
    return nullptr;
  }

  NodePtr assign_node = graph->AddNode(op_desc);
  if (assign_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "Insert Identity node fail.");
    return nullptr;
  }

  return assign_node;
}


///
/// @brief Clear Status, used for subgraph pass
/// @return SUCCESS
///
Status HcclContinuousMemcpyPass::ClearStatus() {
  node_num_map_.clear();
  return SUCCESS;
}
}  // namespace ge
