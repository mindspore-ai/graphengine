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

#include "graph/passes/hccl_memcpy_pass.h"

#include <string>

#include "common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge/ge_util.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"

namespace {
const int32_t kAnchorSize = 1;
const int kAnchorNum = 0;
}  // namespace
namespace ge {
Status HcclMemcpyPass::Run(ge::ComputeGraphPtr graph) {
  GE_IF_BOOL_EXEC(graph == nullptr, GELOGE(PARAM_INVALID, "param [graph] must not be null."); return PARAM_INVALID);
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
    if (!NeedInsertMemcpyOp(op_desc)) {
      continue;
    }

    GELOGI("hcom op is:%s.", op_desc->GetName().c_str());
    for (auto &hccl_in_anchor : node->GetAllInDataAnchors()) {
      if (hccl_in_anchor == nullptr) {
        continue;
      }
      auto src_out_anchor = hccl_in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(src_out_anchor);

      int32_t src_out_anchor_size = src_out_anchor->GetPeerInDataAnchors().size();
      if (src_out_anchor_size == kAnchorSize) {
        // Memcpyasync needs to be inserted between constant (/data) and hcomallreduce to avoid constant being cleared.
        NodePtr src_node = src_out_anchor->GetOwnerNode();
        std::string src_type = src_node->GetType();
        bool check_src_type = (src_type == CONSTANTOP) || (src_type == DATA);
        if (check_src_type && node->GetType() == HCOMALLREDUCE) {
          Status ret = ModifyEdgeConnection(graph, src_out_anchor, hccl_in_anchor);
          if (ret != SUCCESS) {
            GELOGE(INTERNAL_ERROR, "Failed to modify the connection.");
            return ret;
          }
        }
        continue;
      }

      Status ret = ModifyEdgeConnection(graph, src_out_anchor, hccl_in_anchor);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to modify the connection.");
        return ret;
      }
    }
  }
  return SUCCESS;
}

///
/// @brief Add MemcpyAsync Node
/// @param [in] ge::ComputeGraphPtr graph
/// @param [in] ge::OutDataAnchorPtr in_node
/// @return ge::NodePtr
///
NodePtr HcclMemcpyPass::CreateMemcpyNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor) {
  GE_IF_BOOL_EXEC(graph == nullptr, return nullptr);
  NodePtr pre_node = out_data_anchor->GetOwnerNode();
  OpDescPtr pre_op_desc = pre_node->GetOpDesc();
  if (pre_op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "OpDesc of pre node is invalid.");
    return nullptr;
  }

  std::string node_name = pre_node->GetName() + "_" + MEMCPYASYNC;
  node_name = CheckDuplicateName(node_name);
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name.c_str(), MEMCPYASYNC);
  if (op_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "Create MemcpyAsync op: MakeShared op_desc fail.");
    return nullptr;
  }
  GELOGI("Create MemcpyAsync op:%s.", op_desc->GetName().c_str());

  graphStatus ret = op_desc->AddInputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Create MemcpyAsync op: add input desc fail.");
    return nullptr;
  }

  ret = op_desc->AddOutputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Create MemcpyAsync op: add output desc fail.");
    return nullptr;
  }

  NodePtr memcpy_node = graph->AddNode(op_desc);
  if (memcpy_node == nullptr) {
    GELOGE(INTERNAL_ERROR, "Insert MemcpyAsync node fail.");
    return nullptr;
  }

  return memcpy_node;
}

///
/// @brief Check duplicate node_name
/// @param [in] std::string& node_name
/// @return std::string
///
std::string HcclMemcpyPass::CheckDuplicateName(const std::string &node_name) {
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
/// @brief Check hcom op
/// @param [in] ge::ConstOpDescPtr op_desc
/// @return bool
///
bool HcclMemcpyPass::NeedInsertMemcpyOp(const ge::ConstOpDescPtr &op_desc) const {
  return (op_desc->GetType() == HCOMALLGATHER || op_desc->GetType() == HCOMALLREDUCE ||
          op_desc->GetType() == HCOMREDUCESCATTER);
}

///
/// @brief Modify edge connection
/// @param [in] ComputeGraphPtr graph
/// @param [in] OutDataAnchorPtr src_out_anchor
/// @param [in] InDataAnchorPtr hccl_in_anchor
/// @return status
///
Status HcclMemcpyPass::ModifyEdgeConnection(const ComputeGraphPtr &graph, const OutDataAnchorPtr &src_out_anchor,
                                            const InDataAnchorPtr &hccl_in_anchor) {
  GELOGI("The op %s need insert memcpy async op.", src_out_anchor->GetOwnerNode()->GetName().c_str());
  NodePtr memcpy_node = CreateMemcpyNode(graph, src_out_anchor);
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

}  // namespace ge
