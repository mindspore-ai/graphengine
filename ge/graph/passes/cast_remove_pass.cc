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

#include "graph/passes/cast_remove_pass.h"
#include <string>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "graph/common/transop_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"

namespace ge {
Status CastRemovePass::Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "Param [node] must not be null.");
    return PARAM_INVALID;
  }
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param op_desc of node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "OpDesc of param [node] must not be null.");
    return PARAM_INVALID;
  }

  // begin with not trans op, and only has one out data anchor
  if (TransOpUtil::IsTransOp(node) || node->GetAllOutDataAnchorsSize() != 1) {
    return SUCCESS;
  }

  std::vector<NodePtr> nodes_to_fuse;
  NodePtr end_node = GetTheEndNode(node, nodes_to_fuse);
  if (nodes_to_fuse.empty()) {
    return SUCCESS;
  }
  OpDescPtr end_op_desc = end_node->GetOpDesc();
  if (end_op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "op_desc of end_node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "OpDesc of end node must not be null.");
    return PARAM_INVALID;
  }

  if (!CheckPrecisionLoss(nodes_to_fuse)) {
    return SUCCESS;
  }

  DataType type = DT_UNDEFINED;
  if (!HasSameDataType(op_desc, end_op_desc, type)) {
    return SUCCESS;
  }
  auto instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "gelib is not initilized!");
    return FAILED;
  }

  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  return DoFuse(ops_kernel_manager, type, nodes_to_fuse);
}

bool CastRemovePass::CheckPrecisionLoss(const std::vector<NodePtr> &nodes_to_fuse) {
  for (const NodePtr &node : nodes_to_fuse) {
    if (!TransOpUtil::CheckPrecisionLoss(node)) {
      return false;
    }
  }
  return true;
}

bool CastRemovePass::HasSameDataType(OpDescPtr &begin_op_desc, OpDescPtr &end_op_desc, DataType &type) const {
  if (begin_op_desc->GetName() == end_op_desc->GetName()) {
    return false;
  }
  auto end_out_desc = end_op_desc->MutableOutputDesc(0);
  DataType end_out_datatype = end_out_desc->GetDataType();

  auto begin_out_desc = begin_op_desc->MutableOutputDesc(0);
  DataType begin_out_datatype = begin_out_desc->GetDataType();
  if (begin_out_datatype == end_out_datatype && (begin_out_datatype == DT_FLOAT16 || begin_out_datatype == DT_FLOAT)) {
    type = begin_out_datatype;
    return true;
  }
  return false;
}

// op1->TransData->Cast->TransposeD->Cast->TransData->op2
// change to be
// op1->TransData->TransposeD->TransData->op2
Status CastRemovePass::DoFuse(const OpsKernelManager &ops_kernel_manager,
                              const DataType &type,
                              std::vector<NodePtr> &nodes_to_fuse) {
  std::vector<size_t> to_be_deleted_cast_index;
  for (size_t i = 0; i < nodes_to_fuse.size(); i++) {
    NodePtr node = nodes_to_fuse[i];
    if (node->GetType() == CAST) {
      to_be_deleted_cast_index.emplace_back(i);
      continue;
    }
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      REPORT_INNER_ERROR("E19999", "Find nullptr op_desc in node, check invalid");
      GELOGE(FAILED, "OpDesc must not be null.");
      return FAILED;
    }
    auto in_desc = op_desc->MutableInputDesc(0);
    auto out_desc = op_desc->MutableOutputDesc(0);
    auto in_desc_org_dtype = in_desc->GetDataType();
    auto out_desc_org_dtype = out_desc->GetDataType();
    in_desc->SetDataType(type);
    out_desc->SetDataType(type);
    bool is_supported = false;
    for (const auto &ops_kernel_store_info : ops_kernel_manager.GetAllOpsKernelInfoStores()) {
      map<string, OpInfo> op_infos;
      ops_kernel_store_info.second->GetAllOpsKernelInfo(op_infos);
      if (op_infos.find(op_desc->GetType()) == op_infos.end()) {
        continue;
      }
      string un_supported_reason;
      is_supported = ops_kernel_store_info.second->CheckAccuracySupported(op_desc, un_supported_reason);
      if (is_supported) {
        break;
      }
    }
    if (!is_supported) {
      // if no operator_info_store supported, do nothing
      in_desc->SetDataType(in_desc_org_dtype);
      out_desc->SetDataType(out_desc_org_dtype);
      to_be_deleted_cast_index.clear();
      return SUCCESS;
    }

    // add attr to changed TransData, then will be rebuild
    if (!AttrUtils::SetBool(op_desc, ATTR_NEED_COMPILE, true)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s of op:%s(%s) failed",
                        ATTR_NEED_COMPILE.c_str(),
                        op_desc->GetName().c_str(),
                        op_desc->GetType().c_str());
      GELOGE(FAILED, "Set ATTR_NEED_COMPILE Attr fail.");
      return FAILED;
    }
    GELOGI("CastRemovePass, change %s %s datatype to be %s.", node->GetType().c_str(), node->GetName().c_str(),
           TypeUtils::DataTypeToSerialString(type).c_str());
  }
  return DoRemoveCast(to_be_deleted_cast_index, nodes_to_fuse);
}

Status CastRemovePass::DoRemoveCast(const std::vector<size_t> &to_be_deleted_cast_index,
                                    std::vector<NodePtr> &nodes_to_fuse) {
  for (auto &cast_idx : to_be_deleted_cast_index) {
    GELOGI("CastRemovePass, remove Cast %s.", nodes_to_fuse[cast_idx]->GetName().c_str());
    if (IsolateAndDeleteNode(nodes_to_fuse[cast_idx], {0}) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Isolate and delete node:%s(%s) failed when CastRemovePass %s",
                        nodes_to_fuse[cast_idx]->GetName().c_str(),
                        nodes_to_fuse[cast_idx]->GetType().c_str(),
                        __FUNCTION__);
      GELOGE(FAILED, "IsolateAndDeleteNode %s failed.", nodes_to_fuse[cast_idx]->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

NodePtr CastRemovePass::GetTheEndNode(NodePtr begin_node, std::vector<NodePtr> &nodes_to_fuse) {
  while (begin_node->GetOutDataNodes().size() == 1) {
    auto out_node = begin_node->GetOutDataNodes().at(0);
    if (!TransOpUtil::IsTransOp(out_node)) {
      return begin_node;  // when seen not trans op
    }
    begin_node = out_node;
    nodes_to_fuse.emplace_back(begin_node);
  }
  return begin_node;  // when seen branch
}
}  // namespace ge
