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

#include "graph/passes/no_reshape_op_remove_pass.h"

#include <string>
#include <vector>

#include "common/op/attr_value_util.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace {
const char *const kReshapeName = "Reshape_3";
}  // namespace
Status NoReshapeOpRemovePass::Run(ge::NodePtr &node) {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "param [node] must not be null");
    return PARAM_INVALID;
  }
  OpDescPtr op_desc_ptr = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc_ptr);
  if ((op_desc_ptr->GetType() == EXPANDDIMS) || (op_desc_ptr->GetType() == SQUEEZE)) {
    return CheckNodeShapeAndForamt(node);
  } else if (op_desc_ptr->GetType() == RESHAPE) {
    if (op_desc_ptr->GetName() == kReshapeName) {
      std::vector<string> types;
      std::list<ge::NodePtr> path;
      path.push_back(node);
      types.emplace_back(PERMUTE);
      types.emplace_back(TRANSDATA);
      types.emplace_back(CORRELATION);
      // check reshape out data node fit specific type
      bool reshape_correlation_flag = true;
      for (const auto &type : types) {
        if (!CheckOutDataNodesType(type, path)) {
          reshape_correlation_flag = false;
          break;
        }
      }
      if (reshape_correlation_flag) {
        path.pop_front();
        GE_IF_BOOL_EXEC(!AttrUtils::SetBool(path.front()->GetOpDesc(), "reshape_correlation", reshape_correlation_flag),
                        GELOGE(INTERNAL_ERROR, "set reshape_correlation failed");
                        return INTERNAL_ERROR);
      }
      path.clear();
      types.clear();
    }

    if (domi::GetContext().format == domi::DOMI_TENSOR_NCHW && !op_desc_ptr->HasAttr(PERMUTE_ATTR_ORDER)) {
      std::list<ge::NodePtr> path;
      path.push_back(node);
      string correlation = CORRELATION;
      if (CheckOutDataNodesType(correlation, path)) {
        op_desc_ptr->SetType(PERMUTE);
        if (AttrUtils::SetListInt(op_desc_ptr, PERMUTE_ATTR_ORDER, vector<int64_t>{2, 3, 0, 1})) {
          GELOGE(INTERNAL_ERROR, "Set permute attr order failed");
          return INTERNAL_ERROR;
        }
        path.clear();
        return SUCCESS;
      }
    }

    // prefer handle linked reshape than single reshape
    vector<ge::NodePtr> delete_nodes = CheckLinkedReshape(node);
    if (delete_nodes.empty()) {
      return CheckNodeShapeAndForamt(node);
    }
    Status ret;
    for (NodePtr &delete_node : delete_nodes) {
      GE_CHECK_NOTNULL(delete_node);
      GELOGI("NoReshapeOpRemovePass remove node:%s", delete_node->GetName().c_str());
      ret = IsolateAndDeleteNode(delete_node, {0});
      if (ret != SUCCESS) {
        GELOGE(ret, "NoReshapeOpRemovePass remove node failed,ret:%u", ret);
        return ret;
      }
    }
  }
  return SUCCESS;
}

bool NoReshapeOpRemovePass::CheckOutDataNodesType(const string &type, std::list<ge::NodePtr> &path) {
  if (path.empty()) {
    return false;
  }
  Node::Vistor<NodePtr> out_data_nodes = path.back()->GetOutDataNodes();
  bool flag = false;
  GE_IF_BOOL_EXEC(out_data_nodes.at(0)->GetOpDesc() == nullptr, GELOGE(FAILED, "out_data_nodes GetOpDesc is nullptr");
                  return false);
  if ((out_data_nodes.size() == 1) && (out_data_nodes.at(0)->GetOpDesc()->GetType() == type)) {
    path.push_back(out_data_nodes.at(0));
    flag = true;
  }
  return flag;
}

// if single node input and output shape is same can be delete
Status NoReshapeOpRemovePass::CheckNodeShapeAndForamt(ge::NodePtr &node) {
  bool to_be_deleted = false;
  OpDescPtr op_desc_ptr = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc_ptr);
  if (op_desc_ptr->GetAllInputsDescPtr().empty()) {
    GELOGE(INTERNAL_ERROR, "Input num check fail. node name:%s", op_desc_ptr->GetName().c_str());
    return INTERNAL_ERROR;
  }
  GE_CHECK_NOTNULL(op_desc_ptr->GetInputDescPtr(0));
  if (op_desc_ptr->GetInputDescPtr(0)->GetFormat() == FORMAT_ND) {
    to_be_deleted = true;
  } else {
    to_be_deleted = true;
    // compare input and output dims
    std::vector<int64_t> input_4dims;
    GE_CHK_STATUS_RET(OpUtils::TransferDim(op_desc_ptr->GetInputDesc(0).GetShape().GetDims(), input_4dims),
                      "transfer dim failed");

    std::vector<int64_t> output_4dims;
    GE_CHK_STATUS_RET(OpUtils::TransferDim(op_desc_ptr->GetOutputDesc(0).GetShape().GetDims(), output_4dims),
                      "transfer dim failed");

    size_t vec_size = (input_4dims.size() > output_4dims.size()) ? output_4dims.size() : input_4dims.size();

    for (size_t i = 0; i < vec_size; i++) {
      if (input_4dims[i] != output_4dims[i]) {
        to_be_deleted = false;
        break;
      }
    }
  }
  if (to_be_deleted) {
    GELOGI("NoReshapeOpRemovePass remove node:%s", node->GetName().c_str());
    return IsolateAndDeleteNode(node, {0});
  }
  return SUCCESS;
}

// check Reshape->Reshape linked case if can be delete
vector<ge::NodePtr> NoReshapeOpRemovePass::CheckLinkedReshape(ge::NodePtr &node) {
  std::list<ge::NodePtr> node_path;
  std::vector<ge::NodePtr> delete_nodes;
  GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, GELOGE(FAILED, "Node OpDesc is nullptr"); return delete_nodes);
  const auto &cur_input_desc = node->GetOpDesc()->GetInputDesc(0);
  vector<int64_t> cur_input_dims = cur_input_desc.GetShape().GetDims();
  Format cur_input_format = cur_input_desc.GetFormat();
  node_path.push_back(node);
  // from front to back find longest sequence reshape can be delete
  while (!node_path.empty()) {
    const auto src_node = node_path.back();
    if (src_node == nullptr) {
      continue;
    }
    Node::Vistor<NodePtr> out_data_nodes = src_node->GetOutDataNodes();
    if ((out_data_nodes.size() == 1) && (out_data_nodes.at(0)->GetOpDesc() != nullptr)
        && (out_data_nodes.at(0)->GetOpDesc()->GetType() == RESHAPE)) {
      NodePtr dst_node = out_data_nodes.at(0);
      node_path.push_back(dst_node);
      GeTensorDesc dst_output_desc = dst_node->GetOpDesc()->GetOutputDesc(0);
      vector<int64_t> dst_output_dims = dst_output_desc.GetShape().GetDims();
      if ((cur_input_dims.size() == dst_output_dims.size()) && (cur_input_format == dst_output_desc.GetFormat())) {
        bool is_reshape_delete = true;
        for (size_t i = 0; i < cur_input_dims.size(); i++) {
          if (cur_input_dims[i] != dst_output_dims[i]) {
            is_reshape_delete = false;
          }
        }
        if (is_reshape_delete) {
          delete_nodes.insert(delete_nodes.begin(), node_path.begin(), node_path.end());
        }
      }
    } else {
      break;
    }
  }
  node_path.clear();
  return delete_nodes;
}
}  // namespace ge
