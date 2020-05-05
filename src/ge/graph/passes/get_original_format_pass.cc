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

#include "graph/passes/get_original_format_pass.h"

#include <vector>

#include "common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "framework/omg/omg_inner_types.h"

using domi::DOMI_TENSOR_NCHW;
using domi::DOMI_TENSOR_NHWC;
using domi::DOMI_TENSOR_RESERVED;
using domi::FAILED;
using domi::PARAM_INVALID;
using domi::SUCCESS;

using domi::GetContext;

namespace ge {
Status GetOriginalFormatPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  GE_RETURN_WITH_LOG_IF_ERROR(SetOriginalFormat(graph), "SetOriginalFormat failed");

  return SUCCESS;
}

Status GetOriginalFormatPass::SetOriginalFormat(const ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  int64_t ori_format = 0;
  int64_t tmp_format = 0;

  for (auto &node_ptr : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr);

    GE_IF_BOOL_EXEC(!AttrUtils::SetInt(node_ptr->GetOpDesc(), ATTR_NAME_INFERRED_FORMAT, DOMI_TENSOR_RESERVED),
                    GELOGE(FAILED, "set ATTR_NAME_INFERRED_FORMAT failed");
                    return FAILED);
  }

  for (auto &node_ptr : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr);
    OpDescPtr desc_ptr = node_ptr->GetOpDesc();
    GE_CHECK_NOTNULL(desc_ptr);
    auto is_data = (desc_ptr->GetType() == DATA_TYPE || desc_ptr->GetType() == AIPP_DATA_TYPE);
    if (is_data) {
      GELOGI("Data node: %s,format :%d", node_ptr->GetName().c_str(), domi::GetContext().format);
      ori_format = static_cast<int64_t>(domi::GetContext().format);
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(desc_ptr, ATTR_NAME_FORMAT, ori_format),
                      GELOGE(FAILED, "set ATTR_NAME_FORMAT failed");
                      return FAILED);
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(desc_ptr, ATTR_NAME_INFERRED_FORMAT, ori_format),
                      GELOGE(FAILED, "set ATTR_NAME_INFERRED_FORMAT failed");
                      return FAILED);
      continue;
    }
    int32_t i = 0;
    bool continue_flag = false;
    bool ignore_pred_format = false;
    for (auto &bias_node_ptr : node_ptr->GetInDataNodes()) {
      GE_CHECK_NOTNULL(bias_node_ptr);

      OpDescPtr bias_op_ptr = bias_node_ptr->GetOpDesc();
      GE_CHECK_NOTNULL(bias_op_ptr);

      if (bias_op_ptr->GetType() == BIASADD) {
        ignore_pred_format = true;
        std::size_t tmp_size = ge::OpDescUtils::GetNonConstInputsSize(bias_node_ptr);
        GE_IF_BOOL_EXEC(tmp_size > 2 || tmp_size == 0,
                        GELOGW("bias_node is node followed by %zu nodes, should be 1 or 2", tmp_size);
                        continue_flag = true; break);
        OpDescPtr tmp_first_op_ptr = bias_node_ptr->GetInDataNodes().at(0)->GetOpDesc();
        GE_CHECK_NOTNULL(tmp_first_op_ptr);
        bias_op_ptr = tmp_first_op_ptr;

        // if biasadd have 2 input edges, format should be same
        if (tmp_size == 2) {
          int64_t first_input_format = 0;
          int64_t second_input_format = 0;
          OpDescPtr tmpSecondOpPtr = bias_node_ptr->GetInDataNodes().at(1)->GetOpDesc();
          GE_CHECK_NOTNULL(tmpSecondOpPtr);
          GE_IF_BOOL_EXEC(
            !AttrUtils::GetInt(tmp_first_op_ptr, ATTR_NAME_FORMAT, first_input_format), continue_flag = true; break);
          GE_IF_BOOL_EXEC(
            !AttrUtils::GetInt(tmpSecondOpPtr, ATTR_NAME_FORMAT, second_input_format), continue_flag = true; break);

          if (first_input_format != second_input_format) {
            GELOGW("biasadd node is followed two nodes with different format, get original format failed");
            continue_flag = true;
            break;
          }
        }
      }
      GE_IF_BOOL_EXEC(!AttrUtils::GetInt(bias_op_ptr, ATTR_NAME_FORMAT, tmp_format), continue_flag = true; break;);
      if (i == 0) {
        ori_format = tmp_format;
      }

      GE_IF_BOOL_EXEC(tmp_format != ori_format,
                      GELOGW("node: %s , original format of src nodes must be same!", bias_node_ptr->GetName().c_str());
                      continue_flag = true; break;);

      i++;
    }

    GE_IF_BOOL_EXEC(continue_flag, continue);
    OpDescPtr tmp_op_ptr = node_ptr->GetOpDesc();
    GE_CHECK_NOTNULL(tmp_op_ptr);

    if (IsFormatTranspose(tmp_op_ptr, static_cast<int32_t>(ori_format))) {
      ori_format = (ori_format == DOMI_TENSOR_NCHW) ? DOMI_TENSOR_NHWC : DOMI_TENSOR_NCHW;
    }

    if (ignore_pred_format) {
      GE_IF_BOOL_EXEC(!AttrUtils::SetBool(tmp_op_ptr, ATTR_NAME_IGNORE_PRED_FORMAT, true),
                      GELOGE(FAILED, "remove edge failed");
                      return FAILED);
    }

    // Do not reset ATTR_NAME_FORMAT if it is set in the OpParser.
    if (!tmp_op_ptr->HasAttr(ATTR_NAME_FORMAT)) {
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(tmp_op_ptr, ATTR_NAME_FORMAT, ori_format),
                      GELOGE(FAILED, "set ATTR_NAME_FORMAT failed");
                      return FAILED);
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(tmp_op_ptr, ATTR_NAME_INFERRED_FORMAT, ori_format),
                      GELOGE(FAILED, "set ATTR_NAME_INFERRED_FORMAT failed");
                      return FAILED);
    } else {
      int64_t existingFormat = 0;
      GE_RETURN_WITH_LOG_IF_FALSE(AttrUtils::GetInt(tmp_op_ptr, ATTR_NAME_FORMAT, existingFormat),
                                  "Get existing_format attr failed");
      if (!AttrUtils::SetInt(tmp_op_ptr, ATTR_NAME_INFERRED_FORMAT, existingFormat)) {
        GELOGE(FAILED, "set ATTR_NAME_INFERRED_FORMAT failed");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

bool GetOriginalFormatPass::IsFormatTranspose(const ge::OpDescPtr op_ptr, int32_t ori_format) {
  GE_CHK_BOOL_EXEC(op_ptr != nullptr, return false, "opdef is nullptr");
  if (op_ptr->GetType() == PERMUTE) {
    vector<int32_t> index_list;
    GE_IF_BOOL_EXEC(!AttrUtils::GetListInt(op_ptr, PERMUTE_ATTR_ORDER, index_list), return false);

    auto index_size = index_list.size();

    GE_IF_BOOL_EXEC(static_cast<int32_t>(index_size) != PERMUTE_ORDER_NUM, return false);

    int32_t perm_nchw[4] = {0, 2, 3, 1};  // 4 format nums, {0,2,3,1} NCHW -> NHWC
    int32_t perm_nhwc[4] = {0, 3, 1, 2};  // 4 format nums, {0,3,1,2} NHWC -> NCHW
    bool is_nchw = true;
    bool is_nhwc = true;
    for (size_t i = 0; i < index_size; ++i) {
      is_nchw = (perm_nchw[i] != index_list[i]) ? false : is_nchw;
      is_nhwc = (perm_nhwc[i] != index_list[i]) ? false : is_nhwc;
    }
    bool ret = (is_nchw && ori_format == DOMI_TENSOR_NCHW && !is_nhwc) ||
               (is_nhwc && ori_format == DOMI_TENSOR_NHWC && !is_nchw);

    return ret;
  }
  return false;
}
}  // namespace ge
