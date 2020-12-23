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
#include "keep_dtype_option.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"

namespace ge {
namespace {
const size_t kMaxOpsNum = 10;
}  // namespace
bool IsOriginalOpFind(OpDescPtr &op_desc, const std::string &op_name) {
  std::vector<std::string> original_op_names;
  if (!AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names)) {
    return false;
  }

  for (auto &origin_name : original_op_names) {
    if (origin_name == op_name) {
      return true;
    }
  }

  return false;
}

void KeepDtypeReportError(const std::vector<std::string> &invalid_list) {
  std::stringstream error_ops;
  for (size_t i = 0; i < invalid_list.size(); i++) {
    if (i == kMaxOpsNum) {
      error_ops << "...";
      break;
    }
    error_ops << invalid_list[i] << " ";
  }
  std::string err_msg = "config file contains ";
  err_msg = err_msg.append(std::to_string(invalid_list.size()))
                   .append(" operators not in the graph, op names:")
                   .append(error_ops.str());
  ErrorManager::GetInstance().ATCReportErrMessage(
      "E10042", {"parameter", "reason"}, {"keep_dtype", err_msg.c_str()});
  GELOGE(FAILED, "%s", err_msg.c_str());
}

Status DealKeepDtypeOption(const ComputeGraphPtr &graph, const std::string &keep_dtype) {
  GE_CHECK_NOTNULL(graph);
  if (keep_dtype.empty()) {
    return SUCCESS;
  }
  std::string real_path = RealPath(keep_dtype.c_str());
  if (real_path.empty()) {
    GELOGE(PARAM_INVALID, "Can not get real path for %s.", keep_dtype.c_str());
    return PARAM_INVALID;
  }
  std::ifstream ifs(real_path);
  if (!ifs.is_open()) {
    GELOGE(FAILED, "Open file %s failed", keep_dtype.c_str());
    return FAILED;
  }

  std::string op_name;
  std::vector<std::string> invalid_list;
  while (std::getline(ifs, op_name)) {
    if (op_name.empty()) {
      continue;
    }
    op_name = StringUtils::Trim(op_name);
    bool is_find = false;
    for (auto &node_ptr : graph->GetDirectNode()) {
      auto op_desc = node_ptr->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);

      if ((op_desc->GetName() == op_name) || IsOriginalOpFind(op_desc, op_name)) {
        is_find = true;
        (void)AttrUtils::SetInt(op_desc, ATTR_NAME_KEEP_DTYPE, 1);
      }
    }
    if (!is_find) {
      invalid_list.push_back(op_name);
    }
  }

  if (!invalid_list.empty()) {
    KeepDtypeReportError(invalid_list);
    return PARAM_INVALID;
  }

  return SUCCESS;
}
}  // namespace ge
