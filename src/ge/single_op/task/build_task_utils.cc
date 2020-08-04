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

#include "single_op/task/build_task_utils.h"

#include "runtime/rt.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace {
const uint64_t kSessionId = UINT64_MAX;
uint8_t *kVarBase = nullptr;
const uint64_t kLogicVarBase = 0;
const uint64_t kVarSize = 0;
}  // namespace

std::vector<std::vector<void *>> BuildTaskUtils::GetAddresses(const OpDescPtr &op_desc,
                                                              const SingleOpModelParam &param) {
  std::vector<std::vector<void *>> ret;
  RuntimeParam runtime_para;
  runtime_para.mem_size = param.memory_size;
  runtime_para.logic_mem_base = param.base_addr;
  runtime_para.mem_base = param.mem_base;
  runtime_para.weight_size = param.weight_size;
  runtime_para.logic_weight_base = param.weight_addr;
  runtime_para.weight_base = param.weight_base;
  runtime_para.var_size = kVarSize;
  runtime_para.logic_var_base = kLogicVarBase;
  runtime_para.var_base = kVarBase;
  runtime_para.session_id = kSessionId;

  ret.emplace_back(ModelUtils::GetInputDataAddrs(runtime_para, op_desc));
  ret.emplace_back(ModelUtils::GetOutputDataAddrs(runtime_para, op_desc));
  ret.emplace_back(ModelUtils::GetWorkspaceDataAddrs(runtime_para, op_desc));
  return ret;
}

std::vector<void *> BuildTaskUtils::JoinAddresses(const std::vector<std::vector<void *>> &addresses) {
  std::vector<void *> ret;
  for (auto &address : addresses) {
    ret.insert(ret.end(), address.begin(), address.end());
  }
  return ret;
}

std::vector<void *> BuildTaskUtils::GetKernelArgs(const OpDescPtr &op_desc, const SingleOpModelParam &param) {
  auto addresses = GetAddresses(op_desc, param);
  return JoinAddresses(addresses);
}
}  // namespace ge
