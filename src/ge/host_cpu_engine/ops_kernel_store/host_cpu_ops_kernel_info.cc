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

#include "host_cpu_engine/ops_kernel_store/host_cpu_ops_kernel_info.h"
#include <memory>
#include "common/constant/constant.h"
#include "ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "op/op_factory.h"

namespace ge {
namespace host_cpu {
using domi::TaskDef;
using std::map;
using std::string;
using std::vector;

Status HostCpuOpsKernelInfoStore::Initialize(const map<string, string> &options) {
  GELOGI("HostCpuOpsKernelInfoStore init start.");
  OpInfo default_op_info = {.engine = kHostCpuEngineName,
                            .opKernelLib = kHostCpuOpKernelLibName,
                            .computeCost = 0,
                            .flagPartial = false,
                            .flagAsync = false,
                            .isAtomic = false};
  // Init op_info_map_
  auto all_ops = OpFactory::Instance().GetAllOps();
  for (auto &op : all_ops) {
    op_info_map_[op] = default_op_info;
  }

  GELOGI("HostCpuOpsKernelInfoStore inited success. op num=%zu", op_info_map_.size());

  return SUCCESS;
}

Status HostCpuOpsKernelInfoStore::Finalize() {
  op_info_map_.clear();
  return SUCCESS;
}

void HostCpuOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, OpInfo> &infos) const { infos = op_info_map_; }

bool HostCpuOpsKernelInfoStore::CheckSupported(const OpDescPtr &op_desc, std::string &) const {
  if (op_desc == nullptr) {
    return false;
  }
  return op_info_map_.count(op_desc->GetType()) > 0;
}
}  // namespace host_cpu
}  // namespace ge
