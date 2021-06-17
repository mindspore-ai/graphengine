/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "stub_ops_kernel_store.h"
#include <memory>
#include "ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "op/stub_op_factory.h"

namespace ge {
namespace st {
using domi::TaskDef;
using std::map;
using std::string;
using std::vector;

Status StubOpsKernelInfoStore::Initialize(const map<string, string> &options) {
  GELOGI("StubOpsKernelInfoStore init start.");
  string engine_name;
  for (const auto &engine_2_lib : kStubEngine2KernelLib) {
    if (engine_2_lib.second == store_name_) {
      engine_name = engine_2_lib.first;
    }
  }
  if (engine_name.empty()) {
    return FAILED;
  }

  OpInfo default_op_info = {.engine = engine_name,
                            .opKernelLib = store_name_,
                            .computeCost = 0,
                            .flagPartial = false,
                            .flagAsync = false,
                            .isAtomic = false};
  // Init op_info_map_
  auto all_ops_in_store = OpFactory::Instance().GetAllOps(store_name_);
  for (auto &op : all_ops_in_store) {
    op_info_map_[op] = default_op_info;
  }

  GELOGI("StubOpsKernelInfoStore inited success. op num=%zu", op_info_map_.size());
  return SUCCESS;
}

Status StubOpsKernelInfoStore::Finalize() {
  op_info_map_.clear();
  return SUCCESS;
}

void StubOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, OpInfo> &infos) const {
  infos = op_info_map_;
}

bool StubOpsKernelInfoStore::CheckSupported(const OpDescPtr &op_desc, std::string &) const {
  if (op_desc == nullptr) {
    return false;
  }
  return op_info_map_.count(op_desc->GetType()) > 0;
}
}  // namespace st
}  // namespace ge
