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

#include "stub_op_factory.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "graph/op_desc.h"

namespace ge {
namespace st {
OpFactory &OpFactory::Instance() {
  static OpFactory instance;
  return instance;
}

std::shared_ptr<Op> OpFactory::CreateOp(const Node &node, RunContext &run_context) {
  auto iter = op_creator_map_.find(node.GetType());
  if (iter != op_creator_map_.end()) {
    return iter->second(node, run_context);
  }
  GELOGE(FAILED, "Not supported OP, type = %s, name = %s", node.GetType().c_str(), node.GetName().c_str());
  return nullptr;
}

void OpFactory::RegisterCreator(const std::string &type, const std::string &kernel_lib, const OP_CREATOR_FUNC &func) {
  if (func == nullptr) {
    GELOGW("Func is NULL.");
    return;
  }

  if (all_store_ops_.find(kernel_lib) != all_store_ops_.end()) {
    all_store_ops_[kernel_lib].emplace_back(type);
  } else {
    all_store_ops_[kernel_lib] = {type};
  }
}
}  // namespace st
}  // namespace ge
