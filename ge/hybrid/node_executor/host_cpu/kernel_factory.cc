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

#include "hybrid/node_executor/host_cpu/kernel_factory.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace hybrid {
namespace host_cpu {
KernelFactory &KernelFactory::Instance() {
  static KernelFactory instance;
  return instance;
}

std::shared_ptr<Kernel> KernelFactory::CreateKernel(const NodePtr &node) {
  if (node == nullptr) {
    GELOGW("node is NULL.");
    return nullptr;
  }
  auto iter = kernel_creator_map_.find(node->GetType());
  if (iter != kernel_creator_map_.end()) {
    return iter->second(node);
  }
  REPORT_INNER_ERROR("E19999", "Not supported because kernel_creator_map_ not contain type:%s, name = %s",
                     node->GetType().c_str(), node->GetName().c_str());
  GELOGE(FAILED, "[Find][NodeType]Not supported because kernel_creator_map_ not contain type = %s, name = %s",
         node->GetType().c_str(), node->GetName().c_str());
  return nullptr;
}

void KernelFactory::RegisterCreator(const std::string &type, const KERNEL_CREATOR_FUNC &func) {
  if (func == nullptr) {
    GELOGW("Func is NULL.");
    return;
  }
  auto iter = kernel_creator_map_.find(type);
  if (iter != kernel_creator_map_.end()) {
    GELOGW("%s creator already exist", type.c_str());
    return;
  }
  kernel_creator_map_[type] = func;
}
}  // namespace host_cpu
}  // namespace hybrid
}  // namespace ge
