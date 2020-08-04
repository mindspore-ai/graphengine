/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "register/op_kernel_registry.h"

namespace ge {
class OpKernelRegistry::OpKernelRegistryImpl {

};

OpKernelRegistry::OpKernelRegistry() {
}

OpKernelRegistry::~OpKernelRegistry() {

}

bool OpKernelRegistry::IsRegistered(const std::string &op_type) {
  return false;
}

std::unique_ptr<HostCpuOp> OpKernelRegistry::CreateHostCpuOp(const std::string &op_type) {
  return nullptr;
}

void OpKernelRegistry::RegisterHostCpuOp(const std::string &op_type, CreateFn create_fn) {
}

HostCpuOpRegistrar::HostCpuOpRegistrar(const char *op_type, HostCpuOp *(*create_fn)()) {

}
} // namespace ge