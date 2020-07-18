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