#include "register/op_kernel_registry.h"
#include <mutex>
#include <map>
#include "graph/debug/ge_log.h"

namespace ge {
class OpKernelRegistry::OpKernelRegistryImpl {
 public:
  void RegisterHostCpuOp(const std::string &op_type, OpKernelRegistry::CreateFn create_fn) {
    std::lock_guard<std::mutex> lock(mu_);
    create_fns_[op_type] = create_fn;
  }

  OpKernelRegistry::CreateFn GetCreateFn(const std::string &op_type) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = create_fns_.find(op_type);
    if (it == create_fns_.end()) {
      return nullptr;
    }

    return it->second;
  }

 private:
  std::mutex mu_;
  std::map<std::string, OpKernelRegistry::CreateFn> create_fns_;
};

OpKernelRegistry::OpKernelRegistry() {
  impl_ = std::unique_ptr<OpKernelRegistryImpl>(new(std::nothrow) OpKernelRegistryImpl);
}

OpKernelRegistry::~OpKernelRegistry() {
}

bool OpKernelRegistry::IsRegistered(const std::string &op_type) {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to invoke IsRegistered %s, OpKernelRegistry is not properly initialized",
           op_type.c_str());
    return false;
  }

  return impl_->GetCreateFn(op_type) != nullptr;
}

void OpKernelRegistry::RegisterHostCpuOp(const std::string &op_type, CreateFn create_fn) {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to register %s, OpKernelRegistry is not properly initialized", op_type.c_str());
    return;
  }

  impl_->RegisterHostCpuOp(op_type, create_fn);
}
std::unique_ptr<HostCpuOp> OpKernelRegistry::CreateHostCpuOp(const std::string &op_type) {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to create op for %s, OpKernelRegistry is not properly initialized",
           op_type.c_str());
    return nullptr;
  }

  auto create_fn = impl_->GetCreateFn(op_type);
  if (create_fn == nullptr) {
    GELOGD("Host Cpu op is not registered. op type = %s", op_type.c_str());
    return nullptr;
  }

  return std::unique_ptr<HostCpuOp>(create_fn());
}

HostCpuOpRegistrar::HostCpuOpRegistrar(const char *op_type, HostCpuOp *(*create_fn)()) {
  if (op_type == nullptr) {
    GELOGE(PARAM_INVALID, "Failed to register host cpu op, op type is null");
    return;
  }

  OpKernelRegistry::GetInstance().RegisterHostCpuOp(op_type, create_fn);
}
} // namespace ge