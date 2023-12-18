/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TBE_HANDLE_STORE_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TBE_HANDLE_STORE_H_

#include <cstdint>

#include <map>
#include <set>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "framework/common/fmk_types.h"
#include "graph/op_kernel_bin.h"

namespace ge {
class TbeHandleInfo {
 public:
  TbeHandleInfo(void *const handle, const OpKernelBinPtr &kernel) : handle_(handle), kernel_(kernel) {}

  ~TbeHandleInfo() { handle_ = nullptr; }

  void used_inc(const uint32_t num = 1U);
  void used_dec(const uint32_t num = 1U);
  uint32_t used_num() const;

  void *handle() const;

 private:
  uint32_t used_ = 0U;

  void *handle_;
  OpKernelBinPtr kernel_;
};

class TBEHandleStore {
 public:
  static TBEHandleStore &GetInstance();

  /// @ingroup ge
  /// @brief Find Registered TBE handle by name.
  /// @param [in] name: TBE handle name to find.
  /// @param [out] handle: TBE handle addr found.
  /// @return true: found / false: not found.
  bool FindTBEHandle(const std::string &name, void *&handle);

  /// @ingroup ge
  /// @brief Store registered TBE handle info.
  /// @param [in] name: TBE handle name to store.
  /// @param [in] handle: TBE handle addr to store.
  /// @param [in] kernel: TBE kernel bin to store.
  /// @return NA
  void StoreTBEHandle(const std::string &name, void *handle, const OpKernelBinPtr &kernel);

  /// @ingroup ge
  /// @brief Increase reference of registered TBE handle info.
  /// @param [in] name: handle name increase reference.
  /// @return NA
  void ReferTBEHandle(const std::string &name);

  /// @ingroup ge
  /// @brief Erase TBE registered handle record.
  /// @param [in] names: handle names erase.
  /// @return NA
  void EraseTBEHandle(const std::map<std::string, uint32_t> &names);

  void* GetUniqueIdPtr(void *const handle);

 private:
  TBEHandleStore() = default;
  ~TBEHandleStore() = default;

  std::mutex mutex_;
  std::unordered_map<std::string, TbeHandleInfo> kernels_;
  std::unordered_map<void *, std::list<uint8_t>> handle_unique_ids_;
};

class KernelHolder {
 public:
  KernelHolder(const char_t *const stub_func,
               const std::shared_ptr<OpKernelBin> &kernel_bin);

  ~KernelHolder();

  void SetBinHandle(void *const bin_handle) { bin_handle_ = bin_handle; }

 private:
  friend class KernelBinRegistry;
  const char_t *stub_func_;
  void *bin_handle_ = nullptr;
  std::shared_ptr<ge::OpKernelBin> kernel_bin_;
};

class HandleHolder {
 public:
  explicit HandleHolder(void *const bin_handle);
  ~HandleHolder();

  void SetBinHandle(void *const bin_handle) { bin_handle_ = bin_handle; }
  const void *GetBinHandle() const { return bin_handle_; }

 private:
  friend class HandleRegistry;
  void *bin_handle_;
};

class KernelBinRegistry {
 public:
  static KernelBinRegistry &GetInstance() {
    static KernelBinRegistry instance;
    return instance;
  }

  const char_t *GetUnique(const std::string &stub_func);

  const char_t *GetStubFunc(const std::string &stub_name);

  bool AddKernel(const std::string &stub_name, std::unique_ptr<KernelHolder> &&holder);

 private:
  std::map<std::string, std::unique_ptr<KernelHolder>> registered_bins_;
  std::set<std::string> unique_stubs_;
  std::mutex mutex_;
};

class HandleRegistry {
 public:
  static HandleRegistry &GetInstance() {
    static HandleRegistry instance;
    return instance;
  }

  bool AddHandle(std::unique_ptr<HandleHolder> &&holder);

 private:
  std::set<std::unique_ptr<HandleHolder>> registered_handles_;
};
}  // namespace ge

#endif  // NEW_GE_TBE_HANDLE_STORE_H
