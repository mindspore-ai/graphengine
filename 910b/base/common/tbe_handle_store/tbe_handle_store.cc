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

#include "common/tbe_handle_store/tbe_handle_store.h"

#include <limits>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/log.h"
#include "runtime/kernel.h"

namespace ge {
void TbeHandleInfo::used_inc(const uint32_t num) {
  if (used_ > (std::numeric_limits<uint32_t>::max() - num)) {
    REPORT_INNER_ERROR("E19999", "Used:%u reach numeric max", used_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Used[%u] reach numeric max.", used_);
    return;
  }

  used_ += num;
}

void TbeHandleInfo::used_dec(const uint32_t num) {
  if (used_ < (std::numeric_limits<uint32_t>::min() + num)) {
    REPORT_INNER_ERROR("E19999", "Used:%u reach numeric min", used_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Used[%u] reach numeric min.", used_);
    return;
  }

  used_ -= num;
}

uint32_t TbeHandleInfo::used_num() const {
  return used_;
}

void *TbeHandleInfo::handle() const {
  return handle_;
}


TBEHandleStore &TBEHandleStore::GetInstance() {
  static TBEHandleStore instance;

  return instance;
}

///
/// @ingroup ge
/// @brief Find Registered TBE handle by name.
/// @param [in] name: TBE handle name to find.
/// @param [out] handle: handle names record.
/// @return true: found / false: not found.
///
bool TBEHandleStore::FindTBEHandle(const std::string &name, void *&handle) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    return false;
  } else {
    TbeHandleInfo &info = it->second;
    handle = info.handle();
    return true;
  }
}

///
/// @ingroup ge
/// @brief Store registered TBE handle info.
/// @param [in] name: TBE handle name to store.
/// @param [in] handle: TBE handle addr to store.
/// @param [in] kernel: TBE kernel bin to store.
/// @return NA
///
void TBEHandleStore::StoreTBEHandle(const std::string &name, void *handle, const OpKernelBinPtr &kernel) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    TbeHandleInfo info(handle, kernel);
    info.used_inc();
    (void)kernels_.emplace(name, info);
  } else {
    TbeHandleInfo &info = it->second;
    info.used_inc();
  }
}

///
/// @ingroup ge
/// @brief Increase reference of registered TBE handle info.
/// @param [in] name: handle name increase reference.
/// @return NA
///
void TBEHandleStore::ReferTBEHandle(const std::string &name) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    REPORT_INNER_ERROR("E19999", "Kernel:%s not found in stored check invalid", name.c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Kernel[%s] not found in stored.", name.c_str());
    return;
  }

  TbeHandleInfo &info = it->second;
  info.used_inc();
}

///
/// @ingroup ge
/// @brief Erase TBE registered handle record.
/// @param [in] names: handle names erase.
/// @return NA
///
void TBEHandleStore::EraseTBEHandle(const std::map<std::string, uint32_t> &names) {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto &item : names) {
   const auto it = kernels_.find(item.first);
    if (it == kernels_.end()) {
      REPORT_INNER_ERROR("E19999", "Kernel:%s not found in stored check invalid", item.first.c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Kernel[%s] not found in stored.", item.first.c_str());
      continue;
    }

    TbeHandleInfo &info = it->second;
    if (info.used_num() > item.second) {
      info.used_dec(item.second);
    } else {
      const rtError_t rt_ret = rtDevBinaryUnRegister(info.handle());
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_INNER_ERROR("E19999", "Call rtDevBinaryUnRegister failed for Kernel:%s fail, ret:0x%X",
                           item.first.c_str(), rt_ret);
        GELOGE(INTERNAL_ERROR, "[Call][RtDevBinaryUnRegister] Kernel[%s] UnRegister handle fail:%u.",
               item.first.c_str(), rt_ret);
      }
      (void)kernels_.erase(it);
    }
  }
}

KernelHolder::KernelHolder(const char_t *const stub_func,
                           const std::shared_ptr<ge::OpKernelBin> kernel_bin)
    : stub_func_(stub_func), bin_handle_(nullptr), kernel_bin_(kernel_bin) {}

KernelHolder::~KernelHolder() {
  if (bin_handle_ != nullptr) {
    GE_CHK_RT(rtDevBinaryUnRegister(bin_handle_));
  }
}

HandleHolder::HandleHolder(void *const bin_handle)
    : bin_handle_(bin_handle) {}

HandleHolder::~HandleHolder() {
  if (bin_handle_ != nullptr) {
    GE_CHK_RT(rtDevBinaryUnRegister(bin_handle_));
  }
}

const char_t *KernelBinRegistry::GetUnique(const std::string &stub_func) {
  const std::lock_guard<std::mutex> lock(mutex_);
  auto it = unique_stubs_.find(stub_func);
  if (it != unique_stubs_.end()) {
    return it->c_str();
  } else {
    it = unique_stubs_.insert(unique_stubs_.end(), stub_func);
    return it->c_str();
  }
}

const char_t *KernelBinRegistry::GetStubFunc(const std::string &stub_name) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const std::map<std::string, std::unique_ptr<KernelHolder>>::const_iterator iter = registered_bins_.find(stub_name);
  if (iter != registered_bins_.cend()) {
    return iter->second->stub_func_;
  }

  return nullptr;
}

bool KernelBinRegistry::AddKernel(const std::string &stub_name, std::unique_ptr<KernelHolder> &&holder) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto ret = registered_bins_.emplace(stub_name, std::move(holder));
  return ret.second;
}

bool HandleRegistry::AddHandle(std::unique_ptr<HandleHolder> &&holder) {
  const auto ret = registered_handles_.emplace(std::move(holder));
  return ret.second;
}
} // namespace ge
