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

#include "common/tbe_handle_store/kernel_store.h"

namespace {
const uint32_t kKernelItemMagic = 0x5d776efdU;
}

namespace ge {
void KernelStore::AddKernel(const KernelBinPtr &kernel) {
  if (kernel != nullptr) {
    kernels_[kernel->GetName()] = kernel;
  }
}

bool KernelStore::Build() {
  buffer_.clear();
  size_t total_len = 0U;
  for (const auto &item : kernels_) {
    const auto kernel = item.second;
    total_len += sizeof(KernelStoreItemHead);
    total_len += kernel->GetName().length();
    total_len += kernel->GetBinDataSize();
  }

  try {
    buffer_.resize(total_len);
  } catch (std::bad_alloc &e) {
    GELOGE(ge::MEMALLOC_FAILED, "All build memory failed, memory size %zu", total_len);
    GELOGE(ge::MEMALLOC_FAILED, "[Malloc][Memmory]Resize buffer failed, memory size %zu, "
           "exception %s", total_len, e.what());
    REPORT_CALL_ERROR("E19999", "Resize buffer failed, memory size %zu, exception %s",
                      total_len, e.what());
    return false;
  }

  uint8_t *next_buffer = buffer_.data();
  size_t remain_len = total_len;
  for (const auto &item : kernels_) {
    const auto kernel = item.second;
    KernelStoreItemHead kernel_head{};
    kernel_head.magic = kKernelItemMagic;
    kernel_head.name_len = static_cast<uint32_t>(kernel->GetName().length());
    kernel_head.bin_len = static_cast<uint32_t>(kernel->GetBinDataSize());

    GELOGD("get kernel bin name %s, addr %p, size %zu",
           kernel->GetName().c_str(), kernel->GetBinData(), kernel->GetBinDataSize());
    errno_t mem_ret = memcpy_s(next_buffer, remain_len, &kernel_head, sizeof(kernel_head));
    if (mem_ret != EOK) {
      return false;
    }
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(kernel_head)));

    mem_ret = memcpy_s(next_buffer,
                       remain_len - sizeof(kernel_head),
                       kernel->GetName().data(),
                       static_cast<size_t>(kernel_head.name_len));
    if (mem_ret != EOK) {
      return false;
    }
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(
        PtrToValue(next_buffer) + static_cast<size_t>(kernel_head.name_len)));

    mem_ret = memcpy_s(next_buffer, remain_len - sizeof(kernel_head) - kernel_head.name_len, kernel->GetBinData(),
                       static_cast<size_t>(kernel_head.bin_len));
    if (mem_ret != EOK) {
      return false;
    }

    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(
        PtrToValue(next_buffer) + static_cast<size_t>(kernel_head.bin_len)));
    remain_len = remain_len - sizeof(kernel_head) - kernel_head.name_len - kernel_head.bin_len;
  }
  kernels_.clear();
  return true;
}

const uint8_t *KernelStore::Data() const { return buffer_.data(); }

size_t KernelStore::DataSize() const { return buffer_.size(); }

bool KernelStore::Load(const uint8_t *const data, const size_t &len) {
  if ((data == nullptr) || (len == 0U)) {
    return false;
  }
  size_t buffer_len = len;
  while (buffer_len > sizeof(KernelStoreItemHead)) {
    const uint8_t *next_buffer =
      PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(len - buffer_len)));

    const auto *const kernel_head = PtrToPtr<const uint8_t, const KernelStoreItemHead>(next_buffer);
    if (buffer_len < (static_cast<size_t>(kernel_head->name_len) + static_cast<size_t>(kernel_head->bin_len) +
        sizeof(KernelStoreItemHead))) {
      GELOGW("Invalid kernel block remain buffer len %zu, name len %u, bin len %u", buffer_len, kernel_head->name_len,
             kernel_head->bin_len);
      break;
    }

    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(KernelStoreItemHead)));
    std::string name(PtrToPtr<const uint8_t, const char_t>(next_buffer), static_cast<size_t>(kernel_head->name_len));

    next_buffer =
      PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + static_cast<size_t>(kernel_head->name_len)));
    GELOGD("Load kernel from om:%s,%u,%u", name.c_str(), kernel_head->name_len, kernel_head->bin_len);
    std::vector<char> kernel_bin(next_buffer,
                                 PtrToPtr<void, const uint8_t>(ValueToPtr(
                                     PtrToValue(next_buffer) + static_cast<size_t>(kernel_head->bin_len))));
    KernelBinPtr teb_kernel_ptr = ge::MakeShared<KernelBin>(name, std::move(kernel_bin));
    if (teb_kernel_ptr != nullptr) {
      (void) kernels_.emplace(name, teb_kernel_ptr);
    }
    buffer_len -= sizeof(KernelStoreItemHead) + kernel_head->name_len + kernel_head->bin_len;
  }

  return true;
}

KernelBinPtr KernelStore::FindKernel(const std::string &name) const {
  const auto it = kernels_.find(name);
  if (it != kernels_.end()) {
    return it->second;
  }
  return nullptr;
}

bool KernelStore::IsEmpty() const {
  return kernels_.empty() && buffer_.empty();
}
}  // namespace ge
