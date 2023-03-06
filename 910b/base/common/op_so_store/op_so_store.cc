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

#include "op_so_store.h"
#include "common/checker.h"

namespace {
const uint32_t kKernelItemMagic = 0x5d776efdU;
}

namespace ge {
void OpSoStore::AddKernel(const OpSoBinPtr &so_bin_ptr) {
  if (so_bin_ptr != nullptr) {
    kernels_.push_back(so_bin_ptr);
  }

  return;
}

std::vector<OpSoBinPtr> OpSoStore::GetSoBin() const { return kernels_; }

const uint8_t *OpSoStore::Data() const { return buffer_.get(); }

size_t OpSoStore::DataSize() const { return static_cast<size_t>(buffer_size_); }

uint32_t OpSoStore::GetKernelNum() const { return static_cast<uint32_t>(kernels_.size()); };

bool OpSoStore::CalculateAndAllocMem() {
  size_t total_len = sizeof(SoStoreHead);
  for (const auto &item : kernels_) {
    total_len += sizeof(SoStoreItemHead);
    total_len += item->GetSoName().length();
    total_len += item->GetVendorName().length();
    total_len += item->GetBinDataSize();
  }
  buffer_size_ = static_cast<uint32_t>(total_len);
  buffer_ = std::shared_ptr<uint8_t> (new (std::nothrow) uint8_t[total_len],
      std::default_delete<uint8_t[]>());
  GE_ASSERT_NOTNULL(buffer_, "[Malloc][Memmory]Resize buffer failed, memory size %zu", total_len);
  return true;
}

bool OpSoStore::Build() {
  GE_ASSERT_TRUE(CalculateAndAllocMem());

  size_t remain_len = buffer_size_;

  auto next_buffer = buffer_.get();
  SoStoreHead so_store_head;
  so_store_head.so_num = static_cast<uint32_t>(kernels_.size());
  GE_ASSERT_EOK(memcpy_s(next_buffer, remain_len, &so_store_head, sizeof(SoStoreHead)));
  remain_len -= sizeof(SoStoreHead);
  next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(SoStoreHead)));
  for (const auto &item : kernels_) {
    SoStoreItemHead so_bin_head{};
    so_bin_head.magic = kKernelItemMagic;
    so_bin_head.so_name_len = static_cast<uint32_t>(item->GetSoName().length());
    so_bin_head.vendor_name_len = static_cast<uint32_t>(item->GetVendorName().length());
    so_bin_head.bin_len = static_cast<uint32_t>(item->GetBinDataSize());

    GELOGD("get so name %s, vendor name:%s, size %zu",
           item->GetSoName().c_str(), item->GetVendorName().c_str(), item->GetBinDataSize());

    GE_ASSERT_EOK(memcpy_s(next_buffer, remain_len, &so_bin_head, sizeof(SoStoreItemHead)));
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(SoStoreItemHead)));

    GE_ASSERT_EOK(memcpy_s(next_buffer, remain_len - sizeof(SoStoreItemHead),
        item->GetSoName().data(), static_cast<size_t>(so_bin_head.so_name_len)));
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + so_bin_head.so_name_len));

    GE_ASSERT_EOK(memcpy_s(next_buffer, remain_len - sizeof(SoStoreItemHead) - so_bin_head.so_name_len,
        item->GetVendorName().data(), static_cast<size_t>(so_bin_head.vendor_name_len)));
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + so_bin_head.vendor_name_len));

    GE_ASSERT_EOK(memcpy_s(next_buffer,
        remain_len - sizeof(SoStoreItemHead) - so_bin_head.so_name_len - so_bin_head.vendor_name_len,
        item->GetBinData(), static_cast<size_t>(so_bin_head.bin_len)));
    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) +
        static_cast<size_t>(so_bin_head.bin_len)));

    remain_len = remain_len - sizeof(SoStoreItemHead) - so_bin_head.so_name_len -
        so_bin_head.vendor_name_len - so_bin_head.bin_len;
  }

  return true;
}

bool OpSoStore::Load(const uint8_t *const data, const size_t &len) {
  if ((data == nullptr) || (len == 0U)) {
    return false;
  }

  size_t buffer_len = len;
  if (buffer_len > sizeof(SoStoreHead)) {
    SoStoreHead so_store_head;
    GE_ASSERT_EOK(memcpy_s(&so_store_head, sizeof(SoStoreHead), data, sizeof(SoStoreHead)));
    so_num_ = so_store_head.so_num;
    buffer_len -= sizeof(SoStoreHead);
  }
  while (buffer_len > sizeof(SoStoreItemHead)) {
    const uint8_t *next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(data) + (len - buffer_len)));

    const auto *const so_bin_head = PtrToPtr<const uint8_t, const SoStoreItemHead>(next_buffer);
    if (buffer_len < (static_cast<size_t>(so_bin_head->so_name_len) + static_cast<size_t>(so_bin_head->bin_len) +
        static_cast<size_t>(so_bin_head->vendor_name_len) + sizeof(SoStoreItemHead))) {
      GELOGW("Invalid so block remain buffer len %zu, so name len %u, vendor name len:%u, bin len %u",
          buffer_len, so_bin_head->so_name_len, so_bin_head->vendor_name_len, so_bin_head->bin_len);
      break;
    }

    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + sizeof(SoStoreItemHead)));
    std::string so_name(PtrToPtr<const uint8_t, const char_t>(next_buffer),
        static_cast<size_t>(so_bin_head->so_name_len));

    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + so_bin_head->so_name_len));
    std::string vendor_name(PtrToPtr<const uint8_t, const char_t>(next_buffer),
        static_cast<size_t>(so_bin_head->vendor_name_len));

    next_buffer = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(next_buffer) + so_bin_head->vendor_name_len));
    GELOGD("Load vendor:%s so:%s from om, bin len:%u", vendor_name.c_str(), so_name.c_str(), so_bin_head->bin_len);

    std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new(std::nothrow) char[so_bin_head->bin_len]);
    GE_ASSERT_EOK(memcpy_s(so_bin.get(), static_cast<size_t>(so_bin_head->bin_len), next_buffer,
                           static_cast<size_t>(so_bin_head->bin_len)));

    const OpSoBinPtr so_bin_ptr
        = ge::MakeShared<OpSoBin>(so_name, vendor_name, std::move(so_bin), so_bin_head->bin_len);
    if (so_bin_ptr != nullptr) {
      (void) kernels_.push_back(so_bin_ptr);
    }
    buffer_len -= sizeof(SoStoreItemHead) + so_bin_head->so_name_len + so_bin_head->vendor_name_len +
        so_bin_head->bin_len;
  }
  GELOGD("read so num:%zu so bin num:%u from om", so_num_, kernels_.size());
  return true;
}
}  // namespace ge
