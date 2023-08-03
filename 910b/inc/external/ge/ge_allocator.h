/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef AIR_CXX_INC_EXTERNAL_GE_ALLOCATOR_H_
#define AIR_CXX_INC_EXTERNAL_GE_ALLOCATOR_H_
#include <cstdlib>
#include <memory>
namespace ge {
class MemBlock;
class Allocator {
 public:
  Allocator() = default;
  virtual ~Allocator() = default;
  Allocator(const Allocator &) = delete;
  Allocator &operator=(const Allocator &) = delete;

  virtual MemBlock *Malloc(size_t size) = 0;
  virtual void Free(MemBlock *block) = 0;

  // Apply for suggested address memory, default ignore suggested address
  virtual MemBlock *MallocAdvise(size_t size, void *addr) {
    (void)addr;
    return Malloc(size);
  }
};

class MemBlock {
 public:
  MemBlock(Allocator &allocator, void *addr, size_t block_size)
      : allocator_(allocator), addr_(addr), count_(1U), block_size_(block_size) {}
  virtual ~MemBlock() = default;
  const void *GetAddr() const {
    return addr_;
  }
  void *GetAddr() {
    return addr_;
  }
  size_t GetSize() const {
    return block_size_;
  }
  void SetSize(const size_t mem_size) {
    block_size_ = mem_size;
  }
  void Free() {
    if (GetCount() > 0U) {
      if (SubCount() == 0U) {
        return allocator_.Free(this);
      }
    }
  }

  size_t AddCount() {
    return ++count_;
  }
  size_t SubCount() {
    return --count_;
  }
  size_t GetCount() const {
    return count_;
  }
 private:
  Allocator &allocator_;
  void *addr_;
  size_t count_;
  size_t block_size_;
};

using AllocatorPtr = std::shared_ptr<Allocator>;
}  // namespace ge
#endif  // AIR_CXX_INC_EXTERNAL_GE_ALLOCATOR_H_
