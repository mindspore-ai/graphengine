/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef GE_HYBRID_COMMON_TENSOR_VALUE_H_
#define GE_HYBRID_COMMON_TENSOR_VALUE_H_

#include <atomic>
#include <cstddef>
#include <memory>

namespace ge {
namespace hybrid {
class NpuMemoryAllocator;

class TensorBuffer {
 public:
  static std::unique_ptr<TensorBuffer> Create(NpuMemoryAllocator *allocator, size_t size);

  static std::unique_ptr<TensorBuffer> Create(void *buffer, size_t size);

  ~TensorBuffer();

  void *GetData() { return buffer_; }

  size_t GetSize() const { return size_; }

 private:
  TensorBuffer(NpuMemoryAllocator *allocator, void *buffer, size_t size);

  NpuMemoryAllocator *allocator_ = nullptr;
  void *buffer_ = nullptr;
  size_t size_ = 0;
};

class TensorValue {
 public:
  TensorValue() = default;

  explicit TensorValue(std::shared_ptr<TensorBuffer> buffer);

  TensorValue(void *buffer, size_t size);

  ~TensorValue();

  void Destroy();

  bool IsEmpty() { return ref_buffer_ == nullptr && buffer_ == nullptr; }

  const void *GetData() const;

  std::string DebugString() const;

  void SetName(const std::string &name) { name_ = name; }

  void *MutableData();

  size_t GetSize() const;

 private:
  std::shared_ptr<TensorBuffer> buffer_;
  std::string name_;
  // for weights and variables
  void *ref_buffer_ = nullptr;
  size_t ref_size_ = 0;
  // shape
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_COMMON_TENSOR_VALUE_H_
