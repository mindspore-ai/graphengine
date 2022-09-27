/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#ifndef AIR_CXX_BLOCK_H
#define AIR_CXX_BLOCK_H
#include "exe_graph/runtime/tensor_data.h"

namespace gert {
namespace memory {
class Block {
 public:
  // todo use TensorAddress
  virtual const void *Addr() const = 0;
  virtual void *MutableAddr() = 0;
  virtual size_t GetSize() const = 0;
  virtual size_t AddCount() = 0;
  virtual size_t SubCount() = 0;
  virtual size_t GetCount() const = 0;
  virtual TensorData ToTensorData() = 0;
  virtual void Free() = 0;
  virtual ~Block() = default;
};
}  // namespace memory
}  // namespace gert
#endif  // AIR_CXX_BLOCK_H
