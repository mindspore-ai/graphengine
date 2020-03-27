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

#ifndef GE_SINGLE_OP_STREAM_RESOURCE_H_
#define GE_SINGLE_OP_STREAM_RESOURCE_H_

#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>

#include "common/ge_inner_error_codes.h"
#include "runtime/stream.h"
#include "single_op/single_op.h"

namespace ge {
class StreamResource {
 public:
  StreamResource() = default;
  ~StreamResource();

  StreamResource(const StreamResource &) = delete;
  StreamResource(StreamResource &&) = delete;
  StreamResource &operator=(const StreamResource &) = delete;
  StreamResource &operator=(StreamResource &&) = delete;

  void CacheOperator(const void *key, SingleOp *single_op);

  SingleOp *GetOperator(const void *key);

  uint8_t *MallocMemory(size_t size);
  uint8_t *MallocWeight(size_t size);

 private:
  static uint8_t *DoMallocMemory(size_t size, size_t &max_allocated, std::vector<uint8_t *> &allocated);

  size_t max_memory_size_ = 0;
  size_t max_weight_size_ = 0;
  std::vector<uint8_t *> memory_list_;
  std::vector<uint8_t *> weight_list_;
  std::unordered_map<const void *, SingleOp *> op_map_;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_STREAM_RESOURCE_H_
