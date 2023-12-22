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

#ifndef AIR_CXX_RUNTIME_STREAM_ALLOCATOR_H_
#define AIR_CXX_RUNTIME_STREAM_ALLOCATOR_H_

#include <memory>
#include <cstdlib>
#include "runtime/stream.h"
#include "common/checker.h"
#include "common/ge_visibility.h"
#include "framework/common/ge_inner_error_codes.h"
#include "exe_graph/runtime/continuous_vector.h"
namespace gert {
class VISIBILITY_EXPORT StreamAllocator {
 public:
  static constexpr size_t kMaxStreamNum = 2024U;
  explicit StreamAllocator(int32_t priority = RT_STREAM_PRIORITY_DEFAULT, uint32_t flags = RT_STREAM_DEFAULT);
  StreamAllocator(const StreamAllocator &) = delete;
  StreamAllocator &operator=(const StreamAllocator &) = delete;

  ~StreamAllocator();

  TypedContinuousVector<rtStream_t> *AcquireStreams(size_t stream_num);

 private:
  TypedContinuousVector<rtStream_t> *Streams();

 private:
  std::unique_ptr<uint8_t[]> streams_holder_;
  int32_t default_priority_;
  uint32_t default_flags_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_STREAM_ALLOCATOR_H_