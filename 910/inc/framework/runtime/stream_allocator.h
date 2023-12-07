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

#include "runtime/stream.h"

#include <list>
#include <vector>

#include "common/ge_visibility.h"
#include "framework/common/ge_inner_error_codes.h"

namespace gert {
class VISIBILITY_EXPORT StreamAllocator {
 public:
  explicit StreamAllocator(int32_t priority = RT_STREAM_PRIORITY_DEFAULT, uint32_t flags = RT_STREAM_DEFAULT)
      : default_priority_(priority), default_flags_(flags) {}
  StreamAllocator(const StreamAllocator &) = delete;
  StreamAllocator &operator=(const StreamAllocator &) = delete;
  ~StreamAllocator();

  ge::Status AcquireStreams(size_t stream_num, std::vector<rtStream_t> &rt_streams);
  ge::Status ReleaseStreams(const std::vector<rtStream_t> &rt_streams);
  ge::Status Finalize();
  ge::Status GetAvailStreamsNum(size_t &stream_num) const;

 private:
  std::list<rtStream_t> stream_pool_;
  size_t streams_created_total_ = 0U;
  int32_t default_priority_;
  uint32_t default_flags_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_STREAM_ALLOCATOR_H_