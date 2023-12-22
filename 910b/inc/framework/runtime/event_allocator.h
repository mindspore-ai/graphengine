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

#ifndef AIR_CXX_RUNTIME_EVENT_ALLOCATOR_H_
#define AIR_CXX_RUNTIME_EVENT_ALLOCATOR_H_

#include <cstdlib>
#include "runtime/event.h"
#include "common/checker.h"
#include "common/ge_visibility.h"
#include "framework/common/ge_inner_error_codes.h"
#include "exe_graph/runtime/continuous_vector.h"

namespace gert {
class VISIBILITY_EXPORT EventAllocator {
 public:
  static constexpr size_t kMaxEventNum = 4096U;
  explicit EventAllocator(uint32_t flag = RT_EVENT_DDSYNC_NS)
      : events_holder_(ContinuousVector::Create<rtEvent_t>(kMaxEventNum)), default_flag_(flag) {}
  EventAllocator(const EventAllocator &) = delete;
  EventAllocator &operator=(const EventAllocator &) = delete;
  ~EventAllocator();

  TypedContinuousVector<rtEvent_t> *AcquireEvents(size_t event_num);

 private:
  TypedContinuousVector<rtEvent_t> *Events();

 private:
  std::unique_ptr<uint8_t[]> events_holder_;
  uint32_t default_flag_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_EVENT_ALLOCATOR_H_