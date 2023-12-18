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

#include "runtime/stream.h"

#include <list>
#include <vector>

#include "common/ge_visibility.h"
#include "framework/common/ge_inner_error_codes.h"

namespace gert {
class VISIBILITY_EXPORT EventAllocator {
 public:
  explicit EventAllocator(uint32_t flag = RT_EVENT_DDSYNC_NS)
      : default_flag_(flag) {}
  EventAllocator(const EventAllocator &) = delete;
  EventAllocator &operator=(const EventAllocator &) = delete;
  ~EventAllocator();

  ge::Status AcquireEvents(size_t event_num, std::vector<rtEvent_t> &rt_events);
  ge::Status ReleaseEvents(const std::vector<rtEvent_t> &rt_events);
  ge::Status Finalize();
  ge::Status GetAvailEventsNum(size_t &event_num) const;

 private:
  std::list<rtEvent_t> event_pool_;
  size_t events_created_total_ = 0U;
  uint32_t default_flag_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_EVENT_ALLOCATOR_H_