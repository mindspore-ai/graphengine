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

#ifndef RUNTIME_V2_KERNEL_MEMORY_UTIL_DEVICE_MEMORY_STATS_H
#define RUNTIME_V2_KERNEL_MEMORY_UTIL_DEVICE_MEMORY_STATS_H
#include <atomic>
#include <queue>
#include <mutex>
#include "toolchain/prof_common.h"
#include "toolchain/prof_api.h"

namespace gert {
  struct MemoryRecorder {
    int64_t size;
    uint64_t addr;
    uint64_t total_allocate_memory;
    uint64_t total_reserve_memory;
    uint64_t time_stamp;
  };

  class DeviceMemoryRecorder {
   public:
    static uint64_t GetTotalAllocateMemory() { return total_allocate_memory_.load(); }
    static uint64_t GetTotalReserveMemory() { return total_reserve_memory_.load(); }
    static void AddTotalReserveMemory(const uint64_t &num) { total_reserve_memory_ += num; }
    static void ReduceTotalReserveMemory(const uint64_t &num) { total_reserve_memory_ -= num; }
    static void AddTotalAllocateMemory(const uint64_t &num) { total_allocate_memory_ += num; }
    static void ReduceTotalAllocateMemory(const uint64_t &num) { total_allocate_memory_ -= num; }
    static void ClearReserveMemory() { total_reserve_memory_.store(0UL); }
    static void SetRecorder(const void *const addr, const int64_t size);
    static const MemoryRecorder GetRecorder();
    static bool IsRecorderEmpty();
   private:
    static std::atomic<uint64_t> total_allocate_memory_;
    static std::atomic<uint64_t> total_reserve_memory_;
    static std::queue<MemoryRecorder> memory_record_queue_;
    static std::mutex mtx_;
  };
}
#endif