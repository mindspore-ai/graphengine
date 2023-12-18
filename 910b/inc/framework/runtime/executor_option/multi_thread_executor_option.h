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

#ifndef AIR_CXX_MULTI_THREAD_EXECUTOR_OPTION_H
#define AIR_CXX_MULTI_THREAD_EXECUTOR_OPTION_H

#include <cstddef>
#include "framework/runtime/executor_option/executor_option.h"

namespace gert {
constexpr size_t kLeastCoreNumber = 3U;    // least core num, one for schedule, two for workers
constexpr size_t kLeastThreadNumber = 2U;  // least new thread num, one for normal worker, one for memory worker

class VISIBILITY_EXPORT MultiThreadExecutorOption : public ExecutorOption {
 public:
  MultiThreadExecutorOption() : MultiThreadExecutorOption(3U) {}
  explicit MultiThreadExecutorOption(size_t thread_num)
      : ExecutorOption(ExecutorType::kTopologicalMultiThread), thread_num_(thread_num) {}
  MultiThreadExecutorOption(ExecutorType executor_type, size_t thread_num)
      : ExecutorOption(executor_type), thread_num_(thread_num) {}

  size_t GetThreadNum() const {
    return thread_num_;
  }

 private:
  /**
   * 多线程数量
   * 取值范围：2 <= thread_num
   */
  size_t thread_num_;
};
}  // namespace gert
#endif  // AIR_CXX_MULTI_THREAD_EXECUTOR_OPTION_H
