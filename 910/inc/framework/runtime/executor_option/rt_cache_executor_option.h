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

#ifndef AIR_CXX_RT_CACHE_EXECUTOR_OPTION_H
#define AIR_CXX_RT_CACHE_EXECUTOR_OPTION_H

#include "framework/runtime/executor_option/executor_option.h"

namespace gert {
enum class RtCacheMode {
  /**
   * 关闭
   */
  kTurnOff,

  /**
   * 开启HostCache。
   * 若开启此选项，按照输入shape来缓存执行数据，省去部分只依赖输入shape的执行节点（infershape/tiling/malloc等),
   * 节约host调度时间
   */
  kHostCache,

  /**
   * 开启DeviceCache。todo, 待支持以后补充描述。
   */
  kDeviceCache,

  kEnd
};
class VISIBILITY_EXPORT RtCacheExecutorOption : public ExecutorOption {
 public:
  RtCacheExecutorOption() : ExecutorOption(ExecutorType::kHostCache), rt_cache_mode_(RtCacheMode::kTurnOff) {}
  explicit RtCacheExecutorOption(RtCacheMode rt_cache_mode)
      : ExecutorOption(ExecutorType::kHostCache), rt_cache_mode_(rt_cache_mode) {}
  const RtCacheMode &GetCacheMode() const {
    return rt_cache_mode_;
  }

 private:
  /**
   * 动态shape运行时Cache的模式。
   * 启用该模式，会通过缓存运行时数据的方式，提升host/device调度性能。
   */
  RtCacheMode rt_cache_mode_;
};
}  // namespace gert

#endif  // AIR_CXX_RT_CACHE_EXECUTOR_OPTION_H
