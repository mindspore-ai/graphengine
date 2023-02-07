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
#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_TRACER_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_TRACER_H_
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_visibility.h"

namespace gert {
class VISIBILITY_EXPORT GlobalTracer {
 public:
  static GlobalTracer *GetInstance() {
    static GlobalTracer global_tracer;
    return &global_tracer;
  }
  uint64_t GetEnableFlags() const {
    return static_cast<uint64_t>(IsLogEnable(GE_MODULE_NAME, DLOG_INFO));
  };
 private:
   GlobalTracer() {};
   ~GlobalTracer() {};
   GlobalTracer(const GlobalTracer &) = delete;
   GlobalTracer(GlobalTracer &&) = delete;
   GlobalTracer &operator=(const GlobalTracer &) = delete;
   GlobalTracer &operator=(GlobalTracer &&) = delete;
};
}
#endif // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_TRACER_H_