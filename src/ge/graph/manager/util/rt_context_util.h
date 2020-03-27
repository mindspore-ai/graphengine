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

#ifndef GE_GRAPH_MANAGER_UTIL_RT_CONTEXT_UTIL_H_
#define GE_GRAPH_MANAGER_UTIL_RT_CONTEXT_UTIL_H_

#include <vector>

#include "runtime/context.h"

namespace ge {
class RtContextUtil {
 public:
  static RtContextUtil &GetInstance() {
    static RtContextUtil instance;
    return instance;
  }

  void AddrtContext(rtContext_t context);

  void DestroyrtContexts();

  RtContextUtil &operator=(const RtContextUtil &) = delete;
  RtContextUtil(const RtContextUtil &RtContextUtil) = delete;

 private:
  RtContextUtil() = default;
  ~RtContextUtil() {}

  std::vector<rtContext_t> rtContexts_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_UTIL_RT_CONTEXT_UTIL_H_

