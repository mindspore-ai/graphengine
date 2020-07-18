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
#include <map>
#include <mutex>

#include "runtime/context.h"

namespace ge {
class RtContextUtil {
 public:
  static RtContextUtil &GetInstance() {
    static RtContextUtil instance;
    return instance;
  }

  void AddRtContext(uint64_t session_id, rtContext_t context);

  const rtContext_t GetNormalModeContext() const { return before_prerun_ctx_; }

  void SetNormalModeContext(rtContext_t context) { before_prerun_ctx_ = context; }

  void DestroyRtContexts(uint64_t session_id);
  void DestroyAllRtContexts();

  RtContextUtil &operator=(const RtContextUtil &) = delete;
  RtContextUtil(const RtContextUtil &RtContextUtil) = delete;

 private:
  RtContextUtil() = default;
  ~RtContextUtil() {}

  void DestroyRtContexts(uint64_t session_id, std::vector<rtContext_t> &contexts);

  std::map<uint64_t, std::vector<rtContext_t>> rt_contexts_;
  rtContext_t before_prerun_ctx_ = nullptr;

  std::mutex ctx_mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_UTIL_RT_CONTEXT_UTIL_H_
