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

#include "graph/manager/util/rt_context_util.h"

#include "framework/common/debug/ge_log.h"

namespace ge {
void RtContextUtil::AddRtContext(uint64_t session_id, rtContext_t context) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  rt_contexts_[session_id].emplace_back(context);
}

void RtContextUtil::DestroyRtContexts(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  auto &contexts = rt_contexts_[session_id];
  DestroyRtContexts(session_id, contexts);
  auto iter = rt_contexts_.find(session_id);
  if (iter != rt_contexts_.end()) {
    rt_contexts_.erase(iter);
  }
}

void RtContextUtil::DestroyAllRtContexts() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  for (auto &ctx_pair : rt_contexts_) {
    DestroyRtContexts(ctx_pair.first, ctx_pair.second);
  }
  rt_contexts_.clear();
}

void RtContextUtil::DestroyRtContexts(uint64_t session_id, std::vector<rtContext_t> &contexts) {
  GELOGI("Runtime context handle number of session %lu is %zu.", session_id, contexts.size());
  for (auto &rtContext : contexts) {
    (void)rtCtxDestroy(rtContext);
  }
  contexts.clear();
}
}  // namespace ge
