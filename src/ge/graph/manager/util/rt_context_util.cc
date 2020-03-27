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
void RtContextUtil::AddrtContext(rtContext_t context) { rtContexts_.emplace_back(context); }

void RtContextUtil::DestroyrtContexts() {
  GELOGI("The size of runtime context handle is %zu.", rtContexts_.size());
  for (auto &rtContext : rtContexts_) {
    (void)rtCtxDestroy(rtContext);
  }
  rtContexts_.clear();
}
}  // namespace ge
