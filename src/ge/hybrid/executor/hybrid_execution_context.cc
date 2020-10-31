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

#include "hybrid_execution_context.h"

namespace ge {
namespace hybrid {
void GraphExecutionContext::SetErrorCode(Status error_code) {
  std::lock_guard<std::mutex> lk(mu);
  this->status = error_code;
}

Status GraphExecutionContext::GetStatus() const {
  std::lock_guard<std::mutex> lk(mu);
  return this->status;
}
}  // namespace hybrid
}  // namespace ge