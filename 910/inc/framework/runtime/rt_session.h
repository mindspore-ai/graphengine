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
#ifndef AIR_RUNTIME_SESSION_H
#define AIR_RUNTIME_SESSION_H
#include "rt_var_manager.h"
namespace gert {
class RtSession {
 public:
  RtSession() = default;
  explicit RtSession(uint64_t session_id) : session_id_(session_id) {}
  uint64_t GetSessionId() const {
    return session_id_;
  }
  void SetSessionId(uint64_t session_id) {
    session_id_ = session_id;
  }
  const RtVarManager *GetVarManager() const {
    return var_manager_;
  }
  void SetVarManager(RtVarManager *var_manager) {
    var_manager_ = var_manager;
  }

 private:
  uint64_t session_id_{0};
  RtVarManager *var_manager_{nullptr};
};
}
#endif  // AIR_RUNTIME_SESSION_H
