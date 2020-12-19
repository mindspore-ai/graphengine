/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef INC_GRAPH_GE_CONTEXT_H_
#define INC_GRAPH_GE_CONTEXT_H_

#include <string>
#include "graph/ge_error_codes.h"

namespace ge {
class GEContext {
 public:
  graphStatus GetOption(const std::string &key, std::string &option);
  bool GetHostExecFlag();
  uint64_t SessionId();
  uint32_t DeviceId();
  uint64_t TraceId();
  void Init();
  void SetSessionId(uint64_t session_id);
  void SetCtxDeviceId(uint32_t device_id);
 private:
  thread_local static uint64_t session_id_;
  uint32_t device_id_ = 0;
  uint64_t trace_id_ = 0;
};  // class GEContext

/// Get context
/// @return
GEContext &GetContext();
}  // namespace ge

#endif  //  INC_GRAPH_GE_CONTEXT_H_
