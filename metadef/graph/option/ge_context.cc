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

#include "./ge_context.h"
#include "./ge_global_options.h"
#include "./ge_local_context.h"
#include "framework/common/ge_types.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace {
const int64_t kMinTrainingTraceJobId = 256;
const int kDecimal = 10;
const char *kHostExecPlacement = "HOST";
}  // namespace
GEContext &GetContext() {
  static GEContext ge_context{};
  return ge_context;
}

graphStatus GEContext::GetOption(const std::string &key, std::string &option) {
  return GetThreadLocalContext().GetOption(key, option);
}

bool GEContext::GetHostExecFlag() {
  std::string exec_placement;
  if (GetThreadLocalContext().GetOption(GE_OPTION_EXEC_PLACEMENT, exec_placement) != GRAPH_SUCCESS) {
    GELOGW("get option OPTION_EXEC_PLACEMENT failed.");
    return false;
  }
  GELOGD("Option ge.exec.placement is %s.", exec_placement.c_str());
  return exec_placement == kHostExecPlacement;
}

std::map<std::string, std::string> &GetMutableGlobalOptions() {
  static std::map<std::string, std::string> global_options{};
  return global_options;
}

void GEContext::Init() {
  string session_id;
  (void)GetOption("ge.exec.sessionId", session_id);
  try {
    session_id_ = static_cast<uint64_t>(std::stoi(session_id.c_str()));
  } catch (std::invalid_argument &) {
    GELOGW("%s transform to int failed.", session_id.c_str());
  } catch (std::out_of_range &) {
    GELOGW("%s transform to int failed.", session_id.c_str());
  }

  string device_id;
  (void)GetOption("ge.exec.deviceId", device_id);
  try {
    device_id_ = static_cast<uint32_t>(std::stoi(device_id.c_str()));
  } catch (std::invalid_argument &) {
    GELOGW("%s transform to int failed.", device_id.c_str());
  } catch (std::out_of_range &) {
    GELOGW("%s transform to int failed.", device_id.c_str());
  }

  string job_id;
  (void)GetOption("ge.exec.jobId", job_id);
  std::string s_job_id = "";
  for (auto c : job_id) {
    if (c >= '0' && c <= '9') {
      s_job_id += c;
    }
  }
  if (s_job_id == "") {
    trace_id_ = kMinTrainingTraceJobId;
    return;
  }
  int64_t d_job_id = std::strtoll(s_job_id.c_str(), nullptr, kDecimal);
  if (d_job_id < kMinTrainingTraceJobId) {
    trace_id_ = d_job_id + kMinTrainingTraceJobId;
  } else {
    trace_id_ = d_job_id;
  }
}

uint64_t GEContext::SessionId() { return session_id_; }

uint32_t GEContext::DeviceId() { return device_id_; }

uint64_t GEContext::TraceId() { return trace_id_; }

void GEContext::SetSessionId(uint64_t session_id) { session_id_ = session_id; }

void GEContext::SetCtxDeviceId(uint32_t device_id) { device_id_ = device_id; }

}  // namespace ge
