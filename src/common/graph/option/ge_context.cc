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
#include "framework/common/debug/ge_log.h"

namespace ge {
GEContext &GetContext() {
  static GEContext ge_context{};
  return ge_context;
}

graphStatus GEContext::GetOption(const std::string &key, std::string &option) {
  return GetThreadLocalContext().GetOption(key, option);
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
  try {
    job_id_ = static_cast<uint64_t>(std::stoi(job_id.c_str()));
  } catch (std::invalid_argument &) {
    GELOGW("%s transform to int failed.", job_id.c_str());
  } catch (std::out_of_range &) {
    GELOGW("%s transform to int failed.", job_id.c_str());
  }
}

uint64_t GEContext::SessionId() { return session_id_; }

uint32_t GEContext::DeviceId() { return device_id_; }

uint64_t GEContext::JobId() { return job_id_; }

void GEContext::SetCtxDeviceId(uint32_t device_id) { device_id_ = device_id; }
}  // namespace ge
