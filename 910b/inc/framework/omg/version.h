/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef INC_FRAMEWORK_OMG_VERSION_H_
#define INC_FRAMEWORK_OMG_VERSION_H_

#include <string>
#include "ge/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
class GE_FUNC_VISIBILITY PlatformVersionManager {
 public:
  PlatformVersionManager() = delete;
  ~PlatformVersionManager() = delete;
  static Status GetPlatformVersion(std::string &ver) {
    ver = "1.11.z";  // format x.y.z
    GELOGI("current platform version: %s.", ver.c_str());
    return SUCCESS;
  }
};  // class PlatformVersionManager
}  // namespace ge

#endif  // INC_FRAMEWORK_OMG_VERSION_H_
