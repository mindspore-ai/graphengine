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

#ifndef INC_FRAMEWORK_OMG_VERSION_H_
#define INC_FRAMEWORK_OMG_VERSION_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/debug/log.h"
#include "common/string_util.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
class PlatformVersionManager {
 public:
  PlatformVersionManager() = delete;
  ~PlatformVersionManager() = delete;
  static Status GetPlatformVersion(std::string &ver) {
#if defined PLATFORM_PHOENIX
    ver = "3.51.z";
#elif defined PLATFORM_ORLANDO
    ver = "3.31.z";
#elif defined PLATFORM_MINI
    ver = "1.11.z";
#elif defined PLATFORM_CLOUD
    ver = "1.61.z";
#endif
    std::vector<std::string> version_splits = StringUtils::Split(ver, '.');
    GE_IF_BOOL_EXEC(version_splits.size() < 3, GELOGW("Read platform version error!"); return FAILED;);

    GELOGI("Read current platform version: %s.", ver.c_str());
    return SUCCESS;
  }
};  // class PlatformManager
}  // namespace ge

#endif  // INC_FRAMEWORK_OMG_VERSION_H_
