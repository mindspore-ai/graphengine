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

#include "common/plugin/op_tiling_manager.h"

#include <string>

#include "common/plugin/plugin_manager.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/log.h"
#include "mmpa/mmpa_api.h"

namespace ge {
void OpTilingManager::ClearHandles() noexcept {
  for (const auto &handle : handles_) {
    if (mmDlclose(handle.second) != 0) {
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGE(FAILED, "[Close][Handle]Failed, handle of %s: %s", handle.first.c_str(), error);
      REPORT_CALL_ERROR("E19999", "Failed to close handle of %s: %s",
                        handle.first.c_str(), error);
    }
  }
  handles_.clear();
}

OpTilingManager::~OpTilingManager() { ClearHandles(); }

void OpTilingManager::LoadSo() {
  std::string op_tiling_path;
  const Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get op tiling path!");
    return;
  }

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  std::vector<std::string> path_vec;
  PluginManager::SplitPath(op_tiling_path, path_vec);
  for (const auto &path : path_vec) {
    std::string root_path = path + "op_master/";
    std::string so_name = "libopmaster.so";
    char_t resolved_path[MMPA_MAX_PATH] = {};
    const INT32 result = mmRealPath(root_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH);
    if (result != EN_OK) {
      GELOGW("[FindSo][Check] Get path with op_master [%s] failed", root_path.c_str());
      root_path = path + "op_tiling/";
      so_name = "liboptiling.so";
    }
    std::string so_path = root_path + "lib/" + os_type + "/" + cpu_type + "/" + so_name;
    void *handle = mmDlopen(so_path.c_str(),
                            static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                            static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
    if (handle == nullptr) {
      GELOGW("Failed to dlopen %s! errmsg:%s", so_path.c_str(), mmDlerror());
      so_path = root_path + so_name;
      handle = mmDlopen(so_path.c_str(),
                        static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                        static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
      if (handle == nullptr) {
        GELOGW("Failed to dlopen %s! errmsg:%s", so_path.c_str(), mmDlerror());
      } else {
        GELOGI("dlopen file %s successfully!", so_path.c_str());
        handles_[so_path] = handle;
      }
    } else {
      GELOGI("dlopen file %s successfully!", so_path.c_str());
      handles_[so_path] = handle;
    }
  }
}

OpTilingManager &OpTilingManager::GetInstance() {
  static OpTilingManager instance;
  return instance;
}
}  // namespace ge
