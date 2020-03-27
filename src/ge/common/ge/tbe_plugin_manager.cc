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

#include "common/ge/tbe_plugin_manager.h"

#include <dirent.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "common/ge/ge_util.h"
#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/engine/dnnengine.h"
#include "framework/omg/omg_inner_types.h"
#include "external/ge/ge_api_types.h"
#include "register/op_registry.h"
#include "graph/opsproto_manager.h"

namespace ge {
// Get Singleton Instance
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY TBEPluginManager &TBEPluginManager::Instance() {
  static TBEPluginManager instance_ptr_;
  return instance_ptr_;
}

void TBEPluginManager::ClearHandles_() {
  for (const auto &handle : handles_vec_) {
    if (dlclose(handle) != 0) {
      GELOGW("Failed to close handle: %s", dlerror());
    }
  }
  handles_vec_.clear();
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void TBEPluginManager::Finalize() { ClearHandles_(); }

string TBEPluginManager::GetPath() {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(&TBEPluginManager::GetPath), &dl_info) == 0) {
    GELOGW("Failed to read so path!");
    return string();
  } else {
    string so_path = dl_info.dli_fname;
    char path[PATH_MAX] = {0};
    if (so_path.length() >= PATH_MAX) {
      GELOGW("File path is too long!");
      return string();
    }
    if (realpath(so_path.c_str(), path) == nullptr) {
      GELOGW("Failed to get realpath of %s", so_path.c_str());
      return string();
    }

    so_path = path;
    so_path = so_path.substr(0, so_path.rfind('/') + 1);
    return so_path;
  }
}

Status TBEPluginManager::CheckCustomAiCpuOpLib() {
  std::vector<std::string> vec_op_type;

  domi::OpRegistry::Instance()->GetOpTypeByImplyType(vec_op_type, domi::ImplyType::CUSTOM);
  for (size_t i = 0; i < vec_op_type.size(); i++) {
    bool aicpu_so_exist = false;
    std::string ai_cpu_so_name = "lib" + vec_op_type[i] + "_aicpu.so";
    for (size_t j = 0; j < domi::GetContext().aicpu_op_run_paths.size(); j++) {
      string bin_file_path = domi::GetContext().aicpu_op_run_paths[j];
      if (bin_file_path.size() >= ai_cpu_so_name.size() &&
          bin_file_path.compare(bin_file_path.size() - ai_cpu_so_name.size(), ai_cpu_so_name.size(), ai_cpu_so_name) ==
              0) {
        aicpu_so_exist = true;
        break;
      }
    }
    if (!aicpu_so_exist) {
      GELOGE(FAILED, "Can't find aicpu run so(%s), please check the plugin path!", ai_cpu_so_name.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

void TBEPluginManager::SaveDdkVersion(const std::string &ddk_version) {
  if (ddk_version.empty()) {
    return;
  }
  GELOGI("Input ddk version : %s.", ddk_version.c_str());

  // Save DDK version number to omgcontext
  domi::GetContext().ddk_version = ddk_version;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void TBEPluginManager::InitPreparation(
    const std::map<string, string> &options) {
  Status ret = CheckCustomAiCpuOpLib();
  if (ret != SUCCESS) {
    GELOGE(ret, "Check custom aicpu run so failed!");
    return;
  } else {
    auto ddk_version = options.find("ge.DDK_version");
    if (ddk_version != options.end()) {
      SaveDdkVersion(ddk_version->second);
    } else {
      GELOGW("No ddkVersion!");
      return;
    }
  }
}
}  // namespace ge
