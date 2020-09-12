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

#include "graph/opsproto_manager.h"
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_log.h"

namespace ge {
OpsProtoManager *OpsProtoManager::Instance() {
  static OpsProtoManager instance;
  return &instance;
}

bool OpsProtoManager::Initialize(const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (is_init_) {
    GELOGI("OpsProtoManager is already initialized.");
    return true;
  }

  /*lint -e1561*/
  auto proto_iter = options.find("ge.opsProtoLibPath");
  /*lint +e1561*/
  if (proto_iter == options.end()) {
    GELOGW("ge.opsProtoLibPath option not set, return.");
    return false;
  }

  pluginPath_ = proto_iter->second;
  LoadOpsProtoPluginSo(pluginPath_);

  is_init_ = true;

  return true;
}

void OpsProtoManager::Finalize() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!is_init_) {
    GELOGI("OpsProtoManager is not initialized.");
    return;
  }

  for (auto handle : handles_) {
    if (handle != nullptr) {
      if (dlclose(handle) != 0) {
        GELOGW("failed to close handle, message: %s", dlerror());
        continue;
      }
      GELOGI("close opsprotomanager handler success");
    } else {
      GELOGW("close opsprotomanager handler failure, handler is nullptr");
    }
  }

  is_init_ = false;
}

static std::vector<std::string> Split(const std::string &str, char delim) {
  std::vector<std::string> elems;
  if (str.empty()) {
    elems.emplace_back("");
    return elems;
  }

  std::stringstream ss(str);
  std::string item;

  while (getline(ss, item, delim)) {
    elems.push_back(item);
  }

  auto str_size = str.size();
  if (str_size > 0 && str[str_size - 1] == delim) {
    elems.emplace_back("");
  }

  return elems;
}

static void FindParserSo(const std::string &path, std::vector<std::string> &file_list) {
  // Lib plugin path not exist
  if (path.empty()) {
    GELOGI("realPath is empty");
    return;
  }
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(path.size() >= PATH_MAX, return, "path is invalid");

  char resolved_path[PATH_MAX] = {0};

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  if (realpath(path.c_str(), resolved_path) == nullptr) {
    GELOGW("the path [%s] not exsit.", path.c_str());
    return;
  }

  struct dirent *dent = nullptr;
  DIR *dir = opendir(resolved_path);
  // Lib plugin path not exist
  if (dir == nullptr) {
    GELOGW("Open directory %s failed,maybe it is not exit or not a dir", resolved_path);
    return;
  }

  while ((dent = readdir(dir)) != nullptr) {
    if (strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0) {
      continue;
    }
    std::string name = dent->d_name;
    std::string full_name = path + "/" + name;
    const std::string so_suff = ".so";

    if (dent->d_type != DT_DIR && name.size() >= so_suff.size() &&
        name.compare(name.size() - so_suff.size(), so_suff.size(), so_suff) == 0) {
      file_list.push_back(full_name);
      GELOGI("OpsProtoManager Parse full name = %s \n", full_name.c_str());
    }
  }
  if (closedir(dir) != 0) {
    GELOGW("close dir fail.");
  }
}

static void GetPluginSoFileList(const std::string &path, std::vector<std::string> &file_list) {
  // Support multi lib directory with ":" as delimiter
  std::vector<std::string> v_path = Split(path, ':');

  for (size_t i = 0; i < v_path.size(); ++i) {
    FindParserSo(v_path[i], file_list);
    GELOGI("OpsProtoManager full name = %s", v_path[i].c_str());
  }
}

void OpsProtoManager::LoadOpsProtoPluginSo(std::string &path) {
  if (path.empty()) {
    GELOGE(GRAPH_FAILED, "filePath is invalid. please check your text file %s.", path.c_str());
    return;
  }
  std::vector<std::string> file_list;

  // If there is .so file in the lib path
  GetPluginSoFileList(path, file_list);

  // Not found any .so file in the lib path
  if (file_list.empty()) {
    GELOGE(GRAPH_FAILED, "OpsProtoManager can not find any plugin file in pluginPath: %s \n", path.c_str());
    return;
  }
  // Warning message
  GELOGW("The shared library will not be checked. Please ensure that the source of the shared library is trusted.");

  // Load .so file
  for (auto elem : file_list) {
    void *handle = dlopen(elem.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      GELOGW("OpsProtoManager dlopen failed, plugin name:%s. Message(%s).", elem.c_str(), dlerror());
      continue;
    } else {
      // Close dl when the program exist, not close here
      GELOGI("OpsProtoManager plugin load %s success.", elem.c_str());
      handles_.push_back(handle);
    }
  }
}
}  // namespace ge
