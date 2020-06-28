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
#include "graph/utils/type_utils.h"

namespace ge {
std::map<string, string> TBEPluginManager::options_ = {};

// Get Singleton Instance
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY TBEPluginManager &TBEPluginManager::Instance() {
  static TBEPluginManager instance_ptr_;
  return instance_ptr_;
}

Status TBEPluginManager::ClearHandles_() {
  Status ret = SUCCESS;
  for (const auto &handle : handles_vec_) {
    if (dlclose(handle) != 0) {
      ret = FAILED;
      GELOGW("Failed to close handle: %s", dlerror());
    }
  }
  handles_vec_.clear();
  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status TBEPluginManager::Finalize() {
  Status ret = ClearHandles_();
  return ret;
}

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

void TBEPluginManager::ProcessSoFullName(vector<string> &file_list, string &caffe_parser_path, string &full_name,
                                         const string &caffe_parser_so_suff, const string &aicpu_so_suff,
                                         const string &aicpu_host_so_suff) {
  if (full_name.size() >= caffe_parser_so_suff.size() &&
      full_name.compare(full_name.size() - caffe_parser_so_suff.size(), caffe_parser_so_suff.size(),
                        caffe_parser_so_suff) == 0) {
    caffe_parser_path = full_name;
  } else if ((full_name.size() >= aicpu_so_suff.size() &&
              full_name.compare(full_name.size() - aicpu_so_suff.size(), aicpu_so_suff.size(), aicpu_so_suff) == 0) ||
             (full_name.size() >= aicpu_host_so_suff.size() &&
              full_name.compare(full_name.size() - aicpu_host_so_suff.size(), aicpu_host_so_suff.size(),
                                aicpu_host_so_suff) == 0)) {
    // aicpu so, Put the file path into the omgcontext and save into the model in the builder stage.
    domi::GetContext().aicpu_op_run_paths.push_back(full_name);
  } else {
    // Save parser so path into file_list vector
    file_list.push_back(full_name);
  }
}

void TBEPluginManager::FindParserSo(const string &path, vector<string> &file_list, string &caffe_parser_path) {
  // Path, change to absolute path
  string real_path = RealPath(path.c_str());
  // Plugin path does not exist
  if (real_path.empty()) {
    GELOGW("RealPath is empty.");
    return;
  }
  struct stat stat_buf;
  if ((stat(real_path.c_str(), &stat_buf) != 0) || (!S_ISDIR(stat_buf.st_mode))) {
    GELOGW("%s is not a dir.", real_path.c_str());
    return;
  }
  struct dirent *dent(0);
  DIR *dir = opendir(real_path.c_str());
  // Plugin path does not exist
  if (dir == nullptr) {
    GELOGW("Open directory %s failed.", real_path.c_str());
    return;
  }

  while ((dent = readdir(dir)) != nullptr) {
    if (strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0) continue;
    string name = dent->d_name;
    string full_name = real_path + "/" + name;
    const string so_suff = ".so";
    const string caffe_parser_so_suff = "lib_caffe_parser.so";
    const string aicpu_so_suff = "_aicpu.so";
    const string aicpu_host_so_suff = "_online.so";
    if (name.size() >= so_suff.size() && name.compare(name.size() - so_suff.size(), so_suff.size(), so_suff) == 0) {
      ProcessSoFullName(file_list, caffe_parser_path, full_name, caffe_parser_so_suff, aicpu_so_suff,
                        aicpu_host_so_suff);
    } else {
      FindParserSo(full_name, file_list, caffe_parser_path);
    }
  }
  closedir(dir);
}

void TBEPluginManager::GetPluginSoFileList(const string &path, vector<string> &file_list, string &caffe_parser_path) {
  // Support to split multiple so directories by ":"
  vector<string> v_path = StringUtils::Split(path, ':');
  for (size_t i = 0; i < v_path.size(); ++i) {
    FindParserSo(v_path[i], file_list, caffe_parser_path);
    GELOGI("CustomOpLib full name = %s", v_path[i].c_str());
  }
}

void TBEPluginManager::GetCustomOpPath(std::string &customop_path) {
  GELOGI("Enter get custom op path schedule");
  std::string fmk_type;
  domi::FrameworkType type = domi::TENSORFLOW;
  auto it = options_.find(FRAMEWORK_TYPE);
  if (it != options_.end()) {
    type = static_cast<domi::FrameworkType>(std::strtol(it->second.c_str(), nullptr, 10));
  }
  fmk_type = ge::TypeUtils::FmkTypeToSerialString(type);
  GELOGI("Framework type is %s.", fmk_type.c_str());

  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env != nullptr) {
    std::string path = path_env;
    customop_path = (path + "/framework/custom" + "/:") + (path + "/framework/built-in/" + fmk_type);
    GELOGI("Get custom so path from env : %s", path_env);
    return;
  }
  std::string path_base = GetPath();
  GELOGI("path_base is %s", path_base.c_str());
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);
  customop_path = (path_base + "ops/framework/custom" + "/:") + (path_base + "ops/framework/built-in/" + fmk_type);
  return;
}

void TBEPluginManager::LoadCustomOpLib() {
  LoadPluginSo();

  std::vector<OpRegistrationData> registration_datas = domi::OpRegistry::Instance()->registrationDatas;
  GELOGI("The size of registration_datas is: %zu", registration_datas.size());
  for (OpRegistrationData reg_data : registration_datas) {
    bool ret = CheckRegisterStatus(reg_data);
    if (ret) {
      GELOGD("Begin to register optype: %s, imply_type: %u", reg_data.GetOmOptype().c_str(),
             static_cast<uint32_t>(reg_data.GetImplyType()));
      domi::OpRegistry::Instance()->Register(reg_data);
    }
  }
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void TBEPluginManager::LoadPluginSo() {
  vector<string> file_list;
  string caffe_parser_path;
  std::string plugin_path;
  GetCustomOpPath(plugin_path);

  // Whether there are files in the plugin so path
  GetPluginSoFileList(plugin_path, file_list, caffe_parser_path);

  //  No file
  if (file_list.empty()) {
    // Print log
    GELOGW("Can not find any plugin file in plugin_path: %s", plugin_path.c_str());
  }

  GELOGW("The shared library will not be checked. Please ensure that the source of the shared library is trusted.");

  // Load other so files except lib_caffe_parser.so in the plugin so path
  for (auto elem : file_list) {
    StringUtils::Trim(elem);

    void *handle = dlopen(elem.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (handle == nullptr) {
      GELOGW("dlopen failed, plugin name:%s. Message(%s).", elem.c_str(), dlerror());
    } else if (find(handles_vec_.begin(), handles_vec_.end(), handle) == handles_vec_.end()) {
      // Close dl when the program exist, not close here
      GELOGI("Plugin load %s success.", elem.c_str());
      handles_vec_.push_back(handle);
    } else {
      GELOGI("Plugin so has already been loaded, no need to load again.");
    }
  }
}

bool TBEPluginManager::CheckRegisterStatus(const OpRegistrationData &reg_data) {
  bool ret = true;
  static char *parser_priority = std::getenv("PARSER_PRIORITY");
  static bool keep_cce = parser_priority != nullptr && string(parser_priority) == "cce";
  auto ori_optype_set = reg_data.GetOriginOpTypeSet();
  for (const auto &op_type : ori_optype_set) {
    domi::ImplyType imply_type = domi::OpRegistry::Instance()->GetImplyTypeByOriOpType(op_type);
    GELOGD("Enter into reg_data loop. op_type = %s , om_optype_ = %s", op_type.c_str(), reg_data.GetOmOptype().c_str());
    if (imply_type != domi::ImplyType::BUILDIN) {
      if ((keep_cce && reg_data.GetImplyType() != domi::ImplyType::CCE) ||
          (!keep_cce && reg_data.GetImplyType() != domi::ImplyType::TVM)) {
        GELOGD("op_type[%s] does not need to be changed, om_optype:%s.", op_type.c_str(),
               reg_data.GetOmOptype().c_str());
        ret = false;
      } else {
        GELOGI("op_type[%s] will be changed to om_optype:%s.", op_type.c_str(), reg_data.GetOmOptype().c_str());
      }
    } else {
      GELOGD("First register in ge initialize, original type: %s, om_optype: %s, imply type: %d.", op_type.c_str(),
             reg_data.GetOmOptype().c_str(), static_cast<int>(reg_data.GetImplyType()));
    }
  }
  return ret;
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

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void TBEPluginManager::InitPreparation(
  const std::map<string, string> &options) {
  options_.insert(options.begin(), options.end());
  // Load TBE plugin
  TBEPluginManager::Instance().LoadCustomOpLib();
  Status ret = CheckCustomAiCpuOpLib();
  if (ret != SUCCESS) {
    GELOGE(ret, "Check custom aicpu run so failed!");
    return;
  }
}
}  // namespace ge
