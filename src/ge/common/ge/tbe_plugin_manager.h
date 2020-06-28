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

#ifndef GE_COMMON_GE_TBE_PLUGIN_MANAGER_H_
#define GE_COMMON_GE_TBE_PLUGIN_MANAGER_H_

#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "external/ge/ge_api_error_codes.h"
#include "external/register/register.h"

namespace ge {
using SoHandlesVec = std::vector<void *>;
using std::function;
using std::map;
using std::string;
using std::vector;

class TBEPluginManager {
 public:
  Status Finalize();

  // Get TBEPluginManager singleton instance
  static TBEPluginManager &Instance();

  static string GetPath();

  static void InitPreparation(const std::map<string, string> &options);

  void LoadPluginSo();

 private:
  TBEPluginManager() = default;
  ~TBEPluginManager() = default;
  Status ClearHandles_();

  static void ProcessSoFullName(vector<string> &file_list, string &caffe_parser_path, string &full_name,
                                const string &caffe_parser_so_suff, const string &aicpu_so_suff,
                                const string &aicpu_host_so_suff);
  static void FindParserSo(const string &path, vector<string> &file_list, string &caffe_parser_path);
  static void GetPluginSoFileList(const string &path, vector<string> &file_list, string &caffe_parser_path);
  static void GetCustomOpPath(std::string &customop_path);
  void LoadCustomOpLib();
  static Status CheckCustomAiCpuOpLib();
  static bool CheckRegisterStatus(const OpRegistrationData &reg_data);

  SoHandlesVec handles_vec_;
  static std::map<string, string> options_;
};
}  // namespace ge

#endif  // GE_COMMON_GE_TBE_PLUGIN_MANAGER_H_
