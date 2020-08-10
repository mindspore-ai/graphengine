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

#ifndef ERROR_MANAGER_H_
#define ERROR_MANAGER_H_

#include <map>
#include <string>
#include <vector>

class ErrorManager {
 public:
  ///
  /// @brief Obtain  ErrorManager instance
  /// @return ErrorManager instance
  ///
  static ErrorManager &GetInstance();

  ///
  /// @brief init
  /// @param [in] path: current so path
  /// @return int 0(success) -1(fail)
  ///
  int Init(std::string path);

  ///
  /// @brief Report error message
  /// @param [in] error_code: error code
  /// @param [in] args_map: parameter map
  /// @return int 0(success) -1(fail)
  ///
  int ReportErrMessage(std::string error_code, const std::map<std::string, std::string> &args_map);

  ///
  /// @brief output error message
  /// @param [in] handle: print handle
  /// @return int 0(success) -1(fail)
  ///
  int OutputErrMessage(int handle);

  ///
  /// @brief output  message
  /// @param [in] handle: print handle
  /// @return int 0(success) -1(fail)
  ///
  int OutputMessage(int handle);

  ///
  /// @brief Report error message
  /// @param [in] key: vector parameter key
  /// @param [in] value: vector parameter value
  ///
  void ATCReportErrMessage(std::string error_code, const std::vector<std::string> &key = {},
                           const std::vector<std::string> &value = {});

 private:
  struct ErrorInfo {
    std::string error_id;
    std::string error_message;
    std::vector<std::string> arg_list;
  };

  ErrorManager() {}
  ~ErrorManager() {}

  ErrorManager(const ErrorManager &) = delete;
  ErrorManager(ErrorManager &&) = delete;
  ErrorManager &operator=(const ErrorManager &) = delete;
  ErrorManager &operator=(ErrorManager &&) = delete;

  int ParseJsonFile(std::string path);

  int ReadJsonFile(const std::string &file_path, void *handle);

  bool is_init_ = false;
  std::map<std::string, ErrorInfo> error_map_;
  std::vector<std::string> error_messages_;
  std::vector<std::string> warning_messages_;
};

#endif  // ERROR_MANAGER_H_
