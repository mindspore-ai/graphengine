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

#include "common/util/error_manager/error_manager.h"

  ErrorManager &ErrorManager::GetInstance() {
    static ErrorManager instance;
    return instance;
  }

  ///
  /// @brief init
  /// @param [in] path: current so path
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::Init(std::string path) { return 0; }

  ///
  /// @brief Report error message
  /// @param [in] error_code: error code
  /// @param [in] args_map: parameter map
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::ReportErrMessage(std::string error_code, const std::map<std::string, std::string> &args_map) {
    return 0;
  }

  ///
  /// @brief output error message
  /// @param [in] handle: print handle
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::OutputErrMessage(int handle) { return 0; }

  ///
  /// @brief output  message
  /// @param [in] handle: print handle
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::OutputMessage(int handle) { return 0; }

  ///
  /// @brief Report error message
  /// @param [in] key: vector parameter key
  /// @param [in] value: vector parameter value
  ///
  void ErrorManager::ATCReportErrMessage(std::string error_code, const std::vector<std::string> &key,
                                         const std::vector<std::string> &value) { 
  }

  ///
  /// @brief report graph compile failed message such as error code and op_name in mstune case
  /// @param [in] msg: failed message map, key is error code, value is op_name
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::ReportMstuneCompileFailedMsg(const std::string &root_graph_name,
		                                 const std::map<std::string, std::string> &msg) { return 0; }

  ///
  /// @brief get graph compile failed message in mstune case
  /// @param [in] graph_name: graph name
  /// @param [out] msg_map: failed message map, key is error code, value is op_name list
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::GetMstuneCompileFailedMsg(const std::string &graph_name, std::map<std::string, std::vector<std::string>> &msg_map) { return 0; }


