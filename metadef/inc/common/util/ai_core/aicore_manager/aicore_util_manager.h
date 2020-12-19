/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef AICORE_UTIL_MANAGER_H_
#define AICORE_UTIL_MANAGER_H_

#include <string>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class AICoreUtilManager {
 public:
  static AICoreUtilManager &Instance();
  /*
   * to initialize the aicore configuration
   * param[in] the options of init
   * param[in] engine Name
   * param[in] socVersion soc version from ge
   * return Status(SUCCESS/FAILED)
   */
  Status Initialize(const std::map<std::string, std::string> &options, std::string &soc_version);

  /*
   * to release the source of fusion manager
   * return Status(SUCCESS/FAILED)
   */
  Status Finalize();

 private:
  AICoreUtilManager();
  ~AICoreUtilManager();
  bool is_init_;
};
}  // namespace fe
#endif // AICORE_UTIL_MANAGER_H