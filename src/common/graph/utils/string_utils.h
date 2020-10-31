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

#ifndef COMMON_GRAPH_UTILS_STRING_UTILS_H_
#define COMMON_GRAPH_UTILS_STRING_UTILS_H_

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
#include "securec.h"

namespace ge {
class StringUtils {
 public:
  static std::string &Ltrim(std::string &s) {
    (void)s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
    return s;
  }

  static std::string &Rtrim(std::string &s) {
    (void)s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); }).base(), s.end());
    return s;
  }

  /// @ingroup domi_common
  /// @brief trim space
  static std::string &Trim(std::string &s) { return Ltrim(Rtrim(s)); }

  // split string
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
};
}  // namespace ge
#endif  // COMMON_GRAPH_UTILS_STRING_UTILS_H_
