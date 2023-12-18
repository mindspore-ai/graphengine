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

#ifndef INC_FRAMEWORK_COMMON_STRING_UTIL_H_
#define INC_FRAMEWORK_COMMON_STRING_UTIL_H_
#include "common/ge_common/string_util.h"
namespace ge {
template<typename Iterator>
static std::string StrJoin(Iterator begin, Iterator end, const std::string &separator) {
  if (begin == end) {
    return "";
  }
  std::stringstream str_stream;
  str_stream << *begin;
  for (Iterator it = std::next(begin); it != end; ++it) {
    str_stream << separator << *it;
  }
  return str_stream.str();
}
}
#endif  // INC_FRAMEWORK_COMMON_STRING_UTIL_H_
