/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef AIR_INC_FRAMEWORK_COMMON_PROTO_UTIL_H_
#define AIR_INC_FRAMEWORK_COMMON_PROTO_UTIL_H_

#include <sstream>
#include <string>

#include "google/protobuf/text_format.h"
#include "external/ge/ge_error_codes.h"
#include "external/graph/types.h"

namespace ge {
/**
 * @ingroup ge_common
 * @brief Converts RepeatedField to String.
 * @param [in] rpd_field  RepeatedField
 * @return string
 */
template <typename T>
GE_FUNC_VISIBILITY std::string ToString(const google::protobuf::RepeatedField<T> &rpd_field) {
  std::stringstream ss;
  ss << "[";
  for (const T x : rpd_field) {
    ss << x;
    ss << ", ";
  }
  // Delete the two extra characters at the end of the line.
  std::string str = ss.str().substr(0U, ss.str().length() - 2U);
  str += "]";
  return str;
}

/**
 *  @ingroup ge_common
 *  @brief RepeatedPtrField->String
 *  @param [in] const rpd_field  RepeatedPtrField
 *  @return String
 */
template <typename T>
GE_FUNC_VISIBILITY std::string ToString(const google::protobuf::RepeatedPtrField<T> &rpd_ptr_field) {
  std::stringstream ss;
  ss << "[";
  for (const T &x : rpd_ptr_field) {
    ss << x;
    ss << ", ";
  }
  std::string str_ret = ss.str().substr(0U, ss.str().length() - 2U);
  str_ret += "]";
  return str_ret;
}

/**
 * @ingroup ge_common
 * @brief Reads the proto structure from an array.
 * @param [in] data proto data to be read
 * @param [in] size proto data size
 * @param [out] proto Memory for storing the proto file
 * @return true success
 * @return false fail
 */
GE_FUNC_VISIBILITY bool ReadProtoFromArray(const void *const data, const int32_t size,
                                           google::protobuf::Message *const proto);

/**
 * @ingroup ge_common
 * @brief Reads the proto file in the text format.
 * @param [in] file path of proto file
 * @param [out] message Memory for storing the proto file
 * @return true success
 * @return false fail
 */
GE_FUNC_VISIBILITY bool ReadProtoFromText(const char_t *const file, google::protobuf::Message *const message);
}  // namespace ge
#endif  // AIR_INC_FRAMEWORK_COMMON_PROTO_UTIL_H_
