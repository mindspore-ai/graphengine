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

#ifndef INC_GRAPH_UTILS_TYPE_UTILS_H_
#define INC_GRAPH_UTILS_TYPE_UTILS_H_

#include <map>
#include <unordered_set>
#include <string>
#include "graph/def_types.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/usr_types.h"
#include "register/register_types.h"
#include "external/register/register_fmk_types.h"

namespace ge {
class TypeUtils {
 public:
  static bool IsDataTypeValid(DataType dt);
  static bool IsFormatValid(Format format);
  static bool IsInternalFormat(Format format);

  static std::string ImplyTypeToSerialString(domi::ImplyType imply_type);
  static std::string DataTypeToSerialString(DataType data_type);
  static DataType SerialStringToDataType(const std::string &str);
  static std::string FormatToSerialString(Format format);
  static Format SerialStringToFormat(const std::string &str);
  static Format DataFormatToFormat(const std::string &str);
  static Format DomiFormatToFormat(domi::domiTensorFormat_t domi_format);
  static std::string FmkTypeToSerialString(domi::FrameworkType fmk_type);

  static graphStatus Usr2DefQuantizeFactorParams(const UsrQuantizeFactorParams &usr, QuantizeFactorParams &def);
  static graphStatus Def2UsrQuantizeFactorParams(const QuantizeFactorParams &def, UsrQuantizeFactorParams &usr);

  static bool GetDataTypeLength(ge::DataType data_type, uint32_t &length);
  static bool CheckUint64MulOverflow(uint64_t a, uint32_t b);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_TYPE_UTILS_H_
