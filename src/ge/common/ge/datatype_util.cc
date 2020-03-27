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

#include "common/ge/datatype_util.h"

#include <map>

namespace {
const std::vector<ge::DataType> kEmptyDatatypeVector;
std::map<ge::DataType, std::vector<ge::DataType>> g_translatable_data_type = {
    // key:src datatype, value:dst datatype
    {ge::DT_FLOAT, {ge::DT_FLOAT16, ge::DT_FLOAT}},
    {ge::DT_BOOL, {ge::DT_INT32}},
    {ge::DT_FLOAT16, {ge::DT_FLOAT, ge::DT_FLOAT16}},
    {ge::DT_INT64, {ge::DT_INT32}}};

std::map<ge::DataType, std::vector<ge::DataType>> g_reverse_translatable_data_type = {
    // key:dst datatype,value:src datatype
    {ge::DT_FLOAT16, {ge::DT_FLOAT, ge::DT_FLOAT16}},
    {ge::DT_INT32, {ge::DT_BOOL, ge::DT_INT64}},
    {ge::DT_FLOAT, {ge::DT_FLOAT16, ge::DT_FLOAT}}};
}  // namespace

namespace ge {
bool DataTypeUtil::DataTypeTranslatable(const ge::DataType &src_out_data_type, const ge::DataType &dst_in_data_type) {
  auto search = g_translatable_data_type.find(src_out_data_type);
  if (search == g_translatable_data_type.end()) {
    return false;
  }

  for (auto data_type : search->second) {
    if (data_type == dst_in_data_type) {
      return true;
    }
  }

  return false;
}

const std::vector<ge::DataType> &DataTypeUtil::GetTranslatableDataTypesBySrc(const ge::DataType &src_out_data_type) {
  auto search = g_translatable_data_type.find(src_out_data_type);
  if (search == g_translatable_data_type.end()) {
    return kEmptyDatatypeVector;
  }

  return search->second;
}

const std::vector<ge::DataType> &DataTypeUtil::GetTranslatableDataTypesByDst(const ge::DataType &dst_in_data_type) {
  auto search = g_reverse_translatable_data_type.find(dst_in_data_type);
  if (search == g_reverse_translatable_data_type.end()) {
    return kEmptyDatatypeVector;
  }

  return search->second;
}
}  // namespace ge
