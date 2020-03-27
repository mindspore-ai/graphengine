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

#ifndef GE_GRAPH_MANAGER_UTIL_HCOM_UTIL_H_
#define GE_GRAPH_MANAGER_UTIL_HCOM_UTIL_H_

#include <map>
#include <string>
#include <vector>

#include "common/debug/log.h"
#include "common/string_util.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/op_desc.h"
#include "hccl/hcom.h"

namespace ge {
using std::string;
using std::vector;

static std::map<int64_t, hcclDataType_t> kConstOpHcomDataType = {
    {ge::DT_FLOAT, HCCL_DATA_TYPE_FLOAT},
    {ge::DT_FLOAT16, HCCL_DATA_TYPE_HALF},
    {ge::DT_INT8, HCCL_DATA_TYPE_INT8},
    {ge::DT_INT32, HCCL_DATA_TYPE_INT},
};

static std::map<hcclDataType_t, int32_t> kConstOpHcomDataTypeSize  = {
    {HCCL_DATA_TYPE_FLOAT, sizeof(float)},
    {HCCL_DATA_TYPE_HALF, sizeof(float) / 2},
    {HCCL_DATA_TYPE_INT8, sizeof(int8_t)},
    {HCCL_DATA_TYPE_INT, sizeof(int32_t)},
};

class HcomOmeUtil {
 public:
  ///
  /// @ingroup domi_ome
  /// @brief GetHcomDataType
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcomDataType(const ge::ConstOpDescPtr &op_desc, hcclDataType_t &data_type);

  ///
  /// @ingroup domi_ome
  /// @brief GetHcomTypeSize
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcomTypeSize(hcclDataType_t data_type, int32_t &size);

  ///
  /// @ingroup domi_ome
  /// @brief GetHcomCount
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcomCount(const ge::ConstOpDescPtr &op_desc, hcclDataType_t data_type, bool is_allgather,
                             int &count);

  ///
  /// @ingroup domi_ome
  /// @brief GetHcomOperationType
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcomOperationType(const ge::ConstOpDescPtr &op_desc, hcclRedOp_t &op_type);

  ///
  /// @ingroup domi_ome
  /// @brief GetHcomRootId
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcomRootId(const ge::ConstOpDescPtr &op_desc, int64_t &root_id);
};
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_UTIL_HCOM_UTIL_H_
