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
#ifndef AIR_CXX_GE_RT_CONST_TYPES_H
#define AIR_CXX_GE_RT_CONST_TYPES_H
#include <string>
#include "external/graph/types.h"
namespace gert {
// ConstData节点所表示的类型
// ConstData不同于Data, 它将含有执行时信息语义，因此类型需要在lowering时确定
// 如果需要新增ConstData类型，请在下面的枚举中增加, 同时需要在kConstDataTypes对应增加key
enum class ConstDataType {
  kRtSession = 0,
  KWeight,
  kModelDesc,
  kTypeEnd,
};

// ConstData节点的unique key，用于在global data中索引对应的holder。
const ge::char_t* const kConstDataTypes[] = {"RtSession", "OuterWeightMem", "ModelDesc"};
inline std::string GetConstDataTypeStr(ConstDataType type) {
  auto len = sizeof(kConstDataTypes) / sizeof(ge::char_t *);
  if (static_cast<size_t>(type) >= len) {
    return "";
  }
  return kConstDataTypes[static_cast<int>(type)];
}
}
#endif  // AIR_CXX_GE_RT_CONST_TYPES_H
