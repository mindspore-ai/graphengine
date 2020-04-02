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

#include "external/graph/attr_value.h"
#include "debug/ge_log.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_attr_value.h"

namespace ge {
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrValue::AttrValue() { impl = ComGraphMakeShared<AttrValueImpl>(); }

#define ATTR_VALUE_SET_GET_IMP(type)                 \
  graphStatus AttrValue::GetValue(type &val) const { \
    if (impl != nullptr) {                           \
      GELOGW("GetValue failed.");                    \
      return impl->geAttrValue_.GetValue<type>(val); \
    }                                                \
    return GRAPH_FAILED;                             \
  }

ATTR_VALUE_SET_GET_IMP(AttrValue::STR)
ATTR_VALUE_SET_GET_IMP(AttrValue::INT)
ATTR_VALUE_SET_GET_IMP(AttrValue::FLOAT)
}  // namespace ge
