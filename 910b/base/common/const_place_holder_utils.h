/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef GE_COMMON_CONST_PLACE_HOLDER_UTILS_H_
#define GE_COMMON_CONST_PLACE_HOLDER_UTILS_H_

#include "framework/common/types.h"
#include "graph/op_desc.h"
#include "ge/ge_api_types.h"

namespace ge {
  ge::Status GetConstPlaceHolderAddr(const OpDescPtr &op_desc, uint8_t* &dev_address);
}  // namespace ge

#endif  // GE_COMMON_CONST_PLACE_HOLDER_UTILS_H_
