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

#ifndef INC_FRAMEWORK_COMMON_L2_CACHE_OPTIMIZE_H_
#define INC_FRAMEWORK_COMMON_L2_CACHE_OPTIMIZE_H_

#include <cstdint>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/compute_graph.h"

namespace ge {
// Size of RC memory alignment, 2M
constexpr size_t ALIGN_SIZE = 2097152U;

constexpr uint32_t RC_VALUE_DEFAULT = 1U;
constexpr uint32_t RC_VALUE_MAX = 32U;

}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_L2_CACHE_OPTIMIZE_H_