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

#ifndef AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_BIN_CACHE_DEF_H_
#define AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_BIN_CACHE_DEF_H_
#include <cstdint>
namespace ge {
namespace fuzz_compile {
enum NodeBinMode : std::int32_t
{
  kOneNodeSingleBinMode,
  kOneNodeMultipleBinsMode,
  kNodeBinModeEnd
};
}
}
#endif // AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_BIN_CACHE_DEF_H_
