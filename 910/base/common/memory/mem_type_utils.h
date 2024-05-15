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
#ifndef GE_COMMON_MEMORY_MEM_TYPE_UTILS_H
#define GE_COMMON_MEMORY_MEM_TYPE_UTILS_H

#include <string>

#include "runtime/mem.h"
#include "ge/ge_api_types.h"
#include "graph/ge_error_codes.h"

namespace ge {
class MemTypeUtils {
 public:
  static graphStatus RtMemTypeToExternalMemType(const rtMemType_t rt_mem_type, MemoryType &external_mem_type);
  static graphStatus ExternalMemTypeToRtMemType(const MemoryType external_mem_type, rtMemType_t &rt_mem_type);
  static std::string ToString(const rtMemType_t rt_mem_type);
  static std::string ToString(const MemoryType external_mem_type);
  static bool IsMemoryTypeSpecial(const int64_t memory_type);
};
}  // namespace ge
#endif  // GE_COMMON_MEMORY_MEM_TYPE_UTILS_H
