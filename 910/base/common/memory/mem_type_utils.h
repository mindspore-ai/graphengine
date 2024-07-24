/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

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
