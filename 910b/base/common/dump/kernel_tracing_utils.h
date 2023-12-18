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

#ifndef GE_COMMON_KERNEL_TRACING_UTILS_H_
#define GE_COMMON_KERNEL_TRACING_UTILS_H_

#include <sstream>
#include <iomanip>

namespace gert {
template <typename T, bool IsPointer = std::is_pointer<T>::value>
void PrintHex(const T *p, size_t num, std::stringstream &ss) {
  for (size_t i = 0; i < num; ++i) {
    if (!IsPointer) {
      // 通过std::setw设置输出位宽为2倍的sizeof(T)
      ss << "0x" << std::setfill('0') << std::setw(static_cast<int32_t>(sizeof(T)) * 2) << std::hex << +p[i] << ' ';
    } else {
      ss << p[i] << ' ';
    }
  }
}
} // namespace gert

#endif // GE_COMMON_KERNEL_TRACING_UTILS_H_