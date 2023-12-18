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

#ifndef INC_FRAMEWORK_COMMON_GE_FORMAT_UTIL_H_
#define INC_FRAMEWORK_COMMON_GE_FORMAT_UTIL_H_

#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/tensor.h"

namespace ge {
class GE_FUNC_VISIBILITY GeFormatUtil {
 public:
  ///
  /// @name   TransShape
  /// @brief  transform the shape of tensor according to destination format
  /// @param  [in] src_desc       source tensor desc
  /// @param  [in] dst_format     destination format
  /// @param  [out] dst_shape     destination shape
  /// @return Status
  ///
  static Status TransShape(const TensorDesc &src_desc, const Format dst_format, std::vector<int64_t> &dst_shape);
};
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_GE_FORMAT_UTIL_H_
