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

#ifndef GE_COMMON_FORMATS_UTILS_FORMATS_TRANS_UTILS_H_
#define GE_COMMON_FORMATS_UTILS_FORMATS_TRANS_UTILS_H_

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include "external/graph/types.h"
#include "graph/ge_tensor.h"
#include "formats/register_format_transfer.h"

namespace ge {
namespace formats {
int64_t GetCubeSizeByDataType(const DataType data_type);

/**
 * Convert a std::vector to a std::string using ','
 * @tparam T
 * @param vec
 * @return
 */
template <typename T>
std::string JoinToString(const std::vector<T> &vec) {
  std::stringstream ss;
  bool first = true;
  for (auto &ele : vec) {
    if (first) {
      first = false;
    } else {
      ss << ",";
    }
    ss << ele;
  }
  return ss.str();
}

std::string ShapeToString(const GeShape &shape);

std::string ShapeToString(const std::vector<int64_t> &shape);

std::string RangeToString(const std::vector<std::pair<int64_t, int64_t>> &ranges);

int64_t GetItemNumByShape(const std::vector<int64_t> &shape);

bool CheckShapeValid(const std::vector<int64_t> &shape, const int64_t expect_dims);

bool IsShapeValid(const std::vector<int64_t> &shape);

bool IsShapeEqual(const GeShape &src, const GeShape &dst);

bool IsTransShapeSrcCorrect(const TransArgs &args, const std::vector<int64_t> &expect_shape);

bool IsTransShapeDstCorrect(const TransArgs &args, const std::vector<int64_t> &expect_shape);

template <typename T>
T Ceil(const T n1, const T n2) {
  if (n1 == 0) {
    return 0;
  }
  return (n2 != 0) ? (((n1 - 1) / n2) + 1) : 0;
}

inline int64_t Measure(int64_t x, int64_t y) {
  int64_t z = y;
  if (y == 0) {
    return -1;
  }
  while ((x % y) != 0) {
    z = x % y;
    x = y;
    y = z;
  }
  return z;
}

// least common multiple
inline int64_t Lcm(const int64_t a, const int64_t b) {
  if (b == 0) {
    return -1;
  }
  const int64_t temp = (a * b) / (Measure(a, b));
  return temp;
}
}  // namespace formats
}  // namespace ge
#endif  // GE_COMMON_FORMATS_UTILS_FORMATS_TRANS_UTILS_H_
