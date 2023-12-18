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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TABLE_DRIVEN_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TABLE_DRIVEN_H_

#include <cstddef>
#include <utility>
#include <array>

namespace gert {
template <size_t DIM1, size_t DIM2, typename TE>
class TableDriven2 {
 public:
  explicit TableDriven2(const TE &default_value) noexcept : default_value_(default_value) {
    for (auto &row : elements_) {
      for (auto &element : row) {
        element = default_value;
      }
    }
  }
  TE Find(size_t src, size_t dst) const {
    if ((src >= DIM1) || (dst >= DIM2)) {
      return default_value_;
    }
    return elements_[src][dst];
  }
  const TE *FindPointer(size_t src, size_t dst) const {
    if ((src >= DIM1) || (dst >= DIM2)) {
      return nullptr;
    }
    return &elements_[src][dst];
  }
  TE *FindPointer(size_t src, size_t dst) {
    if ((src >= DIM1) || (dst >= DIM2)) {
      return nullptr;
    }
    return &elements_[src][dst];
  }
  template <typename... Arg>
  TableDriven2 &Add(size_t src, size_t dst, Arg &&...arg) {
    auto &element = elements_[src][dst];
    element = TE(std::forward<Arg>(arg)...);
    return *this;
  }

 private:
  TE default_value_;
  std::array<std::array<TE, DIM2>, DIM1> elements_;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_TABLE_DRIVEN_H_
