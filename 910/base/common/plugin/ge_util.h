/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GE_COMMON_GE_GE_UTIL_H_
#define GE_COMMON_GE_GE_UTIL_H_

#include <iostream>
#include <memory>
#include <utility>
#include "common/util/mem_utils.h"
#define GE_DELETE_ASSIGN_AND_COPY(Classname)        \
  Classname &operator=(const Classname &) & = delete; \
  Classname(const Classname &) = delete

namespace ge {
template <typename T>
struct MakeUniq {
  using unique_object = std::unique_ptr<T>;
};

template <typename T>
struct MakeUniq<T[]> {
  using unique_array = std::unique_ptr<T[]>;
};

template <typename T, size_t B>
struct MakeUniq<T[B]> {
  struct invalid_type { };
};

template <typename T, typename... Args>
static inline typename MakeUniq<T>::unique_object MakeUnique(Args &&... args) {
  using T_nc = typename std::remove_const<T>::type;
  return std::unique_ptr<T>(new (std::nothrow) T_nc(std::forward<Args>(args)...));
}

template <typename T>
static inline typename MakeUniq<T>::unique_array MakeUnique(const size_t num) {
  return std::unique_ptr<T>(new (std::nothrow) typename std::remove_extent<T>::type[num]());
}

template <typename T, typename... Args>
static inline typename MakeUniq<T>::invalid_type MakeUnique(Args &&...) = delete;
}  // namespace ge
#endif  // GE_COMMON_GE_GE_UTIL_H_
