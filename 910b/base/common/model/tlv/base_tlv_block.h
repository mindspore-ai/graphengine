/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef BASE_TLV_BLOCK_H_
#define BASE_TLV_BLOCK_H_

#include "graph/def_types.h"

namespace ge {
class BaseTlvBlock {
public:
  virtual size_t Size() = 0;
  virtual bool Serilize(uint8_t ** const addr, size_t &left_size) = 0;
  virtual bool NeedSave() = 0;

  BaseTlvBlock() = default;
  virtual ~BaseTlvBlock() = default;

  BaseTlvBlock &operator=(const BaseTlvBlock &) & = delete;
  BaseTlvBlock(const BaseTlvBlock &) = delete;
};
}
#endif