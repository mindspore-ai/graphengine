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

#ifndef INC_FRAMEWORK_COMMON_TLV_TLV_H_
#define INC_FRAMEWORK_COMMON_TLV_TLV_H_

#if defined(__cplusplus)
extern "C" {
#endif

#pragma pack(1)  // single-byte alignment

// tlv struct
struct TlvHead {
  uint32_t type;
  uint32_t len;
  uint8_t data[0];
};

#pragma pack()

#if defined(__cplusplus)
}
#endif

#endif  // INC_FRAMEWORK_COMMON_TLV_TLV_H_