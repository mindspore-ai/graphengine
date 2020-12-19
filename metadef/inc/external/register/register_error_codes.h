/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef INC_EXTERNAL_REGISTER_REGISTER_ERROR_CODES_H_
#define INC_EXTERNAL_REGISTER_REGISTER_ERROR_CODES_H_

#define SYSID_FWK 3     // Subsystem ID
#define MODID_COMMON 0  // Common module ID

#define DECLARE_ERRORNO(sysid, modid, name, value) \
  const domi::Status name =                        \
      ((0xFF & ((uint8_t)sysid)) << 24) | ((0xFF & ((uint8_t)modid)) << 16) | (0xFFFF & ((uint16_t)value));

#define DECLARE_ERRORNO_COMMON(name, value) DECLARE_ERRORNO(SYSID_FWK, MODID_COMMON, name, value)

namespace domi {
using Status = uint32_t;

// General error code
DECLARE_ERRORNO(0, 0, SUCCESS, 0);
DECLARE_ERRORNO(0xFF, 0xFF, FAILED, 0xFFFFFFFF);
DECLARE_ERRORNO_COMMON(PARAM_INVALID, 1);  // 50331649
DECLARE_ERRORNO(SYSID_FWK, 1, SCOPE_NOT_CHANGED, 201);
}  // namespace domi

#endif  // INC_EXTERNAL_REGISTER_REGISTER_ERROR_CODES_H_
