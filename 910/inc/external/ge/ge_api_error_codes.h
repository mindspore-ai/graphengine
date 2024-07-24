/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GE_GE_API_ERROR_CODES_H_
#define INC_EXTERNAL_GE_GE_API_ERROR_CODES_H_

#include <map>
#include <string>
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "graph/ascend_string.h"

#ifdef __GNUC__
#define ATTRIBUTED_DEPRECATED(replacement) __attribute__((deprecated("Please use " #replacement " instead.")))
#else
#define ATTRIBUTED_DEPRECATED(replacement) __declspec(deprecated("Please use " #replacement " instead."))
#endif
#ifndef GE_ERRORNO_EXTERNAL
#define GE_ERRORNO_EXTERNAL(name, desc) const ErrorNoRegisterar g_errorno_##name((name), (desc))
#endif
#ifndef GE_ERRORNO
// Code compose(4 byte), runtime: 2 bit,  type: 2 bit,   level: 3 bit,  sysid: 8 bit, modid: 5 bit, value: 12 bit
#define GE_ERRORNO(runtime, type, level, sysid, modid, name, value, desc)                               \
  constexpr ge::Status name = (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(runtime))) << 30U) | \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(type))) << 28U) |    \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(level))) << 25U) |   \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(sysid))) << 17U) |   \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(modid))) << 12U) |   \
                              (static_cast<uint32_t>(0x0FFFU) & (static_cast<uint32_t>(value)));        \
  const ErrorNoRegisterar g_errorno_##name((name), (desc))


namespace ge {
class GE_FUNC_VISIBILITY StatusFactory {
 public:
  static StatusFactory *Instance() {
    static StatusFactory instance;
    return &instance;
  }

  void RegisterErrorNo(const uint32_t err, const std::string &desc) {
    // Avoid repeated addition
    if (err_desc_.find(err) != err_desc_.end()) {
      return;
    }
    err_desc_[err] = desc;
  }

  void RegisterErrorNo(const uint32_t err, const char *const desc) {
    if (desc == nullptr) {
      return;
    }
    const std::string error_desc = desc;
    if (err_desc_.find(err) != err_desc_.end()) {
      return;
    }
    err_desc_[err] = error_desc;
  }

  std::string GetErrDesc(const uint32_t err) {
    const auto iter_find = static_cast<const std::map<uint32_t, std::string>::const_iterator>(err_desc_.find(err));
    if (iter_find == err_desc_.cend()) {
      return "";
    }
    return iter_find->second;
  }

  AscendString GetErrDescV2(const uint32_t err) {
    const auto iter_find = static_cast<const std::map<uint32_t, std::string>::const_iterator>(err_desc_.find(err));
    if (iter_find == err_desc_.cend()) {
      return AscendString("");
    }
    return AscendString(iter_find->second.c_str());
  }

 protected:
  StatusFactory() = default;
  ~StatusFactory() = default;

 private:
  std::map<uint32_t, std::string> err_desc_;
};

class GE_FUNC_VISIBILITY ErrorNoRegisterar {
 public:
  ErrorNoRegisterar(const uint32_t err, const std::string &desc) noexcept {
    StatusFactory::Instance()->RegisterErrorNo(err, desc);
  }
  ErrorNoRegisterar(const uint32_t err, const char *const desc) noexcept {
    StatusFactory::Instance()->RegisterErrorNo(err, desc);
  }
  ~ErrorNoRegisterar() = default;
};

// General error code
GE_ERRORNO(0, 0, 0, 0, 0, SUCCESS, 0, "success");
GE_ERRORNO(0b11, 0b11, 0b111, 0xFFU, 0b11111, FAILED, 0xFFFU, "failed"); /*lint !e401*/

}  // namespace ge
#endif
#endif  // INC_EXTERNAL_GE_GE_API_ERROR_CODES_H_
