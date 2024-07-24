/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_FRAMEWORK_COMMON_OP_TYPES_H_
#define INC_FRAMEWORK_COMMON_OP_TYPES_H_

#include <set>
#include <string>

#include "graph/types.h"
#include "graph/compiler_def.h"

namespace ge {
class GE_FUNC_VISIBILITY OpTypeContainer {
 public:
  static OpTypeContainer *Instance() {
    static OpTypeContainer instance;
    return &instance;
  }
  ~OpTypeContainer() = default;

  bool Register(const std::string &op_type) { return op_type_list_.insert(op_type).second; }

  bool IsExisting(const std::string &op_type) {
    return op_type_list_.find(op_type) != op_type_list_.end();
  }

 protected:
  OpTypeContainer() {}

 private:
  std::set<std::string> op_type_list_;
};
}  // namespace ge

#define REGISTER_OPTYPE_DECLARE(var_name, str_name) \
  FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const char_t *var_name

#define REGISTER_OPTYPE_DEFINE(var_name, str_name)           \
  const char_t *var_name = str_name;                         \
  const bool VAR_UNUSED g_##var_name##_reg = OpTypeContainer::Instance()->Register(str_name)

#define IS_OPTYPE_EXISTING(str_name) (ge::OpTypeContainer::Instance()->IsExisting(str_name))
#endif  // INC_FRAMEWORK_COMMON_OP_TYPES_H_

