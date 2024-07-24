/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_FRAMEWORK_ENGINE_DNNENGINE_H_
#define INC_FRAMEWORK_ENGINE_DNNENGINE_H_

#include <map>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/types.h"

namespace ge {
enum class PriorityEnum : uint32_t {
  COST_0 = 0,
  COST_1 = 1,
  COST_2 = 2,
  COST_3 = 3,
  COST_4 = 4,
  COST_9 = 9,
  COST_10 = 10,
};

struct DNNEngineAttribute {
  std::string engine_name;
  std::vector<std::string> mem_type;
  PriorityEnum compute_cost;
  enum RuntimeType runtime_type;  // HOST, DEVICE
  // If engine input format must be specific, set this attribute, else set FORMAT_RESERVED
  Format engine_input_format;
  Format engine_output_format;
  bool atomic_engine_flag;
};

class GE_FUNC_VISIBILITY DNNEngine {
 public:
  DNNEngine() = default;
  explicit DNNEngine(const DNNEngineAttribute &attrs) {
    engine_attribute_ = attrs;
  }
  virtual ~DNNEngine() = default;
  Status Initialize(const std::map<std::string, std::string> &options) const {
   (void)options;
    return SUCCESS;
  }
  Status Finalize() const {
    return SUCCESS;
  }
  void GetAttributes(DNNEngineAttribute &attr) const {
    attr = engine_attribute_;
  }
  bool IsAtomic() const {
    return engine_attribute_.atomic_engine_flag;
  }

 protected:
  DNNEngineAttribute engine_attribute_;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_ENGINE_DNNENGINE_H_
