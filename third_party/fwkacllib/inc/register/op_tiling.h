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

#ifndef INC_REGISTER_OP_TILING_H_
#define INC_REGISTER_OP_TILING_H_

#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "register/op_tiling_registry.h"
#include <nlohmann/json.hpp>

namespace optiling {

#define REGISTER_OP_TILING_FUNC(optype, opfunc) REGISTER_OP_TILING_FUNC_UNIQ_HELPER(optype, opfunc, __COUNTER__)
#define REGISTER_OP_TILING_FUNC_UNIQ_HELPER(optype, opfunc, counter) \
  REGISTER_OP_TILING_FUNC_UNIQ(optype, opfunc, counter)
#define REGISTER_OP_TILING_FUNC_UNIQ(optype, opfunc, counter) \
  static OpTilingInterf g_##optype##TilingInterf##counter(#optype, opfunc)

using OpTilingFuncOld =
  std::function<bool(const std::string &, const TeOpParas &, const nlohmann::json &, OpRunInfo &)>;

class FMK_FUNC_HOST_VISIBILITY OpTilingInterf {
 public:
  OpTilingInterf(std::string op_type, OpTilingFuncOld func);
  ~OpTilingInterf() = default;
  static std::string OpTilingUuid;
};

extern "C" ge::graphStatus OpParaCalculate(const ge::Node &node, OpRunInfo &run_info);
extern "C" ge::graphStatus OpAtomicCalculate(const ge::Node &node, OpRunInfo &run_info);

}  // namespace optiling

#endif  // INC_REGISTER_OP_TILING_H_
