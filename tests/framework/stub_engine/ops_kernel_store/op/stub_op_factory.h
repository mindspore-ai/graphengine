/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_OP_OP_FACTORY_H_
#define GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_OP_OP_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/ge/ge_util.h"
#include "stub_engine/ops_kernel_store/op/op.h"
#include "inc/st_types.h"

namespace ge {
namespace st {
using OP_CREATOR_FUNC = std::function<std::shared_ptr<Op>(const Node &, RunContext &)>;

/**
 * manage all the op, support create op.
 */
class GE_FUNC_VISIBILITY OpFactory {
 public:
  static OpFactory &Instance();

  /**
   *  @brief create Op.
   *  @param [in] node share ptr of node
   *  @param [in] run_context run context
   *  @return not nullptr success
   *  @return nullptr fail
   */
  std::shared_ptr<Op> CreateOp(const Node &node, RunContext &run_context);

  /**
   *  @brief Register Op create function.
   *  @param [in] type Op type
   *  @param [in] func Op create func
   */
  void RegisterCreator(const std::string &type, const std::string &lib_name, const OP_CREATOR_FUNC &func);

  const std::vector<std::string> &GetAllOps() const {
    return all_ops_;
  }

  const std::vector<std::string> &GetAllOps(std::string lib_name) const {
    auto iter = all_store_ops_.find(lib_name);
    if (iter == all_store_ops_.end()) {
      return all_ops_;
    }
    return iter->second;
  }

  bool CheckSupported(const std::string &type) {
    return op_creator_map_.find(type) != op_creator_map_.end();
  }

  OpFactory(const OpFactory &) = delete;
  OpFactory &operator=(const OpFactory &) = delete;
  OpFactory(OpFactory &&) = delete;
  OpFactory &operator=(OpFactory &&) = delete;

 private:
  OpFactory() = default;
  ~OpFactory() = default;

  // the op creator function map
  std::map<std::string, OP_CREATOR_FUNC> op_creator_map_;
  std::map<std::string, std::map<std::string, OP_CREATOR_FUNC>> lib_op_creator_map_;
  std::vector<std::string> all_ops_;
  std::map<std::string, vector<std::string>> all_store_ops_;
};

class GE_FUNC_VISIBILITY OpRegistrar {
 public:
  OpRegistrar(const std::string &type, const std::string &kernel_lib, const OP_CREATOR_FUNC &func) {
    OpFactory::Instance().RegisterCreator(type, kernel_lib, func);
  }
  ~OpRegistrar() = default;

  OpRegistrar(const OpRegistrar &) = delete;
  OpRegistrar &operator=(const OpRegistrar &) = delete;
  OpRegistrar(OpRegistrar &&) = delete;
  OpRegistrar &operator=(OpRegistrar &&) = delete;
};

#define REGISTER_OP_CREATOR(type, lib_name, clazz)                                                                     \
  std::shared_ptr<Op> Creator_##type##Op(const Node &node, RunContext &run_context) {                                  \
    return MakeShared<clazz>(node, run_context);                                                                       \
  }                                                                                                                    \
  OpRegistrar g_##type##Op_creator(#type, #lib_name, Creator_##type##Op)
}  // namespace st
}  // namespace ge

#endif  // GE_HOST_CPU_ENGINE_OPS_KERNEL_STORE_OP_OP_FACTORY_H_
