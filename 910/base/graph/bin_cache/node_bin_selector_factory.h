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

#ifndef AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_FACTORY_H_
#define AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_FACTORY_H_
#include <array>
#include <memory>
#include "node_bin_selector.h"
#include "bin_cache_def.h"
#include "common/plugin/ge_util.h"

namespace ge {
class NodeBinSelectorFactory {
 public:
  static NodeBinSelectorFactory &GetInstance();
  NodeBinSelector *GetNodeBinSelector(const fuzz_compile::NodeBinMode node_bin_type);

  template <typename T>
  class Register {
   public:
    explicit Register(const fuzz_compile::NodeBinMode node_bin_mode) {
      if (node_bin_mode < fuzz_compile::kNodeBinModeEnd) {
        NodeBinSelectorFactory::GetInstance().selectors_[node_bin_mode] = MakeUnique<T>();
      }
    }

    template<typename... Args>
    Register(const fuzz_compile::NodeBinMode node_bin_mode, Args... args) {
      if (node_bin_mode < fuzz_compile::kNodeBinModeEnd) {
        NodeBinSelectorFactory::GetInstance().selectors_[node_bin_mode] = MakeUnique<T>(args...);
      }
    }
  };

 private:
  NodeBinSelectorFactory();
  std::array<std::unique_ptr<NodeBinSelector>, fuzz_compile::kNodeBinModeEnd> selectors_;
};
}  // namespace ge

#define REGISTER_BIN_SELECTOR(node_bin_mode, T, ...) \
  static ge::NodeBinSelectorFactory::Register<T> Register_##T##_(node_bin_mode, ##__VA_ARGS__)
#endif // AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_FACTORY_H_
