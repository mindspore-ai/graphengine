/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "node_bin_selector_factory.h"

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
namespace ge {
NodeBinSelectorFactory &NodeBinSelectorFactory::GetInstance() {
  static NodeBinSelectorFactory factory;
  return factory;
}
NodeBinSelector *NodeBinSelectorFactory::GetNodeBinSelector(NodeBinMode node_bin_type) {
  if (node_bin_type < kNodeBinModeEnd) {
    return selectors_[node_bin_type].get();
  } else {
    GELOGE(FAILED, "node bin type %u is out of range.", static_cast<uint32_t>(node_bin_type));
    return nullptr;
  }
}
NodeBinSelectorFactory::NodeBinSelectorFactory() = default;
}