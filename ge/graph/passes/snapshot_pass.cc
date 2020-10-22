/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "graph/passes/snapshot_pass.h"
#include <string>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/common/omg_util.h"

namespace ge {
Status SnapshotPass::Run(NodePtr &node) {
  if (node == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return FAILED;
  }
  string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    GELOGE(status_ret, "SnapshotPass get original type failed.");
    return status_ret;
  }
  if (type == SNAPSHOT) {
    return IsolateAndDeleteNode(node, {0});
  }
  return SUCCESS;
}
}  // namespace ge
