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

#ifndef GE_GRAPH_MANAGER_TRANS_VAR_DATA_UTILS_H_
#define GE_GRAPH_MANAGER_TRANS_VAR_DATA_UTILS_H_

#include <string>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "runtime/context.h"
#include "graph_var_manager.h"

namespace ge {
class TransVarDataUtils {
 public:
  static ge::Status SyncVarData2BroadCast(const string &var_name, const ge::GeTensorDesc &src_tensor_desc,
                                          uint8_t *dst_addr, int64_t dst_addr_size, uint64_t session_id_);
  static ge::Status SyncBroadCastData2Var(uint8_t *src_addr, int64_t src_addr_size, const string &var_name,
                                          const ge::GeTensorDesc &dst_tensor_desc, uint64_t session_id_);

  static ge::Status TransAllVarData(const std::vector<NodePtr> &variable_nodes, uint64_t session_id,
                                    rtContext_t context, uint32_t graph_id, uint32_t thread_num = 16);

  static ge::Status CopyVarData(const ComputeGraphPtr &compute_graph, uint64_t session_id, uint32_t device_id);

 private:
  static ge::Status SyncTensorToHost(const string &var_name, const ge::GeTensorDesc &src_tensor_desc,
                                     uint8_t **host_addr, int64_t &addr_size, uint64_t session_id_);
  static ge::Status SyncTensorToDevice(const string &var_name, const uint8_t *host_addr, uint32_t addr_size,
                                       const ge::GeTensorDesc &dst_tensor_desc, uint64_t session_id_);
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_TRANS_VAR_DATA_UTILS_H_
