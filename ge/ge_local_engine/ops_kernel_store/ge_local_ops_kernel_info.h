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

#ifndef GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_GE_LOCAL_OPS_KERNEL_INFO_H_
#define GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_GE_LOCAL_OPS_KERNEL_INFO_H_

#include <map>
#include <string>
#include <vector>

#include "common/opskernel/ops_kernel_info_store.h"

namespace ge {
namespace ge_local {
class GeLocalOpsKernelInfoStore : public OpsKernelInfoStore {
 public:
  GeLocalOpsKernelInfoStore() = default;

  ~GeLocalOpsKernelInfoStore() override = default;

  /**
   * Initialize related resources of the ge local kernelinfo store
   * @return status whether this operation success
   */
  Status Initialize(const std::map<std::string, std::string> &options) override;

  /**
   * Release related resources of the ge local kernel info store
   * @return status whether this operation success
   */
  Status Finalize() override;

  /**
   * Check to see if an operator is fully supported or partially supported.
   * @param op_desc OpDesc information
   * @param reason unsupported reason
   * @return bool value indicate whether the operator is fully supported
   */
  bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override;

  /**
   * Returns the full operator information.
   * @param infos reference of a map,
   *        contain operator's name and detailed information
   */
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override;

  /**
   * Calc the running size of Operator,
   * then GE will alloc the mem size from runtime
   * @param ge_node Node information
   * @return status whether this operation success
   */
  Status CalcOpRunningParam(ge::Node &ge_node) override;

  /**
   * call the runtime's interface to generate the task
   * @param node Node information
   * @param context run context info
   * @return status whether this operation success
   */
  Status GenerateTask(const ge::Node &ge_node, ge::RunContext &context, std::vector<domi::TaskDef> &tasks) override;

  /**
   * Create session
   * @param session_options Session Options
   * @return status whether this operation success
   */
  Status CreateSession(const std::map<std::string, std::string> &session_options) override;

  /**
   * Destroy session
   * @param session_options Session Options
   * @return status whether this operation success
   */
  Status DestroySession(const std::map<std::string, std::string> &session_options) override;

  // Copy prohibited
  GeLocalOpsKernelInfoStore(const GeLocalOpsKernelInfoStore &ops_kernel_store) = delete;

  // Move prohibited
  GeLocalOpsKernelInfoStore(const GeLocalOpsKernelInfoStore &&ops_kernel_store) = delete;

  // Copy prohibited
  GeLocalOpsKernelInfoStore &operator=(const GeLocalOpsKernelInfoStore &ops_kernel_store) = delete;

  // Move prohibited
  GeLocalOpsKernelInfoStore &operator=(GeLocalOpsKernelInfoStore &&ops_kernel_store) = delete;

 private:
  /**
   * Calc memSize for constant which type is DT_STRING.
   * @param op_desc OpDesc information
   * @param mem_size output size
   * @return whether this operation success
   */
  Status CalcConstantStrMemSize(const OpDescPtr &op_desc, int64_t &mem_size);

  // store op name and OpInfo key-value pair
  std::map<std::string, ge::OpInfo> op_info_map_;
};
}  // namespace ge_local
}  // namespace ge

#endif  // GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_GE_LOCAL_OPS_KERNEL_INFO_H_
