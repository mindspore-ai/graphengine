/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef GE_GE_LOCAL_ENGINE_ENGINE_HOST_CPU_ENGINE_H_
#define GE_GE_LOCAL_ENGINE_ENGINE_HOST_CPU_ENGINE_H_

#include <mutex>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/fmk_error_codes.h"
#include "graph/node.h"
#include "external/graph/operator.h"
#include "external/../register/register.h"

namespace ge {
class HostCpuEngine {
 public:
  ~HostCpuEngine() = default;

  static HostCpuEngine &GetInstance();

  Status Initialize(const std::string &path_base);

  void Finalize() const;

  static Status Run(const NodePtr &node, HostCpuOp &kernel, const std::vector<ConstGeTensorPtr> &inputs,
                    std::vector<GeTensorPtr> &outputs);

  void *GetConstantFoldingHandle() const { return constant_folding_handle_; }

 private:
  HostCpuEngine() = default;

  Status LoadLibs(std::vector<std::string> &lib_paths);

  Status LoadLib(const std::string &lib_path);

  static Status GetEngineRealPath(std::string &path);

  static Status ListSoFiles(const std::string &base_dir, std::vector<std::string> &names);

  static bool IsSoFile(const std::string &file_name);

  static Status PrepareInputs(const ConstOpDescPtr &op_desc, const std::vector<ConstGeTensorPtr> &inputs,
                              std::map<std::string, const Tensor> &named_inputs);

  static Status PrepareOutputs(const ConstOpDescPtr &op_desc, std::vector<GeTensorPtr> &outputs,
                               std::map<std::string, Tensor> &named_outputs);

  static Status RunInternal(const OpDescPtr &op_desc, HostCpuOp &op_kernel,
                            const std::map<std::string, const Tensor> &named_inputs,
                            std::map<std::string, Tensor> &named_outputs);

  std::mutex mu_;
  std::vector<void *> lib_handles_;
  void *constant_folding_handle_ = nullptr;
  bool initialized_ = false;
};
}  // namespace ge
#endif  // GE_GE_LOCAL_ENGINE_ENGINE_HOST_CPU_ENGINE_H_
