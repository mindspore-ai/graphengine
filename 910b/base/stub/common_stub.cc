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
#include "local_engine/engine/host_cpu_engine.h"
#include "common/plugin/runtime_plugin_loader.h"
#include "common/plugin/opp_so_manager.h"

namespace ge {
HostCpuEngine &HostCpuEngine::GetInstance() {
  static HostCpuEngine instance;
  return instance;
}

ge::Status HostCpuEngine::Initialize(const std::string &path_base) {
  (void)path_base;
  initialized_ = true;
  return SUCCESS;
}

void HostCpuEngine::Finalize() const {
  GELOGI("start HostCpuEngine::Finalize");
}

Status HostCpuEngine::PrepareInputs(const ge::ConstOpDescPtr &op_desc,
                                    const std::vector<ConstGeTensorPtr> &inputs,
                                    std::map<std::string, const Tensor> &named_inputs) {
  (void)op_desc;
  (void)inputs;
  (void)named_inputs;
  return SUCCESS;
}

Status HostCpuEngine::PrepareOutputs(const ge::ConstOpDescPtr &op_desc,
                                     std::vector<GeTensorPtr> &outputs,
                                     std::map<std::string, Tensor> &named_outputs) {
  (void)op_desc;
  (void)outputs;
  (void)named_outputs;
  return SUCCESS;
}

Status HostCpuEngine::RunInternal(const ge::OpDescPtr &op_desc, HostCpuOp &op_kernel,
                                  const std::map<std::string, const Tensor> &named_inputs,
                                  std::map<std::string, Tensor> &named_outputs) {
  (void)op_desc;
  (void)op_kernel;
  (void)named_inputs;
  (void)named_outputs;
  return SUCCESS;
}

Status HostCpuEngine::Run(const NodePtr &node, HostCpuOp &kernel, const std::vector<ConstGeTensorPtr> &inputs,
                          std::vector<GeTensorPtr> &outputs) {
  (void)node;
  (void)kernel;
  (void)inputs;
  (void)outputs;
  return SUCCESS;
}

Status HostCpuEngine::ListSoFiles(const std::string &base_dir, std::vector<std::string> &names) {
  (void)base_dir;
  (void)names;
  return SUCCESS;
}

bool HostCpuEngine::IsSoFile(const std::string &file_name) {
  (void)file_name;
  return true;
}

Status HostCpuEngine::LoadLibs(std::vector<std::string> &lib_paths) {
  (void)lib_paths;
  return SUCCESS;
}

Status HostCpuEngine::LoadLib(const std::string &lib_path) {
  (void)lib_path;
  return SUCCESS;
}

Status HostCpuEngine::GetEngineRealPath(std::string &path) {
  (void)path;
  return SUCCESS;
}

RuntimePluginLoader &RuntimePluginLoader::GetInstance() {
  static RuntimePluginLoader instance;
  return instance;
}

graphStatus RuntimePluginLoader::Initialize(const std::string &path_base) {
  (void)path_base;
  return GRAPH_SUCCESS;
}

void OppSoManager::LoadOppPackage() {}

OppSoManager &OppSoManager::GetInstance() {
  static OppSoManager instance;
  return instance;
}
} // namespace ge
