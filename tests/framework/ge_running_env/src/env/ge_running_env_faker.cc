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

#include <map>
#include <algorithm>
#include "external/ge/ge_api.h"
#include "opskernel_manager/ops_kernel_builder_manager.h"
#include "init/gelib.h"
#include "utility"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_default_running_env.h"
#include "ge_running_env/env_installer.h"
#include "op/fake_op_repo.h"

FAKE_NS_BEGIN

namespace {
OpsKernelManager& getKernelManger() {
  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  return instancePtr->OpsKernelManagerObj();
}

struct InitEnv {
  static InitEnv& GetInstance() {
    static InitEnv instance;
    return instance;
  }

  void reset(std::map<string, OpsKernelInfoStorePtr>& ops_kernel_info_stores,
             std::map<string, OpsKernelBuilderPtr>& builders) {
    std::set<string> remove_info_names;
    for (auto iter : ops_kernel_info_stores) {
      if (kernel_info_names.find(iter.first) == kernel_info_names.end()) {
        remove_info_names.insert(iter.first);
      }
    }
    for (auto info_name : remove_info_names) {
      ops_kernel_info_stores.erase(info_name);
      builders.erase(info_name);
    }
  }

 private:
  InitEnv() {
    for (auto iter : getKernelManger().GetAllOpsKernelInfoStores()) {
      kernel_info_names.insert(iter.first);
    }
  }

 private:
  std::set<string> kernel_info_names;
};
}  // namespace

GeRunningEnvFaker::GeRunningEnvFaker()
    : op_kernel_info_(const_cast<std::map<string, vector<OpInfo>>&>(getKernelManger().GetAllOpsKernelInfo())),
      ops_kernel_info_stores_(
        const_cast<std::map<string, OpsKernelInfoStorePtr>&>(getKernelManger().GetAllOpsKernelInfoStores())),
      ops_kernel_optimizers_(
        const_cast<std::map<string, GraphOptimizerPtr>&>(getKernelManger().GetAllGraphOptimizerObjs())),
      ops_kernel_builders_(const_cast<std::map<string, OpsKernelBuilderPtr>&>(
        OpsKernelBuilderManager::Instance().GetAllOpsKernelBuilders())) {
  Reset();
}

GeRunningEnvFaker& GeRunningEnvFaker::Reset() {
  InitEnv& init_env = InitEnv::GetInstance();
  FakeOpRepo::Reset();
  init_env.reset(ops_kernel_info_stores_, ops_kernel_builders_);
  flush();
  return *this;
}

void GeRunningEnvFaker::BackupEnv() { InitEnv::GetInstance(); }

GeRunningEnvFaker& GeRunningEnvFaker::Install(const EnvInstaller& installer) {
  installer.Install();
  installer.InstallTo(ops_kernel_info_stores_);
  installer.InstallTo(ops_kernel_optimizers_);
  installer.InstallTo(ops_kernel_builders_);
  flush();
  return *this;
}

void GeRunningEnvFaker::flush() {
  op_kernel_info_.clear();
  getKernelManger().GetOpsKernelInfo("");
}

GeRunningEnvFaker& GeRunningEnvFaker::InstallDefault() {
  Reset();
  GeDefaultRunningEnv::InstallTo(*this);
  return *this;
}

FAKE_NS_END
