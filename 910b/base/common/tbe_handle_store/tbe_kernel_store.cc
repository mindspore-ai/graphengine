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

#include "common/tbe_handle_store/tbe_kernel_store.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace {
const std::string kTbeMixCubeCoreType = "MIX_AIC";
const std::string kTbeMixVectorCoreType = "MIX_AIV";
const std::string kTbeMixCubeTBEKernelNamePrefix = "_mix_aic";
const std::string kTbeMixVectorTBEKernelNamePrefix = "_mix_aiv";
}

namespace ge {
TBEKernelStore::TBEKernelStore() : KernelStore() {}

void TBEKernelStore::AddTBEKernel(const TBEKernelPtr &kernel) {
  AddKernel(kernel);
}

void TBEKernelStore::LoadTBEKernelBinToOpDesc(const std::shared_ptr<ge::OpDesc> &op_desc) const {
  if (op_desc != nullptr) {
    std::string core_type;
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
    if ((core_type == kTbeMixCubeCoreType) || (core_type == kTbeMixVectorCoreType)) {
      const auto LoadMixTbeKernel = [&op_desc, this](const std::string &prefix) {
        std::string kernel_name;
        (void)AttrUtils::GetStr(op_desc, prefix + ATTR_NAME_TBE_KERNEL_NAME, kernel_name);
        const auto &kernel = FindKernel(kernel_name);
        if (kernel != nullptr) {
          const std::string ext_kernel_name = prefix + std::string(OP_EXTATTR_NAME_TBE_KERNEL);
          (void)op_desc->SetExtAttr(ext_kernel_name, kernel);
          GELOGI("LoadTBEKernelBinToOpDesc: Set attr %s for tbe kernel: [%s, %zu] successfully",
                 ext_kernel_name.c_str(), kernel->GetName().c_str(), kernel->GetBinDataSize());
        }
      };
      std::vector<std::string> names_prefix;
      (void)AttrUtils::GetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
      for (const auto &prefix : names_prefix) {
        LoadMixTbeKernel(prefix);
      }
    } else {
      std::string kernel_name;
      const auto status = AttrUtils::GetStr(op_desc, ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD, kernel_name);
      const auto &kernel_bin = FindKernel(status ? kernel_name : op_desc->GetName());
      if (kernel_bin != nullptr) {
        GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin),
                        GELOGW("LoadKernelTBEBinToOpDesc: SetExtAttr for kernel_bin failed");)
        GELOGI("Load tbe kernel:%s, %zu", kernel_bin->GetName().c_str(), kernel_bin->GetBinDataSize());

        std::string atomic_kernel_name;
        (void)AttrUtils::GetStr(op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, atomic_kernel_name);
        if (!atomic_kernel_name.empty()) {
          GELOGI("Get atomic kernel name is %s.", atomic_kernel_name.c_str());
          const auto atomic_kernel_bin = FindKernel(atomic_kernel_name);
          GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, atomic_kernel_bin),
                          GELOGW("LoadKernelTBEBinToOpDesc: SetExtAttr for atomic kernel_bin failed");)
        }
      }
    }
  }
}
}  // namespace ge
