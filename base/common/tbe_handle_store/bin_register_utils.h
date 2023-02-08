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
#ifndef GE_HYBRID_KERNEL_AICORE_BIN_REGISTER_UTILS_H_
#define GE_HYBRID_KERNEL_AICORE_BIN_REGISTER_UTILS_H_

#include "graph/op_desc.h"
#include "ge/ge_api_types.h"

namespace ge {
struct AttrNameOfBinOnOp {
  std::string kTbeKernel;
  std::string kTvmMetaData;
  std::string kKernelNameSuffix;
  std::string kTvmMagicName;
};

class BinRegisterUtils {
  public:
  /**
   * @brief Call rtDevBinaryRegister to register bin
   * @param op_desc
   * @param stub_name
   *        Name of stub func. For op in SingleOp model, stub name is log_id + stub_func(defined in kernel_def).
   *        For op in hybrid model, stub name is stub_func(defined in kernel_def).
   * @param stub_func
   * @return Status
   */
   static Status RegisterBin(const OpDesc &op_desc, const std::string &stub_name, const AttrNameOfBinOnOp &attr_names,
                             void *&stub_func);

   /**
    * @brief Call rtRegisterAllKernel to register bin
    * @param op_desc
    *        input param
    * @param handle
    *        outpu param. After register bin, handle will be update.
    * @return Status
    */
   static Status RegisterBinWithHandle(const OpDesc &op_desc, const AttrNameOfBinOnOp &attr_names, void *&handle);
};
}  // namespace ge
#endif  // GE_HYBRID_KERNEL_AICORE_BIN_REGISTER_UTIL_H_
