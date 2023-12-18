/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef INC_FRAMEWORK_COMMON_TLV_pre_model_desc_H_
#define INC_FRAMEWORK_COMMON_TLV_pre_model_desc_H_

namespace ge {
#pragma pack(1)  // single-byte alignment

enum KERNEL_ARG_UPADTE_TYPE {
  KERNEL_ARG_UPDATE_TYPE_ADDR,
  KERNEL_ARG_UPDATE_TYPE_TS,
  KERNEL_ARG_UPDATE_TYPE_P2P,
  KERNEL_ARG_UPDATE_TYPE_CPU_KERNEL_ARGS,
  KERNEL_ARG_UPDATE_TYPE_SESSIONID,
  KERNEL_ARG_UPDATE_TYPE_KERNELID,
  KERNEL_ARG_UPDATE_TYPE_EVENTID,
  KERNEL_ARG_UPDATE_TYPE_BUFF
};
enum KERNEL_ARG_UPADTE_ADDR_TYPE {
  KERNEL_ARG_UPADTE_ADDR_TYPE_ARGS,
  KERNEL_ARG_UPADTE_ADDR_TYPE_WORKSPACE,
  KERNEL_ARG_UPADTE_ADDR_TYPE_WEIGHT,
  KERNEL_ARG_UPADTE_ADDR_TYPE_L1,
  KERNEL_ARG_UPADTE_ADDR_TYPE_TS,
  KERNEL_ARG_UPADTE_ADDR_TYPE_P2P,
  KERNEL_ARG_UPADTE_ADDR_TYPE_VAR,
  KERNEL_ARG_UPADTE_ADDR_TYPE_KERNEL_BIN,
  KERNEL_ARG_UPADTE_ADDR_TYPE_BUFF
};

/********************************************************************************************/
#pragma pack()  // Cancels single-byte alignment
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_TLV_pre_model_desc_H_