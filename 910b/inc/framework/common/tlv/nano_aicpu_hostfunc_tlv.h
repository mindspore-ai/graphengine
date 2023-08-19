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

#ifndef INC_FRAMEWORK_COMMON_TLV_nano_aicpu_hostfunc_tlv_H_
#define INC_FRAMEWORK_COMMON_TLV_nano_aicpu_hostfunc_tlv_H_

#include "framework/common/tlv/tlv.h"

namespace ge {
#pragma pack(1)  // single-byte alignment

struct HostFuncHead {
  uint64_t len;
};

enum NANO_HOSTFUNC_L1_TLV_TYPE {
  L1_TLV_KERNEL_SO_NAME,
  L1_TLV_KERNEL_FUNC_NAME,
  L1_TLV_EXT_INFO,
  L1_TLV_TYPE_INPUT_TENSOR_DESC,
  L1_TLV_TYPE_OUTPUT_TENSOR_DESC,
  L1_TLV_ATTR
};

enum NANO_HOSTFUNC_ATTR_TYPE {
  L2_TLV_TYPE_SHAPE_DIMS,
  L2_TLV_TYPE_ATTR_INT,
  L2_TLV_TYPE_ATTR_FLOAT,
  L2_TLV_TYPE_ATTR_BOOL,
  L2_TLV_TYPE_ATTR_STRING,
  L2_TLV_TYPE_ATTR_LIST_INT,
  L2_TLV_TYPE_ATTR_LIST_FLOAT,
  L2_TLV_TYPE_ATTR_LIST_BOOL,
  L2_TLV_TYPE_ATTR_LIST_STRING,
  L2_TLV_TYPE_ATTR_LIST_LIST_INT,
  L2_TLV_TYPE_ATTR_LIST_LIST_FLOAT
};

struct ExtInfoTlv1 {
  uint32_t kernel_id;
  uint32_t session_id;
};

struct TensorDescTlv1 {
  uint32_t tensor_num;
  uint8_t param[0];
};

struct TensorDescParamTlv1 {
  uint32_t name_len;
  uint8_t name[0];
  int32_t dtype;
  int32_t format;
  int8_t mem_base_type; // data, weight, workspace
  uint64_t mem_offset;
  uint32_t dims_len;
  int64_t dims[0];
  uint32_t shape_range_len;
  int64_t shape_range[0];
};

struct AttrDescTlv1 {
  uint32_t num;
  uint8_t attr_param[0];
};

struct AttrParamVal {
  uint32_t type;
  uint32_t name_len;
  uint8_t name[0];
  uint32_t value_list_num;
  uint32_t value_list_info[0];
  uint32_t value_len;
  uint8_t value[0];
};

struct KernelSoName {
  uint8_t name[0];
};

struct KernelFuncName {
  uint8_t name[0];
};


/********************************************************************************************/
#pragma pack()  // Cancels single-byte alignment
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_TLV_nano_aicpu_hostfunc_tlv_H_