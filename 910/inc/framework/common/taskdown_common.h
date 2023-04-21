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

#ifndef INC_FRAMEWORK_COMMON_TASKDOWN_COMMON_H_
#define INC_FRAMEWORK_COMMON_TASKDOWN_COMMON_H_

#include "runtime/rt.h"

namespace ge {

constexpr int32_t CC_FUSION_OP_MAX = 32;

enum class ccStatus_t : uint32_t {
  CC_STATUS_SUCCESS = 0,         /**< succ */
  CC_STATUS_NOT_INITIALIZED = 1, /**< not init */
  CC_STATUS_ALLOC_FAILED = 2,    /**< alloc mem failed */
  CC_STATUS_BAD_PARAM = 3,       /**< para check failed */
  CC_STATUS_INTERNAL_ERROR = 4,  /**< internal error */
  CC_STATUS_KERNEL_ERROR = 5,    /**< kernel error */
  CC_STATUS_RUNTIME_ERROR = 6,   /**< runtime error */
  CC_STATUS_NOT_SUPPORTED = 7,   /**< unsupport error */
  CC_STATUS_INVALID_VALUE = 7,   /**< invalid value error for blas*/
  CC_STATUS_RESERVED      = 8,   /**< just for check */
};

enum class ccKernelType : uint32_t {
  CCE_AI_CORE = 0, /* cce aicore */
  CCE_AI_CPU = 1,  /* cce aicpu */
  TE = 2,          /* te operator */
  CUSTOMIZED = 3,  /* customized operator */
  TE_AI_CORE = 4,  /* te aicore operator */
  TE_AI_CPU = 5,   /* te aicpu operator */
  AI_CPU = 6,      /* aicpu */
  CUST_AI_CPU = 7, /* custom aicpu */
  HOST_CPU = 8,    /* host cpu */
  DVPP = 9,        /* dvpp */
  INVALID = 10000  /* unknown kernel type */
};

using ccOpContext = struct tagOpContext {
  ccKernelType kernelType;
  uint32_t opId;
  uint32_t kernelFuncId;
  uint32_t opIndex;
  uint32_t opCount;
  uint32_t opIndex2[CC_FUSION_OP_MAX];
  bool isFlowtable;
  uint16_t *argsOffset;
  uint32_t argsCount;
  uint64_t genDataBaseAddr;
  uint64_t genDataBaseSize;
  uint64_t genWeightBaseAddr;
  uint64_t genWeightBaseSize;
  uint64_t genVariableBaseAddr;
  uint64_t genVariableBaseSize;
  uint64_t l2ctrlSize;
};

enum class tagOpTensorFormat : uint32_t {
  OP_TENSOR_FORMAT_NC1HWC0 = 0,
  OP_TENSOR_FORMAT_ND,
  OP_TENSOR_FORMAT_RESERVED
};

enum class tagOpDataType : uint32_t {
  OP_DATA_FLOAT = 0,            /**< float type */
  OP_DATA_HALF,                 /**< fp16 type */
  OP_DATA_INT8,                 /**< int8 type */
  OP_DATA_INT32,                /**< int32 type */
  OP_DATA_UINT8,                /**< uint8 type */
  OP_DATA_HALF_UINT16_PROPOSAL, /**< mixed type for proposal */
  OP_DATA_RESERVED
};

// AICPU Tensor
using ccAICPUTensor = struct tagOpTensor {
  // real dim info
  tagOpTensorFormat format;
  tagOpDataType data_type;
  int32_t dim_cnt;
  int32_t mm;
  int32_t dim[8];
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_TASKDOWN_COMMON_H_
