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
#ifndef OP_API_OP_API_COMMON_INC_EXTERNAL_ACL_META_H
#define OP_API_OP_API_COMMON_INC_EXTERNAL_ACL_META_H

#include <cstdint>
#include <cstdlib>
#include <acl/acl_base.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define ACL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif
#endif

#ifdef __GNUC__
#define ACL_DEPRECATED __attribute__((deprecated))
#define ACL_DEPRECATED_MESSAGE(message) __attribute__((deprecated(message)))
#elif defined(_MSC_VER)
#define ACL_DEPRECATED __declspec(deprecated)
#define ACL_DEPRECATED_MESSAGE(message) __declspec(deprecated(message))
#else
#define ACL_DEPRECATED
#define ACL_DEPRECATED_MESSAGE(message)
#endif

#ifndef ACLNN_META
#define ACLNN_META
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

typedef int32_t aclnnStatus;
constexpr aclnnStatus OK = 0;
#endif

ACL_FUNC_VISIBILITY aclTensor *aclCreateTensor(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                                               const int64_t *stride, int64_t offset, aclFormat format,
                                               const int64_t *storageDims, uint64_t storageDimsNum,
                                               void *tensorData);

ACL_FUNC_VISIBILITY aclScalar *aclCreateScalar(void *value, aclDataType dataType);
ACL_FUNC_VISIBILITY aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);
ACL_FUNC_VISIBILITY aclFloatArray *aclCreateFloatArray(const float *value, uint64_t size);
ACL_FUNC_VISIBILITY aclBoolArray *aclCreateBoolArray(const bool *value, uint64_t size);
ACL_FUNC_VISIBILITY aclTensorList *aclCreateTensorList(const aclTensor *const *value, uint64_t size);

ACL_FUNC_VISIBILITY aclnnStatus aclDestroyTensor(const aclTensor *tensor);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyScalar(const aclScalar *scalar);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyIntArray(const aclIntArray *array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyFloatArray(const aclFloatArray *array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyBoolArray(const aclBoolArray *array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyTensorList(const aclTensorList *array);
ACL_FUNC_VISIBILITY aclnnStatus aclGetViewShape(const aclTensor *tensor, int64_t **viewDims, uint64_t *viewDimsNum);
ACL_FUNC_VISIBILITY aclnnStatus aclGetStorageShape(const aclTensor *tensor,
                                                   int64_t **storageDims,
                                                   uint64_t *storageDimsNum);
ACL_FUNC_VISIBILITY aclnnStatus aclGetViewStrides(const aclTensor *tensor,
                                                  int64_t **stridesValue,
                                                  uint64_t *stridesNum);
ACL_FUNC_VISIBILITY aclnnStatus aclGetViewOffset(const aclTensor *tensor, int64_t *offset);
ACL_FUNC_VISIBILITY aclnnStatus aclGetFormat(const aclTensor *tensor, aclFormat *format);
ACL_FUNC_VISIBILITY aclnnStatus aclGetDataType(const aclTensor *tensor, aclDataType *dataType);
ACL_FUNC_VISIBILITY aclnnStatus aclGetIntArraySize(const aclIntArray *array, uint64_t *size);
ACL_FUNC_VISIBILITY aclnnStatus aclGetFloatArraySize(const aclFloatArray *array, uint64_t *size);
ACL_FUNC_VISIBILITY aclnnStatus aclGetBoolArraySize(const aclBoolArray *array, uint64_t *size);
ACL_FUNC_VISIBILITY aclnnStatus aclGetTensorListSize(const aclTensorList *tensorList, uint64_t *size);

#ifdef __cplusplus
}
#endif

#endif // OP_API_OP_API_COMMON_INC_EXTERNAL_ACL_META_H
