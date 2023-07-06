/**
* @file acl_rt_allocator.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_ACL_ACL_RT_ALLOCATOR_H_
#define INC_EXTERNAL_ACL_ACL_RT_ALLOCATOR_H_

#include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *aclrtAllocatorDesc;
typedef void *aclrtAllocator;
typedef void *aclrtAllocatorBlock;
typedef void *aclrtAllocatorAddr;

typedef void *(*aclrtAllocatorAllocFunc)(aclrtAllocator allocator, size_t size);
typedef void (*aclrtAllocatorFreeFunc)(aclrtAllocator allocator, aclrtAllocatorBlock block);
typedef void *(*aclrtAllocatorAllocAdviseFunc)(aclrtAllocator allocator, size_t size, aclrtAllocatorAddr addr);
typedef void *(*aclrtAllocatorGetAddrFromBlockFunc)(aclrtAllocatorBlock block);


ACL_FUNC_VISIBILITY aclrtAllocatorDesc aclrtAllocatorCreateDesc();

ACL_FUNC_VISIBILITY aclError aclrtAllocatorDestroyDesc(aclrtAllocatorDesc allocatorDesc);

ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetObjToDesc(aclrtAllocatorDesc allocatorDesc, aclrtAllocator allocator);

ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetAllocFuncToDesc(aclrtAllocatorDesc allocatorDesc,
                                                              aclrtAllocatorAllocFunc func);
ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetFreeFuncToDesc(aclrtAllocatorDesc allocatorDesc,
                                                            aclrtAllocatorFreeFunc func);

ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetAllocAdviseFuncToDesc(aclrtAllocatorDesc allocatorDesc,
                                                                    aclrtAllocatorAllocAdviseFunc func);

ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetGetAddrFromBlockFuncToDesc(aclrtAllocatorDesc allocatorDesc,
                                                                         aclrtAllocatorGetAddrFromBlockFunc func);

ACL_FUNC_VISIBILITY aclError aclrtAllocatorRegister(aclrtStream stream, aclrtAllocatorDesc allocatorDesc);

ACL_FUNC_VISIBILITY aclError aclrtAllocatorUnregister(aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_RT_ALLOCATOR_H_
