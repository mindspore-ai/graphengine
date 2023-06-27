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

#ifndef INC_FRAMEWORK_MEMORY_ALLOCATOR_DESC_H_
#define INC_FRAMEWORK_MEMORY_ALLOCATOR_DESC_H_
#include <cstdlib>

namespace ge {
using AllocFunc = void *(*)(void *obj, size_t size);
using FreeFunc = void (*)(void *obj, void *block);
using AllocAdviseFunc = void *(*)(void *obj, size_t size, void *addr);
using GetAddrFromBlockFunc = void *(*)(void *block);

using AllocatorDesc = struct AllocatorDesc {
    AllocFunc alloc_func;
    FreeFunc free_func;
    AllocAdviseFunc alloc_advise_func;
    GetAddrFromBlockFunc get_addr_from_block_func;

    void *obj;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_MEMORY_ALLOCATOR_DESC_H_
