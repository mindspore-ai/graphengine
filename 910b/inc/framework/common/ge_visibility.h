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

#ifndef AIR_CXX_INC_FRAMEWORK_COMMON_GE_VISIBILITY_H_
#define AIR_CXX_INC_FRAMEWORK_COMMON_GE_VISIBILITY_H_

#if defined(_MSC_VER)
#define VISIBILITY_EXPORT _declspec(dllexport)
#define VISIBILITY_HIDDEN _declspec(dllimport)
#else
#define VISIBILITY_EXPORT __attribute__((visibility("default")))
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

#endif  // AIR_CXX_INC_FRAMEWORK_COMMON_GE_VISIBILITY_H_
