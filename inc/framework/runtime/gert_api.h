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

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_GERT_API_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_GERT_API_H_
#include "model_v2_executor.h"
#include "common/ge_types.h"
#include "common/ge_visibility.h"
#include "mem_allocator.h"

namespace gert {
VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromFile(const char *model_path, ge::graphStatus &error_code);

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           ge::graphStatus &error_code);
VISIBILITY_EXPORT
ge::graphStatus IsDynamicModel(const char *model_path, bool &is_dynamic_model);

VISIBILITY_EXPORT
std::unique_ptr<ExternalAllocators> CreateSingleOpAllocator();

}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_GERT_API_H_
