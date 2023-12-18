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
#include "stream_executor.h"
#include "common/ge_types.h"
#include "common/ge_visibility.h"
#include "mem_allocator.h"
#include "memory/allocator_desc.h"
#include "exe_graph/lowering/lowering_opt.h"

namespace gert {
/**
 * Allocator 工厂类，创建Allocator
 */
class VISIBILITY_EXPORT AllocatorFactory {
 public:
  // 根据placement创建allocator
  static std::unique_ptr<ge::Allocator> Create(const TensorPlacement &placement);
 private:
  AllocatorFactory() = default;
};

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromFile(const ge::char_t *model_path, ge::graphStatus &error_code);

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           ge::graphStatus &error_code);

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           const ExecutorOption &executor_option,
                                                           ge::graphStatus &error_code);

VISIBILITY_EXPORT
std::unique_ptr<StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data, const void *weight_ptr,
                                                                const size_t weight_size, ge::graphStatus &error_code);

/**
 * 将ModelData加载为StreamExecutor，本函数等同为LoadStreamExecutorFromModelData(model_data, {}, error_cde);
 * @param model_data model_data从文件中读取后的内容
 * @param error_code 如果load失败（返回了空指针），那么本变量返回对应错误码
 * @return 成功时返回StreamExecutor的指针，失败时返回空指针。
 *         返回值类型是unique_ptr，因此返回的StreamExecutor生命周期由外部管理。
 */
VISIBILITY_EXPORT
std::unique_ptr<StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                ge::graphStatus &error_code);

/**
 * 将ModelData加载为StreamExecutor
 * @param model_data model_data从文件中读取后的内容
 * @param optimize_option 优化选项
 * @param error_code 如果load失败（返回了空指针），那么本变量返回对应错误码
 * @return 成功时返回StreamExecutor的指针，失败时返回空指针。
 *         返回值类型是unique_ptr，因此返回的StreamExecutor生命周期由外部管理。
 */

VISIBILITY_EXPORT
std::unique_ptr<StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                const LoweringOption &optimize_option,
                                                                ge::graphStatus &error_code);
VISIBILITY_EXPORT
ge::graphStatus IsDynamicModel(const void *const model, size_t model_size, bool &is_dynamic_model);
VISIBILITY_EXPORT
ge::graphStatus IsDynamicModel(const ge::char_t *model_path, bool &is_dynamic_model);

VISIBILITY_EXPORT
ge::graphStatus LoadDataFromFile(const ge::char_t *model_path, ge::ModelData &model_data);

VISIBILITY_EXPORT
std::unique_ptr<ge::Allocator> CreateExternalAllocator(const ge::AllocatorDesc * const allocatorDesc);
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_GERT_API_H_
