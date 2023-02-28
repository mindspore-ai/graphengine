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

namespace gert {
struct OptimizeOption {
  /**
   * 是否相信用户传入的输出tensor上的shape，如果开启了本选项，可以省去计算图上输出节点的InferShape，提升一点Host调度性能。
   * 与此同时，也会损失掉对外部传入的输出Tensor Shape、TensorData长度的校验能力。
   *
   * 约束：
   * 1. 如果一个节点有多个输出，并且部分输出并不是网络的输出，
   *    那么这个节点的InferShape不会被省掉，体现为在这个节点上，本选项会被忽略。
   * 2. 如果一个节点没有InferShape函数，例如第三类、第四类算子，
   *    需要从Device拷贝回Shape，那么在这个节点上，本选项会被忽略。
   * 3. 本选项是个加载时选项，一旦选定后，意味着后续本model的每次调用都需要用户传入输出shape，否则可能会导致执行失败
   */
  bool trust_shape_on_out_tensor = false;

  /**
   * 总是零拷贝开关，默认关闭。如果本开关打开，含义是外部调用者总是保证会正确地申请输出内存，包含：
   * 1. 申请的输出内存大于等于输出shape所以计算出的Tensor大小
   * 2. 输出内存的placement正确
   *
   * 打开本开关后，可以提升一点Host调度性能。与此同时，对于零拷贝失效的回退处理将不再进行，
   * 在外部申请的输出内存错误、或未申请输出内存时，执行报错。
   */
  bool always_zero_copy = false;

  /**
   * 二进制兼容保留字段，增加option时，对应缩减删除reserved长度
   */
  uint8_t reserved[6U + 8U] = {0U};
};

/**
 * Allocator 工厂类，创建Allocator
 */
class VISIBILITY_EXPORT AllocatorFactory {
 public:
  // 根据placement创建allocator
  static std::unique_ptr<memory::MemAllocator> Create(const TensorPlacement &placement);
 private:
  AllocatorFactory() = default;
};

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromFile(const ge::char_t *model_path, ge::graphStatus &error_code);

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
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
                                                                const OptimizeOption &optimize_option,
                                                                ge::graphStatus &error_code);
VISIBILITY_EXPORT
ge::graphStatus IsDynamicModel(const void *const model, size_t model_size, bool &is_dynamic_model);
VISIBILITY_EXPORT
ge::graphStatus IsDynamicModel(const ge::char_t *model_path, bool &is_dynamic_model);

VISIBILITY_EXPORT
ge::graphStatus LoadDataFromFile(const ge::char_t *model_path, ge::ModelData &model_data);
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_GERT_API_H_
