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

#ifndef AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
#define AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
#include <memory>
#include "graph/compute_graph.h"
#include "graph/ge_error_codes.h"
#include "model_desc.h"
#include "runtime/stream.h"
#include "exe_graph/runtime/tensor.h"
#include "common/ge_visibility.h"
#include "exe_graph_resource_guard.h"
#include "exe_graph_executor.h"
#include "subscriber/executor_subscribers_scheduler.h"
#include "common/ge_types.h"
#include "mem_allocator.h"

namespace gert {
enum class ExecutorState { kInit, kLoaded };
enum SubExeGraphType { kInitExeGraph, kMainExeGraph, kDeInitExeGraph, kSubExeGraphTypeEnd };
inline const ge::char_t *GetSubExeGraphTypeStr(const SubExeGraphType type) {
  constexpr const ge::char_t *kSubExeGraphTypeStrs[kSubExeGraphTypeEnd] = {"Init", "Main", "DeInit"};
  return kSubExeGraphTypeStrs[type];
}

enum class ExecuteArgIndex { kExternalAllocator = -2, kStream, kEnd };

struct ModelExecuteArg {
  rtStream_t stream;
  ExternalAllocators *external_allocator;
  ModelExecuteArg() : stream(nullptr), external_allocator(nullptr) {}
  ModelExecuteArg(const rtStream_t stream_, ExternalAllocators *const external_allocator_ = nullptr)
      : stream(stream_), external_allocator(external_allocator_) {}
};
static_assert(std::is_standard_layout<ModelExecuteArg>::value, "The class ModelExecuteArg must be a POD");
class VISIBILITY_EXPORT ModelV2Executor {
 public:
  static std::unique_ptr<ModelV2Executor> Create(const ge::ComputeGraphPtr &root_graph, const ge::ModelData &model_data,
                                                 const std::shared_ptr<ge::GeRootModel> &root_model);
  static std::unique_ptr<ModelV2Executor> Create(const ge::ComputeGraphPtr &root_graph);

  ge::graphStatus Load();
  /**
   * 加载模型，本接口需要在模型执行前被调用。加载流程会完成模型的初始化、将重要数据拷贝到NPU等整个模型生命周期内仅需要执行一次的行为。
   * @param arg
   * 模型的执行参数，需要注意的是，此处传入的执行参数应该与Execute接口传入的执行参数具有相同的stream和allocator，
   *            否则在load完成后，外部需要调用流同步以保证不出现时序问题
   * @return 成功时返回`ge::GRAPH_SUCCESS`
   */
  ge::graphStatus Load(const ModelExecuteArg &arg);

  /**
   * 异步执行模型，本接口将模型异步下发到NPU执行，本接口返回不代表模型执行完成，用户需要手动调用流同步等待模型执行完成。
   * 调用本接口前，请确保已经调用`Load`接口
   *
   * 用户可以通过多种方式指定输出Tensor，其行为分别为：
   *
   * * 调用本接口前，用户自行申请了足量空间的输出内存，并通过输出Tensor传入：执行完成后，输出内容被写入到用户申请的输出Tensor。
   *   若用户申请的输出Tensor不够长，那么本接口返回失败。
   * * 用户生成了输出Tensor，但是没有申请输出内存，将不包含输出内存的Tensor传入：本接口内部主动申请输出内存，并将输出内存传出。
   *   若用户没有在arg中指定Allocator，那么本接口输出的内存生命周期与本Executor一致；
   *   如果用户在arg中传入了Allocator，那么输出内存将使用用户传入的Allocator申请
   *
   * 注意：
   *
   * 1. 本接口不支持并发调用
   * 2.
   * 如果外部指定了Allocator，那么建议Allocator应该与stream绑定，如果出现同一个allocator，匹配不同的stream多次调用Execute接口时，
   *    需要满足两个条件：不可以并发调用，在切换stream执行中间，需要对上一条stream做流同步
   * 3.
   * 若外部指定了Allocator，在模型执行完成前，不可以将Allocator中的内存归还给操作系统（即使这块内存已经由执行器归还给Allocator）
   *
   * @param arg 执行参数
   * @param inputs 网络的输入tensor，从调用本接口开始，到流同步等待本模型执行结束之前，用户需要保证传入的Tensor有效
   * @param input_num 输入tensor的数量
   * @param outputs 网络的输出tensor
   * @param output_num 输出tensor的数量
   * @return 成功时返回`ge::GRAPH_SUCCESS`
   */
  ge::graphStatus Execute(const ModelExecuteArg &arg, Tensor **inputs, size_t input_num, Tensor **outputs,
                          size_t output_num);
  ge::graphStatus ExecuteSync(Tensor **inputs, size_t input_num, Tensor **outputs, size_t output_num);
  ge::graphStatus UnLoad();

  const ModelDesc &GetModelDesc() const;
  void SetModelDesc(ModelDesc *model_desc);
  ExeGraphExecutor *GetExeGraphExecutor(const SubExeGraphType type) {
    if (type >= kSubExeGraphTypeEnd) {
      return nullptr;
    }
    return &graphs_[static_cast<size_t>(type)];
  }
  ExecutorSubscribersScheduler &GetSubscribers();
  const ExecutorSubscribersScheduler &GetSubscribers() const;

  ModelV2Executor(const ModelV2Executor &) = delete;
  ModelV2Executor(ModelV2Executor &&) = delete;
  ModelV2Executor &operator=(const ModelV2Executor &) = delete;
  ModelV2Executor &operator=(ModelV2Executor &&) = delete;

 private:
  friend class ModelV2ExecutorBuilder;
  friend class ModelV2ExecutorTestHelper;
  ModelV2Executor();

 private:
  ResourceGuard resource_guard_;
  std::array<ExeGraphExecutor, kSubExeGraphTypeEnd> graphs_;
  ModelDesc *model_desc_ = nullptr;
  rtStream_t default_stream_ = nullptr;
  ExecutorSubscribersScheduler subscribers_;
  ExecutorState state_ = ExecutorState::kInit;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
