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
#ifndef GE_COMMMON_RUNTIME_TILING_KERNEL_CONTEXT_BUILDER_H_
#define GE_COMMMON_RUNTIME_TILING_KERNEL_CONTEXT_BUILDER_H_

#include "graph/node.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context_builder.h"
#include "register/op_impl_space_registry.h"

namespace gert {
class TilingContextBuilder {
 public:
  TilingContextBuilder &CompileInfo(void *compile_info);
  TilingContextBuilder &PlatformInfo(void *platform_info);
  TilingContextBuilder &TilingData(void *tiling_data);
  TilingContextBuilder &Workspace(ContinuousVector *workspace);
  TilingContextBuilder &SpaceRegistry(const gert::OpImplSpaceRegistryPtr &space_registry);
  KernelContextHolder Build(const ge::Operator &op);

 private:
  ge::graphStatus GetDependInputTensorAddr(const ge::Operator &op, const size_t input_idx, TensorAddress &address);
  ge::graphStatus BuildRtTensor(const ge::GeTensorDesc &tensor_desc, const TensorAddress address,
                                std::unique_ptr<uint8_t[]> &rt_tensor_holder) const;
  ge::graphStatus BuildRTInputTensors(const ge::Operator &op);
  ge::graphStatus BuildRTOutputShapes(const ge::Operator &op);

  void *compile_info_{nullptr};
  void *platform_info_{nullptr};
  std::vector<std::unique_ptr<ge::Tensor>> depend_ge_tensor_holders_;
  std::vector<std::unique_ptr<uint8_t[]>> rt_tensor_holders_;
  std::vector<void *> outputs_ {TilingContext::kOutputNum};
  KernelRunContextBuilder base_builder_;
  gert::OpImplSpaceRegistryPtr space_registry_;
};

class AtomicTilingContextBuilder {
 public:
  AtomicTilingContextBuilder &CompileInfo(void *compile_info);
  AtomicTilingContextBuilder &CleanWorkspaceSizes(ContinuousVector *workspace_sizes);
  AtomicTilingContextBuilder &CleanOutputSizes(const std::vector<int64_t> &output_sizes);
  AtomicTilingContextBuilder &TilingData(void *tiling_data);
  AtomicTilingContextBuilder &Workspace(ContinuousVector *workspace);
  KernelContextHolder Build(const ge::Operator &op);

 private:
  void *compile_info_{nullptr};
  void *worksapce_sizes_{nullptr};
  std::vector<int64_t> clean_output_sizes_;
  std::vector<void *> outputs_ {TilingContext::kOutputNum};
  KernelRunContextBuilder base_builder_;
};
}  // namespace gert
#endif // GE_COMMMON_RUNTIME_TILING_KERNEL_CONTEXT_BUILDER_H_
