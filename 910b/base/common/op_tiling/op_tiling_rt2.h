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

#ifndef GE_COMMON_TILING_OP_TILING_RT2_H_
#define GE_COMMON_TILING_OP_TILING_RT2_H_

#include "graph/operator.h"
#include "graph/op_desc.h"
#include "register/op_tiling_registry.h"
#include "platform/platform_info.h"
#include "exe_graph/runtime/kernel_context.h"
#include "runtime/kernel.h"
#include "register/op_impl_space_registry.h"

namespace optiling {
using OutputsConvertorFun = std::function<ge::graphStatus(gert::KernelContext *kernel_context)>;
bool EnableRt2Tiling(const ge::OpDescPtr &op_desc);
bool EnableAtomicRt2Tiling(const ge::OpDescPtr &op_desc);
ge::graphStatus RtParseAndTiling(const ge::Operator &op, const char_t * const compile_info,
                                 const fe::PlatFormInfos &platform_infos, const OutputsConvertorFun &callback,
                                 const gert::OpImplSpaceRegistryPtr &space_registry);
ge::graphStatus AicoreRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                       OpRunInfoV2 &run_info);
ge::graphStatus AtomicRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                       OpRunInfoV2 &run_info);
ge::graphStatus SoftSyncOpRtParseAndTiling(const ge::Operator &op, fe::PlatFormInfos &platform_infos,
                                           OpRunInfoV2 &run_info, const gert::OpImplSpaceRegistryPtr &space_registry);
ge::graphStatus FftsRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                     std::vector<OpRunInfoV2> &op_run_infos);
}  // namespace optiling

#endif  // GE_COMMON_TILING_OP_TILING_RT2_H_
