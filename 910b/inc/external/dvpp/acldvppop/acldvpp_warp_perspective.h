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

#ifndef ACLDVPP_WARP_PERSPECTIVE_H_
#define ACLDVPP_WARP_PERSPECTIVE_H_

#include "acldvpp_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
* @brief acldvppWarpPerspective 的第一阶段接口，根据具体的计算流程，计算workspace大小。
* @param [in] self: npu device侧的aclTensor，数据类型支持 FLOAT 和 UINT8，
*                   仅支持连续的Tensor，数据格式支持NCHW、NHWC。
* @param [in] matrix: aclFloatArray类型，3x4的透射变换矩阵。
* @param [in] interpolationMode: uint32_t，缩放插值算法，
                                 取值与对应缩放插值算法对应关系为：0: BILINEAR，1: NEAREST。
* @param [in] paddingMode：uint32_t 类型，该变量值与对应填充模式对应关系为 0：CONSTANT, 1：EDGE。
* @param [in] fill: aclFloatArray类型，表示在每个通道上填充的值。
* @param [in] out: npu device侧的aclTensor，数据类型支持 FLOAT 和 UINT8，
*                  仅支持连续的Tensor，数据格式支持NCHW、NHWC，且 数据格式 和 数据类型 需要与out一致。
* @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含了算子计算流程。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppWarpPerspectiveGetWorkspaceSize(
    const aclTensor* self, const aclFloatArray* matrix, uint32_t interpolationMode, uint32_t paddingMode,
    const aclFloatArray* fill, aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
* @brief acldvppWarpPerspective 的第二阶段接口，用于执行计算。
* @param [in] workspace: 在npu device侧申请的workspace内存起址。
* @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 acldvppWarpPerspectiveGetWorkspaceSize 获取。
* @param [in] executor: op执行器，包含了算子计算流程。
* @param [in] stream: acl stream流。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppWarpPerspective(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_WRAP_PERSPECTIVE_H_