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

#ifndef ACLDVPP_ROTATE_H_
#define ACLDVPP_ROTATE_H_

#include "acldvpp_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
* @brief acldvppRotate的第一阶段接口，根据具体的计算流程，计算workspace大小。
* @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT和UINT8，
*                   仅支持连续的Tensor，数据格式支持NCHW、NHWC。
* @param [in] angle: float类型，旋转的角度，以度为单位，逆时针方向。
* @param [in] interpolationMode: uint32_t，缩放插值算法，
                                 取值与对应缩放插值算法对应关系为：0: BILINEAR，1: NEAREST。
* @param [in] expand: bool类型，若为True，将扩展图像尺寸大小使其足以容纳整个旋转图像；
*                     若为False，则保持图像尺寸大小不变。请注意，扩展时将假设图像为中心旋转且未进行平移。
* @param [in] center: aclIntArray类型， 旋转中心，以图像左上角为原点，旋转中心的位置按照 (宽度, 高度) 格式指定。
* @param [in] paddingMode：uint32_t 类型，该变量值与对应填充模式对应关系为 0：CONSTANT, 1：EDGE。
* @param [in] fill: aclFloatArray类型，表示在每个通道上填充的值。
* @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT和UINT8，
*                  仅支持连续的Tensor，数据格式支持NCHW、NHWC，且数据格式和数据类型需要与self一致。
* @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含了算子计算流程。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppRotateGetWorkspaceSize(const aclTensor *self, float angle, uint32_t interpolationMode, bool expand,
    const aclIntArray *center, uint32_t paddingMode, const aclFloatArray *fill,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
/**
* @brief acldvppRotate的第二阶段接口，用于执行计算。
* @param [in] workspace: 在npu device侧申请的workspace内存起址。
* @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口acldvppRotateGetWorkspaceSize获取。
* @param [in] executor: op执行器，包含了算子计算流程。
* @param [in] stream: acl stream流。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppRotate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_ROTATE_H_