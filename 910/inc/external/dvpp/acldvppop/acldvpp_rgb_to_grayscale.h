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

#ifndef ACLDVPP_RGB_TO_GRAYSCALE_H_
#define ACLDVPP_RGB_TO_GRAYSCALE_H_

#include "acldvpp_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
* @brief acldvppRgbToGrayscale 的第一阶段接口，根据具体的计算流程，计算workspace大小。
* @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT和UINT8，
*                   仅支持连续的Tensor，数据格式支持NCHW、NHWC。
* @param [in] outputChannelsNum: uint32_t 类型，输出灰度图的通道数，取值可为1或3。当取值为3时，返回图像各通道的像素值将相同。
* @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT和UINT8，
*                  仅支持连续的Tensor，数据格式支持NCHW、NHWC，且数据格式和数据类型 需要与self一致。
* @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含了算子计算流程。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppRgbToGrayscaleGetWorkspaceSize(const aclTensor *self, uint32_t outputChannelsNum,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
/**
* @brief acldvppRgbToGrayscale的第二阶段接口，用于执行计算。
* @param [in] workspace: 在npu device侧申请的workspace内存起址。
* @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口acldvppRgbToGrayscaleGetWorkspaceSize获取。
* @param [in] executor: op执行器，包含了算子计算流程。
* @param [in] stream: acl stream流。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppRgbToGrayscale(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_RGB_TO_GRAYSCALE_H_