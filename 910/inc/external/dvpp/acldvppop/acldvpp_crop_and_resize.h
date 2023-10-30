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

#ifndef ACLDVPP_CROP_AND_RESIZE_H_
#define ACLDVPP_CROP_AND_RESIZE_H_

#include "acldvpp_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
* @brief acldvppCropAndResizeGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
* @param [in] self: npu device侧的aclTensor，数据类型支持 FLOAT 和 UINT8,
*                   仅支持连续的Tensor，数据格式支持NCHW、NHWC，Channels: [1，3]。
* @param [in] top: uint32_t，抠图的上边界位置。
* @param [in] left: uint32_t，抠图的左边界位置。
* @param [in] height: uint32_t，抠图的高度。
* @param [in] width: uint32_t，抠图的宽度。
* @param [in] size: aclIntArray，表示缩放之后的宽、高。resize宽高必须与输出宽高一致。
* @param [in] interpolationMode: uint32_t，缩放插值算法
                                 取值与对应缩放插值算法对应关系为：0: LINEAR/BILINEAR，1: NEAREST，2: BICUBIC
* @param [in] out: npu device侧的aclTensor，数据类型支持 FLOAT 和 UINT8,
*                  仅支持连续的Tensor，数据格式支持NCHW、NHWC，Channels: [1，3]。且数据格式、数据类型和通道数需要与out一致。
* @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含了算子计算流程
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppCropAndResizeGetWorkspaceSize(const aclTensor *self, uint32_t top, uint32_t left, uint32_t height,
    uint32_t width, const aclIntArray* size, uint32_t interpolationMode, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
* @brief acldvppCropAndResize 的第二段接口，用于执行计算。
* @param [in] workspace: 在npu device侧申请的workspace内存起址。
* @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 acldvppCropAndResizeGetWorkspaceSize 获取。
* @param [in] executor: op执行器，包含了算子计算流程。
* @param [in] stream: acl stream流。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppCropAndResize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_CROP_AND_RESIZE_H_